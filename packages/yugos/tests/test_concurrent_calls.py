"""Concurrent spend cannot exceed reserved funds or erase an in-flight bill."""

import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from yugos.inference.actions import CallAction, ModelChoice
from yugos.inference.calls import (
    CallBudget,
    CallError,
    CallLimits,
    MeteredCalls,
    ModelResponse,
    ProviderCallFailed,
    QuestionLimitExceeded,
    RequestRejected,
    UnresolvedSpend,
)
from yugos.inference.chat import ModelInput
from yugos.spend import OpenReservation

ACTION = CallAction(
    model=ModelChoice(provider_id="fixture", model_id="m"), input=ModelInput.from_text("question")
)


@pytest.mark.parametrize("provider_s", [0.25, 2.0])
def test_host_work_clock_excludes_persistence_but_still_enforces_work_budget(monkeypatch, provider_s):
    wall, active, records = [0.0], [0.0], []
    monkeypatch.setattr("yugos.inference.calls.time", SimpleNamespace(monotonic=lambda: wall[0]))

    def prepare(intent):
        wall[0] += 60
        return intent.request

    def complete(request):
        wall[0] += 90
        active[0] += provider_s
        return ModelResponse(text="answer", cost_usd=0.02, latency_s=provider_s)

    def record(call):
        wall[0] += 120
        records.append(call)

    meter = MeteredCalls("run", SimpleNamespace(quote=lambda _: 0.1, complete=complete),
                        CallLimits(max_cost_usd=1, max_question_cost_usd=1), record, prepare_call=prepare)
    if provider_s > 1:
        with pytest.raises(QuestionLimitExceeded, match="time limit"):
            meter.execute(ACTION, "q", deadline=1, clock=lambda: active[0])
    else:
        assert meter.execute(ACTION, "q", deadline=1, clock=lambda: active[0]).text == "answer"
    assert len(records) == 1 and meter.cost_usd == 0.02


@pytest.mark.parametrize("preparation_s", [0.4, 1.0, 2.0])
def test_host_preparation_consumes_the_question_allowance(monkeypatch, preparation_s):
    now, dispatched, records = [0.0], [], []
    monkeypatch.setattr("yugos.inference.calls.time", SimpleNamespace(monotonic=lambda: now[0]))

    def prepare(intent):
        now[0] += preparation_s
        return intent.request.model_copy(update={"call_id": "prepared"})

    def complete(request):
        dispatched.append(request)
        return ModelResponse(text="answer", cost_usd=0.02, latency_s=0)

    meter = MeteredCalls(
        "run",
        SimpleNamespace(quote=lambda _: 0.1, complete=complete),
        CallLimits(max_cost_usd=1, max_question_cost_usd=1),
        records.append,
        prepare_call=prepare,
    )
    if preparation_s >= 1:
        with pytest.raises(QuestionLimitExceeded, match="time limit"):
            meter.execute(ACTION, "q", deadline=1.0)
        assert not dispatched and not records and not meter.budget.reservations
        assert not meter.calls and not meter.budget.stopped.is_set()
    else:
        meter.execute(ACTION, "q", deadline=1.0)
        assert dispatched[0].timeout_s == pytest.approx(1.0 - preparation_s)
        assert records[0].request == dispatched[0]


def test_host_identity_is_prepared_after_local_admission_and_used_for_accounting():
    intents, dispatched, records = [], [], []

    def prepare(intent):
        intents.append(intent)
        return intent.request.model_copy(
            update={"call_id": f"retained-{intent.question_id}-{intent.step}"}
        )

    def complete(request):
        dispatched.append(request.call_id)
        return ModelResponse(text="answer", cost_usd=0.02, latency_s=0)

    meter = MeteredCalls(
        "run",
        SimpleNamespace(quote=lambda _: 0.1, complete=complete),
        CallLimits(max_cost_usd=1, max_question_cost_usd=1),
        records.append,
        prepare_call=prepare,
    )
    meter.execute(ACTION, "question", deadline=time.monotonic() + 10)
    assert dispatched == [records[0].request.call_id] == ["retained-question-0"]
    assert (
        intents[0].run_id == "run" and intents[0].question_id == "question" and intents[0].step == 0
    )
    assert intents[0].quote_usd == 0.1
    assert not meter.budget.reservations and meter.cost_usd == 0.02


def test_host_preparation_cannot_increase_the_quoted_output_limit():
    dispatched = []
    meter = MeteredCalls(
        "run",
        SimpleNamespace(quote=lambda _: 0.1, complete=lambda r: dispatched.append(r)),
        CallLimits(max_cost_usd=1, max_question_cost_usd=1, max_output_tokens=10),
        lambda _: None,
        prepare_call=lambda i: i.request.model_copy(update={"max_output_tokens": 20}),
    )
    with pytest.raises(CallError, match="changed the request"):
        meter.execute(ACTION, "question", deadline=time.monotonic() + 10)
    assert not dispatched and not meter.budget.reservations and meter.budget.stopped.is_set()


def test_host_preparation_cannot_mutate_a_nested_quoted_message():
    dispatched = []

    def prepare(intent):
        intent.request.input.messages[0].content = "different and larger request"
        return intent.request

    meter = MeteredCalls(
        "run",
        SimpleNamespace(quote=lambda _: 0.1, complete=lambda r: dispatched.append(r)),
        CallLimits(max_cost_usd=1, max_question_cost_usd=1),
        lambda _: None,
        prepare_call=prepare,
    )
    with pytest.raises(CallError, match="changed the request"):
        meter.execute(ACTION.model_copy(deep=True), "question", deadline=time.monotonic() + 10)
    assert not dispatched and not meter.budget.reservations


def test_pre_dispatch_rejection_preserves_budget_and_later_admission():
    records = []
    quotes = 0

    def quote(request):
        nonlocal quotes
        quotes += 1
        if quotes == 1:
            raise RequestRejected("input exceeds policy")
        return 0.1

    meter = MeteredCalls(
        "r",
        SimpleNamespace(
            quote=quote,
            complete=lambda _: ModelResponse(text="answered", cost_usd=0.01, latency_s=0),
        ),
        CallLimits(max_cost_usd=1, max_question_cost_usd=1),
        records.append,
    )
    with pytest.raises(RequestRejected, match="input exceeds policy"):
        meter.execute(ACTION, "oversized", deadline=time.monotonic() + 10)
    assert not records and not meter.budget.reservations
    assert meter.execute(ACTION, "later", deadline=time.monotonic() + 10).text == "answered"
    assert records[0].step == 0 and len(records) == 1


def test_four_calls_overlap_and_each_bill_is_retained():
    barrier = Barrier(4)
    records = []

    def complete(request):
        barrier.wait(timeout=5)
        return ModelResponse(text=request.call_id, cost_usd=0.02, latency_s=0)

    meter = MeteredCalls(
        "r",
        SimpleNamespace(quote=lambda r: 0.1, complete=complete),
        CallLimits(max_cost_usd=0.4, max_question_cost_usd=0.1),
        records.append,
    )
    with ThreadPoolExecutor(4) as pool:
        futures = [
            pool.submit(meter.execute, ACTION, str(i), deadline=time.monotonic() + 10)
            for i in range(4)
        ]
        responses = [f.result() for f in futures]
    assert len({r.text for r in responses}) == len(records) == 4
    assert meter.cost_usd == pytest.approx(0.08)
    assert not meter.budget.reservations


def test_unknown_cost_keeps_its_reservation_while_later_questions_continue():
    records = []
    attempts = 0

    def complete(request):
        nonlocal attempts
        attempts += 1
        return ModelResponse(
            text="rejected" if attempts == 1 else "answered",
            cost_usd=0 if attempts == 1 else 0.01,
            cost_complete=attempts != 1,
            reservation_usd=None if attempts != 1 else 0.1,
            latency_s=0,
        )

    meter = MeteredCalls(
        "r",
        SimpleNamespace(quote=lambda _: 0.1, complete=complete),
        CallLimits(max_cost_usd=0.3, max_question_cost_usd=0.1),
        records.append,
    )
    with pytest.raises(UnresolvedSpend) as caught:
        meter.execute(ACTION, "first", deadline=time.monotonic() + 10)
    assert caught.value.pending == records[0]
    assert meter.budget.reservations[records[0].request.call_id] == ("first", 0.1)
    answer = meter.execute(ACTION, "second", deadline=time.monotonic() + 10)
    assert answer.text == "answered"
    assert len(records) == 2 and meter.cost_usd == pytest.approx(0.01)
    assert set(meter.budget.reservations) == {records[0].request.call_id}


def test_known_cost_provider_error_releases_reservation_and_continues():
    records = []
    attempts = 0

    def complete(request):
        nonlocal attempts
        attempts += 1
        return ModelResponse(
            text="" if attempts == 1 else "answered",
            cost_usd=0.02,
            latency_s=0,
            error="invalid_prompt" if attempts == 1 else None,
        )

    meter = MeteredCalls(
        "r",
        SimpleNamespace(quote=lambda _: 0.1, complete=complete),
        CallLimits(max_cost_usd=0.3, max_question_cost_usd=0.1),
        records.append,
    )
    with pytest.raises(ProviderCallFailed) as caught:
        meter.execute(ACTION, "first", deadline=time.monotonic() + 10)
    assert caught.value.pending == records[0]
    assert not meter.budget.reservations
    assert meter.execute(ACTION, "second", deadline=time.monotonic() + 10).text == "answered"
    assert meter.cost_usd == pytest.approx(0.04)


@pytest.mark.parametrize("failure", ["none", "missing_usage", "persistence"])
def test_shared_budget_includes_inflight_quotes_and_retains_late_bills(failure):
    started, release = Event(), Event()
    limits = CallLimits(max_cost_usd=0.1, max_question_cost_usd=0.1)
    budget = CallBudget(limits)
    records = []

    def complete(request):
        started.set()
        assert release.wait(timeout=5)
        return ModelResponse(
            text="paid",
            cost_usd=0.02,
            latency_s=0,
            cost_complete=failure != "missing_usage",
            reservation_usd=None if failure != "missing_usage" else 0.08,
        )

    def record(call):
        records.append(call)
        if failure == "persistence":
            raise OSError("disk unavailable")

    client = SimpleNamespace(quote=lambda r: 0.08, complete=complete)
    candidate = MeteredCalls("r", client, limits, record, budget=budget)
    judge = MeteredCalls("r:services", client, limits, record, budget=budget)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(candidate.execute, ACTION, "first", deadline=time.monotonic() + 10)
        try:
            assert started.wait(timeout=5)
            with pytest.raises(CallError, match="budget cannot fund"):
                judge.execute(ACTION, "second", deadline=time.monotonic() + 10)
        finally:
            release.set()
        if failure == "none":
            future.result()
        else:
            with pytest.raises(CallError):
                future.result()
    assert len(records) == 1
    assert candidate.cost_usd == (0 if failure == "missing_usage" else 0.02)
    assert not judge.calls
    assert bool(budget.reservations) == (failure != "none")
    with pytest.raises(CallError, match="stopped"):
        candidate.execute(ACTION, "third", deadline=time.monotonic() + 10)


def test_restored_history_counts_against_remaining_budget_and_question_calls():
    limits = CallLimits(max_cost_usd=0.03, max_question_cost_usd=0.03, max_calls_per_question=1)
    client = SimpleNamespace(
        quote=lambda r: 0.02,
        complete=lambda r: ModelResponse(text="paid", cost_usd=0.02, latency_s=0),
    )
    first = MeteredCalls("r", client, limits, lambda _: None)
    first.execute(ACTION, "q1", deadline=time.monotonic() + 10)
    resumed = MeteredCalls("r", client, limits, lambda _: None)
    resumed.restore(tuple(first.calls))
    from yugos.inference.calls import QuestionLimitExceeded

    with pytest.raises(QuestionLimitExceeded, match="call limit"):
        resumed.execute(ACTION, "q1", deadline=time.monotonic() + 10)
    with pytest.raises(CallError, match="budget cannot fund"):
        resumed.execute(ACTION, "q2", deadline=time.monotonic() + 10)
    assert len(resumed.calls) == 1 and resumed.cost_usd == 0.02


@pytest.mark.parametrize("defect", ["duplicate", "unknown_cost", "different_run"])
def test_invalid_history_is_rejected_without_partially_restoring(defect):
    limits = CallLimits(max_cost_usd=1, max_question_cost_usd=1)
    client = SimpleNamespace(
        quote=lambda r: 0.1,
        complete=lambda r: ModelResponse(text="paid", cost_usd=0.02, latency_s=0),
    )
    first = MeteredCalls("r", client, limits, lambda _: None)
    first.execute(ACTION, "q", deadline=time.monotonic() + 10)
    call = first.calls[0]
    history = (
        (call, call)
        if defect == "duplicate"
        else (
            call.model_copy(update={"run_id": "other"})
            if defect == "different_run"
            else call.model_copy(
                update={
                    "response": call.response.model_copy(
                        update={"spend": OpenReservation(reservation_usd=Decimal("0.1"))}
                    )
                }
            ),
        )
    )
    resumed = MeteredCalls("r", client, limits, lambda _: None)
    with pytest.raises(ValueError):
        resumed.restore(history)
    assert not resumed.calls and not resumed.budget.calls


def test_restored_unknown_cost_keeps_reservation_and_allows_other_work():
    from yugos.inference.calls import UnresolvedSpend

    limits = CallLimits(max_cost_usd=0.05, max_question_cost_usd=0.05)
    client = SimpleNamespace(
        quote=lambda r: 0.02,
        complete=lambda r: ModelResponse(
            text="", cost_usd=0, cost_complete=False, reservation_usd=0.02, latency_s=0
        ),
    )
    first = MeteredCalls("r", client, limits, lambda _: None)
    with pytest.raises(UnresolvedSpend):
        first.execute(ACTION, "q1", deadline=time.monotonic() + 10)
    call = first.calls[0]
    resumed = MeteredCalls("r", client, limits, lambda _: None)
    resumed.restore((call,), reservations={call.request.call_id: 0.02})
    client.complete = lambda r: ModelResponse(text="done", cost_usd=0.02, latency_s=0)
    resumed.execute(ACTION, "q2", deadline=time.monotonic() + 10)
    with pytest.raises(CallError, match="budget cannot fund"):
        resumed.execute(ACTION, "q3", deadline=time.monotonic() + 10)
    assert resumed.calls[0].response.spend.status == "unresolved"
    assert resumed.budget.reservations[call.request.call_id] == ("q1", 0.02)
    assert resumed.calls[1].request.call_id != call.request.call_id


def _unresolved_on_quote(quote):
    records = []

    def complete(request):
        raise RuntimeError("provider down")

    meter = MeteredCalls(
        "r",
        SimpleNamespace(quote=lambda r: quote, complete=complete),
        CallLimits(max_cost_usd=1, max_question_cost_usd=1),
        records.append,
    )
    with pytest.raises(UnresolvedSpend):
        meter.execute(ACTION, "q", deadline=time.monotonic() + 10)
    assert len(records) == 1
    assert records[0].response.spend.status == "unresolved"
    assert records[0].response.spend.reservation_usd == Decimal("0.01")
    question_id, held = meter.budget.reservations[records[0].request.call_id]
    assert question_id == "q"
    assert held == pytest.approx(0.01)


def test_zero_quote_unresolved_reserves_one_cent():
    _unresolved_on_quote(0)


def test_subcent_quote_unresolved_reserves_one_cent():
    _unresolved_on_quote(0.009)


@pytest.mark.parametrize("same_question", [False, True])
def test_budget_fitting_waits_for_settlement_before_reducing_output(same_question):
    entered, release, waiting = Event(), Event(), Event()
    requests, fitted = [], []

    def quote(request):
        if requests:
            waiting.set()
        return 0.8

    def complete(request):
        requests.append(request)
        if len(requests) == 1:
            entered.set()
            assert release.wait(5)
        return ModelResponse(text="answer", cost_usd=0.1, latency_s=0)

    meter = MeteredCalls(
        "r", SimpleNamespace(quote=quote, complete=complete),
        CallLimits(max_cost_usd=1, max_question_cost_usd=1), lambda _: None,
        fit_request=lambda request, allowance: fitted.append(allowance),
    )
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(meter.execute, ACTION, "q", deadline=time.monotonic() + 10)
        assert entered.wait(5)
        second = pool.submit(meter.execute, ACTION, "q" if same_question else "q2",
                             deadline=time.monotonic() + 10)
        try:
            assert waiting.wait(5)
            assert not second.done() and not fitted and not meter.budget.stopped.is_set()
        finally:
            release.set()
        assert first.result().text == second.result().text == "answer"
    assert not fitted and all(r.max_output_tokens is None for r in requests)
    assert meter.cost_usd == pytest.approx(0.2) and not meter.budget.reservations


def test_budget_fitting_uses_remaining_spend_and_retains_the_funded_request():
    requests, intents = [], []

    def complete(request):
        requests.append(request)
        return ModelResponse(text="answer", cost_usd=0.6 if len(requests) == 1 else 0.1, latency_s=0)

    meter = MeteredCalls(
        "r", SimpleNamespace(quote=lambda r: (r.max_output_tokens or 80) / 100, complete=complete),
        CallLimits(max_cost_usd=1, max_question_cost_usd=1), lambda _: None,
        fit_request=lambda request, available: request.model_copy(
            update={"max_output_tokens": int(round(available * 100))}),
        prepare_call=lambda intent: intents.append(intent) or intent.request,
    )
    meter.execute(ACTION, "q", deadline=time.monotonic() + 10)
    meter.execute(ACTION, "q2", deadline=time.monotonic() + 10)
    assert requests[1].max_output_tokens == 40 and intents[1].quote_usd == 0.4
    assert meter.calls[1].request == requests[1] and meter.cost_usd == pytest.approx(0.7)
    assert not meter.budget.reservations


def test_waiting_for_a_reservation_obeys_deadlines_without_stopping_other_calls():
    entered, release = Event(), Event()

    def complete(request):
        entered.set()
        assert release.wait(5)
        return ModelResponse(text="answer", cost_usd=0.1, latency_s=0)

    meter = MeteredCalls(
        "r", SimpleNamespace(quote=lambda _: 0.8, complete=complete),
        CallLimits(max_cost_usd=1, max_question_cost_usd=1), lambda _: None,
        fit_request=lambda *_: pytest.fail("active holds must settle first"),
    )
    with ThreadPoolExecutor(1) as pool:
        first = pool.submit(meter.execute, ACTION, "q", deadline=time.monotonic() + 10)
        try:
            assert entered.wait(5)
            with pytest.raises(QuestionLimitExceeded, match="time limit"):
                meter.execute(ACTION, "q2", deadline=time.monotonic() + 0.05)
            assert not meter.budget.stopped.is_set()
        finally:
            release.set()
        assert first.result().text == "answer"


def test_unresolved_holds_stay_funded_without_waiting_for_a_finished_call():
    attempts = []

    def complete(request):
        attempts.append(request)
        if len(attempts) == 1:
            return ModelResponse(text="unknown", spend=OpenReservation(reservation_usd="0.8"), latency_s=0)
        return ModelResponse(text="answer", cost_usd=0.1, latency_s=0)

    meter = MeteredCalls(
        "r", SimpleNamespace(quote=lambda r: (r.max_output_tokens or 80) / 100, complete=complete),
        CallLimits(max_cost_usd=1, max_question_cost_usd=1), lambda _: None,
        fit_request=lambda request, available: request.model_copy(
            update={"max_output_tokens": int(round(available * 100))}),
    )
    with pytest.raises(UnresolvedSpend):
        meter.execute(ACTION, "q", deadline=time.monotonic() + 10)
    meter.execute(ACTION, "q2", deadline=time.monotonic() + 10)
    assert attempts[1].max_output_tokens == 20
    assert list(meter.budget.reservations.values()) == [("q", 0.8)]
    assert meter.cost_usd == pytest.approx(0.1)


@pytest.mark.parametrize("defect", ["input", "model", "output"])
def test_budget_fitter_cannot_change_the_request_or_raise_a_native_cap(defect):
    def fit(request, available):
        if defect == "input":
            request.input.messages[0].content = "changed"
            return request
        if defect == "model":
            return request.model_copy(update={"model": ModelChoice(provider_id="fixture", model_id="other")})
        return request.model_copy(update={"max_output_tokens": 11})

    meter = MeteredCalls(
        "r", SimpleNamespace(quote=lambda _: 0.8, complete=lambda _: pytest.fail("unfunded dispatch")),
        CallLimits(max_cost_usd=0.5, max_question_cost_usd=0.5), lambda _: None,
        fit_request=fit,
    )
    with pytest.raises(CallError, match="cost bound is unavailable"):
        meter.execute(ACTION, "q", max_output_tokens=10, deadline=time.monotonic() + 10)
    assert not meter.calls and not meter.budget.reservations
    assert ACTION.input.messages[0].content == "question"


def test_stopping_the_budget_wakes_waiters_without_another_provider_call():
    entered, release, waiting = Event(), Event(), Event()
    requests = []

    def quote(request):
        if requests:
            waiting.set()
        return 0.8

    def complete(request):
        requests.append(request)
        entered.set()
        assert release.wait(5)
        return ModelResponse(text="answer", cost_usd=0.1, latency_s=0)

    meter = MeteredCalls(
        "r", SimpleNamespace(quote=quote, complete=complete),
        CallLimits(max_cost_usd=1, max_question_cost_usd=1), lambda _: None,
        fit_request=lambda *_: pytest.fail("stopped call cannot be fitted"),
    )
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(meter.execute, ACTION, "q", deadline=time.monotonic() + 10)
        assert entered.wait(5)
        second = pool.submit(meter.execute, ACTION, "q2", deadline=time.monotonic() + 10)
        try:
            assert waiting.wait(5)
            meter.budget.stop()
            with pytest.raises(CallError, match="stopped"):
                second.result(timeout=1)
        finally:
            release.set()
        assert first.result().text == "answer"
    assert len(requests) == 1 and meter.cost_usd == 0.1 and not meter.budget.reservations

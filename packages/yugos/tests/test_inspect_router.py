import asyncio
import json
import threading
from decimal import Decimal
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai import Task  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from inspect_ai.model import GenerateConfig, ModelOutput, RetryDecision  # noqa: E402
from inspect_ai.scorer import includes  # noqa: E402
from inspect_ai.solver import generate, solver, use_tools  # noqa: E402
from inspect_ai.tool import tool  # noqa: E402
from inspect_ai.util import LimitExceededError  # noqa: E402
from inspect_fakes import ScriptedRouter, exact_question, pack_task, reply, run  # noqa: E402

from yugos.inference.calls import QuestionLimitExceeded  # noqa: E402
from yugos.inspect_engine import (  # noqa: E402
    QUESTION_EXHAUSTED,
    JsonlReceipts,
    MemoryReceipts,
    ProviderReceipt,
    QuestionError,
    QuestionFailures,
    RouterError,
    RouterReply,
    RouterToolCall,
    RouterUnavailable,
    RunError,
    measure_log,
    router_model,
    run_competition_eval,
    versions,
)
from yugos.inspect_engine.receipts import receipts_by_sample  # noqa: E402
from yugos.inspect_engine.router_model import RouterModel  # noqa: E402
from yugos.submission.protocol import Failure  # noqa: E402
from yugos.submission.session import (  # noqa: E402
    RequestOutcome,
    RequestState,
    SessionError,
    SessionExit,
)


def receipt(receipt_id="r1", sample_uuid="s1", cost="0.25", **changes):
    return ProviderReceipt(
        **{
            "receipt_id": receipt_id,
            "sample_uuid": sample_uuid,
            "provider": "openai",
            "model": "gpt-5.6-luna",
            "cost_usd": Decimal(cost),
            "started_ns": 10,
            "ended_ns": 20,
        }
        | changes
    )


def test_pinned_versions_guard_the_private_sample_api(monkeypatch):
    # Catches running on an Inspect release whose private sample API was never verified.
    versions.require_pinned_inspect()
    monkeypatch.setattr(versions, "version", lambda package: "0.3.999")
    with pytest.raises(versions.UnsupportedInspectVersion, match="0.3.999"):
        versions.require_pinned_inspect()


def test_each_call_carries_its_sample_identity_but_never_the_answer_key(tmp_path):
    # Catches a router reaching targets or metadata, or receipts that cannot join the log.
    questions = [
        exact_question("q1", "reference-key-alpha", source_benchmark="secret-source"),
        exact_question("q2", "reference-key-beta", source_benchmark="secret-source"),
    ]
    router = ScriptedRouter({q.prompt: "no idea" for q in questions})
    log = run(pack_task(questions, name="pack"), router, tmp_path)

    logged = {s.uuid: (str(s.id), s.epoch) for s in log.samples}
    assert {c.scope.sample_uuid: (c.scope.sample_id, c.scope.epoch) for c in router.calls} == logged
    seen_by_router = json.dumps([c.request.model_dump(mode="json") for c in router.calls])
    for secret in ("reference-key", "secret-source", "grading", "workload_class"):
        assert secret not in seen_by_router


def test_router_reply_becomes_output_with_tools_usage_and_operator_time(tmp_path):
    # Catches lost tool calls, dropped tool schemas, or unset ModelOutput.time.
    @tool
    def lookup():
        async def execute(city: str) -> str:
            """Look up a city.

            Args:
                city: City name.
            """
            return "sunny"

        return execute

    calls = []

    async def client(call):
        calls.append(call)
        return RouterReply(
            tool_calls=(RouterToolCall(id="t1", function="lookup", arguments={"city": "Oslo"}),),
            finish_reason="tool_calls",
            input_tokens=11,
            output_tokens=5,
        )

    task = Task(
        dataset=MemoryDataset(
            [
                Sample(
                    id="w",
                    input="Weather in Oslo?",
                    target="sunny",
                    metadata={"workload_class": "agentic"},
                )
            ]
        ),
        solver=[use_tools(lookup()), generate(tool_calls="none")],
        scorer=includes(),
    )
    model = router_model(client, submission_id="sub-tools", config=GenerateConfig(temperature=0))
    log = run_competition_eval(task, model, log_dir=str(tmp_path), display="none")
    [sample] = log.samples

    [call] = calls
    assert call.request.tools[0].name == "lookup"
    assert call.request.config.model_dump(exclude_none=True) == {"temperature": 0}
    [tool_call] = sample.output.message.tool_calls
    assert (tool_call.function, tool_call.arguments) == ("lookup", {"city": "Oslo"})
    assert sample.output.stop_reason == "tool_calls"
    [measured] = measure_log(log, [])
    assert not measured.measurement.outcome.unrealized
    assert not measured.measurement.non_answer
    assert (sample.output.usage.input_tokens, sample.output.usage.output_tokens) == (11, 5)
    [event] = [e for e in sample.events if e.event == "model"]
    assert event.output.time is not None and event.output.time >= 0


def test_only_infrastructure_unavailability_is_retried():
    # Catches retrying a miner failure (a second attempt) or failing on a transient outage.
    api = RouterModel("sub", client=None)
    decision = api.should_retry(RouterUnavailable("proxy restarting", retry_after=2))
    assert isinstance(decision, RetryDecision) and decision.retry
    assert api.should_retry(RouterError("router crashed")) is False
    assert api.should_retry(TimeoutError()) is False


class FailureHost(QuestionFailures):
    """The smallest host of the shared question boundary."""

    fatal_errors = (ConnectionAbortedError,)

    def __init__(self):
        self.meter = SimpleNamespace(budget=SimpleNamespace(stopped=threading.Event()))
        self.service_meter = None
        self.session = SimpleNamespace(session_exit=None)
        self.limits = SimpleNamespace(
            question_timeout_s=60, max_calls_per_question=4, max_question_cost_usd=1
        )
        self.stopped, self.ended = [], []

    def stop_run(self, call, exc):
        self.stopped.append(exc)
        self.meter.budget.stopped.set()

    def known_failure(self, call, exc, role):
        if isinstance(exc, TimeoutError):
            return RouterUnavailable("not billed; retrying")
        return None

    def question_ended(self, call, outcome):
        self.ended.append(outcome)

    def outcome(self, exc, **kwargs):
        try:
            with self.boundary("call", **kwargs):
                raise exc
        except BaseException as raised:
            return raised


def submission_fault(state, *, closed=False):
    failure = Failure(code="request.failed", request_id="r", detail="router bug")
    exit = SessionExit(failure, "submission") if closed else None
    return SessionError(failure, outcome=RequestOutcome("r", state, session_exit=exit))


def test_unfamiliar_question_errors_leave_only_their_question_incomplete():
    host = FailureHost()
    outcome = host.outcome(LookupError("new provider error"))
    assert isinstance(outcome, QuestionError) and "LookupError: new provider error" in str(outcome)
    assert isinstance(outcome.__cause__, LookupError)
    assert host.ended == [outcome] and not host.stopped


def test_known_question_outcomes_are_classified_once():
    host = FailureHost()
    assert isinstance(host.outcome(TimeoutError()), RouterUnavailable)
    assert host.ended == []  # a retry does not end the question
    limit = host.outcome(QuestionLimitExceeded("question call limit reached"))
    assert isinstance(limit, LimitExceededError) and limit.type == "turn"
    assert str(limit.message).startswith(QUESTION_EXHAUSTED)
    assert isinstance(host.outcome(submission_fault(RequestState.FAILED)), RouterError)
    # A service role never owns the submission's protocol faults.
    served = host.outcome(submission_fault(RequestState.FAILED), role="grader")
    assert isinstance(served, QuestionError)
    assert not host.stopped


@pytest.mark.parametrize(
    "exc",
    [RunError("shared protocol is inconsistent"), ConnectionAbortedError("worker lost")],
)
def test_explicit_fatal_errors_stop_the_run(exc):
    host = FailureHost()
    assert host.outcome(exc) is exc
    assert host.stopped == [exc] and not host.ended


def test_any_error_after_shared_state_stops_also_stops_the_run():
    host = FailureHost()
    host.meter.budget.stopped.set()
    exc = LookupError("after the budget stopped")
    assert host.outcome(exc) is exc and host.stopped == [exc]

    host = FailureHost()
    host.session.session_exit = object()
    interrupted = submission_fault(RequestState.INTERRUPTED, closed=True)
    assert host.outcome(interrupted) is interrupted and host.stopped == [interrupted]


@pytest.mark.parametrize("owner", ["submission", "infrastructure"])
def test_a_stop_noticed_by_another_question_keeps_the_session_exit_cause(owner):
    # A provider error can overtake the protocol fault that closed the worker.
    host = FailureHost()
    fault = Failure(code="request.protocol", request_id="r", detail="bad envelope")
    host.session.session_exit = SessionExit(fault, owner)
    exc = LookupError("provider error in another question")
    assert host.outcome(exc) is exc
    [cause] = host.stopped
    assert isinstance(cause, SessionError) and cause.owner == owner
    assert cause.failure == fault and cause.outcome.session_exit is host.session.session_exit


def test_a_protocol_fault_fails_its_request_and_stops_the_shared_worker():
    host = FailureHost()
    host.session.session_exit = object()
    fault = submission_fault(RequestState.FAILED, closed=True)
    outcome = host.outcome(fault)
    assert isinstance(outcome, RouterError)
    assert host.ended == [outcome] and host.stopped == [fault]


def test_cancellation_passes_through_the_boundary():
    host = FailureHost()
    assert isinstance(host.outcome(KeyboardInterrupt()), KeyboardInterrupt)
    assert not host.stopped and not host.ended


def test_swallowed_question_failure_cannot_be_scored_or_dispatch_again(tmp_path):
    calls = []
    host = FailureHost()

    async def client(call):
        calls.append(call.scope.sample_id)
        with host.boundary(call):
            if call.scope.sample_id == "broken":
                raise LookupError("unfamiliar callback failure")
            return reply("yes")

    @solver
    def swallowing_solver():
        async def solve(state, generate):
            for _ in range(2 if state.sample_id == "broken" else 1):
                try:
                    await generate(state)
                except Exception:
                    state.output = ModelOutput.from_content("fixture", "yes")
            return state

        return solve

    task = pack_task(
        [exact_question("broken", "yes"), exact_question("healthy", "yes")], name="pack"
    )
    task.solver = swallowing_solver()
    log = run_competition_eval(
        task, router_model(client, submission_id="sub"), log_dir=str(tmp_path), display="none"
    )
    assert calls.count("broken") == calls.count("healthy") == 1
    # Reloading the log must retain the failure even though the solver swallowed it.
    rows = {
        row.measurement.question_id: row for row in measure_log(read_eval_log(log.location), [])
    }
    assert rows["broken"].failure_owner == "infrastructure"
    assert rows["broken"].grade is None and rows["broken"].measurement.outcome.unrealized
    assert rows["healthy"].grade.score == 1


def test_a_caught_submission_failure_cannot_dispatch_again(tmp_path):
    calls = []
    host = FailureHost()

    async def client(call):
        calls.append(call.scope.sample_id)
        with host.boundary(call):
            if call.scope.sample_id == "broken":
                raise RouterError("router bug")
            return reply("yes")

    @solver
    def retrying_solver():
        async def solve(state, generate):
            for _ in range(2):
                try:
                    await generate(state)
                except RouterError:
                    state.output = ModelOutput.from_content("fixture", "yes")
            return state

        return solve

    task = pack_task(
        [exact_question("broken", "yes"), exact_question("healthy", "yes")], name="pack"
    )
    task.solver = retrying_solver()
    log = run_competition_eval(
        task, router_model(client, submission_id="sub"), log_dir=str(tmp_path), display="none"
    )
    assert calls.count("broken") == 1 and calls.count("healthy") == 2
    rows = {
        row.measurement.question_id: row for row in measure_log(read_eval_log(log.location), [])
    }
    assert rows["broken"].failure_owner == "submission" and rows["broken"].grade is None
    assert rows["healthy"].grade.score == 1


def test_submissions_do_not_share_a_connection_pool():
    assert RouterModel("a", None).connection_key() != RouterModel("b", None).connection_key()


def test_inspect_can_construct_a_named_role_without_dispatching(monkeypatch):
    import importlib

    from inspect_ai.model import get_model

    module = importlib.import_module("yugos.inspect_engine.router_model")
    model = get_model("yugos-router/service:grader", memoize=False)
    assert str(model) == "yugos-router/service:grader"
    monkeypatch.setattr(module, "active_sample_scope", lambda: object())
    with pytest.raises(RuntimeError, match="no bound client"):
        asyncio.run(model.api.generate([], [], "auto", GenerateConfig()))


def test_router_calls_outside_a_sample_are_refused():
    # Unattributable calls would be unmetered spend.
    api = RouterModel("sub", client=None)
    with pytest.raises(RuntimeError, match="inside an Inspect sample"):
        asyncio.run(api.generate([], [], "auto", GenerateConfig()))


def test_transient_unavailability_is_retried_and_router_errors_are_not(tmp_path):
    flaky, broken = exact_question("flaky", "yes"), exact_question("broken", "yes")
    attempts = {"flaky": 0, "broken": 0}

    async def client(call):
        attempts[call.scope.sample_id] += 1
        if call.scope.sample_id == "broken":
            raise RouterError("submission crashed")
        if attempts["flaky"] == 1:
            raise RouterUnavailable("proxy restarting")
        return reply("yes")

    model = router_model(client, submission_id="sub")
    log = run_competition_eval(
        pack_task([flaky, broken], name="pack"), model, log_dir=str(tmp_path), display="none"
    )
    rows = {row.measurement.question_id: row for row in measure_log(log, [])}
    assert attempts == {"flaky": 2, "broken": 1}
    assert rows["flaky"].measurement.outcome.score == 1
    assert rows["broken"].failure_owner == "submission"
    assert rows["broken"].measurement.non_answer


def test_receipts_round_trip_and_reject_duplicates(tmp_path):
    sink = JsonlReceipts(tmp_path / "receipts.jsonl")
    sink.append(receipt("r1"))
    sink.append(receipt("r2", cost="0.10"))
    assert sink.receipts() == (receipt("r1"), receipt("r2", cost="0.10"))
    with pytest.raises(ValueError, match="duplicate receipt"):
        receipts_by_sample([receipt("r1"), receipt("r1")])
    with pytest.raises(ValueError):
        receipt(started_ns=30, ended_ns=20)
    memory = MemoryReceipts()
    memory.append(receipt())
    assert memory.receipts() == (receipt(),)


@pytest.mark.parametrize("measured_s", [None, 0.0, 0.02])
def test_host_duration_preserves_tool_work_and_excludes_model_waits(tmp_path, measured_s):
    # A ModelOutput.time alone is overwritten by Inspect; zero also skips its reconciliation.
    import anyio

    @tool
    def lookup():
        async def execute() -> str:
            """Look up the answer."""
            await anyio.sleep(0.08)
            return "yes"

        return execute

    calls = 0

    async def client(call):
        nonlocal calls
        calls += 1
        await anyio.sleep(0.2)  # Host bookkeeping, not declared model work.
        return RouterReply(
            content="yes" if calls == 2 else "",
            tool_calls=() if calls == 2 else (RouterToolCall(id="lookup", function="lookup"),),
            finish_reason="stop" if calls == 2 else "tool_calls",
            time_s=measured_s,
        )

    task = Task(
        dataset=[
            Sample(id="q", input="Look it up", target="yes", metadata={"workload_class": "agentic"})
        ],
        solver=[use_tools(lookup()), generate()],
        scorer=includes(),
    )
    log = run_competition_eval(
        task, router_model(client, submission_id="timed"), log_dir=str(tmp_path), display="none"
    )
    [row] = measure_log(log, [])
    assert calls == 2 and row.grade.score == 1
    if measured_s is None:
        assert row.active_ns / 1e9 >= 0.48
    else:
        assert row.active_ns / 1e9 == pytest.approx(0.08 + 2 * measured_s, abs=0.07)
        events = [e for e in log.samples[0].events if e.event == "model"]
        assert all(e.output.time == measured_s and e.call.time == measured_s for e in events)


def test_concurrent_samples_keep_their_own_host_duration(tmp_path):
    import anyio

    async def client(call):
        await anyio.sleep(0.15)
        return RouterReply(content="yes", finish_reason="stop", time_s=float(call.scope.sample_id))

    questions = [exact_question("0.02", "yes"), exact_question("0.07", "yes")]
    log = run_competition_eval(
        pack_task(questions, name="timed"),
        router_model(client, submission_id="concurrent"),
        log_dir=str(tmp_path),
        display="none",
    )
    for row in measure_log(log, []):
        assert row.active_ns / 1e9 == pytest.approx(float(row.measurement.question_id), abs=0.06)


@pytest.mark.parametrize("ending", ["success", "error", "cancelled", "exhausted"])
@pytest.mark.parametrize("durations", [(0.0, 0.0), (0.02, 0.03)])
@pytest.mark.parametrize("backoff", [0.0, 0.03])
def test_complete_generation_reconciles_retry_waits_once(
    tmp_path, monkeypatch, ending, durations, backoff
):
    import anyio
    import tenacity
    from inspect_ai._util.working import sample_working_time
    from inspect_ai.model import ModelOutput, _retry
    from inspect_ai.solver import solver

    from yugos.inspect_engine import record_model_time

    monkeypatch.setattr(
        _retry, "wait_exponential_jitter", lambda **kwargs: tenacity.wait_fixed(backoff)
    )
    calls, observed = [], []
    terminal = (
        asyncio.CancelledError
        if ending == "cancelled"
        else tenacity.RetryError
        if ending == "exhausted"
        else RouterError
    )

    async def client(call):
        calls.append(call)
        await anyio.sleep(0.15)  # Host bookkeeping on both attempts.
        if len(calls) == 1 or ending == "exhausted":
            record_model_time(durations[0] if len(calls) == 1 else durations[1])
            raise RouterUnavailable("not dispatched")
        if ending != "success":
            record_model_time(durations[1])
            raise terminal("terminated")
        return RouterReply(content="yes", finish_reason="stop", time_s=durations[1])

    @solver
    def observe_generation():
        async def solve(state, generate):
            started = sample_working_time()
            try:
                state = await generate(state)
            except terminal:
                if ending == "success":
                    raise
            observed.append(sample_working_time() - started)
            state.output = ModelOutput.from_content("timed", "yes")
            return state

        return solve

    task = Task(
        dataset=[Sample(id="q", input="question", target="yes")],
        solver=observe_generation(),
        scorer=includes(),
    )
    log = run_competition_eval(
        task, router_model(client, submission_id="retry"), log_dir=str(tmp_path), display="none"
    )
    assert log.status == "success"
    assert len(calls) == (3 if ending == "exhausted" else 2)
    expected = durations[0] + durations[1] * (len(calls) - 1)
    assert observed == pytest.approx([expected], abs=0.025)

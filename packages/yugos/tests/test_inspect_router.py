import asyncio
import json
from decimal import Decimal

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai import Task  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.model import GenerateConfig, RetryDecision  # noqa: E402
from inspect_ai.scorer import includes  # noqa: E402
from inspect_ai.solver import generate, use_tools  # noqa: E402
from inspect_ai.tool import tool  # noqa: E402
from inspect_fakes import ScriptedRouter, exact_question, pack_task, reply, run  # noqa: E402

from yugos.inspect_engine import (  # noqa: E402
    JsonlReceipts,
    MemoryReceipts,
    ProviderReceipt,
    RouterError,
    RouterReply,
    RouterToolCall,
    RouterUnavailable,
    measure_log,
    router_model,
    run_competition_eval,
    versions,
)
from yugos.inspect_engine.receipts import receipts_by_sample  # noqa: E402
from yugos.inspect_engine.router_model import RouterModel  # noqa: E402


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

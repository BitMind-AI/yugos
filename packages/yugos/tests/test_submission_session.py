"""Session outcomes preserve the origin and extent of a worker failure."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event, RLock, current_thread
from types import SimpleNamespace

import pytest

from yugos.inference.actions import ModelChoice
from yugos.inference.chat import ModelInput
from yugos.submission.protocol import ROUTING_RUNTIME
from yugos.submission.session import (
    CleanupReport,
    RequestState,
    SessionError,
    SubmissionSession,
    WorkerTimeouts,
)

MODEL = ModelChoice(provider_id="fixture", model_id="model")
INPUT = ModelInput.from_text("question")


class Worker:
    def __init__(self):
        self.sent = []
        self.failure = None

    def exchange(self, raw, *, timeout_s):
        request = json.loads(raw)
        self.sent.append(request)
        if self.failure is not None and request["kind"] == self.failure[0]:
            if self.failure[1] == "transport":
                raise TimeoutError("transport lost")
            return "invalid JSON\n"
        reply = {
            "schema": request["schema"],
            "session_id": request["session_id"],
            "sequence": request["sequence"] + 1,
        }
        if request["kind"] == "start":
            reply.update(
                kind="prepared",
                metadata={
                    "submission_sha256": request["submission_sha256"],
                    "manifest_sha256": "b" * 64,
                    "name": "fixture",
                    "runtime": ROUTING_RUNTIME,
                    "version": "1",
                    "required_models": [MODEL.model_dump()],
                },
            )
        elif request["kind"] == "load":
            reply.update(kind="loaded")
        elif request["kind"] == "shutdown":
            reply.update(kind="stopped")
        else:
            reply.update(kind="routed", request_id=request["request_id"], model=MODEL.model_dump())
        return json.dumps(reply) + "\n"

    def close(self, *, timeout_s):
        return CleanupReport(reaped=True)


def test_concurrent_decision_clock_excludes_waiting_for_another_request(monkeypatch):
    clock = [0]
    monkeypatch.setattr(
        "yugos.submission.session.time",
        SimpleNamespace(monotonic_ns=lambda: clock[0], monotonic=lambda: clock[0] / 1e9),
    )
    worker = Worker()
    session = SubmissionSession(worker, frozenset([MODEL]), WorkerTimeouts())
    session.load("a" * 64, None)
    entered, queued, release = Event(), Event(), Event()
    lock = RLock()
    first_thread = []

    class ObservedLock:
        def __enter__(self):
            if first_thread and current_thread() is not first_thread[0]:
                queued.set()
            lock.acquire()

        def __exit__(self, *args):
            lock.release()

    session._lock = ObservedLock()
    exchange = worker.exchange

    def decision(raw, **kwargs):
        request = json.loads(raw)
        if request["kind"] == "route":
            if request["request_id"] == "slow":
                first_thread.append(current_thread())
                entered.set()
                assert release.wait(timeout=5)
                clock[0] += 30_000_000_000
            else:
                clock[0] += 2_000_000_000
        return exchange(raw, **kwargs)

    worker.exchange = decision
    with ThreadPoolExecutor(2) as pool:
        slow = pool.submit(session.route, INPUT, "slow", timeout_s=60)
        assert entered.wait(timeout=5)
        fast = pool.submit(session.route, INPUT, "fast", timeout_s=60)
        try:
            assert queued.wait(timeout=5)
        finally:
            release.set()
        slow.result(timeout=5)
        fast.result(timeout=5)
    assert session.request_outcome("slow").decision_time_ns == 30_000_000_000
    assert session.request_outcome("fast").decision_time_ns == 2_000_000_000
    session.close()
    assert session.request_outcome("fast").decision_time_ns == 2_000_000_000
    assert session.request_outcome("slow").decision_time_ns == 30_000_000_000


@pytest.mark.parametrize("failure", ["protocol", "transport"])
def test_session_exit_preserves_completed_and_interrupts_queued_requests(failure):
    worker = Worker()
    session = SubmissionSession(worker, frozenset([MODEL]), WorkerTimeouts())
    session.load("a" * 64, None)
    session.route(INPUT, "healthy", timeout_s=10)
    worker.failure = "route", failure
    with pytest.raises(SessionError) as caught:
        session.route(INPUT, "fault", timeout_s=10)
    error = caught.value
    assert error.outcome.request_id == "fault"
    assert error.outcome.state == (
        RequestState.FAILED if failure == "protocol" else RequestState.INTERRUPTED
    )
    assert error.owner == ("submission" if failure == "protocol" else "infrastructure")
    assert session.request_outcome("healthy").state == RequestState.COMPLETED
    sent = len(worker.sent)
    with pytest.raises(SessionError) as queued:
        session.route(INPUT, "queued", timeout_s=10)
    assert len(worker.sent) == sent
    assert queued.value.outcome.state == RequestState.INTERRUPTED
    assert queued.value.outcome.session_exit == error.outcome.session_exit
    assert queued.value.owner == error.owner

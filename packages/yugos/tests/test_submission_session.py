"""Session outcomes preserve the origin and extent of a worker failure."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event, RLock, current_thread
from types import SimpleNamespace

import pytest

from yugos.inference.actions import CallOutput, ModelChoice
from yugos.inference.chat import ModelInput
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
                    "version": "1",
                    "required_models": [MODEL.model_dump()],
                },
            )
        elif request["kind"] == "load":
            reply.update(kind="loaded")
        elif request["kind"] == "shutdown":
            reply.update(kind="stopped")
        else:
            reply.update(kind="finished", request_id=request["request_id"])
            if request["kind"] == "solve":
                reply.update(
                    kind="call_requested",
                    action={
                        "model": MODEL.model_dump(),
                        "input": INPUT.model_dump(mode="json"),
                    },
                )
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
    session = SubmissionSession(worker, frozenset([MODEL]), WorkerTimeouts(), max_active_requests=2)
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
        if request["kind"] == "solve":
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
        slow = pool.submit(session.solve, INPUT, "slow", timeout_s=60)
        assert entered.wait(timeout=5)
        fast = pool.submit(session.solve, INPUT, "fast", timeout_s=60)
        try:
            assert queued.wait(timeout=5)
        finally:
            release.set()
        slow.result(timeout=5)
        fast.result(timeout=5)
    assert session.request_outcome("slow").decision_time_ns == 30_000_000_000
    assert session.request_outcome("fast").decision_time_ns == 2_000_000_000
    session.resume(CallOutput(text="answer", finish_reason="stop"), request_id="fast", timeout_s=10)
    session.cancel("slow")
    session.close()
    assert session.request_outcome("fast").decision_time_ns == 2_000_000_000
    assert session.request_outcome("slow").decision_time_ns == 30_000_000_000


@pytest.mark.parametrize("phase", ["solve", "resume", "cancel"])
@pytest.mark.parametrize("failure", ["protocol", "transport"])
def test_session_exit_distinguishes_the_origin_from_active_and_queued_requests(phase, failure):
    worker = Worker()
    session = SubmissionSession(worker, frozenset([MODEL]), WorkerTimeouts(), max_active_requests=4)
    session.load("a" * 64, None)
    session.solve(INPUT, "healthy", timeout_s=10)
    session.resume(
        CallOutput(text="answer", finish_reason="stop"), request_id="healthy", timeout_s=10
    )
    session.solve(INPUT, "active", timeout_s=10)
    if phase != "solve":
        session.solve(INPUT, "fault", timeout_s=10)
    worker.failure = phase, failure
    with pytest.raises(SessionError) as caught:
        if phase == "solve":
            session.solve(INPUT, "fault", timeout_s=10)
        elif phase == "resume":
            session.resume(
                CallOutput(text="answer", finish_reason="stop"), request_id="fault", timeout_s=10
            )
        else:
            session.cancel("fault")
    error = caught.value
    assert error.outcome.request_id == "fault"
    assert error.outcome.state == (
        RequestState.FAILED if failure == "protocol" else RequestState.INTERRUPTED
    )
    assert error.owner == ("submission" if failure == "protocol" else "infrastructure")
    assert session.session_exit.failure.request_id == "fault"
    assert session.request_outcome("healthy").state == RequestState.COMPLETED
    assert session.request_outcome("active").state == RequestState.INTERRUPTED
    sent = len(worker.sent)
    with pytest.raises(SessionError) as queued:
        session.solve(INPUT, "queued", timeout_s=10)
    assert len(worker.sent) == sent
    assert queued.value.outcome.state == RequestState.INTERRUPTED
    assert queued.value.outcome.session_exit == error.outcome.session_exit
    assert queued.value.owner == error.owner


def test_finished_and_cancelled_requests_survive_session_close():
    worker = Worker()
    session = SubmissionSession(worker, frozenset([MODEL]), WorkerTimeouts())
    session.load("a" * 64, None)
    session.solve(INPUT, "completed", timeout_s=10)
    session.resume(
        CallOutput(text="answer", finish_reason="stop"), request_id="completed", timeout_s=10
    )
    session.solve(INPUT, "cancelled", timeout_s=10)
    session.cancel("cancelled")
    session.close()
    assert session.request_outcome("completed").state == RequestState.COMPLETED
    assert session.request_outcome("cancelled").state == RequestState.CANCELLED

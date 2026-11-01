"""Host a miner's router bundle locally, behind the same worker protocol the service uses.

The bundle runs in a `python -m yugos.submission.worker` subprocess driven by a
`SubmissionSession`. Each Inspect generate call becomes one submission request:
the router selects a model, `MeteredCalls` executes the original request, and
each executed call becomes a `ProviderReceipt` for the sample. Local hosting is
not a sandbox: run only bundles you trust, such as your own.
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import anyio
from inspect_ai.hooks import Hooks, hooks
from inspect_ai.util import LimitExceededError

from yugos.inference.actions import CallAction, ModelChoice
from yugos.inference.calls import CallLimits, CallRecord, MeteredCalls, QuestionLimitExceeded
from yugos.inspect_engine import (
    MemoryReceipts,
    ProviderReceipt,
    QuestionFailures,
    ReceiptSink,
    RouterCall,
    RouterReply,
    RouterToolCall,
    RunError,
)
from yugos.submission.protocol import MAX_FRAME_BYTES
from yugos.submission.session import CleanupReport

_SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)


def worker_environment(env=None) -> dict[str, str]:
    """The worker's environment: the caller's, without credentials, as in hosted runs."""
    return {
        k: v for k, v in (os.environ if env is None else env).items() if not _SECRET_ENV.search(k)
    }


class SubprocessWorker:
    """A `yugos.submission.session.Worker` running the bundle in a local Python subprocess."""

    def __init__(self, bundle: Path, runtime_root: Path, *, log: Path, env=None) -> None:
        self.runtime_root = Path(runtime_root)
        self.runtime_root.mkdir(parents=True, exist_ok=False)
        self._log = Path(log).open("a", encoding="utf-8")
        self._process = subprocess.Popen(
            [sys.executable, "-I", "-m", "yugos.submission.worker", str(bundle), str(runtime_root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            encoding="utf-8",
            env=worker_environment(env),
        )
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader = threading.Thread(
            target=self._read, args=(self._process.stdout,), daemon=True
        )
        self._reader.start()

    def _read(self, stream: TextIO) -> None:
        try:
            while line := stream.readline(MAX_FRAME_BYTES + 2):
                self._lines.put(line)
        finally:
            self._lines.put(None)

    def exchange(self, message: str, *, timeout_s: float) -> str:
        try:
            self._process.stdin.write(message)
            self._process.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise OSError("worker input closed") from exc
        try:
            line = self._lines.get(timeout=max(timeout_s, 0.001))
        except queue.Empty:
            raise TimeoutError("worker did not answer in time") from None
        if line is None:
            raise EOFError("worker exited")
        return line

    def close(self, *, timeout_s: float) -> CleanupReport:
        forced = False
        try:
            self._process.stdin.close()
        except OSError:
            pass
        try:
            self._process.wait(timeout=max(timeout_s, 0.1))
        except subprocess.TimeoutExpired:
            forced = True
            self._process.kill()
            self._process.wait(timeout=10)
        self._log.close()
        # Prepared bundles are read-only; restore write access so they can be removed.
        for path in [self.runtime_root, *self.runtime_root.rglob("*")]:
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o700)
        shutil.rmtree(self.runtime_root, ignore_errors=True)
        return CleanupReport(
            reaped=self._process.poll() is not None and not self.runtime_root.exists(),
            forced_termination=forced,
        )


def reply(response, calls: list[CallRecord]) -> RouterReply:
    """The router's final provider answer as a RouterReply, keeping stop semantics."""
    finish = {
        "stop": "stop",
        "length": "length",
        "max_tokens": "length",
        "tool_calls": "tool_calls",
        "content_filter": "content_filter",
    }.get(response.finish_reason)
    if finish is None:
        raise ValueError(f"provider finish reason {response.finish_reason!r} is not supported")
    return RouterReply(
        content=response.text,
        provider_state=response.provider_state,
        finish_reason=finish,
        tool_calls=tuple(
            RouterToolCall(id=c.id, function=c.function, arguments=c.arguments)
            for c in response.tool_calls
        ),
        input_tokens=sum(c.response.input_tokens for c in calls),
        output_tokens=sum(c.response.output_tokens for c in calls),
    )


def pinned_call(call: RouterCall, pinned: Mapping[str, Any], model: ModelChoice) -> RouterCall:
    """`call` with the role's pinned settings applied over the ones the task requested."""
    if not pinned:
        return call
    config = call.request.config.merge(pinned)
    if config.reasoning_effort is not None and config.reasoning_effort != model.effort:
        raise ValueError("the pinned reasoning effort differs from the bound model's")
    return call.model_copy(update={"request": call.request.model_copy(update={"config": config})})


@dataclass
class _Attempt:
    """One sample's shared deadline and call identities across its generate calls."""

    sample_uuid: str
    deadline: float
    calls: set[str] = field(default_factory=set)


class LocalRouter(QuestionFailures):
    """The RouterClient for one loaded submission, plus accounted service-role calls.

    Router and service calls share the run's spend cap. Failures are classified at
    Yugos's shared question boundary, as in hosted evaluation.
    """

    def __init__(
        self,
        session,
        *,
        run_id: str,
        router_client,
        service_client=None,
        limits: CallLimits,
        receipts: ReceiptSink,
        service_receipts: ReceiptSink | None = None,
        record=None,
    ) -> None:
        self.session = session
        self.limits = limits
        self._record_extra = record or (lambda _call: None)
        self.meter = MeteredCalls(run_id, router_client, limits, self._recorder(receipts))
        self.service_meter = (
            MeteredCalls(
                f"{run_id}:services",
                service_client,
                limits,
                self._recorder(service_receipts or MemoryReceipts()),
                budget=self.meter.budget,
            )
            if service_client is not None
            else None
        )
        self.attempts: dict[str, _Attempt] = {}
        self.failure: Exception | None = None
        self.scoring: set[str] = set()
        self._sample_uuid: str | None = None
        self._started_ns = 0
        self._lock = anyio.Lock()

    def _recorder(self, sink):
        def record(call: CallRecord) -> None:
            self._record_extra(call)
            spend = call.response.spend
            if spend.status != "known":
                return
            sink.append(
                ProviderReceipt(
                    receipt_id=call.request.call_id,
                    sample_uuid=self._sample_uuid,
                    provider=call.request.model.provider_id,
                    model=call.request.model.model_id,
                    reasoning_effort=call.request.model.effort,
                    cost_usd=spend.usd,
                    started_ns=self._started_ns,
                    ended_ns=time.monotonic_ns(),
                )
            )

        return record

    def fail(self, exc: Exception) -> None:
        self.meter.budget.stop()
        if self.failure is None:
            self.failure = exc

    def _attempt(self, call: RouterCall) -> _Attempt:
        if self.failure is not None:
            raise RuntimeError(f"local evaluation stopped: {self.failure}")
        attempt = self.attempts.setdefault(
            call.scope.sample_uuid,
            _Attempt(call.scope.sample_uuid, time.monotonic() + self.limits.question_timeout_s),
        )
        if call.call_id in attempt.calls:
            raise RunError("Inspect repeated a model invocation")
        attempt.calls.add(call.call_id)
        return attempt

    def _dispatch(self, meter: MeteredCalls, action: CallAction, call: RouterCall, deadline):
        """Execute one call within the run and sample caps shared by router and services."""
        self._sample_uuid = call.scope.sample_uuid
        self._started_ns = time.monotonic_ns()
        return meter.execute(
            action,
            call.scope.sample_id,
            deadline=deadline,
            max_output_tokens=call.request.config.max_tokens,
        )

    async def __call__(self, call: RouterCall) -> RouterReply:
        with self.boundary(call):
            # Finish the blocking worker/provider transaction even if Inspect cancels.
            async with self._lock:
                return await anyio.to_thread.run_sync(self._generate, call)

    def stop_run(self, call: RouterCall, exc: Exception) -> None:
        self.fail(exc)

    def _generate(self, call: RouterCall) -> RouterReply:
        if call.scope.sample_uuid in self.scoring:
            raise RunError("the router cannot be called while its sample is scored")
        attempt = self._attempt(call)
        request_id = uuid.uuid4().hex
        first_call = len(self.meter.calls)
        remaining = attempt.deadline - time.monotonic()
        if remaining <= 0:
            raise QuestionLimitExceeded("question time limit reached", limit="time")
        choice = self.session.route(call.request, request_id, timeout_s=remaining)
        if time.monotonic() > attempt.deadline:
            raise QuestionLimitExceeded("question time limit reached", limit="time")
        action = CallAction(model=choice, input=call.request)
        response = self._dispatch(self.meter, action, call, attempt.deadline)
        return reply(response, self.meter.calls[first_call:])

    async def service_call(
        self, model: ModelChoice, call: RouterCall, *, pinned: Mapping[str, Any] | None = None
    ) -> RouterReply:
        """A grader or simulator role's call, metered separately under the shared caps.

        `pinned` generation settings (wire names) replace whatever the call requested;
        a pinned reasoning effort is carried by `model` instead.
        """
        with self.boundary(call, role="service"):
            async with self._lock:
                return await anyio.to_thread.run_sync(
                    self._service_call, model, pinned_call(call, pinned or {}, model)
                )

    def _service_call(self, model: ModelChoice, call: RouterCall) -> RouterReply:
        if self.service_meter is None:
            raise RuntimeError("no service roles are configured")
        attempt = self._attempt(call)
        deadline = attempt.deadline
        if call.scope.sample_uuid in self.scoring:
            deadline = time.monotonic() + self.limits.question_timeout_s
        action = CallAction(model=model, input=call.request)
        response = self._dispatch(self.service_meter, action, call, deadline)
        return reply(response, self.service_meter.calls[-1:])


# The run whose model calls the guard hooks below check, if any.
active_router: ContextVar[tuple[LocalRouter, frozenset[str]] | None] = ContextVar(
    "yugos_evals_local_router", default=None
)


@hooks(name="yugos-evals-local", description="Refuse unaccounted model calls in local runs")
class _AccountingGuard(Hooks):
    async def on_before_model_generate(self, data):
        active = active_router.get()
        if active is None:
            return
        router, accounted = active
        if data.cache is not None or data.model_name not in accounted:
            error = RuntimeError(f"unaccounted model call to {data.model_name} was refused")
            router.fail(error)
            # Inspect propagates limits raised by hooks before the API is called.
            raise LimitExceededError("operator", value=1, limit=0, message=str(error))

    async def on_sample_scoring(self, data):
        # `sample_id` here is the sample's UUID; its router calls are now refused.
        active = active_router.get()
        if active is not None:
            active[0].scoring.add(data.sample_id)

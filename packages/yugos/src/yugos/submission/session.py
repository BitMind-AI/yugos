"""Controller-side use of one worker supplied by an external execution service."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, replace
from enum import StrEnum
from threading import RLock
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from yugos.inference.actions import CallAction, CallOutput, ModelChoice, require_supported_models
from yugos.inference.chat import ModelInput

from .protocol import (
    CallRequested,
    Cancel,
    Failed,
    Failure,
    Finished,
    Load,
    Loaded,
    Message,
    Prepared,
    PreparedMetadata,
    Resume,
    Shutdown,
    Solve,
    Start,
    Stopped,
    decode_line,
    encode_line,
)


class CleanupReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    reaped: bool
    forced_termination: bool = False
    warnings: tuple[str, ...] = ()


class Worker(Protocol):
    """An already isolated worker. The adapter enforces deadlines and actual teardown."""

    def exchange(self, message: str, *, timeout_s: float) -> str: ...

    def close(self, *, timeout_s: float) -> CleanupReport: ...


class WorkerTimeouts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    prepare_s: float = Field(default=120, gt=0)
    load_s: float = Field(default=120, gt=0)
    decision_s: float = Field(default=8, gt=0)
    close_s: float = Field(default=10, gt=0)


class RequestState(StrEnum):
    """Progress established by the worker protocol, independent of benchmark scores."""

    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class SessionExit:
    """Why the session stopped, retaining the original failing request when known."""

    failure: Failure | None
    owner: Literal["submission", "infrastructure"]


@dataclass(frozen=True)
class RequestOutcome:
    """One request's progress, independent of the cause of session exit."""

    request_id: str
    state: RequestState
    session_exit: SessionExit | None = None
    decision_time_ns: int = 0


class SessionError(RuntimeError):
    def __init__(self, failure: Failure, *, outcome: RequestOutcome | None = None):
        super().__init__(failure.detail)
        self.failure = failure
        self.outcome = outcome

    @property
    def owner(self) -> Literal["submission", "infrastructure"]:
        """Attribute session exit to its original cause, including affected requests."""
        if self.outcome is not None and self.outcome.session_exit is not None:
            return self.outcome.session_exit.owner
        return self.failure.owner


class SubmissionSession:
    """Load declared dependencies once and isolate miner decision failures from later questions."""

    def __init__(
        self,
        worker: Worker,
        allowed_models: frozenset[ModelChoice],
        timeouts: WorkerTimeouts,
        *,
        max_active_requests: int = 1,
    ):
        if type(max_active_requests) is not int or not 1 <= max_active_requests <= 8:
            raise ValueError("max_active_requests must be between 1 and 8")
        self.max_active_requests = max_active_requests
        self._lock = RLock()
        self._worker = worker
        self._allowed = allowed_models
        self._timeouts = timeouts
        self._id = uuid.uuid4().hex
        self._sequence = 0
        self._serving = False
        self._session_exit: SessionExit | None = None
        self._requests: dict[str, RequestOutcome] = {}
        self._request_id: str | None = None
        self.metadata: PreparedMetadata | None = None
        self.cleanup: CleanupReport | None = None

    @property
    def session_exit(self) -> SessionExit | None:
        """Return the cause that stopped request admission."""
        with self._lock:
            return self._session_exit

    def request_outcome(self, request_id: str) -> RequestOutcome | None:
        """Read a request's latest outcome, including interruption by another request."""
        with self._lock:
            return self._requests.get(request_id)

    def decision_time_ns(self, request_id: str) -> int:
        """Return worker exchange time, excluding waiting for the session lock."""
        with self._lock:
            outcome = self._requests.get(request_id)
            return outcome.decision_time_ns if outcome else 0

    def _outcome(self, request_id, state):
        previous = self._requests.get(request_id)
        outcome = RequestOutcome(
            request_id, state, self._session_exit,
            previous.decision_time_ns if previous else 0,
        )
        self._requests[request_id] = outcome
        return outcome

    def _request_error(self, failure):
        outcome = self._requests[failure.request_id]
        return SessionError(failure, outcome=outcome)

    def _exchange(self, message: Message, timeout_s: float) -> Message:
        """Send a request and verify that the reply belongs to this session and sequence."""
        encoded = encode_line(message)
        started_ns = time.monotonic_ns()
        try:
            raw = self._worker.exchange(encoded, timeout_s=timeout_s)
        finally:
            if isinstance(message, (Solve, Resume)):
                outcome = self._requests[message.request_id]
                self._requests[message.request_id] = replace(
                    outcome,
                    decision_time_ns=outcome.decision_time_ns
                    + max(0, time.monotonic_ns() - started_ns),
                )
        try:
            reply = decode_line(raw)
            if reply.session_id != self._id or reply.sequence != message.sequence + 1:
                raise ValueError("mismatched worker envelope")
        except (TypeError, ValueError) as exc:
            self._protocol_error(message, "worker returned an invalid message", exc)
        self._sequence = message.sequence + 2
        return reply

    def _protocol_error(self, message: Message, detail: str, cause: Exception | None = None):
        """Classify a protocol violation by the phase in which it occurred."""
        solving = isinstance(message, (Solve, Resume, Cancel))
        if solving:
            failure = Failure(code="request.protocol", request_id=message.request_id, detail=detail)
            self._disable(failure, owner="submission")
            raise self._request_error(failure) from cause
        raise SessionError(Failure(code="handshake.protocol", detail=detail)) from cause

    def _disable(self, failure: Failure, *, owner: Literal["submission", "infrastructure"]) -> None:
        """Close an unsynchronized worker; later questions fail without sending it more work."""
        self._session_exit = SessionExit(failure, owner)
        for request_id, outcome in tuple(self._requests.items()):
            if outcome.state == RequestState.ACTIVE:
                state = (
                    RequestState.FAILED
                    if request_id == failure.request_id and owner == "submission"
                    else RequestState.INTERRUPTED
                )
                self._outcome(request_id, state)
        if failure.request_id is not None:
            self._outcome(
                failure.request_id,
                RequestState.FAILED if owner == "submission" else RequestState.INTERRUPTED,
            )
        self._serving = False
        self.close()

    def load(self, submission_sha256: str, manifest_sha256: str | None) -> PreparedMetadata:
        """Validate the miner's prepared identity and dependencies before loading code.

        Unsupported dependencies raise SessionError; evaluation terminates and
        its caller retains the failure and settles any actual spend.
        """
        if self._sequence or self.cleanup is not None:
            raise RuntimeError("a session can load only once")
        loading = False
        try:
            start = Start(
                session_id=self._id,
                sequence=0,
                submission_sha256=submission_sha256,
                expected_manifest_sha256=manifest_sha256,
                max_active_requests=self.max_active_requests,
            )
            reply = self._exchange(start, self._timeouts.prepare_s)
            if isinstance(reply, Failed) and reply.failure.code.startswith("preparation."):
                raise SessionError(reply.failure)
            if not isinstance(reply, Prepared):
                self._protocol_error(start, "worker did not return prepared metadata")
            metadata = reply.metadata
            if (
                metadata.submission_sha256 != submission_sha256
                or manifest_sha256 is not None
                and metadata.manifest_sha256 != manifest_sha256
            ):
                self._protocol_error(start, "prepared metadata does not match the evaluation")
            try:
                require_supported_models(metadata.required_models, self._allowed)
            except ValueError as exc:
                raise SessionError(
                    Failure(code="preparation.dependencies_unsupported", detail=str(exc)[:1000])
                ) from exc
            self._allowed = frozenset(metadata.required_models)
            self.metadata = metadata
            loading = True
            request = Load(session_id=self._id, sequence=self._sequence)
            reply = self._exchange(request, self._timeouts.load_s)
            if isinstance(reply, Failed) and reply.failure.code == "load.failed":
                raise SessionError(reply.failure)
            if not isinstance(reply, Loaded):
                self._protocol_error(request, "worker did not confirm model load")
        except (TimeoutError, EOFError, OSError) as exc:
            raise SessionError(
                Failure(
                    code="load.failed" if loading else "preparation.internal",
                    detail="worker startup failed",
                )
            ) from exc
        self._serving = True
        return metadata

    def solve(self, input: ModelInput, request_id: str, *, timeout_s: float) -> CallAction:
        """Begin a question and get its first call, replacing any unfinished question."""
        with self._lock:
            self._request_id = request_id
            return self._next(
                Solve(
                    session_id=self._id, sequence=self._sequence, request_id=request_id, input=input
                ),
                timeout_s,
            )

    def resume(
        self, output: CallOutput, *, timeout_s: float, request_id: str | None = None
    ) -> CallAction | None:
        """Advance the identified generator; concurrent callers must supply its ID."""
        with self._lock:
            if self.max_active_requests > 1 and request_id is None:
                raise ValueError("concurrent resume requires a request ID")
            return self._next(
                Resume(
                    session_id=self._id,
                    sequence=self._sequence,
                    request_id=request_id or self._request_id,
                    output=output,
                ),
                timeout_s,
            )

    def cancel(self, request_id: str) -> None:
        """Release a finished or abandoned request without disturbing other generators."""
        with self._lock:
            if not self._serving:
                return
            request = Cancel(session_id=self._id, sequence=self._sequence, request_id=request_id)
            try:
                reply = self._exchange(request, self._timeouts.decision_s)
                if not isinstance(reply, Finished) or reply.request_id != request_id:
                    self._protocol_error(request, "worker did not confirm request cleanup")
                self._outcome(request_id, RequestState.CANCELLED)
            except (TimeoutError, EOFError, OSError) as exc:
                failure = Failure(
                    code="request.protocol",
                    request_id=request_id,
                    detail="worker request cleanup failed",
                )
                self._disable(failure, owner="infrastructure")
                raise self._request_error(failure) from exc

    def _next(self, request: Solve | Resume, timeout_s: float) -> CallAction | None:
        """Permit calls only to the miner's declared pool; never substitute a model.

        Request errors let evaluation score the question zero. Transport/protocol
        failures close this worker; later questions fail without further exchanges.
        """
        if self._session_exit is not None:
            self._outcome(request.request_id, RequestState.INTERRUPTED)
            failure = self._session_exit.failure or Failure(
                code="request.failed", request_id=request.request_id, detail="worker session closed"
            )
            raise self._request_error(failure.model_copy(update={"request_id": request.request_id}))
        if not self._serving:
            raise RuntimeError("worker is not serving")
        self._outcome(request.request_id, RequestState.ACTIVE)
        try:
            reply = self._exchange(request, min(timeout_s, self._timeouts.decision_s))
        except (TimeoutError, EOFError, OSError):
            failure = Failure(
                code="request.failed",
                request_id=request.request_id,
                detail="worker decision timed out or exited",
            )
            self._disable(failure, owner="infrastructure")
            raise self._request_error(failure)
        if isinstance(reply, Failed):
            if (
                reply.failure.code not in {"request.failed", "request.invalid_action"}
                or reply.failure.request_id != request.request_id
            ):
                self._protocol_error(request, "worker returned an unrelated failure")
            self._outcome(request.request_id, RequestState.FAILED)
            raise self._request_error(reply.failure)
        if (
            not isinstance(reply, (CallRequested, Finished))
            or reply.request_id != request.request_id
        ):
            self._protocol_error(request, "worker returned an unrelated decision")
        if isinstance(reply, Finished) and isinstance(request, Resume):
            self._outcome(request.request_id, RequestState.COMPLETED)
            return None
        if isinstance(reply, Finished) or reply.action.model not in self._allowed:
            self._outcome(request.request_id, RequestState.FAILED)
            raise self._request_error(
                Failure(
                    code="request.invalid_action",
                    request_id=request.request_id,
                    detail="submission finished without a call or requested an undeclared model",
                )
            )
        return reply.action

    def close(self) -> CleanupReport:
        """Request shutdown and report whether the execution adapter confirmed teardown."""
        with self._lock:
            if self.cleanup is not None and self.cleanup.reaped:
                return self.cleanup
            deadline = time.monotonic() + self._timeouts.close_s
            warnings = list(self.cleanup.warnings) if self.cleanup else []
            try:
                if self._serving:
                    request = Shutdown(session_id=self._id, sequence=self._sequence)
                    reply = self._exchange(request, max(0, deadline - time.monotonic()))
                    if not isinstance(reply, Stopped):
                        warnings.append("worker did not acknowledge shutdown")
            except Exception:
                warnings.append("worker shutdown exchange failed")
            finally:
                self._serving = False
                if self._session_exit is None:
                    self._session_exit = SessionExit(None, "infrastructure")
                    for request_id, outcome in tuple(self._requests.items()):
                        if outcome.state == RequestState.ACTIVE:
                            self._outcome(request_id, RequestState.INTERRUPTED)
                try:
                    report = CleanupReport.model_validate(
                        self._worker.close(timeout_s=max(0, deadline - time.monotonic()))
                    )
                except Exception:
                    report = CleanupReport(
                        reaped=False, warnings=("worker teardown could not be confirmed",)
                    )
            self.cleanup = report.model_copy(
                update={
                    "warnings": tuple(dict.fromkeys((*warnings, *report.warnings))),
                    "forced_termination": report.forced_termination
                    or bool(self.cleanup and self.cleanup.forced_termination),
                }
            )
            return self.cleanup

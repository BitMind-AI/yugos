"""Execute selected model calls with bounded spend and caller-owned persistence."""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from threading import Condition, Event, RLock
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..spend import OpenReservation, Spend, apply_legacy_spend
from .actions import CallAction, ModelChoice
from .chat import ModelInput, ProviderState, ToolCall


class CallLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    max_cost_usd: float = Field(gt=0)
    max_question_cost_usd: float = Field(gt=0)
    max_calls_per_question: int = Field(default=16, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    call_timeout_s: float = Field(default=120, gt=0)
    question_timeout_s: float = Field(default=300, gt=0)


class ModelRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    call_id: str
    model: ModelChoice
    input: ModelInput
    max_output_tokens: int | None = Field(default=None, ge=1)
    timeout_s: float = Field(gt=0)


def committed_usd(calls) -> float:
    """Sum known bills. An open reservation stays in the reservation map."""
    return math.fsum(
        float(call.response.spend.usd) for call in calls if call.response.spend.status == "known"
    )


class ModelResponse(BaseModel):
    """Trusted adapter measurements. Known spend includes a real zero."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, revalidate_instances="always"
    )
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    provider_state: ProviderState | None = None
    spend: Spend
    latency_s: float = Field(ge=0)
    ttft_ms: float = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    finish_reason: str | None = None
    provider_request_id: str | None = None
    error: str | None = None

    @model_validator(mode="before")
    @classmethod
    def load_legacy_cost(cls, data: Any) -> Any:
        return apply_legacy_spend(data, amount_key="cost_usd", drop_amount=True)


class ModelClient(Protocol):
    """The caller supplies pricing, provider access, usage, and deadline enforcement.

    quote() must return an upper bound on the entire request's cost without
    dispatching inference. complete() must make one bounded attempt and include
    all known spend in its response, even on failure.
    With no output cap, quote() must use a finite provider/model bound or fail
    before dispatch. Omitting the cap never disables spend or time limits.
    """

    def quote(self, request: ModelRequest) -> float: ...

    def complete(self, request: ModelRequest) -> ModelResponse: ...


class CallRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["call"] = "call"
    run_id: str
    question_id: str
    step: int
    request: ModelRequest
    response: ModelResponse


class QuestionLimitExceeded(RuntimeError):
    """This question cannot finish within its call, spend, or time limits; `limit` names which."""

    def __init__(self, detail: str, *, limit: Literal["time", "calls", "cost"]):
        super().__init__(detail)
        self.limit = limit


class RequestRejected(ValueError):
    """The client rejected a request before reservation or provider dispatch."""


class CallError(RuntimeError):
    def __init__(
        self, detail: str, *, owner: str = "infrastructure", pending: CallRecord | None = None
    ):
        super().__init__(detail)
        self.owner = owner
        self.pending = pending


class UnresolvedSpend(CallError):
    """One provider attempt has unknown cost; its reservation remains held."""


class ProviderCallFailed(CallError):
    """One provider attempt failed with its cost accounted for."""


class CallBudget:
    """Reserve quoted spend atomically across calls, including shared service budgets."""

    def __init__(self, limits: CallLimits):
        self.limits = limits
        self.lock = RLock()
        self.changed = Condition(self.lock)
        self.stopped = Event()
        self._stop_reason: tuple[str, str, CallRecord | None] | None = None
        self.calls: list[CallRecord] = []
        self.reservations: dict[str, tuple[str, float]] = {}

    def stop(self, reason: CallError | None = None) -> None:
        """Prevent new admissions while already admitted calls retain their evidence."""
        with self.changed:
            if not self.stopped.is_set():
                self._stop_reason = (
                    (str(reason), reason.owner, reason.pending) if reason is not None else None
                )
                self.stopped.set()
            self.changed.notify_all()

    def _raise_if_stopped(self) -> None:
        """Give each waiter its own exception without losing the original owner."""
        if self.stopped.is_set():
            reason = self._stop_reason
            if reason is None:
                raise CallError("evaluation stopped; no further calls can be admitted")
            detail, owner, pending = reason
            raise CallError(detail, owner=owner, pending=pending)


@dataclass(frozen=True)
class CallIntent:
    """Local admission facts supplied to the host before provider dispatch.

    Preparation may replace the call ID. The meter subsequently reduces the
    request timeout by time spent preparing, before reserving or dispatching.
    """

    run_id: str
    question_id: str
    step: int
    request: ModelRequest
    quote_usd: float


class MeteredCalls:
    """Thread-safe call admission and settlement with caller-owned durable records.

    An optional host `fit_request` can narrow output to the available dollars.
    Admission waits for active holds first, then re-quotes the fitted request;
    its full upper bound must fit before any intent or provider call is made.
    """

    def __init__(
        self,
        run_id: str,
        client: ModelClient,
        limits: CallLimits,
        record: Callable[[CallRecord], None],
        *,
        budget: CallBudget | None = None,
        prepare_call: Callable[[CallIntent], ModelRequest] | None = None,
        fit_request: Callable[[ModelRequest, float], ModelRequest | None] | None = None,
    ):
        self.run_id, self.client, self.limits, self.record = run_id, client, limits, record
        self.budget = budget or CallBudget(limits)
        self.calls: list[CallRecord] = []
        self._steps: dict[str, int] = {}
        self.prepare_call = prepare_call
        self.fit_request = fit_request

    def _quote(self, request):
        quote = float(self.client.quote(request))
        if not math.isfinite(quote) or quote < 0:
            raise ValueError("invalid quote")
        return float(OpenReservation(reservation_usd=Decimal(str(quote))).reservation_usd)

    def _fit_budget(self, request, quote, question_id, deadline, clock):
        """Wait for active holds before asking the host to narrow an unaffordable response."""
        budget, limits = self.budget, self.budget.limits
        while True:
            budget._raise_if_stopped()
            remaining = deadline - clock()
            if remaining <= 0:
                raise QuestionLimitExceeded("question time limit reached", limit="time")
            question_calls = [c for c in budget.calls if c.question_id == question_id]
            available = limits.max_cost_usd - committed_usd(budget.calls) - math.fsum(
                cost for _, cost in budget.reservations.values()
            )
            question_available = limits.max_question_cost_usd - committed_usd(question_calls) - math.fsum(
                cost for qid, cost in budget.reservations.values() if qid == question_id
            )
            if quote <= min(available, question_available) + 1e-12:
                return request, quote
            completed = {c.request.call_id for c in budget.calls}
            active = [
                qid for call_id, (qid, _) in budget.reservations.items()
                if call_id not in completed
            ]
            if (quote > available + 1e-12 and active) or (
                quote > question_available + 1e-12 and question_id in active
            ):
                budget.changed.wait(timeout=min(remaining, 1.0))
                continue
            allowance = min(available, question_available)
            if allowance <= 0:
                return request, quote
            original = request.model_dump(exclude={"max_output_tokens"})
            fitted = self.fit_request(request.model_copy(deep=True), allowance)
            if fitted is None:
                return request, quote
            fitted = ModelRequest.model_validate(fitted.model_dump())
            if (
                fitted.model_dump(exclude={"max_output_tokens"}) != original
                or fitted.max_output_tokens is None
                or (request.max_output_tokens is not None
                    and fitted.max_output_tokens > request.max_output_tokens)
            ):
                raise ValueError("budget fitting may only narrow the output allowance")
            return fitted, self._quote(fitted)

    @property
    def cost_usd(self) -> float:
        with self.budget.lock:
            return committed_usd(self.calls)

    def restore(
        self, calls: tuple[CallRecord, ...], *, reservations: dict[str, float] | None = None
    ) -> None:
        """Charge verified durable history before admitting new calls.

        The host must verify the journal and account for every dispatched request.
        This restores accounting only; it never retries or replays a provider call.
        """
        reservations = dict(reservations or {})
        unresolved = {c.request.call_id for c in calls if c.response.spend.status != "known"}
        if set(reservations) != unresolved or any(
            not math.isfinite(value) or value <= 0 for value in reservations.values()
        ):
            raise ValueError("unresolved history requires its retained reservations")
        with self.budget.lock:
            if self.calls or self.budget.reservations or self._steps:
                raise ValueError("call history must be restored before execution")
            ids = {c.request.call_id for c in self.budget.calls}
            steps = set()
            for call in calls:
                key = (call.question_id, call.step)
                if (
                    call.run_id != self.run_id
                    or call.request.call_id in ids
                    or key in steps
                    or call.step < 0
                ):
                    raise ValueError(
                        "call history is incomplete, duplicated, or belongs to another run"
                    )
                ids.add(call.request.call_id)
                steps.add(key)
            self.calls.extend(calls)
            self.budget.calls.extend(calls)
            for call in calls:
                if call.request.call_id in reservations:
                    self.budget.reservations[call.request.call_id] = (
                        call.question_id,
                        reservations[call.request.call_id],
                    )
                self._steps[call.question_id] = max(
                    self._steps.get(call.question_id, 0), call.step + 1
                )

    def execute(
        self,
        action: CallAction,
        question_id: str,
        *,
        deadline: float,
        max_output_tokens: int | None = None,
        clock: Callable[[], float] | None = None,
    ) -> ModelResponse:
        """Reserve before dispatch; persist each response before releasing its reservation."""
        # The host may supply a measured-work clock so its persistence and
        # recovery waits cannot turn a completed answer into a miner timeout.
        clock = clock or time.monotonic
        budget = self.budget
        limits = budget.limits
        remaining = deadline - clock()
        if remaining <= 0:
            raise QuestionLimitExceeded("question time limit reached", limit="time")
        caps = [
            c
            for c in (
                self.limits.max_output_tokens,
                action.input.config.max_tokens,
                max_output_tokens,
            )
            if c is not None
        ]
        request = ModelRequest(
            call_id=uuid.uuid4().hex,
            model=action.model,
            input=action.input,
            max_output_tokens=min(caps) if caps else None,
            timeout_s=min(remaining, self.limits.call_timeout_s),
        )
        with budget.lock:
            budget._raise_if_stopped()
            question_calls = [c for c in budget.calls if c.question_id == question_id]
            pending = [cost for qid, cost in budget.reservations.values() if qid == question_id]
            if len(question_calls) + len(pending) >= limits.max_calls_per_question:
                raise QuestionLimitExceeded("question call limit reached", limit="calls")
            try:
                quote = self._quote(request)
                if self.fit_request is not None:
                    request, quote = self._fit_budget(request, quote, question_id, deadline, clock)
                    question_calls = [c for c in budget.calls if c.question_id == question_id]
                    pending = [cost for qid, cost in budget.reservations.values() if qid == question_id]
                    if len(question_calls) + len(pending) >= limits.max_calls_per_question:
                        raise QuestionLimitExceeded("question call limit reached", limit="calls")
            except (CallError, QuestionLimitExceeded):
                raise
            except RequestRejected:
                raise
            except Exception as exc:
                budget.stop()
                raise CallError("provider cost bound is unavailable") from exc
            committed = committed_usd(budget.calls)
            reserved = math.fsum(cost for _, cost in budget.reservations.values())
            if quote > limits.max_cost_usd - committed - reserved + 1e-12:
                failure = CallError("evaluation budget cannot fund the next call", owner="budget")
                budget.stop(failure)
                raise failure
            question_cost = committed_usd(question_calls)
            if quote > limits.max_question_cost_usd - question_cost - math.fsum(pending) + 1e-12:
                raise QuestionLimitExceeded(
                    "question budget cannot fund the next call", limit="cost"
                )
            remaining = deadline - clock()
            if remaining <= 0:
                raise QuestionLimitExceeded("question time limit reached", limit="time")
            request = request.model_copy(
                update={"timeout_s": min(remaining, self.limits.call_timeout_s)}
            )
            step = self._steps.get(question_id, 0)
            if self.prepare_call is not None:
                quoted_request = request.model_dump(exclude={"call_id"})
                try:
                    prepared = ModelRequest.model_validate(
                        self.prepare_call(
                            CallIntent(
                                self.run_id,
                                question_id,
                                step,
                                request,
                                quote,
                            )
                        )
                    )
                except Exception as exc:
                    budget.stop()
                    raise CallError("call intent could not be prepared") from exc
                if (
                    prepared.model_dump(exclude={"call_id"}) != quoted_request
                    or not prepared.call_id
                    or prepared.call_id in budget.reservations
                    or any(c.request.call_id == prepared.call_id for c in budget.calls)
                ):
                    budget.stop()
                    raise CallError("prepared call changed the request or reused its identity")
                request = prepared
            remaining = deadline - clock()
            if remaining <= 0:
                raise QuestionLimitExceeded("question time limit reached", limit="time")
            request = request.model_copy(update={"timeout_s": min(request.timeout_s, remaining)})
            self._steps[question_id] = step + 1
            budget.reservations[request.call_id] = (question_id, quote)
        started = time.monotonic()
        try:
            response = ModelResponse.model_validate(self.client.complete(request))
        except Exception as exc:
            # An exception cannot establish that a provider did not charge us.
            response = ModelResponse(
                text="",
                spend=OpenReservation(reservation_usd=Decimal(str(quote))),
                latency_s=max(0, time.monotonic() - started),
                error=type(exc).__name__,
            )
        if response.spend.status != "known":
            response = response.model_copy(
                update={"spend": OpenReservation(reservation_usd=Decimal(str(quote)))}
            )
        call = CallRecord(
            run_id=self.run_id,
            question_id=question_id,
            step=step,
            request=request,
            response=response,
        )
        with budget.lock:
            self.calls.append(call)
            budget.calls.append(call)
            try:
                self.record(call)
            except Exception as exc:
                budget.stop()
                raise CallError("call record could not be persisted", pending=call) from exc
            budget.changed.notify_all()
            if response.spend.status == "known" and float(response.spend.usd) > quote + 1e-12:
                budget.stop()
                raise CallError("provider cost exceeded its declared upper bound", pending=call)
            if response.spend.status != "known":
                # Keep this quote reserved, but let other questions spend the
                # remaining budget. The caller must record this sample as
                # incomplete and the run cannot settle until reconciliation.
                raise UnresolvedSpend(
                    "provider spend is incomplete and needs reconciliation", pending=call
                )
            del budget.reservations[request.call_id]
            if response.error is not None:
                raise ProviderCallFailed(f"provider request failed: {response.error}", pending=call)
        if clock() > deadline:
            raise QuestionLimitExceeded("question time limit reached", limit="time")
        return call.response

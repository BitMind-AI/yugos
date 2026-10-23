"""Expose a submitted router to Inspect as a model.

Inspect's solvers and agents call `generate()`; the RouterModel forwards each
call to an injected router client using the native Inspect request types. The
client (Ycore in production) runs the isolated router and its metered provider
proxy, which writes one `ProviderReceipt` per upstream call keyed by the
operator-side `SampleScope`. The router itself receives only `ModelInput`:
the conversation, tools and generation settings, never targets or metadata.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    Model,
    ModelAPI,
    ModelCall,
    ModelOutput,
    ModelUsage,
    RetryDecision,
    StopReason,
    modelapi,
)
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo
from inspect_ai.util import store
from pydantic import BaseModel, ConfigDict, Field

from yugos.inference.chat import PROVIDER_STATE_KEY, ModelInput, ProviderState

from .versions import SampleScope, active_sample_scope, require_pinned_inspect

require_pinned_inspect()

API_NAME = "yugos-router"
# Sample-store key recording a failure the submission owns (see RouterError).
ROUTER_FAILURE_KEY = "yugos.router_failure"


class RouterCall(BaseModel):
    """One generate call: the router's request plus operator-only attribution."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    submission_id: str = Field(min_length=1)
    call_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    scope: SampleScope
    request: ModelInput


class RouterToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(min_length=1)
    function: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class RouterReply(BaseModel):
    """The router's final message for one generate call, as observed by the operator."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    content: str = ""
    tool_calls: tuple[RouterToolCall, ...] = ()
    provider_state: ProviderState | None = None
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"]
    # Provider-reported token usage, informational only; spend comes from receipts.
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    # Host-measured model work, excluding its persistence and recovery waits.
    time_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)


RouterClient = Callable[[RouterCall], Awaitable[RouterReply]]


class RouterError(RuntimeError):
    """The submission failed this call. The sample errors and scores zero; never retried."""


class RouterUnavailable(RuntimeError):
    """Operator infrastructure failed before the router made any provider call.

    Only raise this when retrying cannot duplicate spend; Inspect retries it
    within the fairness profile's bounded retry budget.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


_STOP_REASONS: dict[str, StopReason] = {
    "stop": "stop",
    "length": "max_tokens",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
}


def _output(reply: RouterReply, model: str, elapsed_s: float) -> ModelOutput:
    tool_calls = [
        ToolCall(id=call.id, function=call.function, arguments=call.arguments)
        for call in reply.tool_calls
    ]
    message = ChatMessageAssistant(
        content=reply.content,
        tool_calls=tool_calls or None,
        model=model,
        source="generate",
        metadata={PROVIDER_STATE_KEY: reply.provider_state.model_dump(mode="json")}
        if reply.provider_state is not None
        else None,
    )
    return ModelOutput(
        model=model,
        choices=[
            ChatCompletionChoice(message=message, stop_reason=_STOP_REASONS[reply.finish_reason])
        ],
        usage=ModelUsage(
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            total_tokens=reply.input_tokens + reply.output_tokens,
        ),
        # Operator-measured; Inspect counts the rest of the call as waiting time.
        time=elapsed_s,
    )


@modelapi(name=API_NAME)
class RouterModel(ModelAPI):
    """Inspect ModelAPI for one submission's router."""

    def __init__(
        self,
        model_name: str,
        client: RouterClient | None = None,
        config: GenerateConfig = GenerateConfig(),
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("a submission ID is required")
        if base_url or api_key:
            raise ValueError("router models use an injected client")
        super().__init__(model_name=model_name, config=config)
        self._client = client

    def connection_key(self) -> str:
        # Submissions never share a connection pool within one process.
        return f"submission:{self.model_name}"

    def should_retry(self, ex: Exception) -> bool | RetryDecision:
        if isinstance(ex, RouterUnavailable):
            return RetryDecision.transient(ex.retry_after)
        return False

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput, ModelCall]:
        scope = active_sample_scope()
        if scope is None:
            raise RuntimeError("router calls must run inside an Inspect sample to be metered")
        if self._client is None:
            raise RuntimeError("router model has no bound client")
        request = ModelInput.from_inspect(input, tools, tool_choice, config)
        call = RouterCall(submission_id=self.model_name, scope=scope, request=request)
        started = time.perf_counter()
        try:
            reply = RouterReply.model_validate(await self._client(call))
        except RouterError as exc:
            store().set(ROUTER_FAILURE_KEY, str(exc) or type(exc).__name__)
            raise
        elapsed = time.perf_counter() - started
        duration = reply.time_s if reply.time_s is not None else elapsed
        if duration == 0:
            # Inspect 0.3.272 skips waiting-time reconciliation for a zero time.
            from inspect_ai._util.working import report_sample_waiting_time

            report_sample_waiting_time(elapsed)
        return _output(reply, self.model_name, duration), ModelCall(
            request={}, response={}, time=duration
        )


def router_model(
    client: RouterClient, *, submission_id: str, config: GenerateConfig | None = None
) -> Model:
    """Return an Inspect `Model` that sends every generate call to `client`."""
    config = config or GenerateConfig()
    return Model(RouterModel(submission_id, client, config), config)

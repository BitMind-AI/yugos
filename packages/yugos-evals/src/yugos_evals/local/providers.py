"""Execute a router's model calls with Inspect's own providers and the miner's API keys.

`LocalProviderClient` is a `yugos.inference.calls.ModelClient`: it bounds each call's cost
from the world card before dispatch, makes one provider attempt through
`inspect_ai.model.get_model`, and prices the reported usage with the card's
token rates. The Inspect model, adapter arguments and generation settings come
from `yugos.inspect_engine.providers`, the same mapping hosted execution uses.
Keys come from the environment and never leave this process.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Collection, Mapping
from decimal import Decimal

from inspect_ai._util.error import PrerequisiteError
from inspect_ai.model import (
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.model._providers.providers import (
    validate_anthropic_client,
    validate_google_client,
    validate_openai_client,
)

from yugos.inference.actions import ModelChoice
from yugos.inference.calls import ModelRequest, ModelResponse
from yugos.inference.chat import ModelInput, ProviderState, provider_messages, validated_tool_calls
from yugos.inspect_engine.providers import InspectGeneration, inspect_generation
from yugos.spend import KnownSpend, OpenReservation

from .pricing import reservation_microusd, usage_cost_microusd
from .world import DEFAULT_PROVIDERS, ProviderAccess, WorldCard

_FINISH = {
    "stop": "stop",
    "max_tokens": "length",
    "model_length": "length",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
}


class MissingCredentials(ValueError):
    """A provider the run needs has no API key in the environment."""


# Inspect's own SDK check and the package it needs, by Inspect model API.
_SDK_CHECKS = {
    "openai": (validate_openai_client, "openai"),
    "openai-api": (validate_openai_client, "openai"),
    "deepseek": (validate_openai_client, "openai"),
    "openrouter": (validate_openai_client, "openai"),
    "anthropic": (validate_anthropic_client, "anthropic"),
    "google": (validate_google_client, "google-genai"),
}


def require_provider_sdk(model_name: str) -> None:
    """Refuse a model whose Inspect provider SDK is missing or too old, before any dispatch."""
    check = _SDK_CHECKS.get(model_name.split("/", 1)[0])
    if check is None:
        return
    validate, package = check
    try:
        validate(model_name)
    except PrerequisiteError as exc:
        raise ValueError(
            f"{model_name} needs the {package!r} package at the version Inspect requires; "
            f"install it with `uv pip install -U {package}`"
        ) from exc


def generation(
    access: ProviderAccess,
    model: ModelChoice,
    input: ModelInput,
    max_output_tokens: int | None,
) -> InspectGeneration:
    """The shared Inspect model and settings for this call; nothing is dispatched."""
    return inspect_generation(
        access.provider_id,
        access.api_format,
        model,
        input,
        max_output_tokens=max_output_tokens,
    )


def _base_url(access: ProviderAccess) -> str | None:
    """The endpoint Inspect's adapter is given for this provider."""
    if access.api_format == "google":
        # The Google SDK adds its API version to the base; use its own endpoint.
        if access.api_base != DEFAULT_PROVIDERS["google"].api_base:
            raise ValueError("a Google api_base override is not supported locally")
        return None
    if access.api_format == "anthropic":
        return access.api_base.removesuffix("/v1")
    return access.api_base


def _finish_reason(output: ModelOutput, has_tool_calls: bool) -> str:
    reason = _FINISH.get(output.stop_reason)
    if reason is None:
        # Inspect reports an unmapped provider reason as "unknown".
        reason = "tool_calls" if has_tool_calls else "stop"
    return reason


class LocalProviderClient:
    """Price, bound and execute calls to world-card models with the caller's own keys."""

    def __init__(
        self,
        card: WorldCard,
        models: Collection[ModelChoice],
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.card = card
        self.env = os.environ if env is None else env
        pool = card.pool()
        unsupported = sorted(str(m.model_dump()) for m in models if m not in pool)
        if unsupported:
            raise ValueError(f"models without world-card prices and limits: {unsupported}")
        self.models = frozenset(models)
        self.access = {m.provider_id: card.provider(m.provider_id) for m in self.models}
        missing = sorted({a.key_env for a in self.access.values() if not self.env.get(a.key_env)})
        if missing:
            raise MissingCredentials(f"set your provider API keys: {', '.join(missing)}")
        for model in self.models:
            limits = self.card.limits[model.provider_id, model.model_id]
            limits.require_current()
            # Refuse unsupported efforts and formats before anything is dispatched.
            access = self.access[model.provider_id]
            _base_url(access)
            spec = generation(
                access, model, ModelInput.from_text("check"), limits.max_output_tokens
            )
            require_provider_sdk(spec.model_name)

    def _limits(self, request: ModelRequest):
        if request.model not in self.models:
            raise ValueError(f"model is not admitted for this run: {request.model.model_dump()}")
        limits = self.card.limits[request.model.provider_id, request.model.model_id]
        limits.require_current()
        if request.max_output_tokens is not None and request.max_output_tokens > (
            limits.max_output_tokens
        ):
            raise ValueError("request exceeds the reviewed model output ceiling")
        return limits

    def quote(self, request: ModelRequest) -> float:
        """Upper bound in USD from the card's limits; nothing is dispatched."""
        limits = self._limits(request)
        prices = self.card.prices[request.model.provider_id, request.model.model_id]
        wire = request.input.model_dump(mode="json", exclude_none=True)
        # A requested cap does not bound billed output on every provider.
        output_tokens = limits.max_output_tokens
        return reservation_microusd(prices, limits.input_tokens(wire), output_tokens) / 1_000_000

    def complete(self, request: ModelRequest) -> ModelResponse:
        """One provider attempt, priced from reported usage; failures report their known spend."""
        self._limits(request)
        started = time.monotonic()
        try:
            output, api_format = asyncio.run(
                asyncio.wait_for(self._generate(request), request.timeout_s)
            )
        except TimeoutError:
            # The provider may still bill an abandoned request.
            return self._unresolved(request, started, "provider call timed out")
        except Exception as exc:  # noqa: BLE001 - report every provider failure as a failed call
            return self._unresolved(request, started, self._describe(exc, request))
        prices = self.card.prices[request.model.provider_id, request.model.model_id]
        microusd = usage_cost_microusd(output.usage, prices)
        if microusd is None:
            spend = OpenReservation(reservation_usd=self._quote_decimal(request))
        else:
            spend = KnownSpend(usd=Decimal(microusd) / Decimal(1_000_000))
        usage = output.usage
        message = output.message
        try:
            tool_calls = validated_tool_calls(message)
        except ValueError as exc:
            tool_calls, error = (), str(exc)
        else:
            error = None
        return ModelResponse(
            text=message.text,
            tool_calls=tool_calls,
            provider_state=ProviderState(
                provider_id=request.model.provider_id,
                model_id=request.model.model_id,
                effort=request.model.effort,
                data={"api": api_format, "message": message.model_dump(mode="json")},
            ),
            spend=spend,
            latency_s=time.monotonic() - started,
            input_tokens=(
                usage.input_tokens
                + (usage.input_tokens_cache_read or 0)
                + (usage.input_tokens_cache_write or 0)
                if usage
                else 0
            ),
            output_tokens=usage.output_tokens if usage else 0,
            finish_reason=_finish_reason(output, bool(tool_calls)),
            error=error,
        )

    def _quote_decimal(self, request: ModelRequest) -> Decimal:
        """The admission quote for an unresolved call. Sub-cent quotes reserve one cent."""
        return OpenReservation(reservation_usd=Decimal(str(self.quote(request)))).reservation_usd

    def _unresolved(self, request: ModelRequest, started: float, error: str) -> ModelResponse:
        return ModelResponse(
            text="",
            spend=OpenReservation(reservation_usd=self._quote_decimal(request)),
            latency_s=max(0.0, time.monotonic() - started),
            error=error,
        )

    def _describe(self, exc: Exception, request: ModelRequest) -> str:
        """An error summary that never echoes the API key."""
        text = f"{type(exc).__name__}: {exc}"[:500]
        key = self.env.get(self.access[request.model.provider_id].key_env)
        return text.replace(key, "***") if key else text

    async def _generate(self, request: ModelRequest) -> tuple[ModelOutput, str]:
        choice = request.model
        access = self.access[choice.provider_id]
        limits = self.card.limits[choice.provider_id, choice.model_id]
        max_tokens = request.max_output_tokens
        if max_tokens is None and access.api_format == "anthropic":
            max_tokens = limits.max_output_tokens
        spec = generation(access, choice, request.input, max_tokens)
        # Transport-level additions only; the shared generation settings are unchanged.
        extra = {}
        if request.input.response_format == "json_object":
            if spec.api_format != "openai":
                raise ValueError(f"json_object output is not supported for {spec.api_format}")
            extra["extra_body"] = {"response_format": {"type": "json_object"}}
        if access.extra_headers:
            extra["extra_headers"] = dict(access.extra_headers)
        config = spec.config.merge(GenerateConfig(**extra)) if extra else spec.config
        transport = {}
        if spec.api_format == "anthropic":
            # With the SDK's default timeout, a large non-streaming request is
            # refused before dispatch; the hosted client sets its own timeout too.
            transport["timeout"] = request.timeout_s
        model = get_model(
            spec.model_name,
            config=config,
            base_url=_base_url(access),
            api_key=self.env[access.key_env],
            memoize=False,
            **spec.model_args,
            **transport,
        )
        tools = list(request.input.tools)
        tool_choice = request.input.tool_choice
        try:
            output = await model.api.generate(
                provider_messages(request.input, choice), tools, tool_choice, config
            )
        finally:
            await model.api.aclose()
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, Exception):
            raise output
        return output, spec.api_format

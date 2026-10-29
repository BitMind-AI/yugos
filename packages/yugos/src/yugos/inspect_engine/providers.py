"""The Inspect model and generation settings for one world-card provider call.

Hosted and local execution both reach provider models through Inspect's own
adapters, so a requested call must become the same Inspect model, adapter
arguments and `GenerateConfig` in both. Requested sampling settings and
reasoning effort pass to Inspect unchanged: Inspect decides what each model
accepts (for example, it omits temperature for OpenAI reasoning models) and how
effort is expressed natively. Callers own credentials, endpoints, HTTP transport,
admission and accounting, and must not rewrite what Inspect serializes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inspect_ai.model import GenerateConfig

from yugos.inference.actions import ModelChoice
from yugos.inference.chat import ModelInput, provider_state

API_FORMATS = ("openai", "openai_responses", "anthropic", "google")
RESPONSES_API_FORMAT = "openai_responses"
# OpenAI-compatible providers with their own Inspect adapter (native effort fields).
DEDICATED_COMPATIBLE_PROVIDERS = frozenset({"deepseek", "openrouter"})


@dataclass(frozen=True)
class InspectGeneration:
    """What to pass to `inspect_ai.model.get_model` and `generate` for one call.

    `model_args` are the adapter arguments that fix wire behavior (no streaming,
    no provider retries or storage); callers add `base_url`, `api_key`,
    `memoize=False` and any HTTP client of their own.
    """

    model_name: str
    api_format: str
    config: GenerateConfig
    model_args: dict[str, Any] = field(default_factory=dict)


def request_api_format(
    provider_id: str, api_format: str, model: ModelChoice, input: ModelInput
) -> str:
    """Use Responses for native OpenAI and SayGM GPT tool conversations.

    Submissions never choose an endpoint; a conversation with tools, earlier tool
    calls, or Responses continuation from this exact model moves to Responses.
    """
    if api_format not in API_FORMATS:
        raise ValueError(f"unsupported api_format {api_format!r}")
    if api_format == RESPONSES_API_FORMAT:
        return api_format
    native_gpt = provider_id == "openai" or (
        provider_id == "gm" and model.model_id.startswith("gpt-")
    )
    if not native_gpt or api_format != "openai":
        return api_format
    if input.tools:
        return RESPONSES_API_FORMAT
    for message in input.messages:
        state = provider_state(message)
        if getattr(message, "tool_calls", None) or (
            state is not None
            and (state.provider_id, state.model_id, state.effort)
            == (model.provider_id, model.model_id, model.effort)
            and state.data.get("api") == RESPONSES_API_FORMAT
        ):
            return RESPONSES_API_FORMAT
    return api_format


def inspect_generation(
    provider_id: str,
    api_format: str,
    model: ModelChoice,
    input: ModelInput,
    *,
    max_output_tokens: int | None,
    cache_prompt: bool = False,
) -> InspectGeneration:
    """The Inspect model name, adapter arguments and `GenerateConfig` for one call.

    `api_format` is the provider's configured format; `request_api_format`
    selects Responses where it applies. Anthropic requires an explicit output
    maximum (callers pass the model's reviewed limit when the request has none).
    `cache_prompt` lets Inspect place Anthropic prompt-cache breakpoints (direct
    and through OpenRouter); providers that cache automatically are unaffected.
    """
    if model.provider_id != provider_id:
        raise ValueError("model belongs to a different provider")
    if max_output_tokens is not None and max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    wire = request_api_format(provider_id, api_format, model, input)
    if wire == "anthropic" and max_output_tokens is None:
        raise ValueError("Anthropic requires a reviewed model-native output maximum")
    requested = input.config.model_dump(exclude_none=True)
    if max_output_tokens is not None:
        requested["max_tokens"] = (
            min(max_output_tokens, input.config.max_tokens)
            if input.config.max_tokens is not None
            else max_output_tokens
        )
    requested.update(max_retries=0, cache_prompt=cache_prompt)
    requested.setdefault("reasoning_summary", "none")
    if model.effort is not None:
        requested["reasoning_effort"] = model.effort
    config = GenerateConfig(**requested)
    model_id = model.model_id
    if wire == "google":
        return InspectGeneration(
            f"google/{model_id}", wire, config, {"streaming": False, "use_adc": False}
        )
    if wire == "anthropic":
        args: dict[str, Any] = {"streaming": False, "max_retries": 0}
        if cache_prompt:
            # Otherwise Inspect may switch a sample to the one-hour TTL after a
            # pause, from in-memory state a replay of the call cannot rebuild.
            args["cache_ttl"] = "5m"
        return InspectGeneration(f"anthropic/{model_id}", wire, config, args)
    if provider_id == "openai" or wire == RESPONSES_API_FORMAT:
        return InspectGeneration(
            f"openai/{model_id}",
            wire,
            config,
            {
                "streaming": False,
                "responses_api": wire == RESPONSES_API_FORMAT,
                "responses_store": False,
                "background": False,
                "service_tier": "default",
                "max_retries": 0,
            },
        )
    name = (
        f"{provider_id}/{model_id}"
        if provider_id in DEDICATED_COMPATIBLE_PROVIDERS
        else f"openai-api/{provider_id}/{model_id}"
    )
    return InspectGeneration(
        name, wire, config, {"stream": False, "strict_tools": False, "max_retries": 0}
    )

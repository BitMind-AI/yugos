"""World-card calls become the same Inspect model and settings the hosted service sends.

Expectations are copied from Ycore's Inspect provider (`_generate` and
tests/test_provider_effort.py at Ycore a0b25da). A mismatch means hosted and
local runs would send different requests for the same router call.
"""

import asyncio
import json

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser, GenerateConfig
from inspect_ai.tool import ToolInfo

from yugos.inference.chat import continuation_metadata

pytest.importorskip("inspect_ai")

from inspect_ai.model import get_model

from yugos.inference.actions import ModelChoice
from yugos.inference.chat import ModelInput, ProviderState, ToolCall
from yugos.inspect_engine.providers import inspect_generation, request_api_format

OPENAI_ARGS = {
    "streaming": False,
    "responses_api": False,
    "responses_store": False,
    "background": False,
    "service_tier": "default",
    "max_retries": 0,
}
COMPATIBLE_ARGS = {"stream": False, "strict_tools": False, "max_retries": 0}


def choice(provider, model="m", effort=None):
    return ModelChoice(provider_id=provider, model_id=model, effort=effort)


def sampled(temperature=0.0, seed=7):
    return ModelInput.from_text("hi").model_copy(
        update={"config": GenerateConfig(temperature=temperature, seed=seed)}
    )


LOOKUP = ToolInfo(name="lookup", description="d", parameters={"type": "object"})


@pytest.mark.parametrize(
    ("provider", "api_format", "model", "name", "wire", "args"),
    [
        ("openai", "openai", "gpt-5", "openai/gpt-5", "openai", OPENAI_ARGS),
        (
            "openai",
            "openai_responses",
            "gpt-5",
            "openai/gpt-5",
            "openai_responses",
            {**OPENAI_ARGS, "responses_api": True},
        ),
        (
            "anthropic",
            "anthropic",
            "claude-sonnet-4-5",
            "anthropic/claude-sonnet-4-5",
            "anthropic",
            {"streaming": False, "max_retries": 0},
        ),
        (
            "google",
            "google",
            "gemini-2.5-flash",
            "google/gemini-2.5-flash",
            "google",
            {"streaming": False, "use_adc": False},
        ),
        (
            "deepseek",
            "openai",
            "deepseek-chat",
            "deepseek/deepseek-chat",
            "openai",
            COMPATIBLE_ARGS,
        ),
        (
            "openrouter",
            "openai",
            "qwen/qwen3-235b-a22b",
            "openrouter/qwen/qwen3-235b-a22b",
            "openai",
            COMPATIBLE_ARGS,
        ),
        ("venice", "openai", "m", "openai-api/venice/m", "openai", COMPATIBLE_ARGS),
        # A provider explicitly configured for Responses must use that adapter.
        (
            "acme",
            "openai_responses",
            "m",
            "openai/m",
            "openai_responses",
            {**OPENAI_ARGS, "responses_api": True},
        ),
    ],
)
@pytest.mark.parametrize("effort", [None, "none", "low", "max"])
def test_calls_map_to_ycores_inspect_model_and_generate_config(
    provider, api_format, model, name, wire, args, effort
):
    spec = inspect_generation(
        provider, api_format, choice(provider, model, effort), sampled(), max_output_tokens=64000
    )
    assert (spec.model_name, spec.api_format, spec.model_args) == (name, wire, args)
    # Requested sampling and effort pass through; Inspect alone decides what is sent.
    assert spec.config.model_dump(exclude_none=True) == {
        "max_retries": 0,
        "max_tokens": 64000,
        "temperature": 0.0,
        "seed": 7,
        "cache_prompt": False,
        "reasoning_summary": "none",
        **({"reasoning_effort": effort} if effort else {}),
    }


@pytest.mark.parametrize(
    "provider, api_format, model",
    [("anthropic", "anthropic", "claude"), ("openrouter", "openai", "anthropic/claude")],
)
def test_prompt_caching_is_opt_in(provider, api_format, model):
    def config(**kwargs):
        return inspect_generation(
            provider,
            api_format,
            choice(provider, model),
            sampled(),
            max_output_tokens=10,
            **kwargs,
        ).config

    assert config().cache_prompt is False
    assert config(cache_prompt=True).cache_prompt is True


def test_unrequested_settings_stay_unset_and_invalid_requests_fail_before_dispatch():
    spec = inspect_generation(
        "openai", "openai", choice("openai"), ModelInput.from_text("hi"), max_output_tokens=None
    )
    assert spec.config.max_tokens is spec.config.temperature is spec.config.seed is None
    with pytest.raises(ValueError, match="reasoning_effort"):
        inspect_generation(
            "openai", "openai", choice("openai", effort="typo"), sampled(), max_output_tokens=10
        )
    with pytest.raises(ValueError, match="output maximum"):
        inspect_generation(
            "anthropic", "anthropic", choice("anthropic"), sampled(), max_output_tokens=None
        )
    with pytest.raises(ValueError, match="different provider"):
        inspect_generation("openai", "openai", choice("venice"), sampled(), max_output_tokens=10)
    with pytest.raises(ValueError, match="api_format"):
        inspect_generation("acme", "cohere", choice("acme"), sampled(), max_output_tokens=10)


@pytest.mark.parametrize("provider", ["openai", "gm"])
def test_gpt_tool_conversations_use_responses_and_only_own_continuation_counts(provider):
    model = choice(provider, "gpt-5", "low")
    plain = ModelInput.from_text("hi")
    assert request_api_format(provider, "openai", model, plain) == "openai"
    with_tools = plain.model_copy(update={"tools": (LOOKUP,)})
    assert request_api_format(provider, "openai", model, with_tools) == "openai_responses"
    called = ModelInput(
        messages=(
            ChatMessageUser(content="hi"),
            ChatMessageAssistant(
                tool_calls=(ToolCall(id="c1", function="lookup", arguments={}),), content=""
            ),
            ChatMessageTool(content="x", tool_call_id="c1"),
        )
    )
    assert request_api_format(provider, "openai", model, called) == "openai_responses"

    def continued(effort):
        state = ProviderState(
            provider_id=provider,
            model_id="gpt-5",
            effort=effort,
            data={"api": "openai_responses", "message": {}},
        )
        return ModelInput(
            messages=(
                ChatMessageUser(content="hi"),
                ChatMessageAssistant(content="a", metadata=continuation_metadata(state)),
                ChatMessageUser(content="again"),
            )
        )

    assert request_api_format(provider, "openai", model, continued("low")) == "openai_responses"
    assert request_api_format(provider, "openai", model, continued("high")) == "openai"
    # Providers without the native GPT Responses path keep their configured API.
    assert request_api_format("venice", "openai", choice("venice"), with_tools) == "openai"


@pytest.mark.parametrize(
    ("effort", "native_effort"), [(None, None), ("none", "none"), ("low", "low"), ("max", "xhigh")]
)
def test_saygm_gpt_tools_serialize_stateless_responses_without_changing_effort(
    effort, native_effort
):
    """Catch the rejected GPT-5.4 tools/reasoning Chat Completions request."""
    import httpx2

    sent = []
    model = choice("gm", "gpt-5.4", effort)
    input = ModelInput(messages=(ChatMessageUser(content="Look up a key"),), tools=(LOOKUP,))
    spec = inspect_generation("gm", "openai", model, input, max_output_tokens=64000)

    def capture(request):
        sent.append((str(request.url), json.loads(request.content)))
        raise RuntimeError("captured")

    adapter = get_model(
        spec.model_name,
        config=spec.config,
        base_url="https://api.saygm.com/v1",
        api_key="gm-test",
        memoize=False,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(capture)),
        **spec.model_args,
    )

    async def generate():
        try:
            await adapter.api.generate(list(input.messages), [LOOKUP], "auto", spec.config)
        except RuntimeError:
            pass
        finally:
            await adapter.api.aclose()

    asyncio.run(generate())
    [(url, body)] = sent
    assert url == "https://api.saygm.com/v1/responses"
    assert body["model"] == "gpt-5.4" and body["store"] is False
    assert spec.config.reasoning_effort == effort
    assert body.get("reasoning") == ({"effort": native_effort} if effort is not None else None)
    assert body["tools"][0]["name"] == "lookup"
    assert "previous_response_id" not in body and "reasoning_effort" not in body


def test_saygm_keeps_plain_gpt_and_other_model_protocols():
    plain = ModelInput.from_text("hi")
    tools = plain.model_copy(update={"tools": (LOOKUP,)})
    assert request_api_format("gm", "openai", choice("gm", "gpt-5.4"), plain) == "openai"
    assert request_api_format("gm", "openai", choice("gm", "glm-5.3"), tools) == "openai"
    assert (
        request_api_format("gm", "anthropic", choice("gm", "claude-haiku-5-5"), tools)
        == "anthropic"
    )


def serialized(provider, api_format, model, effort, base_url):
    """The body Inspect's adapter serializes for one call, captured before any network I/O."""
    httpx2 = pytest.importorskip("httpx2")
    sent = []

    def capture(request):
        sent.append(json.loads(request.content))
        raise RuntimeError("captured")

    spec = inspect_generation(
        provider,
        api_format,
        choice(provider, model, effort),
        sampled(seed=None),
        max_output_tokens=64000,
    )
    model = get_model(
        spec.model_name,
        config=spec.config,
        base_url=base_url,
        api_key="test",
        memoize=False,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(capture), timeout=300),
        **spec.model_args,
    )

    async def generate():
        try:
            await model.api.generate([ChatMessageUser(content="hi")], [], "none", spec.config)
        except Exception:  # noqa: BLE001 - the capture stops every request
            pass
        finally:
            await model.api.aclose()

    asyncio.run(generate())
    [body] = sent
    return body


# Rows from Ycore's test_inspect_temperature_translation_survives_quote_and_dispatch
# (Google needs Ycore's own client hook to capture, so it is not repeated here).
@pytest.mark.parametrize(
    ("provider", "model", "effort", "temperature_omitted", "base_url"),
    [
        ("openai", "gpt-5", "low", True, "https://api.openai.com/v1"),
        ("openai", "gpt-4.1", None, False, "https://api.openai.com/v1"),
        ("anthropic", "claude-sonnet-4-5", "low", True, "https://api.anthropic.com"),
        ("anthropic", "claude-sonnet-4-5", None, False, "https://api.anthropic.com"),
        ("deepseek", "deepseek-chat", "low", False, "https://api.deepseek.com/v1"),
        ("openrouter", "qwen/qwen3-235b-a22b", "low", False, "https://openrouter.ai/api/v1"),
    ],
)
def test_inspect_decides_temperature_as_it_does_for_ycore(
    provider, model, effort, temperature_omitted, base_url
):
    pytest.importorskip("anthropic" if provider == "anthropic" else "openai")
    api_format = "anthropic" if provider == "anthropic" else "openai"
    body = serialized(provider, api_format, model, effort, base_url)
    if temperature_omitted:
        assert "temperature" not in body
    else:
        assert body["temperature"] == 0.0


def test_inspect_chooses_native_reasoning_controls():
    pytest.importorskip("openai")
    deepseek = serialized("deepseek", "openai", "deepseek-chat", "none", "https://x.test/v1")
    assert deepseek["thinking"] == {"type": "disabled"} and "reasoning_effort" not in deepseek
    openrouter = serialized("openrouter", "openai", "q/q", "low", "https://x.test/v1")
    assert openrouter["reasoning"]["effort"] == "low" and "reasoning_effort" not in openrouter
    # An effort the model does not support is dropped by Inspect, not restored.
    unsupported = serialized("openai", "openai", "gpt-4.1", "low", "https://x.test/v1")
    assert "reasoning_effort" not in unsupported
    pytest.importorskip("anthropic")
    adaptive = serialized("anthropic", "anthropic", "claude-opus-4-7", "low", "https://x.test")
    assert (
        adaptive["thinking"]["type"] == "adaptive" and "budget_tokens" not in adaptive["thinking"]
    )
    assert adaptive["output_config"]["effort"] == "low"

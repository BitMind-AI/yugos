"""Tool identities and structured data must survive the worker boundary."""

import pytest
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolInfo
from pydantic import ValidationError

from yugos.inference.actions import CallAction, CallOutput, ModelChoice
from yugos.inference.chat import (
    ModelInput,
    ProviderState,
    ToolCall,
    continuation_metadata,
    provider_messages,
)
from yugos.submission.protocol import CallRequested, Resume, decode_line, encode_line


def test_worker_wire_preserves_tools_state_and_matching_results():
    model = ModelChoice(provider_id="fixture", model_id="router-choice")
    calls = (ToolCall(id="one", function="lookup", arguments={"name": "café"}),)
    state = ProviderState(**model.model_dump(), data={"signed": "opaque"})
    input = ModelInput(
        messages=(
            ChatMessageSystem(content="policy"),
            ChatMessageUser(content="request"),
            ChatMessageAssistant(
                tool_calls=calls, metadata=continuation_metadata(state), content=""
            ),
            ChatMessageTool(tool_call_id="one", content='{"ok":true}'),
        ),
        tools=(ToolInfo(name="lookup", parameters={"type": "object"}, description=""),),
        response_format="json_object",
    )
    fields = dict(session_id="a" * 32, sequence=1, request_id="r")
    action = CallRequested(**fields, action=CallAction(model=model, input=input))
    reply = Resume(
        **fields,
        output=CallOutput(
            text="", finish_reason="tool_calls", tool_calls=calls, provider_state=state
        ),
    )
    for message in (action, reply):
        assert decode_line(encode_line(message)) == message


def test_provider_continuation_accepts_inspect_tool_view_but_rejects_changed_arguments():
    from inspect_ai.tool import ToolCallContent

    model = ModelChoice(provider_id="openai", model_id="gpt-5", effort="low")
    saved_call = ToolCall(id="one", function="lookup", arguments={"name": "café"})
    saved = ChatMessageAssistant(content="", tool_calls=[saved_call])
    state = ProviderState(
        **model.model_dump(),
        data={"api": "openai_responses", "message": saved.model_dump(mode="json")},
    )

    def request(arguments):
        return ModelInput(
            messages=(
                ChatMessageUser(content="request"),
                ChatMessageAssistant(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="one",
                            function="lookup",
                            arguments=arguments,
                            view=ToolCallContent(
                                title="lookup", format="markdown", content="Rendered tool call"
                            ),
                        )
                    ],
                    metadata=continuation_metadata(state),
                ),
                ChatMessageTool(tool_call_id="one", content="result"),
            )
        )

    replayed = provider_messages(request({"name": "café"}), model)
    assert replayed[1].tool_calls == saved.tool_calls
    with pytest.raises(ValueError, match="continuation differs"):
        provider_messages(request({"name": "other"}), model)


@pytest.mark.parametrize("text", ["", "Actual assistant text"])
def test_transferred_tool_only_message_does_not_invent_text_or_mutate_input(text):
    from copy import deepcopy

    origin = ModelChoice(provider_id="anthropic", model_id="claude-sonnet-5-5")
    target = ModelChoice(provider_id="anthropic", model_id="claude-haiku-5-5")
    saved = ChatMessageAssistant(
        content=text, tool_calls=[ToolCall(id="tool-one", function="lookup", arguments={})]
    )
    state = ProviderState(
        **origin.model_dump(), data={"api": "anthropic", "message": saved.model_dump(mode="json")}
    )
    input = ModelInput(
        messages=(
            ChatMessageUser(content="request"),
            saved.model_copy(update={"metadata": continuation_metadata(state)}),
            ChatMessageTool(tool_call_id="tool-one", content="result"),
        )
    )
    before = deepcopy(input)
    transferred = provider_messages(input, target)
    assert transferred[1].content == ([] if text == "" else text)
    assert transferred[1].tool_calls == saved.tool_calls
    assert transferred[1].metadata is None
    assert transferred[2] == input.messages[2]
    assert input == before
    # Native continuation with the originating model remains byte-for-byte as saved.
    assert provider_messages(input, origin)[1].model_dump(mode="json") == saved.model_dump(
        mode="json"
    )


def test_transferred_tool_call_has_no_trailing_anthropic_prefill(monkeypatch):
    """Catch placeholder text after a tool remembered at native position zero."""
    import asyncio

    from inspect_ai.model._providers.anthropic import assistant_internal, message_param

    call = ToolCall(id="tool-switched", function="lookup", arguments={})
    input = ModelInput(
        messages=(
            ChatMessageUser(content="request"),
            ChatMessageAssistant(content="", tool_calls=[call]),
            ChatMessageTool(tool_call_id=call.id, content="tool result"),
        )
    )
    # Inspect retains this insertion position from a preceding tool-only response.
    monkeypatch.setitem(assistant_internal().client_tool_call_positions, call.id, 0)
    messages = provider_messages(
        input, ModelChoice(provider_id="anthropic", model_id="claude-haiku-5-5")
    )

    async def serialize():
        return [await message_param(message) for message in messages]

    wire = asyncio.run(serialize())
    assert wire[1]["content"] == [
        {"type": "tool_use", "id": call.id, "name": call.function, "input": {}}
    ]
    assert wire[2]["role"] == "user"
    assert wire[2]["content"][0]["type"] == "tool_result"
    assert wire[2]["content"][0]["tool_use_id"] == call.id


@pytest.mark.parametrize(
    "history",
    [
        [{"role": "tool", "tool_call_id": "missing", "content": "result"}],
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "a", "function": "f", "arguments": {}}],
            }
        ],
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "a", "function": "f", "arguments": {}}],
            },
            {"role": "user", "content": "interrupted"},
        ],
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "a", "function": "f", "arguments": {}}],
            },
            {"role": "tool", "tool_call_id": "wrong", "content": "result"},
        ],
    ],
)
def test_unmatched_or_unfinished_tools_are_rejected_before_dispatch(history):
    with pytest.raises(ValidationError):
        ModelInput(messages=history)


def test_unknown_response_format_is_rejected():
    with pytest.raises(ValidationError):
        ModelInput(messages=({"role": "user", "content": "question"},), response_format="xml")


def test_native_named_tool_and_generation_config_survive_worker_wire():
    # Catches a named tool, output cap or structured schema lost at worker transport.
    from inspect_ai.model import GenerateConfig, ResponseSchema
    from inspect_ai.tool import ToolFunction

    input = ModelInput(
        messages=(ChatMessageUser(content="Grade the answer"),),
        tools=(
            ToolInfo(name="submit", description="Submit a grade", parameters={"type": "object"}),
        ),
        tool_choice=ToolFunction(name="submit"),
        config=GenerateConfig(
            max_tokens=73,
            top_p=0.8,
            stop_seqs=["DONE"],
            response_schema=ResponseSchema(
                name="grade",
                json_schema={"type": "object", "properties": {"correct": {"type": "boolean"}}},
            ),
        ),
    )
    action = CallRequested(
        session_id="a" * 32,
        sequence=1,
        request_id="r",
        action=CallAction(model=ModelChoice(provider_id="fixture", model_id="judge"), input=input),
    )
    restored = decode_line(encode_line(action)).action.input
    assert restored == input
    assert isinstance(restored.messages[0], ChatMessageUser)
    assert isinstance(restored.tools[0], ToolInfo)
    assert isinstance(restored.tool_choice, ToolFunction)
    assert isinstance(restored.config, GenerateConfig)


@pytest.mark.parametrize(
    "config",
    [
        {"max_tokens": True},
        {"max_tokens": 1.5},
        {"max_retries": 1},
        {"extra_headers": {"Authorization": "secret"}},
        {"extra_body": {"model": "unadmitted"}},
        {"num_choices": 2},
    ],
)
def test_worker_requests_cannot_change_host_execution_or_evade_output_caps(config):
    # Catches native config accidentally gaining authority over paid-call execution.
    with pytest.raises(ValidationError):
        ModelInput(messages=(ChatMessageUser(content="request"),), config=config)


@pytest.mark.parametrize("host_cap", [5, 11])
def test_native_output_cap_is_bounded_before_quote_and_dispatch(host_cap):
    # Catches config.max_tokens being lost or bypassing the host's output ceiling.
    import time

    from inspect_ai.model import GenerateConfig

    from yugos.inference.calls import CallLimits, MeteredCalls, ModelResponse

    calls = []

    class Client:
        def quote(self, request):
            calls.append(request)
            return 0.01

        def complete(self, request):
            assert request.max_output_tokens == calls[0].max_output_tokens
            return ModelResponse(text="answer", cost_usd=0.01, latency_s=0, finish_reason="stop")

    input = ModelInput(
        messages=(ChatMessageUser(content="request"),), config=GenerateConfig(max_tokens=7)
    )
    meter = MeteredCalls(
        "r",
        Client(),
        CallLimits(max_cost_usd=1, max_question_cost_usd=1, max_output_tokens=host_cap),
        lambda call: None,
    )
    meter.execute(
        CallAction(model=ModelChoice(provider_id="fixture", model_id="m"), input=input),
        "q",
        deadline=time.monotonic() + 30,
    )
    assert calls[0].max_output_tokens == min(7, host_cap)

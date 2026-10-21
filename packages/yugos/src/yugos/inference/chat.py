"""Inspect request data shared by miners and evaluator-owned model calls."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Literal

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageUser,
    ContentText,
    GenerateConfig,
)
from inspect_ai.tool import ToolCall, ToolChoice, ToolFunction, ToolInfo
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROVIDER_STATE_KEY = "yugos.provider_state"
# Execution controls belong to the host, not to an isolated submission.
HOST_CONFIG_FIELDS = frozenset(
    {
        "max_retries",
        "timeout",
        "attempt_timeout",
        "stream_idle_timeout",
        "max_connections",
        "adaptive_connections",
        "cache",
        "cache_prompt",
        "batch",
        "fallback_models",
    }
)


class ProviderState(BaseModel):
    """Opaque native continuation data, reusable only with its originating model."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    provider_id: str
    model_id: str
    effort: str | None = None
    data: dict[str, Any]


def provider_state(message: ChatMessage) -> ProviderState | None:
    """Read the model-bound continuation carried in an assistant's public metadata."""
    state = (message.metadata or {}).get(PROVIDER_STATE_KEY)
    return ProviderState.model_validate(state) if state is not None else None


def continuation_metadata(state: ProviderState | None) -> dict | None:
    """Attach model-bound continuation to an Inspect assistant message."""
    return {PROVIDER_STATE_KEY: state.model_dump(mode="json")} if state is not None else None


def public_message(message: ChatMessage) -> ChatMessage:
    """Keep native request content while excluding operator metadata and reasoning text."""
    result = deepcopy(message)
    state = provider_state(message)
    result.metadata = {PROVIDER_STATE_KEY: state.model_dump(mode="json")} if state else None
    result.source = None
    result.id = ""
    if isinstance(result, ChatMessageAssistant):
        result.model = None
    if not isinstance(result.content, str):
        if any(part.type not in {"text", "reasoning"} for part in result.content):
            raise ValueError("only text request content is supported")
        result.content = [part for part in result.content if isinstance(part, ContentText)]
    return result


def request_config(config: GenerateConfig) -> GenerateConfig:
    """Separate the task's generation settings from Inspect's execution controls."""
    return GenerateConfig(**config.model_dump(exclude=HOST_CONFIG_FIELDS, exclude_none=True))


def validated_tool_calls(message: ChatMessageAssistant) -> tuple[ToolCall, ...]:
    """Reject malformed provider tool calls before they enter a worker conversation."""
    calls = message.tool_calls or []
    if any(
        not call.id or not call.function or call.parse_error or call.type != "function"
        for call in calls
    ):
        raise ValueError("provider returned invalid native tool arguments")
    if len({call.id for call in calls}) != len(calls):
        raise ValueError("provider reused a native tool call ID")
    return tuple(calls)


def provider_messages(input: ModelInput, model) -> list[ChatMessage]:
    """Replay signed continuation only with its originating model and matching public data."""
    messages = []
    for message in input.messages:
        state = provider_state(message)
        if state is not None and (state.provider_id, state.model_id, state.effort) == (
            model.provider_id,
            model.model_id,
            model.effort,
        ):
            if set(state.data) != {"api", "message"}:
                raise ValueError("unsupported continuation")
            saved = ChatMessageAssistant.model_validate(state.data["message"])
            if not isinstance(saved.content, str) and any(
                part.type not in {"text", "reasoning"} for part in saved.content
            ):
                raise ValueError("unsupported continuation content")
            # Inspect may add a display-only view after the provider returns.
            # Keep every executable tool-call field bound to the saved response.
            saved_calls = tuple(replace(call, view=None) for call in validated_tool_calls(saved))
            public_calls = tuple(replace(call, view=None) for call in message.tool_calls or ())
            if saved.text != message.text or saved_calls != public_calls:
                raise ValueError("continuation differs from assistant message")
            messages.append(saved)
        else:
            messages.append(message.model_copy(deep=True, update={"metadata": None}))
    return messages


def validate_request_config(config: GenerateConfig) -> None:
    """Reject generation settings that would escape host policy or its cost bounds."""
    fields = config.model_dump(exclude_none=True)
    if set(fields) & HOST_CONFIG_FIELDS:
        raise ValueError("request config contains host execution controls")
    if config.extra_headers or config.extra_body:
        raise ValueError("request config cannot override provider headers or body")
    if config.max_tokens is not None and (
        type(config.max_tokens) is not int or config.max_tokens <= 0
    ):
        raise ValueError("Inspect max_tokens must be a positive integer")
    if config.modalities:
        raise ValueError("only text output is supported")
    if config.num_choices not in (None, 1) or config.best_of not in (None, 1):
        raise ValueError("one completion per call is supported")


class ModelInput(BaseModel):
    """Inspect's model request fields, plus Yugos's portable JSON-object output mode."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    tools: tuple[ToolInfo, ...] = ()
    tool_choice: ToolChoice = "auto"
    config: GenerateConfig = Field(default_factory=GenerateConfig)
    response_format: Literal["text", "json_object"] = "text"

    @classmethod
    def from_text(cls, text: str) -> ModelInput:
        return cls(messages=(ChatMessageUser(content=text),))

    @classmethod
    def from_inspect(cls, messages, tools, tool_choice, config) -> ModelInput:
        """Capture an Inspect call without translating its message, tool or config types."""
        return cls(
            messages=tuple(public_message(message) for message in messages),
            tools=deepcopy(tuple(tools)),
            tool_choice=deepcopy(tool_choice),
            config=request_config(config),
        )

    @field_validator("messages")
    @classmethod
    def public_message_ids(cls, messages):
        # Inspect IDs track its log; request equality must depend on request data.
        return tuple(message.model_copy(deep=True, update={"id": ""}) for message in messages)

    @model_validator(mode="before")
    @classmethod
    def validate_output_cap(cls, data):
        if isinstance(data, dict):
            config = data.get("config", {})
            cap = config.get("max_tokens") if isinstance(config, dict) else config.max_tokens
            if cap is not None and (type(cap) is not int or cap <= 0):
                raise ValueError("Inspect max_tokens must be a positive integer")
        return data

    @model_validator(mode="after")
    def valid_request(self):
        validate_request_config(self.config)
        if self.response_format == "json_object" and self.config.response_schema is not None:
            raise ValueError("JSON-object mode and response_schema cannot both be requested")
        names = {tool.name for tool in self.tools}
        if len(names) != len(self.tools):
            raise ValueError("tool names must be distinct")
        if self.tool_choice == "any" and not self.tools:
            raise ValueError("required tool choice needs tool definitions")
        if isinstance(self.tool_choice, ToolFunction) and self.tool_choice.name not in names:
            raise ValueError("named tool choice needs a matching tool definition")
        pending, seen = set(), set()
        for message in self.messages:
            if message.metadata and set(message.metadata) != {PROVIDER_STATE_KEY}:
                raise ValueError("request contains private message metadata")
            if provider_state(message) is not None and message.role != "assistant":
                raise ValueError("only assistant messages carry provider state")
            if not isinstance(message.content, str) and any(
                part.type != "text" for part in message.content
            ):
                raise ValueError("only text request content is supported")
            if message.role == "tool":
                if message.tool_call_id not in pending:
                    raise ValueError("tool result has no matching outstanding call")
                pending.remove(message.tool_call_id)
            else:
                if pending:
                    raise ValueError("all tool results must precede the next conversation turn")
                for call in (
                    validated_tool_calls(message)
                    if isinstance(message, ChatMessageAssistant)
                    else ()
                ):
                    if call.id in seen:
                        raise ValueError("tool call ID reused in conversation")
                    seen.add(call.id)
                    pending.add(call.id)
        if pending:
            raise ValueError("model input contains unanswered tool calls")
        return self

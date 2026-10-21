"""Model calls requested by a submission and the outputs it can inspect."""

from __future__ import annotations

from collections.abc import Collection
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from .chat import ModelInput, ProviderState, ToolCall

ACTION_SCHEMA = "yugos.action.v2"


class ModelChoice(BaseModel):
    """A provider, model, and optional reasoning effort selected for a call."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    provider_id: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=128)
    effort: str | None = Field(default=None, max_length=32)


def _distinct_models(models: tuple[ModelChoice, ...]) -> tuple[ModelChoice, ...]:
    """Reject duplicate dependency choices when parsing miner declarations."""
    if len(set(models)) != len(models):
        raise ValueError("required_models must contain distinct model choices")
    return models


RequiredModels = Annotated[
    tuple[ModelChoice, ...], Field(min_length=1), AfterValidator(_distinct_models)
]


def require_supported_models(
    required: Collection[ModelChoice], supported: Collection[ModelChoice]
) -> None:
    """Reject miner-declared dependencies outside the operator's supported pool.

    Raises ValueError naming the unsupported choices; callers stop preparation
    before loading miner code or spending on provider calls.
    """
    missing = [model.model_dump() for model in required if model not in supported]
    if missing:
        raise ValueError(f"unsupported required models: {missing}")


class CallAction(BaseModel):
    """One model call with the exact conversation and tools chosen by the submission."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)
    schema_name: Literal[ACTION_SCHEMA] = Field(alias="schema", default=ACTION_SCHEMA)
    kind: Literal["call"] = "call"
    model: ModelChoice
    input: ModelInput


class CallOutput(BaseModel):
    """Provider text and completion status visible to the submission."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    text: str
    finish_reason: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    provider_state: ProviderState | None = None

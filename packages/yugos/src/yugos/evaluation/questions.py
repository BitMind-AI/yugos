"""Pack records hold selected Sample data, task configs and competition labels."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .measurements import WorkloadClass


class Question(BaseModel):
    """One packed Sample with its task config and private evaluation labels."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    question_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    expected_answer: str
    inspect_task: dict
    inspect_sample: dict
    workload_class: WorkloadClass
    source_benchmark: str = ""
    difficulty_decile: int | None = Field(default=None, ge=0, le=9, strict=True)
    # Calibrated panel easiness (weighted mean native score); sets the hard tier.
    panel_solve_share: Decimal | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def drop_empty_grading(cls, value):
        """Older packs stored `grading: null`. The grade is not a question field."""
        if isinstance(value, dict) and "grading" in value and value["grading"] is None:
            return {key: item for key, item in value.items() if key != "grading"}
        return value

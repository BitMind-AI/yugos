"""Policy-independent question outcomes and workload labels."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

WorkloadClass = Literal["chat_qa", "math_code", "agentic"]
_NON_ANSWER = {"length", "max_tokens", "empty", "unparseable"}


class Outcome(BaseModel):
    """A trusted measurement, never a miner's claim about its own performance."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    score: float = Field(ge=0, le=1)
    cost_usd: float = Field(ge=0)
    latency_s: float = Field(ge=0)
    ttft_ms: float = Field(default=0, ge=0)
    unrealized: bool = False
    finish_reason: str | None = None

    @property
    def non_answer(self) -> bool:
        return self.unrealized or (self.finish_reason or "").strip().lower() in _NON_ANSWER


class QuestionMeasurement(BaseModel):
    """One question's observed grade, cost, and timing without competition policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    question_id: str = Field(min_length=1)
    workload_class: WorkloadClass
    source_benchmark: str = ""
    difficulty_decile: int | None = Field(default=None, ge=0, le=9, strict=True)
    outcome: Outcome
    # Live grading can detect truncation even without a transport finish reason.
    truncated: bool = False
    grade_rule: str = ""

    @property
    def non_answer(self) -> bool:
        rule = self.grade_rule.strip().lower()
        return self.outcome.non_answer or self.truncated or rule in _NON_ANSWER | {"truncated"}

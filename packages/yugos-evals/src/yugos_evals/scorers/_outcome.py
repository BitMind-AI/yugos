"""Grade outcomes as Inspect scores, and the completion gate every native scorer applies.

Settled outcomes (a correct, incorrect, invalid or truncated answer) become a
numeric score in [0, 1]. Outcomes the grader could not establish (an invalid
reference, a grader error, an unusable judgment) become `Score.unscored` with a
reason, never a silent zero. Score metadata `code` and `status` use the Yugos
grade vocabulary and are the only fields public logs keep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inspect_ai.scorer import Score
from inspect_ai.solver import TaskState

from yugos.evaluation.grades import GradeCode

# Inspect stop reasons for an answer cut off by a token limit.
_TRUNCATED = frozenset({"max_tokens", "model_length"})
# Inspect's stop reason for an answer the provider blocked (a content filter or
# refusal). The block is the answer the router chose to return.
_BLOCKED = frozenset({"content_filter"})
# Settled model-under-test outcomes and Inspect's standard reason for each.
_REASONS = {
    GradeCode.NOT_ATTEMPTED: "no_response",
    GradeCode.INVALID_ANSWER: "invalid_response_format",
    GradeCode.TRUNCATED: "truncated",
}


@dataclass(frozen=True)
class Outcome:
    """One grade: a `GradeCode`, its status and non-reference diagnostics."""

    code: GradeCode
    status: str = "graded"
    score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        code = GradeCode(self.code)
        object.__setattr__(self, "code", code)
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("grade score must be in [0, 1]")
        if (code == GradeCode.CORRECT) != (self.score > 0):
            raise ValueError("exactly the correct grades have a positive score")

    @property
    def settled(self) -> bool:
        return self.status == "graded" or self.code == GradeCode.TRUNCATED


def to_score(outcome: Outcome, *, answer: str | None = None, failure: str = "scoring_failed"):
    """The outcome as an Inspect score; unsettled outcomes are unscored with a reason."""
    metadata = {"code": outcome.code.value, "status": outcome.status, **outcome.details}
    if not outcome.settled:
        return Score.unscored(reason=failure, answer=answer, metadata=metadata)
    return Score(
        value=outcome.score,
        answer=answer,
        reason=_REASONS.get(outcome.code),
        metadata=metadata,
    )


def completion_gate(state: TaskState) -> Outcome | None:
    """The outcome for an answer that must not be graded, as Yugos grading gates it.

    A sample that ended at a limit before the router answered has no completion
    and is not attempted, like an empty answer; a truncated answer is a graded
    zero; an answer the provider blocked is graded incorrect; a completion that
    ended for another reason than `stop` (a tool call) needs review.
    """
    if not state.output.choices:
        return Outcome(GradeCode.NOT_ATTEMPTED, details={"cause": "no_response"})
    reason = state.output.stop_reason
    if reason in _TRUNCATED:
        return Outcome(
            GradeCode.TRUNCATED, status="skipped", details={"cause": "incomplete_response"}
        )
    if reason in _BLOCKED:
        return Outcome(GradeCode.INCORRECT, details={"cause": "provider_blocked"})
    if reason != "stop":
        return Outcome(
            GradeCode.NEEDS_REVIEW,
            status="unavailable",
            details={"cause": "unknown_completion_status", "stop_reason": reason},
        )
    return None

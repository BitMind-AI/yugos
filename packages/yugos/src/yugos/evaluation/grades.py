"""Terminal grade outcomes shared by grading, measurement and competition rules.

Kept free of grader implementations so measurement can use them without loading
the grading stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal


class GradeCode(StrEnum):
    """Finite vocabulary of terminal grading outcomes."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    INVALID_ANSWER = "invalid_answer"
    NOT_ATTEMPTED = "not_attempted"
    TRUNCATED = "truncated"
    INVALID_GOLD = "invalid_gold"
    GRADER_UNAVAILABLE = "grader_unavailable"
    GRADER_ERROR = "grader_error"
    NEEDS_REVIEW = "needs_review"


GradeStatus = Literal["graded", "skipped", "unavailable", "error"]

_CODES_BY_STATUS = {
    "graded": {
        GradeCode.CORRECT,
        GradeCode.INCORRECT,
        GradeCode.INVALID_ANSWER,
        GradeCode.NOT_ATTEMPTED,
    },
    "skipped": {GradeCode.TRUNCATED},
    "unavailable": {GradeCode.INVALID_GOLD, GradeCode.GRADER_UNAVAILABLE, GradeCode.NEEDS_REVIEW},
    "error": {GradeCode.GRADER_ERROR},
}


@dataclass(frozen=True, kw_only=True)
class GradingResult:
    """A score plus enough provenance to audit how it was produced."""

    score: float
    grader: str
    code: GradeCode
    status: GradeStatus = "graded"
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        code = GradeCode(self.code)
        object.__setattr__(self, "code", code)
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("grade score must be in [0, 1]")
        if self.status != "skipped" and not self.grader:
            raise ValueError("non-skipped results must identify their grader")
        if self.status != "graded" and self.score != 0.0:
            raise ValueError("non-graded results must fail closed")
        if self.status not in _CODES_BY_STATUS:
            raise ValueError(f"unknown grade status: {self.status}")
        if code not in _CODES_BY_STATUS[self.status]:
            raise ValueError(f"{code} is incompatible with status={self.status}")
        if code == GradeCode.CORRECT and self.score == 0.0:
            raise ValueError("correct grades must have a positive score")
        if code != GradeCode.CORRECT and self.score != 0.0:
            raise ValueError("only correct grades may have a positive score")

    @classmethod
    def skipped(cls, code: GradeCode) -> "GradingResult":
        """Represent an answer that intentionally never reached a grader."""
        return cls(
            score=0.0,
            grader="",
            code=code,
            status="skipped",
        )

"""Bounded Math-Verify grading with the frozen comparison rules."""

from __future__ import annotations

from functools import partial

import anyio
from inspect_ai.scorer import Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

from yugos.evaluation.grades import GradeCode

from ._math_process import MathWorkerFailed, compare
from ._outcome import Outcome, completion_gate, to_score

GRADER_VERSION = 1


def grade_math(gold: str, text: str, **options) -> Outcome:
    """Compare in a disposable process; resource failures remain unscored."""
    try:
        return Outcome(**compare("grade", gold, text, **options))
    except MathWorkerFailed as exc:
        return Outcome(
            GradeCode.GRADER_ERROR,
            status="error",
            details={"component": "math_worker", "cause": exc.cause},
        )
    except (TypeError, ValueError):
        return Outcome(
            GradeCode.GRADER_ERROR,
            status="error",
            details={"component": "math_worker", "cause": "invalid_result"},
        )


def reference_exclusion(answer: str) -> str | None:
    """Screen a reference with the same bounded, isolated comparison rules."""
    try:
        return compare("reference", answer)
    except MathWorkerFailed as exc:
        return f"unparseable_reference: math comparison {exc.cause}"


@scorer(metrics=[mean(), stderr()])
def math_answer_scorer(answer_format: str = "auto") -> Scorer:
    """Grade the final answer against the target with Math-Verify (`grade_math`)."""

    if answer_format not in {"auto", "answer_line"}:
        raise ValueError("unsupported math answer format")

    async def score(state: TaskState, target: Target):
        gated = completion_gate(state)
        if gated is not None:
            return to_score(gated)
        options = {}
        if answer_format == "answer_line":
            options = {
                "answer_format": answer_format,
                "question": state.input_text,
            }
        outcome = await anyio.to_thread.run_sync(
            partial(grade_math, target.text, state.output.completion, **options)
        )
        return to_score(outcome)

    return score

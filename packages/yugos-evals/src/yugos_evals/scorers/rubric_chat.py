"""Exact, marked-choice, and partial-credit grading for original chat questions.

A rubric score is the share of required lines present in the response. Any
positive share is a correct grade with that score; a miss is incorrect. A
choice answer counts only an explicit ``Answer: <letter>`` line; the last
such line wins, and prose without that marker is invalid. The grader does
not call a model.
"""

from __future__ import annotations

import re

from inspect_ai.scorer import Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

from yugos.evaluation.grades import GradeCode

from ._outcome import Outcome, completion_gate, to_score

# A choice answer is its own line: "Answer: C", optionally with trailing
# punctuation. Isolated letters, option lists, and the word "answer" in
# prose are not markers. The last matching line is the answer.
_ANSWER_MARKER = re.compile(r"(?im)^[ \t]*answer[ \t]*:[ \t]*([abcd])[ \t]*[.,;:!?)]*[ \t]*$")


def _norm(text: str) -> str:
    return " ".join(text.strip().casefold().split())


def contains_fact(text: str, fact: str) -> bool:
    """True when the normalized fact is a whole whitespace-delimited phrase."""
    needle = _norm(fact)
    if not needle:
        return False
    return f" {needle} " in f" {_norm(text)} "


def grade_rubric_chat(gold: str, completion: str, metadata: dict | None) -> Outcome:
    """Grade one completion against its frozen target and sample metadata."""
    expected = (gold or "").strip()
    response = completion or ""
    if not expected:
        return Outcome(GradeCode.INVALID_GOLD, status="unavailable")
    if not response.strip():
        return Outcome(GradeCode.NOT_ATTEMPTED)
    kind = (metadata or {}).get("format") or "exact"
    if kind == "exact":
        if _norm(response) == _norm(expected):
            return Outcome(GradeCode.CORRECT, score=1.0)
        return Outcome(GradeCode.INCORRECT)
    if kind == "choice":
        found = _ANSWER_MARKER.findall(response.replace("\r\n", "\n").replace("\r", "\n"))
        if not found:
            return Outcome(GradeCode.INVALID_ANSWER)
        if found[-1].upper() == expected.upper():
            return Outcome(GradeCode.CORRECT, score=1.0)
        return Outcome(GradeCode.INCORRECT)
    if kind == "rubric":
        facts = (metadata or {}).get("rubric")
        if not isinstance(facts, list) or not facts or any(not str(fact).strip() for fact in facts):
            return Outcome(GradeCode.INVALID_GOLD, status="unavailable")
        hits = sum(contains_fact(response, fact) for fact in facts)
        if hits == 0:
            return Outcome(GradeCode.INCORRECT)
        return Outcome(GradeCode.CORRECT, score=hits / len(facts))
    return Outcome(
        GradeCode.INVALID_GOLD,
        status="unavailable",
        details={"cause": "unknown_format"},
    )


@scorer(metrics=[mean(), stderr()])
def rubric_chat_scorer() -> Scorer:
    """Grade a frozen chat sample from its `format` metadata."""

    async def score(state: TaskState, target: Target):
        gated = completion_gate(state)
        if gated is not None:
            return to_score(gated)
        outcome = grade_rubric_chat(target.text, state.output.completion, state.metadata)
        return to_score(outcome, answer=state.output.completion)

    return score

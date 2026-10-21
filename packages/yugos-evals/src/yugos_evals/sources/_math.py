"""The boxed final-answer prompt contract and reference screening shared by math sources."""

from __future__ import annotations

from collections.abc import Iterable

from inspect_ai.dataset import Sample

from ..scorers.math import reference_exclusion


def boxed_prompt(problem: str) -> str:
    """The retained AIME prompt around the unmodified problem text."""
    return (
        "Solve the following math problem step by step. The last line of your response "
        "should only contain your final answer inside a \\boxed{} command.\n\n"
        f"{problem}\n\nRemember to put your final answer on the last line using the format "
        "\\boxed{$ANSWER} where $ANSWER is the answer to the problem."
    )


def math_samples(
    rows: Iterable[tuple[str, str, str, dict]],
) -> tuple[list[Sample], list[dict[str, str]]]:
    """(ID, problem, answer, metadata) rows as boxed-prompt samples, and the IDs excluded and why."""
    samples, excluded = [], []
    for sample_id, problem, answer, metadata in rows:
        if not problem.strip() or not answer.strip():
            raise ValueError(f"{sample_id} needs a problem and an answer")
        answer = answer.strip()
        reason = reference_exclusion(answer)
        if reason:
            excluded.append({"id": sample_id, "reason": reason})
            continue
        samples.append(
            Sample(id=sample_id, input=boxed_prompt(problem), target=answer, metadata=metadata)
        )
    return samples, excluded

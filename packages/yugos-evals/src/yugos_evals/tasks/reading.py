"""Reading tasks absent from the pinned Inspect Evals release."""

from __future__ import annotations

from typing import Literal

from inspect_ai import Task, task
from inspect_ai.scorer import choice
from inspect_ai.solver import generate, multiple_choice

from ..scorers.reading import hotpot_answer, longbench_answer
from ..sources.reading import load_belebele, load_hotpotqa, load_longbench_v2


def reading_components(source):
    """Native execution components, independent of source acquisition."""
    if source == "belebele":
        return multiple_choice(), choice()
    if source == "longbench_v2":
        return generate(), longbench_answer()
    if source == "hotpotqa":
        return generate(), hotpot_answer()
    raise ValueError(f"unknown reading source: {source}")


@task
def reading(
    source: Literal["belebele", "longbench_v2", "hotpotqa"] = "belebele",
    languages: list[str] | None = None,
    lengths: list[str] | None = None,
    data_dir: str | None = None,
) -> Task:
    """Load pinned source inputs during preparation and retain complete contexts."""
    if source == "belebele":
        if lengths is not None:
            raise ValueError("Belebele selects languages, not lengths")
        loaded = load_belebele(languages, data_dir=data_dir)
    elif source == "longbench_v2":
        if languages is not None:
            raise ValueError("LongBench selects lengths, not languages")
        loaded = load_longbench_v2(lengths, data_dir=data_dir)
    elif source == "hotpotqa":
        if languages is not None or lengths is not None:
            raise ValueError("HotpotQA has no language or length variants")
        loaded = load_hotpotqa(data_dir=data_dir)
    else:
        raise ValueError(f"unknown reading source: {source}")
    solver, scorer = reading_components(source)
    return Task(
        dataset=loaded.samples,
        solver=solver,
        scorer=scorer,
        metadata={"source": source, **loaded.metadata()},
        version="1-A",
    )

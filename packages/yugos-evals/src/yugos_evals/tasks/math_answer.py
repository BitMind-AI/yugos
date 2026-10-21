"""`yugos_evals/math_answer`: boxed final answers graded with Math-Verify."""

from __future__ import annotations

from typing import Literal

from inspect_ai import Task, task
from inspect_ai.solver import generate

from ..scorers.math import math_answer_scorer
from ..sources.arxivmath import load_arxivmath
from ..sources.livemathbench import DEFAULT_SUBSETS, load_livemathbench

VERSION = "1-A"


@task
def math_answer(
    source: Literal["arxivmath", "livemathbench"] = "arxivmath",
    months: list[str] | None = None,
    subsets: list[str] | None = None,
    data_dir: str | None = None,
) -> Task:
    """Problems from a pinned public math source, answered in one generation.

    `arxivmath` takes `months` (`YYMM`, default every pinned month);
    `livemathbench` takes `subsets` (default `202505/all_en`) and needs
    `HF_TOKEN` unless `data_dir` holds staged files. The loader runs here, at
    preparation; excluded references are listed in the task metadata.
    """
    if source == "arxivmath":
        if subsets is not None:
            raise ValueError("arxivmath selects months, not subsets")
        loaded = load_arxivmath(months, data_dir=data_dir)
    elif source == "livemathbench":
        if months is not None:
            raise ValueError("livemathbench selects subsets, not months")
        loaded = load_livemathbench(subsets or DEFAULT_SUBSETS, data_dir=data_dir)
    else:
        raise ValueError(f"unknown math source: {source!r}")
    return Task(
        dataset=loaded.samples,
        solver=generate(),
        scorer=math_answer_scorer(),
        metadata={"source": source, **loaded.metadata()},
        version=VERSION,
    )

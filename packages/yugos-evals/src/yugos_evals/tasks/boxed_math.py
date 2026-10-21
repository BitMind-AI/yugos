"""`yugos_evals/boxed_math`: original arithmetic graded with Math-Verify.

The factory does not load a dataset. Competition packs freeze their own samples
and `yugos_evals.prepared` rebuilds this task from those samples.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from ..scorers.math import math_answer_scorer

VERSION = "1"


@task
def boxed_math() -> Task:
    """One generation, graded by `math_answer_scorer` against the frozen sample.

    Inspect rejects an empty dataset. The placeholder is replaced before a pack
    is frozen and is not a competition question.
    """
    return Task(
        dataset=[Sample(id="placeholder", input="placeholder", target="1")],
        solver=generate(),
        scorer=math_answer_scorer(),
        version=VERSION,
    )

"""`yugos_evals/rubric_chat`: original chat items graded without a model judge.

The factory does not load a dataset. Competition packs freeze their own samples
and `yugos_evals.prepared` rebuilds this task from those samples.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from ..scorers.rubric_chat import rubric_chat_scorer

VERSION = "1"


@task
def rubric_chat() -> Task:
    """One generation, graded by `rubric_chat_scorer` against the frozen sample.

    Inspect rejects an empty dataset. The placeholder is replaced before a pack
    is frozen and is not a competition question.
    """
    return Task(
        dataset=[Sample(id="placeholder", input="placeholder", target="placeholder")],
        solver=generate(),
        scorer=rubric_chat_scorer(),
        version=VERSION,
    )

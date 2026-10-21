"""`yugos_evals/pairwise_judge`: Arena-Hard v2.0 preference against pinned baselines."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.solver import generate

from yugos.inspect_engine import RolePin, bind_grader_roles, role_pins_metadata

from ..scorers.pairwise import GRADER_ROLE, judge_pin_guard, pairwise_judge_scorer
from ..sources.arenahard_v2 import CATEGORIES, load_arenahard_v2

VERSION = "1-A"


@task
def pairwise_judge(
    categories: list[str] | None = None,
    judge_model: str | None = None,
    data_dir: str | None = None,
) -> Task:
    """Each prompt sent unchanged; the `grader` role judges it against the category baseline.

    The `grader` role must be bound explicitly to a model other than the one
    under test (`judge_model`, if given, pins which); each sample checks the
    binding before the router is called and fails closed otherwise. Upstream's
    v2.0 judge is `openai/gpt-4.1-2025-04-14`; the catalog pins `openai/gpt-5.4-mini`
    and its settings instead.
    """
    loaded = load_arenahard_v2(categories or list(CATEGORIES), data_dir=data_dir)
    metadata = loaded.metadata()
    if judge_model:
        metadata.update(role_pins_metadata({GRADER_ROLE: RolePin(model=judge_model)}))
    task = Task(
        dataset=loaded.samples,
        setup=[judge_pin_guard(judge_model)] if judge_model else None,
        solver=generate(),
        scorer=pairwise_judge_scorer(judge_model),
        metadata=metadata,
        version=VERSION,
    )
    return bind_grader_roles(task, [GRADER_ROLE])

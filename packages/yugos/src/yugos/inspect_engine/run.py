"""Run one submission's competition eval with the shared fairness profile."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from inspect_ai import Task, eval
from inspect_ai.log import EvalLog
from inspect_ai.model import Model

from .fairness import FairnessProfile, competition_eval_options
from .isolation import read_role_pins, require_grader_roles, require_role_pins, task_grader_roles


def run_competition_eval(
    task: Task,
    model: Model,
    *,
    log_dir: str,
    required_roles: Iterable[str] = (),
    model_roles: Mapping[str, str | Model] | None = None,
    profile: FairnessProfile | None = None,
    **options: Any,
) -> EvalLog:
    """Evaluate `task` with the router `model` under the cohort's fairness profile.

    Grader roles are checked before any call so a judge can never fall back to
    the router: the roles in the frozen task plus any in `required_roles`, against
    `model_roles` over the task's own bindings. Roles the task pins must be bound
    to their pinned model with their pinned settings. `options` may add settings
    such as `display`, not override the fairness profile.
    """
    pins = read_role_pins(task.metadata)
    required = sorted({*task_grader_roles(task), *required_roles, *pins})
    bound = {**(task.model_roles or {}), **(model_roles or {})}
    require_grader_roles(required, bound, model_under_test=model)
    require_role_pins(pins, bound)
    fairness = competition_eval_options(profile)
    overridden = fairness.keys() & options.keys()
    if overridden:
        raise ValueError(f"the fairness profile fixes {sorted(overridden)}")
    [log] = eval(
        task,
        model=model,
        model_roles=dict(model_roles) if model_roles else None,
        log_dir=log_dir,
        **fairness,
        **options,
    )
    return log

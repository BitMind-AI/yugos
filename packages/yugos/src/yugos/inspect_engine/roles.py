"""Resolve each frozen task's service roles before runner-specific admission."""

import re
from collections.abc import Mapping
from dataclasses import dataclass

from yugos.inference.actions import ModelChoice
from yugos.inference.chat import validate_request_config

from .frozen import TaskConfig
from .isolation import RolePin, read_role_pins


@dataclass(frozen=True)
class TaskRole:
    """One task's selected service model and optional generation pin."""

    model: ModelChoice
    pin: RolePin | None


def resolve_task_roles(
    task: TaskConfig,
    defaults: Mapping[str, ModelChoice],
) -> dict[str, TaskRole]:
    """Use task pins first and defaults for unpinned roles; runners check admission.

    A pinned role's model carries the pin's `reasoning_effort`, so hosted and local
    runners admit, bill and record the effort the role's calls actually request.
    """
    if any(
        not re.fullmatch(r"[a-z][a-z0-9_]*", role) for role in (*defaults, *task.required_roles)
    ):
        raise ValueError("model roles must use plain role names")
    pins = read_role_pins(task.metadata)
    if set(pins) - set(task.required_roles):
        raise ValueError("task pins roles it does not require")
    bound = {}
    for role in task.required_roles:
        pin = pins.get(role)
        if pin is not None:
            provider, model = pin.model.split("/", 1)
            choice = ModelChoice(
                provider_id=provider, model_id=model, effort=pin.params.get("reasoning_effort")
            )
            validate_request_config(pin.config())
        else:
            if role not in defaults:
                raise ValueError(f"model role {role!r} needs a default binding")
            choice = ModelChoice.model_validate(defaults[role])
        bound[role] = TaskRole(choice, pin)
    return bound

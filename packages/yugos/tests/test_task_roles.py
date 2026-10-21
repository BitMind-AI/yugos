"""Task role selection cannot leak another task's model or settings into a sample."""

from types import SimpleNamespace

import pytest

from yugos.inference.actions import ModelChoice
from yugos.inspect_engine import RolePin, resolve_task_roles, role_pins_metadata


def task(model, **params):
    return SimpleNamespace(
        required_roles=("grader",),
        metadata=role_pins_metadata({"grader": RolePin(model=model, params=params)}),
    )


def test_tasks_select_their_own_models_and_settings():
    first = ModelChoice(provider_id="openai", model_id="first")
    second = ModelChoice(provider_id="anthropic", model_id="second")
    definitions = [
        task("openai/first", temperature=0),
        task("anthropic/second", max_tokens=1024),
        task("openai/first", temperature=0.7),
    ]
    bindings = [resolve_task_roles(t, {"grader": second})["grader"] for t in definitions]
    assert [b.model for b in bindings] == [first, second, first]
    assert [b.pin.params for b in bindings] == [
        {"temperature": 0},
        {"max_tokens": 1024},
        {"temperature": 0.7},
    ]


def test_scoped_role_aliases_fail_closed():
    choice = ModelChoice(provider_id="openai", model_id="first")
    with pytest.raises(ValueError, match="plain role names"):
        resolve_task_roles(task("openai/first"), {"grader@other": choice})


def test_unpinned_roles_require_a_default():
    definition = SimpleNamespace(required_roles=("grader",), metadata={})
    choice = ModelChoice(provider_id="openai", model_id="first")
    with pytest.raises(ValueError, match="default binding"):
        resolve_task_roles(definition, {})
    binding = resolve_task_roles(definition, {"grader": choice})["grader"]
    assert binding.model == choice and binding.pin is None


def test_pinned_reasoning_effort_selects_the_role_model_effort():
    """Runners admit and bill the effort a pinned role's calls request, not the default."""
    binding = resolve_task_roles(
        task("openai/judge", reasoning_effort="low", max_tokens=16384), {}
    )["grader"]
    assert binding.model == ModelChoice(provider_id="openai", model_id="judge", effort="low")

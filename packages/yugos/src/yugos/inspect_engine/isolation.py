"""Keep answer keys and grading away from the model under test and outside readers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from inspect_ai import Task
from inspect_ai._util.registry import is_registry_object, registry_info
from inspect_ai.log import EvalError, EvalLog, EvalSample, EvalSampleReductions, EvalSampleScore
from inspect_ai.model import GenerateConfig, Model, ModelOutput, get_model, model_roles
from inspect_ai.scorer import Score
from inspect_ai.solver import Generate, Solver, TaskState, solver
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .router_model import API_NAME as PROXY_API

# Score metadata that carries no reference material.
_PUBLIC_SCORE_METADATA = ("code", "status")
# Task metadata listing the grader roles a task needs bound (see `bind_grader_roles`).
GRADER_ROLES_KEY = "yugos_grader_roles"
# Task metadata pinning each service role's model and settings: {role: RolePin as JSON}.
ROLE_PINS_KEY = "yugos_role_pins"
# The exception name on a traceback's last line, e.g. `inspect_ai.util.SandboxError: ...`.
_EXCEPTION_NAME = re.compile(r"^(?:[A-Za-z_]\w*\.)*([A-Za-z_]\w*)(?::|$)")


class UnboundGraderRole(ValueError):
    """A judge or grader role is missing or would fall back to the model under test.

    Inspect resolves an unbound role to the model being evaluated, which would
    send the answer key to the miner's router and let it grade itself.
    """


class RolePinMismatch(UnboundGraderRole):
    """A pinned service role is bound to another model or with other settings."""


class RolePin(BaseModel):
    """The model a service role must be bound to and the generation settings it must use.

    `params` are Inspect `GenerateConfig` fields (e.g. `temperature`, `max_tokens`);
    only the pinned keys are checked, so an empty `params` pins the model alone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    model: str = Field(pattern=r"^[^/\s]+/\S+$")
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("params")
    @classmethod
    def _generate_config_fields(cls, params: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(params) - set(GenerateConfig.model_fields))
        if unknown:
            raise ValueError(f"role pin settings are not GenerateConfig fields: {unknown}")
        if any(value is None for value in params.values()):
            raise ValueError("role pin settings need values; omit a setting to leave it unpinned")
        GenerateConfig(**params)
        return dict(sorted(params.items()))

    def config(self) -> GenerateConfig:
        """The settings to bind the role's model with."""
        return GenerateConfig(**self.params)


def role_pins_metadata(pins: Mapping[str, RolePin | Mapping[str, Any]]) -> dict[str, Any]:
    """Task metadata recording each role's model and generation settings."""
    validated = {role: RolePin.model_validate(pin) for role, pin in sorted(pins.items())}
    return {
        ROLE_PINS_KEY: {role: pin.model_dump(mode="json") for role, pin in validated.items()},
    }


def read_role_pins(metadata: Mapping[str, Any] | None) -> dict[str, RolePin]:
    """The role pins recorded in one task's metadata."""
    metadata = metadata or {}
    pins = {
        role: RolePin.model_validate(pin) for role, pin in metadata.get(ROLE_PINS_KEY, {}).items()
    }
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*", role) for role in pins):
        raise ValueError("model roles must use plain role names")
    return pins


def served_model(model: str | Model) -> str:
    """The `provider/model` a role serves; a metered proxy is named for the model it forwards to."""
    api, _, name = str(model).partition("/")
    return name if api == PROXY_API else str(model)


def require_role_pins(
    pins: Mapping[str, RolePin], model_roles: Mapping[str, str | Model] | None
) -> None:
    """Raise unless every pinned role is bound to its pinned model with its pinned settings.

    A role bound by name carries no settings, so it satisfies only a pin without any.
    """
    roles = model_roles or {}
    for role, pin in pins.items():
        bound = roles.get(role)
        if not bound:
            raise UnboundGraderRole(f"model role {role!r} must be bound explicitly")
        expected = pin.config()
        for model in bound if isinstance(bound, list) else [bound]:
            if served_model(model) != pin.model:
                raise RolePinMismatch(f"model role {role!r} must be the pinned {pin.model}")
            config = model.config if isinstance(model, Model) else GenerateConfig()
            differing = sorted(k for k in pin.params if getattr(config, k) != getattr(expected, k))
            if differing:
                raise RolePinMismatch(
                    f"model role {role!r} must use its pinned settings for {differing}"
                )


def require_grader_roles(
    required: Iterable[str],
    model_roles: Mapping[str, str | Model] | None,
    *,
    model_under_test: str | Model,
) -> None:
    """Raise unless every required role is bound to a model other than the one under test."""
    roles = model_roles or {}
    tested = str(model_under_test)
    for role in required:
        bound = roles.get(role)
        if not bound:
            raise UnboundGraderRole(f"model role {role!r} must be bound explicitly")
        # Inspect also accepts a list of models per role and uses the first.
        for model in bound if isinstance(bound, list) else [bound]:
            if model is model_under_test or str(model) == tested:
                raise UnboundGraderRole(f"model role {role!r} is bound to the model under test")


def task_grader_roles(task: Task) -> tuple[str, ...]:
    """The grader roles `task` declares it needs bound, if any."""
    return tuple((task.metadata or {}).get(GRADER_ROLES_KEY, ()))


@solver
def grader_role_guard(roles: list[str], pins: dict[str, dict] | None = None) -> Solver:
    """Fail the sample before any solver runs unless every role is bound to another model.

    Pinned roles must also be bound to their pinned model and settings.
    """
    pinned = {role: RolePin.model_validate(pin) for role, pin in (pins or {}).items()}

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        bound = model_roles()
        require_grader_roles(roles, bound, model_under_test=get_model())
        require_role_pins(pinned, bound)
        return state

    return solve


def bind_grader_roles(task: Task, roles: Iterable[str]) -> Task:
    """Make `task` fail closed unless each grader role is bound to a model other than the router.

    The roles are recorded in the task's metadata, where `run_competition_eval`
    checks them before evaluating, and a setup step checks them again in every
    sample before the router is called, so a task evaluated directly with
    `inspect_ai.eval` cannot fall back to the model under test either. Roles
    pinned in the task's metadata (`ROLE_PINS_KEY`) are required too, and the
    setup step also checks their pinned model and settings.
    """
    pins = read_role_pins(task.metadata)
    roles = sorted({*task_grader_roles(task), *roles, *pins})
    if not roles:
        return task
    existing = [] if task.setup is None else task.setup
    steps = existing if isinstance(existing, list) else [existing]
    guard_name = registry_info(grader_role_guard([])).name
    # Unpinned tasks keep the guard's earlier arguments and task config identity.
    guard = (
        grader_role_guard(roles, {role: pin.model_dump(mode="json") for role, pin in pins.items()})
        if pins
        else grader_role_guard(roles)
    )
    task.setup = [
        guard,
        *(
            step
            for step in steps
            if not (is_registry_object(step) and registry_info(step).name == guard_name)
        ),
    ]
    task.metadata = {**(task.metadata or {}), GRADER_ROLES_KEY: roles}
    return task


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sample_input_text(sample: EvalSample) -> str:
    """A sample's input as text, independent of Inspect's random message IDs."""
    if isinstance(sample.input, str):
        return sample.input
    return json.dumps(
        [message.model_dump(mode="json", exclude={"id"}) for message in sample.input],
        sort_keys=True,
    )


def _public_score(score: Score, keep_inputs: bool) -> Score:
    metadata = {k: v for k, v in (score.metadata or {}).items() if k in _PUBLIC_SCORE_METADATA}
    return Score(
        value=score.value, answer=score.answer if keep_inputs else None, metadata=metadata or None
    )


def _public_error(error: EvalError | None) -> EvalError | None:
    """The error's exception name only; messages and tracebacks can quote references."""
    if error is None:
        return None
    lines = [line for line in error.traceback.splitlines() if line.strip()]
    match = _EXCEPTION_NAME.match(lines[-1]) if lines else None
    return EvalError(message=match.group(1) if match else "error", traceback="", traceback_ansi="")


def _public_sample(sample: EvalSample, keep_inputs: bool) -> EvalSample:
    output = sample.output.model_copy(
        deep=True, update={"metadata": None, "error": None, "fallback": None}
    )
    for choice in output.choices:
        choice.message.metadata = None
        choice.message.content = choice.message.text
    if not keep_inputs:
        output = ModelOutput.from_content(
            model=output.model, content=f"sha256:{_sha256(output.completion)}"
        )
    return EvalSample(
        id=sample.id,
        epoch=sample.epoch,
        input=sample.input if keep_inputs else f"sha256:{_sha256(sample_input_text(sample))}",
        choices=sample.choices if keep_inputs else None,
        target="",
        output=output,
        scores={name: _public_score(s, keep_inputs) for name, s in (sample.scores or {}).items()}
        or None,
        model_usage=sample.model_usage,
        role_usage=sample.role_usage,
        started_at=sample.started_at,
        completed_at=sample.completed_at,
        total_time=sample.total_time,
        working_time=sample.working_time,
        uuid=sample.uuid,
        error=_public_error(sample.error),
        limit=sample.limit,
    )


def _public_reductions(reductions: EvalSampleReductions, keep_inputs: bool) -> EvalSampleReductions:
    return EvalSampleReductions(
        scorer=reductions.scorer,
        reducer=reductions.reducer,
        samples=[
            EvalSampleScore(
                value=s.value, answer=s.answer if keep_inputs else None, sample_id=s.sample_id
            )
            for s in reductions.samples
        ],
    )


def _cleared(model: BaseModel, *fields: str) -> BaseModel:
    """A copy of `model` with `fields` reset to their defaults."""
    defaults = type(model).model_fields
    return model.model_copy(
        update={name: defaults[name].get_default(call_default_factory=True) for name in fields}
    )


def _public_header(log: EvalLog) -> dict[str, object]:
    # Task, solver and scorer arguments and metadata can carry rubrics or references.
    spec = _cleared(
        log.eval,
        "task_args",
        "task_args_passed",
        "task_attribs",
        "solver_args",
        "solver_args_passed",
        "model_args",
        "metadata",
    )
    spec = spec.model_copy(
        update={
            "dataset": spec.dataset.model_copy(update={"location": None}),
            "scorers": None
            if spec.scorers is None
            else [_cleared(scorer, "options", "metadata") for scorer in spec.scorers],
        }
    )
    plan = log.plan.model_copy(
        update={
            "steps": [_cleared(step, "params", "params_passed") for step in log.plan.steps],
            "finish": None
            if log.plan.finish is None
            else _cleared(log.plan.finish, "params", "params_passed"),
        }
    )
    results = log.results
    if results is not None:
        results = _cleared(results, "metadata").model_copy(
            update={"scores": [_cleared(score, "params", "metadata") for score in results.scores]}
        )
    return {
        "eval": spec,
        "plan": plan,
        "results": results,
        "metadata": {},
        "error": _public_error(log.error),
        # Edit and config-update provenance carries free-text reasons.
        "log_updates": None,
        "config_updates": None,
    }


def redacted_log(log: EvalLog, *, data_visibility: Literal["public", "private"]) -> EvalLog:
    """A copy of `log` safe to share outside the operator.

    Targets, sample metadata, transcripts (messages, events, attachments,
    store), score explanations and grader details are removed, because
    judges, scorers and task metadata can all contain the answer key. So are
    task, solver and scorer arguments and metadata in the header, plan and
    results, which can hold a rubric or reference. Sample and eval errors keep
    only the exception name, since a grader's error message or traceback can
    quote the reference. Inputs are replaced by
    their SHA-256 for private data, since held-out prompts stay
    private until retired. Private model outputs are hashed and textual score
    answers are removed, since either can repeat the question or reference. Scores,
    grade codes, usage and timing are kept; provider continuation and reasoning are private. Compute
    `receipt_digest` from the operator's unredacted log.
    """
    if data_visibility not in ("public", "private"):
        raise ValueError("data_visibility must be public or private")
    keep_inputs = data_visibility == "public"
    return log.model_copy(
        update={
            **_public_header(log),
            "samples": None
            if log.samples is None
            else [_public_sample(sample, keep_inputs) for sample in log.samples],
            "reductions": None
            if log.reductions is None
            else [_public_reductions(r, keep_inputs) for r in log.reductions],
        }
    )

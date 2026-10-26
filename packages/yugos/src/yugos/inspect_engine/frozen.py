"""Prepare exact samples and settings; installed code constructs native Inspect tasks."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib.metadata import entry_points, version

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer._scorer import unique_scorer_name
from inspect_ai.util import TokenLimit
from pydantic import BaseModel, ConfigDict, Field

from yugos.competition import digest
from yugos.evaluation import Question

from .assets import frozen_sandbox, inline_file, sandbox_spec
from .isolation import bind_grader_roles, read_role_pins, task_grader_roles
from .versions import require_pinned_inspect

TASK_BUILDERS_GROUP = "yugos.inspect_tasks"


def _task_builder(name: str):
    """Resolve one explicitly registered task constructor from installed code."""
    matches = list(entry_points(group=TASK_BUILDERS_GROUP, name=name))
    if len(matches) != 1 or matches[0].dist is None:
        raise ValueError(f"task builder must have one installed registration: {name}")
    return matches[0]


def _scorer_names(task: Task) -> list[str]:
    names: list[str] = []
    for scorer in task.scorer or ():
        names.append(unique_scorer_name(scorer, names))
    return names


class TaskConfig(BaseModel):
    """Plain settings that rebuild one Inspect Task from selected Samples."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    packages: dict[str, str]
    builder: str = Field(min_length=1)
    builder_args: dict = Field(default_factory=dict)
    origin: dict = Field(default_factory=dict)
    scorer: str
    score_key: str | None = None
    config: dict = Field(default_factory=dict)
    message_limit: int | None = None
    token_limit: int | None = None
    token_limit_type: str | None = None
    time_limit: float | None = None
    working_limit: float | None = None
    turn_limit: int | None = None
    metadata: dict = Field(default_factory=dict)
    version: int | str = 0
    sandbox: dict | None = None
    required_roles: tuple[str, ...] = ()

    def build(self, samples: Sequence[Sample], *, name: str, model_roles=None) -> Task:
        require_pinned_inspect()
        registration = _task_builder(self.builder)
        if registration.dist.name not in self.packages:
            raise ValueError("task builder's distribution is not pinned")
        if set(self.required_roles) - set(model_roles or {}):
            raise ValueError("prepared task requires explicitly bound service model roles")
        frozen_sandbox(sandbox_spec(self.sandbox))
        for sample in samples:
            frozen_sandbox(sample.sandbox)
            assets = [*(sample.files or {}).values(), *([sample.setup] if sample.setup else [])]
            if any(not asset.startswith("data:") for asset in assets):
                raise ValueError("sample assets must be frozen inline before execution")
        expected_samples = [s.model_dump(mode="json") for s in samples]
        options = dict(
            dataset=[s.model_copy(deep=True) for s in samples],
            name=name,
            version=self.version,
            config=GenerateConfig(**self.config),
            sandbox=sandbox_spec(self.sandbox),
            metadata=self.metadata,
            model_roles=dict(model_roles) if model_roles else None,
            message_limit=self.message_limit,
            token_limit=(
                TokenLimit(limit=self.token_limit, type=self.token_limit_type or "all")
                if self.token_limit is not None
                else None
            ),
            turn_limit=self.turn_limit,
            time_limit=self.time_limit,
            working_limit=self.working_limit,
        )
        task = registration.load()(options=options, **self.builder_args)
        if (
            not isinstance(task, Task)
            or [s.model_dump(mode="json") for s in task.dataset] != expected_samples
        ):
            raise ValueError("task builder must use exactly the prepared samples")
        if self.scorer not in _scorer_names(task):
            raise ValueError("selected scorer is absent from the constructed task")
        return bind_grader_roles(task, self.required_roles)


def freeze_task(
    task: Task,
    *,
    builder: str,
    builder_args: dict | None = None,
    origin: dict | None = None,
    source: str,
    workload_class: str,
    sample_ids: Sequence[str | int] | None = None,
    scorer: str | None = None,
    score_key: str | None = None,
    required_roles: Sequence[str] = (),
    sandbox=None,
) -> tuple[Question, ...]:
    """Retain selected data and settings for an installed native task constructor."""
    registration = _task_builder(builder)
    for field in ("model", "sample_source", "checkpoint", "early_stopping", "cost_limit"):
        if getattr(task, field) is not None:
            raise ValueError(f"task {field} is not supported in competition packs")
    required_roles = tuple(
        sorted(
            set(required_roles)
            | set(task.model_roles or {})
            | set(task_grader_roles(task))
            | set(read_role_pins(task.metadata))
        )
    )
    names = _scorer_names(task)
    if scorer is None and len(names) != 1:
        raise ValueError("select the authoritative scorer for a multi-scorer task")
    selected = scorer or names[0]
    if selected not in names:
        raise ValueError("selected scorer is absent from the task")
    task_config = TaskConfig(
        packages={
            **{p: version(p) for p in ("inspect-ai", "inspect-evals", "yugos")},
            registration.dist.name: registration.dist.version,
        },
        builder=builder,
        builder_args=builder_args or {},
        origin=origin or {"task": task.name},
        scorer=selected,
        score_key=score_key,
        config=task.config.model_dump(mode="json", exclude_none=True),
        message_limit=task.message_limit,
        token_limit=task.token_limit,
        token_limit_type=task.token_limit_type,
        turn_limit=task.turn_limit,
        time_limit=task.time_limit,
        working_limit=task.working_limit,
        version=task.version,
        metadata=task.metadata or {},
        sandbox=frozen_sandbox(sandbox or task.sandbox),
        required_roles=required_roles,
    )
    encoded_task_config = task_config.model_dump(mode="json")
    samples = {str(s.id): s for s in task.dataset}
    if len(samples) != len(task.dataset) or "None" in samples:
        raise ValueError("upstream samples require unique stable IDs")
    selected_ids = list(samples) if sample_ids is None else [str(i) for i in sample_ids]
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("selection requires distinct sample IDs")
    if set(selected_ids) - samples.keys():
        raise ValueError("selected sample is absent from the upstream dataset")
    result = []
    for identity in selected_ids:
        sample = samples[identity]
        if sample.checkpoint is not None:
            raise ValueError("sample checkpoints are not supported in frozen competition packs")
        encoded = sample.model_dump(mode="json")
        encoded["sandbox"] = frozen_sandbox(sandbox or sample.sandbox) if sample.sandbox else None
        if sample.files:
            encoded["files"] = {name: inline_file(value) for name, value in sample.files.items()}
        if sample.setup:
            encoded["setup"] = inline_file(sample.setup)
        prompt, target = question_text(sample)
        result.append(
            Question(
                question_id=f"{source}:{identity}",
                prompt=prompt,
                expected_answer=target,
                workload_class=workload_class,
                source_benchmark=source,
                inspect_task=encoded_task_config,
                inspect_sample=encoded,
            )
        )
    return tuple(result)


def restore_tasks(
    questions: Sequence[Question],
    *,
    model_roles: Mapping | Callable[[TaskConfig], Mapping] | None = None,
):
    """Rebuild one Task per distinct serialized task config, in first-seen order."""
    from yugos.competition.difficulty import question_identity

    groups: dict[str, tuple[TaskConfig, list[Sample]]] = {}
    if not questions or len({q.question_id for q in questions}) != len(questions):
        raise ValueError("pack requires distinct question IDs")
    for q in questions:
        task_config = TaskConfig.model_validate(q.inspect_task)
        key = task_config.model_dump_json()
        sample = Sample.model_validate(q.inspect_sample).model_copy(deep=True)
        if question_text(sample) != (q.prompt, q.expected_answer):
            raise ValueError("question text differs from its frozen Inspect sample")
        sample.id = q.question_id
        sample.metadata = {
            **(sample.metadata or {}),
            "workload_class": q.workload_class,
            "source_benchmark": q.source_benchmark,
            "difficulty_decile": q.difficulty_decile,
            "panel_solve_share": None if q.panel_solve_share is None else str(q.panel_solve_share),
            "yugos_question_sha256": question_identity(q),
            "yugos_scorer": task_config.scorer,
            "yugos_score_key": task_config.score_key,
        }
        if key not in groups:
            groups[key] = (task_config, [])
        groups[key][1].append(sample)
    return tuple(
        (
            task_config.build(
                samples,
                name=f"pack_{i}",
                model_roles=(model_roles(task_config) if callable(model_roles) else model_roles),
            ),
            task_config,
        )
        for i, (task_config, samples) in enumerate(groups.values())
    )


EVALUATION_IDENTITY_KIND = "yugos-inspect-recipes-v1"


def evaluation_identity(items: Iterable[Question | TaskConfig | Mapping]) -> dict:
    """Identify the distinct task configs in records, independent of Samples.

    Package pins and each config's canonical digest contribute to the identity.
    The `recipes_sha256` field name remains part of the existing protocol.
    """
    digests: set[str] = set()
    packages: dict[str, str] = {}
    for item in items:
        if isinstance(item, TaskConfig):
            task_config = item
        elif isinstance(item, Mapping):
            task_config = TaskConfig.model_validate(item)
        else:
            task_config = TaskConfig.model_validate(item.inspect_task)
        for package, pinned in task_config.packages.items():
            if packages.setdefault(package, pinned) != pinned:
                raise ValueError(f"task configs pin {package} to different versions")
        digests.add(digest(task_config.model_dump(mode="json")))
    if not digests:
        raise ValueError("evaluation identity requires at least one task config")
    return {
        "kind": EVALUATION_IDENTITY_KIND,
        "packages": dict(sorted(packages.items())),
        "recipes_sha256": digest({"recipes": sorted(digests)}),
    }


def question_text(sample: Sample) -> tuple[str, str]:
    """Readable display fields; execution retains the complete structured sample."""
    prompt = (
        sample.input if isinstance(sample.input, str) else "\n".join(m.text for m in sample.input)
    )
    target = sample.target if isinstance(sample.target, str) else json.dumps(sample.target)
    return prompt, target

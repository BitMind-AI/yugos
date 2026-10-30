"""The Yugos benchmark catalog: which tasks represent each competition category, and how.

An entry names an upstream Inspect Evals task or a `yugos_evals` task, with the
settings Yugos selects. Unlisted task arguments keep the defaults of the pinned
Inspect Evals release. `supported` means the task and settings exist in the pinned
packages; it does not mean the source is in the live competition pack.
`planned` entries have no executable task yet.

`role_pins` fix the model and generation settings of each grader or simulator
role. Freezing an entry (`load_task`, `freeze_entry`) records them in the task
metadata, so hosted and local packs carry identical pins and runs fail closed
unless each pinned role is bound to exactly that model and those settings.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yugos.evaluation.measurements import WorkloadClass
from yugos.inspect_engine import (
    RolePin,
    bind_grader_roles,
    read_role_pins,
    role_pins_metadata,
)

if TYPE_CHECKING:
    from inspect_ai import Task

    from yugos.evaluation import Question

SourceKind = Literal["inspect_evals", "yugos_evals"]
Status = Literal["supported", "planned"]


class BenchmarkEntry(BaseModel):
    """One catalog benchmark and the settings Yugos runs it with."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    category: WorkloadClass
    source: SourceKind
    # Inspect registry name, e.g. `inspect_evals/mmlu_pro` or `yugos_evals/arxivmath`.
    task: str
    args: dict[str, Any] = Field(default_factory=dict)
    # Competition fairness runs every sample once.
    epochs: Literal[1] = 1
    # Authoritative scorer for multi-scorer tasks, and scalar key for dict-valued scores.
    scorer: str | None = None
    score_key: str | None = None
    # Grader and simulator roles, each pinned to a model and its generation settings.
    role_pins: dict[str, RolePin] = Field(default_factory=dict)
    # Digest-pinned image for a task whose upstream sandbox names only a provider.
    sandbox_image: str | None = None
    status: Status
    notes: str = ""

    @model_validator(mode="after")
    def task_matches_source(self) -> BenchmarkEntry:
        if not self.task.startswith(f"{self.source}/"):
            raise ValueError(f"{self.id}: {self.source} tasks are named {self.source}/<task>")
        if self.sandbox_image is not None and not re.fullmatch(
            r"[^\s@]+@sha256:[0-9a-f]{64}", self.sandbox_image
        ):
            raise ValueError(f"{self.id}: sandbox image must be pinned by digest")
        return self


def _upstream(id: str, category: WorkloadClass, task: str | None = None, **fields):
    return BenchmarkEntry(
        id=id,
        category=category,
        source="inspect_evals",
        task=f"inspect_evals/{task or id}",
        status="supported",
        **fields,
    )


def _native(id: str, category: WorkloadClass, task: str, **fields):
    return BenchmarkEntry(
        id=id,
        category=category,
        source="yugos_evals",
        task=f"yugos_evals/{task}",
        status="supported",
        **fields,
    )


def _planned(id: str, category: WorkloadClass, notes: str):
    return BenchmarkEntry(
        id=id,
        category=category,
        source="yugos_evals",
        task=f"yugos_evals/{id}",
        status="planned",
        notes=notes,
    )


# Every grader and simulator role uses one judge model at low reasoning effort,
# replacing the benchmarks' reference judges (gpt-4.1-2025-04-14, o3-mini-2025-01-31),
# which OpenAI retires. Reasoning models ignore temperature, so the pins fix effort
# and an output cap instead. Graders: 16,384 tokens, nine times the longest verdict
# measured at low effort (Arena-Hard, 1,778), bounds a runaway judge without
# truncating a verdict. Simulated Tau2 users write short turns: 4,096 tokens.
_JUDGE = "openai/gpt-5.4-mini"
_GRADER = {"model": _JUDGE, "params": {"reasoning_effort": "low", "max_tokens": 16384}}
_USER = {"model": _JUDGE, "params": {"reasoning_effort": "low", "max_tokens": 4096}}
_TAU2_PINS = {"user": _USER}
_TAU2 = (
    "Needs an accounted `user` role; frozen through core's Tau2 constructor adapter. "
    "User-simulator pin: gpt-5.4-mini at low reasoning effort, max_tokens 4096 "
    "(upstream's per-call temperature 0 does not apply to reasoning models)."
)
_SANDBOX = "Needs prepared, digest-pinned images and an isolated sandbox provider."
# Official python:3.12-slim multi-arch index, resolved 2026-10-07. code_tests needs only
# python3 and its standard library; pass `image` to use another digest-pinned image.
PYTHON_IMAGE = "docker.io/library/python@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f"
# The CAIS judge prompt and schema as Inspect Evals' hle `original.yaml` run config
# encodes them, with one `grader` (the catalog judge rather than upstream's o3-mini).
_HLE_JUDGE = {
    "scorer": "inspect_evals/hle/json_judge",
    "graders": ["grader"],
    "args": {"prompt": "json_judge", "max_grader_attempts": 1},
}

CATALOG: tuple[BenchmarkEntry, ...] = (
    # Chat / QA
    _upstream("mmlu_pro", "chat_qa", args={"fewshot": 0, "shuffle": False}),
    _upstream("gpqa_diamond", "chat_qa", notes="Diamond subset only."),
    _upstream("arc_challenge", "chat_qa"),
    _upstream("arc_easy", "chat_qa"),
    _upstream("musr", "chat_qa", args={"shuffle": False}),
    _upstream(
        "simpleqa",
        "chat_qa",
        role_pins={"grader": _GRADER},
        notes=(
            "Needs an accounted `grader` role, pinned to gpt-5.4-mini at low reasoning "
            "effort (simple-evals uses gpt-4.1 sampled at 0.5). SimpleQA Verified is a "
            "separate source."
        ),
    ),
    _upstream(
        "ifeval",
        "chat_qa",
        score_key="prompt_level_strict",
        notes="Checkers need NLTK punkt_tab. Reassess the two historical reference exclusions.",
    ),
    _upstream(
        "livebench_reasoning",
        "chat_qa",
        task="livebench",
        args={"category": "reasoning"},
        notes=(
            "Pin the release date and retired-task exclusions; three historically "
            "rejected references stay excluded."
        ),
    ),
    _upstream(
        "hle",
        "chat_qa",
        args={"include_multi_modal": False, "scorer": _HLE_JUDGE},
        score_key="score",
        role_pins={"grader": _GRADER},
        notes=(
            "Text only: the router wire carries text. Graded with the official CAIS judge "
            "prompt and structured output by one `grader` (gpt-5.4-mini, low reasoning "
            "effort, instead of the official o3-mini) rather than upstream's default "
            "two-judge panel. "
            "Inspect Evals 0.22.0 resolves the judge only after importing "
            "inspect_evals.hle.scorers."
        ),
    ),
    _upstream(
        "winogrande",
        "chat_qa",
        args={"dataset_name": "winogrande_xl", "fewshot": 0, "shuffle": False},
    ),
    _upstream("hellaswag", "chat_qa", args={"split": "validation", "shuffle": False}),
    _upstream("commonsense_qa", "chat_qa", args={"shuffle": False}),
    _upstream("bbh", "chat_qa", notes="Select and record each BBH subset explicitly."),
    _upstream("medqa", "chat_qa"),
    _upstream("paws", "chat_qa", args={"shuffle": False}),
    _upstream("worldsense", "chat_qa", args={"shuffle": False}),
    _upstream("pubmedqa", "chat_qa"),
    _upstream("bbeh", "chat_qa"),
    _native("belebele", "chat_qa", "reading", args={"source": "belebele"}),
    _native(
        "longbench_v2",
        "chat_qa",
        "reading",
        args={"source": "longbench_v2", "lengths": ["short"]},
        notes="Complete fixed contexts; no model-dependent truncation.",
    ),
    _native(
        "hotpotqa",
        "chat_qa",
        "reading",
        args={"source": "hotpotqa"},
        score_key="joint_f1",
        notes="Labeled distractor validation; joint answer/evidence F1 is authoritative.",
    ),
    _upstream("simpleqa_verified", "chat_qa", role_pins={"grader": _GRADER}),
    _upstream("boolq", "chat_qa", args={"shuffle": False}),
    _upstream(
        "squad",
        "chat_qa",
        args={"shuffle": False},
        scorer="f1",
        notes="SQuAD v2 validation. F1 is authoritative; exact match is also scored.",
    ),
    _upstream("drop", "chat_qa", args={"fewshot": 0}),
    _upstream("piqa", "chat_qa"),
    _upstream("race_h", "chat_qa", args={"shuffle": False}),
    _upstream(
        "truthfulqa",
        "chat_qa",
        args={"target": "mc1", "shuffle": False},
        notes="multiple_choice config; mc1 is the single true answer.",
    ),
    # Math / code
    _upstream("gsm8k", "math_code", args={"fewshot": 0}),
    _upstream("aime2024", "math_code"),
    _upstream("aime2025", "math_code"),
    _upstream("aime2026", "math_code"),
    _upstream(
        "bigcodebench",
        "math_code",
        notes=f"Complete variant at v0.1.2; keep the 1120/1121 duplicate exclusion. {_SANDBOX}",
    ),
    _upstream("ds1000", "math_code", args={"shuffle": False}, notes=_SANDBOX),
    _upstream(
        "humaneval",
        "math_code",
        sandbox_image=PYTHON_IMAGE,
        notes="Unit tests run in the digest-pinned python:3.12-slim image, networking disabled.",
    ),
    _upstream(
        "mbpp",
        "math_code",
        sandbox_image=PYTHON_IMAGE,
        notes=(
            "Sanitized test split. Unit tests run in the digest-pinned python:3.12-slim "
            "image, networking disabled."
        ),
    ),
    _upstream(
        "math",
        "math_code",
        scorer="math_equivalence",
        notes=(
            "All levels and subjects, zero-shot. Deterministic answer equivalence; no model judge."
        ),
    ),
    # Agentic
    _upstream(
        "agent_bench_os",
        "agentic",
        args={"split": "test", "max_attempts": 1},
        notes=f"Per-task initialization and private checks require prepared environments. {_SANDBOX}",
    ),
    _upstream(
        "bfcl",
        "agentic",
        args={
            "categories": [
                "irrelevance",
                "live_irrelevance",
                "live_multiple",
                "live_parallel",
                "live_parallel_multiple",
                "live_relevance",
                "live_simple",
                "multiple",
                "parallel",
                "parallel_multiple",
                "simple_java",
                "simple_javascript",
                "simple_python",
                "sql",
            ]
        },
        notes=(
            "Single-turn AST categories. Multi-turn and executable (exec_*) categories "
            "need their own integration."
        ),
    ),
    _upstream("swe_bench_verified", "agentic", task="swe_bench", notes=_SANDBOX),
    _upstream("swe_bench_verified_mini", "agentic", notes=_SANDBOX),
    _upstream(
        "tau2_airline", "agentic", args={"shuffle": False}, role_pins=_TAU2_PINS, notes=_TAU2
    ),
    _upstream(
        "tau2_banking", "agentic", args={"shuffle": False}, role_pins=_TAU2_PINS, notes=_TAU2
    ),
    _upstream("tau2_retail", "agentic", args={"shuffle": False}, role_pins=_TAU2_PINS, notes=_TAU2),
    _upstream(
        "tau2_telecom", "agentic", args={"shuffle": False}, role_pins=_TAU2_PINS, notes=_TAU2
    ),
    # Not in Inspect Evals; native yugos_evals tasks with public source loaders.
    _native(
        "arxivmath",
        "math_code",
        "math_answer",
        args={"source": "arxivmath"},
        notes=(
            "Every pinned monthly release (Dec 2025-Jun 2026, Aug 2026); references "
            "Math-Verify cannot check are excluded with reasons. CC BY-SA 4.0."
        ),
    ),
    _native(
        "livemathbench",
        "math_code",
        "math_answer",
        args={"source": "livemathbench", "subsets": ["202505/all_en"]},
        notes="Gated on Hugging Face: preparation needs HF_TOKEN or staged files.",
    ),
    _native(
        "cruxeval_o",
        "math_code",
        "cruxeval",
        args={
            "image": "ghcr.io/generality-labs/inspect-eval-bigcodebench:2026-07-28@sha256:d7c88bf92782a7486d64bc1d07cd38ba9e735b5f58235fc0c0f36f8f692a1cfe"
        },
        notes=(
            "Output expressions checked with Python equality in the isolated shared test "
            "supervisor. Single generation without few-shot examples; not the upstream "
            "CoT/few-shot prompt. Images need pyarrow and rpyc."
        ),
    ),
    _native(
        "livecodebench",
        "math_code",
        "code_tests",
        args={"source": "livecodebench", "releases": ["v6"], "image": PYTHON_IMAGE},
        notes=(
            "v6 additions only, not cumulative release_v6; upstream livecodebench_pro "
            "differs. Runs in the official python:3.12-slim image pinned by digest "
            "(`image` overrides it) with networking disabled; needs Docker locally."
        ),
    ),
    _native(
        "livecodebench_plus",
        "math_code",
        "code_tests",
        args={
            "source": "livecodebench_plus",
            "splits": ["medium", "hard"],
            "exclude_releases": ["v6"],
            "image": PYTHON_IMAGE,
        },
        notes=(
            "Medium/hard tasks minus LiveCodeBench v6 overlap. Runs in the official "
            "python:3.12-slim image pinned by digest (`image` overrides it) with "
            "networking disabled; needs Docker locally."
        ),
    ),
    _native(
        "arenahard_v2",
        "chat_qa",
        "pairwise_judge",
        args={"categories": ["hard_prompt", "creative_writing"]},
        role_pins={"grader": _GRADER},
        notes=(
            "Pairwise preference against pinned baselines, not correctness. Needs an "
            "accounted `grader` role, pinned to gpt-5.4-mini at low reasoning effort "
            "instead of upstream's v2.0 judge (gpt-4.1); its verdicts differ from "
            "upstream's (52% exact agreement, favouring the candidate)."
        ),
    ),
    # Planned: these need multi-service environments or prepared Docker infrastructure.
    _planned(
        "appworld",
        "agentic",
        (
            "Portable Inspect lifecycle and end-state scoring with private ground truth; "
            "needs the multi-service AppWorld environment server."
        ),
    ),
    _planned(
        "terminal_bench",
        "agentic",
        (
            "Terminal-Bench 2.0: portable lifecycle, prepared per-task Docker "
            "environments and verifier controls."
        ),
    ),
)


def entries(
    *, category: WorkloadClass | None = None, status: Status | None = None
) -> tuple[BenchmarkEntry, ...]:
    """Catalog entries, optionally filtered by category and status, in catalog order."""
    return tuple(
        e
        for e in CATALOG
        if (category is None or e.category == category) and (status is None or e.status == status)
    )


def entry(id: str) -> BenchmarkEntry:
    """The catalog entry with this id."""
    for e in CATALOG:
        if e.id == id:
            return e
    raise KeyError(f"unknown benchmark: {id}")


def pin_roles(task: Task, benchmark: BenchmarkEntry) -> Task:
    """Record the entry's role pins in `task`'s metadata and make it fail closed without them.

    Pins the task already records must agree with the entry's; their settings are combined.
    """
    if not benchmark.role_pins:
        return task
    pins = read_role_pins(task.metadata)
    for role, pin in benchmark.role_pins.items():
        current = pins.get(role)
        if current is not None:
            if current.model != pin.model or any(
                current.params.get(key, value) != value for key, value in pin.params.items()
            ):
                raise ValueError(f"{benchmark.id}: the task pins model role {role!r} differently")
            pin = RolePin(model=pin.model, params={**current.params, **pin.params})
        pins[role] = pin
    task.metadata = {**(task.metadata or {}), **role_pins_metadata(pins)}
    return bind_grader_roles(task, pins)


def freeze_entry(
    benchmark: BenchmarkEntry,
    *,
    task: Task | None = None,
    sample_ids: Sequence[str | int] | None = None,
    task_args: dict[str, Any] | None = None,
    **options: Any,
) -> tuple[Question, ...]:
    """Freeze the entry's samples as pack questions, with its scorer, score key and role pins.

    `task` is an instance already constructed for this entry (for example from
    staged files); otherwise `load_task` constructs it with `task_args`. `options`
    pass through to core's `freeze_task` (e.g. a prepared `sandbox`). An entry
    `sandbox_image` becomes that sandbox unless `options` already pass one.
    """
    if task is None:
        task = load_task(benchmark, **(task_args or {}))
    elif task_args:
        raise ValueError("pass task_args or a constructed task, not both")
    if benchmark.sandbox_image is not None and "sandbox" not in options:
        from .tasks.code_tests import code_sandbox

        options["sandbox"] = code_sandbox(benchmark.sandbox_image)
    from .prepared import prepare_task

    return prepare_task(
        pin_roles(task, benchmark),
        source=benchmark.id,
        workload_class=benchmark.category,
        sample_ids=sample_ids,
        scorer=benchmark.scorer,
        score_key=benchmark.score_key,
        **options,
    )


def load_task(benchmark: BenchmarkEntry, **args: Any) -> Task:
    """Construct the entry's Inspect task with its catalog settings and role pins, plus `args`.

    Resolves `inspect_evals/<task>` and `yugos_evals/<task>` alike through Inspect's
    registry. Calling the factory may download the benchmark's dataset.
    """
    from inspect_ai._util.entrypoints import ensure_entry_points
    from inspect_ai._util.registry import registry_lookup

    from .sources._common import with_sample_ids

    if benchmark.status != "supported":
        raise ValueError(f"{benchmark.id} has no executable task yet ({benchmark.status})")
    ensure_entry_points(benchmark.source)
    factory = registry_lookup("task", benchmark.task)
    if factory is None:
        raise ValueError(f"{benchmark.task} is not installed")
    task = pin_roles(factory(**{**benchmark.args, **args}), benchmark)
    from .corrections import correct_task

    correct_task(task, benchmark.task.split("/")[1], {**benchmark.args, **args})
    task.dataset = with_sample_ids(task.dataset)
    return task

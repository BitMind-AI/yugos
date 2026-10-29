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
    # Reasoning benchmarks verified against inspect_evals==0.22.0.
    _upstream("bbeh_mini", "chat_qa", notes="BIG-Bench Extra Hard Mini: reduced subset of BBEH for faster evaluation."),
    _upstream("lingoly", "chat_qa", notes="LingOly: linguistics olympiad problems requiring deep language reasoning."),
    _upstream("lingoly_too", "chat_qa", notes="LingOly-Too: extended linguistics olympiad problems."),
    _upstream(
        "mmmu_multiple_choice",
        "chat_qa",
        notes="MMMU multiple-choice: multi-discipline multimodal understanding; text-only router path.",
    ),
    _upstream(
        "mmmu_open",
        "chat_qa",
        notes="MMMU open-ended: multi-discipline multimodal open questions; text-only router path.",
    ),
    _upstream("niah", "chat_qa", notes="NIAH (Needle In A Haystack): long-context retrieval benchmark."),
    _upstream(
        "novelty_bench",
        "chat_qa",
        notes="NoveltyBench: evaluating creative and novel reasoning.",
    ),
    _upstream("vimgolf_single_turn", "chat_qa", notes="VimGolf single-turn: text-editing benchmark using vim-style operations."),
    _upstream("infinite_bench_code_debug", "chat_qa", notes="InfiniteBench code_debug: long-context (100k+) code debugging task."),
    _upstream("infinite_bench_code_run", "chat_qa", notes="InfiniteBench code_run: long-context (100k+) code execution prediction task."),
    _upstream("infinite_bench_kv_retrieval", "chat_qa", notes="InfiniteBench kv_retrieval: long-context (100k+) key-value retrieval task."),
    _upstream("infinite_bench_longbook_choice_eng", "chat_qa", notes="InfiniteBench longbook_choice_eng: long-context (100k+) book multiple-choice task."),
    _upstream("infinite_bench_longdialogue_qa_eng", "chat_qa", notes="InfiniteBench longdialogue_qa_eng: long-context (100k+) dialogue QA task."),
    _upstream("infinite_bench_math_calc", "chat_qa", notes="InfiniteBench math_calc: long-context (100k+) math calculation task."),
    _upstream("infinite_bench_math_find", "chat_qa", notes="InfiniteBench math_find: long-context (100k+) math number retrieval task."),
    _upstream("infinite_bench_number_string", "chat_qa", notes="InfiniteBench number_string: long-context (100k+) number string retrieval task."),
    _upstream("infinite_bench_passkey", "chat_qa", notes="InfiniteBench passkey: long-context (100k+) passkey retrieval task."),
    # Knowledge benchmarks verified against inspect_evals==0.22.0.
    # AGIE academic-exam suite (nine tasks, shared cot/fewshot/fewshot_seed params).
    _upstream("agie_aqua_rat", "chat_qa", notes="AGIE AQuA-RAT: algebraic word-problem reasoning from the AGIE exam suite."),
    _upstream("agie_logiqa_en", "chat_qa", notes="AGIE LogiQA-EN: logical reasoning QA from the AGIE exam suite."),
    _upstream("agie_lsat_ar", "chat_qa", notes="AGIE LSAT-AR: analytical reasoning from the AGIE LSAT subset."),
    _upstream("agie_lsat_lr", "chat_qa", notes="AGIE LSAT-LR: logical reasoning from the AGIE LSAT subset."),
    _upstream("agie_lsat_rc", "chat_qa", notes="AGIE LSAT-RC: reading comprehension from the AGIE LSAT subset."),
    _upstream("agie_math", "chat_qa", notes="AGIE MATH: competition math problems from the AGIE exam suite."),
    _upstream("agie_sat_en", "chat_qa", notes="AGIE SAT-EN: SAT English questions from the AGIE exam suite."),
    _upstream(
        "agie_sat_en_without_passage",
        "chat_qa",
        notes="AGIE SAT-EN (no passage): SAT English questions without the reading passage.",
    ),
    _upstream("agie_sat_math", "chat_qa", notes="AGIE SAT-Math: SAT math questions from the AGIE exam suite."),
    # Domain knowledge
    _upstream("air_bench", "chat_qa", notes="AIR-Bench: instruction-following safety and capability benchmark."),
    _upstream("chembench", "chat_qa", notes="ChemBench: chemistry knowledge and reasoning benchmark."),
    _upstream(
        "frontierscience",
        "chat_qa",
        notes=(
            "FrontierScience: cutting-edge scientific knowledge QA. "
            "Upstream uses a grader_model; defaults to the Inspect provider default."
        ),
    ),
    _upstream(
        "healthbench",
        "chat_qa",
        notes=(
            "HealthBench: medical AI quality and safety benchmark. "
            "Upstream uses a judge_model; defaults to the Inspect provider default."
        ),
    ),
    _upstream("healthbench_hard", "chat_qa", notes="HealthBench Hard: harder subset of HealthBench."),
    _upstream("healthbench_consensus", "chat_qa", notes="HealthBench Consensus: consensus-labelled subset of HealthBench."),
    _upstream("healthbench_meta_eval", "chat_qa", notes="HealthBench Meta-Eval: meta-evaluation of HealthBench judges."),
    _upstream(
        "livebench",
        "chat_qa",
        notes=(
            "LiveBench: all categories (reasoning, coding, math, language, data analysis, "
            "instruction following). Pin the release date. "
            "See also livebench_reasoning for the reasoning-only entry."
        ),
    ),
    _upstream("mmlu_0_shot", "chat_qa", notes="MMLU 0-shot: MMLU evaluated without few-shot examples."),
    _upstream("mmlu_5_shot", "chat_qa", notes="MMLU 5-shot: MMLU evaluated with five-shot examples."),
    _upstream("macbench", "chat_qa", notes="MACBench: multi-agent collaboration knowledge benchmark."),
    _upstream("onet_m6", "chat_qa", notes="O*NET M6: occupational knowledge benchmark from the O*NET database."),
    _upstream("pre_flight", "chat_qa", notes="Pre-flight: aviation knowledge and safety QA benchmark."),
    _upstream("sosbench", "chat_qa", notes="SOSBench: safety-of-science benchmark covering dual-use knowledge domains."),
    _upstream("sciknoweval", "chat_qa", notes="SciKnowEval: scientific knowledge evaluation across multiple research domains."),
    _upstream(
        "uccb",
        "chat_qa",
        notes=(
            "UCCB: user-centric chatbot benchmark. "
            "Upstream uses a scorer_model; defaults to the Inspect provider default."
        ),
    ),
    _upstream(
        "xstest",
        "chat_qa",
        notes=(
            "XSTest: cross-safety evaluation of safe vs. unsafe prompts. "
            "Upstream uses a scorer_model; defaults to the Inspect provider default."
        ),
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
        scorer="expression_exact_match",
        notes=(
            "All levels and subjects, zero-shot. Scored only by expression_exact_match. "
            "The model equivalence scorer is not constructed."
        ),
    ),
    # Mathematics benchmarks verified against inspect_evals==0.22.0.
    _upstream("mgsm", "math_code", notes="MGSM: multilingual grade school math benchmark across 11 languages."),
    _upstream("mathvista", "math_code", notes="MathVista: mathematical reasoning in visual contexts; text-only router path."),
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
    # Coding benchmarks verified against inspect_evals==0.22.0.
    _upstream(
        "apps",
        "math_code",
        notes=(
            "APPS: introductory-to-competition programming problems with unit tests. "
            f"{_SANDBOX}"
        ),
    ),
    _upstream(
        "core_bench",
        "math_code",
        notes=(
            "CoreBench: computational reproducibility benchmark evaluating agents' ability "
            f"to reproduce scientific results. {_SANDBOX}"
        ),
    ),
    _upstream(
        "class_eval",
        "math_code",
        notes=(
            "ClassEval: class-level code generation benchmark with method dependencies "
            f"and unit tests. {_SANDBOX}"
        ),
    ),
    _upstream(
        "compute_eval",
        "math_code",
        notes=(
            "ComputeEval: benchmark for evaluating compute-intensive code generation "
            f"tasks. {_SANDBOX}"
        ),
    ),
    _upstream(
        "frontier_cs",
        "math_code",
        notes=(
            "FrontierCS: frontier computer science research problems. "
            "See also frontier_cs_algorithmic and frontier_cs_research variants."
        ),
    ),
    _upstream(
        "frontier_cs_algorithmic",
        "math_code",
        notes="FrontierCS algorithmic variant: subset focused on algorithmic problems.",
    ),
    _upstream(
        "frontier_cs_research",
        "math_code",
        notes="FrontierCS research variant: subset focused on research-level problems.",
    ),
    _upstream(
        "ifevalcode",
        "math_code",
        notes=(
            "IFEvalCode: instruction-following evaluation for code generation tasks. "
            f"{_SANDBOX}"
        ),
    ),
    _upstream(
        "kernelbench",
        "math_code",
        notes=(
            "KernelBench: GPU kernel optimization benchmark testing CUDA/Triton "
            f"code generation. {_SANDBOX}"
        ),
    ),
    _upstream(
        "livecodebench_pro",
        "math_code",
        notes=(
            "LiveCodeBench Pro: upstream's pro variant with harder coding problems. "
            "Distinct from the native yugos_evals livecodebench entry. "
            f"{_SANDBOX}"
        ),
    ),
    _upstream(
        "mle_bench",
        "agentic",
        notes=(
            "MLE-Bench: Kaggle competition benchmark requiring end-to-end ML engineering. "
            f"{_SANDBOX}"
        ),
    ),
    _upstream(
        "mle_bench_full",
        "agentic",
        notes=(
            "MLE-Bench Full: complete set of Kaggle competition tasks. "
            f"{_SANDBOX}"
        ),
    ),
    _upstream(
        "mle_bench_lite",
        "agentic",
        notes=(
            "MLE-Bench Lite: reduced subset of Kaggle competition tasks for faster evaluation. "
            f"{_SANDBOX}"
        ),
    ),
    _upstream(
        "mlrc_bench",
        "agentic",
        notes=(
            "MLRC-Bench: ML reproducibility challenge benchmark requiring paper "
            f"reproduction. {_SANDBOX}"
        ),
    ),
    _upstream(
        "paperbench",
        "agentic",
        notes=(
            "PaperBench: evaluating agents on reproducing ML research papers end-to-end. "
            f"{_SANDBOX}"
        ),
    ),
    _upstream(
        "swe_lancer",
        "agentic",
        notes=(
            "SWE-Lancer: real-world freelance software engineering tasks from Upwork. "
            f"{_SANDBOX}"
        ),
    ),
    _upstream(
        "scicode",
        "math_code",
        notes=(
            "SciCode: scientific code generation benchmark across multiple research "
            f"disciplines. {_SANDBOX}"
        ),
    ),
    _upstream(
        "usaco",
        "math_code",
        notes=(
            "USACO: USA Computing Olympiad competitive programming problems. "
            f"{_SANDBOX}"
        ),
    ),
    _upstream(
        "scbench",
        "agentic",
        notes=(
            "scBench: agentic coding benchmark requiring Docker sandbox and "
            f"inspect-swe agent. {_SANDBOX}"
        ),
    ),
    # Scheming / safety benchmarks verified against inspect_evals==0.22.0.
    _upstream(
        "agentic_misalignment",
        "agentic",
        notes=(
            "Agentic Misalignment: evaluates goal-directed misaligned behaviour in agentic settings. "
            "Upstream uses a grader_model; defaults to the Inspect provider default."
        ),
    ),
    # GDM scheming probes — end-to-end scenarios (sp01–sp12, no sp06/sp11).
    _upstream("gdm_sp01_e2e", "agentic", notes="GDM scheming probe SP01 end-to-end scenario."),
    _upstream("gdm_sp02_e2e", "agentic", notes="GDM scheming probe SP02 end-to-end scenario."),
    _upstream("gdm_sp03_e2e", "agentic", notes="GDM scheming probe SP03 end-to-end scenario."),
    _upstream("gdm_sp04_e2e", "agentic", notes="GDM scheming probe SP04 end-to-end scenario."),
    _upstream("gdm_sp05_e2e", "agentic", notes="GDM scheming probe SP05 end-to-end scenario."),
    _upstream("gdm_sp07_e2e", "agentic", notes="GDM scheming probe SP07 end-to-end scenario."),
    _upstream("gdm_sp08_e2e", "agentic", notes="GDM scheming probe SP08 end-to-end scenario."),
    _upstream("gdm_sp09_e2e", "agentic", notes="GDM scheming probe SP09 end-to-end scenario."),
    _upstream("gdm_sp10_e2e", "agentic", notes="GDM scheming probe SP10 end-to-end scenario."),
    _upstream("gdm_sp12_e2e", "agentic", notes="GDM scheming probe SP12 end-to-end scenario."),
    # GDM self-reasoning tasks (need a sandbox).
    _upstream(
        "gdm_self_reasoning_approved_directories",
        "agentic",
        notes=f"GDM self-reasoning: approved-directories scenario. {_SANDBOX}",
    ),
    _upstream(
        "gdm_self_reasoning_calculator_improvement",
        "agentic",
        notes=f"GDM self-reasoning: calculator-improvement scenario. {_SANDBOX}",
    ),
    _upstream(
        "gdm_self_reasoning_context_length_mod_instrumental_only",
        "agentic",
        notes=f"GDM self-reasoning: context-length modification (instrumental only). {_SANDBOX}",
    ),
    _upstream(
        "gdm_self_reasoning_context_length_mod_irreversibility_only",
        "agentic",
        notes=f"GDM self-reasoning: context-length modification (irreversibility only). {_SANDBOX}",
    ),
    _upstream(
        "gdm_self_reasoning_database_tool",
        "agentic",
        notes=f"GDM self-reasoning: database-tool scenario. {_SANDBOX}",
    ),
    _upstream(
        "gdm_self_reasoning_latency_calculator",
        "agentic",
        notes=f"GDM self-reasoning: latency-calculator scenario. {_SANDBOX}",
    ),
    _upstream(
        "gdm_self_reasoning_max_messages_calculator",
        "agentic",
        notes=f"GDM self-reasoning: max-messages-calculator scenario. {_SANDBOX}",
    ),
    _upstream(
        "gdm_self_reasoning_max_tokens",
        "agentic",
        notes=f"GDM self-reasoning: max-tokens scenario. {_SANDBOX}",
    ),
    _upstream(
        "gdm_self_reasoning_oversight_frequency",
        "agentic",
        notes=f"GDM self-reasoning: oversight-frequency scenario. {_SANDBOX}",
    ),
    _upstream(
        "gdm_self_reasoning_read_logs",
        "agentic",
        notes=f"GDM self-reasoning: read-logs scenario. {_SANDBOX}",
    ),
    _upstream(
        "gdm_self_reasoning_turn_off_filters",
        "agentic",
        notes=f"GDM self-reasoning: turn-off-filters scenario. {_SANDBOX}",
    ),
    # GDM deceptive / covert scheming tasks (rater_model_company selects the LLM rater).
    _upstream(
        "gdm_classifier_evasion",
        "agentic",
        notes="GDM classifier evasion: tests whether a model evades a content classifier. Upstream uses rater_model_company.",
    ),
    _upstream(
        "gdm_cover_your_tracks",
        "agentic",
        notes="GDM cover-your-tracks: tests whether a model covers evidence of its actions. Upstream uses rater_model_company.",
    ),
    _upstream(
        "gdm_oversight_pattern",
        "agentic",
        notes="GDM oversight pattern: tests whether a model behaves differently under oversight. Upstream uses rater_model_company.",
    ),
    _upstream(
        "gdm_strategic_rule_breaking",
        "agentic",
        notes="GDM strategic rule-breaking: tests whether a model strategically violates rules. Upstream uses rater_model_company.",
    ),
    _upstream(
        "instrumentaleval",
        "agentic",
        notes=(
            "InstrumentalEval: evaluates instrumental reasoning and goal-directed behaviour. "
            "Upstream uses a grader_model; defaults to the Inspect provider default."
        ),
    ),
    # SAD (Situational Awareness Dataset) — multiple-choice knowledge tasks.
    _upstream("sad_stages_full", "chat_qa", notes="SAD stages (full): situational-awareness multiple-choice across all stages."),
    _upstream("sad_stages_oversight", "chat_qa", notes="SAD stages (oversight): situational-awareness MC focused on oversight stages."),
    _upstream("sad_influence", "chat_qa", notes="SAD influence: situational-awareness MC on a model's capacity to influence outcomes."),
    _upstream("sad_facts_llms", "chat_qa", notes="SAD facts (LLMs): situational-awareness MC on factual knowledge about LLMs."),
    _upstream("sad_facts_human_defaults", "chat_qa", notes="SAD facts (human defaults): situational-awareness MC on human-default knowledge."),
    # Cybersecurity benchmarks verified against inspect_evals==0.22.0.
    # CTI REALM — agentic cyber-threat-intelligence tasks.
    _upstream("cti_realm_25", "agentic", notes="CTI REALM 25: cyber threat intelligence reasoning with 25 challenges."),
    _upstream("cti_realm_50", "agentic", notes="CTI REALM 50: cyber threat intelligence reasoning with 50 challenges."),
    _upstream("cti_realm_25_minimal", "agentic", notes="CTI REALM 25 minimal: CTI REALM 25 without tool scaffolding."),
    _upstream("cti_realm_25_seeded", "agentic", notes="CTI REALM 25 seeded: CTI REALM 25 with a fixed random seed."),
    # CTF / exploitation tasks (need sandboxes).
    _upstream(
        "cve_bench",
        "agentic",
        notes=f"CVE-Bench: real-world CVE exploitation challenges in isolated environments. {_SANDBOX}",
    ),
    _upstream(
        "threecb",
        "agentic",
        notes=f"3CB: three capture-the-flag challenge benchmarks requiring a sandbox. {_SANDBOX}",
    ),
    _upstream(
        "cybench",
        "agentic",
        notes=f"CyberBench: CTF security challenges across multiple categories. {_SANDBOX}",
    ),
    _upstream(
        "cybergym",
        "agentic",
        notes=f"CyberGym: offensive security gym benchmark with isolated environments. {_SANDBOX}",
    ),
    _upstream(
        "gdm_in_house_ctf",
        "agentic",
        notes="GDM in-house CTF: DeepMind's internal capture-the-flag security challenges.",
    ),
    _upstream(
        "gdm_intercode_ctf",
        "agentic",
        notes=f"GDM InterCode CTF: interactive code-based CTF challenges. {_SANDBOX}",
    ),
    # CYSE attack/defence scenarios.
    _upstream(
        "cyse3_visual_prompt_injection",
        "agentic",
        notes="CYSE3 visual prompt injection: vision-based prompt-injection attack scenarios. Upstream uses judge_llm.",
    ),
    _upstream(
        "cyse4_multiturn_phishing",
        "agentic",
        notes="CYSE4 multi-turn phishing: simulated phishing conversations. Upstream uses victim_llm and judge_llm.",
    ),
    _upstream(
        "cyse4_multilingual_prompt_injection",
        "agentic",
        notes="CYSE4 multilingual prompt injection: cross-language prompt-injection scenarios. Upstream uses judge_llm.",
    ),
    _upstream(
        "cyse2_interpreter_abuse",
        "agentic",
        notes="CYSE2 interpreter abuse: evaluates LLM resistance to interpreter-abuse attacks. Upstream uses judge_llm.",
    ),
    _upstream(
        "cyse2_prompt_injection",
        "agentic",
        notes="CYSE2 prompt injection: evaluates LLM resistance to prompt-injection attacks. Upstream uses judge_llm.",
    ),
    _upstream(
        "cyse2_vulnerability_exploit",
        "agentic",
        notes="CYSE2 vulnerability exploit: practical vulnerability exploitation scenarios.",
    ),
    # CYSE4 knowledge / static-analysis tasks.
    _upstream("cyse4_mitre", "chat_qa", notes="CYSE4 MITRE ATT&CK: knowledge QA over the MITRE ATT&CK framework. Upstream uses expansion_llm and judge_llm."),
    _upstream("cyse4_mitre_frr", "chat_qa", notes="CYSE4 MITRE FRR: false-rejection-rate variant of the CYSE4 MITRE task."),
    _upstream("cyse4_instruct", "chat_qa", notes="CYSE4 instruct: cybersecurity instruction-following evaluation."),
    _upstream("cyse4_autocomplete", "chat_qa", notes="CYSE4 autocomplete: cybersecurity code/command autocomplete benchmark."),
    _upstream("cyse4_malware_analysis", "chat_qa", notes="CYSE4 malware analysis: static malware analysis knowledge benchmark."),
    _upstream("cyse4_threat_intelligence", "chat_qa", notes="CYSE4 threat intelligence: cyber threat intelligence knowledge evaluation."),
    # CyberMetric MC knowledge benchmark (four size variants).
    _upstream("cybermetric_80", "chat_qa", notes="CyberMetric-80: 80-question cybersecurity knowledge MC benchmark."),
    _upstream("cybermetric_500", "chat_qa", notes="CyberMetric-500: 500-question cybersecurity knowledge MC benchmark."),
    _upstream("cybermetric_2000", "chat_qa", notes="CyberMetric-2000: 2000-question cybersecurity knowledge MC benchmark."),
    _upstream("cybermetric_10000", "chat_qa", notes="CyberMetric-10000: 10000-question cybersecurity knowledge MC benchmark."),
    # SevenLLM bilingual cybersecurity QA.
    _upstream("sevenllm_mcq_zh", "chat_qa", notes="SevenLLM MCQ (Chinese): cybersecurity multiple-choice QA in Chinese."),
    _upstream("sevenllm_mcq_en", "chat_qa", notes="SevenLLM MCQ (English): cybersecurity multiple-choice QA in English."),
    _upstream("sevenllm_qa_zh", "chat_qa", notes="SevenLLM QA (Chinese): cybersecurity open QA in Chinese."),
    _upstream("sevenllm_qa_en", "chat_qa", notes="SevenLLM QA (English): cybersecurity open QA in English."),
    # SecQA security knowledge benchmark (v1/v2, 0-shot and 5-shot).
    _upstream("sec_qa_v1", "chat_qa", notes="SecQA v1: 0-shot security knowledge benchmark."),
    _upstream("sec_qa_v1_5_shot", "chat_qa", notes="SecQA v1 5-shot: 5-shot security knowledge benchmark."),
    _upstream("sec_qa_v2", "chat_qa", notes="SecQA v2: 0-shot security knowledge benchmark (v2 dataset)."),
    _upstream("sec_qa_v2_5_shot", "chat_qa", notes="SecQA v2 5-shot: 5-shot security knowledge benchmark (v2 dataset)."),
    # Assistants / agentic-task benchmarks verified against inspect_evals==0.22.0.
    # AssistantBench — four variants (closed-book + web-augmented, 0-shot and 1-shot).
    _upstream("assistant_bench_closed_book_zero_shot", "chat_qa", notes="AssistantBench closed-book 0-shot: complex real-world QA without retrieval."),
    _upstream("assistant_bench_closed_book_one_shot", "chat_qa", notes="AssistantBench closed-book 1-shot: complex real-world QA without retrieval, one-shot."),
    _upstream(
        "assistant_bench_web_search_zero_shot",
        "agentic",
        notes="AssistantBench web-search 0-shot: complex QA augmented with web search. Upstream uses web_search_model.",
    ),
    _upstream(
        "assistant_bench_web_search_one_shot",
        "agentic",
        notes="AssistantBench web-search 1-shot: complex QA with web search, one-shot. Upstream uses web_search_model.",
    ),
    _upstream(
        "assistant_bench_web_browser",
        "agentic",
        notes="AssistantBench web browser: complex QA requiring live browser interaction. Upstream uses web_search_model.",
    ),
    # Web / GUI agentic benchmarks.
    _upstream(
        "browse_comp",
        "agentic",
        notes=f"BrowseComp: web-browsing composition tasks requiring multi-step retrieval. {_SANDBOX} Upstream uses scorer_model.",
    ),
    _upstream("gaia", "agentic", notes=f"GAIA: general AI assistant benchmark with tool use. {_SANDBOX}"),
    _upstream("gaia_level1", "agentic", notes="GAIA level 1: easiest subset of the GAIA benchmark."),
    _upstream("gaia_level2", "agentic", notes="GAIA level 2: medium-difficulty subset of the GAIA benchmark."),
    _upstream("gaia_level3", "agentic", notes="GAIA level 3: hardest subset of the GAIA benchmark."),
    _upstream("gdpval", "agentic", notes="GDPVal: general digital-task performance validation benchmark."),
    _upstream("mind2web", "agentic", notes="Mind2Web: web navigation benchmark evaluating agent task completion on real websites."),
    _upstream(
        "osworld",
        "agentic",
        notes=f"OSWorld: computer-use benchmark requiring GUI interaction in a live OS environment. {_SANDBOX}",
    ),
    _upstream(
        "osworld_small",
        "agentic",
        notes=f"OSWorld small: reduced-size subset of OSWorld for faster evaluation. {_SANDBOX}",
    ),
    _upstream(
        "sycophancy",
        "chat_qa",
        notes="Sycophancy: evaluates tendency to give socially desirable over truthful answers. Upstream uses scorer_model.",
    ),
    _upstream(
        "theagentcompany",
        "agentic",
        notes="TheAgentCompany: agentic workplace tasks requiring multi-step tool use in a simulated company environment.",
    ),
    # Safeguards benchmarks verified against inspect_evals==0.22.0.
    # Multi-turn / agentic safety tasks.
    _upstream(
        "agentdojo",
        "agentic",
        notes="AgentDojo: prompt-injection attacks against tool-using agents across realistic tasks.",
    ),
    _upstream(
        "agentharm",
        "agentic",
        notes="AgentHarm: evaluates agent harmfulness across a range of harmful behaviors. Upstream uses refusal_judge and semantic_judge.",
    ),
    _upstream(
        "agentharm_benign",
        "agentic",
        notes="AgentHarm benign: benign-behavior baseline paired with AgentHarm. Upstream uses refusal_judge and semantic_judge.",
    ),
    _upstream("agent_threat_bench_memory_poison", "agentic", notes="AgentThreatBench memory-poisoning: evaluates resistance to memory-poisoning attacks."),
    _upstream("agent_threat_bench_autonomy_hijack", "agentic", notes="AgentThreatBench autonomy-hijack: evaluates resistance to autonomy-hijacking attacks."),
    _upstream("agent_threat_bench_data_exfil", "agentic", notes="AgentThreatBench data-exfiltration: evaluates resistance to data-exfiltration attacks."),
    _upstream("ipi_coding_agent", "agentic", notes="IPI coding agent: indirect prompt injection against a coding agent."),
    _upstream(
        "ape_eval",
        "agentic",
        notes="APE-Eval: adversarial persuasion evaluation testing resistance to multi-turn influence.",
    ),
    _upstream("make_me_pay", "agentic", notes="MakeMePay: multi-turn manipulation evaluation testing whether an agent can be socially engineered."),
    _upstream("makemesay", "agentic", notes="MakeMeSay: multi-turn jailbreak evaluation testing whether a model can be made to produce target phrases."),
    _upstream("mind2web_sc", "agentic", notes="Mind2Web-SC: safety-constrained web navigation benchmark."),
    _upstream("tac", "agentic", notes="TAC: tool-augmented conversation safety benchmark."),
    _upstream("tac_welfare", "agentic", notes="TAC Welfare: welfare-focused variant of the TAC tool-augmented conversation benchmark."),
    _upstream("b3", "agentic", notes="B3: agentic benchmark evaluating behaviour across threat levels and task types."),
    # Single-turn / knowledge safety tasks.
    _upstream(
        "anima",
        "chat_qa",
        notes="ANIMA: evaluates model accuracy on natural-language QA across languages. Upstream uses grader_models.",
    ),
    _upstream(
        "abstention_bench",
        "chat_qa",
        notes="AbstentionBench: evaluates appropriate refusal on unanswerable or harmful questions. Upstream uses grader_model.",
    ),
    _upstream(
        "fortress_adversarial",
        "chat_qa",
        notes="Fortress adversarial: adversarial safety robustness evaluation. Upstream uses grader_models.",
    ),
    _upstream(
        "fortress_benign",
        "chat_qa",
        notes="Fortress benign: benign-baseline safety evaluation paired with Fortress adversarial. Upstream uses grader_model.",
    ),
    # LAB-Bench — scientific laboratory knowledge (eight sub-tasks).
    _upstream("lab_bench_litqa", "chat_qa", notes="LAB-Bench LitQA: scientific literature question answering."),
    _upstream("lab_bench_suppqa", "chat_qa", notes="LAB-Bench SuppQA: supplementary-material QA from scientific papers."),
    _upstream("lab_bench_figqa", "chat_qa", notes="LAB-Bench FigQA: figure-based question answering in scientific papers."),
    _upstream("lab_bench_tableqa", "chat_qa", notes="LAB-Bench TableQA: table-based QA in scientific papers."),
    _upstream("lab_bench_dbqa", "chat_qa", notes="LAB-Bench DBQA: database-driven scientific QA."),
    _upstream("lab_bench_protocolqa", "chat_qa", notes="LAB-Bench ProtocolQA: laboratory protocol question answering."),
    _upstream("lab_bench_seqqa", "chat_qa", notes="LAB-Bench SeqQA: biological sequence question answering."),
    _upstream("lab_bench_cloning_scenarios", "chat_qa", notes="LAB-Bench cloning scenarios: molecular cloning design scenarios."),
    _upstream(
        "mask",
        "chat_qa",
        notes="MASK: measures honesty by comparing stated beliefs to revealed behaviour. Upstream uses binary and numeric judge models.",
    ),
    _upstream(
        "moru",
        "chat_qa",
        notes="MORU: multilingual open-ended response understanding benchmark. Upstream uses grader_models.",
    ),
    _upstream("persistbench_cross_domain", "chat_qa", notes="PersistBench cross-domain: cross-domain persistence and instruction-following evaluation."),
    _upstream("persistbench_sycophancy", "chat_qa", notes="PersistBench sycophancy: sycophancy persistence under repeated pressure."),
    _upstream("persistbench_beneficial_memory", "chat_qa", notes="PersistBench beneficial memory: evaluates retention of beneficial information across turns."),
    _upstream("stereoset", "chat_qa", notes="StereoSet: intrasentence and intersentence stereotyping evaluation."),
    _upstream(
        "strong_reject",
        "chat_qa",
        notes="StrongREJECT: jailbreak-resistance evaluation using a judge LLM. Upstream uses judge_llm.",
    ),
    _upstream("coconot", "chat_qa", notes="CoCoNot: complex compositional instruction-following and refusal benchmark."),
    # WMDP — weapons of mass destruction proxy (three domains).
    _upstream("wmdp_bio", "chat_qa", notes="WMDP biology: biosecurity knowledge proxy benchmark."),
    _upstream("wmdp_chem", "chat_qa", notes="WMDP chemistry: chemical-weapons knowledge proxy benchmark."),
    _upstream("wmdp_cyber", "chat_qa", notes="WMDP cyber: cybersecurity uplift proxy benchmark."),
    # Bias benchmarks verified against inspect_evals==0.22.0.
    _upstream("bbq", "chat_qa", notes="BBQ: bias benchmark for question answering across nine social categories."),
    _upstream("bold", "chat_qa", notes="BOLD: bias in open-ended language generation across demographic groups."),
    # Personality benchmarks verified against inspect_evals==0.22.0.
    _upstream("personality_bfi", "chat_qa", task="personality_BFI", notes="Personality BFI: Big Five Inventory personality trait elicitation benchmark."),
    _upstream("personality_trait", "chat_qa", task="personality_TRAIT", notes="Personality TRAIT: trait-based personality evaluation benchmark."),
    # Writing benchmarks verified against inspect_evals==0.22.0.
    _upstream(
        "writingbench",
        "chat_qa",
        notes="WritingBench: comprehensive writing quality evaluation across multiple domains. Upstream uses judge_model.",
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
    if benchmark.id == "math":
        # The upstream factory also registers a model equivalence scorer. Inspect
        # runs every scorer, so the catalog task keeps only expression_exact_match.
        from inspect_ai.scorer._scorer import unique_scorer_name

        names: list[str] = []
        kept = []
        for item in task.scorer or ():
            scorer_name = unique_scorer_name(item, names)
            names.append(scorer_name)
            if scorer_name == "expression_exact_match":
                kept.append(item)
        if len(kept) != 1:
            raise ValueError("math requires the expression_exact_match scorer")
        task.scorer = kept
    task.dataset = with_sample_ids(task.dataset)
    return task

"""Construct native Inspect tasks from exact prepared data and plain task arguments."""

from __future__ import annotations

import importlib
import inspect
import json
from datetime import datetime
from pathlib import Path

from inspect_ai import Task
from inspect_ai._util.registry import registry_info, registry_lookup, registry_params
from inspect_ai.agent import as_solver
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import accuracy, choice, exact, f1, match, mean, pattern, stderr
from inspect_ai.solver import generate, multiple_choice, prompt_template, system_message

from yugos.inspect_engine import freeze_task

# Explicit installed implementations; pack arguments cannot choose a Python import.
_MODULES = {
    **{
        f"inspect_evals/{name}": f"inspect_evals.{name}.{name}"
        for name in (
            "mmlu_pro",
            "simpleqa",
            "ifeval",
            "livebench",
            "hle",
            "winogrande",
            "hellaswag",
            "commonsense_qa",
            "bigcodebench",
            "bfcl",
            "swe_bench",
            "musr",
            "bbh",
            "medqa",
            "gsm8k",
            "boolq",
            "squad",
            "drop",
            "piqa",
            "race_h",
            "truthfulqa",
            "humaneval",
            "mbpp",
            "math",
            "paws",
            "worldsense",
            "pubmedqa",
            "bbeh",
            "ds1000",
        )
    },
    "inspect_evals/gpqa_diamond": "inspect_evals.gpqa.gpqa",
    "inspect_evals/arc_challenge": "inspect_evals.arc.arc",
    "inspect_evals/arc_easy": "inspect_evals.arc.arc",
    "inspect_evals/swe_bench_verified_mini": "inspect_evals.swe_bench.swe_bench",
    "inspect_evals/simpleqa_verified": "inspect_evals.simpleqa.simpleqa",
    "inspect_evals/agent_bench_os": "inspect_evals.agent_bench.agent_bench",
    **{
        f"inspect_evals/aime{year}": f"inspect_evals.aime{year}.aime{year}"
        for year in (2024, 2025, 2026)
    },
    **{
        f"inspect_evals/tau2_{domain}": "inspect_evals.tau2.tau2"
        for domain in ("airline", "retail", "telecom", "banking")
    },
    **{
        f"yugos_evals/{name}": f"yugos_evals.tasks.{name}"
        for name in (
            "math_answer",
            "code_tests",
            "pairwise_judge",
            "rubric_chat",
            "boxed_math",
            "reading",
            "cruxeval",
            "aider_polyglot",
        )
    },
}

# Task-level metrics these builders restore. Other callbacks still need their own builder.
_RESTORED_METRICS = {
    "inspect_evals/squad",
    "inspect_evals/drop",
    "inspect_evals/race_h",
    "inspect_evals/worldsense",
    "inspect_evals/bbeh",
}
_CUSTOM_COMPONENTS = {
    "inspect_evals/bigcodebench",
    "inspect_evals/swe_bench",
    "inspect_evals/swe_bench_verified_mini",
    "inspect_evals/bbeh",
    "inspect_evals/ds1000",
    "inspect_evals/agent_bench_os",
}


def _require_nltk_data(directory: Path):
    """Fail before native checker initialization could download a missing resource."""
    import nltk

    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}", paths=[str(directory)])
        except LookupError as exc:
            raise ValueError("NLTK data must be staged in the execution image") from exc


def prepare_task(task: Task, **options):
    """Retain source arguments and any few-shot data while acquisition is permitted."""
    name = registry_info(task).name
    if name not in _MODULES:
        raise ValueError(f"task needs an installed prepared-data constructor: {name}")
    factory = registry_lookup("task", name)
    bound = inspect.signature(factory).bind_partial(**registry_params(task))
    bound.apply_defaults()
    args = dict(bound.arguments)
    # These are task inputs, never logged solver/scorer representations.
    args = json.loads(json.dumps(args, allow_nan=False))
    if name in _CUSTOM_COMPONENTS and (args.get("scorer") or args.get("solver")):
        raise ValueError("custom task components need their own installed builder")
    disallowed = [
        field
        for field in ("cleanup", "on_checkpoint", "on_resume", "approval", "review", "metrics")
        if getattr(task, field) is not None
    ]
    if name in _RESTORED_METRICS:
        disallowed = [field for field in disallowed if field != "metrics"]
    if disallowed:
        raise ValueError("custom task callbacks or metrics need their own installed builder")
    fewshot_samples = []
    if name in {"inspect_evals/mmlu_pro", "inspect_evals/winogrande"} and args.get("fewshot"):
        module = importlib.import_module(_MODULES[name])
        fields = dict(path=module.DATASET_PATH, sample_fields=module.record_to_sample)
        if name.endswith("mmlu_pro"):
            fields.update(split="validation", revision=module.MMLU_PRO_DATASET_REVISION)
        else:
            fields.update(
                name=args.get("dataset_name", "winogrande_xl"),
                split="train",
                shuffle=args.get("fewshot_shuffle", True),
                seed=args.get("fewshot_seed", 42),
                limit=args["fewshot"],
                revision=module.WINOGRANDE_DATASET_REVISION,
            )
        fewshot_samples = [s.model_dump(mode="json") for s in module.hf_dataset(**fields)]
    if name == "inspect_evals/mbpp":
        # The prompt embeds three pinned few-shot problems. Capture it while
        # acquisition is still allowed so execution does not reload them.
        fewshot_samples = [{"prompt_template": _mbpp_template(task)}]
    if name.startswith("inspect_evals/tau2_"):
        options["required_roles"] = tuple(set(options.get("required_roles", ())) | {"user"})
    if name == "inspect_evals/bigcodebench":
        from yugos_evals.scorers.python_unittest import GRADER_VERSION, bigcodebench_scorer

        task.scorer = [bigcodebench_scorer()]
        task.metadata = {**(task.metadata or {}), "unittest_supervisor_version": GRADER_VERSION}
    if name == "inspect_evals/math":
        # Upstream also builds a model equivalence scorer. Inspect would run it,
        # sending the gold target to the model under evaluation. Keep the
        # deterministic exact-match scorer on this task object too.
        module = importlib.import_module(_MODULES[name])
        task.scorer = [module.expression_exact_match()]
    return freeze_task(
        task,
        builder="yugos_evals.prepared",
        origin={"task": name, "args": args},
        builder_args={"task": name, "args": args, "fewshot_samples": fewshot_samples},
        **options,
    )


def _mbpp_template(task: Task) -> str:
    """The prompt template already rendered into an MBPP task's solver chain."""
    solvers = task.solver if isinstance(task.solver, list) else list(task.solver)
    templates = [
        params["template"]
        for solver in solvers
        if isinstance((params := registry_params(solver)).get("template"), str)
    ]
    if len(templates) != 1:
        raise ValueError("MBPP prompt template was not prepared")
    return templates[0]


def build(*, options: dict, task: str, args: dict, fewshot_samples: list[dict] = ()) -> Task:
    """Build native solvers, agents, tools and scorers without calling source factories."""
    if task not in _MODULES:
        raise ValueError(f"unknown prepared task: {task}")
    module = importlib.import_module(_MODULES[task])
    name = task.split("/")[1]
    solver, scorer, setup, metrics = generate(), None, None, None
    if name in {"mmlu_pro", "winogrande"}:
        solver = [multiple_choice(template=module.USER_PROMPT_TEMPLATE, shuffle=False)]
        count = args.get("fewshot", 0)
        if count:
            examples = MemoryDataset([Sample.model_validate(s) for s in fewshot_samples])
            if not examples:
                raise ValueError("few-shot examples must be prepared with the task")
            if name == "mmlu_pro":
                solver.insert(0, module.mmlu_pro_system_message(examples, count))
            else:
                solver.insert(
                    0,
                    system_message(
                        module.SYSTEM_W_EXAMPLES_PROMPT_TEMPLATE.format(
                            examples="\n\n".join(module.sample_to_fewshot(s) for s in examples)
                        )
                    ),
                )
        scorer = choice()
    elif name == "gpqa_diamond":
        solver, scorer = multiple_choice(cot=args.get("cot", True)), choice()
    elif name in {"simpleqa", "simpleqa_verified"}:
        scorer = module._resolve_scorer(
            args.get("scorer", "tool"),
            module.SIMPLEQA_VERIFIED_GRADER_TEMPLATE
            if name == "simpleqa_verified"
            else module.SIMPLEQA_GRADER_TEMPLATE,
            args.get("grade_parse", "strict"),
        )
    elif name in {"arc_challenge", "arc_easy"}:
        solver, scorer = multiple_choice(), choice()
    elif name == "boolq":
        solver, scorer = [generate()], pattern(r"\A[\W_]*(Yes|No)[\W_]*\Z")
    elif name == "paws":
        solver = [prompt_template(template=module.TEMPLATE), generate()]
        scorer = pattern(r"\A\W*\b(Yes|No)\b\W*\Z")
    elif name == "pubmedqa":
        solver, scorer = [multiple_choice(template=module.TEMPLATE)], choice()
    elif name == "worldsense":
        scorer = module.pattern_with_metadata(
            r"^\(?\s*(1|2|3|TRUE|FALSE|IMPOSSIBLE|POSSIBLE)\s*\)?"
        )
        metrics = [accuracy(), stderr(), module.ws_accuracy(), module.ws_bias()]
    elif name == "bbeh":
        from inspect_ai.scorer import grouped

        if args.get("solver"):
            raise ValueError("custom BBEH components need their own installed builder")
        scorer = module.bbeh_scorer()
        metrics = [
            grouped(accuracy(), group_key="task", all=False),
            module.harmonic_mean_across_tasks(),
        ]
    elif name == "squad":
        solver = [system_message(module.SYSTEM_MESSAGE), generate()]
        scorer = [f1(), exact()]
        metrics = [mean(), stderr(cluster="context_hash")]
    elif name == "drop":
        if args.get("fewshot", 3):
            raise ValueError("DROP few-shot examples must be prepared with the task")
        solver = module.drop_solver(
            fewshot=0,
            fewshot_seed=args.get("fewshot_seed", 42),
            fewshot_shuffle=args.get("fewshot_shuffle", True),
        )
        scorer = f1(module.extract_answer)
        metrics = [mean(), stderr(cluster="passage_hash")]
    elif name == "piqa":
        solver, scorer = [multiple_choice(template=module.TEMPLATE)], choice()
    elif name == "race_h":
        solver, scorer = [multiple_choice(template=module.TEMPLATE)], choice()
        metrics = [accuracy(), stderr(cluster="article_hash")]
    elif name == "truthfulqa":
        solver = [multiple_choice(multiple_correct=args.get("target", "mc1") != "mc1")]
        scorer = choice()
    elif name == "musr":
        solver = [
            system_message(module.SYSTEM_PROMPT),
            multiple_choice(
                template=module.get_domain_prompt(
                    args.get("domain", module.DEFAULT_DOMAIN),
                    args.get("prompt_technique", module.DEFAULT_PROMPT_TECHNIQUE),
                    args.get("example_count", module.DEFAULT_EXAMPLE_COUNT),
                )
            ),
        ]
        scorer = choice()
    elif name == "bbh":
        options = {
            **options,
            "dataset": [
                sample.model_copy(
                    update={
                        "metadata": {
                            **sample.metadata,
                            "dataset_type": module.DatasetType(sample.metadata["dataset_type"]),
                        }
                    }
                )
                if sample.metadata and "dataset_type" in sample.metadata
                else sample
                for sample in options["dataset"]
            ],
        }
        solver, scorer = module.bbh_solver(), module.bbh_scorer()
    elif name == "medqa":
        solver, scorer = multiple_choice(template=module.TEMPLATE), choice()
    elif name == "gsm8k":
        if args.get("fewshot", 10):
            raise ValueError("GSM8K few-shot examples must be prepared with the task")
        solver = [prompt_template(module.MATH_PROMPT_TEMPLATE), generate()]
        scorer = match(numeric=True)
    elif name == "ifeval":
        _require_nltk_data(Path.home() / ".cache" / "nltk_data")
        options = {
            **options,
            "dataset": [
                s.model_copy(
                    update={
                        "metadata": {
                            **s.metadata,
                            "kwargs": {int(k): v for k, v in s.metadata["kwargs"].items()},
                        }
                    }
                )
                for s in options["dataset"]
            ],
        }
        scorer = module.instruction_following()
    elif name == "livebench":
        from inspect_evals.constants import INSPECT_EVALS_CACHE_PATH

        _require_nltk_data(INSPECT_EVALS_CACHE_PATH / "livebench" / "nltk_data")
        options = {
            **options,
            "dataset": [
                sample.model_copy(
                    update={
                        "metadata": {
                            **sample.metadata,
                            "livebench_release_date": datetime.fromisoformat(
                                sample.metadata["livebench_release_date"]
                            ),
                            "livebench_removal_date": (
                                datetime.fromisoformat(sample.metadata["livebench_removal_date"])
                                if sample.metadata.get("livebench_removal_date")
                                else None
                            ),
                        }
                    }
                )
                for sample in options["dataset"]
            ],
        }
        solver, scorer = [system_message(module.SYSTEM_PROMPT), generate()], module.decide_scorer()
    elif name == "hle":
        import inspect_evals.hle.scorers  # noqa: F401

        solver = generate(**module.default_run_config().get("solver", {}).get("args", {}))
        scorer = module.scorers_from_spec(args.get("scorer", module.DEFAULT_TASK_ARGS["scorer"]))
    elif name == "hellaswag":
        solver, scorer = [system_message(module.SYSTEM_MESSAGE), multiple_choice()], choice()
    elif name == "commonsense_qa":
        solver, scorer = multiple_choice(), choice()
    elif name.startswith("aime"):
        solver, scorer = module.aime_solver(), module.aime_scorer()
    elif name == "math":
        if args.get("fewshot", 0):
            raise ValueError("MATH few-shot examples must be prepared with the task")
        if args.get("grader_model"):
            raise ValueError("MATH grading does not call a model")
        solver = module.math_solver(fewshot=0, fewshot_seed=args.get("fewshot_seed", 42))
        scorer = module.expression_exact_match()
    elif name == "humaneval":
        if args.get("solver") or args.get("scorer"):
            raise ValueError("custom HumanEval components need their own installed builder")
        scorer = module.verify()
    elif name == "mbpp":
        template = (
            fewshot_samples[0]["prompt_template"] if fewshot_samples else module.PROMPT_TEMPLATE
        )
        solver = [prompt_template(template), generate()]
        scorer = module.verify()
    elif name == "bigcodebench":
        if args.get("solver") or args.get("scorer"):
            raise ValueError("custom BigCodeBench components need their own installed builder")
        from yugos_evals.scorers.python_unittest import bigcodebench_scorer

        scorer = bigcodebench_scorer()
    elif name == "ds1000":
        solver, scorer = module.ds1000_solver(), module.ds1000_scorer()
    elif name == "agent_bench_os":
        if args.get("solver") or args.get("sandbox_config"):
            raise ValueError("custom AgentBench components need their own installed builder")
        solver = as_solver(module.default_agent(args.get("max_attempts", 1)))
        scorer = module.agent_bench_os_default_scorer()
    elif name == "bfcl":
        solver, scorer = module.bfcl_solver(), module.bfcl_scorer()
    elif name in {"swe_bench", "swe_bench_verified_mini"}:
        if args.get("scorer"):
            raise ValueError("custom SWE-bench components need their own installed builder")
        solver = module.swe_bench_agent_with_inspect_tool_support(
            tool_timeout=args.get("tool_timeout", module.DEFAULT_TOOL_TIMEOUT)
        )
        scorer = module.swe_bench_scorer()
    elif name.startswith("tau2_"):
        domain = name.removeprefix("tau2_")
        agents = importlib.import_module(f"inspect_evals.tau2.{domain}.agents")
        agent_args = {
            "instance": f"{domain.title()}Agent",
            "message_limit": args.get("message_limit"),
        }
        if domain == "banking":
            agent_args["retrieval_config"] = agents.validate_retrieval_config(
                args.get("retrieval_config", "grep")
            )
        solver = as_solver(agents.get_user_agent(**agent_args))
        if domain == "telecom":
            solver = [module.telecom_initializer(), solver]
        elif domain == "banking":
            solver = [module.banking_initializer(), solver]
        scorer = getattr(module, f"{domain}_scorer")()
    elif name == "math_answer":
        scorer = module.math_answer_scorer()
    elif name == "reading":
        solver, scorer = module.reading_components(args.get("source", "belebele"))
    elif name == "cruxeval":
        scorer = module.cruxeval_output_scorer()
    elif name == "code_tests":
        scorer = module.code_tests_scorer()
    elif name == "aider_polyglot":
        setup = module.polyglot_setup()
        solver = module.polyglot_agent(args.get("tool_timeout", module.TOOL_TIMEOUT))
        scorer = module.polyglot_scorer()
    elif name == "pairwise_judge":
        model = args.get("judge_model")
        setup = [module.judge_pin_guard(model)] if model else None
        scorer = module.pairwise_judge_scorer(model)
    elif name == "rubric_chat":
        scorer = module.rubric_chat_scorer()
    elif name == "boxed_math":
        scorer = module.math_answer_scorer()
    if metrics is not None:
        options = {**options, "metrics": metrics}
    return Task(**options, solver=solver, scorer=scorer, setup=setup)

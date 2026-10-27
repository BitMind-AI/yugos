"""Prepared samples use native grading without component serialization or source loading."""

import importlib
import math

import pytest
from evals_fakes import DIGEST_IMAGE, ScriptedRouter, run
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig

from yugos.evaluation import Question
from yugos.inspect_engine import RouterReply, RouterToolCall, restore_tasks, router_model
from yugos_evals.catalog import entries, entry, freeze_entry, load_task


@pytest.mark.parametrize(
    "benchmark",
    entries(status="supported"),
    ids=lambda benchmark: benchmark.id,
)
def test_every_catalog_constructor_uses_prepared_data_without_source_acquisition(
    benchmark, monkeypatch
):
    from inspect_ai._util.entrypoints import ensure_entry_points
    from inspect_ai._util.registry import registry_lookup

    from yugos_evals.prepared import build

    if benchmark.id == "ifeval":
        pytest.importorskip("instruction_following_eval")
    if benchmark.id == "livebench_reasoning":
        pytest.importorskip("livebench")

    ensure_entry_points(benchmark.source)
    factory = registry_lookup("task", benchmark.task)
    module = importlib.import_module(factory.__module__)
    for name in vars(module):
        if name == "hf_dataset" or name.endswith("_dataset") or name.startswith("load_"):
            value = getattr(module, name)
            if callable(value):
                monkeypatch.setattr(
                    module, name, lambda *a, **kw: pytest.fail("source acquired during execution")
                )
    metadata = {"kwargs": {}}
    if benchmark.id == "livebench_reasoning":
        metadata.update(
            category="reasoning",
            task="zebra_puzzle",
            livebench_release_date="2024-11-25T00:00:00",
            livebench_removal_date=None,
        )
    samples = [Sample(id="selected", input="question", target="reference", metadata=metadata)]
    task = build(options={"dataset": samples}, task=benchmark.task, args=benchmark.args)
    assert [s.model_dump(mode="json") for s in task.dataset] == [
        s.model_dump(mode="json") for s in samples
    ]
    assert task.solver is not None and task.scorer


def test_prepared_livebench_restores_dates_for_native_zebra_scorer(monkeypatch):
    from inspect_ai.scorer import includes

    from yugos_evals.prepared import build

    module = importlib.import_module("inspect_evals.livebench.livebench")
    monkeypatch.setattr("yugos_evals.prepared._require_nltk_data", lambda _: None)
    monkeypatch.setattr(module, "decide_scorer", includes)
    sample = Sample(
        id="selected",
        input="question",
        target="answer",
        metadata={
            "category": "reasoning",
            "task": "zebra_puzzle",
            "livebench_release_date": "2024-11-25T00:00:00",
            "livebench_removal_date": "2025-04-02T00:00:00",
        },
    )

    task = build(options={"dataset": [sample]}, task="inspect_evals/livebench", args={})
    [restored] = task.dataset
    assert restored.metadata["livebench_release_date"].strftime("%Y-%m-%d") == "2024-11-25"
    assert restored.metadata["livebench_removal_date"].strftime("%Y-%m-%d") == "2025-04-02"
    assert restored.model_dump(mode="json") == sample.model_dump(mode="json")


@pytest.mark.parametrize("source", ["mmlu_pro", "winogrande"])
@pytest.mark.parametrize("args", [{"fewshot": 0}, {"fewshot": 1}, {}])
def test_native_choice_prompts_and_fewshot_defaults_survive_preparation(
    source, args, monkeypatch, tmp_path
):
    import inspect_ai._util.appdirs as appdirs

    from yugos_evals.prepared import prepare_task

    monkeypatch.setattr(appdirs, "user_data_path", lambda _name: tmp_path / "inspect-data")
    module = importlib.import_module(f"inspect_evals.{source}.{source}")
    example = Sample(
        id="example",
        input="An example [BLANK]?",
        choices=["one", "two"],
        target="B",
        metadata={"subject": "math", "cot_content": "example reasoning"},
    )
    selected = example.model_copy(update={"id": "selected", "input": "A question [BLANK]?"})

    def dataset(**kw):
        evaluation_split = "test" if source == "mmlu_pro" else "validation"
        return MemoryDataset([selected] if kw["split"] == evaluation_split else [example])

    monkeypatch.setattr(module, "hf_dataset", dataset)
    native = getattr(module, source)(**args)
    questions = [
        Question.model_validate_json(q.model_dump_json())
        for q in prepare_task(native, source=source, workload_class="chat_qa")
    ]
    monkeypatch.setattr(module, "hf_dataset", lambda **kw: pytest.fail("few-shot data reloaded"))
    [(prepared, _)] = restore_tasks(questions)
    requests = []
    for label, task in (("native", native), ("prepared", prepared)):
        miner = ScriptedRouter("ANSWER: B")
        log = run(task, miner, tmp_path / label)
        assert log.status == "success"
        [sample] = log.samples
        [score] = sample.scores.values()
        assert score.value == "C"
        requests.append(miner.calls[0].request)
    assert requests[0] == requests[1]


@pytest.mark.parametrize(
    "benchmark_id,module_path,loader,answer",
    [
        ("gsm8k", "inspect_evals.gsm8k.gsm8k", "hf_dataset", "ANSWER: 2"),
        ("arc_challenge", "inspect_evals.arc.arc", "hf_dataset", "ANSWER: B"),
        ("arc_easy", "inspect_evals.arc.arc", "hf_dataset", "ANSWER: B"),
        ("musr", "inspect_evals.musr.musr", "hf_dataset", "ANSWER: B"),
        ("medqa", "inspect_evals.medqa.medqa", "load_hf_dataset_with_script", "ANSWER: B"),
        ("boolq", "inspect_evals.boolq.boolq", "hf_dataset", "Yes"),
        ("piqa", "inspect_evals.piqa.piqa", "load_hf_dataset_with_script", "ANSWER: B"),
        ("race_h", "inspect_evals.race_h.race_h", "hf_dataset", "ANSWER: B"),
        ("truthfulqa", "inspect_evals.truthfulqa.truthfulqa", "hf_dataset", "ANSWER: B"),
        ("drop", "inspect_evals.drop.drop", "hf_dataset", "ANSWER: B"),
        ("paws", "inspect_evals.paws.paws", "hf_dataset", "Yes"),
        ("pubmedqa", "inspect_evals.pubmedqa.pubmedqa", "hf_dataset", "ANSWER: B"),
    ],
)
def test_new_text_tasks_keep_native_request_and_grade_without_reloading_source(
    benchmark_id, module_path, loader, answer, monkeypatch, tmp_path
):
    # A wrong prepared prompt or scorer can silently change a hosted grade.
    import inspect_ai._util.appdirs as appdirs

    monkeypatch.setattr(appdirs, "user_data_path", lambda _name: tmp_path / "inspect-data")
    module = importlib.import_module(module_path)
    # Cluster metrics on these tasks reject a sample that omits the cluster id.
    clusters = {"race_h": "article_hash", "drop": "passage_hash"}
    sample = Sample(
        id="selected",
        input="What is one plus one?",
        choices=None if benchmark_id in {"gsm8k", "boolq", "drop", "paws"} else ["one", "two"],
        target={"gsm8k": "2", "boolq": "Yes", "paws": "Yes"}.get(benchmark_id, "B"),
        metadata={clusters[benchmark_id]: "cluster"} if benchmark_id in clusters else None,
    )
    if benchmark_id == "pubmedqa":
        import json
        from pathlib import Path

        test_ids = json.loads(
            (Path(module.__file__).parent / "data/test_ground_truth.json").read_text()
        )
        sample.id, sample.metadata = next(iter(test_ids)), {}
    monkeypatch.setattr(module, loader, lambda **kw: MemoryDataset([sample]))
    benchmark = entry(benchmark_id)
    native = load_task(benchmark)
    questions = freeze_entry(benchmark, task=native)
    monkeypatch.setattr(module, loader, lambda **kw: pytest.fail("source reloaded"))
    [(prepared, _)] = restore_tasks(questions)

    results = []
    for name, task in (("native", native), ("prepared", prepared)):
        miner = ScriptedRouter(answer)
        log = run(task, miner, tmp_path / name)
        assert log.status == "success"
        [scored] = log.samples
        [score] = scored.scores.values()
        results.append((miner.calls[0].request, score.value))
    assert results[0] == results[1]


@pytest.mark.parametrize("answer,grade", [("ANSWER: A", "C"), ("ANSWER: B", "I")])
def test_presented_choice_prompt_reaches_the_router_as_written(
    answer, grade, monkeypatch, tmp_path
):
    # A leftover native template would let a router recognise the benchmark.
    import inspect_ai._util.appdirs as appdirs
    from inspect_ai import task_with

    from yugos_evals.prepared import prepare_task

    monkeypatch.setattr(appdirs, "user_data_path", lambda _name: tmp_path / "inspect-data")
    module = importlib.import_module("inspect_evals.arc.arc")
    original = Sample(id="q", input="What is one plus one?", choices=["one", "two"], target="B")
    monkeypatch.setattr(module, "hf_dataset", lambda **kw: MemoryDataset([original]))
    native = load_task(entry("arc_challenge"))
    reworded = Sample(
        id="q~r",
        input="Add one to one.\n(A) two\n(B) one\nFinish with a line ANSWER: <letter>.",
        choices=["two", "one"],
        target="A",
    )
    questions = prepare_task(
        task_with(native, dataset=MemoryDataset([reworded])),
        presented=True,
        source="private_reasoning",
        workload_class="chat_qa",
    )
    [(prepared, _)] = restore_tasks(questions)
    miner = ScriptedRouter(answer)
    log = run(prepared, miner, tmp_path / "presented")
    [scored] = log.samples
    [score] = scored.scores.values()
    assert score.value == grade
    [message] = miner.calls[0].request.messages
    assert message.text == reworded.input


@pytest.mark.parametrize("benchmark_id", ["worldsense", "bbeh"])
@pytest.mark.parametrize("correct", [True, False])
def test_reasoning_task_metrics_and_grades_survive_preparation(
    benchmark_id, correct, monkeypatch, tmp_path
):
    module = importlib.import_module(f"inspect_evals.{benchmark_id}.{benchmark_id}")
    world = benchmark_id == "worldsense"
    target = "TRUE" if world else "2"
    answer = target if correct else ("FALSE" if world else "3")
    metadata = (
        {"tuple_ID": "fixture", "problemname": "Infer.normal", "problemsize": 2}
        if world
        else {"task": "boolean_expressions"}
    )
    sample = Sample(id="selected", input="A fixture question", target=target, metadata=metadata)
    loader = "load_worldsense_dataset" if world else "load_bbeh_dataset"
    monkeypatch.setattr(module, loader, lambda *a, **kw: MemoryDataset([sample]))
    benchmark = entry(benchmark_id)
    native = load_task(benchmark)
    questions = freeze_entry(benchmark, task=native)
    monkeypatch.setattr(module, loader, lambda *a, **kw: pytest.fail("source reloaded"))
    [(prepared, _)] = restore_tasks(questions)
    results = []
    for name, task in (("native", native), ("prepared", prepared)):
        miner = ScriptedRouter(answer)
        log = run(task, miner, tmp_path / name)
        assert log.status == "success"
        [scored] = log.samples
        [score] = scored.scores.values()
        assert score.value == ("C" if correct else "I")
        results.append((miner.calls[0].request, log.results.scores, score.value))
    assert results[0] == results[1]


@pytest.mark.parametrize(
    ("benchmark_id", "module_path"),
    [("squad", "inspect_evals.squad.squad"), ("math", "inspect_evals.math.math")],
)
def test_multi_scorer_entries_freeze_the_authoritative_scorer(
    benchmark_id, module_path, monkeypatch
):
    # Squad and MATH register several scorers; freezing must keep the catalog's
    # choice, and restoration must rebuild that scorer without reloading data.
    from inspect_ai.scorer._scorer import unique_scorer_name

    module = importlib.import_module(module_path)
    monkeypatch.setattr(
        module,
        "hf_dataset",
        lambda **kw: MemoryDataset([Sample(id="selected", input="question", target="answer")]),
    )
    benchmark = entry(benchmark_id)
    questions = freeze_entry(benchmark, task=load_task(benchmark))
    assert [question.inspect_task["scorer"] for question in questions] == [benchmark.scorer]
    monkeypatch.setattr(module, "hf_dataset", lambda **kw: pytest.fail("source reloaded"))
    [(prepared, _)] = restore_tasks(questions)
    names: list[str] = []
    for scorer in prepared.scorer or ():
        names.append(unique_scorer_name(scorer, names))
    assert benchmark.scorer in names


def test_prepared_math_grades_without_calling_a_model(monkeypatch, tmp_path):
    # expression_equivalance calls get_model(None) with the gold target in the
    # prompt. Restoring or grading MATH must not construct or run that scorer.
    from inspect_ai.scorer import CORRECT
    from inspect_ai.scorer._scorer import unique_scorer_name

    module = importlib.import_module("inspect_evals.math.math")
    monkeypatch.setattr(
        module,
        "hf_dataset",
        lambda **kwargs: MemoryDataset([Sample(id="selected", input="What is 1 + 1?", target="2")]),
    )
    benchmark = entry("math")
    native = load_task(benchmark)
    questions = freeze_entry(benchmark, task=native)
    monkeypatch.setattr(module, "hf_dataset", lambda **kwargs: pytest.fail("source reloaded"))

    def banned_scorer(*args, **kwargs):
        raise AssertionError("MATH constructed a scorer other than expression_exact_match")

    monkeypatch.setattr(module, "expression_equivalance", banned_scorer)
    monkeypatch.setattr(module, "expression_exact_match_sympy", banned_scorer)
    monkeypatch.setattr(
        "inspect_evals.math.utils.get_model",
        lambda *args, **kwargs: pytest.fail("MATH grading called a model"),
    )
    [(prepared, _)] = restore_tasks(questions)

    def scorer_names(task):
        names: list[str] = []
        for scorer in task.scorer or ():
            names.append(unique_scorer_name(scorer, names))
        return names

    assert scorer_names(native) == ["expression_exact_match"]
    assert scorer_names(prepared) == ["expression_exact_match"]
    for label, task in (("native", native), ("prepared", prepared)):
        miner = ScriptedRouter("ANSWER: 2")
        log = run(task, miner, tmp_path / label)
        assert log.status == "success"
        [sample] = log.samples
        assert list(sample.scores) == ["expression_exact_match"]
        assert sample.scores["expression_exact_match"].value == CORRECT
        assert len(miner.calls) == 1
        assert "Look at the following two expressions" not in miner.seen()


@pytest.mark.parametrize("benchmark_id", ["humaneval", "mbpp"])
def test_humaneval_and_mbpp_freeze_the_pinned_python_sandbox(benchmark_id, monkeypatch):
    # Upstream sets sandbox="docker" with no image. Freezing must substitute the
    # catalog's digest-pinned Compose sandbox, the same shape code_tests uses.
    from yugos.inspect_engine.assets import frozen_sandbox
    from yugos_evals.catalog import PYTHON_IMAGE
    from yugos_evals.tasks.code_tests import code_sandbox

    if benchmark_id == "humaneval":
        module = importlib.import_module("inspect_evals.humaneval.humaneval")
        monkeypatch.setattr(
            module,
            "hf_dataset",
            lambda **kwargs: MemoryDataset(
                [Sample(id="HumanEval/0", input="def f():\n", target="pass\n")]
            ),
        )
    else:
        module = importlib.import_module("inspect_evals.mbpp.mbpp")

        class Rows(list):
            def filter(self, predicate):
                return Rows(row for row in self if predicate(row))

        monkeypatch.setattr(
            module,
            "load_dataset",
            lambda *args, **kwargs: Rows(
                [
                    {
                        "task_id": task_id,
                        "text": f"add {task_id}",
                        "test_list": [f"assert f({task_id}) == {task_id}"],
                        "code": f"def f(x):\n    return {task_id}\n",
                    }
                    for task_id in (2, 3, 4)
                ]
            ),
        )
        monkeypatch.setattr(
            module,
            "hf_dataset",
            lambda **kwargs: MemoryDataset(
                [
                    Sample(
                        id="10",
                        input="solve",
                        target=["assert True"],
                        metadata={"test_list_str": "assert True"},
                    )
                ]
            ),
        )
    benchmark = entry(benchmark_id)
    questions = freeze_entry(benchmark, task=load_task(benchmark))
    expected = frozen_sandbox(code_sandbox(PYTHON_IMAGE))
    assert [question.inspect_task["sandbox"] for question in questions] == [expected]
    [(prepared, definition)] = restore_tasks(questions)
    assert definition.sandbox == expected
    service = prepared.sandbox.config.services["default"]
    assert service.image == PYTHON_IMAGE
    assert service.network_mode == "none"
    assert service.command == "tail -f /dev/null"


def test_prepared_mbpp_keeps_the_captured_fewshot_prompt(monkeypatch):
    # The sanitized prompt embeds three pinned examples. Dropping them, or
    # reloading the dataset while restoring, would grade a different task.
    from inspect_ai._util.registry import registry_params

    from yugos_evals.tasks.code_tests import code_sandbox

    module = importlib.import_module("inspect_evals.mbpp.mbpp")

    class Rows(list):
        def filter(self, predicate):
            return Rows(row for row in self if predicate(row))

    monkeypatch.setattr(
        module,
        "load_dataset",
        lambda *args, **kwargs: Rows(
            [
                {
                    "task_id": task_id,
                    "text": f"add {task_id}",
                    "test_list": [f"assert f({task_id}) == {task_id}"],
                    "code": f"def f(x):\n    return {task_id}\n",
                }
                for task_id in (2, 3, 4)
            ]
        ),
    )
    monkeypatch.setattr(
        module,
        "hf_dataset",
        lambda **kwargs: MemoryDataset(
            [
                Sample(
                    id="10",
                    input="solve",
                    target=["assert True"],
                    metadata={"test_list_str": "assert True"},
                )
            ]
        ),
    )
    benchmark = entry("mbpp")
    native = load_task(benchmark)
    questions = freeze_entry(benchmark, task=native, sandbox=code_sandbox(DIGEST_IMAGE))
    monkeypatch.setattr(
        module, "load_dataset", lambda *args, **kwargs: pytest.fail("source reloaded")
    )
    monkeypatch.setattr(module, "hf_dataset", lambda **kwargs: pytest.fail("source reloaded"))
    [(prepared, definition)] = restore_tasks(questions)
    assert definition.scorer == "verify"

    def template(task):
        for solver in task.solver:
            stored = registry_params(solver).get("template")
            if isinstance(stored, str):
                return stored
        raise AssertionError("prompt template missing")

    captured = template(native)
    assert template(prepared) == captured
    assert "add 2" in captured and "add 4" in captured


def test_tau2_banking_prepared_task_uses_the_initializer_and_db_scorer():
    from inspect_ai._util.registry import registry_info

    from yugos_evals.prepared import build

    task = build(
        options={"dataset": [Sample(id="selected", input="question", target="done")]},
        task="inspect_evals/tau2_banking",
        args=entry("tau2_banking").args,
    )
    solvers = list(task.solver)
    assert registry_info(solvers[0]).name.endswith("banking_initializer")
    assert registry_info(solvers[1]).name.endswith("user_simulator_agent")
    assert registry_info(task.scorer[0]).name.endswith("banking_scorer")


def test_bbh_restores_dataset_type_needed_by_native_solver_and_scorer():
    # Freezing turns BBH's Enum into an integer; without restoration both native
    # components silently select the wrong branch for binary-choice samples.
    from inspect_evals.bbh.bbh import DatasetType

    from yugos_evals.prepared import build

    sample = Sample(
        id="selected",
        input="Is the statement true?",
        target="yes",
        metadata={
            "dataset_type": DatasetType.BINARY_CHOICE.value,
            "dataset_name": "boolean_expressions",
        },
    )
    task = build(options={"dataset": [sample]}, task="inspect_evals/bbh", args={})
    assert task.dataset[0].metadata["dataset_type"] is DatasetType.BINARY_CHOICE
    assert task.dataset[0].model_dump(mode="json") == sample.model_dump(mode="json")


@pytest.mark.parametrize(
    ("grade", "expected"),
    [("CORRECT", "C"), ("INCORRECT", "I"), ("NOT_ATTEMPTED", "N"), ("invalid", None)],
)
@pytest.mark.parametrize("scorer", ["tool", "original"])
@pytest.mark.parametrize("source_id", ["simpleqa", "simpleqa_verified"])
def test_simpleqa_native_and_prepared_grading_agree_without_loading_sources(
    grade, expected, scorer, source_id, tmp_path, monkeypatch
):
    import inspect_ai._util.appdirs as appdirs

    monkeypatch.setattr(appdirs, "user_data_path", lambda _name: tmp_path / "inspect-data")
    module = importlib.import_module("inspect_evals.simpleqa.simpleqa")
    loader = "simpleqa_verified_dataset" if source_id == "simpleqa_verified" else "simpleqa_dataset"
    monkeypatch.setattr(
        module,
        loader,
        lambda: MemoryDataset([Sample(input="question", target="private reference")]),
    )
    benchmark = entry(source_id)
    if scorer == "original":
        benchmark = benchmark.model_copy(update={"score_key": "correct"})
    native = load_task(benchmark, scorer=scorer)
    native.config = GenerateConfig(max_tokens=1024)
    questions = [
        Question.model_validate_json(q.model_dump_json())
        for q in freeze_entry(benchmark, task=native)
    ]
    monkeypatch.setattr(module, loader, lambda: pytest.fail("source reloaded"))
    monkeypatch.setattr(module, source_id, lambda **kw: pytest.fail("source factory called"))

    calls = []

    async def judge(call):
        calls.append(call)
        if scorer == "original":
            letter = {"CORRECT": "A", "INCORRECT": "B", "NOT_ATTEMPTED": "C"}.get(grade, "?")
            return RouterReply(content=letter, finish_reason="stop")
        return RouterReply(
            tool_calls=(
                RouterToolCall(id="verdict", function="submit", arguments={"grade": grade}),
            ),
            finish_reason="tool_calls",
        )

    pin = benchmark.role_pins["grader"]
    grader = router_model(judge, submission_id=pin.model, config=pin.config())
    [(prepared, _)] = restore_tasks(questions, model_roles={"grader": grader})
    values = []
    for label, task in (("native", native), ("prepared", prepared)):
        miner = ScriptedRouter("candidate answer")
        log = run(task, miner, tmp_path / label, model_roles={"grader": grader})
        assert log.status == "success"
        [sample] = log.samples
        [score] = sample.scores.values()
        values.append(score.value)
        assert "private reference" not in miner.seen()
        assert miner.calls[0].request.config.max_tokens == 1024
    if scorer == "original":
        expected = {
            "correct": float(grade == "CORRECT"),
            "incorrect": float(grade == "INCORRECT"),
            "not_attempted": float(grade not in {"CORRECT", "INCORRECT"}),
        }
        assert values == [expected, expected]
    elif grade == "invalid":
        assert all(math.isnan(value) for value in values)
    else:
        assert values == [expected, expected]
    assert len(calls) == 2
    assert calls[0].request == calls[1].request
    assert "private reference" in str(calls[1].request.messages)


@pytest.mark.parametrize("mutation", ["replace", "in_place"])
def test_prepared_builder_must_be_installed_pinned_and_keep_the_selected_data(
    monkeypatch, mutation
):
    module = importlib.import_module("inspect_evals.simpleqa.simpleqa")
    monkeypatch.setattr(
        module,
        "simpleqa_dataset",
        lambda: MemoryDataset([Sample(input="question", target="reference")]),
    )
    [question] = freeze_entry(entry("simpleqa"))
    pin = entry("simpleqa").role_pins["grader"]

    async def no_calls(call):
        pytest.fail("provider called during restoration")

    roles = {"grader": router_model(no_calls, submission_id=pin.model, config=pin.config())}
    unknown = Question.model_validate_json(question.model_dump_json())
    unknown.inspect_task["builder"] = "unknown/task"
    with pytest.raises(ValueError, match="installed registration"):
        restore_tasks([unknown], model_roles=roles)
    unpinned = Question.model_validate_json(question.model_dump_json())
    unpinned.inspect_task["packages"].pop("yugos-evals")
    with pytest.raises(ValueError, match="distribution is not pinned"):
        restore_tasks([unpinned], model_roles=roles)

    import yugos_evals.prepared as builders

    original = builders.build

    def substitute(**kw):
        task = original(**kw)
        if mutation == "replace":
            task.dataset = MemoryDataset([Sample(input="different", target="different")])
        else:
            task.dataset[0].target = "different"
        return task

    monkeypatch.setattr(builders, "build", substitute)
    with pytest.raises(ValueError, match="exactly the prepared samples"):
        restore_tasks([question], model_roles=roles)


@pytest.mark.parametrize("task", ["inspect_evals/ifeval", "inspect_evals/livebench"])
def test_missing_checker_resources_fail_before_any_download(task, monkeypatch):
    nltk = pytest.importorskip("nltk")
    from yugos_evals.prepared import build

    def missing(*a, **kw):
        raise LookupError("fixture resource absent")

    monkeypatch.setattr(nltk.data, "find", missing)
    monkeypatch.setattr(nltk, "download", lambda *a, **kw: pytest.fail("runtime download"))
    with pytest.raises(ValueError, match="staged in the execution image"):
        build(options={"dataset": []}, task=task, args={})


@pytest.mark.parametrize("success", [True, False])
def test_ds1000_restores_native_requests_and_checker_invocation(success, monkeypatch, tmp_path):
    from inspect_ai.util import ExecResult

    module = importlib.import_module("inspect_evals.ds1000.ds1000")
    sample = Sample(
        id="q",
        input="Complete the program.",
        target="private reference",
        metadata={
            "code_context": "def test_execution(solution): pass",
            "library": "Pandas",
            "perturbation_type": "Origin",
        },
    )
    monkeypatch.setattr(module, "ds1000_dataset", lambda *a, **kw: MemoryDataset([sample]))
    native = load_task(entry("ds1000"))
    native.sandbox = None
    frozen = freeze_entry(entry("ds1000"), task=native)
    monkeypatch.setattr(module, "ds1000_dataset", lambda *a, **kw: pytest.fail("source reloaded"))
    [(prepared, _)] = restore_tasks(frozen)

    class Checker:
        def __init__(self):
            self.writes = []
            self.execs = []

        async def write_file(self, path, content):
            self.writes.append((path, content))

        async def exec(self, cmd, **kwargs):
            self.execs.append((cmd, kwargs))
            return ExecResult(success, 0 if success else 1, "", "")

    invocations = []
    requests = []
    for label, task in [("native", native), ("prepared", prepared)]:
        checker = Checker()
        monkeypatch.setattr(module, "sandbox", lambda: checker)
        router = ScriptedRouter("<code>result = 2</code>")
        log = run(task, router, tmp_path / label)
        assert log.status == "success"
        assert next(iter(log.samples[0].scores.values())).value == ("C" if success else "I")
        assert "private reference" not in router.seen()
        invocations.append((checker.writes, checker.execs))
        requests.append(router.calls[0].request)
    # Pack IDs namespace the temporary filename; checker code and execution
    # settings must stay identical, and each command must name its staged file.
    for writes, executions in invocations:
        assert len(writes) == len(executions) == 1
        assert executions[0][0] == ["python", writes[0][0]]
    assert invocations[0][0][0][1] == invocations[1][0][0][1]
    assert invocations[0][1][0][1] == invocations[1][1][0][1]
    assert requests[0] == requests[1]


@pytest.mark.parametrize("answer,correct", [("private answer", True), ("wrong", False)])
def test_agentbench_restores_native_submission_and_match_grading(
    answer, correct, monkeypatch, tmp_path
):
    module = importlib.import_module("inspect_evals.agent_bench.agent_bench")
    sample = Sample(
        id="q",
        input="Find the answer.",
        target="",
        metadata={"evaluation": {"match": "private answer"}},
    )
    monkeypatch.setattr(module, "agent_bench_os_dataset", lambda **kw: MemoryDataset([sample]))
    native = load_task(entry("agent_bench_os"))
    frozen = freeze_entry(entry("agent_bench_os"), task=native)
    monkeypatch.setattr(
        module, "agent_bench_os_dataset", lambda **kw: pytest.fail("source reloaded")
    )
    [(prepared, _)] = restore_tasks(frozen)
    requests = []
    for label, task in [("native", native), ("prepared", prepared)]:
        seen = []

        async def reply(call):
            seen.append(call.request)
            return RouterReply(
                tool_calls=(
                    RouterToolCall(id="done", function="submit", arguments={"answer": answer}),
                ),
                finish_reason="tool_calls",
            )

        log = run(task, reply, tmp_path / label)
        assert log.status == "success"
        assert next(iter(log.samples[0].scores.values())).value is correct
        assert len(seen) == 1 and "private answer" not in str(seen[0].messages)
        requests.append(seen[0])
    assert requests[0] == requests[1]

"""The catalog names real tasks with settings their factories accept."""

import inspect

import pytest
from inspect_ai._util.entrypoints import ensure_entry_points
from inspect_ai._util.registry import registry_lookup

from yugos_evals.catalog import CATALOG, PYTHON_IMAGE, BenchmarkEntry, entries, entry


def test_ids_are_unique_and_lookup_fails_for_unknown_ids():
    ids = [e.id for e in CATALOG]
    assert len(ids) == len(set(ids))
    assert entry(ids[0]) is CATALOG[0]
    with pytest.raises(KeyError):
        entry("not_a_benchmark")


def test_filters_preserve_catalog_order():
    planned = entries(status="planned")
    assert planned and all(e.status == "planned" for e in planned)
    agentic = entries(category="agentic")
    assert list(agentic) == [e for e in CATALOG if e.category == "agentic"]


@pytest.mark.parametrize(
    "fields",
    [
        {"category": "vision"},
        {"source": "inspect_evals", "task": "yugos_evals/x"},
        {"epochs": 2},
        {"id": "Bad-Id"},
        {"sandbox_image": "python:3.12-slim"},
    ],
)
def test_entries_reject_invalid_fields(fields):
    base = {
        "id": "x",
        "category": "chat_qa",
        "source": "yugos_evals",
        "task": "yugos_evals/x",
        "status": "planned",
    }
    with pytest.raises(ValueError):
        BenchmarkEntry(**{**base, **fields})


@pytest.mark.parametrize(
    "benchmark", [e for e in CATALOG if e.status == "supported"], ids=lambda e: e.id
)
def test_supported_tasks_resolve_and_accept_their_settings(benchmark):
    # Catches a renamed upstream task or argument after an Inspect Evals upgrade,
    # without calling the factory (which would download the dataset).
    package = benchmark.task.split("/")[0]
    ensure_entry_points(package)
    factory = registry_lookup("task", benchmark.task)
    assert factory is not None, benchmark.task
    inspect.signature(factory).bind_partial(**benchmark.args)


@pytest.mark.parametrize(
    ("benchmark_id", "category", "task", "scorer", "args"),
    [
        ("arc_easy", "chat_qa", "inspect_evals/arc_easy", None, {}),
        ("boolq", "chat_qa", "inspect_evals/boolq", None, {"shuffle": False}),
        ("squad", "chat_qa", "inspect_evals/squad", "f1", {"shuffle": False}),
        ("drop", "chat_qa", "inspect_evals/drop", None, {"fewshot": 0}),
        ("piqa", "chat_qa", "inspect_evals/piqa", None, {}),
        ("race_h", "chat_qa", "inspect_evals/race_h", None, {"shuffle": False}),
        (
            "truthfulqa",
            "chat_qa",
            "inspect_evals/truthfulqa",
            None,
            {"target": "mc1", "shuffle": False},
        ),
        ("humaneval", "math_code", "inspect_evals/humaneval", None, {}),
        ("mbpp", "math_code", "inspect_evals/mbpp", None, {}),
        ("math", "math_code", "inspect_evals/math", "math_answer_scorer", {}),
        (
            "tau2_banking",
            "agentic",
            "inspect_evals/tau2_banking",
            None,
            {"shuffle": False},
        ),
        (
            "swe_bench_verified_mini",
            "agentic",
            "inspect_evals/swe_bench_verified_mini",
            None,
            {},
        ),
    ],
)
def test_added_benchmarks_resolve_with_their_category_and_scorer(
    benchmark_id, category, task, scorer, args
):
    # Locks the twelve added ids: a renamed task, a drifted category, or a
    # forgotten authoritative scorer would otherwise pass the generic loops.
    benchmark = entry(benchmark_id)
    assert benchmark is next(item for item in CATALOG if item.id == benchmark_id)
    assert benchmark.category == category
    assert benchmark.task == task
    assert benchmark.status == "supported"
    assert benchmark.epochs == 1
    assert benchmark.scorer == scorer
    assert benchmark.args == args
    if benchmark_id in {"humaneval", "mbpp"}:
        assert benchmark.sandbox_image == PYTHON_IMAGE
    else:
        assert benchmark.sandbox_image is None
    if benchmark_id == "tau2_banking":
        assert benchmark.role_pins == entry("tau2_airline").role_pins
    else:
        assert not benchmark.role_pins


def test_role_pins_name_direct_openai_models_for_service_roles():
    pinned = {e.id: e.role_pins for e in CATALOG if e.role_pins}
    assert {"simpleqa", "hle", "arenahard_v2", "tau2_airline"} <= pinned.keys()
    for pins in pinned.values():
        # Pins use the direct provider route, never an aggregator.
        assert all(pin.model.startswith("openai/") for pin in pins.values())


def test_hle_judge_spec_builds_one_scorer_for_the_pinned_grader():
    # Catches a judge spec the pinned Inspect Evals release no longer accepts, or a
    # multi-judge panel whose authoritative scorer would be ambiguous when frozen.
    # Inspect Evals 0.22.0 registers the HLE judges only once their module is imported;
    # its own default spec fails the same way without this import (upstream defect).
    import inspect_evals.hle.scorers  # noqa: F401
    from inspect_evals.hle.hle import scorers_from_spec

    hle = entry("hle")
    [scorer] = scorers_from_spec(hle.args["scorer"])
    assert hle.args["scorer"]["graders"] == list(hle.role_pins) == ["grader"]


def _pinned_entry(**pins):
    return BenchmarkEntry(
        id="pinned",
        category="chat_qa",
        source="yugos_evals",
        task="yugos_evals/pinned",
        status="supported",
        role_pins=pins,
    )


def _task(metadata=None):
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import model_graded_qa
    from inspect_ai.solver import generate

    return Task(
        dataset=[Sample(id=f"s{i}", input=f"Question {i}?", target="A") for i in range(3)],
        solver=generate(),
        scorer=model_graded_qa(),
        metadata=metadata,
    )


def test_freezing_an_entry_records_its_pins_in_every_question(monkeypatch):
    from yugos.inspect_engine import read_role_pins, restore_tasks
    from yugos_evals.catalog import freeze_entry

    benchmark = _pinned_entry(grader={"model": "mockllm/model", "params": {"temperature": 0}})
    from evals_fakes import freeze_fixture

    monkeypatch.setattr("yugos_evals.prepared.prepare_task", freeze_fixture)
    questions = freeze_entry(benchmark, task=_task(), sample_ids=["s0", "s2"])
    assert [q.question_id for q in questions] == ["pinned:s0", "pinned:s2"]
    for question in questions:
        assert question.inspect_task["required_roles"] == ["grader"]
        pins = read_role_pins(question.inspect_task["metadata"])
        assert pins == benchmark.role_pins
    with pytest.raises(ValueError, match="explicitly bound"):
        restore_tasks(questions)


def test_task_pins_must_agree_with_the_catalog():
    from yugos.inspect_engine import RolePin, read_role_pins, role_pins_metadata
    from yugos_evals.catalog import pin_roles

    benchmark = _pinned_entry(grader={"model": "a/judge", "params": {"temperature": 0}})
    # A task's model-only pin gains the catalog's settings.
    task = pin_roles(_task(role_pins_metadata({"grader": RolePin(model="a/judge")})), benchmark)
    assert read_role_pins(task.metadata) == benchmark.role_pins
    for other in (RolePin(model="a/other"), RolePin(model="a/judge", params={"temperature": 1})):
        with pytest.raises(ValueError, match="differently"):
            pin_roles(_task(role_pins_metadata({"grader": other})), benchmark)


def test_unnamed_upstream_samples_freeze_with_order_independent_ids(monkeypatch):
    # Catches SimpleQA's missing IDs, row-number identities, and target-based IDs.
    from inspect_ai.dataset import MemoryDataset, Sample

    from yugos.inspect_engine import restore_tasks
    from yugos_evals.catalog import freeze_entry, load_task

    samples = [
        Sample(input="First?", target="one"),
        Sample(input="Second?", target="two"),
        Sample(id="upstream-id", input="Third?", target="three"),
    ]
    upstream = MemoryDataset(samples, name="official", location="pinned.csv", shuffled=True)
    import importlib

    module = importlib.import_module("inspect_evals.winogrande.winogrande")
    monkeypatch.setattr(module, "hf_dataset", lambda **kw: upstream)
    benchmark = entry("winogrande")  # No model roles needed for this source-shape fixture.
    task = load_task(benchmark)
    ids = {sample.input: sample.id for sample in task.dataset}
    assert len(set(ids.values())) == len(samples) and None not in ids.values()
    assert ids["Third?"] == "upstream-id" and samples[0].id is None
    assert (task.dataset.name, task.dataset.location, task.dataset.shuffled) == (
        upstream.name,
        upstream.location,
        upstream.shuffled,
    )
    samples.reverse()
    samples[-1].target = "changed reference"
    assert {s.input: s.id for s in load_task(benchmark).dataset} == ids
    questions = freeze_entry(benchmark, task=task, sample_ids=[ids["First?"]])
    [(restored, _)] = restore_tasks(questions)
    assert restored.dataset[0].input == "First?" and restored.dataset[0].target == "one"


def test_duplicate_unnamed_inputs_fail_before_freezing(monkeypatch):
    import importlib

    from inspect_ai.dataset import MemoryDataset, Sample

    from yugos_evals.catalog import freeze_entry, load_task

    module = importlib.import_module("inspect_evals.winogrande.winogrande")
    monkeypatch.setattr(
        module,
        "hf_dataset",
        lambda **kw: MemoryDataset(
            [Sample(input="Same?", target="one"), Sample(input="Same?", target="two")]
        ),
    )
    benchmark = entry("winogrande")
    with pytest.raises(ValueError, match="unique stable IDs"):
        freeze_entry(benchmark, task=load_task(benchmark))

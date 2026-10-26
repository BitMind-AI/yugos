"""Reading contracts retain source context, evidence metrics and offline execution."""

import json

import pytest
from evals_fakes import ScriptedRouter, run, task_state
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Target

from yugos.inspect_engine import restore_tasks
from yugos_evals.catalog import entry, freeze_entry, load_task
from yugos_evals.scorers.reading import hotpot_metrics, longbench_answer
from yugos_evals.sources._common import LoadedSource


def test_belebele_translations_share_master_identity(tmp_path):
    from yugos_evals.sources.reading import load_belebele

    row = {
        "link": "passage",
        "question_number": "1",
        "flores_passage": "A passage",
        "question": "Which?",
        "correct_answer_num": "2",
        **{f"mc_answer{i}": f"choice {i}" for i in range(1, 5)},
    }
    (tmp_path / "data").mkdir()
    for language in ["fra_Latn", "spa_Latn"]:
        (tmp_path / f"data/{language}.jsonl").write_text(json.dumps(row) + "\n")
    samples = load_belebele(["fra_Latn", "spa_Latn"], data_dir=tmp_path).samples
    assert samples[0].id != samples[1].id
    assert samples[0].metadata["master_question_id"] == samples[1].metadata["master_question_id"]
    assert samples[0].target == "B"


def test_longbench_preserves_the_entire_context(monkeypatch, tmp_path):
    from yugos_evals.sources import reading

    context = "middle information " * 20000
    row = {
        "_id": "q",
        "context": context,
        "question": "What?",
        "answer": "C",
        "domain": "documents",
        "sub_domain": "single",
        "difficulty": "hard",
        "length": "short",
        **{f"choice_{letter}": letter for letter in "ABCD"},
    }
    path = tmp_path / "data.json"
    path.write_text(json.dumps([row]))
    monkeypatch.setattr(reading, "hf_file", lambda *a, **kw: path)
    [sample] = reading.load_longbench_v2(["short"]).samples
    assert context.strip() in sample.input
    assert sample.target == "C" and "(C) C" in sample.input


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("The correct answer is (B)", "C"),
        ("**The correct answer is B**", "C"),
        ("The correct answer is (A)", "I"),
        ("B", "I"),
    ],
)
@pytest.mark.anyio
async def test_longbench_uses_the_published_answer_extraction(answer, expected):
    result = await longbench_answer()(task_state(answer), Target("B"))
    assert result.value == expected


def test_hotpot_joint_f1_combines_precision_and_recall():
    reference = {"answer": "New York City", "supporting_facts": [["a", 0], ["b", 1]]}
    prediction = {"answer": "New York", "supporting_facts": [["a", 0]]}
    result = hotpot_metrics(prediction, reference)
    assert result["f1"] == pytest.approx(0.8)
    assert result["sp_f1"] == pytest.approx(2 / 3)
    assert result["joint_f1"] == pytest.approx(0.5)
    assert result["joint_em"] == 0


def test_hotpot_boolean_answers_and_malformed_evidence():
    reference = {"answer": "yes", "supporting_facts": [["a", 0]]}
    assert (
        hotpot_metrics({"answer": "no", "supporting_facts": [["a", 0]]}, reference)["joint_f1"] == 0
    )
    assert (
        hotpot_metrics({"answer": "yes", "supporting_facts": [["a", True]]}, reference)["joint_f1"]
        == 0
    )
    assert (
        hotpot_metrics({"answer": "Yes!", "supporting_facts": [["a", 0], ["a", 0]]}, reference)[
            "joint_f1"
        ]
        == 1
    )


@pytest.mark.parametrize("benchmark_id", ["belebele", "longbench_v2", "hotpotqa"])
def test_reading_freeze_restores_request_and_grade_without_fetching(
    benchmark_id, monkeypatch, tmp_path
):
    import importlib

    module = importlib.import_module("yugos_evals.tasks.reading")
    reference = json.dumps({"answer": "yes", "supporting_facts": [["a", 0]]})
    sample = Sample(
        id="q",
        input="Read the supplied text",
        target=reference if benchmark_id == "hotpotqa" else "B",
        choices=["one", "two"] if benchmark_id == "belebele" else None,
    )
    loader = {
        "belebele": "load_belebele",
        "longbench_v2": "load_longbench_v2",
        "hotpotqa": "load_hotpotqa",
    }[benchmark_id]
    monkeypatch.setattr(module, loader, lambda *a, **kw: LoadedSource([sample]))
    benchmark = entry(benchmark_id)
    native = load_task(benchmark)
    questions = freeze_entry(benchmark, task=native)
    monkeypatch.setattr(
        module, loader, lambda *a, **kw: pytest.fail("source acquired during execution")
    )
    [(prepared, _)] = restore_tasks(questions)
    answer = {
        "belebele": "ANSWER: B",
        "longbench_v2": "The correct answer is (B)",
        "hotpotqa": reference,
    }[benchmark_id]
    results = []
    for name, task in [("native", native), ("prepared", prepared)]:
        router = ScriptedRouter(answer)
        log = run(task, router, tmp_path / name)
        assert log.status == "success"
        [scored] = log.samples
        [score] = scored.scores.values()
        assert score.value == "C" or isinstance(score.value, dict) and score.value["joint_f1"] == 1
        results.append((router.calls[0].request, score.value))
    assert results[0] == results[1]

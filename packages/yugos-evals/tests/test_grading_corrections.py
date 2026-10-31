"""Reviewed examples, source drift and benchmark-owned grading contracts."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
from evals_fakes import LocalProcesses, task_state
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Target

from yugos_evals._sample_patches import PATCHES
from yugos_evals.prepared import _prepare_grading, build
from yugos_evals.scorers.math import math_answer_scorer
from yugos_evals.scorers.mbpp import mbpp_with_imports
from yugos_evals.scorers.simpleqa import simpleqa_numeric_scorer

FIXTURES = Path(__file__).with_name("fixtures")
SAMPLES = json.loads((FIXTURES / "grading_samples.json").read_text())


def test_reviewed_math_equivalents_and_wrong_answers():
    score = math_answer_scorer(answer_format="answer_line")
    for line in (FIXTURES / "math_equivalence.jsonl").read_text().splitlines():
        case = json.loads(line)
        state = task_state("ANSWER: " + case["answer"], input=case["problem"])
        assert anyio.run(score, state, Target(case["target"])).value == 1, case["id"]
    for gold, answer in [
        (r"\text{Sunday}", "not Sunday"),
        (r"\text{A,D}", "A,C"),
        ("(8,3,-1)", "(3,8,-1)"),
        ("1,2", "2,1"),
        ("1,2", "12"),
        ("2.2", "-2.2"),
        ("42", "41 or 42"),
        ("42", r"\boxed{41}\boxed{42}"),
        ("42", "42}41{"),
        (r"(-\infty,0]", r"(-\infty,0)"),
        ("1200_7", "1200_8"),
        ("4", "4 feet per second"),
        ("50%", "50"),
    ]:
        state = task_state("ANSWER: " + answer, input="Express your answer in inches per second.")
        assert anyio.run(score, state, Target(gold)).value == 0, (gold, answer)
    assert anyio.run(score, task_state("ANSWER: 0.5"), Target("50%")).value == 1
    percent = task_state(r"ANSWER: 50\%", input="Express your answer as a percentage.")
    assert anyio.run(score, percent, Target("50")).value == 1
    truncated = task_state("ANSWER: 42", stop_reason="max_tokens")
    assert anyio.run(score, truncated, Target("42")).value == 0


@pytest.mark.parametrize(
    "gold,answer,order,expected",
    [
        (r"1,\frac12", "0.5,1", "unordered", 1),
        (r"1,\frac{10000001}{10000000}", "1.0,1", "unordered", 1),
        ("1,1,2", "1,2,2", "unordered", 0),
        ("1,2", "2,1", "ordered", 0),
        ("1,234", "234,1", "unordered", 1),
        ("371043400", "371,043,400", "auto", 1),
        ("371,043,400", "371043400", "auto", 1),
    ],
)
def test_shared_math_handles_equivalent_parts_and_grouped_numbers(gold, answer, order, expected):
    question = {
        "unordered": "Enter the roots, separated by commas.",
        "ordered": "Enter the roots in ascending order, separated by commas.",
        "auto": "Compute.",
    }[order]
    state = task_state("ANSWER: " + answer, input=question)
    assert (
        anyio.run(math_answer_scorer(answer_format="answer_line"), state, Target(gold)).value
        == expected
    )


@pytest.mark.parametrize(
    "question,gold,answer,expected",
    [
        ("Enter the solutions, separated by commas.", r"2,\frac14", "0.25,2", 1),
        ("Enter the roots in descending order, separated by commas.", "2,1", "1,2", 0),
        ("Find the point's coordinates.", "(1,2,3)", "(3,2,1)", 0),
        (
            "A train moves in feet per second. Express the speed in meters per second?",
            "4",
            "4 meters per second",
            1,
        ),
        (
            "A train moves in feet per second. Express the speed in meters per second?",
            "4",
            "4 feet per second",
            0,
        ),
        ("How many books must Ada buy?", "12", "12 books", 1),
        ("How many of the books must Ada buy?", "12", "12 of", 0),
        ("How many more books must Ada buy?", "12", "12 more", 0),
        ("How many books must Ada buy?", "12", "12 marbles", 0),
        ("Express your answer in percent.", "50", r"50\%", 1),
        ("Express your answer in percent.", "50", "50 percent", 1),
        ("Express your answer in percent.", "50", r"50\text{percent}", 1),
        ("Express your answer in percent.", "50", "0.5", 0),
        ("Express your final answer in degrees.", "60", r"60^\circ", 1),
        ("Express your final answer in degrees.", "60", "60 radians", 0),
        ("Express your final answer in inches per second.", "4", "4 inches per second", 1),
        ("Express your final answer in inches per second.", "4", "4 feet per second", 0),
        ("Compute.", r"50\text{percent}", r"50\%", 1),
        ("Compute.", r"30^\circ,60^\circ", "30 degrees,60", 1),
        ("Compute.", r"30^\circ,60^\circ", r"30,60^{\circ}", 1),
        ("Compute.", r"30\text{degrees},60\text{degrees}", r"30^\circ,60 degrees", 1),
        ("Compute.", r"30^\circ,60^\circ", "30 radians,60", 0),
        ("Compute.", r"30^\circ,60^\circ", "60,30 degrees", 0),
        ("Compute.", r"30^\circ,60^\circ", "30 degrees,61", 0),
    ],
)
def test_math_rules_apply_to_new_questions_without_item_hints(question, gold, answer, expected):
    state = task_state("ANSWER: " + answer, input=question)
    assert (
        anyio.run(math_answer_scorer(answer_format="answer_line"), state, Target(gold)).value
        == expected
    )


def test_new_grader_does_not_change_legacy_recipes():
    sample = Sample(input="Compute.", target=r"\frac{1}{8}")
    state = task_state(r"ANSWER: \(\frac{1}{8}\)")
    for version, expected in [(0, "I"), (1, 1.0)]:
        task = build(
            options={"dataset": [sample]},
            task="inspect_evals/math",
            args={},
            grader_version=version,
        )
        [score] = task.scorer
        assert anyio.run(score, state, Target(sample.target)).value == expected
    with pytest.raises(ValueError, match="grader version"):
        build(options={"dataset": [sample]}, task="inspect_evals/math", args={}, grader_version=99)


def test_benchmark_versions_advance_independently(monkeypatch):
    from yugos_evals.scorers import mbpp

    def prepare(name):
        return _prepare_grading(
            Task(dataset=[Sample(id="fixture", input="q", target="42")]), name, {}
        )

    math_version, mbpp_version = prepare("math"), prepare("mbpp")
    monkeypatch.setattr(mbpp, "GRADER_VERSION", mbpp_version + 1)
    assert prepare("math") == math_version
    assert prepare("mbpp") == mbpp_version + 1


@pytest.mark.parametrize("row", SAMPLES, ids=lambda row: row["task"] + ":" + row["sample"]["id"])
def test_reviewed_sample_edits_are_checked_and_idempotent(row):
    sample = Sample.model_validate(row["sample"])
    task = Task(dataset=[sample])
    _prepare_grading(task, row["task"], {})
    after = sample.model_dump_json()
    for field, (_, replacement) in PATCHES[row["task"]][str(sample.id)].items():
        actual = (
            sample.metadata[field[9:]] if field.startswith("metadata.") else getattr(sample, field)
        )
        assert actual == replacement
    _prepare_grading(task, row["task"], {})
    assert sample.model_dump_json() == after
    drift = Sample.model_validate(row["sample"])
    if row["task"] == "mbpp" and str(drift.id) == "248":
        drift.input += " changed"
    else:
        drift.target = "changed reference"
    with pytest.raises(ValueError, match="reviewed original changed"):
        _prepare_grading(Task(dataset=[drift]), row["task"], {})


def test_native_code_checks_preserve_future_imports_and_cover_whitespace(monkeypatch):
    environment = LocalProcesses()

    async def execute(cmd, **kwargs):
        return await environment.exec(["python3", *cmd[1:]], **kwargs)

    for module_name in ["inspect_evals.mbpp.mbpp", "inspect_evals.humaneval.humaneval"]:
        monkeypatch.setattr(
            importlib.import_module(module_name), "sandbox", lambda: SimpleNamespace(exec=execute)
        )
    original = next(row["sample"] for row in SAMPLES if row["task"] == "humaneval")
    sample = Sample.model_validate(original)
    _prepare_grading(Task(dataset=[sample]), "humaneval", {})
    scorer = importlib.import_module("inspect_evals.humaneval.humaneval").verify()
    assert (
        anyio.run(
            scorer, task_state(original["target"], metadata=sample.metadata), Target(sample.target)
        ).value
        == "I"
    )
    assert (
        anyio.run(
            scorer, task_state(sample.target, metadata=sample.metadata), Target(sample.target)
        ).value
        == "C"
    )
    code = '"""Trusted fixture."""\nfrom __future__ import annotations\ndef harmonic_sum(n):\n    return sum(1/i for i in range(1,n+1))'
    state = task_state(code, metadata={"test_imports": ["import math"]})
    assert (
        anyio.run(
            mbpp_with_imports(), state, Target(["assert math.isclose(harmonic_sum(4),25/12)"])
        ).value
        == "C"
    )
    assert state.output.completion == code
    code = '"""π fixture."""; from __future__ import annotations; value = math.sqrt(4)\ndef f():\n    return value'
    state = task_state(code, metadata={"test_imports": ["import math"]})
    assert anyio.run(mbpp_with_imports(), state, Target(["assert f() == 2"])).value == "C"


@pytest.mark.parametrize(
    "gold,extracted,response,expected",
    [
        ("27,898", "28,500", "28,500", "I"),
        ("27,898", "27,898", "27,898", "C"),
        ("120000", "124000", "124000", "C"),
        ("120000", "115000", "115000", "C"),
        ("120000", "113000", "113000", "I"),
        ("28000", "28 thousand", "28 thousand", "C"),
        ("27,898", "27,898", "28,500", "C"),
    ],
)
def test_numeric_checks_respect_precision_and_keep_judge_fallbacks(
    monkeypatch, gold, extracted, response, expected
):
    from inspect_ai.scorer import Score
    from inspect_evals.simpleqa.grader_templates import SIMPLEQA_GRADER_TEMPLATE

    import yugos_evals.scorers.simpleqa as simpleqa

    def native(**kwargs):
        async def score(state, target):
            return Score(
                value="C",
                metadata={
                    "grader_payload": {
                        "grade": "CORRECT",
                        "extracted_answer": extracted,
                        "explanation": "fixture",
                    }
                },
            )

        return score

    monkeypatch.setattr(simpleqa, "schema_tool_graded_scorer", native)
    state = task_state("The population was " + response + " in 2015.")
    result = anyio.run(simpleqa_numeric_scorer(SIMPLEQA_GRADER_TEMPLATE), state, Target(gold))
    assert result.value == expected
    if extracted == "28 thousand" or extracted != response:
        assert result.metadata["numeric_check"] == "judge_fallback"

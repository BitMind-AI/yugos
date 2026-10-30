"""Regressions for rejected equivalents, weak tests and numeric judge mistakes."""

import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
from evals_fakes import LocalProcesses, task_state
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Target

from yugos_evals.corrections import correct_task
from yugos_evals.prepared import build
from yugos_evals.scorers.benchmark_fixes import mbpp_with_imports, simpleqa_checked
from yugos_evals.scorers.math_equivalence import math_equivalence


def test_reviewed_math_equivalents_and_wrong_answers():
    cases = Path(__file__).with_name("fixtures").joinpath("math_equivalence.jsonl")
    score = math_equivalence()
    for line in cases.read_text().splitlines():
        case = json.loads(line)
        state = task_state("ANSWER: " + case["answer"], input=case["problem"])
        assert anyio.run(score, state, Target(case["target"])).value == 1, case["id"]
    for gold, answer in [
        ("Sunday", "not Sunday"),
        ("A,D", "A,C"),
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
        state = task_state("ANSWER: " + answer, input="Speed expressed in inches per second?")
        assert anyio.run(score, state, Target(gold)).value == 0, (gold, answer)
    truncated = task_state("ANSWER: 42", stop_reason="max_tokens")
    assert anyio.run(score, truncated, Target("42")).value == 0
    assert anyio.run(score, task_state("ANSWER: 0.5"), Target("50%")).value == 1
    percent = task_state(r"ANSWER: 50\%", input="What percentage is half?")
    assert anyio.run(score, percent, Target("50")).value == 1
    ordered = task_state(
        "ANSWER: 2,1", input="Enter the roots in increasing order, separated by commas."
    )
    assert anyio.run(score, ordered, Target("1,2")).value == 0


def test_new_grading_revision_does_not_change_legacy_recipes():
    sample = Sample(input="Compute.", target=r"\frac{1}{8}")
    state = task_state(r"ANSWER: \(\frac{1}{8}\)")
    for revision, expected in [(0, "I"), (1, 1.0)]:
        task = build(
            options={"dataset": [sample]},
            task="inspect_evals/math",
            args={},
            grading_revision=revision,
        )
        [score] = task.scorer
        assert anyio.run(score, state, Target(sample.target)).value == expected
    with pytest.raises(ValueError, match="grading revision"):
        build(
            options={"dataset": [sample]}, task="inspect_evals/math", args={}, grading_revision=99
        )


def test_native_code_checks_use_imports_and_repaired_samples(monkeypatch):
    environment = LocalProcesses()

    async def execute(cmd, **kwargs):
        return await environment.exec(["python3", *cmd[1:]], **kwargs)

    for module_name in ["inspect_evals.mbpp.mbpp", "inspect_evals.humaneval.humaneval"]:
        monkeypatch.setattr(
            importlib.import_module(module_name), "sandbox", lambda: SimpleNamespace(exec=execute)
        )
    sample = Sample(
        id="HumanEval/125",
        input="Split whitespace.",
        target='    if " " in txt:\n        return txt.split()\n    return 3\n',
        metadata={
            "prompt": "def split_words(txt):\n    pass\n",
            "entry_point": "split_words",
            "test": "def check(candidate):\n    assert candidate('abcdef') == 3\n",
        },
    )
    task = Task(dataset=[sample])
    correct_task(task, "humaneval", {})
    once = sample.model_dump_json()
    correct_task(task, "humaneval", {})
    assert sample.model_dump_json() == once
    scorer = importlib.import_module("inspect_evals.humaneval.humaneval").verify()
    bad = task_state(
        'def split_words(txt):\n    return txt.split() if " " in txt else 3',
        metadata=sample.metadata,
    )
    assert anyio.run(scorer, bad, Target(sample.target)).value == "I"
    good = task_state(sample.target, metadata=sample.metadata)
    assert anyio.run(scorer, good, Target(sample.target)).value == "C"
    state = task_state(
        "def harmonic_sum(n):\n    return sum(1/i for i in range(1,n+1))",
        metadata={"test_imports": ["import math"]},
    )
    assert (
        anyio.run(
            mbpp_with_imports(), state, Target(["assert math.isclose(harmonic_sum(4), 25/12)"])
        ).value
        == "C"
    )
    sample = Sample(
        id="143",
        input="Count lists in a tuple.",
        target=["assert True", "assert True", "assert False"],
    )
    task = Task(dataset=[sample])
    correct_task(task, "mbpp", {})
    assert ",))" in sample.target[2]
    assert sample.metadata["test_list"] == sample.target
    sample = Sample(id="248", input="harmonic sum of n-1", target=[])
    correct_task(Task(dataset=[sample]), "mbpp", {})
    assert "1/n" in sample.input and "n-1" not in sample.input


@pytest.mark.parametrize(
    ("extracted", "response", "expected"),
    [("28,500", "28,500", "I"), ("27,898", "27,898", "C"), ("27,898", "28,500", None)],
)
def test_numeric_check_overrides_an_incorrect_judge_pass(
    monkeypatch, extracted, response, expected
):
    from inspect_ai.scorer import Score
    from inspect_evals.simpleqa.grader_templates import SIMPLEQA_GRADER_TEMPLATE

    import yugos_evals.scorers.benchmark_fixes as fixes

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

    monkeypatch.setattr(fixes, "schema_tool_graded_scorer", native)
    scorer = simpleqa_checked(SIMPLEQA_GRADER_TEMPLATE)
    state = task_state("The population was " + response + " in 2015.")
    value = anyio.run(scorer, state, Target("27,898")).value
    assert math.isnan(value) if expected is None else value == expected

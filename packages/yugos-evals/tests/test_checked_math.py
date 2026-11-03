"""Coverage for versioned math answer comparison."""

import json
from pathlib import Path

import pytest

from yugos.evaluation.grades import GradeCode
from yugos_evals.scorers._math_compare import grade_math as compare_math
from yugos_evals.scorers.math import grade_math, reference_exclusion


@pytest.mark.parametrize(
    "gold,answer",
    [
        (r"20\cdot\binom{40}{20}", r"20\binom{40}{20}"),
        (r"f(x)=x^2", r"f(n)=n^2"),
        (r"7\cdot2031-4=14213", "14213"),
        (r"\text{No}", "no"),
        (r"\text{Finite}", "finite"),
        (r"abc", r"a\cdot b\cdot c"),
        (r"\frac13", "0.333333"),
        (r"\sqrt2", "1.414214"),
        (r"\frac{2^{2048}}{2^{2049}}", r"\frac12"),
        ("2.2", r"\frac{11}{5}"),
    ],
)
def test_equivalent_complete_answers(gold, answer):
    assert grade_math(gold, rf"\boxed{{{answer}}}", comparison_version=2).score == 1


@pytest.mark.parametrize(
    "gold,answer",
    [
        (r"\frac{2^{2049}-1}{3\cdot2^{2048}-1}", r"\frac{2^{2048}}{2^{2049}}"),
        (r"\sqrt{\frac{2^{2049}-1}{3\cdot2^{2048}-1}}", r"\frac{2^{2048}}{3\cdot2^{2048}}"),
        (r"\frac{300}{2^{2053}}", r"\frac{\binom{2053}{400}}{2^{2053}}"),
        (r"\frac{300}{2^{2053}}", "0"),
        (r"f(x)=x^2", r"f(n)=n^3"),
        ("4", r"4\text{ when }x>0"),
        ("2", r"2\text{ or }3"),
        ("2", "2\nANSWER: 3"),
        ("2", "\\boxed{2}\n\\boxed{3}"),
        (r"u<3", r"0<u<3"),
        (r"a+b", r"A+B"),
        (r"10^{-50}", "0.0"),
        (r"10^{-50}", r"2.0\cdot10^{-50}"),
        ("1000000", "1000001.0"),
    ],
)
def test_distinct_or_ambiguous_answers_score_zero(gold, answer):
    assert grade_math(gold, rf"\boxed{{{answer}}}", comparison_version=2).score == 0


def test_last_explicit_answer_overrides_previous_box():
    assert grade_math("2", "\\boxed{2}\nANSWER: 3", comparison_version=2).score == 0
    assert grade_math("3", "\\boxed{2}\nANSWER: 3", comparison_version=2).score == 1


@pytest.mark.parametrize(
    "reference",
    [
        r"f(x)=cx+d",
        r"4\text{ when }x>0",
        "The minimum is $2$. The maximum is $7$.",
        r"\log(2)",
        "1,2",
    ],
)
def test_unsupported_reference_requires_review_at_preparation(reference):
    assert reference_exclusion(reference, comparison_version=2)


@pytest.mark.parametrize(
    "case",
    [
        json.loads(line)
        for line in (Path(__file__).parent / "fixtures/math_equivalence.jsonl")
        .read_text()
        .splitlines()
    ],
    ids=lambda case: case["id"],
)
def test_existing_math_presentation_cases_keep_credit(case):
    outcome = compare_math(
        case["target"],
        "ANSWER: " + case["answer"],
        answer_format="answer_line",
        question=case["problem"],
        comparison_version=2,
    )
    assert outcome.code == GradeCode.CORRECT


def test_nested_box_cannot_discard_outer_expression():
    assert grade_math("2", r"\boxed{1+\boxed{2}}", comparison_version=2).score == 0


def test_prepared_tasks_restore_each_supported_comparison_version():
    import anyio
    from evals_fakes import task_state
    from inspect_ai._util.registry import registry_params
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import Target

    from yugos_evals.prepared import build

    gold = r"\frac34"
    response = r"\boxed{\frac68}"
    for task_name, versions in [
        ("yugos_evals/boxed_math", [0, 2]),
        ("inspect_evals/math", [1, 2]),
    ]:
        for version in versions:
            task = build(
                options={"dataset": [Sample(id="q", input="Compute", target=gold)]},
                task=task_name,
                args={},
                grader_version=version,
            )
            completion = "ANSWER: " + response if task_name == "inspect_evals/math" else response
            scorer = task.scorer[0]
            assert registry_params(scorer)["comparison_version"] == version
            result = anyio.run(scorer, task_state(completion), Target(gold))
            assert result.value == 1


def test_preparation_requires_a_supported_reference():
    from inspect_ai.dataset import MemoryDataset, Sample

    from yugos_evals.prepared import prepare_task
    from yugos_evals.tasks.boxed_math import boxed_math

    task = boxed_math()
    task.dataset = MemoryDataset(
        [Sample(id="family", input="Find all functions", target="f(x)=cx+d")]
    )
    with pytest.raises(ValueError, match="unsupported math reference"):
        prepare_task(task, source="test", workload_class="math_code")


def test_checked_comparison_keeps_cheap_exact_sum_refutation(monkeypatch):
    from yugos_evals.scorers import _math_process

    monkeypatch.setattr(_math_process, "TIMEOUT_S", 3)
    outcome = grade_math("7", r"\boxed{3+\sum_{k=1}^{100000000}(1-f(k))}", comparison_version=2)
    assert outcome.code == GradeCode.INCORRECT

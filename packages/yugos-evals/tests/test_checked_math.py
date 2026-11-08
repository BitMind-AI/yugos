"""Coverage for math answer comparison."""

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
        (r"\frac{1}{24}", "0.041667"),
        (r"\frac{1}{81}", "0.012346"),
        (r"\sqrt2", "1.414214"),
        (r"\frac{2^{2048}}{2^{2049}}", r"\frac12"),
        ("2.2", r"\frac{11}{5}"),
    ],
)
def test_equivalent_complete_answers(gold, answer):
    assert grade_math(gold, rf"\boxed{{{answer}}}").score == 1


@pytest.mark.parametrize(
    "gold,answer",
    [
        (r"\frac{2^{2049}-1}{3\cdot2^{2048}-1}", r"\frac{2^{2048}}{2^{2049}}"),
        (r"\sqrt{\frac{2^{2049}-1}{3\cdot2^{2048}-1}}", r"\frac{2^{2048}}{3\cdot2^{2048}}"),
        (r"\frac{300}{2^{2053}}", r"\frac{\binom{2053}{400}}{2^{2053}}"),
        (r"\frac{300}{2^{2053}}", "0"),
        (r"f(x)=x^2", r"f(n)=n^3"),
        (r"h(x)=x^2", r"h(t)=x^2"),
        ("4", r"4\text{ when }x>0"),
        ("2", r"2\text{ or }3"),
        ("2", "2\nANSWER: 3"),
        ("2", "\\boxed{2}\n\\boxed{3}"),
        (r"u<3", r"0<u<3"),
        (r"a+b", r"A+B"),
        (r"10^{-50}", r"2.0\cdot10^{-50}"),
        ("1000000", "1000001.0"),
    ],
)
def test_distinct_or_ambiguous_answers_score_zero(gold, answer):
    assert grade_math(gold, rf"\boxed{{{answer}}}").score == 0


def test_last_explicit_answer_overrides_previous_box():
    assert grade_math("2", "\\boxed{2}\nANSWER: 3").score == 0
    assert grade_math("3", "\\boxed{2}\nANSWER: 3").score == 1


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
    assert reference_exclusion(reference)


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
    )
    assert outcome.code == GradeCode.CORRECT


@pytest.mark.parametrize("reference", ["2", r"\prod_{i=1}^{4}i!"])
def test_nested_box_cannot_discard_outer_expression(reference):
    assert grade_math(reference, rf"\boxed{{1+\boxed{{{reference}}}}}").score == 0


@pytest.mark.parametrize(
    "reference,answer",
    [("(1,2,3)", "(3,2,1)"), ("1,2,3", "(3,2,1)"), ("(1,2,3)", "3,2,1")],
)
def test_unordered_answers_match_tuples_and_columns(reference, answer):
    result = compare_math(
        reference,
        "ANSWER: " + answer,
        answer_format="answer_line",
        question="Enter the roots, separated by commas.",
    )
    assert result.code == GradeCode.CORRECT


def test_unsupported_parser_result_leaves_a_scored_outcome(monkeypatch):
    from yugos_evals.scorers import _math_compare

    parse = _math_compare.latex2sympy
    monkeypatch.setattr(
        _math_compare,
        "latex2sympy",
        lambda value, **options: [1, 2] if value == "x" else parse(value, **options),
    )
    result = compare_math("2", r"\boxed{x}")
    assert result.settled and result.code == GradeCode.INVALID_ANSWER


def test_checked_comparison_keeps_cheap_exact_sum_refutation(monkeypatch):
    from yugos_evals.scorers import _math_process

    monkeypatch.setattr(_math_process, "TIMEOUT_S", 3)
    outcome = grade_math("7", r"\boxed{3+\sum_{k=1}^{100000000}(1-f(k))}")
    assert outcome.code == GradeCode.INCORRECT

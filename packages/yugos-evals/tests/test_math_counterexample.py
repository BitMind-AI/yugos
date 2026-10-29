"""A cheap exact refutation must not change the frozen equality rules."""

import pytest
from math_verify import verify
from sympy import Function, Integer, Sum, Symbol

from yugos_evals.scorers._math_compare import _integer_counterexample
from yugos_evals.scorers.math import grade_math


def test_large_undefined_function_sum_has_an_exact_integer_counterexample():
    result = grade_math("7", r"\boxed{3+\sum_{k=1}^{100000000}(1-f(k))}")
    assert result.status == "graded" and result.score == 0
    assert result.code.value == "incorrect"


@pytest.mark.parametrize("terms", [1, 3, 11])
@pytest.mark.parametrize("gold", [0, 7, 29])
def test_refutations_agree_with_existing_strict_verifier(terms, gold):
    k = Symbol("k")
    expression = 3 + Sum(1 - Function("f")(k), (k, 1, terms))
    if _integer_counterexample(Integer(gold), expression):
        assert not verify(Integer(gold), expression, strict=True, timeout_seconds=5)


def test_matching_interpretation_is_not_accepted_as_equality():
    k = Symbol("k")
    expression = 3 + Sum(1 - Function("f")(k), (k, 1, 4))
    assert not _integer_counterexample(Integer(7), expression)
    assert not verify(Integer(7), expression, strict=True, timeout_seconds=5)


@pytest.mark.parametrize(
    "expression",
    [
        r"\boxed{\sum_{k=1}^{4}(f(k)-f(k))+7}",
        r"\boxed{\sum_{k=1}^{3} k +1}",
    ],
)
def test_valid_cancellation_and_defined_finite_sums_remain_correct(expression):
    result = grade_math("7", expression)
    assert result.status == "graded" and result.score == 1


def test_non_integer_or_symbolic_bounds_do_not_supply_a_refutation():
    k, n = Symbol("k"), Symbol("n", integer=True)
    expression = Sum(Function("f")(k), (k, 1, n))
    assert not _integer_counterexample(Integer(7), expression)
    assert not _integer_counterexample(Symbol("x"), expression)
    restricted = Function("positive", positive=True)(k)
    assert not _integer_counterexample(Integer(7), Sum(restricted, (k, 1, 4)))

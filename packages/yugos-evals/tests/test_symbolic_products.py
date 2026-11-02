"""Factorial products require exact proofs, not expansion or guessed mismatches."""

import pytest

from yugos_evals.scorers._math_compare import grade_math


@pytest.mark.parametrize(
    "reference,answer",
    [
        (r"2! \cdot 4! \cdots 2^{2023}!", r"\prod_{k=1}^{2024}(2^{k-1})!"),
        (r"\prod_{i=2}^{8}(3^i)!", r"\prod_{j=1}^{7}(3^{j+1})!"),
        (r"\prod_{i=1}^{5}(2^i)!", r"\prod_{i=1}^{5}(4^{i}/2^i)!"),
        (r"\prod_{i=1}^{4}i!", r"2!\cdot3!\cdot4!"),
    ],
)
def test_equivalent_products_normalize_without_factorial_expansion(reference, answer, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a factorial was expanded")

    monkeypatch.setattr("math.factorial", forbidden)
    monkeypatch.setattr("sympy.factorial", forbidden)
    result = grade_math(reference, rf"\boxed{{{answer}}}")
    assert result.settled and result.score == 1


@pytest.mark.parametrize("answer", [r"\prod_{i=1}^{6}(2^i)!", r"(2^{9})!", "0", "14"])
def test_proven_mismatches_are_incorrect(answer):
    result = grade_math(r"\prod_{i=1}^{8}(2^i)!", rf"\boxed{{{answer}}}")
    assert result.settled and result.score == 0


def test_explicit_indexed_definition_is_resolved():
    result = grade_math(
        r"\prod_{i=1}^{6}(2^i)!",
        r"Let \(m_j = 2^{j+1}\). Thus \boxed{\prod_{j=0}^{5}m_j!}",
    )
    assert result.settled and result.score == 1


@pytest.mark.parametrize(
    "answer",
    [
        r"\prod_{i=1}^{8}(2^i)!+1",
        r"\prod_{i=1}^{8}(2^i)!\cdot x",
        r"\prod_{i=1}^{8}(2^i)!=0",
        r"\prod_{i=1}^{8}(3^{i-1})!",
        r"\prod_{i=1}^{1000000}(2^i)!",
        r"\prod_{i=1}^{8}(2^{1000000})!",
        r"\left(\prod_{i=0}^{2024}m_i!\right)"
        r"\left(\left\lfloor\frac{F}{Q^E}\right\rfloor\bmod Q\right)",
    ],
)
def test_unresolved_or_oversized_expressions_never_invent_a_zero(answer):
    result = grade_math(r"\prod_{i=1}^{8}(2^i)!", rf"\boxed{{{answer}}}")
    assert not result.settled
    assert result.details["cause"] == "symbolic_comparison_unresolved"


def test_conflicting_definitions_are_unresolved():
    result = grade_math(
        r"\prod_{i=1}^{6}(2^i)!",
        r"m_j = 2^j\n m_j = 3^j\n \boxed{\prod_{j=1}^{6}m_j!}",
    )
    assert not result.settled


@pytest.mark.parametrize("definition", [r"m_0=99, m_j=2^j", r"m_j=2^j\quad(j\ge1)"])
def test_index_definitions_do_not_drop_exceptions_or_domain_restrictions(definition):
    result = grade_math(r"\prod_{i=0}^{5}(2^i)!", definition + r"\boxed{\prod_{j=0}^{5}m_j!}")
    assert not result.settled


def test_reference_screening_uses_the_same_complete_symbolic_parser(monkeypatch):
    from yugos_evals.scorers import _math_compare

    monkeypatch.setattr(_math_compare, "parse", lambda *a, **kw: pytest.fail("expanding parser"))
    assert _math_compare.reference_exclusion(r"\prod_{i=1}^{2023}(2^i)!") is None
    assert _math_compare.reference_exclusion(r"\prod_{i=1}^{n}(2^i)!") is not None


def test_zero_coefficients_and_unary_precedence_do_not_invent_inequalities():
    assert grade_math(r"0*\prod_{i=1}^{4}i!", r"\boxed{0*99!}").score == 1
    result = grade_math(r"\prod_{i=1}^{4}i!", r"\boxed{(-2^2)!}")
    assert not result.settled


def test_floor_modulo_and_explicit_scalar_definitions_keep_exact_arithmetic():
    result = grade_math(
        r"4*\prod_{i=1}^{4}i!",
        r"Let \(F=123456\), \(Q=10\), \(E=2\). "
        r"\boxed{(\prod_{i=1}^{4}i!)(\lfloor\frac{F}{Q^E}\rfloor\bmod Q)}",
    )
    assert result.settled and result.score == 1


def test_symbolic_worker_timeout_remains_a_saved_unresolved_comparison(monkeypatch):
    from yugos_evals.scorers import math
    from yugos_evals.scorers._math_process import MathWorkerFailed

    def timeout(*args, **kwargs):
        raise MathWorkerFailed("timeout")

    monkeypatch.setattr(math, "compare", timeout)
    result = math.grade_math(r"\prod_{i=1}^{8}(2^i)!", r"\boxed{1}")
    assert not result.settled and result.status == "unavailable"
    assert result.details["cause"] == "symbolic_comparison_unresolved"


def test_every_settled_small_product_agrees_with_exact_integer_arithmetic():
    from math import factorial, prod

    for base in (2, 3, 4):
        reference = rf"\prod_{{i=1}}^{{3}}({base}^i)!"
        expected = prod(factorial(base**i) for i in range(1, 4))
        for other in (2, 3, 4):
            for lower, upper in ((0, 3), (1, 2), (1, 3), (1, 4)):
                answer = rf"\prod_{{j={lower}}}^{{{upper}}}({other}^j)!"
                outcome = grade_math(reference, rf"\boxed{{{answer}}}")
                actual = prod(factorial(other**j) for j in range(lower, upper + 1))
                if outcome.settled:
                    assert outcome.score == int(expected == actual)

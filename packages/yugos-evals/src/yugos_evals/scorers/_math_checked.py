"""Whole-answer parsing and equivalence for versioned math grading."""

from __future__ import annotations

import re

from latex2sympy2_extended import NormalizationConfig, latex2sympy
from latex2sympy2_extended.latex2sympy2 import ConversionConfig
from sympy import Eq, FiniteSet, Float, MatrixBase, Rational, Symbol, Tuple, simplify
from sympy.core.function import AppliedUndef

from yugos.evaluation.grades import GradeCode

from ._outcome import Outcome

_CONFIG = NormalizationConfig(units=False, malformed_operators=True, nits=True, boxed="none")
_CONVERSION = ConversionConfig(lowercase_symbols=False)


class Unsupported(ValueError):
    """The answer cannot be interpreted without discarding meaningful content."""


def _unwrap(value: str) -> str:
    from ._math_compare import _unwrap_answer

    return _unwrap_answer(value)


def _parse(value: str):
    value = _unwrap(value)
    # Use explicit multiplication between a numeric coefficient and a binomial.
    value = re.sub(r"(\d)\s*(\\(?:d|t)?binom)", r"\1\\cdot\2", value)
    literal = re.fullmatch(r"\\(?:text|textrm|mbox)\{([^{}]+)\}", value)
    if literal:
        if not re.fullmatch(r"[A-Za-z]+", literal[1].strip()):
            raise Unsupported("prose_requires_structured_reference")
        return literal[1].strip().casefold()
    if value.casefold() in {"yes", "no", "finite", "infinite"}:
        return value.casefold()
    if re.search(
        r"\\(?:text|textrm|mbox)|\b(?:there|where|for|all|when|otherwise|and|or)\b", value, re.I
    ):
        raise Unsupported("prose_requires_structured_reference")
    if re.search(r"\\log(?![a-zA-Z])(?!\s*_)", value):
        raise Unsupported("ambiguous_log_base")
    if not value or r"\boxed" in value:
        raise Unsupported("unparseable")
    try:
        parsed = latex2sympy(value, normalization_config=_CONFIG, conversion_config=_CONVERSION)
    except Exception as exc:
        raise Unsupported("unparseable") from exc
    return parsed


def _definition(value):
    if isinstance(value, Eq) and isinstance(value.lhs, AppliedUndef):
        args = value.lhs.args
        if len(args) == 1 and isinstance(args[0], Symbol):
            return value.lhs.func, args[0], value.rhs
    return None


def _admit(value):
    definition = _definition(value)
    if definition and definition[2].free_symbols - {definition[1]}:
        raise Unsupported("parameter_family_requires_structured_reference")


def _equal(gold, answer) -> bool:
    if isinstance(gold, str) or isinstance(answer, str):
        return isinstance(gold, str) and isinstance(answer, str) and gold == answer
    if gold == answer:
        return True
    gdef, adef = _definition(gold), _definition(answer)
    if gdef and adef:
        return gdef[0] == adef[0] and _equal(gdef[2], adef[2].subs(adef[1], gdef[1]))
    if isinstance(gold, Eq) and not isinstance(answer, Eq):
        if (
            (gdef and answer.free_symbols <= {gdef[1]})
            or (gold.lhs.is_number and _equal(gold.lhs, gold.rhs))
            or (isinstance(gold.lhs, Symbol) and gold.lhs not in gold.rhs.free_symbols)
        ):
            return _equal(gold.rhs, answer)
    if isinstance(answer, Eq) and not isinstance(gold, Eq):
        if isinstance(answer.lhs, Symbol) and answer.lhs not in answer.rhs.free_symbols:
            return _equal(gold, answer.rhs)
    if isinstance(gold, Eq) or isinstance(answer, Eq):
        return (
            isinstance(gold, Eq)
            and isinstance(answer, Eq)
            and (_equal(gold.lhs, answer.lhs) and _equal(gold.rhs, answer.rhs))
        )
    if isinstance(gold, MatrixBase) and gold.cols == 1 and isinstance(answer, Tuple):
        gold = Tuple(*gold)
    elif isinstance(answer, MatrixBase) and answer.cols == 1 and isinstance(gold, Tuple):
        answer = Tuple(*answer)
    if isinstance(gold, MatrixBase) or isinstance(answer, MatrixBase):
        return (
            isinstance(gold, MatrixBase)
            and isinstance(answer, MatrixBase)
            and (gold.shape == answer.shape and all(_equal(g, a) for g, a in zip(gold, answer)))
        )
    if isinstance(gold, Tuple) or isinstance(answer, Tuple):
        return (
            isinstance(gold, Tuple)
            and isinstance(answer, Tuple)
            and (len(gold) == len(answer) and all(_equal(g, a) for g, a in zip(gold, answer)))
        )
    if isinstance(gold, FiniteSet) or isinstance(answer, FiniteSet):
        if not isinstance(gold, FiniteSet) or not isinstance(answer, FiniteSet):
            return False
        remaining = list(answer)
        for g in gold:
            match = next((i for i, a in enumerate(remaining) if _equal(g, a)), None)
            if match is None:
                return False
            remaining.pop(match)
        return not remaining
    # Relational predicates and intervals must describe the same complete set.
    if (
        gold.is_Relational
        or answer.is_Relational
        or gold.func.__name__ == "And"
        or answer.func.__name__ == "And"
    ):
        try:
            return gold.as_set() == answer.as_set()
        except (NotImplementedError, AttributeError):
            return False
    try:

        def exact(value):
            return value.replace(lambda x: isinstance(x, Float), lambda x: Rational(str(x)))

        difference = simplify(exact(gold) - exact(answer))
        if difference.is_zero is True:
            return True
        # Retain six-decimal approximations for explicit decimal answers. A
        # relative bound keeps tiny nonzero values distinct.
        if gold.is_number and answer.is_number:
            if isinstance(gold, Float) or isinstance(answer, Float):
                tolerance = min(Rational(1, 2_000_000), abs(exact(gold)) / 1_000_000)
                return bool(abs(difference) <= tolerance)
    except (TypeError, AttributeError):
        pass
    return False


def _final(text: str) -> str:
    boxes = []
    position = 0
    while (start := text.find(r"\boxed{", position)) >= 0:
        depth, end = 1, start + len(r"\boxed{")
        while end < len(text) and depth:
            depth += int(text[end] == "{") - int(text[end] == "}")
            end += 1
        boxes.append((start, end, depth))
        position = end
    lines = list(re.finditer(r"(?im)^\s*(?:final\s+)?answer\s*[:=]\s*(.+?)\s*$", text))
    if lines and (not boxes or lines[-1].start() > boxes[-1][0]):
        return lines[-1][1]
    if not boxes:
        return text.strip()
    start, end, depth = boxes[-1]
    if depth:
        raise Unsupported("incomplete_final_box")
    return text[start + len(r"\boxed{") : end - 1]


def reference_exclusion(answer: str) -> str | None:
    """Require a complete parse, rather than a self-matching extracted fragment."""
    try:
        from ._math_compare import _top_level_comma

        if _top_level_comma(_unwrap(answer)):
            return "unstructured_multi_part_reference"
        _admit(_parse(answer))
    except Unsupported as exc:
        return str(exc)
    return None


def grade_math(gold: str, text: str, *, unordered=False) -> Outcome:
    """Compare the complete final field without numerical chopping."""
    try:
        reference = _parse(gold)
        _admit(reference)
    except Unsupported as exc:
        return Outcome(GradeCode.INVALID_GOLD, status="unavailable", details={"cause": str(exc)})
    try:
        final = _final(text)
        if isinstance(reference, str):
            raw = _unwrap(final)
            literal = re.fullmatch(r"\\(?:text|textrm|mbox)\{([^{}]+)\}", raw)
            answer = (literal[1] if literal else raw).strip().casefold()
        else:
            answer = _parse(final)
    except Unsupported as exc:
        return Outcome(GradeCode.INVALID_ANSWER, details={"cause": str(exc)})
    from ._math_compare import _integer_counterexample

    if _integer_counterexample(reference, answer):
        return Outcome(GradeCode.INCORRECT)
    if unordered and isinstance(reference, MatrixBase) and isinstance(answer, MatrixBase):
        remaining = list(answer)
        matched = reference.shape == answer.shape
        for value in reference:
            index = next((i for i, item in enumerate(remaining) if _equal(value, item)), None)
            if index is None:
                matched = False
                break
            remaining.pop(index)
    else:
        matched = _equal(reference, answer)
    return Outcome(GradeCode.CORRECT, score=1) if matched else Outcome(GradeCode.INCORRECT)

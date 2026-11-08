"""Math answer comparison and reference screening."""

from __future__ import annotations

import re

from latex2sympy2_extended import NormalizationConfig, latex2sympy
from latex2sympy2_extended.latex2sympy2 import ConversionConfig
from sympy import Dummy, Eq, FiniteSet, Float, MatrixBase, Rational, Symbol, Tuple, simplify
from sympy.core.function import AppliedUndef

from yugos.evaluation.grades import GradeCode

from ._outcome import Outcome

MAX_MATH_TEXT_CHARS = 65_536
_PLAIN_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_CONFIG = NormalizationConfig(units=False, malformed_operators=True, nits=True, boxed="none")
_CONVERSION = ConversionConfig(lowercase_symbols=False)


class Unsupported(ValueError):
    """The answer cannot be interpreted without discarding meaningful content."""


def _integer_counterexample(gold, response) -> bool:
    """Refute a formal identity before SymPy expands a huge finite sum.

    Undefined functions have no supplied definition in the comparison.
    Any identity with a numeric integer must hold when those functions are zero.
    An exact, different integer under that interpretation proves a mismatch;
    matching or unresolved interpretations prove nothing and use the usual comparison.
    """
    from sympy import Integer, Sum
    from sympy.core.function import AppliedUndef

    if not isinstance(gold, Integer) or not hasattr(response, "atoms"):
        return False
    sums, functions = response.atoms(Sum), response.atoms(AppliedUndef)
    zero = Integer(0)
    if (
        not sums
        or not functions
        or any(
            getattr(zero, "is_" + name, None) is not value
            for function in functions
            for name, value in getattr(function.func, "_kwargs", {}).items()
        )
    ):
        return False
    if any(
        len(limit) != 3 or not isinstance(limit[1], Integer) or not isinstance(limit[2], Integer)
        for total in sums
        for limit in total.limits
    ):
        return False
    try:
        witness = response.replace(
            lambda node: isinstance(node, AppliedUndef), lambda node: zero
        ).doit()
        return isinstance(witness, Integer) and witness != gold
    except Exception:  # The ordinary verifier still owns every unresolved case.
        return False


def _unwrap_answer(value: str) -> str:
    value = value.strip()
    wrappers = (
        ("$$", "$$"),
        ("$", "$"),
        (r"\(", r"\)"),
        (r"\[", r"\]"),
        (r"\boxed{", "}"),
    )
    while True:
        for left, right in wrappers:
            if value.startswith(left) and value.endswith(right):
                value = value[len(left) : -len(right)].strip()
                break
        else:
            return value


def _balanced_answer(value: str) -> bool:
    depth = 0
    for char in value:
        depth += int(char == "{") - int(char == "}")
        if depth < 0:
            return False
    return depth == 0 and r"\boxed" not in value


def _question_contract(question: str) -> tuple[str, str]:
    """Recognize explicit unit requests and comma-separated roots, without item IDs."""
    ordered = re.search(
        r"\b(?:(?:ascending|descending|increasing|decreasing|given|that)\s+order|ordered)\b",
        question,
        re.I,
    )
    roots = re.search(r"\b(?:roots|solutions)\b[^.?!\n]*\bseparated by commas\b", question, re.I)
    order = "ordered" if ordered else "unordered" if roots else "auto"
    explicit_units = re.search(
        r"\bexpress(?:ed|\s+(?:your|the)\s+(?:final\s+)?[a-z]+)?\s+in\s+"
        r"([a-z]+(?:\s+per\s+[a-z]+)?)\s*[?.]",
        question,
        re.I,
    )
    count = re.search(
        r"\bhow many\s+([a-z]{2,}s)\s+(?:must|does|did|will|are|were|do|can|should|would|could)\b",
        question,
        re.I,
    )
    if explicit_units:
        return explicit_units[1].lower(), order
    if count and count[1].lower() != "less":
        return count[1].lower(), order
    if re.search(r"\b(?:as a percentage|in percent|what percent(?:age)?)\b", question, re.I):
        return "%", order
    return "", order


def _strip_answer_units(value: str, units: str) -> str:
    """Normalize each comma-separated value, keeping unrecognized units intact."""
    suffix = r"(?:\\text\{\s*" + re.escape(units) + r"\s*\}|" + re.escape(units) + r")\s*$"
    if units == "degrees":
        suffix = r"(?:\^\{?\\circ\}?|" + suffix + r")\s*$"
    elif units in {"%", "percent", "percentage"}:
        suffix = r"(?:\\?%|\\text\{\s*percent(?:age)?\s*\}|percent(?:age)?)\s*$"
    return ",".join(
        re.sub(suffix, "", part.strip(), flags=re.I).strip() for part in _comma_parts(value)
    )


def _line_answers(expected: str, response: str, units: str, answer_order: str):
    """Normalize the explicit final field for comparison."""
    from inspect_evals.math.utils import extract_answer

    expected, response = _unwrap_answer(expected), _unwrap_answer(extract_answer(response))
    if not response:
        return Outcome(GradeCode.INVALID_ANSWER)
    if not _balanced_answer(expected):
        return Outcome(GradeCode.INVALID_GOLD, status="unavailable")
    if not _balanced_answer(response):
        return Outcome(GradeCode.INVALID_ANSWER)
    prose = re.fullmatch(r"\\(?:text|textrm|mbox)\{([^{}]+)\}", expected)
    if prose:
        actual = re.sub(r"^\\(?:text|textrm|mbox)\{([^{}]+)\}$", r"\1", response)
        values = [re.sub(r"\s*,\s*", ",", value.strip()).casefold() for value in (prose[1], actual)]
        return (
            Outcome(GradeCode.CORRECT, score=1)
            if values[0] == values[1]
            else Outcome(GradeCode.INCORRECT)
        )
    if re.search(r"\b(?:or|and)\b", response):
        return Outcome(GradeCode.INVALID_ANSWER, details={"cause": "ambiguous_answer"})
    if not units:
        reference_units = set()
        for part in _comma_parts(expected):
            suffix = re.fullmatch(r"[-+]?\d+(?:\.\d+)?\\text\{\s*([a-z ]+)\}", part.strip())
            if suffix:
                reference_units.add(suffix[1].strip())
            elif re.search(r"\^\{?\\circ\}?$", part.strip()):
                reference_units.add("degrees")
        if len(reference_units) == 1:
            units = reference_units.pop()
    if units:
        expected, response = (_strip_answer_units(value, units) for value in (expected, response))
    expected, response = (
        re.sub(r"\\?%", r"\\cdot\\frac{1}{100}", value) for value in (expected, response)
    )
    expected, response = (value.replace(r"\!", "") for value in (expected, response))
    if answer_order == "auto" and all(
        _PLAIN_NUMBER.fullmatch(value) for value in (expected, response)
    ):
        expected, response = expected.replace(",", ""), response.replace(",", "")
    if any(re.fullmatch(r"\d+_\{?\d+\}?", value) for value in (expected, response)):
        values = [value.replace("{", "").replace("}", "") for value in (expected, response)]
        return (
            Outcome(GradeCode.CORRECT, score=1)
            if values[0] == values[1]
            else Outcome(GradeCode.INCORRECT)
        )
    if _top_level_comma(expected) or _top_level_comma(response):
        expected, response = (
            r"\begin{pmatrix}" + r"\\".join(_comma_parts(value)) + r"\end{pmatrix}"
            if _top_level_comma(value)
            else value
            for value in (expected, response)
        )
    return expected, rf"\boxed{{{response}}}"


def _parse(value: str):
    value = _unwrap_answer(value)
    value = re.sub(
        r"(?<=\S)\s*\\text\{\s*(?:degrees|(?:(?:square|cubic)\s+)?units)\s*\}$",
        "",
        value,
        flags=re.I,
    )
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


def _unordered_equal(gold, answer) -> bool:
    """Match each value once, including overlapping decimal approximations."""
    gold, answer = tuple(gold), tuple(answer)
    if len(gold) != len(answer):
        return False
    matched = {}

    def assign(index, seen):
        for candidate, value in enumerate(answer):
            if candidate in seen or not _equal(gold[index], value):
                continue
            seen.add(candidate)
            if candidate not in matched or assign(matched[candidate], seen):
                matched[candidate] = index
                return True
        return False

    return all(assign(index, set()) for index in range(len(gold)))


def _equal(gold, answer) -> bool:
    if isinstance(gold, str) or isinstance(answer, str):
        return isinstance(gold, str) and isinstance(answer, str) and gold == answer
    if gold == answer:
        return True
    gdef, adef = _definition(gold), _definition(answer)
    if gdef and adef:
        argument = Dummy()
        return gdef[0] == adef[0] and _equal(
            gdef[2].subs(gdef[1], argument), adef[2].subs(adef[1], argument)
        )
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
        return (
            isinstance(gold, FiniteSet)
            and isinstance(answer, FiniteSet)
            and _unordered_equal(gold, answer)
        )
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


def grade_math(
    gold: str,
    text: str,
    *,
    answer_format: str = "auto",
    question: str = "",
) -> Outcome:
    """Compare a response's final answer with the reference; parser failures never escape."""
    expected = (gold or "").strip()
    response = (text or "").strip()
    if not expected:
        return Outcome(GradeCode.INVALID_GOLD, status="unavailable")
    if not response:
        return Outcome(GradeCode.NOT_ATTEMPTED)
    from ._symbolic_products import compare_products

    context = response
    answer_order = "auto"
    if answer_format == "answer_line":
        if max(len(expected), len(response), len(question)) > MAX_MATH_TEXT_CHARS:
            return Outcome(GradeCode.INVALID_ANSWER, details={"cause": "input_too_long"})
        units, answer_order = _question_contract(question)
        prepared = _line_answers(expected, response, units, answer_order)
        if isinstance(prepared, Outcome):
            return prepared
        expected, response = prepared
    if len(expected) <= MAX_MATH_TEXT_CHARS:
        symbolic = compare_products(expected, response, context=context)
        if symbolic is not None:
            return symbolic
    if max(len(expected), len(response), len(question)) > MAX_MATH_TEXT_CHARS:
        return Outcome(GradeCode.INVALID_ANSWER, details={"cause": "input_too_long"})
    return _compare_answer(expected, response, unordered=answer_order == "unordered")


def _compare_answer(gold: str, text: str, *, unordered=False) -> Outcome:
    """Compare the complete final answer with its reference."""
    try:
        reference = _parse(gold)
        _admit(reference)
    except Unsupported as exc:
        return Outcome(GradeCode.INVALID_GOLD, status="unavailable", details={"cause": str(exc)})
    try:
        final = _final(text)
        if isinstance(reference, str):
            raw = _unwrap_answer(final)
            literal = re.fullmatch(r"\\(?:text|textrm|mbox)\{([^{}]+)\}", raw)
            answer = (literal[1] if literal else raw).strip().casefold()
        else:
            answer = _parse(final)
    except Unsupported as exc:
        return Outcome(GradeCode.INVALID_ANSWER, details={"cause": str(exc)})
    if _integer_counterexample(reference, answer):
        return Outcome(GradeCode.INCORRECT)
    if unordered and isinstance(reference, MatrixBase) and isinstance(answer, MatrixBase):
        matched = reference.shape == answer.shape and _unordered_equal(reference, answer)
    else:
        matched = _equal(reference, answer)
    return Outcome(GradeCode.CORRECT, score=1) if matched else Outcome(GradeCode.INCORRECT)


def _top_level_comma(answer: str) -> bool:
    """Detect answer lists that are not a tuple, set or interval."""
    return len(_comma_parts(answer)) > 1


def _comma_parts(answer: str) -> list[str]:
    depth = 0
    start, parts = 0, []
    for index, char in enumerate(answer):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth <= 0 and (index == 0 or answer[index - 1] != "\\"):
            parts.append(answer[start:index])
            start = index + 1
    return [*parts, answer[start:]]


def reference_exclusion(answer: str) -> str | None:
    """Report a reference format the comparison cannot interpret."""
    from ._symbolic_products import compare_products

    if _top_level_comma(_unwrap_answer(answer)):
        return "unstructured_multi_part_reference"
    symbolic = compare_products(answer, rf"\boxed{{{answer}}}")
    if symbolic is not None:
        return None if symbolic.code == GradeCode.CORRECT else "unsupported_symbolic_product"
    try:
        _admit(_parse(answer))
    except Unsupported as exc:
        return str(exc)
    return None

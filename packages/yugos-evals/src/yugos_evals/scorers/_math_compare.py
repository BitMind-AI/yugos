"""Math-Verify answer grading, and the reference checks sources apply before packing.

The default auto format preserves Yugos's frozen legacy math rules: the reference is boxed and
parsed with Math-Verify, the response's boxed (else last) expression is parsed
with shorthand normalization, and the two must verify strictly. References the
grader cannot check faithfully are excluded when a source is loaded
(`reference_exclusion`), not graded; no LLM-judge fallback is used.
"""

from __future__ import annotations

import re

from math_verify import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    LatexNormalizationConfig,
    parse,
    verify,
)

from yugos.evaluation.grades import GradeCode

from ._outcome import Outcome

MAX_MATH_TEXT_CHARS = 65_536

_PREDICTION_EXTRACTION = (
    LatexExtractionConfig(
        boxed_match_priority=0,
        normalization_config=LatexNormalizationConfig(
            basic_latex=True,
            units=True,
            # Gold uses shorthand such as `\frac43` and `\sqrt2`; accept the same
            # notation from a response.
            malformed_operators=True,
            nits=False,
            boxed="all",
            equations=False,
        ),
    ),
    ExprExtractionConfig(),
)
_GOLD_EXTRACTION = (
    LatexExtractionConfig(
        boxed_match_priority=0,
        normalization_config=LatexNormalizationConfig(
            basic_latex=True,
            units=True,
            malformed_operators=True,
            nits=True,
            boxed="all",
            equations=False,
        ),
    ),
    ExprExtractionConfig(),
)
# The gold parse without unit normalization or a bare-number fallback.
_STRICT_GOLD_EXTRACTION = (
    LatexExtractionConfig(
        boxed_match_priority=0,
        normalization_config=LatexNormalizationConfig(
            basic_latex=True,
            units=False,
            malformed_operators=True,
            nits=True,
            boxed="all",
            equations=False,
        ),
    ),
)
_PLAIN_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_EXPLICIT_LINE = re.compile(r"(?im)^\s*(?:final\s+)?answer\s*[:=]\s*(.+?)\s*$")
# Math-Verify reads a bare \log as base 10; sources such as ArXivMath mean natural
# logarithms. A subscript, even after a space (\log _{2}), names the base.
_BARE_LOG = re.compile(r"\\log(?![a-zA-Z])(?!\s*_)")
# Words in a text box are prose; operator names such as \operatorname{lcm} are math.
_TEXT = re.compile(r"\\(?:text|textrm|textbf|textit|textsf|texttt|mbox)\s*\{[^{}]*[a-zA-Z]")


def _ambiguous_numeric_answer(gold: str, text: str) -> bool:
    if not _PLAIN_NUMBER.fullmatch(gold):
        return False
    lines = _EXPLICIT_LINE.findall(text)
    return bool(lines and len(_PLAIN_NUMBER.findall(lines[-1])) != 1)


def _integer_counterexample(gold, response) -> bool:
    """Refute a formal identity before SymPy expands a huge finite sum.

    Undefined functions have no supplied definition in the frozen comparison.
    Any identity with a numeric integer must hold when those functions are zero.
    An exact, different integer under that interpretation proves a mismatch;
    matching or unresolved interpretations prove nothing and use normal verify.
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


def _line_answers(expected: str, response: str, units: str, answer_order: str):
    """Prepare the explicit final field; Math-Verify still owns expression normalization."""
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
        suffix = re.search(r"^([-+]?\d+(?:\.\d+)?)\\text\{\s*([a-z ]+)\}$", expected)
        if suffix:
            units = suffix[2].strip()
        elif re.search(r"\^\{?\\circ\}?$", expected):
            units = "degrees"
    if units:
        suffix = r"(?:\\text\{\s*" + re.escape(units) + r"\s*\}|" + re.escape(units) + r")\s*$"
        if units == "degrees":
            suffix = r"(?:\^\{?\\circ\}?|" + suffix + r")\s*$"
        elif units == "%":
            suffix = r"\\?%\s*$"
        expected, response = (
            re.sub(suffix, "", value, flags=re.I).strip() for value in (expected, response)
        )
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


def _same_math_values(gold, answer, unordered: bool) -> bool:
    """Compare coordinates structurally, and unordered roots without losing multiplicity."""
    from sympy import MatrixBase, Tuple

    if unordered:
        gold = Tuple(*gold) if isinstance(gold, MatrixBase) and gold.cols == 1 else gold
        answer = Tuple(*answer) if isinstance(answer, MatrixBase) and answer.cols == 1 else answer
    if isinstance(gold, MatrixBase) and gold.cols == 1 and isinstance(answer, Tuple):
        gold = Tuple(*gold)
    elif isinstance(answer, MatrixBase) and answer.cols == 1 and isinstance(gold, Tuple):
        answer = Tuple(*answer)
    if isinstance(gold, Tuple) != isinstance(answer, Tuple):
        return False
    if unordered and isinstance(gold, Tuple) and isinstance(answer, Tuple):
        if len(gold) != len(answer):
            return False
        matched = {}

        def assign(index, seen):
            for candidate in range(len(answer)):
                if candidate in seen or not verify(
                    [gold[index]], [answer[candidate]], strict=True, timeout_seconds=0
                ):
                    continue
                seen.add(candidate)
                if candidate not in matched or assign(matched[candidate], seen):
                    matched[candidate] = index
                    return True
            return False

        return all(assign(index, set()) for index in range(len(gold)))
    return bool(verify([gold], [answer], strict=True, timeout_seconds=0))


def grade_math(
    gold: str,
    text: str,
    *,
    answer_format: str = "auto",
    units: str = "",
    answer_order: str = "auto",
) -> Outcome:
    """Compare a response's final answer with the reference; parser failures never escape."""
    expected = (gold or "").strip()
    response = (text or "").strip()
    if not expected:
        return Outcome(GradeCode.INVALID_GOLD, status="unavailable")
    if not response:
        return Outcome(GradeCode.NOT_ATTEMPTED)
    if answer_format == "auto" and _ambiguous_numeric_answer(expected, response):
        return Outcome(GradeCode.INVALID_ANSWER, details={"cause": "ambiguous_answer"})
    if len(expected) > MAX_MATH_TEXT_CHARS or len(response) > MAX_MATH_TEXT_CHARS:
        return Outcome(GradeCode.INVALID_ANSWER, details={"cause": "input_too_long"})
    if answer_format == "answer_line":
        prepared = _line_answers(expected, response, units, answer_order)
        if isinstance(prepared, Outcome):
            return prepared
        expected, response = prepared
    try:
        parsed_gold = parse(
            f"\\boxed{{{expected}}}",
            extraction_config=_STRICT_GOLD_EXTRACTION
            if answer_format == "answer_line"
            else _GOLD_EXTRACTION,
            fallback_mode="no_fallback",
            parsing_timeout=0,
        )
        if not parsed_gold:
            return Outcome(
                GradeCode.INVALID_GOLD, status="unavailable", details={"cause": "unparseable"}
            )
        parsed_response = parse(
            response,
            extraction_config=_STRICT_GOLD_EXTRACTION
            if answer_format == "answer_line"
            else _PREDICTION_EXTRACTION,
            fallback_mode="no_fallback",
            parsing_timeout=0,
        )
        if not parsed_response:
            return Outcome(GradeCode.INVALID_ANSWER, details={"cause": "unparseable"})
        if len(parsed_gold) == len(parsed_response) == 1 and _integer_counterexample(
            parsed_gold[0], parsed_response[0]
        ):
            return Outcome(GradeCode.INCORRECT)
        if answer_format == "answer_line":
            matched = len(parsed_gold) == len(parsed_response) == 1 and _same_math_values(
                parsed_gold[0], parsed_response[0], answer_order == "unordered"
            )
        else:
            matched = bool(verify(parsed_gold, parsed_response, strict=True, timeout_seconds=0))
    except Exception as exc:  # noqa: BLE001 - one response must not abort a scoring run.
        return Outcome(
            GradeCode.GRADER_ERROR,
            status="error",
            details={"component": "parser", "error_type": type(exc).__name__},
        )
    if matched:
        return Outcome(GradeCode.CORRECT, score=1.0)
    return Outcome(GradeCode.INCORRECT)


def _top_level_comma(answer: str) -> bool:
    """Detect answer lists that are not a tuple, set or interval Math-Verify can compare."""
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
    """Why `grade_math` cannot faithfully check this reference, or None.

    Prose answers, comma-separated parts outside a tuple, set or interval, a bare
    `\\log`, a reference that does not verify against itself, and one whose parse
    silently drops part of the expression are excluded.
    """
    if _TEXT.search(answer):
        return "textual_reference: math-verify cannot compare prose answers"
    if _top_level_comma(answer):
        return (
            "unstructured_multi_part_reference: comma-separated parts outside a tuple, "
            "set or interval"
        )
    if _BARE_LOG.search(answer):
        return (
            "ambiguous_log_base: math-verify reads bare \\log as base 10, so an equivalent "
            "\\ln answer grades incorrect"
        )
    probe = grade_math(answer, f"\\boxed{{{answer}}}")
    if probe.code == GradeCode.INVALID_GOLD:
        return "unparseable_reference: invalid expected answer"
    if probe.code != GradeCode.CORRECT:
        return f"unparseable_reference: math reference could not be verified: {probe.code.value}"
    # The grader normalizes units and may fall back to a bare number; an
    # independent LaTeX-only parse exposes references it silently truncates.
    boxed = f"\\boxed{{{answer}}}"
    try:
        graded = parse(
            boxed,
            extraction_config=_GOLD_EXTRACTION,
            fallback_mode="no_fallback",
            parsing_timeout=0,
        )
        complete = parse(
            boxed,
            extraction_config=_STRICT_GOLD_EXTRACTION,
            fallback_mode="no_fallback",
            parsing_timeout=0,
        )
        whole = bool(complete) and (
            verify(graded, complete, strict=True, timeout_seconds=0)
            and verify(complete, graded, strict=True, timeout_seconds=0)
        )
    except Exception as exc:  # noqa: BLE001 - one reference must not abort a dataset load.
        return f"unparseable_reference: completeness probe raised {type(exc).__name__}"
    if not whole:
        return "partial_reference_parse: the grader's parse drops part of the reference"
    return None

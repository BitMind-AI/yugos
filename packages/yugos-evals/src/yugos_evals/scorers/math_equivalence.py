"""Deterministic MATH grading with presentation and expression equivalence."""

import re
from collections import Counter
from functools import partial

import anyio
from inspect_ai.scorer import Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_evals.math.utils import extract_answer
from math_verify import parse, verify

from yugos.evaluation.grades import GradeCode

from ._math_compare import _STRICT_GOLD_EXTRACTION, MAX_MATH_TEXT_CHARS
from ._math_process import MathWorkerFailed, compare
from ._outcome import Outcome, completion_gate, to_score


def _normalize(text: str, units: str, grouped: bool) -> str:
    text = text.strip()
    for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if text.startswith(left) and text.endswith(right):
            text = text[len(left) : -len(right)].strip()
    if text.startswith(r"\boxed{") and text.endswith("}"):
        text = text[7:-1]
    text = re.sub(r"\\(?:text|textrm|mbox)\{([^{}]*)\}", r"\1", text)
    text = text.replace(r"^{\circ}", " degrees").replace(r"^\circ", " degrees")
    if units == "%":
        text = re.sub(r"\\?%$", "", text.rstrip())
    elif units and text.casefold().rstrip().endswith(units):
        text = text.rstrip()[: -len(units)]
    text = re.sub(r"\\(?:left|right|[!,;])", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"\\?%", r"\\cdot\\frac{1}{100}", text)
    if grouped and re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
        text = text.replace(",", "")
    return text


def _units(reference: str, problem: str) -> str:
    if re.search(
        r"\bwhat percent(?:age)?\b|\bexpressed (?:as|in) (?:a )?percent(?:age)?\b",
        problem.casefold(),
    ):
        return "%"
    explicit = re.search(r".+\\text\{([a-z ]+)\}$", reference)
    if explicit:
        return explicit[1].strip().casefold()
    if re.search(r"\^\{?\\circ\}?$", reference):
        return "degrees"
    requested = re.search(r"\bexpressed in ([a-z ]+)[?.]?$", problem.casefold())
    count = re.search(r"\bhow many ([a-z]+)\b", problem.casefold())
    return requested[1].strip() if requested else count[1] if count else ""


def _parts(text: str) -> list[str]:
    depth, start, parts = 0, 0, []
    for i, char in enumerate(text):
        depth += int(char in "({[") - int(char in ")}]")
        if char == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    return [*parts, text[start:]]


def _equivalent(reference: str, answer: str, problem: str) -> bool:
    units = _units(reference, problem)
    grouped = r",\!" in reference
    gold, prediction = (_normalize(value, units, grouped) for value in (reference, answer))
    if not gold or not prediction:
        return False
    for value in (gold, prediction):
        depth = 0
        for char in value:
            depth += int(char == "{") - int(char == "}")
            if depth < 0:
                return False
        if depth or r"\boxed" in value:
            return False
    if "point" in problem.casefold():

        def point(value):
            matrix = re.fullmatch(r"\\begin\{pmatrix\}(.+)\\end\{pmatrix\}", value)
            if matrix and "&" not in matrix[1]:
                return "(" + matrix[1].replace("\\\\", ",") + ")"
            return value

        gold, prediction = point(gold), point(prediction)
    if gold == prediction:
        return True
    if any(re.fullmatch(r"\d+_\{?\d+\}?", value) for value in (gold, prediction)):
        return gold.replace("{", "").replace("}", "") == prediction.replace("{", "").replace(
            "}", ""
        )
    gold_parts, answer_parts = _parts(gold), _parts(prediction)
    if len(gold_parts) > 1 or len(answer_parts) > 1:
        if (
            "roots" in problem.casefold()
            and "comma" in problem.casefold()
            and not re.search(
                r"\b(order|ascending|descending|increasing|decreasing|smallest|largest)\b",
                problem.casefold(),
            )
        ):
            return Counter(gold_parts) == Counter(answer_parts)
        return len(gold_parts) == len(answer_parts) and all(
            _equivalent(g, a, problem) for g, a in zip(gold_parts, answer_parts)
        )
    # Do not turn prose into a commutative product of single-letter symbols.
    for value in (gold, prediction):
        words = re.sub(r"\\(?:begin|end|operatorname)\{[^{}]*\}|\\[a-zA-Z]+", "", value)
        if re.search(r"[a-zA-Z]{2,}", words):
            return gold.isalpha() and prediction.casefold() == gold.casefold()
    parsed = [
        parse(
            rf"\boxed{{{value}}}",
            extraction_config=_STRICT_GOLD_EXTRACTION,
            fallback_mode="no_fallback",
            parsing_timeout=2,
        )
        for value in (gold, prediction)
    ]
    if not parsed[0]:
        raise ValueError("unparseable reference")
    return bool(parsed[1]) and bool(verify(*parsed, strict=True, timeout_seconds=2))


@scorer(metrics=[accuracy(), stderr()])
def math_equivalence():
    """Grade the final ANSWER using bounded, deterministic equivalence checks."""

    async def score(state: TaskState, target: Target):
        answer = extract_answer(state.output.completion)
        outcome = completion_gate(state)
        if outcome is None:
            if not target.text.strip():
                outcome = Outcome(GradeCode.INVALID_GOLD, status="unavailable")
            elif max(len(answer), len(target.text)) > MAX_MATH_TEXT_CHARS:
                outcome = Outcome(GradeCode.INVALID_ANSWER)
            else:
                try:
                    correct = await anyio.to_thread.run_sync(
                        partial(
                            compare, "equivalence", target.text, answer, problem=state.input_text
                        )
                    )
                    outcome = (
                        Outcome(GradeCode.CORRECT, score=1)
                        if correct
                        else Outcome(GradeCode.INCORRECT)
                    )
                except MathWorkerFailed as exc:
                    outcome = Outcome(
                        GradeCode.GRADER_ERROR,
                        status="error",
                        details={"component": "math_worker", "cause": exc.cause},
                    )
                except Exception:
                    outcome = Outcome(GradeCode.GRADER_ERROR, status="error")
        return to_score(outcome, answer=answer)

    return score

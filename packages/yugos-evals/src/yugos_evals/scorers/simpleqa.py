"""Yugos SimpleQA tool rubric; its scores differ from the upstream tool scorer."""

import re
from decimal import Decimal
from typing import Literal

from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, accuracy, scorer, stderr
from inspect_evals.simpleqa.grader_templates import SIMPLEQA_TOOL_INSTRUCTIONS
from inspect_evals.utils.scorers import schema_tool_graded_scorer
from typing_extensions import TypedDict

GRADER_VERSION = 1


class _Judgment(TypedDict):
    grade: Literal["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
    extracted_answer: str
    explanation: str


def _number(text):
    text = text.strip().replace("**", "")
    if len(text) <= 100 and re.fullmatch(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text):
        return Decimal(text.replace(",", ""))
    return None


@scorer(metrics=[accuracy(), stderr()])
def simpleqa_numeric_scorer(grader_template: str):
    """Use a Yugos-specific tool rubric with structured answers and numeric checks."""
    native = schema_tool_graded_scorer(
        template=grader_template,
        response_schema=_Judgment,
        instructions=SIMPLEQA_TOOL_INSTRUCTIONS
        + "\nExtract the response's final answer verbatim into extracted_answer. "
        "For numeric answers extract only the number, preserving digits and sign. "
        "Include a short explanation applying the rubric's numeric precision rule.",
        grade_map={"CORRECT": CORRECT, "INCORRECT": INCORRECT, "NOT_ATTEMPTED": NOANSWER},
        model_role="grader",
    )

    async def score(state, target):
        result = await native(state, target)
        payload = (result.metadata or {}).get("grader_payload", {})
        expected = _number(target.text)
        answer = _number(payload.get("extracted_answer", ""))
        if result.value == CORRECT and expected is not None and expected.as_tuple().exponent == 0:
            numbers = re.findall(
                r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])",
                state.output.completion,
            )
            if answer is None or not any(_number(number) == answer for number in numbers):
                result.metadata = {**result.metadata, "numeric_check": "judge_fallback"}
                return result
            precision = Decimal(10) ** max(0, expected.normalize().as_tuple().exponent)
            result.metadata = {**result.metadata, "numeric_check": "checked"}
            if abs(expected - answer) > precision / 2:
                result.value = INCORRECT
                result.metadata["numeric_precision_override"] = True
                result.explanation = (
                    "The extracted number does not match the reference's precision."
                )
        return result

    return score

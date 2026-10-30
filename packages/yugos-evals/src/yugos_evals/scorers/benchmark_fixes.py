"""Small corrections to native MBPP execution and SimpleQA numeric judging."""

import re
from copy import copy
from decimal import Decimal
from typing import Literal

from inspect_ai.model import ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, Score, accuracy, scorer, stderr
from inspect_evals.mbpp.mbpp import find_code, verify
from inspect_evals.simpleqa.grader_templates import SIMPLEQA_TOOL_INSTRUCTIONS
from inspect_evals.utils.scorers import schema_tool_graded_scorer
from typing_extensions import TypedDict


@scorer(metrics=[accuracy(), stderr()])
def mbpp_with_imports():
    """Run native MBPP verification with its declared imports inside the sandbox."""
    native = verify()

    async def score(state, target):
        amended = copy(state)
        imports = "\n".join(state.metadata.get("test_imports", []))
        amended.output = ModelOutput.from_content(
            state.output.model, imports + "\n" + find_code(state.output.completion)
        )
        result = await native(amended, target)
        result.answer = state.output.completion
        return result

    return score


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
def simpleqa_checked(grader_template: str):
    """Check a structured native judgment against explicit integer precision."""
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
        # Bare integer references have unit precision; scaled/decimal answers retain the rubric.
        if result.value == CORRECT and expected is not None and expected.as_tuple().exponent == 0:
            numbers = re.findall(
                r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])",
                state.output.completion,
            )
            if answer is None or not any(_number(number) == answer for number in numbers):
                return Score.unscored(reason="grader_failed", metadata=result.metadata)
            if abs(expected - answer) >= Decimal("0.5"):
                result.value = INCORRECT
                result.metadata = {**result.metadata, "numeric_precision_override": True}
                result.explanation = (
                    "The extracted number does not match the reference's precision."
                )
        return result

    return score

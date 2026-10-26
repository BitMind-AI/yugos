"""CRUXEval-O expression equivalence through the isolated Python test supervisor."""

import ast

from inspect_ai.scorer import Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

from yugos.evaluation.grades import GradeCode

from ._outcome import Outcome, completion_gate, to_score
from .python_unittest import grade_unittest


def prediction_program(completion: str, code: str, given_input: str) -> str | None:
    """Parse an expression without evaluating it, retaining the official call exclusion."""
    expression = completion.strip()
    if f"f({given_input})" in expression:
        return None
    try:
        ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError):
        return None
    return code + "\n\ndef task_func():\n    return (\n" + expression + "\n    )\n"


@scorer(metrics=[mean(), stderr()])
def cruxeval_output_scorer() -> Scorer:
    async def score(state: TaskState, target: Target):
        gated = completion_gate(state)
        if gated is not None:
            return to_score(gated)
        metadata = state.metadata or {}
        if not all(isinstance(metadata.get(k), str) for k in ("code", "input")):
            raise ValueError("CRUXEval needs its prepared function and input")
        candidate = prediction_program(state.output.completion, metadata["code"], metadata["input"])
        if candidate is None:
            return to_score(
                Outcome(GradeCode.INCORRECT, score=0, details={"cause": "invalid_expression"})
            )
        # The trusted assertion receives only a primitive value from the fresh
        # candidate process. Candidate exit, callbacks and test edits are not verdicts.
        tests = (
            "import unittest\nclass TestCases(unittest.TestCase):\n"
            f"    def test_output(self):\n        self.assertEqual(({target.text}), task_func())\n"
        )
        return to_score(await grade_unittest(sandbox(), candidate, tests, timeout_s=8))

    return score

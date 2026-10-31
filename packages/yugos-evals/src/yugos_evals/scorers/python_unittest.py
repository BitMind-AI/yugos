"""Score BigCodeBench unittest tasks without using candidate process exit as a verdict."""

from __future__ import annotations

import ast
import json
from importlib.resources import files

from inspect_ai.scorer import Scorer, Target, mean, scorer, std
from inspect_ai.solver import TaskState
from inspect_ai.util import SandboxEnvironment, sandbox
from inspect_ai.util._sandbox.local import LocalSandboxEnvironment
from inspect_evals.bigcodebench.bigcodebench import find_code

from yugos.evaluation.grades import GradeCode

from ._outcome import Outcome, completion_gate, to_score

RUNNER = files(__package__).joinpath("python_unittest_runner.py").read_text(encoding="utf-8")
VERIFY_TIMEOUT = 180  # Each trusted test invokes the candidate in a separate process.
GRADER_VERSION = 2


def library_names(value) -> tuple[str, ...]:
    """Accept the native dataset's list or its Python-list string representation."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("BigCodeBench libraries must be a list of names") from exc
    if not isinstance(value, (tuple, list)) or not all(isinstance(v, str) for v in value):
        raise ValueError("BigCodeBench libraries must be a list of names")
    return tuple(value)


async def grade_unittest(
    env: SandboxEnvironment,
    candidate: str,
    tests: str,
    *,
    timeout_s: int = VERIFY_TIMEOUT,
    libs: tuple[str, ...] = (),
    reference: str = "",
    http_fixtures: dict | None = None,
) -> Outcome:
    try:
        env.as_type(LocalSandboxEnvironment)
    except TypeError:
        pass
    else:
        raise RuntimeError("candidate code requires an isolated sandbox")
    if not candidate.strip():
        return Outcome(GradeCode.INVALID_ANSWER, details={"cause": "no_code_in_response"})
    if not tests.strip():
        return Outcome(GradeCode.INVALID_GOLD, status="unavailable", details={"cause": "no_tests"})
    request = {
        "candidate": candidate,
        "tests": tests,
        "timeout_s": max(1, timeout_s - 5),
        "libs": libs,
        "reference": reference,
        "http_fixtures": http_fixtures or {},
        "worker_source": RUNNER,
    }
    try:
        result = await env.exec(
            ["python", "-I", "-c", RUNNER],
            input=json.dumps(request),
            timeout=timeout_s,
            timeout_retry=False,
        )
    except TimeoutError:
        return Outcome(GradeCode.INCORRECT, details={"cause": "timeout"})
    if not result.success:
        return Outcome(
            GradeCode.GRADER_UNAVAILABLE,
            status="unavailable",
            details={"cause": "supervisor_failed", "returncode": result.returncode},
        )
    try:
        report = json.loads(result.stdout)
        if not isinstance(report, dict) or type(report.get("passed")) is not bool:
            raise ValueError("invalid report")
        cause = report["cause"]
        if not isinstance(cause, str):
            raise ValueError("missing cause")
        if cause not in {
            "all_pass",
            "test_failure",
            "timeout",
            "unsupported_result",
            "missing_test_cases",
            "test_error",
        } or report["passed"] != (cause == "all_pass"):
            raise ValueError("contradictory report")
        if cause in {"all_pass", "test_failure"} and not (
            type(report.get("tests_run")) is int
            and type(report.get("tests_expected")) is int
            and 0 < report["tests_run"] == report["tests_expected"]
        ):
            raise ValueError("incomplete tests")
    except (ValueError, KeyError, TypeError):
        return Outcome(
            GradeCode.GRADER_UNAVAILABLE,
            status="unavailable",
            details={"cause": "invalid_supervisor_report"},
        )
    if cause == "unsupported_result":
        return Outcome(
            GradeCode.GRADER_UNAVAILABLE,
            status="unavailable",
            details={"cause": cause},
        )
    if cause in {"missing_test_cases", "test_error"}:
        return Outcome(
            GradeCode.INVALID_GOLD,
            status="unavailable",
            details={"cause": cause},
        )
    return Outcome(
        GradeCode.CORRECT if report["passed"] else GradeCode.INCORRECT,
        score=1.0 if report["passed"] else 0.0,
        details={
            "cause": cause,
            **{key: report[key] for key in ("tests_run", "tests_expected") if key in report},
        },
    )


async def preflight_unittest(
    env: SandboxEnvironment,
    reference: str,
    tests: str,
    **settings,
) -> dict:
    """Qualify unchanged tests against their reference and a failing control, without inference."""
    gold = await grade_unittest(env, reference, tests, reference=reference, **settings)
    wrong = await grade_unittest(
        env,
        "def task_func(*args, **kwargs): raise AssertionError('preflight wrong control')",
        tests,
        reference=reference,
        **settings,
    )
    return {
        "passed": (
            gold.code == GradeCode.CORRECT
            and wrong.code == GradeCode.INCORRECT
            and wrong.details.get("cause") == "test_failure"
        ),
        "reference": {"code": gold.code.value, "status": gold.status, **gold.details},
        "wrong_control": {"code": wrong.code.value, "status": wrong.status, **wrong.details},
    }


@scorer(metrics=[mean(), std()])
def bigcodebench_scorer() -> Scorer:
    async def score(state: TaskState, target: Target):
        gated = completion_gate(state)
        if gated is not None:
            return to_score(gated)
        candidate = find_code(state.output.completion)
        tests = state.metadata.get("test") if state.metadata else None
        if not isinstance(tests, str):
            return to_score(
                Outcome(GradeCode.INVALID_GOLD, status="unavailable", details={"cause": "no_tests"})
            )
        return to_score(
            await grade_unittest(
                sandbox(),
                candidate,
                tests,
                libs=library_names(state.metadata.get("libs", ())),
                reference=target.text,
                http_fixtures=state.metadata.get("http_fixtures"),
            )
        )

    return score

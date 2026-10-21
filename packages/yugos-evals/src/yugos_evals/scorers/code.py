"""LiveCodeBench-style stdin/stdout and functional tests, executed only in the sample's sandbox.

For each test case in order, the scorer runs `code_runner` in the sandbox with
the candidate code and that case's input. Expected outputs never enter the
sandbox; outputs are compared here, with Yugos's LiveCodeBench rules, and the
first failing case decides the grade. Nothing the candidate writes (stdout,
files, exit status) is read as a verdict. A trusted probe runs before any
candidate code; if the sandbox cannot run it, the sample errors as an
infrastructure failure rather than scoring zero. A `local` sandbox is refused.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from importlib.resources import files

from inspect_ai.scorer import Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import OutputLimitExceededError, SandboxEnvironment, sandbox
from inspect_ai.util._sandbox.local import LocalSandboxEnvironment

from yugos.evaluation.grades import GradeCode

from . import code_runner
from ._outcome import Outcome, completion_gate, to_score

CASE_TIMEOUT_S = 10
RUNNER = files(__package__).joinpath("code_runner.py").read_text(encoding="utf-8")
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)
# Failure causes a candidate's report may claim; anything else is a malformed report.
_CANDIDATE_CAUSES = frozenset({"exec_error", "result_not_json", "no_callable_defined"})
_REPORTED = re.compile(r"(exec_error|result_not_json):\s*([^:]+)(?::\s*(.*))?", re.S)


class SandboxUnavailable(RuntimeError):
    """The sample's sandbox could not run the trusted test runner."""


def extract_code(text: str) -> str:
    """The last fenced block, else the whole response."""
    blocks = _FENCE.findall(text or "")
    return blocks[-1] if blocks else (text or "").strip()


def parse_tests(target: str) -> dict:
    """Validate the `{"kind": "lcb", "cases": [...], "fn_name": ...}` reference."""
    gold = json.loads(target)
    cases = gold.get("cases") if isinstance(gold, dict) else None
    if (
        not isinstance(gold, dict)
        or gold.get("kind") != "lcb"
        or not isinstance(cases, list)
        or not cases
        or any(not isinstance(c, dict) or "input" not in c or "output" not in c for c in cases)
    ):
        raise ValueError("gold_not_test_cases")
    return gold


def _norm_stdio(text: str) -> str:
    return "\n".join(line.rstrip() for line in (text or "").strip("\n").splitlines())


def stdio_match(got: str, expected: str) -> bool:
    """Equal lines after trailing-space normalization, or equal Decimal tokens per line."""
    got_lines = _norm_stdio(got).splitlines()
    expected_lines = _norm_stdio(expected).splitlines()
    if len(got_lines) != len(expected_lines):
        return False
    for a, b in zip(got_lines, expected_lines):
        if a == b:
            continue
        try:
            if [Decimal(x) for x in a.split()] == [Decimal(x) for x in b.split()]:
                continue
        except Exception:
            return False
        return False
    return True


def functional_match(got, expected: str) -> bool:
    """The return value equals the expected JSON value, or its text does."""
    exp = (expected or "").strip()
    try:
        parsed = json.loads(exp)
        if parsed == got or json.loads(json.dumps(got)) == parsed:
            return True
    except (TypeError, ValueError):
        pass
    return str(got).strip() == exp


def _candidate_failure(cause: str, **details) -> Outcome:
    return Outcome(GradeCode.INVALID_ANSWER, details={"cause": cause, **details})


def _case_outcome(stdout: str, returncode: int, functional: bool):
    """The candidate's reported output for one case, or the failure it caused."""
    if returncode != 0:
        return None, _candidate_failure("candidate_exit", exit_code=returncode)
    lines = [line for line in stdout.splitlines() if line.strip()]
    try:
        report = json.loads(lines[-1])
    except (IndexError, ValueError):
        report = None
    if not isinstance(report, dict):
        return None, _candidate_failure("candidate_bad_output")
    if not report.get("ok"):
        reason = str(report.get("reason") or "")
        match = _REPORTED.fullmatch(reason)
        if match:
            return None, _candidate_failure(
                match[1], error_type=match[2].strip(), error_message=match[3] or None
            )
        if reason in _CANDIDATE_CAUSES:
            return None, _candidate_failure(reason)
        return None, _candidate_failure("candidate_bad_output")
    if functional:
        if "value" not in report:
            return None, _candidate_failure("candidate_bad_output")
        return report["value"], None
    value = report.get("stdout")
    if not isinstance(value, str):
        return None, _candidate_failure("candidate_bad_output")
    return value, None


def _refuse_local(env: SandboxEnvironment) -> None:
    try:
        env.as_type(LocalSandboxEnvironment)
    except TypeError:
        return
    raise SandboxUnavailable("candidate code never runs in a local sandbox")


async def _run(env: SandboxEnvironment, python: str, request: dict, timeout_s: int):
    return await env.exec(
        [python, "-I", "-c", RUNNER],
        input=json.dumps(request),
        timeout=timeout_s,
        timeout_retry=False,
    )


async def _probe(env: SandboxEnvironment, python: str) -> None:
    try:
        result = await _run(env, python, {"op": "probe"}, CASE_TIMEOUT_S)
        lines = result.stdout.strip().splitlines()
        report = json.loads(lines[-1]) if result.success and lines else None
    except Exception as exc:
        raise SandboxUnavailable(f"sandbox runner probe failed: {type(exc).__name__}") from exc
    if report != {"ok": True, "probe": code_runner.PROBE}:
        raise SandboxUnavailable("sandbox cannot run the test runner")


async def grade_tests(
    env: SandboxEnvironment,
    target: str,
    response: str,
    *,
    python: str = "python3",
    timeout_s: int = CASE_TIMEOUT_S,
) -> Outcome:
    """Run each case in `env` and compare its output here; the first failure decides."""
    try:
        gold = parse_tests(target)
    except (ValueError, TypeError) as exc:
        return Outcome(GradeCode.INVALID_GOLD, status="unavailable", details={"cause": str(exc)})
    code = extract_code(response)
    if not code.strip():
        return _candidate_failure("no_code_in_response")
    _refuse_local(env)
    await _probe(env, python)
    entry_point = str(gold.get("fn_name") or "")
    for index, case in enumerate(gold["cases"]):
        testtype = str(case.get("testtype") or "stdin").lower()
        functional = testtype == "functional"
        request = {
            "op": "functional" if functional else "stdio",
            "code": code,
            "entry_point": entry_point,
            "input": str(case["input"]),
        }
        try:
            result = await _run(env, python, request, timeout_s)
        except TimeoutError:
            return _candidate_failure("timeout")
        except OutputLimitExceededError:
            return _candidate_failure("output_limit")
        value, failure = _case_outcome(result.stdout, result.returncode, functional)
        if failure is not None:
            return failure
        expected = str(case["output"])
        matched = functional_match(value, expected) if functional else stdio_match(value, expected)
        if not matched:
            return Outcome(GradeCode.INCORRECT, details={"cause": f"case_{index}_mismatch"})
    return Outcome(GradeCode.CORRECT, score=1.0, details={"cause": "all_pass"})


@scorer(metrics=[mean(), stderr()])
def code_tests_scorer(
    python: str = "python3", timeout_s: int = CASE_TIMEOUT_S, sandbox_name: str | None = None
) -> Scorer:
    """Run the answer's last Python code block against the target's tests in the sandbox."""

    async def score(state: TaskState, target: Target):
        gated = completion_gate(state)
        if gated is not None:
            return to_score(gated)
        outcome = await grade_tests(
            sandbox(sandbox_name),
            target.text,
            state.output.completion,
            python=python,
            timeout_s=timeout_s,
        )
        return to_score(outcome)

    return score

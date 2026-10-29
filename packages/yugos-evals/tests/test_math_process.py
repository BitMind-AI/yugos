"""A wedged symbolic grader must end without inventing a grade or leaking a child."""

import math
import os
import sys
import time
from pathlib import Path

import anyio
import pytest
from evals_fakes import task_state
from inspect_ai.scorer import Target

from yugos.evaluation.grades import GradeCode
from yugos_evals.scorers import _math_process
from yugos_evals.scorers.math import grade_math, math_answer_scorer, reference_exclusion


@pytest.fixture
def worker_fixture(monkeypatch):
    tests = str(Path(__file__).parent)
    monkeypatch.setenv("PYTHONPATH", tests + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setattr(_math_process, "WORKER_MODULE", "math_worker_fixture")
    monkeypatch.setattr(_math_process, "TIMEOUT_S", 1)


def test_timeout_is_unscored_and_reaps_even_a_worker_ignoring_termination(tmp_path, worker_fixture):
    marker = tmp_path / "child.pid"
    start = time.monotonic()
    score = anyio.run(math_answer_scorer(), task_state("stall"), Target(str(marker)))
    assert time.monotonic() - start < 10
    assert math.isnan(score.value)
    assert score.metadata == {
        "code": "grader_error",
        "status": "error",
        "component": "math_worker",
        "cause": "timeout",
    }
    pid = int(marker.read_text())
    if sys.platform != "win32":
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.parametrize(
    "failure,cause", [("crash", "process_failed"), ("invalid", "invalid_result")]
)
def test_crashed_and_malformed_workers_never_settle_a_grade(
    failure, cause, tmp_path, worker_fixture
):
    outcome = grade_math(str(tmp_path / "child.pid"), failure)
    assert not outcome.settled and outcome.code == GradeCode.GRADER_ERROR
    assert outcome.details["cause"] == cause


def test_reference_screening_has_the_same_deadline(tmp_path, worker_fixture):
    assert reference_exclusion(str(tmp_path / "child.pid")) == (
        "unparseable_reference: math comparison timeout"
    )


def test_symbolic_sum_cannot_hold_the_grader_and_the_next_comparison_still_works(monkeypatch):
    # SymPy tries a direct sum for this family, as in the observed live stack.
    # Only the trusted parser runs; this string is never executed as Python.
    monkeypatch.setattr(_math_process, "TIMEOUT_S", 3)
    start = time.monotonic()
    outcome = grade_math("0", r"\boxed{\sum_{k=1}^{1000000000} k^k}")
    assert time.monotonic() - start < 12
    assert outcome.code == GradeCode.GRADER_ERROR and not outcome.settled
    assert outcome.details["cause"] == "timeout"
    assert grade_math("42", r"\boxed{42}").code == GradeCode.CORRECT

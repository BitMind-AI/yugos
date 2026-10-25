"""Output predictions retain gold privacy and require isolated expression evaluation."""

import importlib

import pytest
from evals_fakes import task_state
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Target

from yugos.evaluation.grades import GradeCode
from yugos.inspect_engine import restore_tasks
from yugos_evals.catalog import entry, freeze_entry, load_task
from yugos_evals.scorers._outcome import Outcome
from yugos_evals.scorers.cruxeval import cruxeval_output_scorer, prediction_program
from yugos_evals.sources._common import LoadedSource


def test_crux_predictions_are_parsed_without_evaluation(tmp_path):
    path = tmp_path / "executed"
    expression = f"__import__('pathlib').Path({str(path)!r}).write_text('bad')"
    assert prediction_program(expression, "def f(x): return x", "1")
    assert not path.exists()
    assert prediction_program("f(1)", "def f(x): return x", "1") is None
    assert prediction_program("2; print(2)", "def f(x): return x", "1") is None
    assert prediction_program("1 + 1", "def f(x): return x", "1")


@pytest.mark.anyio
async def test_crux_uses_private_equality_test_and_shared_supervisor(monkeypatch):
    module = importlib.import_module("yugos_evals.scorers.cruxeval")
    state = task_state("1 + 1")
    state.metadata = {"code": "def f(x): return x + 1", "input": "1"}
    calls = []

    async def grade(env, candidate, tests, **kwargs):
        calls.append((candidate, tests, kwargs))
        return Outcome(GradeCode.CORRECT, score=1)

    monkeypatch.setattr(module, "sandbox", lambda: object())
    monkeypatch.setattr(module, "grade_unittest", grade)
    result = await cruxeval_output_scorer()(state, Target("2"))
    assert result.value == 1
    candidate, tests, kwargs = calls[0]
    assert "self.assertEqual((2), task_func())" in tests
    assert "self.assertEqual" not in candidate
    assert kwargs["timeout_s"] == 8
    state.output.completion = "f(1)"
    assert (await cruxeval_output_scorer()(state, Target("2"))).value == 0
    assert len(calls) == 1


def test_crux_freeze_restores_without_reloading_source(monkeypatch):
    module = importlib.import_module("yugos_evals.tasks.cruxeval")
    sample = Sample(
        id="q",
        input="Predict f(1)",
        target="2",
        metadata={"code": "def f(x): return x+1", "input": "1"},
    )
    monkeypatch.setattr(module, "load_cruxeval", lambda **kw: LoadedSource([sample]))
    task = load_task(entry("cruxeval_o"))
    frozen = freeze_entry(entry("cruxeval_o"), task=task)

    def fail(**kw):
        raise AssertionError("execution acquired source")

    monkeypatch.setattr(module, "load_cruxeval", fail)
    [(restored, _)] = restore_tasks(frozen)
    assert restored.dataset[0].target == "2"
    assert restored.dataset[0].sandbox is None
    assert restored.sandbox is not None
    assert restored.dataset[0].input == "Predict f(1)"

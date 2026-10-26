"""SWE-bench-format datasets grade by their own recipe and restore from frozen samples."""

import importlib
import json

from evals_fakes import DIGEST_IMAGE
from inspect_ai.util import ComposeConfig, SandboxEnvironmentSpec

from yugos.inspect_engine import restore_tasks
from yugos_evals.prepared import prepare_task

harness = importlib.import_module("yugos_evals.tasks.swe_harness")

TEST_PATCH = "diff --git a/tests/test_a.py b/tests/test_a.py\n--- a/tests/test_a.py\n+++ b/tests/test_a.py\n"


def _row(**extra):
    return {
        "instance_id": "org__repo-7", "repo": "org/repo", "version": "1.0",
        "base_commit": "abc123", "problem_statement": "Fix it", "patch": "gold",
        "test_patch": TEST_PATCH, "FAIL_TO_PASS": json.dumps(["tests/test_a.py::test_new"]),
        "PASS_TO_PASS": json.dumps(["tests/test_a.py::test_old"]),
        **extra,
    }  # fmt: skip


def test_each_family_runs_its_own_recipe_and_grading_rule():
    rebench = harness._sample(
        _row(image_name="swerebench/x:latest",
             install_config=json.dumps({"test_cmd": "pytest -rA", "log_parser": "parse_log_pytest"})),
        "rebench",
    ).metadata  # fmt: skip
    script = harness.eval_script(rebench)
    assert "conda activate testbed" in script
    assert "pytest -rA tests/test_a.py" in script
    assert script.index("git apply") < script.index(harness.START) < script.index(harness.END)
    live = harness._sample(
        _row(test_cmds=["pytest -rA"], rebuild_cmds=["make"], log_parser="pytest"), "live"
    ).metadata
    assert "make" in harness.eval_script(live)
    log = f"{harness.START}\nPASSED tests/test_a.py::test_new\n{harness.END}"
    # Live accepts an absent PASS_TO_PASS test; standard SWE-bench grading does not.
    assert harness.resolved(live, log)[0]
    assert not harness.resolved(rebench, log)[0]
    assert harness.resolved(rebench, log + "\nPASSED tests/test_a.py::test_old")[0] is False
    full = f"{harness.START}\nPASSED tests/test_a.py::test_new\nPASSED tests/test_a.py::test_old\n{harness.END}"
    assert harness.resolved(rebench, full)[0]


def test_frozen_rows_restore_the_harness_scorer_and_pinned_sandbox(monkeypatch):
    rows = [_row(image="swebench/x:latest", eval_script="echo ok", log_parser="parse_log_pytest")]
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: rows)
    task = harness.swe_harness(dataset="fixture/rows", family="eval_script", revision="r")
    pinned = SandboxEnvironmentSpec(
        "modal", ComposeConfig.model_validate({"services": {"default": {"image": DIGEST_IMAGE}}})
    )
    [question] = prepare_task(task, source="swe_ml", workload_class="agentic", sandbox=pinned)
    [(restored, _)] = restore_tasks([question])
    config = restored.sandbox.model_dump(mode="json")["config"]
    assert config["services"]["default"]["image"] == DIGEST_IMAGE
    assert restored.dataset[0].metadata["instance_id"] == "org__repo-7"
    assert "swe_harness_scorer" in str(restored.scorer)

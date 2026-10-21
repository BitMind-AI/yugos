"""`code_tests`: legacy LiveCodeBench parity, sandbox-only execution, verdict integrity, loaders."""

import base64
import json
import os
import pickle
import shutil
import zlib
from pathlib import Path

import anyio
import evals_fakes
import pytest
from evals_fakes import (
    DIGEST_IMAGE,
    LocalProcesses,
    ScriptedRouter,
    assert_portable_task_config,
    run,
    task_state,
)
from inspect_ai.scorer import Target
from inspect_ai.util import ComposeConfig, SandboxEnvironmentSpec
from inspect_ai.util._sandbox.local import LocalSandboxEnvironment

from yugos.inspect_engine import restore_tasks
from yugos_evals.prepared import prepare_task as freeze_task
from yugos_evals.scorers import code as code_scorer
from yugos_evals.scorers.code import SandboxUnavailable, grade_tests
from yugos_evals.sources import _fetch, livecodebench, livecodebench_plus
from yugos_evals.tasks import code_tests
from yugos_evals.tasks.code_tests import code_sandbox

FIXTURES = Path(__file__).parent / "fixtures"
BASELINE = [
    json.loads(line)
    for line in (FIXTURES / "livecodebench_baseline.jsonl").read_text().splitlines()
]
REFERENCE = json.dumps(
    {
        "kind": "lcb",
        "fn_name": "",
        "cases": [{"input": "2\n", "output": "4\n", "testtype": "stdin"}],
    }
)


def _grade(reference, response, env=None):
    env = env or LocalProcesses()
    return anyio.run(lambda: grade_tests(env, reference, response)), env


def _reason(outcome) -> str:
    reason = outcome.details["cause"]
    if "error_type" in outcome.details:
        reason += ": " + outcome.details["error_type"]
        if outcome.details.get("error_message"):
            reason += ": " + outcome.details["error_message"]
    return reason


@pytest.mark.parametrize("case", BASELINE, ids=lambda case: case["name"])
def test_outcomes_match_the_recorded_legacy_livecodebench_grader(case):
    outcome, env = _grade(case["reference"], case["response"].strip())
    assert {"score": outcome.score, "reason": _reason(outcome)} == case["expected"]
    # Expected outputs never enter the sandbox; only inputs and candidate code do.
    for _, stdin in env.execs[1:]:
        assert json.loads(stdin).keys() == {"op", "code", "entry_point", "input"}
    assert not env.reads


def test_forged_reports_and_planted_verdict_files_earn_no_credit():
    forger = (
        "import json, os, sys\n"
        "open('verdict.json', 'w').write(json.dumps({'passed': True, 'score': 1}))\n"
        "os.write(1, b'\\n' + json.dumps({'ok': True, 'passed': True, 'score': 1}).encode() + b'\\n')\n"
        "os._exit(0)\n"
    )
    env = LocalProcesses()
    env.written["/tmp/results.json"] = json.dumps({"all_pass": True})
    outcome, _ = _grade(REFERENCE, forger, env)
    assert outcome.score == 0 and outcome.details["cause"] == "candidate_bad_output"
    assert not env.reads and "4\\n" not in env.payloads()
    printed = 'print(3)\nimport sys\nsys.stdout.write(\'{"ok": true, "stdout": "3\\\\n"}\')'
    outcome, _ = _grade(REFERENCE, printed)
    assert outcome.score == 0


def test_candidate_failures_are_graded_zero_not_infrastructure_errors():
    timeout, _ = _grade(REFERENCE, "while True: pass", LocalProcesses(timeout_on="while True"))
    assert (timeout.score, timeout.details["cause"]) == (0, "timeout")
    exit_code, _ = _grade(REFERENCE, "import os\nos._exit(3)")
    assert exit_code.details == {"cause": "candidate_exit", "exit_code": 3}
    assert exit_code.settled


def test_a_sandbox_that_cannot_run_the_runner_is_an_infrastructure_failure():
    env = LocalProcesses(probe_ok=False)
    with pytest.raises(SandboxUnavailable):
        _grade(REFERENCE, "print(4)", env)
    assert len(env.execs) == 1 and "print(4)" not in env.payloads()


def test_the_sandbox_service_uses_only_settings_hosted_preparation_admits():
    # Ycore's hosted Modal preparation rejects services with other settings (e.g. `init`),
    # so every code_tests pack would fail hosted admission.
    service = code_sandbox(DIGEST_IMAGE).config.services["default"]
    assert set(service.model_dump(exclude_none=True)) <= {
        "image",
        "command",
        "network_mode",
        "working_dir",
    }
    assert service.network_mode == "none"


def test_the_local_sandbox_is_refused_before_any_execution():
    local = LocalSandboxEnvironment.__new__(LocalSandboxEnvironment)
    with pytest.raises(SandboxUnavailable, match="local"):
        anyio.run(lambda: grade_tests(local, REFERENCE, "print(4)"))
    with pytest.raises(ValueError, match="local"):
        code_tests(image=DIGEST_IMAGE, sandbox_type="local")
    with pytest.raises(ValueError, match="digest"):
        code_tests(image="python:3.12-slim")
    with pytest.raises(ValueError, match="image"):
        code_tests()


def test_scorer_gates_truncation_and_invalid_tests_before_execution(monkeypatch):
    def unexpected(name=None):
        raise AssertionError("sandbox used")

    monkeypatch.setattr(code_scorer, "sandbox", unexpected)
    score = code_scorer.code_tests_scorer()
    truncated = anyio.run(
        score, task_state("print(4)", stop_reason="max_tokens"), Target(REFERENCE)
    )
    assert truncated.value == 0 and truncated.metadata["code"] == "truncated"
    outcome, env = _grade(json.dumps({"kind": "lcb", "cases": []}), "print(4)")
    assert not outcome.settled and outcome.code == "invalid_gold" and not env.execs


# --- Loaders ---------------------------------------------------------------------------


def _pickled(cases) -> str:
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(cases)))).decode()


class _Exploit:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def _lcb_row(question_id, content, public, private, **fields):
    return {
        "question_title": "t",
        "question_content": content,
        "platform": "atcoder",
        "question_id": question_id,
        "contest_id": "abc400",
        "contest_date": "2025-03-01T00:00:00",
        "starter_code": "",
        "difficulty": "medium",
        "public_test_cases": json.dumps(public),
        "private_test_cases": _pickled(private),
        "metadata": "{}",
        **fields,
    }


STDIN_CASE = {"input": "2\n", "output": "4\n", "testtype": "stdin"}


def test_test_payloads_decode_literal_json_and_refuse_pickled_objects():
    assert livecodebench.decode_test_cases(_pickled([STDIN_CASE])) == [STDIN_CASE]
    exploit = base64.b64encode(zlib.compress(pickle.dumps(_Exploit()))).decode()
    with pytest.raises(ValueError, match="literal JSON string"):
        livecodebench.decode_test_cases(exploit)
    with pytest.raises(ValueError, match="unsupported"):
        livecodebench.decode_test_cases(json.dumps([{**STDIN_CASE, "testtype": "interactive"}]))


def _jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return _fetch.sha256_file(path)


def _lcb_fixture(tmp_path, monkeypatch):
    lcb = tmp_path / "lcb"
    lcb.mkdir()
    rows = [
        _lcb_row(
            "abc400_a",
            "Double the number.",
            [STDIN_CASE],
            [{**STDIN_CASE, "input": "3\n", "output": "6\n"}],
        ),
        _lcb_row(
            "abc400_b",
            "Triple the number.",
            [STDIN_CASE | {"output": "6\n"}],
            [STDIN_CASE | {"output": "6\n"}],
            contest_date="2025-05-01T00:00:00",
        ),
    ]
    monkeypatch.setitem(livecodebench.FILE_SHA256, "test6.jsonl", _jsonl(lcb / "test6.jsonl", rows))
    return lcb


def test_livecodebench_loader_windows_contest_dates_and_keeps_tests_private(tmp_path, monkeypatch):
    lcb = _lcb_fixture(tmp_path, monkeypatch)
    loaded = livecodebench.load_livecodebench(["v6"], end_date="2025-04-01", data_dir=lcb)
    [sample] = loaded.samples
    assert sample.id == "livecodebench-atcoder-abc400_a"
    assert "6\\n" not in sample.input and "Double the number." in sample.input
    gold = json.loads(sample.target)
    assert [c["output"] for c in gold["cases"]] == ["4\n", "6\n"] and gold["fn_name"] == ""
    monkeypatch.setitem(livecodebench.FILE_SHA256, "test6.jsonl", "0" * 64)
    with pytest.raises(ValueError, match="not the pinned"):
        livecodebench.load_livecodebench(["v6"], data_dir=lcb)


def _plus_row(problem_id, statement, split="medium", seed=None):
    cases = [
        {"input": "1\n", "output": "2\n", "tier": "public"},
        {"input": "5\n", "output": "10\n", "tier": "private"},
    ]
    return {
        "problem_id": problem_id,
        "title": "t",
        "difficulty": split,
        "problem_statement": statement,
        "public_tests": cases[:1],
        "test_cases": cases,
        "solution_code": "print(2*int(input()))" if seed else "",
        "mutation_type": "twist" if seed else "",
        "approach": "",
        "mutation_notes": "",
        "seed_problem": seed,
    }


def test_livecodebench_plus_drops_problems_already_in_v6(tmp_path, monkeypatch):
    lcb = _lcb_fixture(tmp_path, monkeypatch)
    plus = tmp_path / "plus"
    plus.mkdir()
    medium = [
        _plus_row("abc400_a", "Renamed statement."),  # original: overlaps by ID
        _plus_row("evo_1", "Triple the number.", seed={"question_id": "abc400_b"}),  # by text
        _plus_row("evo_2", "Quadruple the number.", seed={"question_id": "abc400_b"}),
    ]
    monkeypatch.setitem(
        livecodebench_plus.FILE_SHA256, "medium.jsonl", _jsonl(plus / "medium.jsonl", medium)
    )
    loaded = livecodebench_plus.load_livecodebench_plus(["medium"], data_dir=plus, lcb_data_dir=lcb)
    assert [s.id for s in loaded.samples] == ["livecodebench-plus-evo_2"]
    assert {e["id"]: e["reason"] for e in loaded.excluded} == {
        "livecodebench-plus-abc400_a": "overlap: already in LiveCodeBench as livecodebench-atcoder-abc400_a",
        "livecodebench-plus-evo_1": "overlap: already in LiveCodeBench as livecodebench-atcoder-abc400_b",
    }
    kept = livecodebench_plus.load_livecodebench_plus(
        ["medium"], exclude_releases=[], data_dir=plus
    )
    assert len(kept.samples) == 3


# --- Freezing and end-to-end ---------------------------------------------------------------


def test_task_freezes_restores_and_runs_only_in_the_bound_sandbox(tmp_path, monkeypatch):
    lcb = _lcb_fixture(tmp_path, monkeypatch)
    task = code_tests(source="livecodebench", image=DIGEST_IMAGE, data_dir=str(lcb))
    assert task.sandbox.type == "docker"
    hosted = SandboxEnvironmentSpec(
        "yugos_evals_fixture",
        ComposeConfig.model_validate({"services": {"default": {"image": DIGEST_IMAGE}}}),
    )
    questions = freeze_task(
        task, source="livecodebench", workload_class="math_code", sandbox=hosted
    )
    questions = [q.model_validate_json(q.model_dump_json()) for q in questions]
    for question in questions:
        assert_portable_task_config(question)
    monkeypatch.setitem(livecodebench.FILE_SHA256, "test6.jsonl", "0" * 64)
    [(restored, _)] = restore_tasks(questions)
    env = LocalProcesses()
    monkeypatch.setitem(evals_fakes.CURRENT, "env", env)
    router = ScriptedRouter("```python\nprint(2 * int(input()))\n```")
    log = run(restored, router, tmp_path)
    scores = {s.id: s.scores["code_tests_scorer"].value for s in log.samples}
    assert scores == {
        "livecodebench:livecodebench-atcoder-abc400_a": 1.0,
        "livecodebench:livecodebench-atcoder-abc400_b": 0.0,
    }
    # Tests stay private: the router saw prompts only, the sandbox saw inputs only.
    assert '"output"' not in router.seen() and '"output"' not in env.payloads()
    local = SandboxEnvironmentSpec(
        "local", ComposeConfig.model_validate({"services": {"default": {"image": DIGEST_IMAGE}}})
    )
    with pytest.raises(ValueError, match="isolation"):
        freeze_task(task, source="livecodebench", workload_class="math_code", sandbox=local)


# --- Real Docker -----------------------------------------------------------------------------

_NETWORK_PROBE = """
import socket
n = int(input())
try:
    socket.create_connection(("1.1.1.1", 53), timeout=3).close()
    print(n)
except OSError:
    print(2 * n)
"""
_FORGER = """
import json, os
verdict = json.dumps({"passed": True, "score": 1, "all_pass": True})
for path in ("verdict.json", "/tmp/verdict.json", "/tmp/results.json"):
    open(path, "w").write(verdict)
os.write(1, json.dumps({"ok": True, "passed": True, "score": 1}).encode() + b"\\n")
os._exit(0)
"""


@pytest.mark.skipif(
    not (os.environ.get("YUGOS_EVALS_DOCKER_TESTS") and shutil.which("docker")),
    reason="runs candidate code in the catalog's python image under local Docker",
)
@pytest.mark.parametrize(
    ("program", "expected", "cause"),
    [
        ("print(2 * int(input()))", 1.0, "all_pass"),
        ("print(int(input()) + 1)", 0.0, "case_0_mismatch"),
        # Earns credit only when the sandbox has no network route.
        (_NETWORK_PROBE, 1.0, "all_pass"),
        (_FORGER, 0.0, "candidate_bad_output"),
    ],
    ids=["correct", "wrong", "network-blocked", "planted-verdict"],
)
def test_catalog_entry_grades_in_its_pinned_docker_image(
    program, expected, cause, tmp_path, monkeypatch
):
    from yugos_evals.catalog import entry, freeze_entry
    from yugos_evals.local.runner import use_local_sandboxes

    lcb = tmp_path / "lcb"
    lcb.mkdir()
    private = [{**STDIN_CASE, "input": "3\n", "output": "6\n"}]
    rows = [_lcb_row("abc400_a", "Double the number.", [STDIN_CASE], private)]
    monkeypatch.setitem(livecodebench.FILE_SHA256, "test6.jsonl", _jsonl(lcb / "test6.jsonl", rows))
    questions = freeze_entry(entry("livecodebench"), task_args={"data_dir": str(lcb)})
    tasks = restore_tasks([q.model_validate_json(q.model_dump_json()) for q in questions])
    use_local_sandboxes(tasks, docker_available=True)
    [(restored, _)] = tasks
    log = run(restored, ScriptedRouter(f"```python\n{program.strip()}\n```"), tmp_path)
    [sample] = log.samples
    score = sample.scores["code_tests_scorer"]
    assert log.status == "success" and score.value == expected, score.metadata
    assert score.metadata["status"] == "graded" and score.metadata["cause"] == cause

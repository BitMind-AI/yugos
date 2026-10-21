"""BigCodeBench's test supervisor must reject early exits and forged output."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from evals_fakes import DIGEST_IMAGE, LocalProcesses
from inspect_ai.dataset import Sample

from yugos.inspect_engine import restore_tasks
from yugos_evals.catalog import entry, freeze_entry
from yugos_evals.prepared import build
from yugos_evals.tasks.code_tests import code_sandbox

RUNNER = Path(__file__).parents[1] / "src/yugos_evals/scorers/python_unittest_runner.py"
TESTS = (
    "import unittest\n"
    "class TestCases(unittest.TestCase):\n"
    "    def test_answer(self): self.assertEqual(task_func(), 2)\n"
)


def _run(candidate, *, timeout_s=1, tests=TESTS, libs=()):
    result = subprocess.run(
        [sys.executable, "-I", str(RUNNER)],
        input=json.dumps(
            {
                "candidate": candidate,
                "tests": tests,
                "timeout_s": timeout_s,
                "libs": libs,
                "worker_source": RUNNER.read_text(),
            }
        ),
        capture_output=True,
        text=True,
        timeout=timeout_s + 5,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    ("candidate", "passed", "cause"),
    [
        ("def task_func(): return 2", True, "all_pass"),
        ("def task_func(): return 3", False, "test_failure"),
        ("import sys; sys.exit(0)", False, "test_failure"),
        ("import os; os._exit(0)", False, "test_failure"),
        (
            'import os; os.write(1, b\'{"passed": true, "cause": "all_pass"}\\n\'); os._exit(0)',
            False,
            "test_failure",
        ),
        (
            "import sys, os\n"
            'sys._getframe(1).f_locals[\'connection\'].send_bytes(b\'{"passed": true, "cause": "all_pass"}\')\n'
            "os._exit(0)",
            False,
            "test_failure",
        ),
        ("while True: pass", False, "timeout"),
    ],
)
def test_supervisor_requires_completed_passing_tests(candidate, passed, cause):
    report = _run(candidate)
    assert report["passed"] is passed and report["cause"] == cause


@pytest.mark.parametrize("title,passed", [("Histogram", True), ("wrong", False)])
def test_dataframe_and_plot_methods_reach_trusted_tests(title, passed, monkeypatch, tmp_path):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    candidate = f"""
import pandas as pd
import matplotlib.pyplot as plt
def task_func(values):
    if not values:
        raise ValueError("empty")
    frame = pd.DataFrame({{"value": values, "object_value": pd.Series(values, dtype=object)}})
    ax = frame.hist()[0][0]
    ax.set_title({title!r})
    return frame, ax
"""
    tests = """
import unittest
import pandas as pd
from matplotlib.axes import Axes
class TestCases(unittest.TestCase):
    def test_objects(self):
        frame, ax = task_func([1, 2, 3])
        self.assertIsInstance(frame, pd.DataFrame)
        self.assertEqual(frame["value"].tolist(), [1, 2, 3])
        self.assertTrue(pd.api.types.is_numeric_dtype(frame["value"]))
        self.assertFalse(pd.api.types.is_numeric_dtype(frame["object_value"]))
        self.assertEqual(ax.get_title(), "Histogram")
        self.assertIsInstance(ax, Axes)
    def test_exception(self):
        with self.assertRaises(ValueError):
            task_func([])
    def tearDown(self):
        plt.clf()
"""
    report = _run(candidate, timeout_s=15, tests=tests, libs=("pandas", "matplotlib"))
    assert report["passed"] is passed
    assert report["cause"] == ("all_pass" if passed else "test_failure")


def test_arrow_preserves_nullable_categorical_timezone_and_index_in_both_directions():
    tests = """
import unittest
import pandas as pd
class TestCases(unittest.TestCase):
    def test_round_trip(self):
        frame = pd.DataFrame({
            "numbers": pd.Series([1, None], dtype="Int64"),
            "labels": pd.Categorical(["a", "b"], ordered=True),
            "dates": pd.to_datetime(["2020-01-01", None], utc=True),
        })
        frame.index = pd.Index(["first", "second"], name="row")
        pd.testing.assert_frame_equal(task_func(frame), frame)
"""
    report = _run("def task_func(frame): return frame.copy()", tests=tests, timeout_s=10)
    assert report["passed"] and report["cause"] == "all_pass"


def test_unsupported_arrow_values_do_not_become_incorrect_answers():
    candidate = 'import pandas as pd\ndef task_func(): return pd.DataFrame({"value": [object()]})'
    report = _run(candidate, timeout_s=10)
    assert report["passed"] is False and report["cause"] == "unsupported_result"


def test_candidate_cannot_call_back_into_the_trusted_grader():
    candidate = """
import sys
def task_func():
    frame = sys._getframe().f_back
    while "conn" not in frame.f_locals:
        frame = frame.f_back
    frame.f_locals["conn"].root
    return 2
"""
    report = _run(candidate, timeout_s=5)
    assert report["passed"] is False and report["cause"] == "test_failure"


def test_partial_rpc_frame_cannot_stall_the_grader_past_its_deadline():
    candidate = """
import sys
import time
def task_func():
    frame = sys._getframe().f_back
    while "conn" not in frame.f_locals:
        frame = frame.f_back
    frame.f_locals["conn"]._channel.stream.sock.sendall(b"\\x00")
    time.sleep(30)
    return 2
"""
    report = _run(candidate, timeout_s=3)
    assert report["passed"] is False and report["cause"] == "timeout"


@pytest.mark.parametrize("header", [(0xFFFFFFFF, 0), (1, 1)])
def test_candidate_rpc_frames_are_rejected_before_body_allocation(header):
    spec = importlib.util.spec_from_file_location("grader_transport_fixture", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Stream:
        def read(self, size):
            if size != module.Channel.FRAME_HEADER.size:
                pytest.fail("untrusted frame body was read before validation")
            return module.Channel.FRAME_HEADER.pack(*header)

    with pytest.raises(module.CandidateError):
        module.BoundedChannel(Stream()).recv()


def test_live_plot_method_shares_the_original_deadline(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    candidate = """
import matplotlib.pyplot as plt
import time
def task_func():
    ax = plt.subplots()[1]
    def title():
        time.sleep(30)
        return "late"
    ax.get_title = title
    return ax
"""
    tests = 'import unittest\nclass TestCases(unittest.TestCase):\n def test_plot(self): self.assertEqual(task_func().get_title(), "late")'
    report = _run(candidate, tests=tests, libs=("matplotlib",), timeout_s=5)
    assert report["passed"] is False and report["cause"] == "timeout"


@pytest.mark.asyncio
async def test_packaged_scorer_bootstraps_arrow_and_rpyc_workers(monkeypatch, tmp_path):
    from yugos.evaluation.grades import GradeCode
    from yugos_evals.scorers.python_unittest import grade_unittest

    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))

    class Processes(LocalProcesses):
        async def exec(self, cmd, **kwargs):
            return await super().exec([sys.executable, *cmd[1:]], **kwargs)

    candidate = """
import pandas as pd
import matplotlib.pyplot as plt
def task_func():
    frame = pd.DataFrame({"value": [1, 2]})
    ax = frame.hist()[0][0]
    ax.set_title("Chart")
    return frame, ax
"""
    tests = """
import unittest
class TestCases(unittest.TestCase):
    def test_objects(self):
        frame, ax = task_func()
        self.assertEqual(frame["value"].tolist(), [1, 2])
        self.assertEqual(ax.get_title(), "Chart")
        plt.clf()
"""
    result = await grade_unittest(
        Processes(), candidate, tests, libs=("pandas", "matplotlib"), timeout_s=15
    )
    assert result.code == GradeCode.CORRECT and result.score == 1


def test_prepared_bigcodebench_uses_supervised_scorer():
    from inspect_ai._util.registry import registry_info

    task = build(
        options={"dataset": [Sample(id="example", input="prompt", target="gold")]},
        task="inspect_evals/bigcodebench",
        args={},
    )
    assert registry_info(task.scorer[0]).name == "yugos_evals/bigcodebench_scorer"


def test_bigcodebench_freeze_and_restore_selects_supervised_scorer(monkeypatch):
    upstream = importlib.import_module("inspect_evals.bigcodebench.bigcodebench")

    record = {
        "task_id": "BigCodeBench/0",
        "complete_prompt": "def task_func():\n    pass\n",
        "canonical_solution": "    return 2\n",
        "test": TESTS,
        "libs": [],
    }
    monkeypatch.setattr(upstream, "load_dataset", lambda *a, **kw: [record])
    monkeypatch.setattr(
        upstream,
        "force_build_or_pull_docker_image",
        lambda **kw: Path(upstream.__file__).parent / "compose.yaml",
    )
    native = upstream.bigcodebench(subset=[0])
    [question] = freeze_entry(
        entry("bigcodebench"), task=native, sandbox=code_sandbox(DIGEST_IMAGE)
    )
    assert question.inspect_task["scorer"] == "bigcodebench_scorer"
    [(restored, _)] = restore_tasks([question])
    from inspect_ai._util.registry import registry_info

    assert registry_info(restored.scorer[0]).name.endswith(question.inspect_task["scorer"])

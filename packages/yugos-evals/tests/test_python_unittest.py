"""BigCodeBench's test supervisor must reject early exits and forged output."""

import importlib
import json
import os
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


@pytest.fixture(scope="session")
def matplotlib_cache(tmp_path_factory):
    """Build fonts before tests measure the candidate execution deadline."""
    cache = tmp_path_factory.mktemp("matplotlib-fonts")
    subprocess.run(
        [sys.executable, "-c", "import matplotlib.font_manager"],
        env={**os.environ, "MPLCONFIGDIR": str(cache), "MPLBACKEND": "Agg"},
        check=True,
        timeout=120,
        capture_output=True,
    )
    return cache


def _run(candidate, *, timeout_s=1, tests=TESTS, libs=(), reference="", http_fixtures=None):
    result = subprocess.run(
        [sys.executable, "-I", str(RUNNER)],
        input=json.dumps(
            {
                "candidate": candidate,
                "tests": tests,
                "timeout_s": timeout_s,
                "libs": libs,
                "reference": reference,
                "http_fixtures": http_fixtures or {},
                "worker_source": RUNNER.read_text(),
            }
        ),
        capture_output=True,
        text=True,
        timeout=timeout_s + 5,
        check=True,
    )
    return json.loads(result.stdout)


def test_mocked_randomness_and_scientific_values_keep_native_test_behavior():
    candidate = """
import random
import numpy as np
def task_func():
    return np.array([random.randint(0, 100), random.randint(0, 100)]), np.float64(1.5)
"""
    tests = """
import unittest
from unittest.mock import patch
class TestCases(unittest.TestCase):
    @patch('random.randint', return_value=50)
    def test_values(self, mocked):
        array, scalar = task_func()
        np.testing.assert_array_equal(array, [50, 50])
        self.assertEqual(array.dtype, np.dtype('int64'))
        self.assertIsInstance(scalar, np.float64)
        self.assertEqual(scalar, 1.5)
        self.assertEqual(mocked.call_count, 2)
        mocked.assert_called_with(0, 100)
"""
    report = _run(candidate, tests=tests, reference=candidate, libs=("numpy",), timeout_s=10)
    assert report["passed"] and report["cause"] == "all_pass"


def test_mock_open_and_native_exception_types_reach_the_candidate():
    candidate = """
import json
import statistics
def task_func(path):
    if not path:
        raise statistics.StatisticsError('empty path')
    with open(path) as handle:
        return json.load(handle)
"""
    tests = """
import unittest
from unittest.mock import patch, mock_open
class TestCases(unittest.TestCase):
    def test_fixture(self):
        with patch('builtins.open', mock_open(read_data='{"x": 2}')) as mocked:
            self.assertEqual(task_func('fixture.json'), {'x': 2})
            mocked.assert_called_once_with('fixture.json')
            mocked.return_value.read.assert_called_once_with()
    def test_exception(self):
        with self.assertRaisesRegex(statistics.StatisticsError, 'empty path'):
            task_func('')
"""
    report = _run(candidate, tests=tests, reference=candidate, timeout_s=10)
    assert report["passed"] and report["cause"] == "all_pass"


def test_model_import_aliases_cannot_replace_trusted_test_helpers():
    candidate = "from os import _exit as helper\ndef task_func(): return 3"
    report = _run(candidate, reference="import math as helper")
    assert report["passed"] is False and report["cause"] == "test_failure"


def test_returned_handlers_observe_later_patches_and_mock_callbacks():
    candidate = """
import smtplib
def task_func():
    class Handler:
        def __init__(self, request):
            self.request = request
        def handle(self):
            with smtplib.SMTP('fixture.example') as connection:
                connection.sendmail('from', 'to', self.request.read())
            self.respond(200)
    return Handler
"""
    tests = """
import unittest
from unittest.mock import MagicMock, patch
class TestCases(unittest.TestCase):
    def test_handler(self):
        handler_type = task_func()
        request = MagicMock()
        request.read.return_value = b'hello'
        handler = handler_type(request)
        handler.respond = MagicMock()
        with patch('smtplib.SMTP') as smtp:
            handler.handle()
            smtp.assert_called_once_with('fixture.example')
            smtp.return_value.__enter__.return_value.sendmail.assert_called_once_with('from', 'to', b'hello')
        handler.respond.assert_called_once_with(200)
        request.read.assert_called_once_with()
"""
    report = _run(candidate, tests=tests, timeout_s=10)
    assert report["passed"] and report["cause"] == "all_pass"


def test_test_output_does_not_corrupt_the_supervisor_report():
    tests = TESTS + "\nprint('test fixture output')\n"
    report = _run("def task_func(): return 2", tests=tests)
    assert report["passed"]


def test_offline_http_snapshot_keeps_nested_test_mocks():
    import base64
    import hashlib

    body = b"frozen response"
    candidate = """
import urllib.request
def task_func(url):
    with urllib.request.urlopen(url) as response:
        return response.read().decode()
"""
    tests = """
import unittest
from unittest.mock import patch
from urllib.error import URLError
class TestCases(unittest.TestCase):
    def test_snapshot(self):
        self.assertEqual(task_func('https://fixture.example/'), 'frozen response')
    @patch('urllib.request.urlopen', side_effect=URLError('native mock'))
    def test_nested_patch(self, mocked):
        with self.assertRaisesRegex(URLError, 'native mock'):
            task_func('https://fixture.example/')
"""
    report = _run(
        candidate,
        tests=tests,
        timeout_s=10,
        http_fixtures={
            "https://fixture.example/": {
                "body_base64": base64.b64encode(body).decode(),
                "sha256": hashlib.sha256(body).hexdigest(),
                "status": 200,
            }
        },
    )
    assert report["passed"]


@pytest.mark.parametrize("value", ["['numpy', 'matplotlib']", ["numpy", "matplotlib"]])
def test_native_library_metadata_is_normalized(value):
    from yugos_evals.scorers.python_unittest import library_names

    assert library_names(value) == ("numpy", "matplotlib")


@pytest.mark.parametrize(
    ("candidate", "passed", "cause"),
    [
        ("def task_func(): return 2", True, "all_pass"),
        ("def task_func(): return 3", False, "test_failure"),
        ("import sys; sys.exit(0)", False, "test_failure"),
        ("import os; os._exit(0)", False, "test_failure"),
        (
            "import unittest\ndef task_func(): raise unittest.SkipTest('forged skip')",
            False,
            "test_failure",
        ),
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
def test_dataframe_and_plot_methods_reach_trusted_tests(
    title, passed, monkeypatch, matplotlib_cache
):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(matplotlib_cache))
    candidate = f"""
import pandas as pd
import matplotlib.pyplot as plt
def task_func(values):
    if not values:
        raise ValueError("empty")
    frame = pd.DataFrame({{"value": values, "object_value": pd.Series(values, dtype=object)}})
    ax = frame.hist()[0][0]
    ax.set_title({title!r})
    ax.plot(values, values)
    return frame, ax
"""
    tests = """
import unittest
import pandas as pd
import numpy as np
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
        self.assertEqual(ax.get_lines()[0].get_ydata().tolist(), [1, 2, 3])
        np.testing.assert_array_equal(ax.get_lines()[0].get_xdata(), [1, 2, 3])
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


@pytest.mark.parametrize(
    "zone", ["UTC", "America/New_York", "Europe/Berlin", "Asia/Tokyo", "Australia/Lord_Howe"]
)
def test_zoneinfo_frames_preserve_provider_dst_nat_and_empty_values_in_both_directions(zone):
    tests = f"""
import unittest
import pandas as pd
from zoneinfo import ZoneInfo
class TestCases(unittest.TestCase):
    def test_round_trip(self):
        times = pd.to_datetime(['2026-11-01T05:30:00Z', None, '2026-11-01T06:30:00Z'])
        times = times.tz_convert(ZoneInfo({zone!r}))
        frame = pd.DataFrame({{'time': times, 'label': ['first', 'missing', 'last']}})
        frame.index = pd.Index(['a', 'b', 'c'], name='row')
        for value in (frame, frame.iloc[:0]):
            result = task_func(value)
            pd.testing.assert_frame_equal(result, value)
            self.assertIs(type(result['time'].dtype.tz), ZoneInfo)
        self.assertEqual(task_func(frame)['time'].iloc[-1].fold, frame['time'].iloc[-1].fold)
"""
    report = _run("def task_func(frame): return frame.copy()", tests=tests, timeout_s=15)
    assert report["passed"] and report["cause"] == "all_pass"


def test_wrong_timestamp_output_gets_native_test_failure_instead_of_unavailable_grade():
    candidate = """
import pandas as pd
from zoneinfo import ZoneInfo
def task_func():
    return pd.DataFrame({'display': pd.date_range('2026-06-01', periods=2, tz=ZoneInfo('UTC'))})
"""
    tests = """
import unittest
class TestCases(unittest.TestCase):
    def test_format(self):
        self.assertEqual(task_func()['display'].tolist(), ['2026-06-01', '2026-06-02'])
"""
    report = _run(candidate, tests=tests, timeout_s=10)
    assert report["passed"] is False and report["cause"] == "test_failure"
    assert report["tests_run"] == report["tests_expected"] == 1


@pytest.mark.parametrize("custom", [False, True])
def test_noncanonical_zoneinfo_stays_unavailable_instead_of_changing_timezone_rules(custom):
    timezone = (
        "next(ZoneInfo.from_file(open(str(Path(p) / 'UTC'), 'rb'), key='America/New_York') "
        "for p in TZPATH if (Path(p) / 'UTC').is_file())"
        if custom
        else "ZoneInfo.no_cache('America/New_York')"
    )
    candidate = f"""
from pathlib import Path
from zoneinfo import TZPATH, ZoneInfo
import pandas as pd
def task_func():
    return pd.DataFrame({{'time': pd.date_range('2026-01-01', periods=2, tz={timezone})}})
"""
    report = _run(candidate, timeout_s=10)
    assert report["passed"] is False and report["cause"] == "unsupported_result"


@pytest.mark.parametrize(
    "columns",
    [
        ((-1, "America/New_York"),),
        ((True, "America/New_York"),),
        (("0", "America/New_York"),),
        ((0, "America/Chicago"),),
        ((0, "../../etc/passwd"),),
        ((0, ["America/New_York"]),),
        ((0, "America/New_York"), (0, "America/New_York")),
        ((0, "America/New_York", "extra"),),
    ],
)
def test_untrusted_timezone_descriptors_cannot_relabel_arrow_columns(columns):
    from zoneinfo import ZoneInfo

    import pandas as pd

    spec = importlib.util.spec_from_file_location("timezone_transport_fixture", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = pd.DataFrame(
        {"time": pd.date_range("2026-01-01", periods=2, tz=ZoneInfo("America/New_York"))}
    )
    with pytest.raises(module.CandidateError):
        module._restore_zoneinfo(frame, columns)


def test_existing_dataframe_payload_without_timezone_descriptors_still_decodes():
    import pandas as pd
    import pyarrow as pa

    spec = importlib.util.spec_from_file_location("legacy_frame_fixture", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = pd.DataFrame({"value": [1, 2]})
    table = pa.Table.from_pandas(frame, preserve_index=True)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    result = module._materialize(("dataframe", (sink.getvalue().to_pybytes(), ())))
    pd.testing.assert_frame_equal(result, frame)


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


def test_live_plot_method_shares_the_original_deadline(monkeypatch, matplotlib_cache):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(matplotlib_cache))
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
async def test_packaged_scorer_bootstraps_arrow_and_rpyc_workers(monkeypatch, matplotlib_cache):
    from yugos.evaluation.grades import GradeCode
    from yugos_evals.scorers.python_unittest import grade_unittest

    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(matplotlib_cache))

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
    monkeypatch.setattr(upstream, "get_images_from_compose", lambda *a, **kw: [DIGEST_IMAGE])
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
    assert question.inspect_task["metadata"]["unittest_supervisor_version"] == 2
    [(restored, _)] = restore_tasks([question])
    from inspect_ai._util.registry import registry_info

    assert registry_info(restored.scorer[0]).name.endswith(question.inspect_task["scorer"])

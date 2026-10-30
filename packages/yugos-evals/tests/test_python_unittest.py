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


@pytest.mark.parametrize("correct", [True, False])
def test_native_paths_work_as_inputs_and_preserve_their_result_type(correct):
    candidate = (
        "def task_func(path):\n path.write_text('updated')\n return path.with_suffix('.done')"
        if correct
        else "def task_func(path): return 'wrong path'"
    )
    tests = """
import unittest
import tempfile
from pathlib import Path
class TestCases(unittest.TestCase):
    def test_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.txt'
            result = task_func(path)
            self.assertEqual(path.read_text(), 'updated')
            self.assertIsInstance(result, Path)
            self.assertEqual(result, path.with_suffix('.done'))
"""
    report = _run(candidate, tests=tests, timeout_s=10)
    assert report["passed"] is correct
    assert report["cause"] == ("all_pass" if correct else "test_failure")


@pytest.mark.parametrize("correct", [True, False])
def test_provided_plain_function_fixtures_are_callable(correct):
    candidate = (
        "def task_func(provided): return provided(3)"
        if correct
        else "def task_func(provided): return 0"
    )
    tests = """
import unittest
class TestCases(unittest.TestCase):
    def test_provided_function(self):
        def provided(value): return 7 * value
        self.assertEqual(task_func(provided), 21)
"""
    report = _run(candidate, tests=tests, timeout_s=10)
    assert report["passed"] is correct
    assert report["cause"] == ("all_pass" if correct else "test_failure")


@pytest.mark.parametrize("correct", [True, False])
def test_popen_test_patch_does_not_intercept_the_grader_worker_launch(correct):
    candidate = (
        "import subprocess\ndef task_func(): return subprocess.Popen(['script'], stdout=subprocess.PIPE).communicate()[0]"
        if correct
        else "def task_func(): return b'wrong'"
    )
    tests = """
import unittest
import subprocess
from unittest.mock import patch, Mock
class TestCases(unittest.TestCase):
    @patch('subprocess.Popen')
    def test_script(self, launch):
        child = Mock()
        child.communicate.return_value = (b'expected', b'')
        launch.return_value = child
        self.assertEqual(task_func(), b'expected')
        launch.assert_called_once_with(['script'], stdout=subprocess.PIPE)
"""
    report = _run(candidate, tests=tests, timeout_s=10)
    assert report["passed"] is correct
    assert report["cause"] == ("all_pass" if correct else "test_failure")


@pytest.mark.parametrize("correct", [True, False])
def test_point_cells_in_series_cross_provided_function_calls(correct):
    candidate = (
        """
import pandas as pd
from shapely.geometry import Point
def task_func(project): return project(pd.Series([Point(1, 2)], dtype=object)).iloc[0]
"""
        if correct
        else "def task_func(project): return None"
    )
    tests = """
import unittest
from shapely.geometry import Point
class TestCases(unittest.TestCase):
    def test_point(self):
        def project(series):
            return series.map(lambda point: Point(point.x + 3, point.y + 5))
        point = task_func(project)
        self.assertIsInstance(point, Point)
        self.assertEqual(tuple(point.coords), ((4.0, 7.0),))
"""
    report = _run(candidate, tests=tests, timeout_s=10)
    assert report["passed"] is correct
    assert report["cause"] == ("all_pass" if correct else "test_failure")


def test_point_transport_keeps_z_coordinates_and_srid_and_rejects_other_geometries():
    import shapely
    from shapely import wkb
    from shapely.geometry import LineString, Point

    spec = importlib.util.spec_from_file_location("point_transport_fixture", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = shapely.set_srid(Point(1, 2, 3), 4326)
    restored = module._materialize(module._transfer(original))
    assert tuple(restored.coords) == tuple(original.coords)
    assert shapely.get_srid(restored) == 4326
    with pytest.raises(module.CandidateError):
        module._materialize(("shapely_point", "not bytes"))
    with pytest.raises(module.CandidateError):
        module._materialize(("shapely_point", b"x" * (module.MAX_POINT_BYTES + 1)))
    with pytest.raises(module.CandidateError):
        module._materialize(("shapely_point", wkb.dumps(LineString([(0, 0), (1, 1)]))))


def test_path_descriptors_cannot_select_custom_constructors_or_pathlike_callbacks():
    spec = importlib.util.spec_from_file_location("path_transport_fixture", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for descriptor in (("CustomPath", "/tmp/file"), ("PosixPath", 7), ["PosixPath", "/tmp/file"]):
        with pytest.raises(module.CandidateError):
            module._materialize(("pathlib_path", descriptor))

    class CustomPath:
        def __fspath__(self):
            pytest.fail("a custom path callback must not run in the test supervisor")

        def __str__(self):
            pytest.fail("custom string conversion must not run in the test supervisor")

    original = CustomPath()
    encoded = module._transfer(original)
    assert encoded == ("reference", original)


@pytest.mark.parametrize("correct", [True, False])
def test_deque_results_preserve_native_equality_and_capacity(correct):
    candidate = (
        "from collections import deque\ndef task_func(): return deque([1, 2], maxlen=3)"
        if correct
        else "def task_func(): return [1, 2]"
    )
    tests = """
import unittest
from collections import deque
class TestCases(unittest.TestCase):
    def test_deque(self):
        value = task_func()
        self.assertIsInstance(value, deque)
        self.assertEqual(value, deque([1, 2], maxlen=3))
        value.append(3)
        value.append(4)
        self.assertEqual(list(value), [2, 3, 4])
"""
    report = _run(candidate, tests=tests, timeout_s=10)
    assert report["passed"] is correct
    assert report["cause"] == ("all_pass" if correct else "test_failure")


def test_deque_payloads_cannot_truncate_values_or_supply_invalid_capacity():
    spec = importlib.util.spec_from_file_location("deque_transport_fixture", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for value in [(-1, ()), (False, ()), (0, (("scalar", 1),)), (2, [1, 2])]:
        with pytest.raises(module.CandidateError):
            module._materialize(("deque", value))


@pytest.mark.parametrize("correct", [True, False])
def test_test_module_imports_reach_the_solution_without_assertions_or_private_globals(correct):
    candidate = (
        """
def task_func(values):
    return np.asarray(values).sum().item(), 'HIDDEN_EXPECTED' in globals(), 'TestCases' in globals()
"""
        if correct
        else "def task_func(values): return 0, True, True"
    )
    tests = """
import unittest
import numpy as np
HIDDEN_EXPECTED = 'test-only-value'
class TestCases(unittest.TestCase):
    def test_library_alias(self):
        self.assertEqual(task_func([1, 2, 3]), (6, False, False))
"""
    report = _run(candidate, tests=tests, timeout_s=10)
    assert report["passed"] is correct
    assert report["cause"] == ("all_pass" if correct else "test_failure")


def test_import_bindings_preserve_module_aliases_from_imports_and_local_scope():
    candidate = """
def task_func():
    return math_alias.sqrt(9), pure_path('a').name, 'local_alias' in globals()
"""
    tests = """
import unittest
import math as math_alias
from pathlib import PurePosixPath as pure_path
class TestCases(unittest.TestCase):
    def test_aliases(self):
        import fractions as local_alias
        self.assertEqual(task_func(), (3.0, 'a', False))
"""
    report = _run(candidate, tests=tests, timeout_s=10)
    assert report["passed"] is True and report["cause"] == "all_pass"


@pytest.mark.parametrize("correct", [True, False])
def test_plot_methods_receive_native_array_arguments(correct, matplotlib_cache, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(matplotlib_cache))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    candidate = (
        "import matplotlib.pyplot as plt\ndef task_func(): return plt.subplots()[1]"
        if correct
        else "def task_func(): return None"
    )
    tests = """
import unittest
import pandas as pd
import numpy as np
class TestCases(unittest.TestCase):
    def test_array_arguments(self):
        ax = task_func()
        x, y = np.array([1, 2, 3]), pd.Series([3, 1, 2])
        line, = ax.plot(x, y)
        np.testing.assert_array_equal(line.get_xdata(), x)
        np.testing.assert_array_equal(line.get_ydata(), y.to_numpy())
        counts, _, _ = ax.hist(pd.Series([1, 2, 2, 3]), bins=[0.5, 1.5, 2.5, 3.5])
        np.testing.assert_array_equal(counts, [1, 2, 1])
"""
    report = _run(candidate, tests=tests, timeout_s=20)
    assert report["passed"] is correct
    assert report["cause"] == ("all_pass" if correct else "test_failure")


@pytest.mark.parametrize("correct", [True, False])
def test_rsa_fixture_signatures_verify_with_the_original_key(correct):
    candidate = (
        """
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
def task_func(key, data):
    return key.sign(data, padding.PKCS1v15(), hashes.SHA256())
"""
        if correct
        else "def task_func(key, data): return b'wrong signature'"
    )
    tests = """
import unittest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
class TestCases(unittest.TestCase):
    def test_signature(self):
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        data = b'fixture input'
        signature = task_func(key, data)
        self.assertIsInstance(signature, bytes)
        key.public_key().verify(signature, data, padding.PKCS1v15(), hashes.SHA256())
"""
    report = _run(candidate, tests=tests, timeout_s=15)
    assert report["passed"] is correct
    assert report["cause"] == ("all_pass" if correct else "test_failure")


@pytest.mark.parametrize("correct", [True, False])
def test_function_inputs_preserve_names_signatures_and_lambda_defaults(correct):
    candidate = (
        """
import inspect
import types
def task_func(function):
    spec = inspect.getfullargspec(function)
    return function.__name__, spec.args, sum(isinstance(v, types.LambdaType) for v in spec.defaults or ())
"""
        if correct
        else "def task_func(function): return 'wrong', [], 0"
    )
    tests = """
import unittest
class TestCases(unittest.TestCase):
    def test_signature(self):
        def sample(x, y=lambda: 3): pass
        self.assertEqual(task_func(sample), ('sample', ['x', 'y'], 1))
    def test_keyword_only_signature(self):
        def sample(x, *, y=4): pass
        self.assertEqual(task_func(sample), ('sample', ['x'], 0))
"""
    report = _run(candidate, tests=tests, timeout_s=10)
    assert report["passed"] is correct
    assert report["cause"] == ("all_pass" if correct else "test_failure")


@pytest.mark.parametrize(
    "attribute", ["__globals__", "__code__", "__closure__", "__dict__", "__array_interface__"]
)
def test_function_metadata_does_not_grant_access_to_trusted_execution_state(attribute):
    candidate = f"def task_func(function): return function._request('get', {attribute!r})"
    tests = """
import unittest
HIDDEN_VALUE = 'trusted-only-value'
class TestCases(unittest.TestCase):
    def test_no_fixture_state_leak(self):
        def sample(x): return HIDDEN_VALUE
        self.assertEqual(task_func(sample), HIDDEN_VALUE)
"""
    report = _run(candidate, tests=tests, timeout_s=10)
    assert report["passed"] is False and report["cause"] == "test_failure"


def test_rsa_transport_rejects_unbounded_malformed_and_other_key_types():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    spec = importlib.util.spec_from_file_location("rsa_transport_fixture", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(module.UnsupportedResult):
        module._materialize(("rsa_private_key", "not bytes"))
    with pytest.raises(module.UnsupportedResult):
        module._materialize(("rsa_private_key", b"x" * (module.MAX_RSA_BYTES + 1)))
    with pytest.raises(ValueError):
        module._materialize(("rsa_private_key", b"not a DER key"))
    other_key = ec.generate_private_key(ec.SECP256R1())
    payload = other_key.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    with pytest.raises(module.UnsupportedResult):
        module._materialize(("rsa_private_key", payload))


@pytest.mark.parametrize("correct", [True, False])
def test_trusted_assertions_can_inspect_candidate_owned_private_fields(correct):
    rows = "[1, 2, 3]" if correct else "[]"
    candidate = f"class Result:\n    def __init__(self): self._rows = {rows}\ndef task_func(): return Result()"
    tests = """
import unittest
class TestCases(unittest.TestCase):
    def test_private_rows(self):
        result = task_func()
        self.assertEqual(result._rows, [1, 2, 3])
"""
    report = _run(candidate, tests=tests, timeout_s=10)
    assert report["passed"] is correct
    assert report["cause"] == ("all_pass" if correct else "test_failure")


@pytest.mark.parametrize("correct", [True, False])
def test_http_error_fixture_keeps_constructor_fields_and_can_be_reraised(correct):
    candidate = (
        """
import urllib.request
from urllib.error import HTTPError
def task_func(rethrow=False):
    try:
        urllib.request.urlopen('https://fixture.example')
    except HTTPError as error:
        if rethrow: raise
        return error.url, error.code, error.msg, dict(error.hdrs)
"""
        if correct
        else "def task_func(rethrow=False): return None"
    )
    tests = """
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
class TestCases(unittest.TestCase):
    def test_fields(self):
        with patch('urllib.request.urlopen', side_effect=HTTPError('https://fixture.example', 403, 'Forbidden', {'x-fixture': 'yes'}, None)):
            self.assertEqual(task_func(), ('https://fixture.example', 403, 'Forbidden', {'x-fixture': 'yes'}))
    def test_rethrow(self):
        with patch('urllib.request.urlopen', side_effect=HTTPError('https://fixture.example', 403, 'Forbidden', {'x-fixture': 'yes'}, None)):
            with self.assertRaises(HTTPError) as caught:
                task_func(True)
            self.assertEqual(caught.exception.code, 403)
            self.assertEqual(dict(caught.exception.hdrs), {'x-fixture': 'yes'})
"""
    report = _run(candidate, tests=tests, timeout_s=10)
    assert report["passed"] is correct
    assert report["cause"] == ("all_pass" if correct else "test_failure")


def test_function_metadata_allows_matplotlib_to_adapt_mocked_show(matplotlib_cache, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(matplotlib_cache))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    candidate = """
import inspect
import pandas as pd
import matplotlib.pyplot as plt
def task_func(function):
    count = len(inspect.getfullargspec(function).args)
    pd.DataFrame({'args': [count]}).plot(kind='bar')
    plt.show()
    return count
"""
    tests = """
import unittest
from unittest.mock import patch
class TestCases(unittest.TestCase):
    @patch('matplotlib.pyplot.show')
    def test_show(self, show):
        def sample(x, y=1): pass
        self.assertEqual(task_func(sample), 2)
        show.assert_called_once_with()
"""
    report = _run(candidate, tests=tests, reference=candidate, timeout_s=20)
    assert report["passed"] and report["cause"] == "all_pass"


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


@pytest.mark.parametrize("correct", [True, False])
def test_nested_connection_errors_preserve_constructor_fields_and_original_message(correct):
    candidate = (
        """
import socket
import requests
from urllib3.connection import HTTPSConnection
from urllib3.connectionpool import HTTPSConnectionPool
from urllib3.exceptions import MaxRetryError, NameResolutionError
def task_func():
    pool = HTTPSConnectionPool('fixture.example')
    connection = HTTPSConnection('fixture.example')
    reason = NameResolutionError('fixture.example', connection, socket.gaierror(-2, 'missing fixture'))
    error = MaxRetryError(pool, '/archive.zip', reason)
    error.args = ('original connection message',)
    raise requests.ConnectionError(error)
"""
        if correct
        else "def task_func(): raise ValueError('wrong exception')"
    )
    tests = """
import unittest
import requests
from urllib3.exceptions import MaxRetryError, NameResolutionError
class TestCases(unittest.TestCase):
    def test_exception(self):
        with self.assertRaises(requests.ConnectionError) as caught:
            task_func()
        error = caught.exception.args[0]
        self.assertIs(type(error), MaxRetryError)
        self.assertEqual(error.url, '/archive.zip')
        self.assertEqual(error.pool.host, 'fixture.example')
        self.assertEqual(str(error), 'original connection message')
        self.assertIs(type(error.reason), NameResolutionError)
        self.assertEqual(error.reason._host, 'fixture.example')
        self.assertEqual(error.reason.conn.host, 'fixture.example')
        self.assertEqual(error.reason._reason.errno, -2)
"""
    report = _run(candidate, tests=tests, timeout_s=10)
    assert report["passed"] is correct
    assert report["cause"] == ("all_pass" if correct else "test_failure")


def test_provided_connection_errors_keep_their_fields_when_caught_in_the_candidate():
    candidate = """
from urllib3.exceptions import NewConnectionError
def task_func(provided):
    try:
        provided()
    except NewConnectionError as error:
        return error.conn.host, error._message, str(error)
"""
    tests = """
import unittest
from urllib3.connection import HTTPSConnection
from urllib3.exceptions import NewConnectionError
class TestCases(unittest.TestCase):
    def test_exception(self):
        error = NewConnectionError(HTTPSConnection('fixture.example'), 'connection refused')
        error.args = ('original fixture message',)
        def provided(): raise error
        self.assertEqual(task_func(provided), ('fixture.example', 'connection refused', 'original fixture message'))
"""
    report = _run(candidate, tests=tests, timeout_s=10)
    assert report["passed"] and report["cause"] == "all_pass"


def test_exception_constructor_references_remain_restricted_to_exact_known_types():
    from urllib3.exceptions import MaxRetryError

    spec = importlib.util.spec_from_file_location("exception_transport_fixture", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class CustomRetryError(MaxRetryError):
        pass

    assert module._urllib3_error_fields(CustomRetryError) == ()
    with pytest.raises(module.CandidateError):
        module._make_exception(("builtins", "ValueError", ("tuple", (("reference", object()),))))
    with pytest.raises(module.CandidateError):
        module._make_exception(("builtins", "ValueError", ("tuple", ()), ("tuple", ())))
    with pytest.raises(module.CandidateError):
        module._make_exception(("untrusted_missing_module", "MaxRetryError", ("tuple", ())))


def test_model_import_aliases_cannot_replace_trusted_test_helpers():
    candidate = "from os import _exit as helper\ndef task_func(): return 3"
    report = _run(candidate, reference="import math as helper")
    assert report["passed"] is False and report["cause"] == "test_failure"


@pytest.mark.parametrize("correct", [True, False])
def test_socket_fixtures_do_not_replace_the_supervisor_transport(correct):
    candidate = (
        """
import socket
def task_func():
    try:
        with socket.socket() as channel:
            channel.connect(('fixture.example', 1234))
        return True
    except OSError:
        return False
"""
        if correct
        else "def task_func(): return False\n"
    )
    tests = """
import socket
import unittest
from unittest.mock import patch
class TestCases(unittest.TestCase):
    @patch('socket.socket')
    def test_open(self, mocked):
        self.assertTrue(task_func())
        mocked.return_value.__enter__.return_value.connect.assert_called_once_with(
            ('fixture.example', 1234)
        )
    @patch('socket.socket')
    def test_closed(self, mocked):
        mocked.return_value.__enter__.return_value.connect.side_effect = OSError
        self.assertFalse(task_func())
"""
    report = _run(candidate, tests=tests, timeout_s=10)
    assert report == {
        "passed": correct,
        "cause": "all_pass" if correct else "test_failure",
        "tests_run": 2,
        "tests_expected": 2,
    }


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


@pytest.mark.parametrize("kind", ["axes", "figure"])
def test_plot_return_types_resolve_without_library_hints(kind, matplotlib_cache, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(matplotlib_cache))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    candidate = f"""
import matplotlib.pyplot as plt
def task_func():
    figure, axes = plt.subplots()
    axes.set_title('example')
    return {"axes" if kind == "axes" else "figure"}
"""
    tests = f"""
import unittest
class TestCases(unittest.TestCase):
    def test_plot(self):
        result = task_func()
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure
        self.assertIsInstance(result, {"Axes" if kind == "axes" else "Figure"})
        axes = {"result" if kind == "axes" else "result.axes[0]"}
        self.assertEqual(axes.get_title(), 'example')
"""
    report = _run(candidate, tests=tests, timeout_s=15)
    assert report == {"passed": True, "cause": "all_pass", "tests_run": 1, "tests_expected": 1}


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


@pytest.mark.parametrize("wrong", [False, True])
def test_mixed_dataframe_cells_round_trip_and_wrong_answers_still_fail(wrong):
    tests = """
import unittest
import pandas as pd
import numpy as np
from decimal import Decimal
class TestCases(unittest.TestCase):
    def test_round_trip(self):
        values = [1, 'apple', {'b': [3, None]}, (4, 5), Decimal('1.20'),
                  b'bytes', None, pd.NA, np.int64(7)]
        frame = pd.DataFrame({'mixed': values, 'number': range(len(values))})
        frame.index = pd.Index([f'row-{i}' for i in range(len(values))], name='row')
        for original in (frame, frame.iloc[:0], frame['mixed']):
            actual = task_func(original)
            if isinstance(original, pd.Series):
                pd.testing.assert_series_equal(actual, original)
            else:
                pd.testing.assert_frame_equal(actual, original)
"""
    candidate = (
        "def task_func(frame):\n result = frame.copy(deep=True)\n"
        + (" if len(result): result.iloc[0] = 'wrong'\n" if wrong else "")
        + " return result\n"
    )
    report = _run(candidate, tests=tests, timeout_s=15)
    assert report["passed"] is not wrong
    assert report["cause"] == ("test_failure" if wrong else "all_pass")
    assert report["tests_run"] == report["tests_expected"] == 1


@pytest.mark.parametrize(
    "columns",
    [
        [(-1, ())],
        ((-1, ()),),
        ((True, ()),),
        ((0, []),),
        ((0, ()),),
        ((0, (("scalar", 1),)), (0, (("scalar", 2),))),
        ((1, (("scalar", 1),)),),
        ((0, (("reference", object()),)),),
    ],
)
def test_object_column_descriptors_cannot_replace_typed_columns_or_admit_references(columns):
    import pandas as pd

    spec = importlib.util.spec_from_file_location("object_transport_fixture", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = pd.DataFrame({"object": pd.Series([None], dtype=object), "number": [1]})
    with pytest.raises(module.CandidateError):
        module._restore_object_columns(frame, columns, allow_axes=False)


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "report,success",
    [
        ("", False),
        ("invalid json", True),
        ('{"passed":true,"cause":"all_pass"}', True),
        ('{"passed":true,"cause":"all_pass","tests_run":1,"tests_expected":2}', True),
        ('{"passed":true,"cause":"test_failure","tests_run":1,"tests_expected":1}', True),
        ('{"passed":true,"cause":"unknown"}', True),
    ],
)
async def test_supervisor_faults_stay_unavailable_without_inference_retries(report, success):
    from types import SimpleNamespace

    from yugos_evals.scorers.python_unittest import grade_unittest

    class Environment:
        calls = 0

        def as_type(self, _):
            raise TypeError

        async def exec(self, *args, **kwargs):
            self.calls += 1
            return SimpleNamespace(success=success, returncode=0 if success else 1, stdout=report)

    env = Environment()
    outcome = await grade_unittest(env, "def task_func(): return 2", TESTS)
    assert outcome.code.value == "grader_unavailable" and not outcome.settled
    assert env.calls == 1


@pytest.mark.asyncio
async def test_reference_preflight_requires_passing_gold_and_failing_control():
    from types import SimpleNamespace

    from yugos_evals.scorers.python_unittest import preflight_unittest

    class Environment:
        def __init__(self, wrong_passes):
            self.requests = []
            self.wrong_passes = wrong_passes

        def as_type(self, _):
            raise TypeError

        async def exec(self, *args, input, **kwargs):
            self.requests.append(json.loads(input))
            passed = len(self.requests) == 1 or self.wrong_passes
            report = dict(
                passed=passed,
                cause="all_pass" if passed else "test_failure",
                tests_run=1,
                tests_expected=1,
            )
            return SimpleNamespace(success=True, returncode=0, stdout=json.dumps(report))

    for wrong_passes in (False, True):
        env = Environment(wrong_passes)
        report = await preflight_unittest(env, "def task_func(): return 2", TESTS)
        assert report["passed"] is not wrong_passes
        assert len(env.requests) == 2
        assert all(r["tests"] == TESTS for r in env.requests)


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

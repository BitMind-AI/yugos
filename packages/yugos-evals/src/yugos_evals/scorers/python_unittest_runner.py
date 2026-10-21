"""Run trusted tests with Arrow values and RPyC references to candidate objects.

Candidate code runs in child processes. The supervisor owns the unittest result,
refuses reverse RPC requests, and never unpickles candidate data. The caller must
still provide an isolated sandbox with resource limits for the whole supervisor.
"""

import json
import os
import socket
import subprocess
import sys
import time
import unittest

import pyarrow as pa
import rpyc
from rpyc.core.channel import Channel
from rpyc.core.protocol import Connection
from rpyc.core.stream import SocketStream

MAX_RPC_BYTES = 1_000_000


class CandidateError(Exception):
    pass


class UnsupportedResult(CandidateError):
    pass


class DeadlineSocketStream(SocketStream):
    """Keep partial frames and blocked writes inside the original deadline."""

    def __init__(self, sock, deadline):
        super().__init__(sock)
        self.deadline = deadline

    def _set_timeout(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("candidate timeout")
        self.sock.settimeout(remaining)

    def read(self, count):
        chunks = []
        while count:
            self._set_timeout()
            chunk = self.sock.recv(min(count, self.MAX_IO_CHUNK))
            if not chunk:
                raise EOFError("candidate connection closed")
            chunks.append(chunk)
            count -= len(chunk)
        return b"".join(chunks)

    def write(self, data):
        while data:
            self._set_timeout()
            count = self.sock.send(data[: self.MAX_IO_CHUNK])
            if not count:
                raise EOFError("candidate connection closed")
            data = data[count:]


class BoundedChannel(Channel):
    """Bound RPyC frames and reject compressed frames from the candidate."""

    def __init__(self, stream):
        super().__init__(stream, compress=False)

    def send(self, data):
        if len(data) > MAX_RPC_BYTES:
            raise UnsupportedResult("RPC frame exceeds the value limit")
        return super().send(data)

    def recv(self):
        header = self.stream.read(self.FRAME_HEADER.size)
        length, compressed = self.FRAME_HEADER.unpack(header)
        if compressed or length > MAX_RPC_BYTES:
            raise CandidateError("invalid candidate RPC frame")
        data = self.stream.read(length + len(self.FLUSHER))
        if not data.endswith(self.FLUSHER):
            raise CandidateError("invalid candidate RPC frame terminator")
        return data[: -len(self.FLUSHER)]


class SupervisorConnection(Connection):
    """The candidate may reply but cannot invoke anything in the supervisor."""

    def _dispatch_request(self, seq, raw_args):
        raise CandidateError("candidate callbacks into the grader are forbidden")

    def sync_request(self, handler, *args):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            self.timeout_callback()
            raise TimeoutError("candidate timeout")
        self._config["sync_request_timeout"] = remaining
        try:
            return super().sync_request(handler, *args)
        except TimeoutError:
            self.timeout_callback()
            raise


def _has_extension(dtype):
    if isinstance(dtype, pa.ExtensionType):
        return True
    if pa.types.is_dictionary(dtype):
        return _has_extension(dtype.value_type)
    return any(_has_extension(dtype.field(index).type) for index in range(dtype.num_fields))


def _arrow_dataframe(payload):
    if type(payload) is not bytes or len(payload) > MAX_RPC_BYTES:
        raise UnsupportedResult("invalid Arrow payload")
    with pa.ipc.open_stream(pa.BufferReader(payload)) as reader:
        if any(_has_extension(field.type) for field in reader.schema):
            raise UnsupportedResult("Arrow extension types are not admitted")
        table = reader.read_all()
    frame = table.to_pandas()
    # Arrow infers numeric storage for numeric object columns. Its pandas
    # metadata retains that distinction, but to_pandas does not restore it.
    metadata = table.schema.pandas_metadata or {}
    for position, column in enumerate(metadata.get("columns", ())[: len(frame.columns)]):
        if column.get("numpy_type") == "object":
            frame.isetitem(position, frame.iloc[:, position].astype(object))
    return frame


def _transfer(value, *, allow_axes=True):
    """Use Arrow for DataFrames and native RPyC references for Matplotlib axes."""
    if type(value).__module__.startswith("pandas"):
        import pandas as pd

        if type(value) is pd.DataFrame:
            try:
                table = pa.Table.from_pandas(value, preserve_index=True)
                sink = pa.BufferOutputStream()
                with pa.ipc.new_stream(sink, table.schema) as writer:
                    writer.write_table(table)
                payload = sink.getvalue().to_pybytes()
                # Reject lossy conversion instead of changing what tests observe.
                pd.testing.assert_frame_equal(value, _arrow_dataframe(payload))
            except (pa.ArrowException, AssertionError, TypeError, ValueError) as exc:
                raise UnsupportedResult("DataFrame cannot round-trip through Arrow") from exc
            return ("arrow", payload)
    if type(value).__module__.startswith("matplotlib"):
        from matplotlib.axes import Axes

        if allow_axes and isinstance(value, Axes):
            return ("axes", value)
    if type(value) in (list, tuple, set):
        return (
            type(value).__name__,
            tuple(_transfer(item, allow_axes=allow_axes) for item in value),
        )
    if type(value) is dict:
        return (
            "dict",
            tuple(
                (_transfer(k, allow_axes=allow_axes), _transfer(v, allow_axes=allow_axes))
                for k, v in value.items()
            ),
        )
    if value is None or type(value) in (bool, int, float, str, bytes):
        return ("scalar", value)
    raise UnsupportedResult(type(value).__name__)


def _materialize(value, *, allow_axes=True):
    if type(value) is not tuple or len(value) != 2:
        raise CandidateError("invalid candidate return value")
    kind, data = value
    if kind == "arrow":
        return _arrow_dataframe(data)
    if kind == "axes" and allow_axes:
        from matplotlib.axes import Axes
        from rpyc.core.netref import BaseNetref

        if not isinstance(data, BaseNetref) or not isinstance(data, Axes):
            raise UnsupportedResult("invalid Matplotlib reference")
        return data
    if kind in ("list", "tuple", "set") and type(data) is tuple:
        return {"list": list, "tuple": tuple, "set": set}[kind](
            _materialize(item, allow_axes=allow_axes) for item in data
        )
    if kind == "dict" and type(data) is tuple:
        return {
            _materialize(k, allow_axes=allow_axes): _materialize(v, allow_axes=allow_axes)
            for k, v in data
        }
    if kind == "scalar" and (data is None or type(data) in (bool, int, float, str, bytes)):
        return data
    raise CandidateError("invalid candidate return value")


def _candidate_serve(sock, candidate):
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)

    class CandidateService(rpyc.Service):
        def exposed_invoke(self, args, kwargs):
            namespace = {"__name__": "__candidate__"}
            exec(compile(candidate, "<candidate>", "exec"), namespace)
            function = namespace.get("task_func")
            if not callable(function):
                raise CandidateError("missing task_func")
            try:
                return (
                    "value",
                    _transfer(
                        function(
                            *_materialize(args, allow_axes=False),
                            **_materialize(kwargs, allow_axes=False),
                        )
                    ),
                )
            except UnsupportedResult:
                return ("unsupported", None)

    # Public access is enabled only in the untrusted candidate process, for
    # native netrefs to plot methods/properties. No pickle or custom exceptions.
    channel = BoundedChannel(SocketStream(sock))
    conn = CandidateService._connect(
        channel,
        config={
            "allow_public_attrs": True,
            "allow_pickle": False,
            "import_custom_exceptions": False,
            "instantiate_custom_exceptions": False,
        },
    )
    try:
        conn.serve_all()
    except (EOFError, OSError):
        pass
    finally:
        conn.close()


def run(request):
    """Evaluate trusted unittest assertions against isolated candidate calls."""
    deadline = time.monotonic() + request["timeout_s"]
    children = []
    unsupported = False
    timed_out = False

    def mark_timeout():
        nonlocal timed_out
        timed_out = True

    def task_func(*args, **kwargs):
        nonlocal unsupported, timed_out
        if time.monotonic() >= deadline:
            timed_out = True
            raise CandidateError("candidate timeout")
        parent, child = socket.socketpair()
        # Only submitted code and the socket enter this fresh interpreter;
        # it inherits neither the supervisor's test data nor its call stack.
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-c",
                request["worker_source"],
                "--candidate",
                str(child.fileno()),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(child.fileno(),),
        )
        child.close()
        conn = SupervisorConnection(
            rpyc.VoidService(),
            BoundedChannel(DeadlineSocketStream(parent, deadline)),
            config={
                "allow_getattr": False,
                "allow_safe_attrs": False,
                "allow_exposed_attrs": False,
                "allow_public_attrs": False,
                "allow_pickle": False,
                "import_custom_exceptions": False,
                "instantiate_custom_exceptions": False,
                "sync_request_timeout": max(0.001, deadline - time.monotonic()),
            },
        )
        children.append((conn, process))
        conn.deadline = deadline
        conn.timeout_callback = mark_timeout
        try:
            process.stdin.write(json.dumps({"candidate": request["candidate"]}).encode("utf-8"))
            process.stdin.close()
            status, value = conn.root.invoke(
                _transfer(args, allow_axes=False),
                _transfer(kwargs, allow_axes=False),
            )
            if status == "unsupported":
                unsupported = True
                raise UnsupportedResult("unsupported candidate return type")
            if status != "value":
                raise CandidateError("invalid candidate response")
            return _materialize(value)
        except UnsupportedResult:
            unsupported = True
            raise
        except TimeoutError:
            timed_out = True
            raise CandidateError("candidate timeout") from None

    namespace = {"__name__": "__main__", "task_func": task_func}
    try:
        if "matplotlib" in request.get("libs", ()):
            import matplotlib.pyplot as plt

            namespace["plt"] = plt
        exec(compile(request["tests"], "<tests>", "exec"), namespace)
        test_case = namespace.get("TestCases")
        if not isinstance(test_case, type) or not issubclass(test_case, unittest.TestCase):
            return {"passed": False, "cause": "missing_test_cases"}
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_case)
        expected = suite.countTestCases()
        result = unittest.TestResult()
        suite.run(result)
        if unsupported:
            return {"passed": False, "cause": "unsupported_result"}
        if timed_out:
            return {"passed": False, "cause": "timeout"}
        passed = expected > 0 and result.testsRun == expected and result.wasSuccessful()
        return {
            "passed": passed,
            "cause": "all_pass" if passed else "test_failure",
            "tests_run": result.testsRun,
            "tests_expected": expected,
        }
    except BaseException as exc:
        return {"passed": False, "cause": "test_error", "error_type": type(exc).__name__}
    finally:
        for conn, process in children:
            # Reap before protocol close: an uncooperative candidate must not
            # prolong cleanup by withholding RPyC's close acknowledgement.
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
            try:
                conn.close()
            except (EOFError, OSError, TimeoutError, CandidateError):
                pass


if __name__ == "__main__":
    if sys.argv[1:2] == ["--candidate"]:
        payload = json.load(sys.stdin)
        _candidate_serve(socket.socket(fileno=int(sys.argv[2])), payload["candidate"])
    else:
        print(json.dumps(run(json.load(sys.stdin)), allow_nan=False), flush=True)

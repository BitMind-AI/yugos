"""Run trusted tests with Arrow values and RPyC references to candidate objects.

Candidate code runs in child processes. The supervisor owns the unittest result,
admits calls only to active test fixtures, and never unpickles candidate data.
The caller must still provide an isolated sandbox with resource limits.
"""

import ast
import base64
import hashlib
import importlib
import io
import json
import math
import os
import socket
import subprocess
import sys
import time
import unittest
import warnings
from _socket import socketpair as _transport_socketpair
from contextlib import ExitStack, redirect_stdout
from datetime import date, datetime, timedelta
from datetime import time as datetime_time
from decimal import Decimal

import pyarrow as pa
import rpyc
from rpyc.core import consts
from rpyc.core.channel import Channel
from rpyc.core.protocol import Connection
from rpyc.core.stream import SocketStream

MAX_RPC_BYTES = 16_000_000
MAX_RSA_BYTES = 16_384
FIXTURE_CALL = 101
_fixture_registry = None
_fixture_connection = None


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
    """Admit only bounded fixture operations, never general reverse RPC requests."""

    def _dispatch_request(self, seq, raw_args):
        handler, boxed = raw_args
        if handler != FIXTURE_CALL:
            self._check_fixture_request(handler, boxed)
            return super()._dispatch_request(seq, raw_args)
        if type(boxed) is not tuple or len(boxed) != 2:
            raise CandidateError("candidate callbacks into the grader are forbidden")
        args = self._unbox(boxed)
        if type(args) is not tuple or len(args) != 1:
            raise CandidateError("invalid fixture request")
        result = self.fixtures.dispatch(args[0])
        self._send(consts.MSG_REPLY, seq, self._box(result))

    def _box(self, obj):
        if getattr(self, "value_tag", None) and type(obj).__module__.startswith(
            ("numpy", "pandas")
        ):
            return super()._box((self.value_tag, _transfer(obj, allow_axes=False)))
        boxed = super()._box(obj)
        if boxed[0] == consts.LABEL_REMOTE_REF:
            self.fixtures.register(obj)
        return boxed

    def _unbox(self, package):
        if package[0] == consts.LABEL_REMOTE_REF and package[1][0].startswith(
            ("matplotlib.axes.", "matplotlib.figure.")
        ):
            # RPyC resolves proxy classes from already imported trusted modules.
            # Load these before creating a plot proxy, including when the task's
            # optional library hints do not mention Matplotlib.
            importlib.import_module("matplotlib.axes")
            importlib.import_module("matplotlib.figure")
        value = super()._unbox(package)
        if package[0] == consts.LABEL_LOCAL_REF and id(value) not in self.fixtures.identities:
            raise CandidateError("unregistered fixture reference")
        if package[0] == consts.LABEL_REMOTE_REF and not getattr(self, "_copying", False):
            name = package[1][0]
            if name in {
                "builtins.list",
                "builtins.dict",
                "builtins.set",
                "builtins.frozenset",
            } or name.startswith(("numpy.", "pandas.")):
                self._copying = True
                try:
                    value = _materialize(self.root.transfer(value))
                finally:
                    self._copying = False
        return value

    def _check_fixture_request(self, handler, boxed):
        if handler == consts.HANDLE_INSPECT:
            if boxed[0] != consts.LABEL_VALUE or len(boxed[1]) != 1:
                raise CandidateError("invalid fixture inspection")
            identity = boxed[1][0]
        else:
            allowed = {
                consts.HANDLE_GETATTR,
                consts.HANDLE_CALL,
                consts.HANDLE_CALLATTR,
                consts.HANDLE_REPR,
                consts.HANDLE_STR,
                consts.HANDLE_CMP,
                consts.HANDLE_HASH,
                consts.HANDLE_DEL,
                consts.HANDLE_BUFFITER,
                consts.HANDLE_CTXEXIT,
            }
            if handler not in allowed or boxed[0] != consts.LABEL_TUPLE:
                raise CandidateError("candidate callbacks into the grader are forbidden")
            first = boxed[1][0]
            if first[0] != consts.LABEL_LOCAL_REF:
                raise CandidateError("fixture object reference required")
            identity = first[1]
        try:
            obj = self._local_objects[identity]
        except KeyError:
            raise CandidateError("unknown fixture reference") from None
        if id(obj) not in self.fixtures.identities:
            raise CandidateError("unregistered fixture reference")
        if handler in {consts.HANDLE_GETATTR, consts.HANDLE_CALLATTR}:
            name_box = boxed[1][1]
            if name_box[0] != consts.LABEL_VALUE:
                raise CandidateError("invalid fixture attribute")
            name = name_box[1]
            # NumPy probes these absent protocols before iterating ordinary
            # containers. Let RPyC return AttributeError without opening private
            # attributes on fixtures that could actually implement a protocol.
            absent_array_protocol = (
                type(obj) in {list, dict, set, frozenset}
                and type(name) is str
                and name in {"__array__", "__array_interface__", "__array_struct__"}
            )
            if (
                type(name) is not str
                or name in self.fixtures._BLOCKED
                or (
                    name.startswith("_")
                    and name not in self.fixtures._SPECIAL
                    and not absent_array_protocol
                )
            ):
                raise CandidateError("fixture attribute is not available")

    def sync_request(self, handler, *args):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            self.timeout_callback()
            raise TimeoutError("candidate timeout")
        self._config["sync_request_timeout"] = remaining
        try:
            if getattr(self, "prepare_callback", None) and not getattr(self, "_preparing", False):
                self._preparing = True
                try:
                    self.prepare_callback()
                finally:
                    self._preparing = False
            return super().sync_request(handler, *args)
        except TimeoutError:
            self.timeout_callback()
            raise


class CandidateConnection(Connection):
    """Restore explicitly tagged trusted array arguments to candidate plot methods."""

    def _unbox(self, package):
        value = super()._unbox(package)
        if (
            type(value) is tuple
            and len(value) == 2
            and type(value[0]) is str
            and value[0] == getattr(self, "value_tag", None)
        ):
            return _materialize(value[1], allow_axes=False)
        return value


def _exception_record(error):
    from urllib.error import HTTPError

    args = (
        (error.msg, error.doc, error.pos)
        if isinstance(error, json.JSONDecodeError)
        else (error.url, error.code, error.msg, error.hdrs, error.fp)
        if isinstance(error, HTTPError)
        else error.args
    )
    return (type(error).__module__, type(error).__qualname__, _transfer(args))


def _make_exception(record, *, candidate=False):
    module_name, name, args = record
    module = importlib.import_module(module_name) if candidate else sys.modules.get(module_name)
    error_type = module
    for part in name.split("."):
        if part.startswith("_") or error_type is None:
            raise CandidateError("unavailable exception type")
        error_type = getattr(error_type, part, None)
    if not isinstance(error_type, type) or not issubclass(error_type, Exception):
        raise CandidateError("unavailable exception type")
    values = _materialize(args, allow_axes=False)
    return error_type(*values)


def _raise_exception(record, *, candidate=False):
    raise _make_exception(record, candidate=candidate)


class FixtureProxy:
    """A candidate-side handle to a particular test fixture, without grader access."""

    def __init__(self, identity, fixture_type=None):
        self.identity = identity
        self.fixture_type = fixture_type

    @property
    def __name__(self):
        if "_name_override" in self.__dict__:
            return self._name_override
        return self._request("get", "__name__")

    @__name__.setter
    def __name__(self, value):
        self._name_override = value

    @property
    def __signature__(self):
        if "_signature_override" in self.__dict__:
            return self._signature_override
        import inspect

        record = self._request("signature")

        def restore(value):
            return value[1] if value[0] else inspect.Parameter.empty

        return inspect.Signature(
            [
                inspect.Parameter(
                    p["name"],
                    p["kind"],
                    default=restore(p["default"]),
                    annotation=restore(p["annotation"]),
                )
                for p in record["parameters"]
            ],
            return_annotation=restore(record["return_annotation"]),
        )

    @__signature__.setter
    def __signature__(self, value):
        self._signature_override = value

    @property
    def __class__(self):
        return self.fixture_type or FixtureProxy

    def _request(self, operation, name="", args=(), kwargs=None):
        payload = (
            self.identity,
            operation,
            name,
            _transfer(args),
            _transfer(kwargs or {}),
        )
        status, value = _fixture_connection.sync_request(FIXTURE_CALL, payload)
        if status == "exception":
            _raise_exception(value, candidate=True)
        if status != "value":
            raise CandidateError("invalid fixture response")
        return _materialize(value, allow_axes=False)

    def __call__(self, *args, **kwargs):
        return self._request("call", args=args, kwargs=kwargs)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._request("get", name)

    def __enter__(self):
        return self._request("method", "__enter__")

    def __exit__(self, *args):
        # Exception objects do not cross the fixture boundary. Native mock-open
        # and connection fixtures only need the exception's absence/presence.
        if any(value is not None for value in args):
            return False
        return self._request("method", "__exit__", args=args)

    def __iter__(self):
        return self._request("method", "__iter__")

    def __next__(self):
        return self._request("method", "__next__")

    def __getitem__(self, key):
        return self._request("method", "__getitem__", args=(key,))

    def __len__(self):
        return self._request("method", "__len__")

    def __bool__(self):
        return self._request("truth")


class FixtureRegistry:
    """Track native test patches and serve only those fixtures to candidate calls."""

    _SPECIAL = {
        "__enter__",
        "__exit__",
        "__iter__",
        "__next__",
        "__getitem__",
        "__len__",
        "__bool__",
        "__call__",
        "__setitem__",
        "__contains__",
        "__eq__",
        "__ne__",
        "__str__",
        "__repr__",
    }
    _BLOCKED = {
        "side_effect",
        "return_value",
        "configure_mock",
        "attach_mock",
        "reset_mock",
        "mock_add_spec",
        "mock_calls",
        "method_calls",
        "call_args",
        "call_args_list",
    }
    _IO = {
        "read",
        "readline",
        "readlines",
        "write",
        "writelines",
        "seek",
        "tell",
        "flush",
        "close",
        "getvalue",
        "closed",
        "name",
        "mode",
    }

    def __init__(self):
        from unittest.mock import _patch

        self.objects = {}
        self.identities = {}
        self.methods = set()
        self.active = []
        self.enter = _patch.__enter__
        self.exit = _patch.__exit__

        def enter(patcher):
            result = self.enter(patcher)
            self.active.append(patcher)
            return result

        def exit(patcher, *args):
            try:
                return self.exit(patcher, *args)
            finally:
                if patcher in self.active:
                    self.active.remove(patcher)

        _patch.__enter__, _patch.__exit__ = enter, exit

    def close(self):
        from unittest.mock import _patch

        _patch.__enter__, _patch.__exit__ = self.enter, self.exit

    def register(self, value, *, method=False):
        identity = self.identities.get(id(value))
        if identity is None:
            if len(self.objects) >= 4096:
                raise UnsupportedResult("fixture handle limit exceeded")
            identity = len(self.objects) + 1
            self.objects[identity] = value
            self.identities[id(value)] = identity
        if method:
            self.methods.add(identity)
        return identity

    def manifest(self):
        from types import ModuleType

        result = []
        for patcher in self.active:
            target = patcher.target
            if isinstance(target, ModuleType):
                module, path = target.__name__, ()
            elif isinstance(target, type):
                module, path = target.__module__, tuple(target.__qualname__.split("."))
            else:
                raise UnsupportedResult("non-importable patch target")
            result.append(
                (module, path, patcher.attribute, _transfer(getattr(target, patcher.attribute)))
            )
        return tuple(result)

    def dispatch(self, request):
        from types import FunctionType
        from unittest.mock import Mock

        try:
            identity, operation, name, encoded_args, encoded_kwargs = request
            if type(identity) is not int or identity not in self.objects:
                raise CandidateError("unknown fixture")
            value = self.objects[identity]
            args = _materialize(encoded_args)
            kwargs = _materialize(encoded_kwargs)
            if type(args) is not tuple or type(kwargs) is not dict:
                raise CandidateError("invalid fixture arguments")
            if operation == "signature":
                if type(value) is not FunctionType:
                    raise AttributeError("__signature__")
                import inspect

                signature = inspect.signature(value)

                def record(v):
                    return (False, None) if v is inspect.Parameter.empty else (True, v)

                result = {
                    "parameters": [
                        {
                            "name": p.name,
                            "kind": int(p.kind),
                            "default": record(p.default),
                            "annotation": record(p.annotation),
                        }
                        for p in signature.parameters.values()
                    ],
                    "return_annotation": record(signature.return_annotation),
                }
            elif operation == "truth":
                result = bool(value)
            elif operation == "call":
                if not isinstance(value, Mock) and identity not in self.methods:
                    raise CandidateError("fixture is not callable")
                result = value(*args, **kwargs)
            elif operation in ("get", "method"):
                if (
                    type(name) is not str
                    or name in self._BLOCKED
                    or (
                        name.startswith("_")
                        and name not in self._SPECIAL
                        and not (type(value) is FunctionType and name == "__name__")
                    )
                ):
                    raise CandidateError("fixture attribute is not available")
                if (
                    not isinstance(value, Mock)
                    and not (
                        isinstance(value, io.IOBase) and (name in self._IO or name in self._SPECIAL)
                    )
                    and not (
                        operation == "method"
                        and name in {"__iter__", "__next__"}
                        and hasattr(value, "__next__")
                    )
                ):
                    if (
                        name.startswith("_")
                        and name not in self._SPECIAL
                        and not (type(value) is FunctionType and name == "__name__")
                    ):
                        raise CandidateError("fixture operation is not available")
                result = getattr(value, name)
                if operation == "method":
                    result = result(*args, **kwargs)
                elif callable(result) and not isinstance(result, Mock):
                    return ("value", ("fixture", self.register(result, method=True)))
            else:
                raise CandidateError("fixture operation is not available")
            return ("value", _transfer(result, allow_axes=False))
        except Exception as error:
            return ("exception", _exception_record(error))


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


def _zoneinfo_columns(frame):
    """Retain canonical ZoneInfo providers that Arrow otherwise replaces."""
    from zoneinfo import ZoneInfo

    import pandas as pd

    columns = []
    for position, dtype in enumerate(frame.dtypes):
        if not isinstance(dtype, pd.DatetimeTZDtype) or type(dtype.tz) is not ZoneInfo:
            continue
        key = dtype.tz.key
        if type(key) is not str or not 0 < len(key) <= 255:
            raise UnsupportedResult("timezone has no portable key")
        try:
            canonical = ZoneInfo(key)
        except (ValueError, KeyError) as exc:
            raise UnsupportedResult("timezone key cannot be restored") from exc
        if dtype.tz is not canonical:
            # A from_file timezone can claim another zone's name. A key alone
            # cannot prove its transition rules, so leave it unavailable.
            raise UnsupportedResult("noncanonical timezone cannot be restored")
        columns.append((position, key))
    return tuple(columns)


def _restore_zoneinfo(frame, columns):
    """Restore only bounded descriptors matching the decoded Arrow schema."""
    from zoneinfo import ZoneInfo

    import pandas as pd

    if type(columns) is not tuple or len(columns) > len(frame.columns):
        raise CandidateError("invalid timezone column metadata")
    seen = set()
    for item in columns:
        if type(item) is not tuple or len(item) != 2:
            raise CandidateError("invalid timezone column descriptor")
        position, key = item
        if (
            type(position) is not int
            or not 0 <= position < len(frame.columns)
            or position in seen
            or type(key) is not str
            or not 0 < len(key) <= 255
        ):
            raise CandidateError("invalid timezone column descriptor")
        dtype = frame.iloc[:, position].dtype
        if not isinstance(dtype, pd.DatetimeTZDtype) or str(dtype.tz) != key:
            raise CandidateError("timezone column differs from Arrow schema")
        try:
            zone = ZoneInfo(key)
        except (ValueError, KeyError) as exc:
            raise CandidateError("invalid timezone key") from exc
        seen.add(position)
        frame.isetitem(position, frame.iloc[:, position].dt.tz_convert(zone))
    return frame


def _restore_object_columns(frame, columns, *, allow_axes):
    """Restore bounded plain values without asking Arrow to infer mixed cells."""
    import numpy as np
    import pandas as pd

    if type(columns) is not tuple or len(columns) > len(frame.columns):
        raise CandidateError("invalid DataFrame object columns")
    seen = set()
    for item in columns:
        if type(item) is not tuple or len(item) != 2:
            raise CandidateError("invalid DataFrame object column")
        column, values = item
        if (
            type(column) is not int
            or not 0 <= column < len(frame.columns)
            or column in seen
            or not pd.api.types.is_object_dtype(frame.iloc[:, column].dtype)
            or type(values) is not tuple
            or len(values) != len(frame)
            or len(values) * 8 > MAX_RPC_BYTES
        ):
            raise CandidateError("invalid DataFrame object column")
        seen.add(column)
        restored = np.empty(len(values), dtype=object)
        for row, value in enumerate(values):
            restored[row] = _materialize(value, allow_axes=allow_axes)
        frame.isetitem(column, restored)
    return frame


def _transfer(value, *, allow_axes=True):
    """Use Arrow for DataFrames and native RPyC references for Matplotlib axes."""
    if isinstance(value, FixtureProxy):
        return ("fixture", value.identity)
    if isinstance(value, Exception):
        return ("exception", _exception_record(value))
    if _fixture_registry is not None:
        from collections.abc import Iterator
        from unittest.mock import Mock

        if isinstance(value, (Mock, io.IOBase, Iterator)):
            return (
                "fixture",
                (
                    _fixture_registry.register(value),
                    type(value).__module__,
                    type(value).__qualname__,
                ),
            )
    from types import FunctionType

    if _fixture_registry is not None and type(value) is FunctionType:
        return ("fixture", (_fixture_registry.register(value), "types", "FunctionType"))
    if type(value).__module__.startswith("numpy"):
        import numpy as np

        if isinstance(value, (np.ndarray, np.generic)):
            array = np.asarray(value)
            if array.dtype.hasobject:
                if array.dtype != object:
                    raise UnsupportedResult("structured object arrays are not supported")
                return (
                    "numpy_objects",
                    (array.shape, tuple(_transfer(v, allow_axes=allow_axes) for v in array.flat)),
                )
            stream = io.BytesIO()
            np.save(stream, array, allow_pickle=False)
            payload = stream.getvalue()
            if len(payload) > MAX_RPC_BYTES:
                raise UnsupportedResult("NumPy value exceeds the value limit")
            return ("numpy_scalar" if isinstance(value, np.generic) else "numpy", payload)
    if type(value).__module__.startswith("pandas"):
        import pandas as pd

        if isinstance(value, pd.Timestamp):
            return ("pandas_timestamp", value.isoformat())
        if value is pd.NaT:
            return ("pandas_nat", None)
        if value is pd.NA:
            return ("pandas_na", None)

        if type(value) is pd.Series:
            return (
                "series",
                (_transfer(value.to_frame(), allow_axes=allow_axes), _transfer(value.name)),
            )
        if type(value) is pd.DataFrame:
            try:
                zones = _zoneinfo_columns(value)
                positions = tuple(
                    index
                    for index, dtype in enumerate(value.dtypes)
                    if pd.api.types.is_object_dtype(dtype)
                )
                if len(value) * len(positions) * 8 > MAX_RPC_BYTES:
                    raise UnsupportedResult("DataFrame object cells exceed the value limit")
                object_columns = tuple(
                    (
                        index,
                        tuple(_transfer(v, allow_axes=False) for v in value.iloc[:, index]),
                    )
                    for index in positions
                )
                portable = value.copy(deep=False)
                for index, _ in object_columns:
                    portable.isetitem(index, pd.Series([None] * len(value), dtype=object).array)
                table = pa.Table.from_pandas(portable, preserve_index=True)
                sink = pa.BufferOutputStream()
                with pa.ipc.new_stream(sink, table.schema) as writer:
                    writer.write_table(table)
                payload = sink.getvalue().to_pybytes()
                # Reject lossy conversion instead of changing what tests observe.
                pd.testing.assert_frame_equal(
                    value,
                    _restore_object_columns(
                        _restore_zoneinfo(_arrow_dataframe(payload), zones),
                        object_columns,
                        allow_axes=False,
                    ),
                )
            except (
                pa.ArrowException,
                AssertionError,
                TypeError,
                ValueError,
                CandidateError,
            ) as exc:
                raise UnsupportedResult("DataFrame cannot round-trip through Arrow") from exc
            return ("dataframe", (payload, object_columns, zones))
    if type(value).__module__.startswith("cryptography"):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

        if isinstance(value, RSAPrivateKey):
            payload = value.private_bytes(
                serialization.Encoding.DER,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
            if len(payload) > MAX_RSA_BYTES:
                raise UnsupportedResult("RSA fixture exceeds the value limit")
            return ("rsa_private_key", payload)
    if type(value).__module__.startswith("matplotlib"):
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure

        if allow_axes and isinstance(value, (Axes, Figure)):
            return ("axes", value)
    if isinstance(value, str):
        return ("scalar", str(value))
    if type(value) is Decimal:
        return ("decimal", str(value))
    from collections import Counter

    if type(value) is Counter:
        return ("counter", _transfer(dict(value), allow_axes=allow_axes))
    if type(value) in (date, datetime, datetime_time):
        return (type(value).__name__, value.isoformat())
    if type(value) is timedelta:
        return ("timedelta", (value.days, value.seconds, value.microseconds))
    if type(value) in (list, tuple, set, frozenset):
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
    if value is None or type(value) in (bool, int, float, complex, str, bytes):
        return ("scalar", value)
    if _fixture_registry is not None:
        return ("fixture", _fixture_registry.register(value))
    if allow_axes:
        return ("reference", value)
    raise UnsupportedResult(type(value).__name__)


def _materialize(value, *, allow_axes=True):
    if type(value) is not tuple or len(value) != 2:
        raise CandidateError("invalid candidate return value")
    kind, data = value
    if kind == "fixture" and type(data) in {int, tuple}:
        identity = data if type(data) is int else data[0]
        if _fixture_registry is not None:
            if identity not in _fixture_registry.objects:
                raise CandidateError("unknown fixture")
            return _fixture_registry.objects[identity]
        fixture_type = None
        if type(data) is tuple:
            try:
                fixture_type = importlib.import_module(data[1])
                for name in data[2].split("."):
                    fixture_type = getattr(fixture_type, name)
                if not isinstance(fixture_type, type):
                    fixture_type = None
            except (ImportError, AttributeError):
                fixture_type = None
        return FixtureProxy(identity, fixture_type)
    if kind == "rsa_private_key":
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

        if type(data) is not bytes or not 0 < len(data) <= MAX_RSA_BYTES:
            raise UnsupportedResult("invalid RSA fixture payload")
        key = serialization.load_der_private_key(data, password=None)
        if not isinstance(key, RSAPrivateKey):
            raise UnsupportedResult("fixture is not an RSA private key")
        return key
    if kind == "exception":
        return _make_exception(data, candidate=_fixture_registry is None)
    if kind in {"pandas_timestamp", "pandas_na", "pandas_nat"}:
        import pandas as pd

        return (
            pd.Timestamp(data)
            if kind == "pandas_timestamp"
            else pd.NA
            if kind == "pandas_na"
            else pd.NaT
        )
    if kind in ("numpy", "numpy_scalar"):
        import numpy as np

        if type(data) is not bytes or len(data) > MAX_RPC_BYTES:
            raise CandidateError("invalid NumPy payload")
        stream = io.BytesIO(data)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise CandidateError("unsupported NumPy file version")
        if (
            dtype.hasobject
            or len(shape) > 16
            or any(type(n) is not int or n < 0 for n in shape)
            or math.prod(shape) * dtype.itemsize != len(data) - stream.tell()
        ):
            raise CandidateError("invalid NumPy shape or dtype")
        array = np.load(io.BytesIO(data), allow_pickle=False)
        return array[()] if kind == "numpy_scalar" else array
    if kind == "numpy_objects":
        import numpy as np

        shape, items = data
        if (
            type(shape) is not tuple
            or any(type(n) is not int or n < 0 for n in shape)
            or math.prod(shape) != len(items)
            or len(items) * 8 > MAX_RPC_BYTES
        ):
            raise CandidateError("invalid object array")
        array = np.empty(shape, dtype=object)
        for index, item in enumerate(items):
            array.flat[index] = _materialize(item, allow_axes=allow_axes)
        return array
    if kind == "series":
        frame, name = data
        result = _materialize(frame, allow_axes=allow_axes).iloc[:, 0]
        result.name = _materialize(name, allow_axes=allow_axes)
        return result
    if kind in ("date", "datetime", "time") and type(data) is str:
        return {"date": date, "datetime": datetime, "time": datetime_time}[kind].fromisoformat(data)
    if kind == "timedelta" and type(data) is tuple and len(data) == 3:
        return timedelta(days=data[0], seconds=data[1], microseconds=data[2])
    if kind == "decimal" and type(data) is str:
        return Decimal(data)
    if kind == "counter":
        from collections import Counter

        return Counter(_materialize(data, allow_axes=allow_axes))
    if kind == "arrow":
        return _arrow_dataframe(data)
    if kind == "dataframe":
        if type(data) is not tuple or len(data) not in (2, 3):
            raise CandidateError("invalid DataFrame payload")
        payload, columns = data[:2]
        zones = data[2] if len(data) == 3 else ()
        frame = _restore_zoneinfo(_arrow_dataframe(payload), zones)
        return _restore_object_columns(frame, columns, allow_axes=False)
    if kind == "axes" and allow_axes:
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure
        from rpyc.core.netref import BaseNetref

        if not isinstance(data, BaseNetref) or not isinstance(data, (Axes, Figure)):
            raise UnsupportedResult("invalid Matplotlib reference")
        return data
    if kind == "reference" and allow_axes:
        from rpyc.core.netref import BaseNetref

        if not isinstance(data, BaseNetref):
            raise CandidateError("invalid candidate object reference")
        return data
    if kind in ("list", "tuple", "set", "frozenset") and type(data) is tuple:
        return {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}[kind](
            _materialize(item, allow_axes=allow_axes) for item in data
        )
    if kind == "dict" and type(data) is tuple:
        return {
            _materialize(k, allow_axes=allow_axes): _materialize(v, allow_axes=allow_axes)
            for k, v in data
        }
    if kind == "scalar" and (data is None or type(data) in (bool, int, float, complex, str, bytes)):
        return data
    raise CandidateError("invalid candidate return value")


def _candidate_serve(sock, candidate, value_tag):
    global _fixture_connection
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)

    originals = {}
    namespace = {"__name__": "__candidate__"}
    loaded = False

    class CandidateService(rpyc.Service):
        _protocol = CandidateConnection

        def exposed_transfer(self, value):
            return _transfer(value)

        def exposed_plot(self, name):
            import matplotlib.pyplot as plt

            if name not in {"gca", "gcf", "get_fignums"}:
                raise CandidateError("invalid plot query")
            return _transfer(getattr(plt, name)())

        def exposed_configure(self, patches=(), django_configured=False):
            if django_configured:
                from django.conf import settings

                if not settings.configured:
                    settings.configure()
            if not loaded:
                return
            for (module, path, attribute), original in originals.items():
                target = namespace if module == "__main__" else importlib.import_module(module)
                for name in path:
                    target = getattr(target, name)
                if isinstance(target, dict):
                    target[attribute] = original
                else:
                    setattr(target, attribute, original)
            originals.clear()
            for module, path, attribute, value in patches:
                target = namespace if module == "__main__" else importlib.import_module(module)
                for name in path:
                    target = getattr(target, name)
                original = (
                    target.get(attribute)
                    if isinstance(target, dict)
                    else getattr(target, attribute)
                )
                originals.setdefault((module, path, attribute), original)
                if isinstance(target, dict):
                    target[attribute] = _materialize(value, allow_axes=False)
                else:
                    setattr(target, attribute, _materialize(value, allow_axes=False))

        def exposed_invoke(self, args, kwargs, patches=(), seed_state=None):
            nonlocal loaded
            import random

            if seed_state is not None:
                state = _materialize(seed_state, allow_axes=False)
                random.setstate(state[0])
                if state[1] is not None:
                    import numpy as np

                    np.random.set_state(state[1])
            exec(compile(candidate, "<candidate>", "exec"), namespace)
            loaded = True
            self.exposed_configure(patches)
            function = namespace.get("task_func")
            if not callable(function):
                raise CandidateError("missing task_func")
            with warnings.catch_warnings(record=True) as caught:
                try:
                    outcome = (
                        "value",
                        _transfer(
                            function(
                                *_materialize(args, allow_axes=False),
                                **_materialize(kwargs, allow_axes=False),
                            )
                        ),
                    )
                except UnsupportedResult:
                    outcome = ("unsupported", None)
                except Exception as error:
                    outcome = ("exception", _exception_record(error))
            numpy_state = (
                sys.modules["numpy.random"].get_state() if "numpy.random" in sys.modules else None
            )
            return (
                *outcome,
                _transfer(
                    (
                        random.getstate(),
                        numpy_state,
                        [
                            (str(w.message), w.category.__module__, w.category.__qualname__)
                            for w in caught
                        ],
                    )
                ),
            )

    # Trusted assertions may inspect candidate-owned private attributes. These
    # permissions apply only in the untrusted process; supervisor callbacks
    # still pass the strict fixture guards. No pickle or custom exceptions.
    channel = BoundedChannel(SocketStream(sock))
    conn = CandidateService._connect(
        channel,
        config={
            "allow_public_attrs": True,
            "allow_all_attrs": True,
            "allow_setattr": True,
            "allow_delattr": True,
            "allow_pickle": False,
            "import_custom_exceptions": False,
            "instantiate_custom_exceptions": False,
        },
    )
    conn.value_tag = value_tag
    _fixture_connection = conn
    try:
        conn.serve_all()
    except (EOFError, OSError):
        pass
    finally:
        conn.close()


def _test_namespace(reference, libraries, task_func):
    """Restore trusted reference setup while keeping the candidate call isolated."""
    statements = [
        node
        for node in ast.parse(reference).body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.ClassDef))
        or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name != "task_func"
    ]
    namespace = {"__name__": "__reference__", "task_func": task_func}
    setup = ast.Module(body=statements, type_ignores=[])
    exec(compile(setup, "<reference-setup>", "exec"), namespace)
    namespace.update(__name__="__main__", task_func=task_func)
    return namespace


def _offline_http(stack, fixtures):
    """Supply declared HTTP response snapshots through ordinary client APIs."""
    if not fixtures:
        return
    from email.message import Message
    from unittest.mock import patch
    from urllib.error import URLError
    from urllib.response import addinfourl

    bodies = {}
    for url, fixture in fixtures.items():
        body = base64.b64decode(fixture["body_base64"], validate=True)
        if len(body) > MAX_RPC_BYTES or hashlib.sha256(body).hexdigest() != fixture["sha256"]:
            raise ValueError("invalid HTTP response snapshot")
        bodies[url] = body

    def urlopen(url, *args, **kwargs):
        url = getattr(url, "full_url", url)
        if url not in bodies:
            raise URLError("URL is not present in the frozen HTTP fixtures")
        fixture = fixtures[url]
        headers = Message()
        for key, value in fixture.get("headers", {}).items():
            headers[key] = value
        return addinfourl(io.BytesIO(bodies[url]), headers, url, fixture.get("status", 200))

    stack.enter_context(patch("urllib.request.urlopen", side_effect=urlopen))
    try:
        import requests
    except ImportError:
        return

    def request(method, url, *args, **kwargs):
        if url not in bodies or method.upper() != "GET":
            raise requests.exceptions.ConnectionError(
                "URL is not present in the frozen HTTP fixtures"
            )
        fixture = fixtures[url]
        response = requests.Response()
        response.url = url
        response.status_code = fixture.get("status", 200)
        response.headers.update(fixture.get("headers", {}))
        response._content = bodies[url]
        response.encoding = requests.utils.get_encoding_from_headers(response.headers)
        return response

    stack.enter_context(patch("requests.sessions.Session.request", side_effect=request))


def run(request):
    """Evaluate trusted unittest assertions against isolated candidate calls."""
    global _fixture_registry
    deadline = time.monotonic() + request["timeout_s"]
    children = []
    unsupported = False
    timed_out = False
    fixtures = FixtureRegistry()
    _fixture_registry = fixtures
    setup = ExitStack()
    original_main = sys.modules.get("__main__")
    last_connection = None

    def mark_timeout():
        nonlocal timed_out
        timed_out = True

    def task_func(*args, **kwargs):
        nonlocal unsupported, timed_out, last_connection
        if time.monotonic() >= deadline:
            timed_out = True
            raise CandidateError("candidate timeout")
        # Tests may mock socket.socket. The supervisor's IPC must remain real
        # while those mocks are forwarded to the candidate as test fixtures.
        parent, child = _transport_socketpair()
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
        conn.fixtures = fixtures
        conn.value_tag = os.urandom(32).hex()
        last_connection = conn
        conn._config["allow_getattr"] = True
        conn._config["allow_safe_attrs"] = True
        conn._config["allow_public_attrs"] = True

        def prepare_call():
            module = sys.modules.get("django.conf")
            configured = bool(module is not None and module.settings.configured)
            conn.root.configure(fixtures.manifest(), configured)

        conn.prepare_callback = prepare_call
        try:
            process.stdin.write(
                json.dumps({"candidate": request["candidate"], "value_tag": conn.value_tag}).encode(
                    "utf-8"
                )
            )
            process.stdin.close()
            import random

            numpy_state = (
                sys.modules["numpy.random"].get_state() if "numpy.random" in sys.modules else None
            )
            status, value, context = conn.root.invoke(
                _transfer(args, allow_axes=False),
                _transfer(kwargs, allow_axes=False),
                fixtures.manifest(),
                _transfer((random.getstate(), numpy_state), allow_axes=False),
            )
            random_state, numpy_state, warning_records = _materialize(context, allow_axes=False)
            random.setstate(random_state)
            if numpy_state is not None:
                import numpy as np

                np.random.set_state(numpy_state)
            for message, module_name, name in warning_records:
                module = sys.modules.get(module_name)
                category = getattr(module, name, UserWarning) if module is not None else UserWarning
                if not isinstance(category, type) or not issubclass(category, Warning):
                    category = UserWarning
                warnings.warn(message, category, stacklevel=2)
            if status == "unsupported":
                unsupported = True
                raise UnsupportedResult("unsupported candidate return type")
            if status == "exception":
                _raise_exception(value)
            if status != "value":
                raise CandidateError("invalid candidate response")
            return _materialize(value)
        except UnsupportedResult:
            unsupported = True
            raise
        except TimeoutError:
            timed_out = True
            raise CandidateError("candidate timeout") from None

    try:
        _offline_http(setup, request.get("http_fixtures", {}))
        namespace = _test_namespace(
            request.get("reference", ""), request.get("libs", ()), task_func
        )
        from types import ModuleType

        test_module = ModuleType("__main__")
        test_module.__dict__.update(namespace)
        sys.modules["__main__"] = test_module
        namespace = test_module.__dict__
        if "matplotlib" in request.get("libs", ()):
            import matplotlib.pyplot as plt

            namespace["plt"] = plt
        exec(compile(request["tests"], "<tests>", "exec"), namespace)
        if "matplotlib.pyplot" in sys.modules:
            pyplot = sys.modules["matplotlib.pyplot"]
            view = ModuleType(pyplot.__name__)
            view.__dict__.update(pyplot.__dict__)

            def query_plot(name):
                def query():
                    if last_connection is None:
                        return getattr(pyplot, name)()
                    return _materialize(last_connection.root.plot(name))

                return query

            for name in ("gca", "gcf", "get_fignums"):
                setattr(view, name, query_plot(name))
            for name, value in tuple(namespace.items()):
                if value is pyplot:
                    namespace[name] = view
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
        passed = (
            expected > 0
            and result.testsRun == expected
            and result.wasSuccessful()
            and not result.skipped
            and not result.expectedFailures
        )
        return {
            "passed": passed,
            "cause": "all_pass" if passed else "test_failure",
            "tests_run": result.testsRun,
            "tests_expected": expected,
        }
    except BaseException as exc:
        return {"passed": False, "cause": "test_error", "error_type": type(exc).__name__}
    finally:
        if original_main is not None:
            sys.modules["__main__"] = original_main
        setup.close()
        fixtures.close()
        _fixture_registry = None
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
        _candidate_serve(
            socket.socket(fileno=int(sys.argv[2])), payload["candidate"], payload["value_tag"]
        )
    else:
        request = json.load(sys.stdin)
        with redirect_stdout(sys.stderr):
            report = run(request)
        print(json.dumps(report, allow_nan=False), flush=True)

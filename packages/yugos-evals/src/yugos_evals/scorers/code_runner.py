"""Run one candidate test case inside the sample's sandbox (standard library only).

`code_tests_scorer` passes this file's source to `python3 -I -c` in the sandbox
and one JSON request on stdin: the candidate code, entry point and one test
input, never an expected output. The last stdout line is a JSON report of what
the candidate produced (its stdout, or a function's return value) or how it
failed. The report is only data: a candidate controls this process and could
forge it, so correctness is decided outside the sandbox by comparing outputs.

Execution rules are ported unchanged from Yugos's LiveCodeBench worker: the
benchmark preamble, independent text and binary stdin, `__main__` programs,
and function selection by entry point, `Solution` method or last callable.
"""

import contextlib
import io
import json
import sys

PROBE = "yugos-evals-code-runner-v1"
MEMORY_MB = 3072
CPU_SECONDS = 8

_PREAMBLE = (
    "from string import *\nfrom re import *\nfrom datetime import *\n"
    "from collections import *\nfrom heapq import *\nfrom bisect import *\n"
    "from copy import *\nfrom math import *\nfrom random import *\n"
    "from statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\n"
    "from json import *\nfrom builtins import *\nfrom typing import *\n"
    "import string\nimport re\nimport datetime\nimport collections\n"
    "import heapq\nimport bisect\nimport copy\nimport math\nimport random\n"
    "import statistics\nimport itertools\nimport functools\nimport operator\n"
    "import io\nimport sys\nimport json\n"
    "sys.setrecursionlimit(50000)\n"
)


def _limits():
    try:
        import resource

        memory = MEMORY_MB * 1024**2
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass


def _resolve_fn(ns, entry_point):
    """Top-level entry point, else a `Solution` method, else the last callable."""
    if entry_point and callable(ns.get(entry_point)):
        return ns[entry_point], entry_point
    sol = ns.get("Solution")
    if isinstance(sol, type):
        try:
            inst = sol()
        except Exception:
            inst = None
        if inst is not None and entry_point and callable(getattr(inst, entry_point, None)):
            return getattr(inst, entry_point), entry_point
        methods = (
            [
                getattr(inst, k)
                for k in dir(inst)
                if not k.startswith("_") and callable(getattr(inst, k, None))
            ]
            if inst is not None
            else []
        )
        if len(methods) == 1:
            return methods[0], entry_point
    fns = {k: v for k, v in ns.items() if callable(v) and not k.startswith("_") and k != "Solution"}
    if fns:
        k = list(fns)[-1]
        return fns[k], k
    return None, entry_point


def _functional_args(raw_input):
    """JSON values per line, else the whole input as JSON, else the raw text."""
    lines = [ln for ln in (raw_input or "").split("\n") if ln.strip()]
    if not lines:
        return []
    try:
        return [json.loads(ln) for ln in lines]
    except Exception:
        try:
            parsed = json.loads(raw_input)
        except Exception:
            return [raw_input]
        return parsed if isinstance(parsed, list) else [parsed]


class _Stdin:
    """Independent text and binary stdin cursors."""

    def __init__(self, text):
        raw = text or ""
        self._s = io.StringIO(raw)
        self.buffer = io.BytesIO(raw.encode("utf-8"))

    def read(self, *args):
        return self._s.read(*args)

    def readline(self, *args):
        return self._s.readline(*args)

    def readlines(self, *args):
        return self._s.readlines(*args)

    def __iter__(self):
        return self._s


def _source(code):
    return compile(_PREAMBLE + "\n" + code, "<candidate>", "exec")


def run_functional(code, entry_point, raw_input, report):
    output = io.StringIO()
    namespace = {"__name__": "_cand"}
    args = _functional_args(raw_input)
    try:
        # One capture spans loading, selection and the call, so stream
        # reassignment by the candidate survives between those steps.
        with contextlib.redirect_stdout(output):
            exec(_source(code), namespace)
            fn, picked = _resolve_fn(namespace, entry_point)
            if not callable(fn):
                report({"ok": False, "reason": "no_callable_defined"})
                return
            value = fn(*args)
    except BaseException as exc:
        report({"ok": False, "reason": f"exec_error: {type(exc).__name__}: {exc}"})
        return
    if isinstance(value, tuple):
        value = list(value)
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        report({"ok": False, "reason": f"result_not_json: {type(exc).__name__}: {exc}"})
        return
    report({"ok": True, "value": value, "entry_point": picked})


def run_stdio(code, raw_input, report):
    output = io.StringIO()
    real_stdin = sys.stdin
    try:
        sys.stdin = _Stdin(raw_input or "")
        with contextlib.redirect_stdout(output):
            exec(_source(code), {"__name__": "__main__"})
    except SystemExit:
        pass
    except BaseException as exc:
        report({"ok": False, "reason": f"exec_error: {type(exc).__name__}: {exc}"})
        return
    finally:
        sys.stdin = real_stdin
    report({"ok": True, "stdout": output.getvalue()})


def main():
    real_stdout = sys.stdout

    def report(message):
        real_stdout.write("\n" + json.dumps(message) + "\n")
        real_stdout.flush()

    request = json.loads(sys.stdin.read())
    op = request.get("op")
    if op == "probe":
        report({"ok": True, "probe": PROBE})
        return
    _limits()
    code, raw_input = request.get("code") or "", request.get("input") or ""
    if op == "functional":
        run_functional(code, str(request.get("entry_point") or ""), raw_input, report)
    else:
        run_stdio(code, raw_input, report)


if __name__ == "__main__":
    main()

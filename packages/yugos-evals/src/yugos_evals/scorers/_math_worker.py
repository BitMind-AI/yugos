"""One trusted math comparison; never import or execute candidate programs."""

from __future__ import annotations

import contextlib
import json
import sys
from dataclasses import asdict


def main():
    # Linux address-space limits cover native SymPy allocations too. macOS
    # reserves a much larger virtual address space before loading the parser.
    # The parent enforces the wall deadline on every supported platform.
    if sys.platform != "win32":
        import resource

        if sys.platform == "linux":
            memory = 1024**3
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    request = json.load(sys.stdin)
    # Parser diagnostics can contain reference/response text. Only the result
    # crosses stdout; the caller discards stderr and exposes no exception text.
    with contextlib.redirect_stdout(sys.stderr):
        from ._math_compare import grade_math, reference_exclusion

        if request["operation"] == "grade":
            result = asdict(grade_math(request["gold"], request["text"]))
        elif request["operation"] == "reference":
            result = reference_exclusion(request["gold"])
        elif request["operation"] == "equivalence":
            from .math_equivalence import _equivalent

            result = _equivalent(request["gold"], request["text"], request["problem"])
        else:
            raise ValueError("unknown math operation")
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()

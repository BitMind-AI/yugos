"""The parts of SWE-bench's harness that its vendored log parsers depend on.

The parsers in this package are SWE-bench 5.0.2's `swebench.harness.log_parsers`
(MIT, see SWEBENCH_LICENSE) with imports pointed here. They read only a test
spec's `instance_id` and `version`.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import NamedTuple


class TestStatus(Enum):
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    XFAIL = "XFAIL"


class TestSpec(NamedTuple):
    instance_id: str
    version: str


def ansi_escape(text: str) -> str:
    return re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])").sub("", text)

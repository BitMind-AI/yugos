"""Give symbolic math a process lifetime independent of the Inspect event loop."""

from __future__ import annotations

import json
import subprocess
import sys

TIMEOUT_S = 30
WORKER_MODULE = "yugos_evals.scorers._math_worker"


class MathWorkerFailed(Exception):
    def __init__(self, cause: str):
        self.cause = cause
        super().__init__(cause)


def compare(operation: str, gold: str, text: str = ""):
    """Reap a slow or crashed worker; never turn its missing answer into a zero."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", WORKER_MODULE],
            input=json.dumps({"operation": operation, "gold": gold, "text": text}),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise MathWorkerFailed("timeout") from None
    except OSError:
        raise MathWorkerFailed("unavailable") from None
    if result.returncode != 0:
        raise MathWorkerFailed("process_failed")
    try:
        value = json.loads(result.stdout)
        if operation == "grade":
            if not isinstance(value, dict):
                raise ValueError
        elif operation == "reference":
            if value is not None and not isinstance(value, str):
                raise ValueError
        else:
            raise ValueError
        return value
    except (ValueError, TypeError):
        raise MathWorkerFailed("invalid_result") from None

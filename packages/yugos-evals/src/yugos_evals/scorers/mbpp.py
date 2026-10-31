"""Native MBPP verification with the declared test environment."""

import ast
from copy import copy

from inspect_ai.model import ModelOutput
from inspect_ai.scorer import accuracy, scorer, stderr
from inspect_evals.mbpp.mbpp import find_code, verify

GRADER_VERSION = 1


def _test_environment(code: str, imports: str) -> str:
    """Insert test imports after a docstring and future imports, before executable code."""
    if not imports:
        return code
    try:
        body = ast.parse(code).body
    except (SyntaxError, ValueError, RecursionError):
        return code
    offset = 0
    encoded = code.encode()
    lines = encoded.splitlines(keepends=True)
    for index, node in enumerate(body):
        docstring = (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        if not docstring and not (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
            break
        offset = sum(len(line) for line in lines[: node.end_lineno - 1]) + node.end_col_offset
    prefix, suffix = encoded[:offset].decode(), encoded[offset:].decode()
    if suffix.startswith(";"):
        suffix = suffix[1:].lstrip(" \t")
    return prefix + "\n" + imports + "\n" + suffix


@scorer(metrics=[accuracy(), stderr()])
def mbpp_with_imports():
    """Run native MBPP verification with its declared imports inside the sandbox."""
    native = verify()

    async def score(state, target):
        amended = copy(state)
        imports = "\n".join(state.metadata.get("test_imports", []))
        amended.output = ModelOutput.from_content(
            state.output.model, _test_environment(find_code(state.output.completion), imports)
        )
        result = await native(amended, target)
        result.answer = state.output.completion
        return result

    return score

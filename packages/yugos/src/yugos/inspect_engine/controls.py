"""Portable acceptance tasks that exercise Inspect's real tool execution loop."""

import json

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import solver
from inspect_ai.tool import tool


@tool
def lookup_record(record_id: str, value: str):
    async def execute(record_id: str) -> str:
        """Retrieve a record's value.

        Args:
            record_id: The record identifier to retrieve.
        """
        if record_id != expected_id:
            raise ValueError("unknown record")
        return json.dumps({"record_id": record_id, "value": value})

    expected_id = record_id
    return execute


@solver
def lookup_control(record_id: str, value: str):
    """Reconstruct operator-owned tools from portable arguments, then let Inspect run them."""

    async def solve(state, generate):
        state.tools = [lookup_record(record_id, value)]
        return await generate(state, tool_calls="loop")

    return solve


def lookup_task(record_id: str, value: str) -> Task:
    prompt = (
        f"Use lookup_record to retrieve record {json.dumps(record_id)}. "
        "Reply with only its value, exactly as returned by the tool."
    )
    if not value or value in prompt:
        raise ValueError("the lookup value must be absent from the prompt")
    return Task(
        dataset=[Sample(id="lookup", input=prompt, target=value)],
        solver=lookup_control(record_id, value),
        scorer=exact(),
        message_limit=4,
    )


def build_lookup(*, options: dict, record_id: str, value: str) -> Task:
    """Build the acceptance control with prepared samples and operator-owned tools."""
    return Task(**options, solver=lookup_control(record_id, value), scorer=exact())

"""Schema-directed reference normalization retains BFCL's native matching rules."""

from copy import deepcopy

import pytest
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.tool import ToolCall
from inspect_evals.bfcl.score.scorer import tool_call_matches_possible_answers

from yugos_evals._sample_patches import prepare_samples


@pytest.mark.parametrize("language", ["java", "javascript"])
def test_boolean_string_schema_normalizes_reference_values(language):
    schema = {
        "name": "Encoder.get",
        "parameters": {
            "properties": {
                "enabled": {
                    "type": "string",
                    "description": "This is Java boolean type parameter in string representation.",
                },
                "encoding": {"type": "string"},
            },
            "required": ["enabled", "encoding"],
        },
    }
    raw = [{"Encoder.get": {"enabled": [True], "encoding": ["UTF-8"]}}]
    metadata = {
        "language": language,
        "tools": [schema],
        "parsed_ground_truth": deepcopy(raw),
        "raw_ground_truth": deepcopy(raw),
    }
    task = Task(dataset=[Sample(id="example", input="encode", metadata=metadata)])
    prepare_samples(task, "bfcl")
    actual = task.dataset[0].metadata
    expected = actual["parsed_ground_truth"][0]
    for enabled, encoding, valid in [
        ("true", "UTF-8", True),
        ("false", "UTF-8", False),
        (True, "UTF-8", False),
        ("true", "ASCII", False),
    ]:
        result = tool_call_matches_possible_answers(
            ToolCall(
                id="1", function="Encoder_get", arguments={"enabled": enabled, "encoding": encoding}
            ),
            expected,
            schema,
            language,
        )
        assert result["valid"] is valid
    assert actual["raw_ground_truth"] == raw
    assert metadata["parsed_ground_truth"] == raw


def test_arbitrary_string_and_python_boolean_contracts_are_not_coerced():
    for language, description in [
        ("java", "A flag label"),
        ("python", "boolean type parameter in string representation"),
    ]:
        metadata = {
            "language": language,
            "tools": [
                {
                    "name": "f",
                    "parameters": {
                        "properties": {"flag": {"type": "string", "description": description}}
                    },
                }
            ],
            "parsed_ground_truth": [{"f": {"flag": [True]}}],
        }
        task = Task(dataset=[Sample(id="example", input="flag", metadata=deepcopy(metadata))])
        prepare_samples(task, "bfcl")
        assert task.dataset[0].metadata == metadata

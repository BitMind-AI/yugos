"""A frozen sandboxed question can be re-presented to a pinned coding harness."""

import asyncio
import importlib
from importlib.metadata import version

import pytest
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, GenerateConfig

from yugos.evaluation import Question
from yugos.inference.chat import ModelInput
from yugos.inspect_engine import restore_tasks
from yugos_evals.catalog import entry, freeze_entry, load_task
from yugos_evals.harnessed import harness_question, portable_request


@pytest.fixture
def sandboxed(monkeypatch) -> Question:
    module = importlib.import_module("inspect_evals.humaneval.humaneval")
    monkeypatch.setattr(
        module,
        "hf_dataset",
        lambda **kwargs: MemoryDataset(
            [Sample(id="HumanEval/0", input="def f():\n", target="pass\n")]
        ),
    )
    benchmark = entry("humaneval")
    [question] = freeze_entry(benchmark, task=load_task(benchmark))
    return question


def test_harnessed_question_keeps_the_native_samples_sandbox_and_scorer(sandboxed):
    harnessed = harness_question(sandboxed, harness="claude_code", version="2.1.311")
    [(native, _)] = restore_tasks([sandboxed])
    [(task, definition)] = restore_tasks([harnessed])

    assert harnessed.question_id == "humaneval@claude_code:HumanEval/0"
    assert harnessed.source_benchmark == "humaneval@claude_code"
    assert (harnessed.prompt, harnessed.expected_answer) == (
        sandboxed.prompt,
        sandboxed.expected_answer,
    )
    assert definition.packages["inspect-swe"] == version("inspect-swe")
    assert definition.sandbox == sandboxed.inspect_task["sandbox"]
    assert definition.scorer == sandboxed.inspect_task["scorer"]
    assert task.dataset[0].input == native.dataset[0].input
    assert task.solver is not native.solver
    # The harness and its release are part of the recipe, so they change the identity.
    other = harness_question(sandboxed, harness="claude_code", version="2.1.312")
    assert other.inspect_task != harnessed.inspect_task


def test_harnessed_question_drops_evidence_measured_with_the_native_agent(sandboxed):
    labelled = sandboxed.model_copy(update={"difficulty_decile": 3, "panel_solve_share": "0.5"})
    harnessed = harness_question(labelled, harness="codex_cli", version="0.80.0")
    assert harnessed.difficulty_decile is None and harnessed.panel_solve_share is None


@pytest.mark.parametrize(
    ("harness", "release", "message"),
    [
        ("aider", "1.0.0", "unknown harness"),
        ("claude_code", "latest", "exact release"),
        ("claude_code", "stable", "exact release"),
        ("codex_cli", "sandbox", "exact release"),
    ],
)
def test_only_known_harnesses_at_exact_releases_are_recorded(sandboxed, harness, release, message):
    with pytest.raises(ValueError, match=message):
        harness_question(sandboxed, harness=harness, version=release)
    forged = harness_question(sandboxed, harness="opencode", version="1.0.0")
    forged.inspect_task["builder_args"].update(harness=harness, version=release)
    with pytest.raises(ValueError, match=message):
        restore_tasks([forged])


def test_a_harness_needs_a_sandbox_and_is_applied_once(sandboxed):
    harnessed = harness_question(sandboxed, harness="opencode", version="1.0.0")
    with pytest.raises(ValueError, match="already presented"):
        harness_question(harnessed, harness="opencode", version="1.0.0")
    bare = Question.model_validate_json(sandboxed.model_dump_json())
    bare.inspect_task["sandbox"] = None
    bare.inspect_sample["sandbox"] = None
    with pytest.raises(ValueError, match="has none"):
        harness_question(bare, harness="opencode", version="1.0.0")


def test_harness_requests_become_acceptable_to_the_router_protocol():
    messages = [ChatMessageUser(content="fix the bug")]
    # What Claude Code and Codex attach to their requests.
    config = GenerateConfig(
        parallel_tool_calls=True,
        extra_body={"metadata": {"user_id": "device"}, "prompt_cache_key": "k", "store": False},
        extra_headers={"x-session": "s"},
    )
    with pytest.raises(ValueError, match="provider headers or body"):
        ModelInput.from_inspect(messages, [], "auto", config)

    filtered = asyncio.run(portable_request(None, messages, [], "auto", config))
    request = ModelInput.from_inspect(*filtered)
    assert request.config.parallel_tool_calls is True
    assert request.config.extra_body is None and request.config.extra_headers is None
    plain = GenerateConfig(temperature=0)
    assert asyncio.run(portable_request(None, messages, [], "auto", plain)) is None

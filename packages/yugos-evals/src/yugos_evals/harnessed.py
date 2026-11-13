"""Prepared sandboxed tasks solved by a real coding harness instead of the task's own agent.

The harness (Claude Code, Codex CLI or OpenCode, from `inspect-swe`) runs inside the
task's sandbox. Its model requests return to Inspect's model on the host, so they are
routed, metered and logged like any other turn and the sandbox needs no network. The
task's samples, setup, sandbox and scorer are unchanged; only the solver is replaced.
"""

from __future__ import annotations

import re
from importlib.metadata import version as installed_version

from inspect_ai import Task, task_with
from inspect_ai.model import GenerateInput

from yugos.inspect_engine.frozen import _task_builder

BUILDER = "yugos_evals.harnessed"
HARNESSES = ("claude_code", "codex_cli", "opencode")
_RELEASE = re.compile(r"\d+(\.\d+)+")


def _check(harness: str, version: str) -> None:
    if harness not in HARNESSES:
        raise ValueError(f"unknown harness: {harness}")
    if not isinstance(version, str) or not _RELEASE.fullmatch(version):
        raise ValueError("a harness is pinned to one exact release")


async def portable_request(model, messages, tools, tool_choice, config):
    """Drop the provider-private headers and body fields a harness adds to its requests.

    They carry session and cache hints for the harness's own provider; the routed
    model may be any provider, and the router protocol refuses them.
    """
    if config.extra_body is None and config.extra_headers is None:
        return None
    portable = config.model_copy(update={"extra_body": None, "extra_headers": None})
    return GenerateInput(messages, tools, tool_choice, portable)


def harness_agent(harness: str, version: str):
    """The named harness at one release, with its model requests made portable."""
    _check(harness, version)
    import inspect_swe

    return getattr(inspect_swe, harness)(version=version, filter=portable_request)


def build(*, options: dict, harness: str, version: str, builder: str, builder_args: dict) -> Task:
    """Build the task `builder` constructs, then replace its solver with the harness."""
    if builder == BUILDER:
        raise ValueError("a harnessed task wraps a native task builder")
    native = _task_builder(builder).load()(options=options, **builder_args)
    if native.sandbox is None and any(sample.sandbox is None for sample in native.dataset):
        raise ValueError("a harness runs in the task's sandbox; this task has none")
    return task_with(native, solver=harness_agent(harness, version))


def harness_question(question, *, harness: str, version: str):
    """A frozen sandboxed question re-presented to a harness, without reloading it.

    The copy is a separate source (`<source>@<harness>`) with no difficulty labels:
    panel evidence for the task's own agent does not describe a harness.
    """
    _check(harness, version)
    config = question.inspect_task
    if config["builder"] == BUILDER:
        raise ValueError("question is already presented to a harness")
    if config.get("sandbox") is None and question.inspect_sample.get("sandbox") is None:
        raise ValueError("a harness runs in the task's sandbox; this question has none")
    source, _, identity = question.question_id.partition(":")
    packages = {
        **config["packages"],
        "yugos-evals": installed_version("yugos-evals"),
        "inspect-swe": installed_version("inspect-swe"),
    }
    builder_args = {
        "harness": harness,
        "version": version,
        "builder": config["builder"],
        "builder_args": config["builder_args"],
    }
    return question.model_copy(
        update={
            "question_id": f"{source}@{harness}:{identity}",
            "source_benchmark": f"{question.source_benchmark or source}@{harness}",
            "difficulty_decile": None,
            "panel_solve_share": None,
            "inspect_task": {
                **config,
                "builder": BUILDER,
                "builder_args": builder_args,
                "packages": packages,
            },
        }
    )

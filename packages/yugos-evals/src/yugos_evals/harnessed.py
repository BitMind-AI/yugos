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
from inspect_ai.model import ContentText, GenerateInput
from inspect_ai.tool import ToolFunction, internal_tool_type

from yugos.inspect_engine.frozen import _task_builder

BUILDER = "yugos_evals.harnessed"
HARNESSES = ("claude_code", "codex_cli", "opencode")
# OpenCode formats requests as one model from its release's own catalog, which drops
# older models; the routed model still answers. This one is in current releases.
OPTIONS = {"opencode": {"opencode_model": "anthropic/claude-sonnet-4-6"}}
_RELEASE = re.compile(r"\d+(\.\d+)+")


def _check(harness: str, version: str) -> None:
    if harness not in HARNESSES:
        raise ValueError(f"unknown harness: {harness}")
    if not isinstance(version, str) or not _RELEASE.fullmatch(version):
        raise ValueError("a harness is pinned to one exact release")


def _text_only(message):
    """The message with each attachment replaced by a note that it was left out."""
    content = message.content
    if isinstance(content, str) or all(part.type in ("text", "reasoning") for part in content):
        return message
    parts = [
        part
        if part.type in ("text", "reasoning")
        else ContentText(text=f"[{part.type} omitted: this session's model reads text only]")
        for part in content
    ]
    return message.model_copy(update={"content": parts})


async def portable_request(model, messages, tools, tool_choice, config):
    """Keep a harness's request to what any routed model can do inside the sandbox.

    Provider-private headers and body fields are dropped: they carry session and cache
    hints for the harness's own provider, the routed model may be any provider, and
    the router protocol refuses them.

    Tools the provider runs itself, such as web search, are dropped too. They run on
    the provider's side with the network access the sandbox denies, and only for the
    providers that offer them. The harness's own tools are untouched: one that needs
    the network is still offered and fails as it would on a machine without it.

    An attachment, such as an image the harness read from disk, becomes a note saying
    it was left out. The router protocol carries text only and refuses the request
    otherwise, which ends the session with the harness reporting a failed model.
    """
    local = [tool for tool in tools if internal_tool_type(tool) is None]
    text = [_text_only(message) for message in messages]
    if (
        len(local) == len(tools)
        and all(new is old for new, old in zip(text, messages, strict=True))
        and config.extra_body is None
        and config.extra_headers is None
    ):
        return None
    named = isinstance(tool_choice, ToolFunction)
    if not local and (named or tool_choice == "any"):
        # The request required a tool and none is left: it is answered without one.
        tool_choice = "none"
    elif named and tool_choice.name not in {tool.name for tool in local}:
        tool_choice = "auto"
    portable = config.model_copy(update={"extra_body": None, "extra_headers": None})
    return GenerateInput(text, local, tool_choice, portable)


def harness_agent(harness: str, version: str):
    """The named harness at one release, with its model requests made portable."""
    _check(harness, version)
    import inspect_swe

    options = OPTIONS.get(harness, {})
    return getattr(inspect_swe, harness)(version=version, filter=portable_request, **options)


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
    panel evidence for the task's own agent does not describe a harness. The task's
    message limit is dropped too: it sizes the native agent's loop, and a harness
    session is bounded by the run's call, cost and time limits instead.
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
                "message_limit": None,
            },
        }
    )

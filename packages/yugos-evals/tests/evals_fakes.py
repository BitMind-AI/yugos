"""Scripted routers and sandboxes for task tests; no provider calls, no containers."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile

from inspect_ai.model import GenerateConfig, ModelAPI, ModelOutput, get_model, modelapi
from inspect_ai.util import ExecResult, SandboxEnvironment, sandboxenv

from yugos.inspect_engine import RouterCall, RouterReply, router_model, run_competition_eval

DIGEST_IMAGE = "fixture/python@sha256:" + "a" * 64


class ScriptedRouter:
    """Answer every call with one scripted reply and record what the router was sent."""

    def __init__(self, content: str, finish_reason: str = "stop"):
        self.reply = RouterReply(content=content, finish_reason=finish_reason)
        self.calls: list[RouterCall] = []

    async def __call__(self, call: RouterCall) -> RouterReply:
        self.calls.append(call)
        return self.reply

    def seen(self) -> str:
        """Everything the router received, as text."""
        return json.dumps([call.request.model_dump(mode="json") for call in self.calls])


def run(task, router: ScriptedRouter, tmp_path, **options):
    model = router_model(router, submission_id="sub")
    return run_competition_eval(
        task, model, log_dir=str(tmp_path / "logs"), display="none", **options
    )


# The judge's reply to a conversation's last message; tests replace it.
JUDGE_RULE = {"reply": lambda text: "[[A=B]]"}


@modelapi(name="yugos_evals_judge")
class ScriptedJudge(ModelAPI):
    """A grader model whose verdict depends only on the judged conversation."""

    def __init__(self, model_name, base_url=None, api_key=None, config=GenerateConfig(), **args):
        super().__init__(model_name, base_url, api_key, [], config)

    async def generate(self, input, tools, tool_choice, config):
        return ModelOutput.from_content(self.model_name, JUDGE_RULE["reply"](input[-1].text))


def judge(reply=None):
    """A grader model; `reply(text)` gives its verdict for the last message."""
    if reply is not None:
        JUDGE_RULE["reply"] = reply
    return get_model("yugos_evals_judge/judge")


class LocalProcesses(SandboxEnvironment):
    """Run the sandbox's commands as local processes: trusted fixture programs only.

    Records every command and input, and every attempt to read a file, so tests
    can check what entered the sandbox and that no verdict was read back.
    """

    def __init__(self, *, probe_ok: bool = True, timeout_on: str | None = None):
        super().__init__()
        self.cwd = tempfile.mkdtemp(prefix="yugos-evals-fixture-")
        self.execs: list[tuple[list[str], str]] = []
        self.reads: list[str] = []
        self.written: dict[str, str] = {}
        self.probe_ok = probe_ok
        self.timeout_on = timeout_on

    async def exec(self, cmd, input=None, timeout=None, **kwargs) -> ExecResult[str]:
        self.execs.append((list(cmd), input or ""))
        request = json.loads(input or "{}")
        if request.get("op") == "probe" and not self.probe_ok:
            return ExecResult(False, 127, "", "python3: not found")
        if self.timeout_on and self.timeout_on in request.get("code", ""):
            raise TimeoutError()
        command = [sys.executable if part == "python3" else part for part in cmd]
        done = subprocess.run(
            command, input=input, capture_output=True, text=True, timeout=timeout, cwd=self.cwd
        )
        return ExecResult(done.returncode == 0, done.returncode, done.stdout, done.stderr)

    async def write_file(self, file, contents):
        self.written[file] = contents

    async def read_file(self, file, text=True):
        self.reads.append(file)
        return self.written.get(file, "")

    def payloads(self) -> str:
        return "\n".join(stdin for _, stdin in self.execs)

    @classmethod
    async def sample_cleanup(cls, task_name, config, environments, interrupted):
        return None


# The provider an end-to-end test binds its frozen tasks to; not an isolation boundary.
CURRENT: dict[str, LocalProcesses] = {}


@sandboxenv(name="yugos_evals_fixture")
class FixtureSandbox(SandboxEnvironment):
    """Registered stand-in for an isolated provider, delegating to `CURRENT["env"]`."""

    async def exec(self, cmd, input=None, cwd=None, env=None, user=None, timeout=None, **kw):
        return await CURRENT["env"].exec(cmd, input=input, timeout=timeout)

    async def write_file(self, file, contents):
        await CURRENT["env"].write_file(file, contents)

    async def read_file(self, file, text=True):
        return await CURRENT["env"].read_file(file, text)

    @classmethod
    async def sample_init(cls, task_name, config, metadata):
        return {"default": cls()}

    @classmethod
    async def sample_cleanup(cls, task_name, config, environments, interrupted):
        return None

    @classmethod
    def config_deserialize(cls, config):
        from inspect_ai.util import ComposeConfig

        return ComposeConfig.model_validate(config)


def task_state(completion: str, *, stop_reason="stop", input="q", metadata=None):
    """A scored sample's state after the router answered `completion`."""
    from inspect_ai.model._model import ModelName
    from inspect_ai.solver import TaskState

    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id="s",
        epoch=1,
        input=input,
        messages=[],
        output=ModelOutput.from_content("mockllm/model", completion, stop_reason=stop_reason),
        metadata=metadata or {},
    )


def assert_portable_task_config(question) -> None:
    """A packed task config names trusted code and pins its package."""
    from importlib.metadata import version

    task_config = question.inspect_task
    assert task_config["builder"] == "yugos_evals.prepared"
    assert task_config["packages"]["yugos-evals"] == version("yugos-evals")


# Test task code remains in Python; the pack selects a fixture constructor by name.
_NATIVE_FIXTURES = {}


def freeze_fixture(task, **options):
    from inspect_ai._util.registry import registry_info
    from inspect_ai.scorer._scorer import unique_scorer_name

    from yugos.inspect_engine.frozen import freeze_task as prepare

    try:
        name = registry_info(task).name
    except ValueError:
        name = ""
    if name.startswith(("inspect_evals/", "yugos_evals/")):
        from yugos_evals.prepared import prepare_task

        return prepare_task(task, **options)
    names = []
    for scorer in task.scorer or ():
        names.append(unique_scorer_name(scorer, names))
    case = (task.name if task.name == "fixture" else options["source"]) + ":" + "+".join(names)
    _NATIVE_FIXTURES[case] = task
    from unittest.mock import patch

    from yugos.inspect_engine import frozen

    with patch.object(frozen, "_task_builder", return_value=_fixture_registration()):
        return prepare(task, builder="tests.yugos_evals", builder_args={"case": case}, **options)


def _fixture_registration():
    from importlib.metadata import version
    from types import SimpleNamespace

    from inspect_ai import task_with

    def build(*, options, case):
        from copy import deepcopy

        return task_with(deepcopy(_NATIVE_FIXTURES[case]), **options)

    return SimpleNamespace(
        dist=SimpleNamespace(name="pytest", version=version("pytest")), load=lambda: build
    )


def install_task_builders(monkeypatch):
    from yugos.inspect_engine import frozen

    original = frozen._task_builder
    monkeypatch.setattr(
        frozen,
        "_task_builder",
        lambda name: _fixture_registration() if name == "tests.yugos_evals" else original(name),
    )

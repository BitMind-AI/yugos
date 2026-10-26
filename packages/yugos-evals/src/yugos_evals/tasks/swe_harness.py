"""`yugos_evals/swe_harness`: SWE-bench-format tasks graded by each dataset's own test recipe.

Inspect Evals' `swe_bench` scorer only knows SWE-bench's original Python repos.
These datasets carry their test recipe per row instead:

- `eval_script` (SWE-bench Multilingual, Multimodal): a complete script that
  resets and patches the test files and runs the selected tests;
- `rebench` (SWE-rebench): an `install_config` with the test command, run on the
  test files the test patch touches, as the SWE-rebench harness does;
- `live` (SWE-bench-Live): `test_cmds` (and `rebuild_cmds`) after the test patch.

Each row's log parser is SWE-bench's own (vendored in `scorers.swe_logs`). The
agent is Inspect Evals' SWE-bench agent, unchanged.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import ComposeConfig, ComposeService, SandboxEnvironmentSpec, sandbox

from ..scorers.swe_logs import PARSER_REGISTRY
from ..scorers.swe_logs._status import TestSpec

VERSION = "1-A"
Family = Literal["eval_script", "rebench", "live"]
PROMPT = "Please solve the following coding issue:\n\n{issue_text}"
START, END = ">>>>> Start Test Output", ">>>>> End Test Output"
_PASS = {"PASSED", "XFAIL"}
_DELIMITER = "EOF_114329324912"


def _list(value) -> list:
    return json.loads(value) if isinstance(value, str) else list(value or [])


def _image(row: dict, family: Family) -> str:
    if family == "live":
        name = row["instance_id"].replace("__", "_1776_").lower()
        return f"starryzhang/sweb.eval.x86_64.{name}:latest"
    return row.get("image") or row.get("image_name") or row["docker_image"]


def _images(row: dict) -> list[str]:
    """Issue screenshots, fetched while preparing so execution needs no network."""
    from urllib.request import Request, urlopen

    assets = row.get("image_assets")
    urls = (json.loads(assets) if isinstance(assets, str) else assets or {}).get(
        "problem_statement", []
    )
    images = []
    for url in urls:
        request = Request(url, headers={"User-Agent": "yugos-evals"})
        with urlopen(request, timeout=60) as response:
            kind = response.headers.get_content_type()
            data = base64.b64encode(response.read(20_000_000)).decode()
        images.append(f"data:{kind};base64,{data}")
    return images


def _sample(row: dict, family: Family) -> Sample:
    service = ComposeService(
        image=_image(row, family),
        command="sleep infinity",
        working_dir="/testbed",
        network_mode="none",
    )
    prompt = PROMPT.format(issue_text=row["problem_statement"])
    images = _images(row) if family == "eval_script" else []
    content = [ContentText(text=prompt), *(ContentImage(image=i) for i in images)]
    recipe = {
        "eval_script": {"eval_script": row.get("eval_script")},
        "rebench": {"install_config": row.get("install_config")},
        "live": {
            "test_cmds": _list(row.get("test_cmds")),
            "rebuild_cmds": _list(row.get("rebuild_cmds")),
        },
    }[family]
    return Sample(
        id=row["instance_id"],
        input=[ChatMessageUser(content=content)] if images else prompt,
        metadata={
            "family": family,
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "version": str(row.get("version") or ""),
            "base_commit": row["base_commit"],
            "patch": row["patch"],
            "test_patch": row["test_patch"],
            "FAIL_TO_PASS": _list(row["FAIL_TO_PASS"]),
            "PASS_TO_PASS": _list(row["PASS_TO_PASS"]),
            "log_parser": row.get("log_parser") or _rebench_config(row).get("log_parser"),
            **recipe,
        },
        sandbox=SandboxEnvironmentSpec(
            "docker", ComposeConfig(services={"default": service})
        ),
    )


def _rebench_config(row: dict) -> dict:
    config = row.get("install_config")
    return json.loads(config) if isinstance(config, str) else config or {}


@task
def swe_harness(
    dataset: str,
    family: Family,
    revision: str,
    split: str = "test",
    tool_timeout: int = 210,
) -> Task:
    """Rows of a SWE-bench-format dataset, loaded at preparation from a pinned revision."""
    from datasets import load_dataset
    from inspect_evals.swe_bench.solvers import swe_bench_agent_with_inspect_tool_support

    rows = load_dataset(dataset, split=split, revision=revision)
    return Task(
        dataset=[_sample(row, family) for row in rows],
        solver=swe_bench_agent_with_inspect_tool_support(tool_timeout=tool_timeout),
        scorer=swe_harness_scorer(),
        version=VERSION,
    )


def _test_files(test_patch: str) -> list[str]:
    return re.findall(r"^diff --git a/.* b/(.*)$", test_patch, flags=re.M)


def eval_script(metadata: dict) -> str:
    """The dataset's test recipe as one script, with SWE-bench's output markers."""
    family = metadata["family"]
    if family == "eval_script":
        return metadata["eval_script"]
    base, patch = metadata["base_commit"], metadata["test_patch"]
    files = " ".join(_test_files(patch))
    reset = f"git checkout {base} {files}" if files else ":"
    setup = [reset, f"git apply -v - <<'{_DELIMITER}'\n{patch}\n{_DELIMITER}"]
    if family == "rebench":
        config = _rebench_config(metadata)
        test_cmd = config.get("test_cmd") or "pytest -rA"
        test_cmd = " ".join(test_cmd) if isinstance(test_cmd, list) else test_cmd
        prelude = [
            "source /opt/miniconda3/bin/activate 2>/dev/null || source /opt/conda/bin/activate",
            "conda activate testbed",
        ]
        run = f"{test_cmd} {files}"
    else:
        prelude = []
        setup += metadata["rebuild_cmds"]
        run = " ; ".join(metadata["test_cmds"])
    return "\n".join(
        ["#!/bin/bash", "set -uxo pipefail", *prelude, "cd /testbed",
         "git config --global --add safe.directory /testbed", *setup,
         f": '{START}'", run, f": '{END}'", reset]
    )  # fmt: skip


def resolved(metadata: dict, log: str) -> tuple[bool, dict]:
    """Grade the marked test output the way the dataset's own harness does."""
    if START in log and END in log:
        log = log.split(START, 1)[1].split(END, 1)[0]
    name = metadata["log_parser"]
    parser = PARSER_REGISTRY["parse_log_pytest" if name == "pytest" else name]
    status = parser(log, TestSpec(metadata["instance_id"], metadata["version"]))
    passed = {t for t, s in status.items() if str(s).upper() in _PASS}
    failed = {t for t, s in status.items() if str(s).upper() in {"FAILED", "ERROR"}}
    f2p, p2p = set(metadata["FAIL_TO_PASS"]), set(metadata["PASS_TO_PASS"])
    if metadata["family"] == "live":
        # SWE-bench-Live: no listed test fails, and every FAIL_TO_PASS test passes.
        ok = not (failed & (f2p | p2p)) and f2p <= passed
    else:
        ok = f2p <= passed and p2p <= passed
    return ok, {"passed": len(passed & (f2p | p2p)), "required": len(f2p | p2p)}


@scorer(metrics=[accuracy(), stderr()])
def swe_harness_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        metadata = state.metadata
        base = metadata["base_commit"]
        diff = await sandbox().exec(
            ["bash", "-c", f"cd /testbed && git add -A && git diff --cached --text {base}"]
        )
        await sandbox().write_file("/tmp/yugos_eval.sh", eval_script(metadata))
        result = await sandbox().exec(["bash", "/tmp/yugos_eval.sh"], timeout=1800)
        ok, counts = resolved(metadata, result.stdout + "\n" + result.stderr)
        return Score(
            value=CORRECT if ok else INCORRECT,
            answer=diff.stdout,
            explanation=f"{counts['passed']} of {counts['required']} listed tests passed",
            metadata=counts,
        )

    return score

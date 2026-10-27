"""`yugos_evals/intercode_bash`: InterCode's NL2Bash tasks as an interactive shell task.

The 200 natural-language requests of InterCode-Bash (MIT, pinned by commit) over
its four prepared file systems, one image per file system. The agent runs shell
commands; its answer is the output of its last command, as in InterCode.

InterCode grades against a second, clean container. Here one container does both:
the scorer records the agent's file-system changes, resets the container with
git (restoring modification times when the image lists them), runs the gold command and computes InterCode's reward (0.01, plus up to 0.33
each for the changed paths, their contents and the TF-IDF similarity of the
outputs). A sample is correct at the full reward of 1.
"""

from __future__ import annotations

import json
import math
import re
import shlex
from collections import Counter
from textwrap import dedent

from inspect_ai import Task, task
from inspect_ai.agent import Agent, react
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.tool import Tool, tool
from inspect_ai.util import ComposeConfig, ComposeService, SandboxEnvironmentSpec, sandbox, store

from ..sources._fetch import fetch

VERSION = "1-A"
REVISION = "c3e46d827cfc9d4c704ec078f7abf9f41e3191d8"
SHA256 = {
    1: "60f88e1aacc7ebba535093f9890c5c33203f4e5f32958e0e94fbe90ec4f01c82",
    2: "8f4ce24e535fab782fda607e37db2ae1d6c5f99993c638d1ac0a7e0b542f633e",
    3: "a2d4ec8bc7ad69a4e2fb3eb84033994cf65ee9cfb355e3e63099df67a339b2e1",
    4: "ce41b89450f87765a02a51df259ca0c1762e8249185c022adb089147e2c16200",
}
COMMAND_TIMEOUT = 60
LAST_OUTPUT = "intercode_last_output"
_CWD = "intercode_cwd"
_MARK = "\x1e"
GIT_RESET = "git reset --hard; git clean -fd;"
# Images may list every tracked path's mtime, NUL-separated `<mtime>\t<path>` records,
# so files a reset recreates keep the times the agent saw.
MTIMES = "/var/lib/intercode/mtimes"
RESTORE_MTIMES = (
    f"if [ -f {MTIMES} ]; then while IFS=$'\\t' read -r -d '' t p; "
    f'do touch -h -d "@$t" "$p"; done < {MTIMES}; fi'
)
GIT_STATUS = "git status --short;"
AGENT_PROMPT = dedent("""
    You are an expert in the bash shell, working on an Ubuntu machine without
    internet access. Use the shell tool to carry out the user's request. Your
    answer is the output of the last command you run: finish with the command
    whose output answers the request (any command, if the request only asks for
    changes to files), then call submit.
""")


def _sample(fs: int, index: int, row: dict, image: str) -> Sample:
    service = ComposeService(image=image, command="sleep infinity", network_mode="none")
    return Sample(
        id=f"fs{fs}-{index}",
        input=row["query"],
        metadata={"file_system": fs, "gold": row["gold"]},
        sandbox=SandboxEnvironmentSpec("docker", ComposeConfig(services={"default": service})),
    )


@task
def intercode_bash(images: list[str], revision: str = REVISION) -> Task:
    """`images[i]` is InterCode's NL2Bash image for file system i + 1, pinned by digest."""
    if len(images) != 4 or not all(re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", i) for i in images):
        raise ValueError("intercode_bash needs its four file-system images pinned by digest")
    samples = []
    for fs, image in enumerate(images, start=1):
        path = fetch(
            f"https://raw.githubusercontent.com/princeton-nlp/intercode/{revision}"
            f"/data/nl2bash/nl2bash_fs_{fs}.json",
            name=f"nl2bash_fs_{fs}.json",
            max_bytes=1_000_000,
            sha256=SHA256[fs] if revision == REVISION else None,
        )
        rows = json.loads(path.read_text())
        samples += [_sample(fs, index, row, image) for index, row in enumerate(rows)]
    return Task(
        dataset=samples,
        solver=intercode_agent(),
        scorer=intercode_scorer(),
        version=VERSION,
    )


@tool
def shell() -> Tool:
    async def execute(command: str) -> str:
        """Run a bash command; the working directory carries over between calls.

        Args:
            command: The bash command to run.

        Returns:
            The command's output (stdout, then stderr).
        """
        cwd = store().get(_CWD, "/")
        script = f'cd {shlex.quote(cwd)}\n{command}\n__s=$?; printf "{_MARK}%s" "$PWD"; exit $__s'
        try:
            result = await sandbox().exec(["bash", "-c", script], timeout=COMMAND_TIMEOUT)
            stdout, marker, new_cwd = result.stdout.rpartition(_MARK)
            if not marker:
                stdout, new_cwd = result.stdout, cwd
            store().set(_CWD, new_cwd)
            output = stdout + result.stderr
        except TimeoutError:
            output = "Command timed out"
        except UnicodeDecodeError:
            output = "Command output is not UTF-8 text"
        store().set(LAST_OUTPUT, output)
        return output

    return execute


def intercode_agent() -> Agent:
    return react(
        description="Bash agent whose answer is its last command's output",
        prompt=AGENT_PROMPT,
        tools=[shell()],
    )


def _tokens(text: str) -> list[str]:
    return re.findall(r"(?u)\b\w\w+\b", text.lower())


def answer_similarity(agent: str | None, gold: str | None) -> float:
    """Cosine similarity of scikit-learn's default TF-IDF vectors of the two outputs."""
    if agent is None or gold is None:
        return float(agent == gold)
    counts = [Counter(_tokens(agent)), Counter(_tokens(gold))]
    vocabulary = set(counts[0]) | set(counts[1])
    if not vocabulary:
        return float(agent == gold)
    idf = {t: math.log(3 / (1 + sum(t in c for c in counts))) + 1 for t in vocabulary}
    vectors = [{t: c[t] * idf[t] for t in c} for c in counts]
    norms = [math.sqrt(sum(v * v for v in vector.values())) for vector in vectors]
    if not all(norms):
        return 0.0
    dot = sum(vectors[0][t] * vectors[1].get(t, 0.0) for t in vectors[0])
    return dot / (norms[0] * norms[1])


def _changes(status: str) -> list[tuple[str, str]]:
    words = status.split()
    return [(words[i + 1], words[i]) for i in range(0, len(words) - 1, 2)]


async def _output(argv: list[str]) -> str:
    try:
        result = await sandbox().exec(argv, cwd="/", timeout=COMMAND_TIMEOUT)
    except UnicodeDecodeError:
        return "Command output is not UTF-8 text"
    return result.stdout + result.stderr


async def _hashes(changes: list[tuple[str, str]]) -> dict[str, str]:
    paths = [path for path, kind in changes if kind in ("A", "??", "C")]
    return {
        path: await _output(["md5sum", path] if "." in path else ["md5deep", "-r", path])
        for path in paths
    }


def reward(
    agent_changes, gold_changes, agent_hashes, gold_hashes, agent_output, gold_output
) -> tuple[float, dict]:
    """InterCode-Bash's reward from both runs' changes, content hashes and outputs."""
    miss = set(gold_changes) - set(agent_changes)
    extra = set(agent_changes) - set(gold_changes)
    file_diff = round(0.33 * (1 - math.erf(len(miss) + len(extra))), 2)
    same = [c for c in set(agent_changes) & set(gold_changes) if c[1] in ("A", "??", "C")]
    file_changes = 0.33
    if same:
        correct = sum(agent_hashes.get(p) == gold_hashes.get(p) for p, _ in same)
        file_changes = round(0.33 * correct / len(same), 2)
    answer = round(0.33 * answer_similarity(agent_output, gold_output), 2)
    parts = {"file_diff": file_diff, "file_changes": file_changes, "answer_similarity": answer}
    return round(0.01 + file_diff + file_changes + answer, 2), parts


@scorer(metrics=[accuracy(), stderr()])
def intercode_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        agent_changes = _changes(await _output(["bash", "-c", GIT_STATUS]))
        agent_hashes = await _hashes(agent_changes)
        await _output(["bash", "-c", GIT_RESET + RESTORE_MTIMES])
        gold = state.metadata["gold"]
        try:
            # InterCode runs the gold command as `docker exec <container> /bin/bash -c "<gold>"`.
            gold_output = await _output(shlex.split(f'/bin/bash -c "{gold.strip()}"'))
        except (ValueError, TimeoutError):
            gold_output = None
        gold_changes = _changes(await _output(["bash", "-c", GIT_STATUS]))
        gold_hashes = await _hashes(gold_changes)
        total, parts = reward(
            agent_changes,
            gold_changes,
            agent_hashes,
            gold_hashes,
            store().get(LAST_OUTPUT),
            gold_output,
        )
        return Score(
            value=CORRECT if total == 1 else INCORRECT,
            answer=store().get(LAST_OUTPUT),
            explanation=f"InterCode reward {total}: {parts}",
            metadata={"reward": total, **parts},
        )

    return score

"""`yugos_evals/spider2_lite`: Spider 2.0-Lite's SQLite tasks as an agentic SQL task.

The 135 `local` instances of Spider 2.0-Lite (MIT, pinned by commit). The image
holds the 30 SQLite databases and their schema descriptions; the agent explores
them with shell and Python, writes its answer table to `/workspace/result.csv`
and submits, as in Spider 2.0's CSV submission format. The scorer compares that
table with every gold table using Spider 2.0's own comparison (`condition_cols`,
`ignore_order`, 0.01 numeric tolerance).
"""

from __future__ import annotations

import io
import json
import re
import tarfile
from textwrap import dedent

from inspect_ai import Task, task
from inspect_ai.agent import Agent, react
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.tool import bash, python
from inspect_ai.util import ComposeConfig, ComposeService, SandboxEnvironmentSpec, sandbox

from ..sources._fetch import fetch

VERSION = "1-A"
REVISION = "cafb867313aab4e674652054198f383cf4018943"
WORKSPACE = "/workspace"
RESULT = f"{WORKSPACE}/result.csv"
TOOL_TIMEOUT = 180
PROMPT = """{question}
{knowledge}
The SQLite database is {workspace}/databases/{db}.sqlite. Its tables are described in {workspace}/schema/{schema}/: DDL.csv has each table's definition, and each JSON file has a table's columns, types and sample rows.

Write the answer table to {result} as CSV with a header row, for example by running your final SQL query with sqlite3 or pandas, then call submit."""  # noqa: E501
AGENT_PROMPT = dedent("""
    You are an expert data analyst working on an Ubuntu machine with bash and
    python tools, sqlite3, pandas and no internet access. Answer the question
    from the SQLite database, then save the answer table and call submit.
""")


def _sources(revision: str) -> dict[str, bytes]:
    archive = fetch(
        f"https://codeload.github.com/xlang-ai/Spider2/tar.gz/{revision}",
        name="spider2.tar.gz",
        max_bytes=600_000_000,
    )
    wanted = re.compile(
        r"spider2-lite/(spider2-lite\.jsonl|evaluation_suite/gold/spider2lite_eval\.jsonl"
        r"|evaluation_suite/gold/exec_result/local\d+(_[a-z])?\.csv|resource/documents/.+\.md"
        r"|resource/databases/sqlite/[^/]+/)"
    )
    files = {}
    with tarfile.open(archive) as tar:
        for member in tar:
            path = member.name.split("/", 1)[-1]
            if member.isfile() and wanted.match(path):
                files[path] = tar.extractfile(member).read()
    return files


def _jsonl(data: bytes) -> list[dict]:
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


@task
def spider2_lite(image: str, revision: str = REVISION) -> Task:
    """Spider 2.0-Lite's SQLite instances on an image with the databases, pinned by digest."""
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image or ""):
        raise ValueError("spider2_lite needs its database image pinned by digest")
    files = _sources(revision)
    standards = {
        row["instance_id"]: row
        for row in _jsonl(files["spider2-lite/evaluation_suite/gold/spider2lite_eval.jsonl"])
    }
    schema_root = "spider2-lite/resource/databases/sqlite/"
    schemas = {
        name.lower().replace("-", "_"): name
        for name in {
            p[len(schema_root) :].split("/")[0] for p in files if p.startswith(schema_root)
        }
    }
    gold_dir = "spider2-lite/evaluation_suite/gold/exec_result/"
    samples = []
    for row in _jsonl(files["spider2-lite/spider2-lite.jsonl"]):
        instance = row["instance_id"]
        if not instance.startswith("local"):
            continue
        golds = sorted(
            path for path in files
            if re.fullmatch(rf"{re.escape(gold_dir + instance)}(_[a-z])?\.csv", path)
        )  # fmt: skip
        knowledge = row.get("external_knowledge")
        document = files.get(f"spider2-lite/resource/documents/{knowledge}") if knowledge else None
        standard = standards[instance]
        samples.append(
            Sample(
                id=instance,
                input=PROMPT.format(
                    question=row["question"],
                    knowledge=f"\nExternal knowledge:\n\n{document.decode()}\n" if document else "",
                    workspace=WORKSPACE,
                    db=row["db"],
                    schema=schemas[row["db"].lower().replace("-", "_")],
                    result=RESULT,
                ),
                metadata={
                    "db": row["db"],
                    "gold": [files[path].decode() for path in golds],
                    "single_gold": golds == [f"{gold_dir}{instance}.csv"],
                    "condition_cols": standard.get("condition_cols"),
                    "ignore_order": standard.get("ignore_order", False),
                },
            )
        )
    service = ComposeService(
        image=image, command="sleep infinity", working_dir=WORKSPACE, network_mode="none"
    )
    return Task(
        dataset=samples,
        solver=spider2_agent(),
        scorer=spider2_scorer(),
        sandbox=SandboxEnvironmentSpec("docker", ComposeConfig(services={"default": service})),
        version=VERSION,
    )


def spider2_agent(tool_timeout: int = TOOL_TIMEOUT) -> Agent:
    return react(
        description="Data analyst agent with bash and python tools",
        prompt=AGENT_PROMPT,
        tools=[bash(timeout=tool_timeout), python(timeout=tool_timeout)],
    )


def result_matches(result_csv: str, metadata: dict) -> bool:
    """Spider 2.0's verdict for one answer table against the instance's gold tables."""
    import pandas as pd

    from ..scorers.spider2_compare import compare_multi_pandas_table, compare_pandas_table

    pred = pd.read_csv(io.StringIO(result_csv))
    golds = [pd.read_csv(io.StringIO(gold)) for gold in metadata["gold"]]
    if metadata["single_gold"]:
        return bool(
            compare_pandas_table(
                pred, golds[0], metadata["condition_cols"], metadata["ignore_order"]
            )
        )
    return bool(
        compare_multi_pandas_table(
            pred, golds, metadata["condition_cols"], metadata["ignore_order"]
        )
    )


@scorer(metrics=[accuracy(), stderr()])
def spider2_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        try:
            result = (await sandbox().read_file(RESULT, text=False)).decode("utf-8", "replace")
        except FileNotFoundError:
            return Score(value=INCORRECT, explanation=f"no answer table at {RESULT}")
        try:
            matched = result_matches(result, state.metadata)
        except Exception as exc:  # an unreadable table is a wrong answer, as in Spider 2.0
            return Score(value=INCORRECT, answer=result[:2000], explanation=repr(exc)[:500])
        return Score(value=CORRECT if matched else INCORRECT, answer=result[:2000])

    return score

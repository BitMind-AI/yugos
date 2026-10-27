"""`yugos_evals/aider_polyglot`: Aider's Polyglot benchmark as an agentic coding task.

The 225 hardest Exercism exercises in C++, Go, Java, JavaScript, Python and Rust,
from Aider's `polyglot-benchmark` repository pinned by commit. Each sample writes
one exercise's starter and test files into `/workspace/<exercise>` (C++ builds
name their targets after the directory); the agent edits the
solution files and may run the tests. The scorer restores the original test files
and runs the language's test command as Aider's harness does (all tests enabled,
3 minutes), and the exercise passes when the command succeeds.

The image needs every toolchain with the exercises' dependencies cached, because
the sandbox has no network. Reference solutions are kept in sample metadata for
qualification only; they are never written to the sandbox.
"""

from __future__ import annotations

import base64
import json
import re
import tarfile
from pathlib import PurePosixPath
from textwrap import dedent

from inspect_ai import Task, task
from inspect_ai.agent import Agent, react
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import bash_session, python, text_editor
from inspect_ai.util import ComposeConfig, ComposeService, SandboxEnvironmentSpec, sandbox

from ..sources._fetch import fetch

VERSION = "1-A"
REVISION = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
WORKSPACE = "/workspace"
TEST_TIMEOUT = 180
TOOL_TIMEOUT = 210
LANGUAGES = ("cpp", "go", "java", "javascript", "python", "rust")
COMMANDS = {
    "cpp": "mkdir -p build && cd build && cmake -DEXERCISM_RUN_ALL_TESTS=1 -G 'Unix Makefiles' .. && make",
    "go": "go test ./...",
    "java": "./gradlew test --offline",
    "javascript": "npm run test",
    "python": "pytest",
    "rust": "cargo test -- --include-ignored",
}
# Aider's prompt addendum, plus where the exercise is and how to test it.
PROMPT = """{instructions}
####

Use the above instructions to modify the supplied files: {file_list}
Don't change the names of existing functions or classes, as they may be referenced from other code like unit tests, etc.
Only use standard libraries, don't suggest installing any packages.

The exercise is in {directory}. Run its tests with `{command}`; the tests are correct, don't change them."""  # noqa: E501
AGENT_PROMPT = dedent("""
    You are an expert software engineer working on a standard Ubuntu machine with
    bash and python tools and no internet access. Solve the coding exercise by
    editing its files. Once you are done, use your submit tool.
""")
_IGNORED = {"CMakeLists.txt", "Cargo.toml"}


def _enable_all_tests(path: str, text: str) -> str:
    """Aider runs every test: Java's `@Disabled` and JavaScript's `xtest` are switched on."""
    if path.endswith(".java"):
        return re.sub(r"@Disabled\([^)]*\)\s*\n", "", text)
    if path.endswith(".spec.js"):
        return re.sub(r"\bxtest\(", "test(", text)
    return text


def _gold_target(example: str, solution: list[str]) -> str:
    name = PurePosixPath(example).name
    if name == "Cargo-example.toml":
        return "Cargo.toml"
    by_name = [s for s in solution if PurePosixPath(s).name == name]
    by_suffix = [s for s in solution if PurePosixPath(s).suffix == PurePosixPath(name).suffix]
    if len(by_name) == 1:
        return by_name[0]
    if len(by_suffix) == 1 and not by_name:
        return by_suffix[0]
    if name.endswith(".java"):
        return f"src/main/java/{name}"
    raise ValueError(f"no solution file for reference file {example}")


def _exercises(revision: str) -> dict[str, dict[str, bytes]]:
    archive = fetch(
        f"https://codeload.github.com/Aider-AI/polyglot-benchmark/tar.gz/{revision}",
        name="polyglot-benchmark.tar.gz",
        max_bytes=200_000_000,
    )
    exercises: dict[str, dict[str, bytes]] = {}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or len(parts) < 6 or parts[2:4] != ("exercises", "practice"):
                continue
            key = f"{parts[1]}/{parts[4]}"
            exercises.setdefault(key, {})["/".join(parts[5:])] = tar.extractfile(member).read()
    return exercises


def _sample(key: str, files: dict[str, bytes]) -> Sample:
    language, name = key.split("/")
    config = json.loads(files[".meta/config.json"])["files"]
    tests = {path: _enable_all_tests(path, files[path].decode()) for path in config["test"]}
    workspace: dict[str, str] = {}
    binary: dict[str, str] = {}
    for path, content in files.items():
        if path.split("/")[0].startswith(".") and "/" in path:
            continue  # .meta, .docs, .approaches: references and their write-ups
        if path in tests:
            workspace[path] = tests[path]
            continue
        try:
            workspace[path] = content.decode()
        except UnicodeDecodeError:
            binary[path] = base64.b64encode(content).decode()
    example = {
        _gold_target(path, config["solution"]): files[path].decode()
        for path in config.get("example", [])
    }
    reference = ".meta/src/reference/java/"
    for path in files:  # Java references may span more files than the config lists
        if language == "java" and path.startswith(reference):
            example.setdefault(f"src/main/java/{path[len(reference) :]}", files[path].decode())
    if language == "rust" and ".meta/Cargo-example.toml" in files:
        example["Cargo.toml"] = files[".meta/Cargo-example.toml"].decode()
    editable = [s for s in config["solution"] if s not in _IGNORED and s not in tests]
    instructions = "".join(
        files[f".docs/{doc}.md"].decode()
        for doc in ("introduction", "instructions", "instructions.append")
        if f".docs/{doc}.md" in files
    )
    command = COMMANDS[language]
    return Sample(
        id=key,
        input=PROMPT.format(
            instructions=instructions,
            file_list=" ".join(PurePosixPath(s).name for s in editable),
            directory=f"{WORKSPACE}/{name}",
            command=command,
        ),
        metadata={
            "language": language,
            "exercise": name,
            "directory": f"{WORKSPACE}/{name}",
            "command": command,
            "files": workspace,
            "binary_files": binary,
            "tests": tests,
            "example": example,
        },
    )


def polyglot_sandbox(image: str) -> SandboxEnvironmentSpec:
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image or ""):
        raise ValueError("aider_polyglot needs its toolchain image pinned by digest")
    service = ComposeService(
        image=image, command="sleep infinity", working_dir=WORKSPACE, network_mode="none"
    )
    return SandboxEnvironmentSpec("docker", ComposeConfig(services={"default": service}))


def polyglot_agent(tool_timeout: int = TOOL_TIMEOUT) -> Agent:
    return react(
        description="Coding agent with python, bash_session and text_editor tools",
        prompt=AGENT_PROMPT,
        tools=[
            python(timeout=tool_timeout),
            bash_session(timeout=tool_timeout),
            text_editor(timeout=tool_timeout),
        ],
    )


@task
def aider_polyglot(
    image: str,
    revision: str = REVISION,
    languages: list[str] | None = None,
    tool_timeout: int = TOOL_TIMEOUT,
) -> Task:
    """One sample per exercise, loaded at preparation from the pinned repository."""
    chosen = set(languages or LANGUAGES)
    if unknown := chosen - set(LANGUAGES):
        raise ValueError(f"unknown languages: {sorted(unknown)}")
    exercises = _exercises(revision)
    return Task(
        dataset=[
            _sample(key, files)
            for key, files in sorted(exercises.items())
            if key.split("/")[0] in chosen
        ],
        setup=polyglot_setup(),
        solver=polyglot_agent(tool_timeout),
        scorer=polyglot_scorer(),
        sandbox=polyglot_sandbox(image),
        version=VERSION,
    )


async def _write(directory: str, files: dict[str, str | bytes]) -> None:
    for path, content in files.items():
        await sandbox().write_file(f"{directory}/{path}", content)


@solver
def polyglot_setup() -> Solver:
    """Write the exercise into the workspace and link the shared JavaScript dependencies."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        metadata = state.metadata
        directory = metadata["directory"]
        await _write(directory, metadata["files"])
        binary = {p: base64.b64decode(c) for p, c in metadata["binary_files"].items()}
        await _write(directory, binary)
        prepare = {
            "java": "chmod +x gradlew",
            "javascript": "ln -sfn /npm-install/node_modules node_modules",
        }.get(metadata["language"])
        if prepare:
            result = await sandbox().exec(["bash", "-c", prepare], cwd=directory)
            if not result.success:
                raise RuntimeError(f"workspace setup failed: {result.stderr}")
        return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def polyglot_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        metadata = state.metadata
        await _write(metadata["directory"], metadata["tests"])
        try:
            result = await sandbox().exec(
                ["bash", "-c", metadata["command"]],
                cwd=metadata["directory"],
                timeout=TEST_TIMEOUT,
            )
            passed, output = result.success, result.stdout + result.stderr
        except TimeoutError:
            passed, output = False, f"tests timed out after {TEST_TIMEOUT} seconds"
        return Score(
            value=CORRECT if passed else INCORRECT,
            explanation=output[-4000:],
        )

    return score

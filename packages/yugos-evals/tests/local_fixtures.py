"""Synthetic packs, real `yugos pack` bundles and a scripted provider; no network or spend."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from evals_fakes import freeze_fixture as freeze_task
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.model import get_model as inspect_get_model
from inspect_ai.scorer import exact, model_graded_qa
from inspect_ai.solver import generate
from safetensors.numpy import save_file

from yugos.evaluation.packs import PackManifest, records_bytes, write_pack
from yugos.miner import main as yugos_cli

CARD = Path(__file__).parent / "fixtures" / "world-card.json"
TEST_MODEL = {"provider_id": "openai", "model_id": "gpt-test"}
USAGE = {"input_tokens": 1000, "output_tokens": 500}
# 1000 input tokens at $1/M plus 500 output tokens at $4/M.
CALL_COST = 0.003

ROUTER = """
from yugos.inference.actions import ModelChoice

def load_model(weights_path):
    class Router:
        def route(self, input):
            prompt = input.messages[-1].content or ""
            if "crash" in prompt:
                raise RuntimeError("router bug")
            return ModelChoice(**MODEL)

    return Router()
"""


def make_bundle(root: Path, *, model=TEST_MODEL, source=ROUTER) -> Path:
    """Package a trivial router with the real `yugos pack` command."""
    root.mkdir(parents=True)
    (root / "model.py").write_text(f"MODEL = {model!r}\n{source}")
    save_file({"w": np.zeros(1, dtype=np.float32)}, str(root / "model.safetensors"))
    (root / "models.json").write_text(json.dumps([model]))
    output = root.parent / f"{root.name}.zip"
    yugos_cli(["pack", str(root), "--output", str(output)])
    return output


def exact_questions(source: str, category: str, items: dict[str, tuple[str, str]]):
    task = Task(
        dataset=[
            Sample(id=i, input=prompt, target=target) for i, (prompt, target) in items.items()
        ],
        solver=generate(),
        scorer=exact(),
        name=source,
    )
    return freeze_task(task, source=source, workload_class=category)


def graded_questions():
    task = Task(
        dataset=[Sample(id="g1", input="Explain why the sky is blue.", target="Rayleigh")],
        solver=generate(),
        scorer=model_graded_qa(),
        name="graded",
    )
    return freeze_task(task, source="graded", workload_class="chat_qa", required_roles=("grader",))


def default_questions():
    return (
        *exact_questions(
            "trivia",
            "chat_qa",
            {"t1": ("Capital of France?", "Paris"), "t2": ("crash the router", "x")},
        ),
        *exact_questions(
            "arithmetic",
            "math_code",
            {"m1": ("What is 2 + 2?", "4"), "m2": ("What is 3 * 3?", "9")},
        ),
    )


def make_pack(path: Path, questions=None, **manifest) -> Path:
    records = records_bytes(questions or default_questions())
    write_pack(
        path,
        PackManifest(
            pack_id="fixture",
            records_sha256=hashlib.sha256(records).hexdigest(),
            source_releases=("fixture",),
            data_visibility="public",
            purpose="development",
            **manifest,
        ),
        records,
    )
    return path


ANSWERS = {
    "Capital of France?": "Paris",
    "What is 2 + 2?": "4",
    "What is 3 * 3?": "10",
    "Explain why the sky is blue.": "Rayleigh scattering.",
}


class ScriptedProvider:
    """Stand-in for `inspect_ai.model.get_model`: Inspect's mockllm with scripted answers."""

    def __init__(self, answers=ANSWERS) -> None:
        self.answers = answers
        self.models: list[dict] = []

    def __call__(self, name, *, config, base_url, api_key, memoize, **model_args):
        self.models.append(
            {"name": name, "base_url": base_url, "api_key": api_key, "config": config, **model_args}
        )

        def respond(input, tools, tool_choice, config):
            prompt = input[-1].text
            if "GRADE" in prompt or "grade" in prompt.lower():
                content = "GRADE: C"
            else:
                content = self.answers[prompt]
            return ModelOutput(
                model=name,
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessageAssistant(content=content), stop_reason="stop"
                    )
                ],
                usage=ModelUsage(**USAGE, total_tokens=sum(USAGE.values())),
            )

        return inspect_get_model("mockllm/model", custom_outputs=respond, memoize=False)

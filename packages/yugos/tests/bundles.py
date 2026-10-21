"""Trusted model fixtures for local worker tests; no downloaded submission code."""

import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from safetensors.numpy import save

MODEL = {"provider_id": "fixture", "model_id": "general"}
MODEL_SOURCE = """
import re
import numpy as np
from safetensors.numpy import load_file

class Router:
    def __init__(self, weights):
        self.weights = weights

    def solve(self, input):
        prompt = input.messages[-1].content
        words = re.findall(r"[a-z]+", prompt.lower())
        features = np.array([words.count(word) for word in ("equation", "python", "function")])
        choice = int(np.argmax(features @ self.weights))
        reply = yield {"kind": "call", "model": {"provider_id": "fixture", "model_id": ("math-model", "code-model")[choice]}, "input": input}
        if reply.finish_reason == "length":
            yield {"kind": "call", "model": {"provider_id": "fixture", "model_id": "general"}, "input": {"messages": [{"role": "user", "content": "Finish " + prompt + "\\nDraft:\\n" + reply.text}]}}

def load_model(weights_path):
    return Router(load_file(weights_path)["weights"])
"""


def make_bundle(
    root: Path,
    *,
    model_source=MODEL_SOURCE,
    manifest_changes=None,
    extra=None,
    reverse_weights=False,
    weights_data=None,
):
    root.mkdir(parents=True, exist_ok=True)
    training = [
        ("Solve an equation", 0),
        ("Find the roots of this equation", 0),
        ("Write a python function", 1),
        ("Debug this python function", 1),
    ]
    x = np.array(
        [
            [
                re.findall(r"[a-z]+", prompt.lower()).count(word)
                for word in ("equation", "python", "function")
            ]
            for prompt, _ in training
        ]
    )
    y = np.eye(2)[[label for _, label in training]]
    weights = np.linalg.lstsq(x, y, rcond=None)[0].astype(np.float32)
    if reverse_weights:
        weights = weights[:, ::-1].copy()
    payloads = {
        "model.py": model_source.encode(),
        "model.safetensors": save({"weights": weights}) if weights_data is None else weights_data,
    }
    manifest = {
        "schema": "yugos.bundle.v3",
        "name": "trained-fixture",
        "runtime": "python-safetensors-v2",
        "required_models": [
            MODEL,
            {"provider_id": "fixture", "model_id": "math-model"},
            {"provider_id": "fixture", "model_id": "code-model"},
        ],
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
        **(manifest_changes or {}),
    }
    manifest_bytes = json.dumps(manifest).encode()
    archive = root / "submission.zip"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("manifest.json", manifest_bytes)
        for name, data in {**payloads, **(extra or {})}.items():
            out.writestr(name, data)
    return (
        archive,
        hashlib.sha256(archive.read_bytes()).hexdigest(),
        hashlib.sha256(manifest_bytes).hexdigest(),
    )

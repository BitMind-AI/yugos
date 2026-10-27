"""Aider Polyglot exercises run every test and restore from frozen samples."""

import importlib
import json

from evals_fakes import DIGEST_IMAGE

from yugos.inspect_engine import restore_tasks
from yugos_evals.prepared import prepare_task

polyglot = importlib.import_module("yugos_evals.tasks.aider_polyglot")

JAVA = {
    ".meta/config.json": json.dumps({"files": {
        "solution": ["src/main/java/Cipher.java"], "test": ["src/test/java/CipherTest.java"],
        "example": [".meta/src/reference/java/Cipher.java"],
    }}).encode(),
    ".meta/src/reference/java/Cipher.java": b"class Cipher { /* reference */ }",
    ".docs/instructions.md": b"# Instructions\n\nEncode it.\n",
    "src/main/java/Cipher.java": b"class Cipher {}",
    "src/test/java/CipherTest.java": b'@Disabled("Remove to run test")\n@Test\nvoid b() {}\n',
    "gradle/wrapper/gradle-wrapper.jar": b"\xff\xfe jar",
}  # fmt: skip


def test_frozen_exercises_restore_with_all_tests_enabled_and_no_reference(monkeypatch):
    monkeypatch.setattr(polyglot, "_exercises", lambda revision: {"java/cipher": JAVA})
    task = polyglot.aider_polyglot(image=DIGEST_IMAGE)
    [question] = prepare_task(task, source="aider_polyglot", workload_class="agentic")
    [(restored, _)] = restore_tasks([question])
    [sample] = restored.dataset
    metadata = sample.metadata
    assert "@Disabled" not in metadata["tests"]["src/test/java/CipherTest.java"]
    assert not any(path.startswith(".") for path in metadata["files"])
    assert "reference" not in json.dumps(metadata["files"])
    assert metadata["example"] == {"src/main/java/Cipher.java": "class Cipher { /* reference */ }"}
    assert list(metadata["binary_files"]) == ["gradle/wrapper/gradle-wrapper.jar"]
    assert "Cipher.java" in sample.input and "./gradlew test --offline" in sample.input
    config = restored.sandbox.model_dump(mode="json")["config"]
    assert config["services"]["default"]["image"] == DIGEST_IMAGE
    assert "polyglot_setup" in str(restored.setup) and "polyglot_scorer" in str(restored.scorer)

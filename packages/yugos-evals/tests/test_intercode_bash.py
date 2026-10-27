"""InterCode-Bash restores from frozen samples and keeps InterCode's reward."""

import importlib
import json

from evals_fakes import DIGEST_IMAGE

from yugos.inspect_engine import restore_tasks
from yugos_evals.prepared import prepare_task

intercode = importlib.import_module("yugos_evals.tasks.intercode_bash")


def test_reward_is_full_only_for_the_gold_changes_and_output():
    changes = [("testbed/new.txt", "??")]
    hashes = {"testbed/new.txt": "abc  testbed/new.txt\n"}
    assert intercode.reward(changes, changes, hashes, hashes, "a.txt b.txt", "a.txt b.txt")[0] == 1
    wrong = {"testbed/new.txt": "def  testbed/new.txt\n"}
    total, parts = intercode.reward(changes, changes, wrong, hashes, "", "a.txt b.txt")
    assert parts == {"file_diff": 0.33, "file_changes": 0.0, "answer_similarity": 0.0}
    assert total == 0.34


def test_frozen_samples_restore_their_scorer(tmp_path, monkeypatch):
    rows = [{"query": "List the files", "gold": "ls /testbed"}]
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(rows))
    monkeypatch.setattr(intercode, "fetch", lambda *a, **k: path)
    task = intercode.intercode_bash(images=[DIGEST_IMAGE] * 4)
    assert len(task.dataset) == 4
    [question] = prepare_task(
        task, source="intercode_bash", workload_class="agentic", sample_ids=["fs2-0"]
    )
    [(restored, _)] = restore_tasks([question])
    assert restored.dataset[0].metadata["gold"] == "ls /testbed"
    assert "intercode_scorer" in str(restored.scorer)

"""Spider 2.0-Lite SQLite tasks restore and grade answer tables as Spider 2.0 does."""

import importlib
import json

from evals_fakes import DIGEST_IMAGE

from yugos.inspect_engine import restore_tasks
from yugos_evals.prepared import prepare_task

spider2 = importlib.import_module("yugos_evals.tasks.spider2_lite")

ROOT = "spider2-lite/"
FILES = {
    ROOT + "spider2-lite.jsonl": json.dumps(
        {"instance_id": "local001", "db": "Db-IMDB", "question": "Top film?", "external_knowledge": None}
    ).encode(),
    ROOT + "evaluation_suite/gold/spider2lite_eval.jsonl": json.dumps(
        {"instance_id": "local001", "condition_cols": [1], "ignore_order": True}
    ).encode(),
    ROOT + "evaluation_suite/gold/exec_result/local001_a.csv": b"title,score\nAlien,9.0\n",
    ROOT + "evaluation_suite/gold/exec_result/local001_b.csv": b"title,score\nAliens,8.9\n",
    ROOT + "resource/databases/sqlite/DB_IMDB/DDL.csv": b"table_name,DDL\n",
}  # fmt: skip


def test_answer_tables_match_any_gold_on_its_condition_columns(monkeypatch):
    monkeypatch.setattr(spider2, "_sources", lambda revision: FILES)
    task = spider2.spider2_lite(image=DIGEST_IMAGE)
    [question] = prepare_task(task, source="spider2_lite", workload_class="agentic")
    [(restored, _)] = restore_tasks([question])
    [sample] = restored.dataset
    assert "/workspace/schema/DB_IMDB/" in sample.input and "Db-IMDB.sqlite" in sample.input
    assert spider2.result_matches("name,rating\nwhatever,8.905\n", sample.metadata)
    assert not spider2.result_matches("name,rating\nAlien,7.0\n", sample.metadata)
    assert "spider2_scorer" in str(restored.scorer)

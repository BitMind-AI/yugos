"""Application progress survives failed writes and concurrent writers are excluded."""

import json

import pytest

from yugos.state import save_json, state_lock


def test_state_lock_rejects_concurrent_writer_and_releases_on_failure(tmp_path):
    path = tmp_path / "state.json"
    with pytest.raises(RuntimeError, match="another process"):
        with state_lock(path), state_lock(path):
            pytest.fail("concurrent writer acquired lock")
    with state_lock(path):
        pass


def test_failed_serialization_preserves_saved_state_and_cleans_temporary_file(tmp_path):
    path = tmp_path / "progress" / "state.json"
    save_json(path, {"result": "café"})
    previous = path.read_bytes()

    # A writer that truncates the destination first would destroy resumable progress.
    with pytest.raises(TypeError):
        save_json(path, {"invalid": object()})
    assert path.read_bytes() == previous
    assert list(path.parent.iterdir()) == [path]

    updated = {"result": "next", "complete": True}
    save_json(path, updated)
    assert json.loads(path.read_text()) == updated
    assert list(path.parent.iterdir()) == [path]

"""Known zero and a missing bill stay different values, including on old JSON."""

import json
from decimal import Decimal
from pathlib import Path

from yugos.competition import CategoryMeasurement, CategoryRule, spend_error
from yugos.spend import KnownSpend, OpenReservation

LEGACY_KEYS = {
    "run_id": "r",
    "miner_id": "r",
    "submission_sha256": "a" * 64,
    "cohort_sha256": "b" * 64,
    "category": "chat_qa",
    "grades": {"q": "1"},
    "question_spend_usd": {"q": "0"},
    "execution_time_ns": 1_000_000_000,
    "execution_complete": True,
    "integrity_pass": True,
}


def rule() -> CategoryRule:
    return CategoryRule(
        category="chat_qa",
        cohort_sha256="b" * 64,
        question_ids=("q",),
        initial_quality=Decimal(0),
        panel_leader_time_ns=1,
    )


def load_unchanged(tmp_path: Path, name: str, payload: dict) -> CategoryMeasurement:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    raw = path.read_bytes()
    loaded = CategoryMeasurement.model_validate(json.loads(raw))
    assert path.read_bytes() == raw
    return loaded


def test_legacy_unresolved_zero_blocks_and_is_not_rewritten(tmp_path):
    # Catches a loader that treats cost 0 as a free completed bill, or rewrites the file.
    row = load_unchanged(
        tmp_path,
        "unresolved.json",
        {
            **LEGACY_KEYS,
            "cost_usd": "0",
            "cost_complete": False,
            "reservation_usd": "0.25",
        },
    )
    assert isinstance(row.spend, OpenReservation)
    assert row.spend.reservation_usd == Decimal("0.25")
    assert "usd" not in type(row.spend).model_fields
    assert row.cost_usd == 0
    assert "unresolved" in rule().measurement_errors(row)


def test_legacy_known_zero_stays_comparable(tmp_path):
    # Catches a known zero being stored as an open reservation.
    row = load_unchanged(
        tmp_path,
        "known-zero.json",
        {**LEGACY_KEYS, "cost_usd": "0", "cost_complete": True},
    )
    assert row.spend == KnownSpend(usd=Decimal(0))
    assert spend_error(row) is None
    assert "unresolved" not in rule().measurement_errors(row)


def test_legacy_positive_known_cost_can_compare(tmp_path):
    # Catches a priced bill failing the spend check.
    payload = {
        **LEGACY_KEYS,
        "cost_usd": "4",
        "cost_complete": True,
        "question_spend_usd": {"q": "4"},
    }
    row = load_unchanged(tmp_path, "known-positive.json", payload)
    assert row.spend == KnownSpend(usd=Decimal(4))
    assert rule().measurement_errors(row) == ()

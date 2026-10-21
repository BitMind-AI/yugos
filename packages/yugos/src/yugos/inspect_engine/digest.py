"""A canonical, replayable digest of an Inspect run and its provider receipts.

`.eval` files embed timestamps, random IDs and non-canonical compression, so
their bytes cannot be compared across reruns or replays. This digest commits
to what was evaluated and observed instead: task identity, each sample's
input, target, output and grade, and the metering receipts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from inspect_ai.log import EvalLog, EvalSample

from yugos.competition import digest

from .isolation import sample_input_text
from .receipts import ProviderReceipt, receipts_by_sample
from .versions import PINNED

RECEIPT_VERSION = "yugos-inspect-receipt-v1"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _target_text(sample: EvalSample) -> str:
    return sample.target if isinstance(sample.target, str) else json.dumps(sample.target)


def _output_text(sample: EvalSample) -> str:
    output = sample.output
    calls = [
        {"function": call.function, "arguments": call.arguments}
        for choice in output.choices
        for call in choice.message.tool_calls or ()
    ]
    stop = output.stop_reason if output.choices else None
    return json.dumps(
        {"completion": output.completion, "stop_reason": stop, "tool_calls": calls},
        sort_keys=True,
    )


def _sample_record(sample: EvalSample) -> dict[str, Any]:
    return {
        "id": str(sample.id),
        "epoch": sample.epoch,
        "input_sha256": _sha256(sample_input_text(sample)),
        "target_sha256": _sha256(_target_text(sample)),
        "output_sha256": _sha256(_output_text(sample)),
        "scores": {
            name: {"value": score.value, "code": (score.metadata or {}).get("code")}
            for name, score in sorted((sample.scores or {}).items())
        },
        "error": sample.error is not None,
        "limit": None if sample.limit is None else sample.limit.type,
    }


def canonical_receipt(
    log: EvalLog, receipts: Iterable[ProviderReceipt], *, dataset_revision: str
) -> dict[str, Any]:
    """The JSON document `receipt_digest` hashes.

    Receipts are attached to samples by (id, epoch) rather than Inspect's random
    sample UUIDs. Logs edited after the run are rejected, since their scores no
    longer come from the recorded run.
    """
    if log.invalidated or log.log_updates or log.config_updates:
        raise ValueError("edited or invalidated logs cannot be committed")
    if not dataset_revision:
        raise ValueError("a dataset revision is required")
    spec = log.eval
    samples = sorted(log.samples or [], key=lambda s: (str(s.id), s.epoch))
    keys = {sample.uuid: (str(sample.id), sample.epoch) for sample in samples}
    grouped = receipts_by_sample(receipts)
    if not grouped.keys() <= keys.keys():
        raise ValueError("a receipt cannot be attributed to a logged sample")
    receipt_rows = sorted(
        (
            {
                "sample": list(keys[uuid]),
                **receipt.model_dump(mode="json", exclude={"sample_uuid"}),
            }
            for uuid, rows in grouped.items()
            for receipt in rows
        ),
        key=lambda row: (row["sample"], row["started_ns"], row["receipt_id"]),
    )
    return {
        "version": RECEIPT_VERSION,
        "task": {
            "pins": PINNED,
            "packages": dict(sorted(spec.packages.items())),
            "name": spec.task,
            "version": spec.task_version,
            "args": spec.task_args,
            "dataset_revision": dataset_revision,
            "epochs": spec.config.epochs,
            "model": spec.model,
            "model_roles": {
                role: [c.model for c in config] if isinstance(config, list) else config.model
                for role, config in sorted((spec.model_roles or {}).items())
            },
            "scorers": sorted(scorer.name for scorer in spec.scorers or ()),
        },
        "status": log.status,
        "samples": [_sample_record(sample) for sample in samples],
        "receipts": receipt_rows,
    }


def receipt_digest(
    log: EvalLog, receipts: Iterable[ProviderReceipt], *, dataset_revision: str
) -> str:
    """SHA-256 of the canonical receipt; equal for identical runs and replays."""
    return digest(canonical_receipt(log, receipts, dataset_revision=dataset_revision))

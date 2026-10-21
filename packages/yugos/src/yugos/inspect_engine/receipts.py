"""Provider-call receipts: the operator's record of what each sample actually spent."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

Name = Annotated[str, Field(min_length=1)]


class ProviderReceipt(BaseModel):
    """One upstream provider call, priced by the metering proxy, never by the router."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    receipt_id: Name
    # Equals `EvalSample.uuid`; the RouterModel supplies it with every router call.
    sample_uuid: Name
    provider: Name
    model: Name
    reasoning_effort: str | None = None
    cost_usd: Decimal = Field(ge=0)
    # Metering-proxy monotonic clock; only differences within one proxy are meaningful.
    started_ns: int = Field(ge=0, strict=True)
    ended_ns: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def ordered(self):
        if self.ended_ns < self.started_ns:
            raise ValueError("a receipt cannot end before it starts")
        return self


class ReceiptSink(Protocol):
    """Durable destination for receipts; `append` returns only after the receipt is kept."""

    def append(self, receipt: ProviderReceipt) -> None: ...


class MemoryReceipts:
    """In-process receipt log for tests and replays."""

    def __init__(self, receipts: Iterable[ProviderReceipt] = ()) -> None:
        self._lock = threading.Lock()
        self._receipts = list(receipts)

    def append(self, receipt: ProviderReceipt) -> None:
        with self._lock:
            self._receipts.append(ProviderReceipt.model_validate(receipt))

    def receipts(self) -> tuple[ProviderReceipt, ...]:
        with self._lock:
            return tuple(self._receipts)


class JsonlReceipts:
    """Append-only JSON Lines receipt log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, receipt: ProviderReceipt) -> None:
        line = ProviderReceipt.model_validate(receipt).model_dump_json() + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as file:
            file.write(line)
            file.flush()

    def receipts(self) -> tuple[ProviderReceipt, ...]:
        if not self.path.exists():
            return ()
        with self.path.open(encoding="utf-8") as file:
            return tuple(ProviderReceipt.model_validate(json.loads(line)) for line in file if line)


def receipts_by_sample(
    receipts: Iterable[ProviderReceipt],
) -> dict[str, tuple[ProviderReceipt, ...]]:
    """Group receipts by sample, rejecting a receipt recorded twice."""
    grouped: dict[str, list[ProviderReceipt]] = {}
    seen: set[str] = set()
    for receipt in receipts:
        if receipt.receipt_id in seen:
            raise ValueError(f"duplicate receipt {receipt.receipt_id}")
        seen.add(receipt.receipt_id)
        grouped.setdefault(receipt.sample_uuid, []).append(receipt)
    return {uuid: tuple(rows) for uuid, rows in grouped.items()}

"""A known bill or an open reservation. Zero is a price, not a missing bill."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class KnownSpend(BaseModel):
    """A bill the provider priced. Zero is a real charge, not a missing one."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    status: Literal["known"] = "known"
    usd: Decimal = Field(ge=0)


QUOTE_FLOOR = Decimal("0.01")


class OpenReservation(BaseModel):
    """A dispatched call whose bill is not known. The held quote is not spend."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    status: Literal["unresolved"] = "unresolved"
    reservation_usd: Decimal = Field(ge=QUOTE_FLOOR)

    @model_validator(mode="before")
    @classmethod
    def floor_subcent_quote(cls, data: Any) -> Any:
        """A non-negative per-call quote below one cent is reserved as one cent."""
        if not isinstance(data, dict) or "reservation_usd" not in data:
            return data
        data = dict(data)
        amount = Decimal(str(data["reservation_usd"]))
        if amount >= 0 and amount < QUOTE_FLOOR:
            data["reservation_usd"] = QUOTE_FLOOR
        return data


Spend = Annotated[KnownSpend | OpenReservation, Field(discriminator="status")]


def apply_legacy_spend(data: Any, *, amount_key: str, drop_amount: bool) -> Any:
    """Read old cost keys into `Spend` without changing the caller's mapping."""
    if not isinstance(data, dict):
        return data
    data = dict(data)
    if "spend" in data:
        data.pop("cost_complete", None)
        if drop_amount:
            data.pop(amount_key, None)
        data.pop("reservation_usd", None)
        return data
    if "cost_complete" not in data and (amount_key not in data or not drop_amount):
        return data
    complete = data.pop("cost_complete", True)
    amount = data.pop(amount_key) if drop_amount else data.get(amount_key, 0)
    if complete:
        data.pop("reservation_usd", None)
        data["spend"] = {"status": "known", "usd": amount}
        return data
    reservation = data.pop("reservation_usd", None)
    if reservation is None:
        parsed = Decimal(str(amount))
        reservation = parsed if parsed > 0 else None
    if reservation is None or Decimal(str(reservation)) < 0:
        raise ValueError("unresolved spend requires a positive reservation_usd")
    data["spend"] = {"status": "unresolved", "reservation_usd": reservation}
    return data

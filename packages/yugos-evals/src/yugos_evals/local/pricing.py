"""World-card token prices and reviewed limits, applied exactly as the hosted meter does.

Prices are USD per million tokens with optional cache rates and context bands; a
call's cost is its token counts times the band's rates, rounded up to a whole
microdollar. Limits bound a call's cost before it is dispatched. These mirror
Ycore's tariff, billing and reservation rules; a count without a price is unpriced
spend, never free.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from decimal import ROUND_CEILING, Decimal

from inspect_ai.model import ModelUsage
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TokenRates(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    input_usd_per_million: float = Field(ge=0)
    output_usd_per_million: float = Field(ge=0)
    cached_usd_per_million: float | None = Field(default=None, ge=0)
    cache_write_usd_per_million: float | None = Field(default=None, ge=0)
    cache_write_1h_usd_per_million: float | None = Field(default=None, ge=0)

    @field_validator(
        "input_usd_per_million",
        "output_usd_per_million",
        "cached_usd_per_million",
        "cache_write_usd_per_million",
        "cache_write_1h_usd_per_million",
        mode="before",
    )
    @classmethod
    def numeric_price(cls, value):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError("prices must be JSON numbers")
        return value


class ContextPriceBand(TokenRates):
    # Exclusive threshold on total input, including cached tokens.
    input_tokens_above: int = Field(strict=True, ge=0)


class TokenPrices(TokenRates):
    context_bands: tuple[ContextPriceBand, ...] = ()

    @model_validator(mode="after")
    def ordered_bands(self):
        thresholds = [band.input_tokens_above for band in self.context_bands]
        if thresholds != sorted(set(thresholds)):
            raise ValueError("context bands must have distinct ascending thresholds")
        return self

    def for_input_tokens(self, tokens: int) -> TokenRates:
        selected: TokenRates = self
        for band in self.context_bands:
            if tokens <= band.input_tokens_above:
                break
            selected = band
        return selected


class InputTokenBound(BaseModel):
    """Reviewed input accounting: tokens per request byte plus provider framing."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    tokens_per_byte: int = Field(strict=True, ge=1)
    overhead_tokens: int = Field(strict=True, ge=0)


class ModelLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_input_tokens: int = Field(strict=True, gt=0)
    # Total billable output, including reasoning, at the native provider ceiling.
    max_output_tokens: int = Field(strict=True, gt=0)
    source: str = Field(min_length=1)
    valid_until: date
    input_token_bound: InputTokenBound | None = None

    def require_current(self) -> None:
        if self.valid_until < datetime.now(UTC).date():
            raise ValueError("model limit review expired; refresh the world card")

    def input_tokens(self, request_json: object) -> int:
        """Bound a request's input tokens; without a reviewed rule, the model's input ceiling."""
        if self.input_token_bound is None:
            return self.max_input_tokens
        payload = json.dumps(
            request_json, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        bound = self.input_token_bound
        return min(
            self.max_input_tokens, len(payload) * bound.tokens_per_byte + bound.overhead_tokens
        )


def reservation_microusd(prices: TokenPrices, input_tokens: int, output_tokens: int) -> int:
    """Upper bound, in microdollars, on a call with these token ceilings across price bands."""
    bands = (prices, *prices.context_bands)
    worst = Decimal(0)
    for index, band in enumerate(bands):
        if index and input_tokens <= band.input_tokens_above:
            break
        upper = input_tokens
        if index + 1 < len(bands):
            upper = min(upper, bands[index + 1].input_tokens_above)
        input_rate = max(
            rate
            for rate in (
                band.input_usd_per_million,
                band.cached_usd_per_million,
                band.cache_write_usd_per_million,
                band.cache_write_1h_usd_per_million,
            )
            if rate is not None
        )
        cost = upper * Decimal(str(input_rate)) + output_tokens * Decimal(
            str(band.output_usd_per_million)
        )
        worst = max(worst, cost)
    return max(1, int(worst.to_integral_value(rounding=ROUND_CEILING)))


def usage_cost_microusd(usage: ModelUsage | None, prices: TokenPrices) -> int | None:
    """Price Inspect's normalized usage; None when usage or a needed rate is missing.

    Inspect reports uncached input, cache reads and cache writes separately and
    includes reasoning in output. Cache writes are priced at the short-lived
    write rate; Inspect does not distinguish one-hour writes.
    """
    if usage is None:
        return None
    cached = usage.input_tokens_cache_read or 0
    writes = usage.input_tokens_cache_write or 0
    counts = (usage.input_tokens, usage.output_tokens, cached, writes)
    if any(type(count) is not int or count < 0 for count in counts):
        return None
    rates = prices.for_input_tokens(usage.input_tokens + cached + writes)
    amount = Decimal(0)
    for count, rate in (
        (usage.input_tokens, rates.input_usd_per_million),
        (usage.output_tokens, rates.output_usd_per_million),
        (cached, rates.cached_usd_per_million),
        (writes, rates.cache_write_usd_per_million),
    ):
        if count:
            if rate is None:
                return None
            amount += count * Decimal(str(rate))
    return int(amount.to_integral_value(rounding=ROUND_CEILING))


def card_token_prices(card: Mapping) -> dict[tuple[str, str], TokenPrices]:
    """Every priced (provider, model) in a world card; outdated price formats are rejected."""
    providers = card.get("providers")
    if not isinstance(providers, list) or not providers:
        raise ValueError("world card providers required")
    result, seen = {}, set()
    for provider in providers:
        if not isinstance(provider, Mapping):
            raise ValueError("invalid world card provider")
        pid = provider.get("provider_id")
        if not isinstance(pid, str) or not pid or pid in seen:
            raise ValueError("world card provider IDs must be distinct")
        seen.add(pid)
        if "price_card" in provider or "cache_price_card" in provider:
            raise ValueError("outdated price format; supply token_prices")
        models = provider.get("token_prices")
        if not isinstance(models, Mapping) or not models:
            raise ValueError("provider token_prices required")
        for model, rates in models.items():
            if not isinstance(model, str) or not model:
                raise ValueError("priced model ID required")
            result[pid, model] = TokenPrices.model_validate(rates)
    return result


def card_model_limits(card: Mapping) -> dict[tuple[str, str], ModelLimits]:
    """Reviewed token limits for each (provider, model) the card lists them for."""
    return {
        (provider["provider_id"], model): ModelLimits.model_validate(limits)
        for provider in card["providers"]
        for model, limits in (provider.get("model_limits") or {}).items()
    }

"""The world card: which provider models a router may call, and what each call costs.

A card is loaded from a file or URL, identified by the same SHA-256 the hosted
service publishes, and cached under that digest. A pack that pins a card is run
against exactly that card.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from yugos.inference.actions import ModelChoice
from yugos.service_client import DEFAULT_MINER_ENDPOINT, service_url

from .pricing import ModelLimits, TokenPrices, card_model_limits, card_token_prices

API_FORMATS = ("openai", "openai_responses", "anthropic", "google")
MAX_CARD_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class ProviderAccess:
    """How to reach one provider: endpoint, wire format and the env var holding its key."""

    provider_id: str
    api_base: str
    api_format: str
    key_env: str
    extra_headers: Mapping[str, str] = field(default_factory=dict)


def _openai_compatible(provider_id: str, api_base: str, key_env: str) -> ProviderAccess:
    return ProviderAccess(provider_id, api_base, "openai", key_env)


# The hosted service's provider endpoints; a card's provider fields override them.
DEFAULT_PROVIDERS: dict[str, ProviderAccess] = {
    "openai": ProviderAccess("openai", "https://api.openai.com/v1", "openai", "OPENAI_API_KEY"),
    "anthropic": ProviderAccess(
        "anthropic", "https://api.anthropic.com/v1", "anthropic", "ANTHROPIC_API_KEY"
    ),
    "google": ProviderAccess(
        "google",
        "https://generativelanguage.googleapis.com/v1beta",
        "google",
        "GOOGLE_API_KEY",
    ),
    "openrouter": ProviderAccess(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "openai",
        "OPENROUTER_API_KEY",
        {"HTTP-Referer": "https://github.com/BitMind-AI/yugos", "X-Title": "yugos-evaluation"},
    ),
    "venice": _openai_compatible("venice", "https://api.venice.ai/api/v1", "VENICE_API_KEY"),
    "gm": _openai_compatible("gm", "https://api.saygm.com/v1", "GM_API_KEY"),
    "engy": _openai_compatible("engy", "https://api.engy.ai/v1", "ENGY_API_KEY"),
    "spacexai": _openai_compatible("spacexai", "https://api.x.ai/v1", "XAI_API_KEY"),
    "nous": _openai_compatible("nous", "https://inference-api.nousresearch.com/v1", "NOUS_API_KEY"),
    "deepseek": _openai_compatible("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "chutes": _openai_compatible("chutes", "https://llm.chutes.ai/v1", "CHUTES_API_KEY"),
}


def world_card_payload(card: Mapping[str, Any]) -> dict[str, Any]:
    """The fields that define metering: ids, roles, aliases and provider prices."""
    return {
        "world_state_id": card.get("world_state_id") or "ws-live",
        "roles": card.get("roles") or {},
        "aliases": card.get("aliases") or {},
        "providers": card.get("providers") or [],
    }


def world_card_sha256(card: Mapping[str, Any]) -> str:
    """The card's published identity."""
    raw = json.dumps(
        world_card_payload(card), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class WorldCard:
    """A validated world card: its priced model pool, reviewed limits and provider access."""

    def __init__(self, card: Mapping[str, Any]) -> None:
        self.card = dict(card)
        self.sha256 = world_card_sha256(card)
        self.prices: dict[tuple[str, str], TokenPrices] = card_token_prices(card)
        self.limits: dict[tuple[str, str], ModelLimits] = card_model_limits(card)
        self._providers = {p["provider_id"]: p for p in card["providers"]}

    def pool(self) -> frozenset[ModelChoice]:
        """Model choices the card admits with both prices and reviewed limits."""
        choices = set()
        for pid, provider in self._providers.items():
            listed = provider.get("models")
            if listed is None:
                listed = [
                    {"model_id": m, "effort_levels": [None]} for m in provider["token_prices"]
                ]
            for model in listed:
                key = (pid, model["model_id"])
                if key not in self.prices or key not in self.limits:
                    continue
                for effort in model.get("effort_levels") or [None]:
                    choices.add(ModelChoice(provider_id=pid, model_id=key[1], effort=effort))
        return frozenset(choices)

    def provider(self, provider_id: str) -> ProviderAccess:
        """How to reach a card provider; unknown endpoints or wire formats are refused."""
        fields = self._providers.get(provider_id)
        if fields is None:
            raise ValueError(f"world card has no provider {provider_id!r}")
        default = DEFAULT_PROVIDERS.get(provider_id)
        api_base = fields.get("api_base") or (default and default.api_base)
        # A card provider with only an endpoint is taken to be OpenAI-compatible.
        api_format = fields.get("api_format") or (default.api_format if default else "openai")
        key_env = fields.get("key_env") or (default and default.key_env)
        if not api_base or not key_env:
            raise ValueError(
                f"provider {provider_id!r} is not supported locally: the world card gives "
                "no api_base/key_env and it is not a known provider"
            )
        if api_format not in API_FORMATS:
            raise ValueError(f"provider {provider_id!r} uses unsupported api_format {api_format!r}")
        return ProviderAccess(
            provider_id,
            api_base.rstrip("/"),
            api_format,
            key_env,
            dict(fields.get("extra_headers") or (default.extra_headers if default else {})),
        )


def default_cache_dir() -> Path:
    root = os.environ.get("YUGOS_EVALS_CACHE") or Path.home() / ".cache" / "yugos-evals"
    return Path(root) / "world-cards"


def default_card_url(sha256: str | None) -> str:
    endpoint = os.environ.get("YUGOS_ENDPOINT") or DEFAULT_MINER_ENDPOINT
    return f"{service_url(endpoint)}/world/cards/{sha256 or 'active'}"


def fetch_url(url: str) -> bytes:
    """GET a card over HTTPS (or localhost HTTP) with a bounded response."""
    import httpx

    service_url(url)
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_bytes():
                data.extend(chunk)
                if len(data) > MAX_CARD_BYTES:
                    raise ValueError("world card exceeds the size limit")
    return bytes(data)


def _parse(raw: bytes) -> dict[str, Any]:
    """Accept a bare card or the service's `{"sha256": ..., "card": {...}}` envelope."""
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("world card must be a JSON object")
    if "providers" not in value and isinstance(value.get("card"), dict):
        card = value["card"]
        if value.get("sha256") and value["sha256"] != world_card_sha256(card):
            raise ValueError("world card does not match its published digest")
        return card
    return value


def load_world_card(
    source: str | None,
    *,
    pinned: str | None = None,
    cache_dir: Path | None = None,
    fetch: Callable[[str], bytes] = fetch_url,
) -> WorldCard:
    """Load the card from `source` (a path or URL), the pinned cache entry, or the service.

    With a pinned digest, only that card is accepted, and a cached copy is used
    without fetching. Without a source or pin, the service's active card is used.
    """
    cache = cache_dir or default_cache_dir()
    if source is None and pinned is not None and (cache / f"{pinned}.json").is_file():
        source = str(cache / f"{pinned}.json")
    if source is None:
        raw = fetch(default_card_url(pinned))
    elif source.startswith(("https://", "http://")):
        raw = fetch(source)
    else:
        raw = Path(source).expanduser().read_bytes()
    card = WorldCard(_parse(raw))
    if pinned is not None and card.sha256 != pinned:
        raise ValueError(
            f"the pack pins world card {pinned}, but {source or 'the service'} "
            f"provided {card.sha256}"
        )
    target = cache / f"{card.sha256}.json"
    if not target.exists():
        cache.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(card.card, sort_keys=True, indent=2))
        temporary.replace(target)
    return card

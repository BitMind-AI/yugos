"""World cards, prices and provider mapping behave like the hosted meter, offline."""

import json

import pytest
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelUsage
from inspect_ai.util import SandboxEnvironmentSpec
from inspect_ai.util._sandbox.compose import ComposeConfig
from local_fixtures import CARD

from yugos.inference.actions import ModelChoice
from yugos.inference.calls import ModelRequest
from yugos.inference.chat import ModelInput
from yugos.inspect_engine.providers import inspect_generation
from yugos_evals.local import providers
from yugos_evals.local.pricing import TokenPrices, reservation_microusd, usage_cost_microusd
from yugos_evals.local.providers import LocalProviderClient, MissingCredentials, generation
from yugos_evals.local.router import worker_environment
from yugos_evals.local.runner import LocalRunError, use_local_sandboxes
from yugos_evals.local.world import WorldCard, load_world_card, world_card_sha256

RAW = json.loads(CARD.read_text())
PRICED = {"m": {"input_usd_per_million": 1, "output_usd_per_million": 1}}


def test_card_is_cached_by_digest_and_verified_against_its_envelope(tmp_path, monkeypatch):
    monkeypatch.setenv("YUGOS_ENDPOINT", "https://example.test/upload")
    fetched = []

    def fetch(url):
        fetched.append(url)
        return json.dumps({"sha256": world_card_sha256(RAW), "card": RAW}).encode()

    card = load_world_card(
        "https://example.test/world/cards/active", cache_dir=tmp_path, fetch=fetch
    )
    assert card.sha256 == world_card_sha256(RAW)
    assert (tmp_path / f"{card.sha256}.json").is_file()
    # A pinned card already in the cache is used without fetching.
    again = load_world_card(None, pinned=card.sha256, cache_dir=tmp_path, fetch=fetch)
    assert again.sha256 == card.sha256 and len(fetched) == 1
    # Without a source or cached pin, the service's card for that pin is fetched.
    load_world_card(None, pinned=card.sha256, cache_dir=tmp_path / "empty", fetch=fetch)
    assert fetched[-1] == f"https://example.test/world/cards/{card.sha256}"

    def forged(url):
        return json.dumps({"sha256": "0" * 64, "card": RAW}).encode()

    with pytest.raises(ValueError, match="published digest"):
        load_world_card("https://example.test/card", cache_dir=tmp_path, fetch=forged)


def test_pool_admits_only_priced_models_with_reviewed_limits():
    pool = WorldCard(RAW).pool()
    assert ModelChoice(provider_id="openai", model_id="gpt-test", effort="low") in pool
    assert ModelChoice(provider_id="openai", model_id="gpt-unreviewed") not in pool
    assert ModelChoice(provider_id="venice", model_id="open-model") in pool


def test_usage_is_priced_like_the_hosted_tariff():
    prices = TokenPrices.model_validate(
        {
            "input_usd_per_million": 1.0,
            "output_usd_per_million": 4.0,
            "cached_usd_per_million": 0.25,
            "context_bands": [
                {
                    "input_tokens_above": 2000,
                    "input_usd_per_million": 2.0,
                    "output_usd_per_million": 8.0,
                }
            ],
        }
    )
    usage = ModelUsage(input_tokens=1000, output_tokens=500, input_tokens_cache_read=400)
    assert usage_cost_microusd(usage, prices) == 1000 + 2000 + 100
    # Total input above the band threshold moves every token to the band's rates.
    big = ModelUsage(input_tokens=2001, output_tokens=1)
    assert usage_cost_microusd(big, prices) == 2001 * 2 + 8
    # The band has no cache rate, so band-priced cache reads are unpriced, never free.
    assert (
        usage_cost_microusd(ModelUsage(input_tokens=2000, input_tokens_cache_read=1), prices)
        is None
    )
    assert usage_cost_microusd(None, prices) is None
    # Fractional microdollars round up.
    assert (
        usage_cost_microusd(
            ModelUsage(input_tokens=1, output_tokens=0),
            TokenPrices(input_usd_per_million=0.1, output_usd_per_million=0),
        )
        == 1
    )
    assert reservation_microusd(prices, 3000, 10) >= usage_cost_microusd(
        ModelUsage(input_tokens=3000, output_tokens=10), prices
    )


def request(model, **changes):
    return ModelRequest(
        call_id="c",
        model=model,
        input=ModelInput.from_text("hello"),
        timeout_s=10,
        **changes,
    )


def test_quotes_bound_calls_and_keys_are_required():
    card = WorldCard(RAW)
    model = ModelChoice(provider_id="openai", model_id="gpt-test")
    with pytest.raises(MissingCredentials, match="OPENAI_API_KEY"):
        LocalProviderClient(card, [model], env={})
    client = LocalProviderClient(card, [model], env={"OPENAI_API_KEY": "k"})
    quote = client.quote(request(model))
    assert 0 < quote < 0.01
    with pytest.raises(ValueError, match="output ceiling"):
        client.quote(request(model, max_output_tokens=5000))
    with pytest.raises(ValueError, match="not admitted"):
        client.quote(request(ModelChoice(provider_id="openai", model_id="gpt-judge")))


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ({"provider_id": "openai"}, "openai/m"),
        ({"provider_id": "anthropic"}, "anthropic/m"),
        ({"provider_id": "google"}, "google/m"),
        ({"provider_id": "deepseek"}, "deepseek/m"),
        ({"provider_id": "openrouter"}, "openrouter/m"),
        ({"provider_id": "venice"}, "openai-api/venice/m"),
        (
            {"provider_id": "acme", "api_base": "https://acme.test/v1", "key_env": "ACME_KEY"},
            "openai-api/acme/m",
        ),
    ],
)
def test_world_card_providers_use_the_hosted_inspect_mapping(provider, expected):
    card = WorldCard({"providers": [{**provider, "token_prices": PRICED}]})
    access = card.provider(provider["provider_id"])
    model = ModelChoice(provider_id=access.provider_id, model_id="m", effort="low")
    spec = generation(access, model, ModelInput.from_text("hi"), 100)
    assert spec.model_name == expected
    # The local runner must not keep its own effort or sampling mapping.
    assert spec == inspect_generation(
        access.provider_id,
        access.api_format,
        model,
        ModelInput.from_text("hi"),
        max_output_tokens=100,
    )
    assert spec.config.reasoning_effort == "low"


def test_unknown_providers_and_efforts_are_refused():
    card = WorldCard({"providers": [{"provider_id": "acme", "token_prices": PRICED}]})
    with pytest.raises(ValueError, match="not supported locally"):
        card.provider("acme")
    raw = json.loads(json.dumps(RAW))
    raw["providers"][0]["models"][0]["effort_levels"] = ["extreme"]
    extreme = ModelChoice(provider_id="openai", model_id="gpt-test", effort="extreme")
    with pytest.raises(ValueError, match="reasoning_effort"):
        LocalProviderClient(WorldCard(raw), [extreme], env={"OPENAI_API_KEY": "k"})


def test_calls_reach_inspect_with_the_shared_settings_unchanged(monkeypatch):
    seen = {}

    def fake_get_model(name, **kwargs):
        seen.update(name=name, **kwargs)
        raise RuntimeError("stop before dispatch")

    monkeypatch.setattr(providers, "get_model", fake_get_model)
    model = ModelChoice(provider_id="openai", model_id="gpt-test", effort="low")
    client = LocalProviderClient(WorldCard(RAW), [model], env={"OPENAI_API_KEY": "k"})
    sampled = ModelInput.from_text("hello").model_copy(update={"temperature": 0.0, "seed": 3})
    call = request(model, max_output_tokens=500).model_copy(update={"input": sampled})
    response = client.complete(call)
    assert response.error and "stop before dispatch" in response.error
    spec = inspect_generation("openai", "openai", model, sampled, max_output_tokens=500)
    assert seen.pop("name") == spec.model_name
    assert seen.pop("config") == spec.config
    assert seen == {
        "base_url": "https://api.openai.com/v1",
        "api_key": "k",
        "memoize": False,
        **spec.model_args,
    }


def test_worker_environment_drops_credentials():
    env = worker_environment({"PATH": "/bin", "OPENAI_API_KEY": "k", "HF_TOKEN": "t", "HOME": "/h"})
    assert env == {"PATH": "/bin", "HOME": "/h"}


def test_frozen_sandboxes_run_under_local_docker_or_fail_clearly():
    compose = ComposeConfig(services={"default": {"image": "img@sha256:" + "a" * 64}})
    task = Task(
        dataset=[Sample(input="q", target="a")],
        sandbox=SandboxEnvironmentSpec(type="modal", config=compose.model_dump(mode="json")),
    )
    with pytest.raises(LocalRunError, match="Docker"):
        use_local_sandboxes([(task, None)], docker_available=False)
    use_local_sandboxes([(task, None)], docker_available=True)
    assert task.sandbox.type == "docker"
    assert task.sandbox.config.services["default"].image == "img@sha256:" + "a" * 64
    unsupported = Task(dataset=[Sample(input="q")], sandbox=("docker", "compose.yaml"))
    with pytest.raises(LocalRunError, match="not yet supported locally"):
        use_local_sandboxes([(unsupported, None)], docker_available=True)

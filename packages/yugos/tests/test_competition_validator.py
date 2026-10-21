"""Freshness, competition version and current hotkey identity precede any weight write."""

from datetime import timedelta

import pytest
from test_competition import policy, submission
from test_validator import NOW, ValidatorChain

from yugos.competition import (
    CompetitionHistory,
    CompetitionPublication,
    CompetitionResult,
    digest,
)
from yugos.validator import validator_step


def publication(history=None, *, now=NOW, paused=False):
    history = history or CompetitionHistory(policy=policy()).challenge(submission("miner-a"))
    payload = CompetitionResult(netuid=7, history=history, paused=paused)
    return CompetitionPublication(id=digest(payload), confirmed_at=now, payload=payload).model_dump(
        mode="json"
    )


def step(tmp_path, raw, state=None, *, now=NOW, chain=None):
    class Service:
        def request(self, method, path, *, signed=False):
            assert method == "GET" and path == "/scoring/snapshots/latest?netuid=7" and signed
            return raw

    return validator_step(
        Service(),
        chain or ValidatorChain(),
        state={} if state is None else state,
        state_path=tmp_path / "state.json",
        blocks_between_weights=10,
        max_age_s=86400,
        version_key=2,
        now=now,
    )


def test_three_kings_publish_85_10_and_5_on_their_current_uids(tmp_path):
    from types import SimpleNamespace

    history = CompetitionHistory(policy=policy())
    for miner, cost in (("miner-a", "10"), ("miner-b", "8"), ("miner-c", "6")):
        history = history.challenge(submission(miner, cost=cost))
    chain = ValidatorChain()
    chain.graph.neurons.append(SimpleNamespace(hotkey="miner-c", uid=11))
    assert "published" in step(tmp_path, publication(history), chain=chain)
    assert chain.published == [
        ({11: pytest.approx(0.85), 8: pytest.approx(0.10), 3: pytest.approx(0.05)}, 2)
    ]


def test_unregistered_current_king_does_not_pay_older_kings(tmp_path):
    from types import SimpleNamespace

    history = CompetitionHistory(policy=policy())
    for miner, cost in (("miner-a", "10"), ("miner-b", "8"), ("miner-c", "6")):
        history = history.challenge(submission(miner, cost=cost))
    chain = ValidatorChain()
    chain.graph.neurons.append(SimpleNamespace(hotkey="miner-c", uid=11))
    chain.graph.neurons = [neuron for neuron in chain.graph.neurons if neuron.hotkey != "miner-c"]
    assert "no eligible registered miners" in step(tmp_path, publication(history), chain=chain)
    assert chain.published == []


def test_replay_publishes_only_overall_champion_and_remaps_current_uid(tmp_path):
    raw, chain = publication(), ValidatorChain()
    assert "published" in step(tmp_path, raw, chain=chain)
    assert chain.published == [({3: 1.0}, 2)]
    chain.graph.neurons[1].uid = 19
    assert "published" in step(tmp_path, raw, chain=chain)
    assert chain.published[-1] == ({19: 1.0}, 2)
    chain.graph.neurons.pop(1)
    assert "no eligible registered miners" in step(tmp_path, raw, chain=chain)
    assert len(chain.published) == 2


def test_fresh_confirmation_allows_unchanged_history_after_24_hours(tmp_path):
    old, state = publication(), {}
    step(tmp_path, old, state)
    later = NOW + timedelta(hours=25)
    refreshed = publication(now=later)
    assert refreshed["id"] == old["id"]
    step(tmp_path, refreshed, state, now=later)
    with pytest.raises(ValueError, match="stale"):
        step(tmp_path, old, now=later)


@pytest.mark.parametrize("defect", ["digest", "version", "netuid", "future"])
def test_invalid_publication_never_writes_weights_or_state(tmp_path, defect):
    raw, chain, state = publication(), ValidatorChain(), {}
    if defect == "digest":
        raw["id"] = "0" * 64
    elif defect == "version":
        raw["payload"]["history"]["policy"]["version"] = "yugos-overall-v1"
    elif defect == "netuid":
        raw["payload"]["netuid"] = 8
    else:
        raw["confirmed_at"] = (NOW + timedelta(seconds=1)).isoformat()
    if defect != "digest":
        raw["id"] = digest(raw["payload"])
    with pytest.raises(ValueError):
        step(tmp_path, raw, state, chain=chain)
    assert not chain.published and not state


def test_pause_and_empty_history_do_not_publish_an_alternative_winner(tmp_path):
    chain = ValidatorChain()
    step(tmp_path, publication(paused=True), chain=chain)
    step(tmp_path, publication(CompetitionHistory(policy=policy())), chain=chain)
    assert not chain.published


def test_incompatible_chain_limits_fail_before_state_or_transaction(tmp_path):
    from types import SimpleNamespace

    from yugos.chain import Chain

    chain = ValidatorChain()
    chain.check_winner_allocation = Chain.check_winner_allocation.__get__(chain)
    state = {}
    for params in (
        {"min_allowed_weights": 2, "max_weights_limit": 65535},
        {"min_allowed_weights": 1, "max_weights_limit": 32767},
        {},
    ):
        chain.subtensor = SimpleNamespace(read=lambda *args, **kwargs: params)
        with pytest.raises(ValueError, match="weight"):
            step(tmp_path, publication(), state, chain=chain)
        assert not chain.published and not state
    chain.subtensor = SimpleNamespace(
        read=lambda *args, **kwargs: {"min_allowed_weights": 1, "max_weights_limit": 65535}
    )
    step(tmp_path, publication(), state, chain=chain)
    assert chain.published == [({3: 1.0}, 2)]


def test_a_new_policy_of_the_same_version_carries_on_with_saved_state(tmp_path):
    state, later = {}, NOW + timedelta(seconds=1)
    step(tmp_path, publication(), state)
    raw = publication(now=later)
    raw["payload"]["history"]["policy"]["bootstrap_seed"] = 1
    raw["id"] = digest(raw["payload"])
    step(tmp_path, raw, state, now=later)
    assert state["snapshot"] == raw

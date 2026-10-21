"""Hosted results must be intact, fresh and for this subnet before any chain publication."""

import copy
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
from bittensor.wallet import Keypair
from btauth import verify_signed
from test_competition import policy, submission

from yugos import validator
from yugos.chain import Chain
from yugos.competition import (
    CompetitionHistory,
    CompetitionPublication,
    CompetitionResult,
    digest,
)
from yugos.service_client import ServiceClient
from yugos.validator import validator_step

NOW = datetime(2026, 9, 13, tzinfo=timezone.utc)
NETUID = 7


def snapshot(*, empty=False):
    history = CompetitionHistory(policy=policy())
    if not empty:
        history = history.challenge(submission("miner-a"))
    payload = CompetitionResult(netuid=7, history=history)
    return CompetitionPublication(id=digest(payload), confirmed_at=NOW, payload=payload).model_dump(
        mode="json"
    )


class Results:
    def __init__(self, raw):
        self.raw = raw

    def request(self, method, path, *, signed=False):
        assert method == "GET" and path == f"/scoring/snapshots/latest?netuid={NETUID}"
        assert signed
        return self.raw


class ValidatorChain(Chain):
    def __init__(self):
        super().__init__(
            SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator")), None, NETUID
        )
        self.graph = SimpleNamespace(
            block=100,
            owner_hotkey="owner",
            neurons=[
                SimpleNamespace(hotkey="validator", uid=0, last_update=0, validator_permit=True),
                SimpleNamespace(hotkey="miner-a", uid=3),
                SimpleNamespace(hotkey="miner-b", uid=8),
            ],
        )
        self.graph.by_hotkey = lambda key: next(
            (n for n in self.graph.neurons if n.hotkey == key), None
        )
        self.published = []
        self.fail = False

    def metagraph(self):
        return self.graph

    def check_winner_allocation(self):
        pass

    def publish(self, weights, *, version_key):
        if self.fail:
            raise RuntimeError("chain rejected update")
        self.published.append((weights, version_key))
        return SimpleNamespace(block_hash="included")


def step(tmp_path, service, chain, state, **kwargs):
    return validator_step(
        service,
        chain,
        state=state,
        state_path=tmp_path / "state.json",
        blocks_between_weights=10,
        max_age_s=100,
        version_key=4,
        now=NOW,
        **kwargs,
    )


def test_weights_follow_current_hotkeys_across_restart_and_uid_reassignment(tmp_path):
    chain, state, results = ValidatorChain(), {}, Results(snapshot())
    assert "published" in step(tmp_path, results, chain, state)
    expected = CompetitionPublication.model_validate(results.raw).payload.weights()
    assert chain.published == [({3: expected["miner-a"]}, 4)]
    restored = json.loads((tmp_path / "state.json").read_bytes())
    assert "waiting" in step(tmp_path, results, chain, restored)
    chain.graph.block += 10
    chain.graph.neurons[1].uid = 17
    chain.graph.neurons.pop()
    step(tmp_path, results, chain, restored)
    assert chain.published[-1] == ({17: expected["miner-a"]}, 4)


@pytest.mark.parametrize("defect", ["hash", "cohort", "old", "future", "legacy"])
def test_bad_snapshot_never_changes_chain_or_durable_state(tmp_path, defect):
    raw = snapshot()
    if defect == "hash":
        raw["id"] = "0" * 64
    elif defect == "cohort":
        raw["payload"]["netuid"] += 1
        raw["id"] = digest(raw["payload"])
    elif defect == "legacy":
        raw = {"id": "a" * 64, "payload": {"schema_version": 1, "observations": [], "weights": {}}}
    else:
        raw["confirmed_at"] = (
            NOW + timedelta(seconds=1 if defect == "future" else -101)
        ).isoformat()
    chain, state = ValidatorChain(), {"previous": "preserved"}
    with pytest.raises(ValueError):
        step(tmp_path, Results(raw), chain, state)
    assert not chain.published and state == {"previous": "preserved"}
    assert not (tmp_path / "state.json").exists()


def test_preview_and_zero_results_preserve_chain_weights(tmp_path):
    chain, state = ValidatorChain(), {}
    assert "preview" in step(tmp_path, Results(snapshot()), chain, state, dry_run=True)
    assert "unchanged" in step(tmp_path, Results(snapshot(empty=True)), chain, {})
    assert not chain.published and "last_success_block" not in state


def test_failed_chain_update_is_retryable_and_never_recorded_as_success(tmp_path):
    chain, state, service = ValidatorChain(), {}, Results(snapshot())
    chain.fail = True
    with pytest.raises(RuntimeError, match="rejected"):
        step(tmp_path, service, chain, state)
    restored = json.loads((tmp_path / "state.json").read_bytes())
    assert "last_success_block" not in restored
    chain.fail = False
    assert "published" in step(tmp_path, service, chain, restored)


def test_snapshot_rollback_is_rejected_even_while_both_results_are_fresh(tmp_path):
    chain, state, service = ValidatorChain(), {}, Results(snapshot())
    step(tmp_path, service, chain, state, dry_run=True)
    service.raw = copy.deepcopy(service.raw)
    service.raw["confirmed_at"] = (NOW - timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="rolled back"):
        step(tmp_path, service, chain, state)
    assert not chain.published


def test_missing_permit_or_unwritable_state_prevents_publication(tmp_path, monkeypatch):
    chain = ValidatorChain()
    chain.graph.neurons[0].validator_permit = False
    with pytest.raises(ValueError, match="permit"):
        step(tmp_path, Results(snapshot()), chain, {})
    chain.graph.neurons[0].validator_permit = True

    def fail(*args):
        raise OSError("storage unavailable")

    monkeypatch.setattr("yugos.validator.save_json", fail)
    with pytest.raises(OSError):
        step(tmp_path, Results(snapshot()), chain, {})
    assert not chain.published


def test_snapshot_request_carries_a_valid_btauth_signature(tmp_path):
    key = Keypair.create_from_uri("//Alice")
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=snapshot())

    client = httpx.Client(transport=httpx.MockTransport(handle))
    service = ServiceClient("https://evaluation.example", client, key)
    assert "preview" in step(tmp_path, service, ValidatorChain(), {}, dry_run=True)
    assert len(calls) == 1
    request = calls[0]
    assert request.method == "GET"
    assert request.url.raw_path == f"/scoring/snapshots/latest?netuid={NETUID}".encode()
    assert request.content == b""
    verify_signed(request, key)


def test_neuron_retries_service_outage_and_persists_a_later_success(tmp_path, monkeypatch):
    key = Keypair.create_from_uri("//Alice")
    chain = ValidatorChain()
    chain.wallet.hotkey = key
    chain.graph.neurons[0].hotkey = key.ss58_address
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(503) if len(requests) == 1 else httpx.Response(200, json=snapshot())

    client = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(validator, "configured_chain", lambda args: chain)
    monkeypatch.setattr(validator.httpx, "Client", lambda **kwargs: client)
    monkeypatch.setattr(validator.signal, "signal", lambda *args: None)

    class Stop:
        waits = 0

        def is_set(self):
            return self.waits == 2

        def wait(self, seconds):
            self.waits += 1

        def set(self):
            self.waits = 2

    monkeypatch.setattr(validator, "threading", SimpleNamespace(Event=Stop))

    class Clock:
        @staticmethod
        def now(tz):
            return NOW

        fromisoformat = datetime.fromisoformat

    monkeypatch.setattr(validator, "datetime", Clock)
    validator.main(
        [
            "--network",
            "test",
            "--netuid",
            str(NETUID),
            "--wallet",
            "fixture",
            "--endpoint",
            "https://evaluation.example",
            "--state",
            str(tmp_path / "state.json"),
        ]
    )
    assert len(requests) == 2 and len(chain.published) == 1
    for request in requests:
        verify_signed(request, key)
    assert (
        json.loads((tmp_path / "state.json").read_bytes())["published_snapshot"] == snapshot()["id"]
    )

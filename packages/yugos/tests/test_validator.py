"""Hosted results must be intact, fresh and for this subnet before any chain publication."""

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
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
from yugos.service_client import ServiceClient, ServiceError
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
        self.calls = 0

    def request(self, method, path, *, signed=False):
        assert method == "GET" and path == f"/scoring/snapshots/latest?netuid={NETUID}"
        assert signed
        self.calls += 1
        return self.raw


class ValidatorChain(Chain):
    def __init__(self):
        self.params = {
            "min_allowed_weights": 1,
            "max_weights_limit": 65535,
            "weights_rate_limit": 0,
        }
        super().__init__(
            SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator")),
            SimpleNamespace(read=lambda name, **kwargs: self.params),
            NETUID,
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
        self.waited = []

    def metagraph(self):
        return self.graph

    def publish(self, weights, *, version_key):
        if self.fail:
            raise RuntimeError("chain rejected update")
        self.published.append((weights, version_key))
        return SimpleNamespace(block_hash="included")

    def wait_for_block(self, block, stop):
        self.waited.append(block)
        stop.set()


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


@pytest.mark.parametrize(
    "chain_interval,chain_last,saved_last,due",
    [(20, 90, 0, 110), (0, 95, 0, 105), (20, 50, 95, 115)],
)
def test_weights_wait_for_both_chain_and_operator_intervals(
    tmp_path, chain_interval, chain_last, saved_last, due
):
    chain, results = ValidatorChain(), Results(snapshot())
    chain.params["weights_rate_limit"] = chain_interval
    chain.graph.neurons[0].last_update = chain_last
    state = {"last_success_block": saved_last}
    state_path = tmp_path / "state.json"
    validator.save_json(state_path, state)
    saved = state_path.read_bytes()
    chain.graph.block = due - 1
    status = step(tmp_path, results, chain, state)
    assert f"weight block {due}" in status and "1 remaining" in status
    assert not chain.published and results.calls == 0
    assert state_path.read_bytes() == saved and state == {"last_success_block": saved_last}
    restored = json.loads(saved)
    chain.graph.block = due
    assert "published" in step(tmp_path, results, chain, restored)
    assert results.calls == 1 and len(chain.published) == 1
    assert restored["last_success_block"] == due


def test_first_weight_update_does_not_wait_for_a_full_interval(tmp_path):
    chain = ValidatorChain()
    chain.params["weights_rate_limit"] = chain.graph.block + 1
    assert "published" in step(tmp_path, Results(snapshot()), chain, {})


def test_miner_uids_are_refreshed_after_fetching_the_snapshot(tmp_path, monkeypatch):
    chain = ValidatorChain()
    monkeypatch.setattr(chain, "metagraph", lambda: copy.deepcopy(chain.graph))

    class ReassignedResults(Results):
        def request(self, *args, **kwargs):
            chain.graph.neurons[1].uid = 17
            return super().request(*args, **kwargs)

    assert "published" in step(tmp_path, ReassignedResults(snapshot()), chain, {})
    assert chain.published == [({17: 1.0}, 4)]


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
    chain.graph.neurons[0].last_update = chain.graph.block
    results = Results(snapshot())
    assert "preview" in step(tmp_path, results, chain, state, dry_run=True)
    assert results.calls == 1
    chain.graph.neurons[0].last_update = 0
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


def test_snapshot_retries_refresh_before_publishing_and_use_completion_time(tmp_path, monkeypatch):
    chain, state, waits, requests = ValidatorChain(), {}, [], []
    key = Keypair.create_from_uri("//Alice")
    refreshed = snapshot()
    refreshed["confirmed_at"] = (NOW + timedelta(minutes=1)).isoformat()

    def handle(request):
        requests.append(request)
        assert not chain.published
        if len(requests) == 1:
            return httpx.Response(409)
        if len(requests) == 2:
            raise httpx.ReadTimeout("temporarily unavailable", request=request)
        return httpx.Response(200, json=refreshed)

    class Clock:
        @staticmethod
        def now(tz):
            assert len(requests) == 3
            return NOW + timedelta(minutes=1)

        fromisoformat = datetime.fromisoformat

    monkeypatch.setattr(validator, "datetime", Clock)
    stop = SimpleNamespace(is_set=lambda: False, wait=waits.append)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        service = ServiceClient("https://evaluation.example", client, key)
        assert "published" in validator_step(
            service,
            chain,
            state=state,
            state_path=tmp_path / "state.json",
            blocks_between_weights=10,
            max_age_s=100,
            version_key=4,
            stop=stop,
        )
    assert len(chain.published) == 1 and state["snapshot"] == refreshed
    assert len(waits) == 2 and 0 < waits[0] < waits[1]
    for request in requests:
        verify_signed(request, key)


@pytest.mark.parametrize("failure", ["outage", "authentication", "cancelled"])
def test_failed_refresh_never_publishes_a_cached_snapshot(tmp_path, monkeypatch, failure):
    chain, requests, waits = ValidatorChain(), [], []
    state = {"snapshot": snapshot()}
    validator.save_json(tmp_path / "state.json", state)
    saved = (tmp_path / "state.json").read_bytes()
    monkeypatch.setattr(validator, "SNAPSHOT_ATTEMPTS", 3)

    def request(*args, **kwargs):
        requests.append(args)
        raise ServiceError(
            "/scoring/snapshots/latest", 403 if failure == "authentication" else 503, b"{}"
        )

    def wait(seconds):
        waits.append(seconds)
        return failure == "cancelled"

    stop = SimpleNamespace(is_set=lambda: False, wait=wait)
    service = SimpleNamespace(request=request)
    if failure == "cancelled":
        assert "stopping" in step(tmp_path, service, chain, state, stop=stop)
    else:
        with pytest.raises(ServiceError):
            step(tmp_path, service, chain, state, stop=stop)
    assert len(requests) == (3 if failure == "outage" else 1)
    assert len(waits) == {"outage": 2, "authentication": 0, "cancelled": 1}[failure]
    assert not chain.published and state == {"snapshot": snapshot()}
    assert (tmp_path / "state.json").read_bytes() == saved


def test_default_state_path_is_stable_and_isolates_every_scope_field(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    scope = {
        "network": "test",
        "netuid": NETUID,
        "endpoint": "https://evaluation.example",
        "hotkey": "validator",
    }
    path = validator.default_state_path(scope)
    assert path == validator.default_state_path(dict(reversed(scope.items())))
    alternatives = {
        "network": "another-network",
        "netuid": NETUID + 1,
        "endpoint": "https://another.example",
        "hotkey": "another-validator",
    }
    paths = [path] + [
        validator.default_state_path(scope | {field: value})
        for field, value in alternatives.items()
    ]
    assert len(set(paths)) == len(paths)


@pytest.mark.parametrize("xdg", [None, "", "relative-state", "absolute"])
def test_default_state_uses_an_absolute_persistent_directory(tmp_path, monkeypatch, xdg):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    if xdg is None:
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    else:
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state") if xdg == "absolute" else xdg)
    root = tmp_path / "state" if xdg == "absolute" else home / ".local" / "state"
    assert (
        validator.default_state_path({"hotkey": "validator"}).parent
        == root / "yugos" / "validators"
    )


@pytest.mark.parametrize("explicit_state", [False, True])
@pytest.mark.parametrize("saved_alias", [False, True])
def test_neuron_retries_service_outage_and_persists_a_later_success(
    tmp_path, monkeypatch, saved_alias, explicit_state
):
    key = Keypair.create_from_uri("//Alice")
    chain = ValidatorChain()
    chain.wallet.hotkey = key
    chain.graph.neurons[0].hotkey = key.ss58_address
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(503) if len(requests) == 1 else httpx.Response(200, json=snapshot())

    client_class = httpx.Client
    monkeypatch.setattr(validator, "configured_chain", lambda args: chain)
    monkeypatch.setattr(
        validator.httpx,
        "Client",
        lambda **kwargs: client_class(transport=httpx.MockTransport(handle)),
    )
    monkeypatch.setattr(validator.signal, "signal", lambda *args: None)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "application-state"))

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
    scope = {
        "network": "test",
        "netuid": NETUID,
        "endpoint": "https://evaluation.example",
        "hotkey": key.ss58_address,
    }
    state_path = tmp_path / "state.json" if explicit_state else validator.default_state_path(scope)
    if saved_alias:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "scope": {
                        "network": "test",
                        "netuid": NETUID,
                        "endpoint": "https://evaluation.example/upload",
                        "hotkey": key.ss58_address,
                    }
                }
            )
        )
    argv = [
        "--network",
        "test",
        "--netuid",
        str(NETUID),
        "--wallet",
        "fixture",
        "--endpoint",
        "https://evaluation.example/upload",
    ]
    if explicit_state:
        argv.extend(["--state", str(state_path)])
    validator.main(argv)
    assert len(requests) == 2 and len(chain.published) == 1
    assert chain.waited and chain.waited[0] > chain.graph.block
    for request in requests:
        assert request.url.path == "/scoring/snapshots/latest"
        verify_signed(request, key)
    assert json.loads(state_path.read_text())["scope"]["endpoint"] == "https://evaluation.example"
    assert json.loads(state_path.read_bytes())["published_snapshot"] == snapshot()["id"]
    validator.main(argv + ["--once"])
    assert len(requests) == 2 and len(chain.published) == 1
    if explicit_state:
        assert not (tmp_path / "application-state").exists()
    else:
        assert list((tmp_path / "application-state").rglob("*.json")) == [state_path]

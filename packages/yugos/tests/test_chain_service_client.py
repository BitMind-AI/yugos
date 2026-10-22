"""Check the real SDK request shapes without opening a chain connection."""

import json
import threading
from types import SimpleNamespace

import httpx
import pytest
from bittensor import SetWeights
from bittensor.result import ExtrinsicResult
from bittensor.wallet import Keypair
from btauth import assert_unsigned, verify_signed

from yugos.chain import (
    Chain,
    add_chain_arguments,
    resolve_subnet,
    submission_commitment,
    subnet_for_network,
)
from yugos.miner.cli import resolve_miner_endpoint
from yugos.service_client import (
    DEFAULT_MINER_ENDPOINT,
    ServiceClient,
    ServiceError,
    service_url,
)


@pytest.mark.parametrize("cancel", [False, True])
def test_block_wait_resumes_after_timeout_and_honors_shutdown(cancel):
    calls, stop = [], threading.Event()

    def wait_for_block(block, *, timeout):
        calls.append(block)
        assert timeout > 0
        if len(calls) == 1:
            if cancel:
                stop.set()
            raise TimeoutError

    chain = Chain(None, SimpleNamespace(wait_for_block=wait_for_block), 7)
    chain.wait_for_block(123, stop)
    assert calls == ([123] if cancel else [123, 123])
    stop.set()
    chain.wait_for_block(456, stop)
    assert 456 not in calls


def test_commitment_and_weights_use_hotkey_operations_and_propagate_failure():
    captured = []
    wallet = object()

    def submit_call(call, actual_wallet, **kwargs):
        captured.append(call)
        assert actual_wallet is wallet and kwargs["signer"] == "hotkey"
        assert kwargs["wait_for_finalization"] and kwargs["wait_for_inclusion"]
        return ExtrinsicResult(success=True, block_hash="finalized")

    def execute(intent, actual_wallet, **kwargs):
        assert isinstance(intent, SetWeights) and actual_wallet is wallet
        assert dict(zip(intent.uids, intent.weights)) == {5: 0.8, 8: 0.2}
        assert intent.netuid == 7 and intent.version_key == 4
        assert kwargs["wait_for_finalization"] and not kwargs["wait_for_registration"]
        return ExtrinsicResult(success=False, message="rejected")

    chain = Chain(wallet, SimpleNamespace(submit_call=submit_call, execute=execute), 7)
    digest = "a" * 64
    assert chain.commit(digest).block_hash == "finalized"
    call = captured[0]
    assert (call.module, call.function, call.params["netuid"]) == (
        "Commitments",
        "set_commitment",
        7,
    )
    (field,) = call.params["info"]["fields"]
    kind, encoded = next(iter(field.items()))
    payload = bytes.fromhex(encoded[2:])
    assert payload.decode() == submission_commitment(digest) and kind == f"Raw{len(payload)}"
    with pytest.raises(Exception):
        chain.publish({5: 0.8, 8: 0.2}, version_key=4)
    with pytest.raises(ValueError):
        chain.commit("g" * 64)
    assert len(captured) == 1


def test_named_networks_select_their_subnets():
    assert subnet_for_network("test") == 169
    assert subnet_for_network("finney") == 34
    args = SimpleNamespace(network="test", netuid=None)
    resolve_subnet(args)
    assert args.netuid == 169
    explicit = SimpleNamespace(network="test", netuid=7)
    resolve_subnet(explicit)
    assert explicit.netuid == 7
    with pytest.raises(ValueError):
        subnet_for_network("wss://custom.example")


def test_network_defaults_to_mainnet_and_test_needs_an_endpoint(monkeypatch):
    import argparse

    monkeypatch.delenv("YUGOS_ENDPOINT", raising=False)
    parser = argparse.ArgumentParser()
    add_chain_arguments(parser)
    args = parser.parse_args(["--wallet", "miner"])
    assert args.network == "finney"
    resolve_subnet(args)
    assert args.netuid == 34
    with pytest.raises(ValueError, match="test network"):
        resolve_miner_endpoint(None, "test")
    assert resolve_miner_endpoint(None, "finney") == DEFAULT_MINER_ENDPOINT
    assert resolve_miner_endpoint("https://other.example", "test") == ("https://other.example")


def test_submit_endpoint_uses_public_default_and_explicit_override_precedence(monkeypatch):
    monkeypatch.delenv("YUGOS_ENDPOINT", raising=False)
    assert resolve_miner_endpoint(None) == DEFAULT_MINER_ENDPOINT
    monkeypatch.setenv("YUGOS_ENDPOINT", "https://configured.example")
    assert resolve_miner_endpoint(None) == "https://configured.example"
    assert resolve_miner_endpoint("https://other.example") == ("https://other.example")


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.example",
        "https://user:secret@example.org",
        "https://example.org?token=1",
        "file:///tmp/results",
    ],
)
def test_untrusted_transport_configuration_is_rejected(url):
    with pytest.raises(ValueError):
        service_url(url)


@pytest.mark.parametrize(
    "base,prefix",
    [("https://evaluation.example/upload/", ""), ("https://evaluation.example/tenant", "/tenant")],
)
def test_service_base_is_canonical_before_signing_encoded_requests(base, prefix):
    # A stale prefix in either the request or its signature breaks the canonical routes.
    key = Keypair.create_from_uri("//Alice")
    path = "/credits/%68istory?limit=1&probe=a%2Fb&probe=c+d"
    calls = []

    def handle(request):
        calls.append(request)
        assert request.url.raw_path == (prefix + path).encode()
        verify_signed(request, key)
        return httpx.Response(200, json={"ok": True})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = ServiceClient(base, client, key).request("GET", path, signed=True)
    assert result == {"ok": True} and len(calls) == 1


def test_redirects_and_oversized_results_are_rejected(monkeypatch):
    monkeypatch.setattr("yugos.service_client.MAX_RESPONSE_BYTES", 32)
    for response in (
        httpx.Response(302, headers={"Location": "https://other.example"}),
        httpx.Response(200, content=b"x" * 33),
    ):
        calls = []

        def handle(request):
            calls.append(request)
            return response

        with httpx.Client(transport=httpx.MockTransport(handle), follow_redirects=True) as client:
            service = ServiceClient("https://evaluation.example", client)
            with pytest.raises((RuntimeError, ValueError)):
                service.request("GET", "/scoring/snapshots/latest")
        assert len(calls) == 1


def test_delayed_completion_polls_the_same_result_without_replaying_signed_post():
    calls = []
    key = Keypair.create_from_uri("//Alice")

    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            assert request.method == "POST"
            assert json.loads(request.content) == {"model_id": 1, "upload_id": "session"}
            verify_signed(request, key)
        else:
            assert request.method == "GET" and request.content == b""
            assert_unsigned(request.headers)
            assert request.url.params["result"] == "same-operation"
        if len(calls) < 3:
            return httpx.Response(303, headers={"Location": "?result=same-operation"})
        return httpx.Response(200, json={"uploaded": True, "model_id": 1})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        service = ServiceClient("https://evaluation.example", client, key)
        result = service.request(
            "POST", "/multipart/complete", {"model_id": 1, "upload_id": "session"}, signed=True
        )
    assert result == {"uploaded": True, "model_id": 1}
    assert len(calls) == 3


def test_expired_signature_is_refreshed_and_result_polling_remains_unsigned():
    calls = []
    key = Keypair.create_from_uri("//Alice")
    payload = {"model_id": 1, "upload_id": "session"}

    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            verify_signed(request, key)
            return httpx.Response(
                401, json={"detail": "nonce is older than the 10s freshness window"}
            )
        if len(calls) == 2:
            assert request.content == calls[0].content
            assert request.headers["X-Bittensor-Nonce"] != calls[0].headers["X-Bittensor-Nonce"]
            verify_signed(request, key)
            return httpx.Response(303, headers={"Location": "?result=same-operation"})
        assert request.method == "GET" and request.content == b""
        assert_unsigned(request.headers)
        return httpx.Response(200, json={"uploaded": True})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = ServiceClient("https://evaluation.example", client, key).request(
            "POST", "/multipart/complete", payload, signed=True
        )
    assert result == {"uploaded": True} and len(calls) == 3


@pytest.mark.parametrize(
    "detail,signed,expected_requests",
    [
        ("nonce is older than the 10s freshness window", True, 3),
        ("nonce is older than the 10s freshness window", False, 1),
        ("signature does not verify against the claimed hotkey", True, 1),
    ],
)
def test_authentication_retries_are_bounded_and_only_refresh_expired_signatures(
    detail, signed, expected_requests
):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(401, json={"detail": detail})

    key = Keypair.create_from_uri("//Alice")
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ServiceError) as error:
            ServiceClient("https://evaluation.example", client, key).request(
                "POST", "/multipart/part", {"part_number": 2}, signed=signed
            )
    assert error.value.status == 401
    assert len(calls) == expected_requests


@pytest.mark.parametrize(
    "location",
    [
        "https://other.example/multipart/complete?result=token",
        "http://evaluation.example/multipart/complete?result=token",
        "/different-operation?result=token",
    ],
)
def test_delayed_results_cannot_redirect_to_another_origin_or_operation(location):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(303, headers={"Location": location})

    with httpx.Client(transport=httpx.MockTransport(handle), follow_redirects=True) as client:
        service = ServiceClient("https://evaluation.example", client)
        with pytest.raises(ValueError, match="unsafe result redirect"):
            service.request("POST", "/multipart/complete", {})
    assert len(calls) == 1

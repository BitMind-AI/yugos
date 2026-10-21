"""Payment commands must sign exact requests and preserve refunds across failures."""

import argparse
import json
from types import SimpleNamespace

import httpx
import pytest
from bittensor.wallet import Keypair
from btauth import assert_unsigned, verify_signed

from yugos.miner import credits as miner_credits
from yugos.miner import main
from yugos.miner.credits import checkout, refund, usd_cents
from yugos.service_client import ServiceClient


def verify(request, key):
    body = verify_signed(request, key)
    return json.loads(body) if body else None


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "Infinity", "1.001", "1e2", "$10"])
def test_amount_rejects_ambiguous_or_fractional_cents(value):
    with pytest.raises(argparse.ArgumentTypeError):
        usd_cents(value)


def test_amount_preserves_cents():
    assert usd_cents("0.29") == 29
    assert usd_cents("1234.5") == 123450


@pytest.mark.parametrize("no_browser", [False, True])
def test_checkout_cli_unsigned_amount_and_handles_headless_browser(monkeypatch, capsys, no_browser):
    key = Keypair.create_from_uri("//Alice")
    url = "https://checkout.stripe.com/c/pay/test-session"
    requests = []
    opened = []

    def handler(request):
        requests.append(request)
        assert request.method == "POST" and request.url.path == "/credits/checkout"
        assert_unsigned(request.headers)
        assert json.loads(request.read()) == {"amount_cents": 1029, "hotkey": key.ss58_address}
        return httpx.Response(200, json={"id": "cs_test", "url": url})

    def browser(link):
        assert url in capsys.readouterr().out  # Link survives a browser-launch failure.
        opened.append(link)
        raise OSError("headless")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(httpx, "Client", lambda **kw: client)
    monkeypatch.setattr(
        miner_credits,
        "configured_wallet",
        lambda args: pytest.fail("public checkout opened wallet"),
    )
    monkeypatch.setattr(miner_credits.webbrowser, "open", browser)
    main(
        [
            "credits",
            "add",
            "--amount-usd",
            "10.29",
            "--endpoint",
            "https://service.example",
            "--to-hotkey",
            key.ss58_address,
            *(["--no-browser"] if no_browser else []),
        ]
    )
    assert len(requests) == 1
    assert opened == ([] if no_browser else [url])
    if no_browser:
        assert url in capsys.readouterr().out


@pytest.mark.parametrize(
    "operation,path", [("balance", "/credits"), ("history", "/credits/history")]
)
def test_read_commands_sign_empty_body_without_chain(monkeypatch, capsys, operation, path):
    key = Keypair.create_from_uri("//Alice")

    def handler(request):
        assert request.method == "GET" and request.url.path == path
        assert verify(request, key) is None
        return httpx.Response(200, json={"balance_cents": 42})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(httpx, "Client", lambda **kw: client)
    monkeypatch.setattr(
        miner_credits, "configured_wallet", lambda args: SimpleNamespace(hotkey=key)
    )
    main(
        [
            "credits",
            operation,
            "--endpoint",
            "https://service.example",
            "--wallet",
            "miner",
            "--hotkey",
            "router-a",
        ]
    )
    assert json.loads(capsys.readouterr().out)["balance_cents"] == 42


@pytest.mark.parametrize("saved_alias", [False, True])
def test_refund_lost_response_reuses_persisted_identity(tmp_path, saved_alias):
    key = Keypair.create_from_uri("//Alice")
    path = tmp_path / "refund.json"
    attempts = []

    def handler(request):
        payload = verify(request, key)
        assert request.url.path == "/credits/refund"
        assert json.loads(path.read_text())["request_id"] == payload["request_id"]
        attempts.append(payload)
        if len(attempts) == 1:
            raise httpx.ReadTimeout("Stripe accepted refund but response was lost")
        return httpx.Response(
            200,
            json={
                "ok": True,
                "request_id": payload["request_id"],
                "refund": {"id": "re_test", "status": "succeeded"},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = ServiceClient("https://service.example", client, key)
        with pytest.raises(httpx.ReadTimeout):
            refund(service, 125, path)
        if saved_alias:
            receipt = json.loads(path.read_text())
            receipt["endpoint"] += "/upload"
            path.write_text(json.dumps(receipt))
        result = refund(service, 125, path)
        assert result["endpoint"] == service.endpoint
        assert refund(service, 125, path) == result
        for other, amount in [
            (service, 126),
            (ServiceClient("https://other.example", client, key), 125),
            (ServiceClient(service.endpoint, client, Keypair.create_from_uri("//Bob")), 125),
        ]:
            with pytest.raises(ValueError, match="another amount, wallet, or service"):
                refund(other, amount, path)
    assert len(attempts) == 2 and attempts[0] == attempts[1]


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/card",
        "http://checkout.stripe.com/pay",
        "https://checkout.stripe.com.evil.example/pay",
    ],
)
def test_checkout_never_opens_untrusted_url(monkeypatch, url):
    key = Keypair.create_from_uri("//Alice")
    monkeypatch.setattr(
        miner_credits.webbrowser, "open", lambda url: pytest.fail("opened untrusted URL")
    )
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"url": url}))
    ) as client:
        with pytest.raises(ValueError, match="Stripe-hosted"):
            checkout(ServiceClient("https://service.example", client), 100, hotkey=key.ss58_address)


def test_checkout_wallet_uses_only_public_hotkey(monkeypatch):
    class PublicWallet:
        hotkeypub = SimpleNamespace(ss58_address="public-destination")

        @property
        def hotkey(self):
            pytest.fail("checkout unlocked the signing key")

    seen = []
    monkeypatch.setattr(miner_credits, "configured_wallet", lambda args: PublicWallet())
    monkeypatch.setattr(miner_credits, "checkout", lambda service, amount, **kw: seen.append(kw))
    main(
        [
            "credits",
            "add",
            "--amount-usd",
            "10",
            "--endpoint",
            "https://service.example",
            "--wallet",
            "miner",
            "--hotkey",
            "router-a",
            "--no-browser",
        ]
    )
    assert seen == [{"hotkey": "public-destination", "open_browser": False}]


@pytest.mark.parametrize("override", [None, "https://other.example"])
def test_checkout_uses_configured_default_and_explicit_override(monkeypatch, override):
    seen = []
    monkeypatch.setenv("YUGOS_ENDPOINT", "https://configured.example")
    monkeypatch.setattr(
        miner_credits, "checkout", lambda service, amount, **kw: seen.append(service.endpoint)
    )
    args = ["credits", "add", "--amount-usd", "10", "--to-hotkey", "public-address"]
    if override:
        args += ["--endpoint", override]
    main(args)
    assert seen == [override or "https://configured.example"]


def test_miner_endpoint_defaults_to_public_service(monkeypatch):
    from yugos.miner.cli import add_miner_endpoint_argument
    from yugos.service_client import DEFAULT_MINER_ENDPOINT

    monkeypatch.delenv("YUGOS_ENDPOINT", raising=False)
    parser = argparse.ArgumentParser()
    add_miner_endpoint_argument(parser)
    assert parser.parse_args([]).endpoint == DEFAULT_MINER_ENDPOINT


@pytest.mark.parametrize(
    "operation,extra",
    [
        ("add", ["--amount-usd", "10"]),
        ("balance", []),
        ("history", []),
        ("refund", ["--amount-usd", "1", "--receipt", "unused-refund.json"]),
    ],
)
def test_credit_commands_reject_implicit_hotkey(monkeypatch, capsys, operation, extra):
    monkeypatch.setattr(
        miner_credits,
        "configured_wallet",
        lambda args: pytest.fail("opened a wallet without selecting a hotkey"),
    )
    with pytest.raises(SystemExit) as exc:
        main(["credits", operation, "--wallet", "miner", *extra])
    assert exc.value.code != 0
    assert "--hotkey" in capsys.readouterr().err

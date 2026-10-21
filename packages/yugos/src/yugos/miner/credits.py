"""Authenticated credit operations for miners; accounting stays with the service."""

from __future__ import annotations

import argparse
import json
import re
import webbrowser
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from yugos.chain import add_wallet_arguments, configured_wallet
from yugos.service_client import ServiceClient
from yugos.state import save_json, state_lock


def usd_cents(value: str) -> int:
    """Parse a positive decimal dollar amount without rounding fractional cents."""
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]{1,2})?", value):
        raise argparse.ArgumentTypeError("use a positive USD amount with at most two decimals")
    dollars, _, fraction = value.partition(".")
    cents = int(dollars) * 100 + int(fraction.ljust(2, "0"))
    if cents <= 0:
        raise argparse.ArgumentTypeError("amount must be positive")
    return cents


def add_credit_commands(commands, *, endpoint_options: argparse.ArgumentParser) -> None:
    """Register wallet-authenticated credit commands on the miner CLI."""
    credits = commands.add_parser("credits", help="fund and manage hosted evaluation credits")
    operations = credits.add_subparsers(dest="credit_command", required=True)
    for name, help_text in (
        ("claim", "verify coldkey billing ownership for a miner hotkey"),
        ("balance", "show balance and benchmark funding requirements"),
        ("history", "show recent deposits, usage charges, and refunds"),
        ("add", "add credits using Stripe's hosted payment page"),
        ("refund", "refund unused credit; reuse the receipt to retry safely"),
    ):
        command = operations.add_parser(name, help=help_text, parents=[endpoint_options])
        add_wallet_arguments(command, required=name != "add", explicit_hotkey=True)
        if name == "add":
            command.add_argument("--amount-usd", required=True, type=usd_cents)
        if name == "refund":
            amounts = command.add_mutually_exclusive_group(required=True)
            amounts.add_argument("--amount-usd", type=usd_cents)
            amounts.add_argument("--all", action="store_true", help="refund the available balance")
        if name == "add":
            command.add_argument(
                "--to-hotkey", help="public destination SS58 address; no wallet needed"
            )
            command.add_argument("--no-browser", action="store_true", help="only print the link")
        if name == "refund":
            command.add_argument(
                "--receipt",
                required=True,
                type=Path,
                help="durable refund file; reuse after interruptions or uncertain outcomes",
            )


def checkout(
    service: ServiceClient, amount_cents: int, *, hotkey: str, open_browser: bool = True
) -> dict:
    """Create a checkout and print its link before attempting to launch a browser."""
    if amount_cents < 100:
        raise ValueError("minimum checkout is $1.00")
    print(f"Funding miner: {hotkey}", flush=True)
    session = service.request(
        "POST", "/credits/checkout", {"amount_cents": amount_cents, "hotkey": hotkey}
    )
    url = session.get("url")
    parsed = urlsplit(url) if isinstance(url, str) else None
    if (
        parsed is None
        or parsed.scheme != "https"
        or parsed.hostname != "checkout.stripe.com"
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 443}
    ):
        raise ValueError("service did not return a Stripe-hosted HTTPS checkout URL")
    print(f"Amount: ${amount_cents / 100:.2f}")
    print(f"Complete payment in your browser: {url}", flush=True)
    if open_browser:
        try:
            opened = webbrowser.open(url)
        except Exception:
            opened = False
        if not opened:
            print("Browser unavailable. Open the link on another device.")
    print("After paying, use credits balance or credits history to check the deposit.")
    return session


def refund(service: ServiceClient, amount_cents: int, receipt_path: Path) -> dict:
    """Persist the refund identity before sending it and reuse it on every retry."""
    if amount_cents <= 0:
        raise ValueError("refund amount must be positive")
    identity = {
        "schema": "yugos.credit-refund.v1",
        "endpoint": service.endpoint,
        "hotkey": service.hotkey.ss58_address,
        "amount_cents": amount_cents,
    }
    with state_lock(receipt_path):
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_bytes())
            if any(receipt.get(key) != value for key, value in identity.items()):
                raise ValueError("refund receipt belongs to another amount, wallet, or service")
            if not isinstance(receipt.get("request_id"), str) or not receipt["request_id"]:
                raise ValueError("refund receipt has no request_id")
        else:
            receipt = {**identity, "request_id": str(uuid4())}
            save_json(receipt_path, receipt)
        response = receipt.get("response") or {}
        batch_pending = any(
            row.get("status") not in {"succeeded", "failed", "canceled"}
            for row in response.get("refunds", [])
        )
        if "response" not in receipt or batch_pending:
            from yugos.service_client import ServiceError

            payload = {"amount_cents": amount_cents, "request_id": receipt["request_id"]}
            try:
                response = service.request("POST", "/credits/refund", payload, signed=True)
            except ServiceError as exc:
                if exc.code != "use_billing_refund":
                    raise
                response = service.request("POST", "/billing/refund", payload, signed=True)
            if (
                response.get("ok") is not True
                or response.get("request_id") != receipt["request_id"]
            ):
                raise ValueError("service did not acknowledge this refund request")
            receipt["response"] = response
            save_json(receipt_path, receipt)
        return receipt


def run_credit_command(args) -> None:
    """Sign payment requests locally; the server verifies chain ownership when needed."""
    if args.credit_command == "add":
        if bool(args.to_hotkey) == bool(args.wallet):
            raise ValueError("provide --wallet and --hotkey, or --to-hotkey, to add credits")
        if args.wallet and not args.hotkey:
            raise ValueError("--hotkey is required with --wallet to select the credit account")
        if args.to_hotkey and args.hotkey:
            raise ValueError("--to-hotkey cannot be combined with --hotkey")
        destination = args.to_hotkey or configured_wallet(args).hotkeypub.ss58_address
        signing_key = None
    else:
        wallet = configured_wallet(args)
        signing_key = wallet.coldkey if args.credit_command == "claim" else wallet.hotkey
    import httpx

    with httpx.Client(timeout=60, follow_redirects=False) as client:
        service = ServiceClient(args.endpoint, client, signing_key)
        if args.credit_command == "add":
            checkout(service, args.amount_usd, hotkey=destination, open_browser=not args.no_browser)
            return
        if args.credit_command == "claim":
            result = service.request(
                "POST", "/billing/claim", {"hotkey": wallet.hotkeypub.ss58_address}, signed=True
            )
        elif args.credit_command == "refund":
            amount = args.amount_usd
            if args.all:
                if args.receipt.exists():
                    amount = json.loads(args.receipt.read_bytes())["amount_cents"]
                else:
                    balance = service.request("GET", "/credits", signed=True)
                    amount = balance.get("available_cents", balance["balance_cents"])
            try:
                result = refund(service, amount, args.receipt)
            except Exception as exc:
                raise RuntimeError(
                    f"{exc}\nRetry with the same --receipt {args.receipt}; "
                    "do not create a new refund receipt for this attempt."
                ) from exc
        else:
            path = "/credits" if args.credit_command == "balance" else "/credits/history"
            result = service.request("GET", path, signed=True)
        print(json.dumps(result, indent=2))

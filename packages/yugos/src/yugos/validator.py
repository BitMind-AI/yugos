"""Read hosted evaluation results and periodically publish Bittensor weights."""

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import httpx

from yugos.chain import add_chain_arguments, configured_chain, resolve_subnet
from yugos.competition import COMPETITION_VERSION, digest, king_weights
from yugos.service_client import ServiceClient, resolve_miner_endpoint, service_url
from yugos.state import save_json, state_lock

logger = logging.getLogger(__name__)
# Validators follow every competition of this major version; a major bump needs an upgrade.
COMPETITION_MAJOR = COMPETITION_VERSION.split(".")[0]


def read_result(raw: dict, *, netuid: int, now: datetime, max_age_s: float):
    """Confirmation time, weights and champion from an intact, fresh hosted result.

    Decisions are read as published, not re-scored, so the validator keeps working
    when the hosted rules, pack or record fields change within a major version.
    """
    try:
        payload = raw["payload"]
        if digest(payload) != raw["id"]:
            raise ValueError("competition publication digest differs")
        version = payload["history"]["policy"]["version"]
        if payload["netuid"] != netuid or version.split(".")[0] != COMPETITION_MAJOR:
            raise ValueError(f"publication is for another subnet or major version ({version})")
        confirmed_at = datetime.fromisoformat(raw["confirmed_at"])
        if (
            confirmed_at.tzinfo is None
            or not 0 <= (now - confirmed_at).total_seconds() <= max_age_s
        ):
            raise ValueError("competition confirmation is stale or dated in the future")
        kings = [
            event["measurement"]["categories"][0]["miner_id"]
            for event in payload["history"]["challenges"]
            if event["decision"]["promoted"]
        ]
        weights = {} if payload.get("paused") else king_weights(kings)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed competition publication: {exc!r}") from exc
    return confirmed_at, weights, kings[-1] if kings else None


def validator_step(
    service: ServiceClient,
    chain,
    *,
    state: dict,
    state_path: Path,
    blocks_between_weights: int,
    max_age_s: float,
    version_key: int,
    dry_run: bool = False,
    now: datetime | None = None,
) -> str:
    """Validate a fresh result, refresh miner identities, and publish when due."""
    current_time = now or datetime.now(timezone.utc)
    raw = service.request(
        "GET",
        "/scoring/snapshots/latest?" + urlencode({"netuid": chain.netuid}),
        signed=True,
    )
    confirmed_at, raw_weights, champion = read_result(
        raw, netuid=chain.netuid, now=current_time, max_age_s=max_age_s
    )
    previous = state.get("snapshot")
    if previous and confirmed_at < datetime.fromisoformat(previous["confirmed_at"]):
        raise ValueError("competition result rolled back")
    graph = chain.metagraph()
    neuron = chain.registered(graph, validator=True)
    by_hotkey = {miner.hotkey: miner.uid for miner in graph.neurons}
    # An absent incumbent must not redistribute its weight to older seats.
    if champion is None or champion not in by_hotkey:
        raw_weights = {}
    weights = {
        by_hotkey[key]: weight
        for key, weight in raw_weights.items()
        if key in by_hotkey and weight > 0
    }
    if weights:
        chain.check_winner_allocation()
    state["snapshot"] = raw
    save_json(state_path, state)
    if not weights:
        return "no eligible registered miners; weights unchanged"
    if dry_run:
        logger.info("preview weights: %s", weights)
        return "preview only; no weights submitted"
    last = max(neuron.last_update, state.get("last_success_block", 0))
    if last and graph.block - last < blocks_between_weights:
        return "waiting for the next weight interval"
    result = chain.publish(weights, version_key=version_key)
    state.update(
        last_success_block=graph.block, published_snapshot=raw["id"], block_hash=result.block_hash
    )
    save_json(state_path, state)
    return f"published weights for {len(weights)} miners from snapshot {raw['id']}"


def main(argv=None) -> None:
    """Run the hosted validator with persistent progress and graceful signal handling."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_chain_arguments(parser)
    parser.add_argument(
        "--endpoint",
        default=None,
        help="service override; required on test, mainnet uses upload.yugos.ai",
    )
    # Accepted and ignored so existing launch commands keep working.
    parser.add_argument("--cohort", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--state", type=Path, required=True, help="durable validator state file")
    parser.add_argument("--blocks-between-weights", type=int, default=100)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--max-snapshot-age-seconds", type=float, default=86400)
    parser.add_argument("--version-key", type=int, default=0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="check results and print weights without publishing"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stop = threading.Event()
    try:
        if (
            args.blocks_between_weights < 1
            or args.version_key < 0
            or any(
                not math.isfinite(v) or v <= 0
                for v in (args.poll_seconds, args.max_snapshot_age_seconds)
            )
        ):
            raise ValueError(
                "intervals and maximum snapshot age must be positive; version key nonnegative"
            )
        resolve_subnet(args)
        args.endpoint = resolve_miner_endpoint(args.endpoint, args.network)
        chain = configured_chain(args)
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: stop.set())
        with state_lock(args.state), httpx.Client(timeout=60, follow_redirects=False) as client:
            service = ServiceClient(args.endpoint, client, chain.wallet.hotkey)
            scope = {
                "network": args.network,
                "netuid": args.netuid,
                "endpoint": service.endpoint,
                "hotkey": chain.wallet.hotkey.ss58_address,
            }
            state = json.loads(args.state.read_bytes()) if args.state.exists() else {}
            # Older state files also pinned a cohort; it no longer matters.
            saved = {k: v for k, v in state.get("scope", scope).items() if k != "cohort"}
            if "endpoint" in saved:
                saved["endpoint"] = service_url(saved["endpoint"])
            if saved != scope:
                raise ValueError("state belongs to another validator or service")
            state["scope"] = scope
            while not stop.is_set():
                try:
                    status = validator_step(
                        service,
                        chain,
                        state=state,
                        state_path=args.state,
                        blocks_between_weights=args.blocks_between_weights,
                        max_age_s=args.max_snapshot_age_seconds,
                        version_key=args.version_key,
                        dry_run=args.dry_run,
                    )
                    logger.info(status)
                except Exception as exc:
                    if args.once:
                        raise
                    logger.error("validator cycle failed: %s", exc)
                if args.once:
                    break
                stop.wait(args.poll_seconds)
    except Exception as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()

"""Read hosted evaluation results and periodically publish Bittensor weights."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import httpx

from yugos.chain import add_chain_arguments, configured_chain, resolve_subnet
from yugos.competition import COMPETITION_VERSION, digest, king_weights
from yugos.service_client import ServiceClient, ServiceError, resolve_miner_endpoint, service_url
from yugos.state import save_json, state_lock

logger = logging.getLogger(__name__)
# Validators follow every competition of this major version; a major bump needs an upgrade.
COMPETITION_MAJOR = COMPETITION_VERSION.split(".")[0]
SNAPSHOT_ATTEMPTS = 10


def default_state_path(scope: dict) -> Path:
    """Keep each validator identity's progress in the user's persistent state directory."""
    root = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    if not root.is_absolute():
        root = Path.home() / ".local" / "state"
    return root / "yugos" / "validators" / f"{digest(scope)}.json"


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


def weight_schedule(chain, state: dict, minimum_interval: int) -> tuple[int, int, int]:
    """Return the current block, next eligible block, and effective weight interval."""
    graph = chain.metagraph()
    neuron = chain.registered(graph, validator=True)
    interval = max(minimum_interval, chain.check_weight_settings())
    last = max(neuron.last_update, state.get("last_success_block", 0))
    return graph.block, last + interval if last else graph.block, interval


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
    stop: threading.Event | None = None,
) -> str | None:
    """Check chain eligibility, then fetch a fresh result and publish when due."""
    block, next_block, _ = weight_schedule(chain, state, blocks_between_weights)
    logger.info("block: %s; next weight set: %s blocks", block, max(0, next_block - block))
    if not dry_run and block < next_block:
        return None
    stop = stop if stop is not None else threading.Event()
    for attempt in range(SNAPSHOT_ATTEMPTS):
        if stop.is_set():
            return "stopping before snapshot refresh"
        try:
            raw = service.request(
                "GET",
                "/scoring/snapshots/latest?" + urlencode({"netuid": chain.netuid}),
                signed=True,
            )
            break
        except (httpx.TransportError, ServiceError) as exc:
            if (
                isinstance(exc, ServiceError)
                and exc.status < 500
                and exc.status not in {408, 409, 429}
            ):
                raise
            if attempt + 1 == SNAPSHOT_ATTEMPTS:
                raise
            delay = min(5 * 2**attempt, 60)
            logger.warning(
                "snapshot refresh failed (%s/%s); retrying in %ss: %s",
                attempt + 1,
                SNAPSHOT_ATTEMPTS,
                delay,
                exc,
            )
            if stop.wait(delay):
                return "stopping during snapshot retry"
    confirmed_at, raw_weights, champion = read_result(
        raw, netuid=chain.netuid, now=now or datetime.now(timezone.utc), max_age_s=max_age_s
    )
    previous = state.get("snapshot")
    if previous and confirmed_at < datetime.fromisoformat(previous["confirmed_at"]):
        raise ValueError("competition result rolled back")
    # Miner UIDs may change while the results request is in flight.
    graph = chain.metagraph()
    chain.registered(graph, validator=True)
    by_hotkey = {miner.hotkey: miner.uid for miner in graph.neurons}
    # An absent incumbent must not redistribute its weight to older seats.
    if champion is None or champion not in by_hotkey:
        raw_weights = {}
    weights = {
        by_hotkey[key]: weight
        for key, weight in raw_weights.items()
        if key in by_hotkey and weight > 0
    }
    state["snapshot"] = raw
    save_json(state_path, state)
    if not weights:
        return "no eligible registered miners; weights unchanged"
    if dry_run:
        logger.info("preview weights: %s", weights)
        return "preview only; no weights submitted"
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
    parser.add_argument(
        "--state",
        type=Path,
        help="state file override; defaults to a validator-specific file under "
        "$XDG_STATE_HOME/yugos/validators or ~/.local/state/yugos/validators",
    )
    parser.add_argument("--blocks-between-weights", type=int, default=100)
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
            or not math.isfinite(args.max_snapshot_age_seconds)
            or args.max_snapshot_age_seconds <= 0
        ):
            raise ValueError(
                "intervals and maximum snapshot age must be positive; version key nonnegative"
            )
        resolve_subnet(args)
        args.endpoint = resolve_miner_endpoint(args.endpoint, args.network)
        chain = configured_chain(args)
        scope = {
            "network": args.network,
            "netuid": args.netuid,
            "endpoint": args.endpoint,
            "hotkey": chain.wallet.hotkey.ss58_address,
        }
        state_path = args.state if args.state is not None else default_state_path(scope)
        logger.info("validator state file: %s", state_path)
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: stop.set())
        with state_lock(state_path), httpx.Client(timeout=60, follow_redirects=False) as client:
            service = ServiceClient(args.endpoint, client, chain.wallet.hotkey)
            state = json.loads(state_path.read_bytes()) if state_path.exists() else {}
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
                        state_path=state_path,
                        blocks_between_weights=args.blocks_between_weights,
                        max_age_s=args.max_snapshot_age_seconds,
                        version_key=args.version_key,
                        dry_run=args.dry_run,
                        stop=stop,
                    )
                    if status:
                        logger.info(status)
                except Exception as exc:
                    if args.once:
                        raise
                    logger.error("validator cycle failed: %s", exc)
                if args.once:
                    break
                if not stop.is_set():
                    block, next_block, interval = weight_schedule(
                        chain, state, args.blocks_between_weights
                    )
                    if next_block <= block:
                        next_block = block + interval
                    while block < next_block and not stop.is_set():
                        chain.wait_for_block(min(block + 5, next_block), stop)
                        if not stop.is_set():
                            block = chain.metagraph().block
                            if block < next_block:
                                logger.info(
                                    "block: %s; next weight set: %s blocks",
                                    block,
                                    next_block - block,
                                )
    except Exception as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()

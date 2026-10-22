"""Wallet and chain operations shared by the miner and validator applications."""

from __future__ import annotations

import argparse
import re


def submission_commitment(digest: str) -> str:
    """Identify a Yugos ZIP by its digest in the chain's bounded metadata field."""
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("submission digest must be a lowercase SHA-256")
    return f"yugos:2:{digest}"


class Chain:
    """Use the configured hotkey for submission commitments and weight updates."""

    def __init__(self, wallet, subtensor, netuid: int):
        self.wallet = wallet
        self.subtensor = subtensor
        self.netuid = netuid

    def metagraph(self):
        return self.subtensor.subnets.metagraph(self.netuid)

    def registered(self, metagraph, *, validator: bool = False):
        """Require registration, and a permit or subnet ownership for validation."""
        neuron = metagraph.by_hotkey(self.wallet.hotkey.ss58_address)
        if neuron is None:
            raise ValueError(f"hotkey is not registered on subnet {self.netuid}")
        if validator and not neuron.validator_permit and neuron.hotkey != metagraph.owner_hotkey:
            raise ValueError("hotkey has no validator permit")
        return neuron

    def commit(self, digest: str):
        """Publish the ZIP digest and wait for a confirmed chain result."""
        from bittensor import calls

        payload = submission_commitment(digest).encode()
        info = {"fields": [{f"Raw{len(payload)}": "0x" + payload.hex()}]}
        result = self.subtensor.submit_call(
            calls.Commitments.set_commitment(self.netuid, info),
            self.wallet,
            signer="hotkey",
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        result.raise_for_failure()
        return result

    def publish(self, weights: dict[int, float], *, version_key: int):
        """Let the SDK enforce chain limits and select the weight publication method."""
        from bittensor import SetWeights

        result = self.subtensor.execute(
            SetWeights(netuid=self.netuid, weights=weights, version_key=version_key),
            self.wallet,
            wait_for_inclusion=True,
            wait_for_finalization=True,
            wait_for_registration=False,
        )
        result.raise_for_failure()
        return result

    def check_weight_settings(self) -> int:
        """Check supported weight settings and return the chain's minimum block interval."""
        params = self.subtensor.read("subnet_hyperparameters", netuid=self.netuid)
        minimum = (params or {}).get("min_allowed_weights")
        maximum = (params or {}).get("max_weights_limit")
        if type(minimum) is not int or type(maximum) is not int:
            raise ValueError("chain did not supply raw weight allocation limits")
        if minimum > 1 or minimum < 0 or maximum != 65535:
            raise ValueError("subnet weight limits do not permit winner-take-all allocation")
        interval = params.get("weights_rate_limit")
        if type(interval) is not int or interval < 0:
            raise ValueError("chain did not supply a nonnegative weights rate limit")
        return interval

    def wait_for_block(self, block: int, stop) -> None:
        """Use the SDK block subscription, checking shutdown between bounded waits."""
        while not stop.is_set():
            try:
                self.subtensor.wait_for_block(block, timeout=60)
                return
            except TimeoutError:
                continue


# Public subnets. A raw chain endpoint still needs an explicit --netuid.
NETWORK_SUBNETS = {"test": 169, "finney": 34}


def subnet_for_network(network: str) -> int:
    """Return the Yugos subnet for testnet or mainnet."""
    try:
        return NETWORK_SUBNETS[network]
    except KeyError:
        raise ValueError("pass --netuid for this network; test is 169 and finney is 34") from None


def resolve_subnet(args):
    """Fill the subnet from the network when the miner or validator omits --netuid."""
    if getattr(args, "netuid", None) is None:
        args.netuid = subnet_for_network(args.network)
    if args.netuid <= 0:
        raise ValueError("netuid must identify a non-root subnet")
    return args


def add_chain_arguments(parser: argparse.ArgumentParser) -> None:
    """Use the same explicit network, subnet, and wallet configuration in both apps."""
    parser.add_argument(
        "--network",
        default="finney",
        help="finney (subnet 34, default) or test (subnet 169)",
    )
    parser.add_argument(
        "--netuid",
        type=int,
        help="subnet override; defaults to 169 on test and 34 on finney",
    )
    add_wallet_arguments(parser)


def add_wallet_arguments(
    parser: argparse.ArgumentParser, *, required: bool = True, explicit_hotkey: bool = False
) -> None:
    """Select an existing signing wallet without requiring a chain connection."""
    parser.add_argument("--wallet", required=required, help="wallet name")
    parser.add_argument(
        "--hotkey",
        required=required and explicit_hotkey,
        default=None if explicit_hotkey else "default",
        help="hotkey name within the wallet",
    )
    parser.add_argument("--wallet-path", default="~/.bittensor/wallets")


def configured_wallet(args):
    """Open an existing wallet without creating or recovering keys."""
    import bittensor as bt

    return bt.Wallet(name=args.wallet, hotkey=args.hotkey, path=args.wallet_path)


def configured_chain(args) -> Chain:
    """Open an existing wallet without creating or recovering keys."""
    import bittensor as bt

    resolve_subnet(args)
    return Chain(
        configured_wallet(args),
        bt.Subtensor(args.network),
        args.netuid,
    )

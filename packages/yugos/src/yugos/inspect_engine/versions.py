"""Exact Inspect pins and the one private Inspect API the engine depends on."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

# Inspect ships several releases a month without a semver promise, and the
# active-sample lookup below is private. Upgrade these together, deliberately.
INSPECT_AI_VERSION = "0.3.272"
INSPECT_EVALS_VERSION = "0.22.0"
# The `yugos[inspect-modal]` pin of the inspect-sandboxes Modal provider, which an
# execution service selects for sandboxed tasks; Yugos itself does not import it.
INSPECT_SANDBOXES_VERSION = "0.6.0"

PINNED = {"inspect-ai": INSPECT_AI_VERSION, "inspect-evals": INSPECT_EVALS_VERSION}


class UnsupportedInspectVersion(RuntimeError):
    """The installed Inspect packages differ from the versions Yugos was verified against."""


def require_pinned_inspect() -> None:
    """Refuse to run on Inspect versions other than the pinned ones."""
    for package, pinned in PINNED.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            raise UnsupportedInspectVersion(
                f"{package} is not installed; install yugos[inspect]"
            ) from None
        if installed != pinned:
            raise UnsupportedInspectVersion(
                f"{package} {installed} is installed; Yugos is verified against {pinned}"
            )


@dataclass(frozen=True)
class SampleScope:
    """Operator-side identity of the Inspect sample a model call belongs to.

    `sample_uuid` equals `EvalSample.uuid` in the log and is the join key for
    receipts. None of these fields are forwarded to a miner's router.
    """

    sample_uuid: str
    sample_id: str
    epoch: int
    eval_id: str


def active_sample_scope() -> SampleScope | None:
    """Return the sample whose solver is currently running, if any.

    Inspect has no public accessor for this. `sample_active()` is private, so
    the pinned version check above gates it and a test exercises it end to end.
    """
    from inspect_ai.log._samples import sample_active

    active = sample_active()
    if active is None:
        return None
    return SampleScope(
        sample_uuid=active.sample_uuid,
        sample_id=str(active.sample.id),
        epoch=active.epoch,
        eval_id=active.eval_id,
    )

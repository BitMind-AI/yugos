"""`yugos_evals/code_tests`: candidate programs run against hidden tests in an isolated sandbox."""

from __future__ import annotations

import re
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.solver import generate
from inspect_ai.util import ComposeConfig, ComposeService, SandboxEnvironmentSpec

from ..scorers.code import code_tests_scorer
from ..sources.livecodebench import load_livecodebench
from ..sources.livecodebench_plus import SPLITS, load_livecodebench_plus

VERSION = "1-A"
_DIGEST_PINNED = r"[^\s@]+@sha256:[0-9a-f]{64}"


def code_sandbox(image: str, sandbox_type: str = "docker") -> SandboxEnvironmentSpec:
    """A one-service sandbox from a prepared, digest-pinned image without network access.

    The image needs only `python3` (3.11+, standard library). The service idles
    like Inspect's default compose service, so images whose own command exits
    (such as the official `python` images) stay up. Hosted runs replace this spec
    when freezing (for Modal: `x-modal: {block_network: true}`).
    """
    if sandbox_type == "local":
        raise ValueError("code_tests never runs candidate code in a local sandbox")
    if not re.fullmatch(_DIGEST_PINNED, image or ""):
        raise ValueError("code_tests needs a prepared image pinned by digest (name@sha256:...)")
    service = ComposeService(
        image=image,
        command="tail -f /dev/null",
        network_mode="none",
        working_dir="/tmp",
    )
    return SandboxEnvironmentSpec(sandbox_type, ComposeConfig(services={"default": service}))


@task
def code_tests(
    source: Literal["livecodebench", "livecodebench_plus"] = "livecodebench",
    image: str | None = None,
    sandbox_type: str = "docker",
    releases: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    splits: list[str] | None = None,
    exclude_releases: list[str] | None = None,
    data_dir: str | None = None,
    lcb_data_dir: str | None = None,
) -> Task:
    """Python code generation graded by stdin/stdout or functional tests in a sandbox.

    `livecodebench` takes `releases` (default `["v6"]`, the v6 additions only)
    and an inclusive contest-date window; `livecodebench_plus` takes `splits`
    and drops problems already in LiveCodeBench `exclude_releases` (default
    `["v6"]`; `lcb_data_dir` stages those files). `image` must be a prepared
    image pinned by digest; `sandbox_type` must be an isolated provider.
    """
    if image is None:
        raise ValueError("code_tests needs `image`: a prepared python3 image pinned by digest")
    spec = code_sandbox(image, sandbox_type)
    if source == "livecodebench":
        if splits is not None or exclude_releases is not None:
            raise ValueError("livecodebench selects releases and dates, not Plus splits")
        loaded = load_livecodebench(
            releases or ["v6"], start_date=start_date, end_date=end_date, data_dir=data_dir
        )
    elif source == "livecodebench_plus":
        if releases is not None or start_date is not None or end_date is not None:
            raise ValueError("livecodebench_plus selects splits, not releases or dates")
        loaded = load_livecodebench_plus(
            splits or list(SPLITS),
            exclude_releases=["v6"] if exclude_releases is None else exclude_releases,
            data_dir=data_dir,
            lcb_data_dir=lcb_data_dir,
        )
    else:
        raise ValueError(f"unknown code source: {source!r}")
    return Task(
        dataset=loaded.samples,
        solver=generate(),
        scorer=code_tests_scorer(),
        sandbox=spec,
        metadata={"source": source, **loaded.metadata()},
        version=VERSION,
    )

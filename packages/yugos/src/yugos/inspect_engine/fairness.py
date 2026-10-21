"""The Inspect settings every competition run uses, identical for all submissions.

Inspect's defaults are unsafe for a competition: adaptive concurrency throttles
a router that returns rate-limit errors while others scale up, retryable errors
retry without bound, one sample error aborts the whole eval, and `time_limit`
counts time spent queued behind other samples.

What the profile can express: static per-submission concurrency, bounded
retries and per-attempt timeouts, no sample-level re-runs, errors scored as
failed samples, one epoch, and a per-sample `working_limit` that excludes
queueing and retry backoff. What it cannot: Inspect checks `working_limit`
only between model calls, so a call in flight is bounded by `attempt_timeout`
instead; there is no per-sample wall-clock limit that ignores queueing, so no
`time_limit` is set; and a spend cap must come from the metering proxy, since
Inspect's `cost_limit` needs static per-token prices a router does not have.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FairnessProfile(BaseModel):
    """Concurrency, retry and limit settings shared by every submission in a cohort."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    # Samples in flight per submission; also its connection-pool size.
    concurrency: int = Field(default=8, ge=1, strict=True)
    max_sandboxes: int = Field(default=8, ge=1, strict=True)
    # Retries apply only to RouterUnavailable, which cannot duplicate spend.
    max_retries: int = Field(default=2, ge=0, strict=True)
    attempt_timeout_s: int = Field(default=600, ge=1, strict=True)
    call_timeout_s: int = Field(default=1800, ge=1, strict=True)
    working_limit_s: int = Field(default=3600, ge=1, strict=True)


def competition_eval_options(profile: FairnessProfile | None = None) -> dict[str, Any]:
    """Keyword arguments for `inspect_ai.eval` that every competition run must pass."""
    profile = profile or FairnessProfile()
    return {
        "max_connections": profile.concurrency,
        "adaptive_connections": False,
        "max_samples": profile.concurrency,
        "max_sandboxes": profile.max_sandboxes,
        "max_retries": profile.max_retries,
        "attempt_timeout": profile.attempt_timeout_s,
        "timeout": profile.call_timeout_s,
        "working_limit": profile.working_limit_s,
        "fail_on_error": False,
        "retry_on_error": 0,
        "score_on_error": False,
        "epochs": 1,
        "log_samples": True,
    }

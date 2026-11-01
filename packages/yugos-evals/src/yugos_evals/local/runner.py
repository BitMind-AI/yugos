"""Run a frozen pack against a local router bundle, with hosted tasks, fairness and pricing.

The run restores the pack's frozen Inspect tasks, hosts the bundle behind the
submission protocol, meters each provider call with world-card prices, runs
every task with `run_competition_eval`, and measures the logs with core's
competition measurements. Local results are for development: the hosted
service's isolation, provider receipts and private packs are not reproduced.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalLog
from inspect_ai.util import SandboxEnvironmentSpec
from inspect_ai.util._sandbox.compose import ComposeConfig

from yugos.competition import competition_grade
from yugos.evaluation.packs import read_pack, records_bytes
from yugos.inference.actions import ModelChoice, require_supported_models
from yugos.inference.calls import CallLimits
from yugos.inspect_engine import (
    FairnessProfile,
    JsonlReceipts,
    TaskConfig,
    category_measurements_from_log,
    measure_log,
    resolve_task_roles,
    restore_tasks,
    router_model,
    run_competition_eval,
)
from yugos.state import save_json
from yugos.submission.bundle import inspect_bundle
from yugos.submission.session import SessionError, SubmissionSession, WorkerTimeouts

from .providers import LocalProviderClient
from .router import LocalRouter, SubprocessWorker, active_router
from .world import load_world_card

RUN_SCHEMA = "yugos-evals.local-run.v1"


class LocalRunError(RuntimeError):
    """The run could not start; nothing was spent."""


@dataclass
class RunOptions:
    pack: Path
    bundle: Path
    max_cost_usd: float
    log_dir: Path
    world_card: str | None = None
    limit: int | None = None
    service_models: Mapping[str, ModelChoice] = field(default_factory=dict)
    limits: CallLimits | None = None
    display: str | None = None


@dataclass
class RunResult:
    path: Path
    summary: dict[str, Any]
    failure: str | None


def parse_model_choice(text: str) -> ModelChoice:
    """`provider/model` or `provider/model@effort`, naming a world-card model."""
    choice, _, effort = text.partition("@")
    provider, sep, model = choice.partition("/")
    if not sep or not provider or not model:
        raise ValueError(f"expected provider/model[@effort], got {text!r}")
    return ModelChoice(provider_id=provider, model_id=model, effort=effort or None)


def use_local_sandboxes(tasks, *, docker_available: bool) -> None:
    """Run frozen sandboxes with Inspect's local Docker provider on the same pinned images.

    A frozen sandbox is a Compose configuration of digest-pinned images for the
    hosted provider; locally the same configuration runs under `docker`.
    """
    owners = [owner for task, _ in tasks for owner in (task, *task.dataset) if owner.sandbox]
    if not owners:
        return
    for owner in owners:
        spec = owner.sandbox
        config = getattr(spec, "config", None)
        if isinstance(config, dict) and config.get("services"):
            config = ComposeConfig.model_validate(config)
        if not isinstance(config, ComposeConfig):
            raise LocalRunError("sandboxed tasks not yet supported locally for this pack")
        owner.sandbox = SandboxEnvironmentSpec(type="docker", config=config)
    if not docker_available:
        raise LocalRunError("this pack has sandboxed tasks; install and start Docker to run it")


def _profile(limits: CallLimits, task) -> FairnessProfile:
    """The hosted executor's profile: one sample at a time, no retries, per-question deadlines."""
    seconds = math.ceil(limits.question_timeout_s)
    working = seconds
    if task.working_limit:
        working = min(seconds, max(1, math.ceil(task.working_limit)))
    return FairnessProfile(
        concurrency=1,
        max_sandboxes=1,
        max_retries=0,
        attempt_timeout_s=seconds,
        call_timeout_s=seconds,
        working_limit_s=working,
    )


def _task_binding_key(definition: TaskConfig) -> str:
    """Service model bindings are independent of sandbox placement."""
    return definition.model_copy(update={"sandbox": None}).model_dump_json()


def run_local(
    options: RunOptions,
    *,
    env: Mapping[str, str] | None = None,
    card_cache: Path | None = None,
    docker_available: bool | None = None,
) -> RunResult:
    """Evaluate the bundle on the pack and write measurements, receipts and Inspect logs.

    Everything that can be checked without spending is checked first, raising
    `LocalRunError`: the pack, world card, bundle declarations, role bindings,
    API keys and sandbox support. A failure after that stops further spending,
    and the partial evidence is still written.
    """
    if not options.max_cost_usd > 0:
        raise LocalRunError("--max-cost must be a positive number of USD")
    if options.limit is not None and options.limit < 1:
        raise LocalRunError("--limit must be positive")
    run_dir = Path(options.log_dir)
    if run_dir.exists():
        raise LocalRunError(f"{run_dir} already exists; choose a new --log-dir")
    pack = read_pack(options.pack)
    questions = pack.questions[: options.limit] if options.limit else pack.questions
    card = load_world_card(
        options.world_card, pinned=pack.manifest.world_card_sha256, cache_dir=card_cache
    )
    metadata = inspect_bundle(Path(options.bundle))
    pool = card.pool()
    try:
        require_supported_models(metadata.required_models, pool)
    except ValueError as exc:
        raise LocalRunError(f"the bundle declares models outside the world card: {exc}") from exc
    task_bindings = {}
    needed = set()
    try:
        for question in questions:
            definition = TaskConfig.model_validate(question.inspect_task)
            key = _task_binding_key(definition)
            if key not in task_bindings:
                task_bindings[key] = resolve_task_roles(definition, options.service_models)
            needed.update(definition.required_roles)
        if set(options.service_models) - needed:
            raise ValueError("the pack does not use some configured model roles")
    except ValueError as exc:
        raise LocalRunError(str(exc)) from exc

    bound = {binding.model for roles in task_bindings.values() for binding in roles.values()}
    try:
        router_client = LocalProviderClient(card, metadata.required_models, env=env)
        service_client = LocalProviderClient(card, bound, env=env) if bound else None
    except ValueError as exc:
        raise LocalRunError(str(exc)) from exc
    limits = (options.limits or CallLimits(max_cost_usd=1, max_question_cost_usd=1)).model_copy(
        update={"max_cost_usd": options.max_cost_usd, "max_question_cost_usd": options.max_cost_usd}
    )
    run_id = uuid.uuid4().hex
    receipts = JsonlReceipts(run_dir / "receipts.jsonl")
    calls_path = run_dir / "calls.jsonl"

    def record(call) -> None:
        with calls_path.open("a", encoding="utf-8") as file:
            file.write(call.model_dump_json() + "\n")

    router = LocalRouter(
        None,
        run_id=run_id,
        router_client=router_client,
        service_client=service_client,
        limits=limits,
        receipts=receipts,
        service_receipts=JsonlReceipts(run_dir / "service-receipts.jsonl"),
        record=record,
    )
    task_models = {
        key: {
            role: router_model(
                partial(
                    router.service_call,
                    binding.model,
                    pinned=binding.pin.params if binding.pin else {},
                ),
                submission_id=f"{binding.model.provider_id}/{binding.model.model_id}",
                config=binding.pin.config() if binding.pin else None,
            )
            for role, binding in roles.items()
        }
        for key, roles in task_bindings.items()
    }

    def roles_for_task(definition):
        return task_models[_task_binding_key(definition)]

    model = router_model(router, submission_id=metadata.submission_sha256)
    try:
        tasks = restore_tasks(questions, model_roles=roles_for_task)
    except ValueError as exc:
        raise LocalRunError(f"the pack cannot be restored: {exc}") from exc
    if docker_available is None:
        docker_available = shutil.which("docker") is not None
    use_local_sandboxes(tasks, docker_available=docker_available)

    run_dir.mkdir(parents=True)
    scratch = Path(tempfile.mkdtemp(prefix="yugos-worker-"))
    worker = SubprocessWorker(
        Path(options.bundle), scratch / "runtime", log=run_dir / "worker.log", env=env
    )
    router.session = SubmissionSession(worker, pool, WorkerTimeouts())
    logs: list[EvalLog] = []
    failure: str | None = None
    token = active_router.set(
        (
            router,
            frozenset(
                {str(model), *(str(m) for roles in task_models.values() for m in roles.values())}
            ),
        )
    )
    try:
        router.session.load(metadata.submission_sha256, metadata.manifest_sha256)
        for task, definition in tasks:
            display = {"display": options.display} if options.display else {}
            log = run_competition_eval(
                task,
                model,
                log_dir=str(run_dir / "logs"),
                required_roles=definition.required_roles,
                model_roles=roles_for_task(definition),
                profile=_profile(limits, task),
                sample_shuffle=False,
                log_realtime=False,
                log_shared=False,
                **display,
            )
            logs.append(log)
            if router.failure is not None or log.status != "success":
                break
    except SessionError as exc:
        failure = f"the bundle could not be loaded ({exc.failure.code}): {exc}"
    except Exception as exc:  # noqa: BLE001 - keep the evidence of a partial run
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        active_router.reset(token)
        cleanup = router.session.close()
        shutil.rmtree(scratch, ignore_errors=True)
    if router.failure is not None:
        failure = f"run stopped: {router.failure}"
    elif failure is None and any(log.status != "success" for log in logs):
        failure = "an Inspect task did not complete; see the logs"

    result = _measurements(
        pack=pack,
        pack_path=Path(options.pack),
        questions=questions,
        logs=logs,
        receipts=receipts.receipts(),
        run_id=run_id,
        metadata=metadata,
        integrity=cleanup.reaped,
        open_reservations=[
            call.response.spend.reservation_usd
            for call in router.meter.budget.calls
            if call.response.spend.status != "known"
        ],
    )
    if failure is None and any(not row["complete"] for row in result["summary"].values()):
        failure = "one or more questions are incomplete; see the logs"
    result.update(
        status="failed" if failure else "complete",
        failure=failure,
        world_card_sha256=card.sha256,
        max_cost_usd=options.max_cost_usd,
        spent_usd={
            "router": router.meter.cost_usd,
            "services": router.service_meter.cost_usd if router.service_meter else 0.0,
        },
        service_models=[choice.model_dump() for choice in sorted(bound, key=str)],
        logs=[log.location for log in logs],
        created_at=datetime.now(UTC).isoformat(),
    )
    save_json(run_dir / "measurements.json", result)
    return RunResult(run_dir, result, failure)


def _measurements(
    *, pack, pack_path, questions, logs, receipts, run_id, metadata, integrity, open_reservations
) -> dict[str, Any]:
    assigned = {q.question_id: q.workload_class for q in questions}
    logged = {s.uuid for log in logs for s in log.samples or ()}
    # Spend in a task whose log was never written cannot be attributed to a question.
    unattributed = [r for r in receipts if r.sample_uuid not in logged]
    receipts = [r for r in receipts if r.sample_uuid in logged]
    cohort = hashlib.sha256(records_bytes(questions)).hexdigest()
    categories = category_measurements_from_log(
        logs,
        receipts,
        assigned,
        run_id=run_id,
        miner_id="local",
        submission_sha256=metadata.submission_sha256,
        cohort_sha256=cohort,
        integrity_pass=integrity,
        open_reservations=open_reservations,
        isolate_errors=True,
    )
    rows = []
    for log in logs:
        uuids = {s.uuid for s in log.samples or ()}
        for row in measure_log(
            log, [r for r in receipts if r.sample_uuid in uuids], isolate_errors=True
        ):
            rows.append(
                {
                    "question_id": row.measurement.question_id,
                    "category": row.measurement.workload_class,
                    "source_benchmark": row.measurement.source_benchmark,
                    "grade": str(competition_grade(row.measurement)),
                    "cost_usd": str(row.spend_usd),
                    "active_s": row.active_ns / 1e9,
                    "grade_rule": row.measurement.grade_rule,
                    "failure_owner": row.failure_owner,
                    "error": row.error,
                    "sample_uuid": row.sample_uuid,
                }
            )
    return {
        "schema": RUN_SCHEMA,
        "comparable_to_hosted": False,
        "pack": {
            "path": str(pack_path),
            "sha256": pack.sha256,
            "pack_id": pack.manifest.pack_id,
            "purpose": pack.manifest.purpose,
            "data_visibility": pack.manifest.data_visibility,
            "records_sha256": pack.manifest.records_sha256,
            "cohort_sha256": cohort,
        },
        "submission": {
            "sha256": metadata.submission_sha256,
            "manifest_sha256": metadata.manifest_sha256,
            "name": metadata.name,
            "version": metadata.version,
        },
        "question_ids": list(assigned),
        "panel_solve_shares": {
            q.question_id: None if q.panel_solve_share is None else str(q.panel_solve_share)
            for q in questions
        },
        "categories": [c.model_dump(mode="json") for c in categories],
        "questions": rows,
        "summary": summarize(categories),
        "unattributed_spend_usd": str(sum((r.cost_usd for r in unattributed), Decimal(0))),
    }


def summarize(categories) -> dict[str, dict[str, Any]]:
    """Per-category quality, spend per question and pre-scoring time."""
    summary = {}
    for row in categories:
        n = len(row.grades)
        summary[row.category] = {
            "questions": n,
            "quality": str(row.quality),
            "cost_usd": str(row.cost_usd),
            "cost_per_question_usd": str(row.cost_usd / n),
            "time_s": row.execution_time_ns / 1e9,
            "time_per_question_s": row.execution_time_ns / 1e9 / n,
            "complete": row.execution_complete
            and row.spend.status == "known"
            and row.integrity_pass,
        }
    return summary


def format_summary(result: dict[str, Any]) -> str:
    rows = [("CATEGORY", "N", "QUALITY", "COST/Q (USD)", "TIME (s)", "TIME/Q (s)", "COMPLETE")]
    for category, row in result["summary"].items():
        rows.append(
            (
                category,
                str(row["questions"]),
                f"{Decimal(row['quality']):.3f}",
                f"{Decimal(row['cost_per_question_usd']):.6f}",
                f"{row['time_s']:.1f}",
                f"{row['time_per_question_s']:.2f}",
                "yes" if row["complete"] else "no",
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]
    spent = result["spent_usd"]
    lines.append(
        f"spend: router ${spent['router']:.6f}, service roles ${spent['services']:.6f} "
        f"(cap ${result['max_cost_usd']:.2f})"
    )
    return "\n".join(lines)


def load_run(path: Path) -> dict[str, Any]:
    """A run's measurements, from its directory or its measurements.json."""
    path = Path(path)
    if path.is_dir():
        path = path / "measurements.json"
    data = json.loads(path.read_text())
    if data.get("schema") != RUN_SCHEMA:
        raise ValueError(f"{path} is not a yugos-evals local run")
    return data

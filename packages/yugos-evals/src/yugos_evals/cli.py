"""`yugos-evals` command: inspect the catalog and run benchmarks locally."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import get_args

from yugos.evaluation.measurements import WorkloadClass

from .catalog import Status, entries


def _list(args: argparse.Namespace) -> int:
    selected = entries(category=args.category, status=args.status)
    if args.json:
        print(json.dumps([e.model_dump(mode="json") for e in selected], indent=2))
        return 0
    rows = [("ID", "CATEGORY", "STATUS", "TASK", "SCORE KEY")]
    rows += [(e.id, e.category, e.status, e.task, e.score_key or "-") for e in selected]
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
    return 0


def _key_value(text: str) -> tuple[str, str]:
    key, sep, value = text.partition("=")
    if not sep or not key or not value:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    return key, value


def _run(args: argparse.Namespace) -> int:
    from .local.runner import (
        LocalRunError,
        RunOptions,
        format_summary,
        parse_model_choice,
        run_local,
    )

    try:
        roles = {role: parse_model_choice(model) for role, model in args.grader_model}
        if len(roles) != len(args.grader_model):
            raise ValueError("bind each model role once")
        log_dir = args.log_dir or Path("yugos-runs") / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        result = run_local(
            RunOptions(
                pack=args.pack,
                bundle=args.bundle,
                max_cost_usd=args.max_cost,
                log_dir=log_dir,
                world_card=args.world_card,
                limit=args.limit,
                service_models=roles,
                display=args.display,
            )
        )
    except (LocalRunError, ValueError, OSError) as exc:
        print(f"yugos-evals run: {exc}", file=sys.stderr)
        return 2
    print(format_summary(result.summary))
    print(f"results: {result.path} (Inspect logs: inspect view --log-dir {result.path / 'logs'})")
    print("local results are for development and are not comparable to hosted scores")
    if result.failure:
        print(f"yugos-evals run: {result.failure}", file=sys.stderr)
        return 1
    return 0


def _compare(args: argparse.Namespace) -> int:
    from .local.compare import compare_runs, format_comparison, load_policy
    from .local.runner import load_run

    try:
        a, b = load_run(args.run_a), load_run(args.run_b)
        result = compare_runs(a, b, load_policy(args.policy))
    except (ValueError, OSError, KeyError) as exc:
        print(f"yugos-evals compare: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.as_json(), indent=2))
    else:
        print(format_comparison(a, b, result))
    return 0


def _pack_build(args: argparse.Namespace) -> int:
    from .local.packbuild import build_local_pack

    out = args.out or Path(f"{args.benchmark}.pack.tar")
    try:
        task_args = {key: json.loads(value) for key, value in args.arg}
        digest, manifest = build_local_pack(
            args.benchmark, out, limit=args.limit, seed=args.seed, task_args=task_args
        )
    except (ValueError, KeyError, OSError) as exc:
        print(f"yugos-evals pack build: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"path": str(out), "sha256": digest, **manifest.model_dump(mode="json", by_alias=True)},
            indent=2,
        )
    )
    print(
        "development pack: public questions without holdouts or calibration; "
        "not comparable to competition packs",
        file=sys.stderr,
    )
    return 0


def _pack_assemble(args: argparse.Namespace) -> int:
    from .local.assemble import assemble_packs

    try:
        assembled = assemble_packs(args.packs, args.out, calibration=args.calibration)
    except (ValueError, OSError) as exc:
        print(f"yugos-evals pack assemble: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "path": str(args.out),
                "sha256": assembled.sha256,
                **assembled.manifest.model_dump(mode="json", by_alias=True),
                "question_counts": dict(assembled.question_counts),
                "question_count": assembled.question_count,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yugos-evals", description="Yugos benchmark catalog and local evaluation."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="Print the benchmark catalog.")
    listing.add_argument("--category", choices=get_args(WorkloadClass))
    listing.add_argument("--status", choices=get_args(Status))
    listing.add_argument("--json", action="store_true", help="Print entries as JSON.")
    listing.set_defaults(handler=_list)

    run = commands.add_parser("run", help="Evaluate a router bundle on a pack locally.")
    run.add_argument("--pack", type=Path, required=True, help="pack.tar to evaluate")
    run.add_argument("--bundle", type=Path, required=True, help="ZIP from `yugos pack`")
    run.add_argument(
        "--max-cost",
        type=float,
        default=100,
        help="hard spend cap in USD for router and service-role calls (default: 100)",
    )
    run.add_argument("--world-card", help="world card JSON file or URL (default: the service's)")
    run.add_argument("--log-dir", type=Path, help="new results directory")
    run.add_argument("--limit", type=int, help="evaluate only the pack's first N questions")
    run.add_argument(
        "--grader-model",
        type=_key_value,
        action="append",
        default=[],
        metavar="ROLE=PROVIDER/MODEL[@EFFORT]",
        help="default world-card model for a task role without a model pin",
    )
    run.add_argument("--display", help="Inspect display mode (e.g. full, plain, none)")
    run.set_defaults(handler=_run)

    compare = commands.add_parser("compare", help="Net value of run B over run A on one pack.")
    compare.add_argument("run_a", type=Path)
    compare.add_argument("run_b", type=Path)
    compare.add_argument("--policy", type=Path, help="net-value policy JSON/YAML override")
    compare.add_argument("--json", action="store_true")
    compare.set_defaults(handler=_compare)

    pack = commands.add_parser("pack", help="Local benchmark packs.")
    pack_commands = pack.add_subparsers(dest="pack_command", required=True)
    build = pack_commands.add_parser("build", help="Build a development pack from the catalog.")
    build.add_argument("--benchmark", required=True, help="catalog ID (see `yugos-evals list`)")
    build.add_argument("--limit", type=int, help="number of samples to select")
    build.add_argument("--seed", type=int, default=0, help="selection seed")
    build.add_argument("--out", type=Path, help="output pack (default: <benchmark>.pack.tar)")
    build.add_argument(
        "--arg",
        type=_key_value,
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="extra task argument, e.g. --arg release='\"2026-09\"'",
    )
    build.set_defaults(handler=_pack_build)

    assemble = pack_commands.add_parser(
        "assemble",
        help="Merge packs that share purpose and data visibility.",
        description=(
            "Pass at least two pack files, or one directory. A directory's inputs are "
            "the *.tar files directly inside it (including *.pack.tar), sorted by "
            "filename. Other files are ignored. Do not mix files and a directory."
        ),
    )
    assemble.add_argument(
        "packs",
        type=Path,
        nargs="+",
        help="two or more pack files, or a single directory of *.tar packs",
    )
    assemble.add_argument("--out", type=Path, required=True, help="output pack (not overwritten)")
    assemble.add_argument(
        "--calibration",
        type=Path,
        help="difficulty-calibration.json to include in the assembled pack",
    )
    assemble.set_defaults(handler=_pack_assemble)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())

"""Package a learned submission and send it to the hosted evaluation service."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

from yugos.chain import add_chain_arguments, configured_chain, resolve_subnet
from yugos.inference.actions import ModelChoice
from yugos.service_client import ServiceClient, resolve_miner_endpoint, service_url
from yugos.state import canonical_json, save_json, state_lock
from yugos.submission.bundle import (
    MAX_EXPANDED_BYTES,
    MAX_MANIFEST_BYTES,
    MAX_MEMBERS,
    BundleManifest,
    _safe_member,
    inspect_bundle,
)

from .display import Activity as _Heartbeat
from .display import (
    finish_submission,
    funding_wait_stopped,
    progress_message,
    submission_error,
    submission_header,
)
from .funding import FundingWaitInterrupted, save_status_url


def add_miner_endpoint_argument(parser, *, defer_default: bool = False) -> None:
    """Add a service override, optionally leaving resolution to the command."""
    parser.add_argument(
        "--endpoint",
        default=None if defer_default else resolve_miner_endpoint(None),
        help="service URL override (or set YUGOS_ENDPOINT)",
    )


def _payload_files(source: Path, destination: Path) -> dict[str, Path]:
    """Discover payloads with bounded traversal, pruning local metadata directories."""
    paths = {}
    pending = [source]
    directories = 1
    visited = 0
    expanded = 0
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                visited += 1
                if visited > 2 * MAX_MEMBERS:
                    raise ValueError("source exceeds the directory traversal limit")
                path = Path(entry.path)
                relative = path.relative_to(source)
                if entry.name.startswith(".") or entry.name == "__pycache__":
                    continue
                if (
                    relative.as_posix() == "manifest.json"
                    or path.suffix in {".zip", ".pyc", ".pyo"}
                    or entry.name.endswith(
                        (
                            ".receipt.json",
                            ".receipt.json.lock",
                            "-receipt.json",
                            "-receipt.json.lock",
                        )
                    )
                ):
                    continue
                filename = _safe_member(relative.as_posix())
                if entry.is_symlink():
                    raise ValueError(f"payload must be a regular file without symlinks: {filename}")
                if entry.is_dir(follow_symlinks=False):
                    directories += 1
                    if directories > MAX_MEMBERS:
                        raise ValueError("source exceeds the directory traversal limit")
                    pending.append(path)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    raise ValueError(f"payload must be a regular file without symlinks: {filename}")
                if path.resolve() == destination:
                    raise ValueError("output must not overwrite a payload file")
                expanded += entry.stat(follow_symlinks=False).st_size
                if len(paths) + 2 > MAX_MEMBERS or expanded > MAX_EXPANDED_BYTES:
                    raise ValueError("payload exceeds the bundle size or member limit")
                paths[filename] = path
    return paths


def package(
    source: Path,
    output: Path,
    *,
    name: str,
    required_models: tuple[ModelChoice, ...],
):
    """Bundle a model directory with generated hashes, versions and declarations.

    Read files without importing miner code. Invalid dependencies or payloads
    fail without publishing an archive. Publish atomically without overwriting an
    existing output; packaging needs no hosted service.
    """
    source = source.resolve()
    if not source.is_dir():
        raise ValueError("source must be a model directory")
    destination = output.resolve()
    if destination in {
        source / name for name in ("model.py", "model.safetensors", "models.json", "manifest.json")
    }:
        raise ValueError("output must not overwrite a payload file")
    paths = _payload_files(source, destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        archive_path = Path(scratch) / "submission.zip"
        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as archive:
            hashes = {}
            expanded = 0
            for filename, path in sorted(paths.items()):
                digest = hashlib.sha256()
                with (
                    path.open("rb") as payload,
                    archive.open(filename, "w", force_zip64=True) as member,
                ):
                    while chunk := payload.read(1024 * 1024):
                        expanded += len(chunk)
                        if expanded > MAX_EXPANDED_BYTES:
                            raise ValueError("payload exceeds the expanded bundle size limit")
                        digest.update(chunk)
                        member.write(chunk)
                hashes[filename] = digest.hexdigest()
            manifest = BundleManifest(
                name=name,
                runtime="python-safetensors-v2",
                required_models=required_models,
                files=hashes,
            )
            raw = canonical_json(manifest.model_dump(by_alias=True))
            if len(raw) > MAX_MANIFEST_BYTES or expanded + len(raw) > MAX_EXPANDED_BYTES:
                raise ValueError("manifest exceeds the expanded bundle size limit")
            with archive.open("manifest.json", "w", force_zip64=True) as member:
                member.write(raw)
        metadata = inspect_bundle(archive_path)
        try:
            output.hardlink_to(archive_path)
        except FileExistsError:
            raise FileExistsError(
                f"Output already exists: {output}. Choose a different path."
            ) from None
    return metadata


def _say(message: str) -> None:
    """Show submission progress immediately. The final receipt stays machine-readable JSON."""
    progress_message(message)


def _inspect_submission(archive: Path):
    with _Heartbeat("Verifying submission ZIP and file hashes."):
        metadata = inspect_bundle(archive)
    _say("Submission ZIP verified.")
    return metadata


def job_progress(result: dict) -> str | None:
    """Summarize a hosted job for the status command."""
    if not isinstance(result, dict):
        return None
    phase = result.get("phase") or result.get("status")
    if not isinstance(phase, str) or not phase:
        return None
    job = result.get("id") or "submission"
    parts = [f"Job {job}: {phase}."]
    hint = result.get("hint")
    if isinstance(hint, str) and hint:
        parts.append(hint if hint.endswith(".") else hint + ".")
    spend = result.get("spend")
    if isinstance(spend, list):
        parts.append(f"{len(spend)} provider calls recorded.")
    return " ".join(parts)


def submit(
    archive: Path,
    receipt_path: Path,
    service: ServiceClient,
    chain,
    *,
    network: str,
    renew_funding: bool = False,
) -> dict:
    """Resume upload, confirmation, and commitment from a durable submission receipt."""
    metadata = _inspect_submission(archive)
    with _Heartbeat("Checking hotkey registration."):
        chain.registered(chain.metagraph())
    _say(f"Hotkey registered on subnet {chain.netuid}.")
    identity = {
        "schema": "yugos.submission.v1",
        "network": network,
        "netuid": chain.netuid,
        "hotkey": chain.wallet.hotkey.ss58_address,
    }
    return _upload(
        archive,
        receipt_path,
        service,
        metadata,
        identity,
        coldkey=chain.wallet.coldkeypub.ss58_address,
        commit=chain.commit,
        renew_funding=renew_funding,
    )


def upload(archive: Path, receipt_path: Path, service: ServiceClient, *, coldkey: str) -> dict:
    """Upload and start hosted evaluation without requiring a chain connection."""
    if service.hotkey is None or not coldkey:
        raise ValueError("hosted upload requires a signing hotkey and public coldkey")
    metadata = _inspect_submission(archive)
    return _upload(
        archive,
        receipt_path,
        service,
        metadata,
        {
            "schema": "yugos.hosted-submission.v1",
            "hotkey": service.hotkey.ss58_address,
            "coldkey": coldkey,
        },
        coldkey=coldkey,
    )


def _submission_state(receipt, data, path):
    """Cache the canonical response shared by submission and status requests."""
    model = data.get("model_id")
    expected = f"bundles/{receipt['hotkey']}/{receipt['submission_sha256']}.zip"
    if (
        type(model) is not int
        or model <= 0
        or data.get("r2_key") != expected
        or data.get("file_hash", receipt["submission_sha256"]) != receipt["submission_sha256"]
    ):
        raise ValueError("service returned an unrelated submission")
    funding = data.get("funding")
    if funding:
        if (
            funding.get("model_id") != model
            or funding.get("hotkey") != receipt["hotkey"]
            or not isinstance(funding.get("portal_token"), str)
            or not funding.get("request_id")
            or any(
                funding.get("quote", {}).get(k) != receipt[k]
                for k in ("network", "netuid")
                if k in receipt
            )
        ):
            raise ValueError("service returned unrelated submission funding")
        receipt["funding"] = {k: funding[k] for k in ("request_id", "portal_token")}
        funding = {k: v for k, v in funding.items() if k != "portal_token"}
    elif "funding" in data:
        receipt.pop("funding", None)
        receipt.pop("status_url", None)
    receipt.update(model_id=model, r2_key=expected)
    if "uploaded" in data:
        receipt["uploaded"] = data["uploaded"] is True
    if "job_id" in data or "funding" in data:
        receipt.pop("job_id", None)
        receipt.pop("confirmed", None)
    job = data.get("job_id") or (funding or {}).get("job_id")
    if job:
        receipt.update(job_id=str(job), confirmed=True, uploaded=True)
    save_status_url(receipt, path)
    save_json(path, receipt)
    return funding


def _upload(
    archive, receipt_path, service, metadata, identity, *, coldkey, commit=None, renew_funding=False
):
    if archive.resolve() == receipt_path.resolve():
        raise ValueError("receipt must not overwrite the submission ZIP")
    identity = {
        **identity,
        "endpoint": service.endpoint,
        "submission_sha256": metadata.submission_sha256,
        "manifest_sha256": metadata.manifest_sha256,
    }
    with state_lock(receipt_path):
        receipt = (
            json.loads(receipt_path.read_bytes()) if receipt_path.exists() else identity.copy()
        )
        if "endpoint" in receipt:
            receipt["endpoint"] = service_url(receipt["endpoint"])
        if any(receipt.get(key) != value for key, value in identity.items()):
            raise ValueError("receipt belongs to another submission, wallet, network, or service")
        if receipt_path.exists():
            _say(f"Resuming from receipt: {receipt_path.resolve()}")
        saved_receipt = receipt_path.exists()
        with _Heartbeat("Checking submission on the service."):
            response = service.request(
                "POST",
                "/submissions",
                {
                    "filename": archive.name,
                    "expected_hash": metadata.submission_sha256,
                    "file_size": archive.stat().st_size,
                    "upload_method": "multipart",
                    "upload_id": (receipt.get("multipart") or {}).get("upload_id"),
                    **{k: identity[k] for k in ("network", "netuid") if k in identity},
                },
                signed=True,
            )["data"]
        funding_status = _submission_state(receipt, response, receipt_path)
        if not saved_receipt:
            _say(f"Submission receipt saved: {receipt_path.resolve()}")
        if renew_funding and receipt.get("funding") and not receipt.get("confirmed"):
            previous = receipt["funding"]
            canceled = service.request(
                "POST", "/billing/intents/" + previous["request_id"] + "/cancel", {}, signed=True
            )
            if canceled.get("job_id"):
                receipt.update(confirmed=True, job_id=str(canceled["job_id"]))
            elif canceled.get("state") in {"canceled", "expired", "attention"}:
                receipt.setdefault("funding_history", []).append(receipt.pop("funding"))
                receipt.pop("status_url", None)
                funding_status = None
            else:
                raise ValueError("previous funding intent could not be canceled")
            save_json(receipt_path, receipt)
        if not receipt.get("uploaded"):
            if response.get("billing_enabled") and response.get("uploaded") is not True:
                from .funding import funding_before_upload

                funding_status = funding_before_upload(
                    service, receipt, receipt_path, status=funding_status
                )
            from .uploads import continue_multipart

            continue_multipart(service, archive, receipt, receipt_path, response)
            receipt["uploaded"] = True
            save_json(receipt_path, receipt)
            _say("Upload stored.")
        else:
            _say(f"Upload already stored (model {receipt['model_id']}).")
        if not receipt.get("confirmed"):
            from yugos.service_client import ServiceError

            from .funding import continue_funding

            if receipt.get("funding"):
                response = continue_funding(
                    service, receipt, receipt_path, previous_status=funding_status
                )
            else:
                try:
                    _say("Confirming the submission.")
                    response = service.request(
                        "POST",
                        "/confirm",
                        {
                            "model_id": receipt["model_id"],
                            "file_hash": metadata.submission_sha256,
                            "coldkey": coldkey,
                        },
                        signed=True,
                    )
                except ServiceError as exc:
                    if exc.code != "use_billing_prepare":
                        raise
                    response = continue_funding(service, receipt, receipt_path)
            if (
                response.get("ok") is not True
                or response.get("model_id") != receipt["model_id"]
                or response.get("r2_key") != receipt["r2_key"]
                or not response.get("job_id")
            ):
                raise ValueError("service did not confirm this submission and evaluation job")
            receipt.update(confirmed=True, job_id=str(response["job_id"]))
            save_json(receipt_path, receipt)
            _say(f"Evaluation job {receipt['job_id']} is confirmed.")
        else:
            _say(f"Evaluation job {receipt['job_id']} was already confirmed.")
            link = save_status_url(receipt, receipt_path)
            if link:
                _say("Private benchmark status and refund link: " + link)
        if commit is not None and not receipt.get("committed"):
            with _Heartbeat(
                "Committing the submission digest on chain. Finalization can take a few minutes."
            ):
                result = commit(metadata.submission_sha256)
            receipt.update(committed=True, block_hash=result.block_hash)
            save_json(receipt_path, receipt)
            _say(f"Committed in block {result.block_hash}.")
        elif receipt.get("committed"):
            _say(f"Chain commitment already recorded ({receipt.get('block_hash')}).")
        return receipt


def main(argv=None) -> None:
    """Expose packaging, resumable submission, and evaluation-job status commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack", help="bundle a model directory and its local assets")
    pack.add_argument("source", type=Path)
    pack.add_argument(
        "--output",
        type=Path,
        help="ZIP destination (defaults to SOURCE/submission-<UTC timestamp>.zip)",
    )
    pack.add_argument("--name", help="submission name (defaults to the source directory name)")
    send = commands.add_parser(
        "submit", help="upload, start evaluation, and commit; may incur charges"
    )
    send.add_argument("archive", type=Path)
    send.add_argument(
        "--renew-funding",
        action="store_true",
        help="cancel an expired/unstarted funding intent and review a new quote",
    )
    send.add_argument(
        "--receipt",
        type=Path,
        help="resume file (defaults to ARCHIVE.receipt.json beside the ZIP; reused on retries)",
    )
    add_miner_endpoint_argument(send, defer_default=True)
    add_chain_arguments(send)
    status = commands.add_parser(
        "status", help="look up submission status and its private link from a ZIP or receipt"
    )
    status.add_argument("receipt", type=Path, metavar="ZIP_OR_RECEIPT")
    add_miner_endpoint_argument(status, defer_default=True)
    add_chain_arguments(status)
    cancel = commands.add_parser(
        "cancel-upload", help="abort an incomplete storage upload; retain the ZIP and receipt"
    )
    cancel.add_argument("receipt", type=Path)
    add_chain_arguments(cancel)
    from .credits import add_credit_commands, run_credit_command

    credit_options = argparse.ArgumentParser(add_help=False)
    add_miner_endpoint_argument(credit_options)
    add_credit_commands(commands, endpoint_options=credit_options)
    args = parser.parse_args(argv)
    try:
        if args.command == "credits":
            run_credit_command(args)
            return
        if args.command == "pack":
            output = args.output or args.source / (
                "submission-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + ".zip"
            )
            declaration = args.source / "models.json"
            if output.resolve() == declaration.resolve():
                raise ValueError("output must not overwrite models.json")
            try:
                models = json.loads(declaration.read_text())
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"Cannot read required model declarations from {declaration}: {exc}"
                ) from exc
            result = package(
                args.source,
                output,
                name=args.name or args.source.resolve().name,
                required_models=models,
            )
            print(
                json.dumps(
                    {"output": str(output.resolve()), **result.model_dump(mode="json")}, indent=2
                )
            )
            return
        import httpx

        resolve_subnet(args)
        if args.command == "submit":
            args.endpoint = resolve_miner_endpoint(args.endpoint, args.network)
            args.receipt = args.receipt or args.archive.with_suffix(".receipt.json")
            submission_header(
                args.archive,
                args.network,
                args.netuid,
                args.hotkey,
                wallet=args.wallet,
                endpoint=args.endpoint,
            )
        with _Heartbeat("Loading wallet and network settings."):
            chain = configured_chain(args)
        with httpx.Client(timeout=60, follow_redirects=False) as client:
            if args.command == "submit":
                service = ServiceClient(args.endpoint, client, chain.wallet.hotkey)
                result = submit(
                    args.archive,
                    args.receipt,
                    service,
                    chain,
                    network=args.network,
                    renew_funding=args.renew_funding,
                )
            else:
                initial_receipt = None
                if args.command == "status" and args.receipt.suffix.lower() == ".zip":
                    metadata = _inspect_submission(args.receipt)
                    archive = args.receipt
                    args.receipt = archive.with_suffix(".receipt.json")
                    if not args.receipt.exists():
                        initial_receipt = {
                            "schema": "yugos.submission.v1",
                            "network": args.network,
                            "netuid": args.netuid,
                            "hotkey": chain.wallet.hotkey.ss58_address,
                            "endpoint": resolve_miner_endpoint(args.endpoint, args.network),
                            "submission_sha256": metadata.submission_sha256,
                            "manifest_sha256": metadata.manifest_sha256,
                        }
                    else:
                        saved = json.loads(args.receipt.read_bytes())
                        if saved.get("submission_sha256") != metadata.submission_sha256:
                            raise ValueError("receipt belongs to a different ZIP")
                with state_lock(args.receipt):
                    receipt = (
                        json.loads(args.receipt.read_bytes())
                        if args.receipt.exists()
                        else initial_receipt
                    )
                    if not receipt:
                        raise ValueError("submission receipt is unavailable")
                    if (
                        initial_receipt
                        and receipt["submission_sha256"] != initial_receipt["submission_sha256"]
                    ):
                        raise ValueError("receipt belongs to a different ZIP")
                    if (receipt["hotkey"], receipt["network"], receipt["netuid"]) != (
                        chain.wallet.hotkey.ss58_address,
                        args.network,
                        chain.netuid,
                    ):
                        raise ValueError("receipt belongs to another wallet or network")
                    service = ServiceClient(receipt["endpoint"], client, chain.wallet.hotkey)
                    receipt["endpoint"] = service.endpoint
                    if (
                        args.command == "status"
                        and args.endpoint
                        and service.endpoint != service_url(args.endpoint)
                    ):
                        raise ValueError("receipt belongs to another service")
                    if args.command == "cancel-upload":
                        if receipt.get("uploaded") or receipt.get("confirmed"):
                            raise ValueError(
                                "upload is already complete; use the funding status link for refunds"
                            )
                        state = receipt.get("multipart")
                        if not state:
                            raise ValueError("receipt has no incomplete multipart upload")
                        result = service.request(
                            "POST",
                            "/multipart/abort",
                            {"model_id": receipt["model_id"], "upload_id": state["upload_id"]},
                            signed=True,
                        )
                        if result.get("ok") is not True:
                            raise ValueError("service did not acknowledge upload cancellation")
                        receipt.pop("multipart")
                        save_json(args.receipt, receipt)
                        print(json.dumps(result, indent=2))
                        return
                    from .funding import submission_status

                    previous_link = save_status_url(receipt, args.receipt)
                    try:
                        with _Heartbeat("Checking submission status on the service."):
                            from urllib.parse import urlencode

                            scope = urlencode({k: receipt[k] for k in ("network", "netuid")})
                            response = service.request(
                                "GET",
                                "/submissions/" + receipt["submission_sha256"] + "?" + scope,
                                {},
                                signed=True,
                            )["data"]
                            view = _submission_state(receipt, response, args.receipt) or {
                                "model_id": response["model_id"],
                                "submission_ready": response.get("uploaded") is True,
                            }
                    except Exception:
                        if previous_link:
                            _say("Private benchmark status and refund link: " + previous_link)
                        raise
                    link = save_status_url(receipt, args.receipt)
                if link:
                    _say("Private benchmark status and refund link: " + link)
                if receipt.get("job_id"):
                    result = service.request(
                        "GET", "/jobs/" + quote(receipt["job_id"], safe=""), signed=True
                    )
                else:
                    result = {
                        **(view or {}),
                        "status": submission_status(view) if view else "not_found",
                        "status_url": link,
                    }
                summary = job_progress(result)
                if summary:
                    _say(summary)
            if args.command != "submit" or not finish_submission(result, receipt_path=args.receipt):
                print(json.dumps(result, indent=2))
    except FundingWaitInterrupted as exc:
        funding_wait_stopped(exc.receipt_path, exc.status_url)
        parser.exit(130)
    except Exception as exc:
        if args.command == "submit":
            submission_error(exc)
            parser.exit(1)
        parser.exit(1, f"{exc}\n")

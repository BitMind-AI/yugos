"""Terminal presentation for miner activity and resumable upload progress."""

from __future__ import annotations

import threading
import time

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich.text import Text

_SUCCESS = "#b5bd68"
_WARNING = "#f0c674"
_ERROR = "#cc6666"
_ACCENT = "#81a2be"

# Status messages must pass through the live display's console so it can track
# their lines instead of erasing them when the spinner redraws.
_CONSOLE = Console(highlight=False, emoji=False)


def _interactive(console):
    return console.is_terminal and not console.is_dumb_terminal


def submission_header(archive, network, netuid, hotkey, *, wallet=None, endpoint=None):
    console = _CONSOLE
    console.print()
    console.print(Text("Submit model", style="bold"))
    phase("Bundle", archive.name)
    phase("Wallet", f"{wallet} / {hotkey}" if wallet else hotkey, style=_ACCENT)
    phase("Network", f"{network} · subnet {netuid}")
    if endpoint:
        phase("Service", endpoint)
    console.print()


def phase(label, message, *, style=None):
    style = {"green": _SUCCESS, "yellow": _WARNING, "cyan": _ACCENT}.get(style, style)
    _CONSOLE.print(Text.assemble((f"  {label:<12}  ", "dim"), (str(message), style)))


def _success(message):
    _CONSOLE.print(Text.assemble(("  ✓ ", _SUCCESS), str(message)))


def progress_message(message):
    """Keep receipt paths and completed phases compact in the interactive CLI."""
    console = _CONSOLE
    if not _interactive(console):
        print(message, flush=True)
        return
    from pathlib import Path

    steps = {"Submission ZIP verified.": "Bundle verified", "Upload stored.": "Upload verified"}
    if message in steps:
        _success(steps[message])
    elif message.startswith("Hotkey registered on subnet "):
        _success(message.rstrip("."))
    elif message.startswith("Resuming from receipt: "):
        phase("Receipt", Path(message.removeprefix("Resuming from receipt: ")).name + " · resuming")
    elif message.startswith("Submission receipt saved: "):
        phase("Receipt", Path(message.removeprefix("Submission receipt saved: ")).name)
    elif message.startswith("Upload already stored (model "):
        model = message.removeprefix("Upload already stored (model ").removesuffix(").")
        _success(f"Submission {model} · upload already stored")
    elif message.startswith("Upload slot reserved (model "):
        phase("Submission", message.removeprefix("Upload slot reserved (model ").removesuffix(")."))
    elif message.startswith(
        (
            "Evaluation job ",
            "Funding ready. Starting evaluation job ",
            "Committed in block ",
            "Chain commitment already recorded ",
        )
    ):
        return  # The final submission summary carries these identifiers and outcomes.
    elif message.startswith("Private benchmark status and refund link: "):
        status_link(message.removeprefix("Private benchmark status and refund link: "))
    else:
        console.print(Text("  " + message, style="dim"))


def finish_submission(result, *, receipt_path=None):
    console = _CONSOLE
    if not _interactive(console):
        return False
    console.print()
    _success("Submission accepted")
    phase("Submission", result.get("model_id", "—"))
    phase("Evaluation", result.get("job_id", "—"))
    if result.get("committed"):
        phase("On-chain", "Commitment confirmed", style=_SUCCESS)
    if receipt_path:
        phase("Receipt", receipt_path.name)
    console.print()
    return True


def funding_summary(balance, *, required=None, shortfall=None):
    """Present available funds separately from provider and execution reservations."""
    if not isinstance(balance, dict) or type(balance.get("balance_cents")) is not int:
        return
    active = balance.get("active_run_id")
    pending = balance.get("billing_held_cents", 0)
    active_held = balance.get(
        "active_held_cents", balance["balance_cents"] - pending if active else 0
    )
    available = balance.get(
        "available_cents", max(0, balance["balance_cents"] - pending - active_held)
    )

    def money(amount):
        return f"${amount / 100:,.2f}"

    _CONSOLE.print()
    _CONSOLE.print(Text("Funding", style="dim"))
    table = Table.grid(padding=(0, 3))
    table.add_column(style="dim", min_width=25)
    table.add_column(justify="right")
    table.add_row("  Available balance", money(available))
    if active_held:
        table.add_row("  Evaluation hold", money(active_held))
    if pending:
        table.add_row("  Pending provider billing", Text(money(pending), style=_WARNING))
    if required is not None:
        table.add_row("  Evaluation allowance", money(required))
    if shortfall:
        table.add_row("  Top-up needed", Text(money(shortfall), style=_WARNING))
    _CONSOLE.print(table)
    _CONSOLE.print()
    if active:
        phase("Held by run", active, style=_WARNING)


def status_link(url):
    console = _CONSOLE
    if _interactive(console):
        console.print(
            Text.assemble(
                ("  ", ""), ("Open funding & submission", f"{_ACCENT} underline link {url}")
            )
        )
    else:
        print("Private funding and submission status: " + url, flush=True)


def funding_next_step(*, reserved, shortfall, before_upload=False):
    console = _CONSOLE
    if _interactive(console):
        console.print(
            Text(
                f"  Add ${shortfall:,.2f} on the funding page "
                + ("while your ZIP uploads." if before_upload else "to start evaluation."),
                style=_WARNING,
            )
        )
    else:
        print(f"Reservation: ${reserved:.2f}; add ${shortfall:.2f}", flush=True)


def funding_wait_stopped(receipt_path, url):
    from pathlib import Path

    _CONSOLE.print()
    _CONSOLE.print(Text("  Funding wait stopped", style=_WARNING))
    phase("Receipt", Path(receipt_path).name)
    _CONSOLE.print("  Rerun the same submit command to continue.")
    _CONSOLE.print(
        "  Your submission remains active; payment confirmation can still start evaluation.",
        style="dim",
    )
    status_link(url)


def funding_wait_hint():
    if _interactive(_CONSOLE):
        _CONSOLE.print()
        _CONSOLE.print("  Ctrl-C stops waiting. Your submission stays active.", style="dim")


def submission_error(error):
    """Explain actionable service failures without exposing transport response dumps."""
    import httpx

    from yugos.service_client import ServiceError

    title = "Submission needs attention"
    message = str(error)
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        title = "Network request timed out"
        message = (
            "No response arrived before the network timeout. The request may still have completed.\n"
            "Keep the same ZIP and receipt, and rerun the same submit command to resume. "
            "If this repeats, report the last progress message shown above."
        )
    elif isinstance(error, ServiceError):
        if error.code == "submission_limit_reached":
            title = "Submission limit reached"
            limit = error.detail.get("submissions_max")
            quota = (
                f"the limit of {limit} new ZIPs per hotkey"
                if type(limit) is int and limit > 0
                else "your hotkey's limit for new ZIPs"
            )
            message = (
                f"You have reached {quota} in a rolling 24-hour window.\n"
                "Wait for an earlier submission to leave that window, then retry. "
                "You can still resume the same ZIP."
            )
        elif error.code == "credits_locked":
            title = "Funding is held"
            message = (
                "An earlier evaluation still holds funds for this hotkey.\n"
                "Your uploaded ZIP is retained. Billing recovery must release the unused allowance; "
                "keep the ZIP and receipt to continue."
            )
        elif error.code == "insufficient_credits":
            title = "More funds needed"
            message = "The available balance does not cover this evaluation. Open the funding status link to add funds."
        elif error.code == "request_signature_expired":
            title = "Authentication timed out"
            message = (
                "The service rejected the request signature as expired after retrying. "
                "Keep the receipt and rerun the same command to resume. "
                "If this repeats, check your device clock."
            )
        elif error.code:
            message = f"The service could not continue: {error.code.replace('_', ' ')}. Keep the receipt to retry."
        else:
            operation = {
                "/billing/register": "Opening your submission page",
                "/billing/prepare": "Preparing evaluation funding",
                "/confirm": "Confirming your submission",
                "/multipart/complete": "Finalizing your upload",
            }.get(error.path, "The submission request")
            message = f"{operation} failed (HTTP {error.status}). Keep the receipt to retry."
    console = Console(stderr=True, highlight=False, emoji=False)
    console.print()
    console.print(Text.assemble(("  Error: ", f"bold {_ERROR}"), title))
    console.print(Text("  " + message))
    console.print()


class Activity:
    """Animate terminal activity, or emit periodic plain-text status in logs."""

    def __init__(self, message, *, interval=30, console=None):
        self.message, self.interval = message, interval
        self.console = console or _CONSOLE
        self._stop = threading.Event()
        self._thread = None
        self._status = None

    @property
    def interactive(self):
        return _interactive(self.console)

    def update(self, message):
        self.message = message
        if self._status is not None:
            self._status.update(Text("  " + message, style="dim"))

    def __enter__(self):
        if _interactive(self.console):
            self._status = self.console.status(
                Text("  " + self.message, style="dim"),
                spinner="dots",
                spinner_style="dim",
                refresh_per_second=4,
            )
            self._status.start()
        else:
            self.console.print(self.message, markup=False, highlight=False)
            started = time.monotonic()

            def beat():
                while not self._stop.wait(self.interval):
                    elapsed = int(time.monotonic() - started)
                    self.console.print(
                        f"{self.message} Still working ({elapsed}s).",
                        markup=False,
                        highlight=False,
                    )

            self._thread = threading.Thread(target=beat, name="yugos-progress", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, *_exc):
        self._stop.set()
        if self._status is not None:
            self._status.stop()
            if exc_type is not None:
                self.console.print(self.message, markup=False, highlight=False)
        if self._thread is not None:
            self._thread.join(timeout=1)
        return False


class UploadProgress:
    """Track streamed bytes separately from durable part acknowledgements."""

    def __init__(self, name, size, parts, *, stored=0, retained=0, console=None, clock=None):
        self.console = console or _CONSOLE
        self.interactive = _interactive(self.console)
        self.size, self.parts, self.stored = size, parts, stored
        self.acknowledged = retained
        self._last_bucket = int(stored * 10 / size)
        self.progress = Progress(
            TextColumn("  {task.description}", style="dim"),
            BarColumn(bar_width=None, complete_style=_ACCENT, finished_style=_SUCCESS),
            TaskProgressColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TextColumn("ETA", style="dim"),
            TimeRemainingColumn(),
            console=self.console,
            disable=not self.interactive,
            expand=True,
            refresh_per_second=4,
            get_time=clock,
        )
        self.task = self.progress.add_task("Starting", total=size, completed=stored)
        self.name = name

    def __enter__(self):
        self.console.print()
        self.console.print(Text("Upload", style="dim"))
        if self.acknowledged:
            self.console.print(
                f"  Resuming {self.acknowledged}/{self.parts} stored parts.", style="dim"
            )
        self.progress.start()
        return self

    def begin(self, number, attempt):
        label = f"Part {number}/{self.parts}"
        if attempt:
            label = f"Retry {number}/{self.parts}"
            self.progress.reset(self.task, total=self.size, completed=self.stored)
            if not self.interactive:
                self.console.print(f"Retrying upload part {number}/{self.parts}.")
        self.progress.update(self.task, description=label, completed=self.stored)

    def advance(self, amount):
        self.progress.advance(self.task, amount)

    def confirm(self, size):
        self.stored += size
        self.acknowledged += 1
        self.progress.update(self.task, completed=self.stored)
        bucket = int(self.stored * 10 / self.size)
        if not self.interactive and bucket > self._last_bucket:
            self.console.print(
                f"Upload {self.stored / self.size:.0%} · "
                f"{self.acknowledged}/{self.parts} parts stored."
            )
            self._last_bucket = bucket

    def __exit__(self, exc_type, *_exc):
        if exc_type is not None:
            self.progress.update(self.task, completed=self.stored, description="Interrupted")
        else:
            self.progress.update(self.task, description="Sent")
        self.progress.stop()
        return False

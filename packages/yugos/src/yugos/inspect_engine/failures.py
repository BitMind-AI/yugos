"""The one question-failure boundary between Inspect samples and a host's model callbacks.

Every router and service-role callback runs inside `QuestionFailures.boundary`.
Shared state decides whether a failure stops the run: once a call budget is
stopped or the submission session has exited, the host stops new admissions,
whatever was raised. Any other error belongs to its question. Yugos's own call
and session errors map to their outcomes here, a host maps only the errors it
alone knows, and anything else leaves the question incomplete (`QuestionError`):
never retried, never graded, and left for automatic or operator resolution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Protocol

from inspect_ai.util import LimitExceededError

from yugos.inference.calls import CallLimits, MeteredCalls, QuestionLimitExceeded
from yugos.submission.protocol import Failure
from yugos.submission.session import RequestOutcome, RequestState, SessionError, SessionExit

from .router_model import QuestionError, RouterCall, RouterError, RouterUnavailable, RunError

# The reason a sample records when its own limits refuse the next turn.
QUESTION_EXHAUSTED = "question budget exhausted before dispatch"

# Outcomes already decided by a callback or a host.
_DECIDED = (QuestionError, RouterError, RouterUnavailable, LimitExceededError)


def question_exhausted(exc: QuestionLimitExceeded, limits: CallLimits) -> LimitExceededError:
    """End the sample at its call, spend or time limit, so it is scored on what it did.

    Each limit stops the next turn before dispatch. An agent that cannot finish
    within its question's budget fails the task; it is not an unmeasured question.
    """
    match exc.limit:
        case "time":
            kind, limit = "time", limits.question_timeout_s
        case "calls":
            kind, limit = "turn", limits.max_calls_per_question
        case "cost":
            kind, limit = "cost", limits.max_question_cost_usd
    return LimitExceededError(
        kind, value=limit or 0, limit=limit or 0, message=f"{QUESTION_EXHAUSTED}: {exc}"
    )


class ExitingSession(Protocol):
    """What the boundary reads from a host's routing session."""

    @property
    def session_exit(self) -> SessionExit | None:
        """Set once a worker closes; sessions with no worker to lose keep it None."""


class QuestionFailures(ABC):
    """Base for a host's router: classify each callback failure in one place.

    Hosts set the attributes below and implement `stop_run`. They may list
    `fatal_errors`, map their own known errors in `known_failure`, and settle
    question state in `question_ended`.
    """

    meter: MeteredCalls
    service_meter: MeteredCalls | None
    # None until a session is loaded, and for routers without one.
    session: ExitingSession | None
    limits: CallLimits
    fatal_errors: tuple[type[Exception], ...] = ()

    @abstractmethod
    def stop_run(self, call: RouterCall, exc: Exception) -> None:
        """Stop new admissions; already admitted calls keep their evidence."""

    def known_failure(
        self, call: RouterCall, exc: Exception, role: str | None
    ) -> BaseException | None:
        """The question outcome for an error only this host knows, or None."""
        return None

    def question_ended(self, call: RouterCall, outcome: BaseException) -> None:
        """Record that this question's callback ended with `outcome`."""

    def run_stopped(self) -> bool:
        budgets = (m.budget for m in (self.meter, self.service_meter) if m is not None)
        return (
            any(budget.stopped.is_set() for budget in budgets) or self._session_exit() is not None
        )

    @contextmanager
    def boundary(self, call: RouterCall, *, role: str | None = None):
        """Contain one callback; `role` names a service role, None the router itself.

        Cancellation and process exits pass through untouched.
        """
        try:
            yield
        except Exception as exc:
            fatal = self.run_stopped() or isinstance(exc, (RunError, *self.fatal_errors))
            if self._submission_failed(exc, role):
                # The request the submission failed is its own, even when the fault
                # also closed the shared worker.
                outcome = RouterError(str(exc))
                self.question_ended(call, outcome)
                if fatal:
                    self.stop_run(call, exc)
                raise outcome from exc
            if fatal:
                self.stop_run(call, self._stop_cause(exc))
                raise
            outcome = self._outcome(call, exc, role)
            if not isinstance(outcome, RouterUnavailable):  # Inspect retries it
                self.question_ended(call, outcome)
            if outcome is exc:
                raise
            raise outcome from exc

    def _session_exit(self):
        return self.session.session_exit if self.session is not None else None

    @staticmethod
    def _submission_failed(exc: Exception, role: str | None) -> bool:
        """Whether the submission failed this router request, not a later bystander."""
        return (
            role is None
            and isinstance(exc, SessionError)
            and exc.owner == "submission"
            and (exc.outcome is None or exc.outcome.state == RequestState.FAILED)
        )

    def _stop_cause(self, exc: Exception) -> Exception:
        """Stop for the session's recorded exit, whichever callback noticed it first.

        Concurrent questions can observe a closed worker before the request that
        closed it reports; the exit's own failure and owner explain the stop.
        """
        exit = self._session_exit()
        if exit is None or (isinstance(exc, SessionError) and exc.outcome is not None):
            return exc
        failure = exit.failure or Failure(
            code="preparation.internal", detail="worker session closed"
        )
        outcome = RequestOutcome(
            failure.request_id or "", RequestState.INTERRUPTED, session_exit=exit
        )
        return SessionError(failure, outcome=outcome)

    def _outcome(self, call, exc, role) -> BaseException:
        if isinstance(exc, _DECIDED):
            return exc
        if (known := self.known_failure(call, exc, role)) is not None:
            return known
        if isinstance(exc, QuestionLimitExceeded):
            return question_exhausted(exc, self.limits)
        return QuestionError(f"{type(exc).__name__}: {exc}")

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

from contextlib import contextmanager

from inspect_ai.util import LimitExceededError

from yugos.inference.calls import QuestionLimitExceeded
from yugos.submission.session import RequestState, SessionError

from .router_model import QuestionError, RouterCall, RouterError, RouterUnavailable, RunError

# The reason a sample records when its own limits refuse the next turn.
QUESTION_EXHAUSTED = "question budget exhausted before dispatch"

# Outcomes already decided by a callback or a host.
_DECIDED = (QuestionError, RouterError, RouterUnavailable, LimitExceededError)


def question_exhausted(exc: QuestionLimitExceeded, limits) -> LimitExceededError:
    """End the sample at its call, spend or time limit, so it is scored on what it did.

    Each limit stops the next turn before dispatch. An agent that cannot finish
    within its question's budget fails the task; it is not an unmeasured question.
    """
    reason = str(exc)
    if "time" in reason:
        kind, limit = "time", limits.question_timeout_s
    elif "call" in reason:
        kind, limit = "turn", limits.max_calls_per_question
    else:
        kind, limit = "cost", limits.max_question_cost_usd
    return LimitExceededError(
        kind, value=limit or 0, limit=limit or 0, message=f"{QUESTION_EXHAUSTED}: {reason}"
    )


class QuestionFailures:
    """Mixin for a host's router: classify each callback failure in one place.

    Hosts provide `meter`, `service_meter` (or None), `session` and
    `limits`, and implement `stop_run`. They may list `fatal_errors`, map their
    own known errors in `known_failure`, and settle question state in
    `question_ended`.
    """

    fatal_errors: tuple[type[Exception], ...] = ()

    def stop_run(self, call: RouterCall, exc: Exception) -> None:
        """Stop new admissions; already admitted calls keep their evidence."""
        raise NotImplementedError

    def known_failure(
        self, call: RouterCall, exc: Exception, role: str | None
    ) -> BaseException | None:
        """The question outcome for an error only this host knows, or None."""
        return None

    def question_ended(self, call: RouterCall, outcome: BaseException) -> None:
        """Record that this question's callback ended with `outcome`."""

    def run_stopped(self) -> bool:
        budgets = (m.budget for m in (self.meter, self.service_meter) if m is not None)
        # Sessions without a worker to lose (e.g. operator panels) never exit.
        return any(budget.stopped.is_set() for budget in budgets) or (
            getattr(self.session, "session_exit", None) is not None
        )

    @contextmanager
    def boundary(self, call: RouterCall, *, role: str | None = None):
        """Contain one callback; `role` names a service role, None the router itself.

        Cancellation and process exits pass through untouched.
        """
        try:
            yield
        except Exception as exc:
            # The request the submission failed is its own, even when the fault
            # also closed the shared worker.
            submission_failed = (
                role is None
                and isinstance(exc, SessionError)
                and exc.owner == "submission"
                and (exc.outcome is None or exc.outcome.state == RequestState.FAILED)
            )
            fatal = self.run_stopped() or isinstance(exc, (RunError, *self.fatal_errors))
            if fatal and not submission_failed:
                self.stop_run(call, exc)
                raise
            outcome = self._outcome(call, exc, role, submission_failed)
            if not isinstance(outcome, RouterUnavailable):
                self.question_ended(call, outcome)
            if fatal:
                self.stop_run(call, exc)
            if outcome is exc:
                raise
            raise outcome from exc

    def _outcome(self, call, exc, role, submission_failed) -> BaseException:
        if submission_failed:
            return RouterError(str(exc))
        if isinstance(exc, _DECIDED):
            return exc
        if (known := self.known_failure(call, exc, role)) is not None:
            return known
        if isinstance(exc, QuestionLimitExceeded):
            return question_exhausted(exc, self.limits)
        return QuestionError(f"{type(exc).__name__}: {exc}")

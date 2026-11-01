"""Evaluate submissions with UK AISI Inspect as the engine (install `yugos[inspect]`).

Inspect runs tasks, agents, sandboxes and scorers; a submission's router is an
Inspect model. Yugos keeps competition rules, frozen task contracts, verification and
answer-key isolation; metering and billing stay with the execution service.
Core Yugos never imports this package.
"""

from .versions import require_pinned_inspect

require_pinned_inspect()

from .digest import canonical_receipt, receipt_digest  # noqa: E402
from .fairness import FairnessProfile, competition_eval_options  # noqa: E402
from .frozen import TaskConfig, evaluation_identity, freeze_task, restore_tasks  # noqa: E402
from .isolation import (  # noqa: E402
    RolePin,
    RolePinMismatch,
    UnboundGraderRole,
    bind_grader_roles,
    read_role_pins,
    redacted_log,
    require_grader_roles,
    require_role_pins,
    role_pins_metadata,
    served_model,
    task_grader_roles,
)
from .measure import (  # noqa: E402
    SampleMeasurement,
    active_seconds,
    category_measurements_from_log,
    category_measurements_from_samples,
    measure_log,
)
from .receipts import (  # noqa: E402
    JsonlReceipts,
    MemoryReceipts,
    ProviderReceipt,
    ReceiptSink,
)
from .roles import TaskRole, resolve_task_roles  # noqa: E402
from .router_model import (  # noqa: E402
    QuestionError,
    RouterCall,
    RouterClient,
    RouterError,
    RouterReply,
    RouterToolCall,
    RouterUnavailable,
    RunError,
    model_working_time,
    question_failure,
    record_model_time,
    router_model,
)
from .run import run_competition_eval  # noqa: E402
from .versions import SampleScope  # noqa: E402

__all__ = [
    "FairnessProfile",
    "TaskConfig",
    "TaskRole",
    "resolve_task_roles",
    "evaluation_identity",
    "freeze_task",
    "restore_tasks",
    "JsonlReceipts",
    "MemoryReceipts",
    "ProviderReceipt",
    "ReceiptSink",
    "RolePin",
    "RolePinMismatch",
    "RouterCall",
    "QuestionError",
    "question_failure",
    "RouterClient",
    "RouterError",
    "RouterReply",
    "RouterToolCall",
    "RouterUnavailable",
    "RunError",
    "SampleMeasurement",
    "SampleScope",
    "UnboundGraderRole",
    "active_seconds",
    "bind_grader_roles",
    "canonical_receipt",
    "category_measurements_from_log",
    "category_measurements_from_samples",
    "competition_eval_options",
    "measure_log",
    "read_role_pins",
    "receipt_digest",
    "redacted_log",
    "require_grader_roles",
    "require_role_pins",
    "role_pins_metadata",
    "model_working_time",
    "record_model_time",
    "router_model",
    "run_competition_eval",
    "served_model",
    "task_grader_roles",
]

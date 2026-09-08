from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from self_cognition.bootstrap import build_container
from self_cognition.core.actions import (
    ActionDecision,
    ActionDecisionPayload,
    ActionDecisionStatus,
    ActionRequest,
)
from self_cognition.core.contributions import (
    CognitiveContribution,
    CognitionType,
)
from self_cognition.core.events import EventEnvelope, EventSource
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.runs import (
    RunCheckpoint,
    RunKind,
    RunRecord,
    RunStatus,
    RunUsage,
    RunBudget,
    run_from_dict,
    run_to_dict,
)
from self_cognition.core.scopes import DataScope, DisclosureScope, SubjectScope
from self_cognition.infrastructure.persistence.file_run_repository import (
    FileRunRepository,
)
from self_cognition.infrastructure.persistence.in_memory_event_store import (
    InMemoryEventStore,
)
from self_cognition.infrastructure.persistence.in_memory_run_repository import (
    InMemoryRunRepository,
)
from self_cognition.runtime.cancellation import CancellationToken
from self_cognition.runtime.recovery import RunRecoveryService
from self_cognition.runtime.run_context import RunContext
from self_cognition.runtime.run_service import RunLifecycle
from self_cognition.tools.executor import ToolExecutor


NOW = datetime(2026, 9, 9, 10, tzinfo=timezone.utc)
MIND = SubjectScope.for_mind("mind-24")


@dataclass
class FixedClock:
    value: datetime = NOW

    def now(self) -> datetime:
        return self.value


def context(*, run_id: UUID | None = None) -> RunContext:
    return RunContext(
        run_id or uuid4(),
        uuid4(),
        NOW + timedelta(minutes=5),
        clock=FixedClock(),
    )


def record(run_id: UUID, *, status: RunStatus = RunStatus.RUNNING) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        kind=RunKind.ACTION,
        subject=MIND,
        correlation_id=uuid4(),
        started_at=NOW,
        updated_at=NOW,
        deadline=NOW + timedelta(minutes=5),
        status=status,
        budget=RunBudget(max_model_calls=2, max_tool_calls=1),
        usage=RunUsage(model_calls=1, tool_calls=1),
        checkpoint=RunCheckpoint(
            uuid4(),
            run_id,
            "action.execute_started",
            NOW,
            action_id=uuid4(),
        ),
    )


def test_run_record_round_trips_through_file_repository(tmp_path: Path) -> None:
    original = record(uuid4())
    repository = FileRunRepository(tmp_path / "runs")

    repository.save(original)

    assert FileRunRepository(tmp_path / "runs").get(original.run_id) == original
    assert run_from_dict(run_to_dict(original)) == original


def test_parent_cancellation_propagates_and_persists() -> None:
    parent = context()
    child = parent.child(uuid4())
    repository = InMemoryRunRepository()
    lifecycle = RunLifecycle(repository)
    lifecycle.begin(parent, RunKind.COGNITIVE_CYCLE, MIND)

    lifecycle.request_cancel(parent.run_id, "user requested stop")

    assert parent.is_cancelled
    assert child.is_cancelled
    assert repository.get(parent.run_id).cancel_requested is True


def test_incomplete_action_with_possible_side_effect_is_terminated() -> None:
    repository = InMemoryRunRepository()
    events = InMemoryEventStore()
    original = record(uuid4())
    repository.save(original)

    recovered = RunRecoveryService(repository, events).recover(NOW)

    assert recovered[0].status is RunStatus.INTERRUPTED
    assert recovered[0].termination_reason == (
        "action result missing after possible side effect"
    )
    assert repository.read_incomplete() == ()

    cycle_id = uuid4()
    cycle = replace(
        original,
        run_id=cycle_id,
        kind=RunKind.COGNITIVE_CYCLE,
        checkpoint=RunCheckpoint(
            uuid4(), cycle_id, "state_reduced", NOW
        ),
    )
    repository.save(cycle)
    assert RunRecoveryService(repository, events).recover(NOW)[0].status is RunStatus.COMPLETED


def test_cognitive_cycle_is_recorded_and_completed() -> None:
    from self_cognition.application.process_event import ProcessEventService
    from self_cognition.blackboard.reducer import StateReducer
    from self_cognition.blackboard.service import CognitiveSpaceService
    from self_cognition.core.protocols import CognitiveModule
    from self_cognition.infrastructure.persistence.in_memory_evidence_repository import (
        InMemoryEvidenceRepository,
    )
    from self_cognition.infrastructure.persistence.in_memory_state_repository import (
        InMemoryStateRepository,
    )
    from self_cognition.runtime.engine import CognitionEngine

    class Module:
        subscriptions = frozenset({"user.message"})
        module_id = "test.stage24"
        module_version = "1"
        deterministic = True

        def run(self, request):
            return (
                CognitiveContribution.set_from_event(
                    request.event,
                    contribution_id=uuid4(),
                    target_field="preferences.stage24",
                    cognition_type=CognitionType.PREFERENCE,
                    value="checkpointed",
                    confidence=1.0,
                    evidence_refs=(EvidenceRef.for_event(request.event),),
                    source_module=self.module_id,
                    module_version=self.module_version,
                ),
            )

    repository = InMemoryRunRepository()
    lifecycle = RunLifecycle(repository)
    service = ProcessEventService(
        InMemoryEventStore(),
        InMemoryEvidenceRepository(),
        InMemoryStateRepository(),
        CognitionEngine((Module(),), CognitiveSpaceService(StateReducer())),
        run_lifecycle=lifecycle,
    )

    result = service.process(EventEnvelope.user_message("mind-24", "hello"), context())
    stored = repository.get(result.run_id)

    assert result.status.value == "succeeded"
    assert stored is not None
    assert stored.kind is RunKind.COGNITIVE_CYCLE
    assert stored.status is RunStatus.COMPLETED
    assert stored.checkpoint.position == "state_reduced"


class CrashExecutor(ToolExecutor):
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, action, decision, context):
        self.calls += 1
        raise RuntimeError("crash after external effect")


def test_action_crash_leaves_checkpoint_and_recovery_does_not_retry(
    tmp_path: Path,
) -> None:
    executor = CrashExecutor()
    app = build_container(tmp_path / "data", tool_executor=executor)
    run = context()
    action = ActionRequest(
        uuid4(),
        uuid4(),
        MIND,
        uuid4(),
        1,
        "read-file",
        "file.read",
        {"path": "notes.txt"},
        (),
        "stage24-action",
        NOW,
    )
    decision = ActionDecision(
        uuid4(),
        action.action_id,
        ActionDecisionStatus.ALLOWED,
        "run the requested action",
        ("active goal",),
        "stage24 test",
        ("bounded test execution",),
        (),
        NOW,
        NOW + timedelta(minutes=5),
        action.action_id,
    )
    app.event_store.append(
        EventEnvelope(
            uuid4(),
            "action.decided",
            None,
            MIND,
            ActionDecisionPayload(action, decision),
            NOW,
            NOW,
            EventSource.SYSTEM,
            DataScope(MIND, DisclosureScope.MIND),
            causation_id=action.proposal_request_id,
        )
    )

    with pytest.raises(RuntimeError, match="crash"):
        app.action.execute(action, decision, run)

    recovered = app.run_recovery.recover(NOW)
    retry = app.action.execute(action, decision, run)

    assert recovered[0].status is RunStatus.INTERRUPTED
    assert retry.error_type == "InterruptedAction"
    assert executor.calls == 1

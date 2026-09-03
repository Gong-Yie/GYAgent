from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from self_cognition.application.process_event import ProcessEventService
from self_cognition.application.results import ProcessEventStatus
from self_cognition.blackboard.reducer import StateReducer
from self_cognition.blackboard.service import CognitiveSpaceService
from self_cognition.cognition.identity.identity_value_extractor import (
    IdentityValueExtractor,
)
from self_cognition.cognition.identity.self_model import SelfModelCognitionModule
from self_cognition.core.contributions import CognitionType
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.evidence import EvidenceSourceKind
from self_cognition.core.events import (
    EventEnvelope,
    EventSource,
    SelfModelObservationPayload,
)
from self_cognition.core.identity import (
    CapabilityExecutionStatus,
    CapabilityKind,
    CapabilityPermission,
    GoalPriority,
    GoalRecord,
    GoalStatus,
    LimitationRecord,
    SelfModel,
    SelfModelAspect,
)
from self_cognition.core.scopes import (
    MindScope,
    SubjectKind,
    SubjectRef,
    SubjectScope,
)
from self_cognition.core.state import StateDecisionStatus, SubjectState
from self_cognition.core.workspace import WorkspaceBuilder
from self_cognition.executive.dialogue.rule_based import RuleBasedDialogueModel
from self_cognition.infrastructure.persistence.in_memory_event_store import (
    InMemoryEventStore,
)
from self_cognition.infrastructure.persistence.in_memory_evidence_repository import (
    InMemoryEvidenceRepository,
)
from self_cognition.infrastructure.persistence.in_memory_state_repository import (
    InMemoryStateRepository,
)
from self_cognition.infrastructure.persistence.serialization import (
    event_from_json,
    event_to_json,
    state_from_json,
    state_to_json,
)
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.runtime.run_context import RunContext
from self_cognition.tools.registry import (
    CapabilityRegistration,
    CapabilityRegistry,
)


NOW = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
MIND = SubjectScope.for_mind("mind-1")
USER = SubjectScope(
    MindScope("mind-1"),
    SubjectRef(SubjectKind.USER, "user-1"),
)


@dataclass(frozen=True, slots=True)
class FixedClock:
    value: datetime = NOW

    def now(self) -> datetime:
        return self.value


def make_engine() -> CognitionEngine:
    return CognitionEngine(
        (IdentityValueExtractor(), SelfModelCognitionModule()),
        CognitiveSpaceService(StateReducer()),
    )


def make_context(identifier: int) -> RunContext:
    return RunContext(
        run_id=UUID(int=identifier),
        correlation_id=UUID(int=identifier + 100),
        deadline=NOW + timedelta(minutes=1),
        clock=FixedClock(),
    )


def self_observation(
    event_id: int,
    aspect: SelfModelAspect,
    field_id: str,
    value: str | LimitationRecord | GoalRecord,
    *,
    actor: SubjectScope | None = USER,
    confirmed: bool = False,
    confidence: float = 1.0,
    expires_at: datetime | None = None,
) -> EventEnvelope:
    return EventEnvelope.self_model_observation(
        MIND,
        SelfModelObservationPayload(
            aspect=aspect,
            field_id=field_id,
            value=value,
            confidence=confidence,
            explicitly_confirmed=confirmed,
            expires_at=expires_at,
        ),
        actor=actor,
        event_id=UUID(int=event_id),
        clock=FixedClock(),
    )


def test_process_service_keeps_user_identity_separate_from_agent_self_model() -> None:
    states = InMemoryStateRepository()
    service = ProcessEventService(
        event_store=InMemoryEventStore(),
        evidence_repository=InMemoryEvidenceRepository(),
        state_repository=states,
        engine=make_engine(),
    )

    user_result = service.process(
        EventEnvelope.user_message(
            USER,
            "我的角色是研究员",
            event_id=UUID(int=1),
            clock=FixedClock(),
        ),
        make_context(1),
    )
    mind_result = service.process(
        self_observation(
            2,
            SelfModelAspect.IDENTITY,
            "role",
            "认知助手",
            confirmed=True,
        ),
        make_context(2),
    )
    value_result = service.process(
        self_observation(
            3,
            SelfModelAspect.VALUE,
            "principle",
            "可靠",
            confirmed=True,
        ),
        make_context(3),
    )

    assert user_result.status is ProcessEventStatus.SUCCEEDED
    assert mind_result.status is ProcessEventStatus.SUCCEEDED
    assert value_result.status is ProcessEventStatus.SUCCEEDED
    assert states.load(USER).get("identity.role").value == "研究员"
    assert states.load(MIND).get("identity.role").value == "认知助手"
    assert states.load(MIND).get("values.principle").cognition_type is (
        CognitionType.PREFERENCE
    )


def test_mind_identity_requires_confirmation_or_system_prior() -> None:
    engine = make_engine()
    state = SubjectState.empty(
        MIND.subject.subject_id,
        mind_id=MIND.mind.mind_id,
        subject_kind=SubjectKind.MIND,
    )

    state = engine.process(
        self_observation(
            10,
            SelfModelAspect.IDENTITY,
            "role",
            "未确认角色",
        ),
        state,
    )
    assert "identity.role" not in state.entries
    assert state.changes[-1].status is StateDecisionStatus.PENDING

    state = engine.process(
        self_observation(
            11,
            SelfModelAspect.IDENTITY,
            "role",
            "认知助手",
            actor=None,
        ),
        state,
    )
    assert state.get("identity.role").value == "认知助手"
    assert state.get("identity.role").evidence_refs[-1].source_kind is (
        EvidenceSourceKind.SYSTEM_PRIOR
    )

    state = engine.process(
        self_observation(
            12,
            SelfModelAspect.IDENTITY,
            "role",
            "普通助手",
        ),
        state,
    )
    assert state.get("identity.role").value == "认知助手"
    assert state.changes[-1].status is StateDecisionStatus.PENDING

    state = engine.process(
        self_observation(
            13,
            SelfModelAspect.IDENTITY,
            "role",
            "普通助手",
            confirmed=True,
        ),
        state,
    )
    assert state.get("identity.role").value == "普通助手"
    assert len(state.get("identity.role").evidence_refs) == 2
    workspace = WorkspaceBuilder().build("你是谁？", state, as_of=NOW)
    answer = RuleBasedDialogueModel().respond("你是谁？", workspace)
    assert answer.text == "我的自我认知包括：普通助手。"
    assert answer.evidence_refs


def test_capability_comes_from_registry_and_tool_result_evidence() -> None:
    registry = CapabilityRegistry(
        (
            CapabilityRegistration(
                capability_id="filesystem.read",
                name="读取工作区文件",
                kind=CapabilityKind.TOOL,
                permission=CapabilityPermission.GRANTED,
            ),
        )
    )
    engine = make_engine()
    state = SubjectState.empty(
        "mind-1",
        mind_id="mind-1",
        subject_kind=SubjectKind.MIND,
    )

    state = engine.process(
        registry.registration_event(
            "filesystem.read",
            MIND,
            event_id=UUID(int=20),
            clock=FixedClock(),
        ),
        state,
    )
    registered = SelfModel.from_state(state, as_of=NOW).capabilities[0]
    assert registered.available is True
    assert registered.verified is False

    state = engine.process(
        registry.record_execution(
            "filesystem.read",
            MIND,
            succeeded=True,
            event_id=UUID(int=21),
            clock=FixedClock(),
        ),
        state,
    )
    verified = SelfModel.from_state(state, as_of=NOW).capabilities[0]
    assert verified.execution_status is CapabilityExecutionStatus.SUCCEEDED
    assert verified.verified is True
    assert state.get("capabilities.filesystem.read").evidence_refs[-1].source_kind is (
        EvidenceSourceKind.TOOL_RESULT
    )
    workspace = WorkspaceBuilder().build("你能做什么？", state, as_of=NOW)
    answer = RuleBasedDialogueModel().respond("你能做什么？", workspace)
    assert answer.text == "我当前可用的能力包括：读取工作区文件（已有成功执行证据）。"
    assert answer.evidence_refs[-1].source_kind is EvidenceSourceKind.TOOL_RESULT
    with pytest.raises(ContractValidationError, match="system or tool"):
        EventEnvelope.capability_observed(
            MIND,
            verified,
            source=EventSource.MODEL,
        )


def test_self_model_exposes_structured_limitations_goals_and_answers() -> None:
    limitation = LimitationRecord(
        limitation_id="network",
        description="不能访问外部网络",
        reason="当前运行环境未授予网络权限",
        applies_to="external_network",
        recovery_condition="运行环境授予网络权限并产生成功证据",
    )
    goal = GoalRecord(
        goal_id="stage-16",
        description="完成阶段16自我模型",
        source="user-1",
        priority=GoalPriority.HIGH,
        status=GoalStatus.ACTIVE,
        completion_conditions=("聚焦测试通过", "主体隔离成立"),
    )
    engine = make_engine()
    state = SubjectState.empty(
        "mind-1",
        mind_id="mind-1",
        subject_kind=SubjectKind.MIND,
    )
    for event in (
        self_observation(
            30,
            SelfModelAspect.LIMITATION,
            limitation.limitation_id,
            limitation,
            expires_at=NOW + timedelta(days=1),
        ),
        self_observation(
            31,
            SelfModelAspect.GOAL,
            goal.goal_id,
            goal,
        ),
    ):
        state = engine.process(event, state)

    model = SelfModel.from_state(state, as_of=NOW)
    assert model.active_limitations == (limitation,)
    assert model.active_goals == (goal,)
    assert SelfModel.from_state(
        state,
        as_of=NOW + timedelta(days=2),
    ).active_limitations == ()
    assert state.get("limitations.network").cognition_type is CognitionType.LIMITATION
    assert state.get("goals.stage-16").cognition_type is CognitionType.GOAL

    workspace = WorkspaceBuilder().build("你当前的目标是什么？", state, as_of=NOW)
    answer = RuleBasedDialogueModel().respond("你当前的目标是什么？", workspace)
    assert answer.text == "我当前的目标是：完成阶段16自我模型。"
    assert answer.evidence_refs[0].evidence_id == UUID(int=31)
    limitation_workspace = WorkspaceBuilder().build(
        "你不能做什么？",
        state,
        as_of=NOW,
    )
    limitation_answer = RuleBasedDialogueModel().respond(
        "你不能做什么？",
        limitation_workspace,
    )
    assert "不能访问外部网络" in limitation_answer.text
    assert limitation_answer.evidence_refs[0].evidence_id == UUID(int=30)


def test_self_model_events_and_state_round_trip_without_schema_change() -> None:
    event = self_observation(
        40,
        SelfModelAspect.GOAL,
        "stage-16",
        GoalRecord(
            goal_id="stage-16",
            description="完成阶段16",
            source="user-1",
            priority=GoalPriority.NORMAL,
            status=GoalStatus.ACTIVE,
            completion_conditions=("测试通过",),
        ),
    )
    restored_event = event_from_json(event_to_json(event))
    state = make_engine().process(
        restored_event,
        SubjectState.empty(
            "mind-1",
            mind_id="mind-1",
            subject_kind=SubjectKind.MIND,
        ),
    )

    assert restored_event == event
    assert state_from_json(state_to_json(state)) == state
    assert SelfModel.from_state(state, as_of=NOW).goals[0].goal_id == "stage-16"

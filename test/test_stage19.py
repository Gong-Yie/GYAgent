import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from self_cognition.application.results import ProcessEventStatus
from self_cognition.blackboard.reducer import StateReducer
from self_cognition.blackboard.service import CognitiveSpaceService
from self_cognition.bootstrap import ApplicationContainer, build_container
from self_cognition.cognition.affect.affect_extractor import AffectExtractor
from self_cognition.cognition.metacognition.conflict_extractor import (
    ConflictMetacognitionExtractor,
)
from self_cognition.cognition.metacognition.correction import UserCorrectionModule
from self_cognition.core.affect import AffectAssessment, decay_assessment
from self_cognition.core.cognition import (
    CognitionFailureType,
    CognitionRequest,
    CognitionResultStatus,
)
from self_cognition.core.contributions import (
    CognitiveContribution,
    CognitionType,
    ContributionOperation,
)
from self_cognition.core.deletions import DeletionPlan, DeletionSelector, DeletionStatus
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import (
    AssessmentRequestPayload,
    ConflictReviewPayload,
    EventEnvelope,
)
from self_cognition.core.memories import MemoryLifecycleStatus, MemoryRecord, MemoryType
from self_cognition.core.metacognition import (
    ConflictReview,
    ConflictStatus,
    EvidenceBasis,
    FailureCause,
    KnowledgeStatus,
    MetacognitiveAssessment,
    SuggestedAction,
)
from self_cognition.core.model_outputs import (
    ContributionCandidate,
    ModelExtractionResult,
)
from self_cognition.core.scopes import (
    DataScope,
    DisclosureScope,
    MindScope,
    SubjectKind,
    SubjectRef,
    SubjectScope,
)
from self_cognition.core.state import StateDecisionStatus, SubjectState
from self_cognition.core.workspace import (
    RetrievalBudget,
    RetrievalQuery,
    RetrievalSource,
    WorkspaceBuilder,
)
from self_cognition.executive.dialogue.rule_based import RuleBasedDialogueModel
from self_cognition.infrastructure.llm.openai_responses import (
    OpenAIResponsesCognitionModel,
)
from self_cognition.infrastructure.persistence.file_deletion_repository import (
    _plan_from_json,
    _plan_to_json,
)
from self_cognition.infrastructure.persistence.serialization import (
    event_from_json,
    event_to_json,
    state_from_dict,
    state_from_json,
    state_to_dict,
    state_to_json,
)
from self_cognition.runtime.cognition_context import ReadOnlyCognitionContext
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.runtime.run_context import RunContext

NOW = datetime(2026, 9, 5, 8, tzinfo=timezone.utc)
META_FIELD = "metacognition.assessments.deploy"


@dataclass(frozen=True)
class FixedClock:
    value: datetime = NOW

    def now(self) -> datetime:
        return self.value


def context() -> RunContext:
    return RunContext(uuid4(), uuid4(), NOW + timedelta(hours=1), clock=FixedClock())


def subject(name: str = "alice", mind: str = "mind-19") -> SubjectScope:
    return SubjectScope(MindScope(mind), SubjectRef(SubjectKind.USER, name))


def message(
    name: str = "alice", mind: str = "mind-19", text: str = "部署任务失败"
) -> EventEnvelope:
    return EventEnvelope.user_message(subject(name, mind), text, clock=FixedClock())


def appraisal() -> AffectAssessment:
    return AffectAssessment("部署任务", (), "担忧", "negative", "task", 0.8, NOW)


def assessment() -> MetacognitiveAssessment:
    return MetacognitiveAssessment(
        "部署任务",
        KnowledgeStatus.UNKNOWN,
        EvidenceBasis.HYPOTHESIS,
        "现有证据不足以确定失败原因",
        FailureCause.UNKNOWN,
        (SuggestedAction.ASK,),
    )


class FakeAssessmentModel:
    def __init__(
        self,
        value: object,
        *,
        kind: str = "metacognition",
        cognition_type: CognitionType = CognitionType.UNKNOWN,
        target_field: str = META_FIELD,
        operation: ContributionOperation = ContributionOperation.SET,
    ) -> None:
        self.value = value
        self.kind = kind
        self.cognition_type = cognition_type
        self.target_field = target_field
        self.operation = operation
        self.calls = 0
        self.forbidden = False
        self.forge_evidence = False
        self.cancel = False

    def extract(self, request: CognitionRequest) -> ModelExtractionResult:
        assert not self.forbidden, "a persisted/deleted result must not call the model"
        self.calls += 1
        packet = request.context.query(
            RetrievalQuery(
                request.event.subject,
                "部署任务 preferences.study_time",
                purpose=f"cognition:{self.kind}",
                field_patterns=(
                    "metacognition.*",
                    "affect.*",
                    "preferences.*",
                    "goals.*",
                ),
                budget=RetrievalBudget(4096, 16),
            )
        )
        refs = [EvidenceRef.for_event(request.event)]
        if isinstance(request.event.payload, AssessmentRequestPayload):
            refs.append(EvidenceRef.for_event(request.event.payload.source_event))
        refs.extend(ref for item in packet.items for ref in item.evidence_refs)
        evidence_ids = tuple(dict.fromkeys(str(ref.evidence_id) for ref in refs))
        if self.forge_evidence:
            evidence_ids += (str(uuid4()),)
        run = request.run_context
        assert run is not None
        response = EventEnvelope.model_response(
            request.event,
            model="deterministic-fake",
            response_id=f"fake-{self.calls}",
            raw_output=json.dumps(self.value, ensure_ascii=False),
            clock=run.clock,
            run_id=run.run_id,
            correlation_id=run.correlation_id,
        )
        run.emit_event(response)
        if self.cancel:
            run.cancel()
        return ModelExtractionResult(
            response.payload.response_id,
            (
                ContributionCandidate(
                    self.target_field,
                    self.operation,
                    self.cognition_type,
                    self.value,
                    0.6,
                    evidence_ids,
                ),
            ),
            EvidenceRef.for_event(response),
        )


def process(container: ApplicationContainer, event: EventEnvelope) -> SubjectState:
    result = container.process_event.process(event, context())
    assert result.status is ProcessEventStatus.SUCCEEDED, result.error_type
    assert result.state is not None
    return result.state


@pytest.mark.parametrize(
    ("status", "basis", "cause", "cognition_type"),
    [
        (
            KnowledgeStatus.KNOWN,
            EvidenceBasis.DIRECT,
            FailureCause.PERMISSION,
            CognitionType.FACT,
        ),
        (
            KnowledgeStatus.KNOWN,
            EvidenceBasis.INFERENCE,
            FailureCause.ENVIRONMENT,
            CognitionType.INFERENCE,
        ),
        (
            KnowledgeStatus.KNOWN,
            EvidenceBasis.HYPOTHESIS,
            FailureCause.INPUT,
            CognitionType.INFERENCE,
        ),
        (
            KnowledgeStatus.CONFLICT,
            EvidenceBasis.INFERENCE,
            FailureCause.MODEL,
            CognitionType.INFERENCE,
        ),
        (
            KnowledgeStatus.EXPIRED,
            EvidenceBasis.DIRECT,
            FailureCause.STRATEGY,
            CognitionType.INFERENCE,
        ),
        (
            KnowledgeStatus.UNKNOWN,
            EvidenceBasis.HYPOTHESIS,
            FailureCause.UNKNOWN,
            CognitionType.UNKNOWN,
        ),
    ],
)
def test_generic_assessment_persists_evidence_and_replays_without_model(
    tmp_path: Path,
    status: KnowledgeStatus,
    basis: EvidenceBasis,
    cause: FailureCause,
    cognition_type: CognitionType,
) -> None:
    value = replace(assessment(), status=status, basis=basis, failure_cause=cause)
    model = FakeAssessmentModel(value.to_state_value(), cognition_type=cognition_type)
    container = build_container(tmp_path, metacognition_model=model)
    origin = message()
    container.event_store.append(origin)
    event = EventEnvelope.assessment_requested(
        origin, "评估部署任务", clock=FixedClock()
    )
    state = process(container, event)
    atom = state.get(META_FIELD)
    assert state.subject_kind is SubjectKind.MIND
    assert container.state_repository.load(origin.subject) is None
    assert atom.cognition_type is cognition_type
    assert {origin.event_id, event.event_id}.issubset(
        ref.evidence_id for ref in atom.evidence_refs
    )
    assert MetacognitiveAssessment.from_state_value(atom.value) == value
    assert event_from_json(event_to_json(event)) == event
    packet = container.workspace_builder.build("部署任务", state, as_of=NOW)
    response = container.dialogue_model.respond("部署任务", packet)
    assert "不是统计校准" in response.text and "尚未执行" in response.text
    assert response.evidence_refs
    model.forbidden = True
    assert container.replay.replay(state.subject_scope) == state
    assert process(container, event) == state
    assert state_from_json(state_to_json(state)) == state


def test_affect_has_target_goals_scope_and_read_only_decay(tmp_path: Path) -> None:
    value = appraisal().to_state_value()
    model = FakeAssessmentModel(
        value,
        kind="affect",
        cognition_type=CognitionType.AFFECT,
        target_field="affect.current.deploy",
    )
    container = build_container(tmp_path, affect_model=model)
    origin = message()
    user_state = process(container, origin)
    event = EventEnvelope.assessment_requested(
        origin, "评估部署任务的目标影响", clock=FixedClock()
    )
    mind_state = process(container, event)
    assert user_state.subject_kind is SubjectKind.USER
    assert mind_state.subject_kind is SubjectKind.MIND
    stored = mind_state.get("affect.current.deploy").value
    packet = container.workspace_builder.build(
        "部署任务", mind_state, as_of=NOW + timedelta(hours=1)
    )
    affect_item = next(
        item for item in packet.items if item.target_field == "affect.current.deploy"
    )
    assert affect_item.content["current_intensity"] == 0.4
    assert "不代表真实感受" in container.dialogue_model.respond("部署任务", packet).text
    assert stored == value and "current_intensity" not in stored
    assert decay_assessment(value, NOW - timedelta(seconds=1)) is None
    assert decay_assessment(value, NOW + timedelta(hours=4)) is None
    expired_atom = replace(
        mind_state.get("affect.current.deploy"), expires_at=NOW + timedelta(minutes=1)
    )
    expired_state = replace(mind_state, entries={"affect.current.deploy": expired_atom})
    query = RetrievalQuery(
        mind_state.subject_scope,
        "部署任务",
        purpose="cognition:metacognition",
        field_patterns=("affect.*",),
    )
    packet = container.workspace_builder.build(
        "部署任务", expired_state, as_of=NOW + timedelta(hours=1), query=query
    )
    assert packet.items[0].content["status"] == "expired"
    assert expired_state.get("affect.current.deploy").value == stored


@pytest.mark.parametrize(
    "problem", ["target", "type", "evidence", "goal", "time", "cancel"]
)
def test_invalid_or_cancelled_model_assessments_never_write_state(problem: str) -> None:
    value = appraisal().to_state_value()
    model = FakeAssessmentModel(
        value,
        kind="affect",
        cognition_type=CognitionType.AFFECT,
        target_field="affect.current.deploy",
    )
    if problem == "target":
        model.target_field = "values.principle"
    elif problem == "type":
        model.cognition_type = CognitionType.FACT
    elif problem == "evidence":
        model.forge_evidence = True
    elif problem == "goal":
        value["goal_ids"] = ["invented-goal"]
    elif problem == "time":
        value["assessed_at"] = (NOW + timedelta(days=1)).isoformat()
    else:
        model.cancel = True
    event = message()
    state = SubjectState.empty("alice", mind_id="mind-19")
    engine = CognitionEngine(
        (AffectExtractor(model),), CognitiveSpaceService(StateReducer())
    )
    results = engine.analyze(event, state, context())
    assert results[0].status is (
        CognitionResultStatus.CANCELLED
        if problem == "cancel"
        else CognitionResultStatus.FAILED
    )
    assert results[0].failure_type is (
        CognitionFailureType.CANCELLED
        if problem == "cancel"
        else CognitionFailureType.INVALID_OUTPUT
    )
    assert engine.reduce(state, results, decided_at=NOW) is state


@pytest.mark.parametrize(
    "status", [ConflictStatus.RESOLVED, ConflictStatus.INVALIDATED]
)
def test_conflict_lifecycle_requires_confirmation_and_preserves_candidates(
    status: ConflictStatus,
) -> None:
    seed = message(text="我既喜欢早上学习，也喜欢晚上学习")
    engine = CognitionEngine(
        (ConflictMetacognitionExtractor(),), CognitiveSpaceService(StateReducer())
    )
    state = engine.process(seed, SubjectState.empty("alice", mind_id="mind-19"))
    conflict = next(iter(state.conflicts))
    review = ConflictReview(
        conflict.candidate_contribution_ids, ConflictStatus.OPEN, "需要人工检查", True
    )

    def contribution(
        value: ConflictReview, current: SubjectState, confirmed: bool = False
    ) -> CognitiveContribution:
        return replace(
            CognitiveContribution.set_from_event(
                seed,
                contribution_id=uuid4(),
                target_field=conflict.target_field,
                cognition_type=CognitionType.INFERENCE,
                value=value.to_state_value(),
                confidence=0.8,
                evidence_refs=(EvidenceRef.for_event(seed),),
                source_module="metacognition.test",
                module_version="1",
                explicitly_confirmed=confirmed,
                target_version=current.version,
            ),
            operation=ContributionOperation.REVIEW_CONFLICT,
        )

    reducer = StateReducer()
    state = reducer.apply(state, contribution(review, state), decided_at=NOW)
    selected = (
        conflict.candidate_contribution_ids[0]
        if status is ConflictStatus.RESOLVED
        else None
    )
    review = replace(
        review,
        status=status,
        reason="已检查证据及适用时间",
        selected_contribution_id=selected,
    )
    pending = reducer.apply(state, contribution(review, state), decided_at=NOW)
    assert pending.changes[-1].status is StateDecisionStatus.PENDING
    assert next(iter(pending.conflicts)).status is ConflictStatus.OPEN
    event = EventEnvelope.conflict_reviewed(
        state.subject_scope,
        ConflictReviewPayload(conflict.target_field, review),
        actor=state.subject_scope,
        clock=FixedClock(),
    )
    final = engine.process(event, pending)
    closed = next(iter(final.conflicts))
    assert closed.status is status and closed.confirmed
    assert closed.conflict_id == conflict.conflict_id
    assert closed.candidate_contribution_ids == conflict.candidate_contribution_ids
    assert closed.resolution_reason == review.reason
    assert (
        not WorkspaceBuilder()
        .build("我喜欢什么时候学习？", final, as_of=NOW)
        .items_from(RetrievalSource.CONFLICT)
    )
    assert state_from_json(state_to_json(final)) == final
    assert event_from_json(event_to_json(event)) == event
    legacy = state_to_dict(state)
    legacy["schema_version"] = 4
    legacy["conflicts"] = [
        {
            key: item[key]
            for key in ("target_field", "candidate_contribution_ids", "reason")
        }
        for item in legacy["conflicts"]
    ]
    migrated = state_from_dict(legacy)
    assert migrated.changes == state.changes
    assert next(iter(migrated.conflicts)).status is ConflictStatus.OPEN


def test_explicit_correction_closes_conflict_without_rewriting_original() -> None:
    engine = CognitionEngine(
        (ConflictMetacognitionExtractor(), UserCorrectionModule()),
        CognitiveSpaceService(StateReducer()),
    )
    original = message(text="我既喜欢早上学习，也喜欢晚上学习")
    state = engine.process(original, SubjectState.empty("alice", mind_id="mind-19"))
    correction = EventEnvelope.correction(
        original.subject,
        target_field="preferences.study_time",
        cognition_type="preference",
        value="早上",
        clock=FixedClock(),
    )
    final = engine.process(correction, state)
    assert next(iter(final.conflicts)).status is ConflictStatus.RESOLVED
    packet = WorkspaceBuilder().build("我喜欢什么时候学习？", final, as_of=NOW)
    assert (
        RuleBasedDialogueModel().respond("我喜欢什么时候学习？", packet).text
        == "你喜欢早上学习。"
    )
    assert original.payload.text == "我既喜欢早上学习，也喜欢晚上学习"


@pytest.mark.parametrize("kind", ["metacognition", "affect"])
def test_responses_profiles_use_structured_values_and_fresh_context(kind: str) -> None:
    event = EventEnvelope.assessment_requested(
        message(), "评估部署任务", clock=FixedClock()
    )
    value = (
        assessment().to_state_value()
        if kind == "metacognition"
        else appraisal().to_state_value()
    )
    calls: list[dict[str, object]] = []

    def create(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            id="response",
            output_text=json.dumps(
                {
                    "candidates": [
                        {
                            "target_field": (
                                META_FIELD
                                if kind == "metacognition"
                                else "affect.current.deploy"
                            ),
                            "operation": "set",
                            "cognition_type": (
                                "unknown" if kind == "metacognition" else "affect"
                            ),
                            "value": value,
                            "confidence": 0.6,
                            "evidence_ids": [
                                str(event.event_id),
                                str(event.payload.source_event.event_id),
                            ],
                        }
                    ]
                }
            ),
        )

    model = OpenAIResponsesCognitionModel(
        SimpleNamespace(responses=SimpleNamespace(create=create)),
        "fake",
        assessment_kind=kind,
    )
    module = (
        ConflictMetacognitionExtractor(model)
        if kind == "metacognition"
        else AffectExtractor(model)
    )
    state = SubjectState.empty(
        "mind-19", mind_id="mind-19", subject_kind=SubjectKind.MIND
    )
    request = CognitionRequest(
        event, ReadOnlyCognitionContext(WorkspaceBuilder(), state, NOW), context()
    )
    assert module.run(request)[0].value == value
    assert module.run(replace(request, run_context=context()))[0].value == value
    assert all(
        call["store"] is False and "previous_response_id" not in call for call in calls
    )
    assert "source_event=" in calls[0]["input"]
    schema = calls[0]["text"]["format"]["schema"]
    assert (
        schema["properties"]["candidates"]["items"]["properties"]["value"].get("type")
        != "string"
    )


def setup_dependencies(
    path: Path,
) -> tuple[ApplicationContainer, FakeAssessmentModel, EventEnvelope, EventEnvelope]:
    model = FakeAssessmentModel(assessment().to_state_value())
    container = build_container(path, metacognition_model=model)
    first, second = (
        container.process_event.enqueue(event, context())
        for event in (message(), message("bob", text="部署任务补充了独立事实"))
    )
    for origin in (first, second):
        request = EventEnvelope.assessment_requested(
            origin, "评估部署任务", clock=FixedClock()
        )
        state = process(container, request)
    atom = state.get(META_FIELD)
    memory = MemoryRecord(
        uuid4(),
        MemoryType.SEMANTIC,
        state.subject_scope,
        atom.scope,
        "混合来源评估",
        atom.evidence_refs,
        0.6,
        0.5,
        0.5,
        1.0,
        1,
        MemoryLifecycleStatus.ACTIVE,
        NOW,
        "test.derived",
        "1",
    )
    container.memory_repository.save(memory, 0)
    return container, model, first, second


def test_dependency_deletion_preserves_other_sources_and_cannot_revive(
    tmp_path: Path,
) -> None:
    container, model, origin, other = setup_dependencies(tmp_path)
    unrelated = container.process_event.enqueue(
        message("carol", "other-mind"), context()
    )
    plan = container.forget.dry_run(
        DeletionSelector(origin.subject, delete_subject=True), now=NOW
    )
    assert {impact.subject.subject.kind for impact in plan.impacts} == {
        SubjectKind.USER,
        SubjectKind.MIND,
    }
    assert all(other.event_id not in impact.event_ids for impact in plan.impacts)
    assert _plan_from_json(_plan_to_json(plan)) == plan
    model.forbidden = True
    result = container.forget.execute(plan, now=NOW)
    assert result.status is DeletionStatus.COMPLETED
    assert container.event_store.read_by_subject(other.subject) == (other,)
    assert container.event_store.read_by_subject(unrelated.subject) == (unrelated,)
    mind = SubjectScope.for_mind("mind-19")
    assert container.memory_repository.read_by_subject(mind) == ()
    assert not container.replay.replay(mind).entries
    container.event_store.append(origin)
    assert container.event_store.read_by_subject(origin.subject) == ()
    refused = container.process_event.process(
        replace(origin, run_id=None, correlation_id=None), context()
    )
    assert refused.status is ProcessEventStatus.FAILED
    restarted = build_container(tmp_path, metacognition_model=model)
    assert not restarted.replay.replay(mind).entries
    assert container.forget.execute(plan, now=NOW) == result


def test_interrupted_cross_subject_deletion_recovers_without_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container, model, origin, other = setup_dependencies(tmp_path)
    plan = container.forget.dry_run(
        DeletionSelector(origin.subject, delete_subject=True), now=NOW
    )
    original_delete = container.memory_repository.delete

    def crash(subject_scope: SubjectScope, memory_ids: tuple[UUID, ...]) -> None:
        original_delete(subject_scope, memory_ids)
        raise SystemExit("simulated interruption after a committed deletion")

    monkeypatch.setattr(container.memory_repository, "delete", crash)
    with pytest.raises(SystemExit):
        container.forget.execute(plan, now=NOW)
    assert (
        container.deletion_repository.get(plan.plan_id).status
        is DeletionStatus.EXECUTING
    )
    model.forbidden = True
    restarted = build_container(tmp_path, metacognition_model=model)
    assert (
        restarted.deletion_repository.get(plan.plan_id).status
        is DeletionStatus.COMPLETED
    )
    assert restarted.event_store.read_by_subject(other.subject) == (other,)
    assert not restarted.replay.replay(SubjectScope.for_mind("mind-19")).entries


def test_legacy_plan_is_not_silently_expanded(tmp_path: Path) -> None:
    container, model, origin, _ = setup_dependencies(tmp_path)
    legacy = DeletionPlan(
        uuid4(),
        DeletionSelector(origin.subject, delete_subject=True),
        (),
        (origin.event_id,),
        NOW,
    )
    container.deletion_repository.save(legacy)
    assert _plan_from_json(_plan_to_json(legacy)) == legacy
    with pytest.raises(ValueError, match="new dry-run"):
        container.forget.execute(legacy, now=NOW)
    assert container.event_store.read_by_subject(origin.subject) == (origin,)
    assert (
        container.deletion_repository.get(legacy.plan_id).status
        is DeletionStatus.PLANNED
    )


def test_assessment_source_must_exist_and_cannot_cross_minds(tmp_path: Path) -> None:
    model = FakeAssessmentModel(assessment().to_state_value())
    container = build_container(tmp_path, metacognition_model=model)
    event = EventEnvelope.assessment_requested(
        message(), "评估部署任务", clock=FixedClock()
    )
    result = container.process_event.process(event, context())
    assert result.status is ProcessEventStatus.FAILED and model.calls == 0
    with pytest.raises(ContractValidationError):
        replace(
            event,
            subject=SubjectScope.for_mind("other-mind"),
            scope=DataScope(
                SubjectScope.for_mind("other-mind"), DisclosureScope.PRIVATE
            ),
        )
    with pytest.raises(ContractValidationError):
        EventEnvelope.assessment_requested(event, "递归评估", clock=FixedClock())


@pytest.mark.parametrize(
    "field,value",
    [
        ("half_life_seconds", float("inf")),
        ("initial_intensity", True),
        ("active_threshold", 0),
    ],
)
def test_affect_rejects_invalid_dynamics(field: str, value: object) -> None:
    payload = appraisal().to_state_value()
    payload[field] = value
    with pytest.raises(ContractValidationError):
        AffectAssessment.from_state_value(payload)

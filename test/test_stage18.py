from datetime import datetime, timezone
from uuid import UUID

from self_cognition.blackboard.reducer import StateReducer
from self_cognition.blackboard.service import CognitiveSpaceService
from self_cognition.cognition.metacognition.correction import UserCorrectionModule
from self_cognition.cognition.narrative.narrative_extractor import (
    NarrativeExtractor,
)
from self_cognition.cognition.relationship.relationship_extractor import (
    RelationshipExtractor,
)
from self_cognition.core.narratives import NarrativeLayer, NarrativeRecord
from self_cognition.core.relationships import RelationshipState
from self_cognition.core.events import Event
from self_cognition.core.scopes import (
    ConversationScope,
    DisclosureScope,
    MindScope,
    SubjectKind,
    SubjectRef,
    SubjectScope,
)
from self_cognition.core.state import SubjectState
from self_cognition.infrastructure.persistence.serialization import (
    state_from_json,
    state_to_json,
)
from self_cognition.runtime.engine import CognitionEngine


class FixedClock:
    def __init__(self, value: datetime) -> None:
        self._value = value

    def now(self) -> datetime:
        return self._value


def _engine(*modules: object) -> CognitionEngine:
    return CognitionEngine(
        tuple(modules),
        CognitiveSpaceService(StateReducer()),
    )


def test_structured_relationship_preserves_direction_scope_and_evidence():
    event = Event.user_message(
        SubjectScope.legacy_user("user-1"),
        "我和小明在研究项目中合作过",
        event_id=UUID(int=1801),
        clock=FixedClock(datetime(2026, 9, 4, tzinfo=timezone.utc)),
        disclosure=DisclosureScope.GROUP,
        conversation=ConversationScope("conversation-1", group_id="group-1"),
    )

    contribution = RelationshipExtractor().process(event)[0]
    relationship = RelationshipState.from_state_value(contribution.value)

    assert relationship.source == event.subject
    assert relationship.target.subject.subject_id == "小明"
    assert relationship.relation == "合作"
    assert relationship.context == "研究项目"
    assert relationship.shared_experience_ids == (event.event_id,)
    assert relationship.scope.disclosure is DisclosureScope.GROUP
    assert relationship.scope.conversation == event.scope.conversation
    assert contribution.evidence_refs[0].evidence_id == event.event_id


def test_group_message_stays_with_original_speaker_and_mind_isolation():
    event = Event.user_message(
        SubjectScope.legacy_user("user-1"),
        "我和小明在研究项目中合作过",
        event_id=UUID(int=1802),
        conversation=ConversationScope("conversation-1", group_id="group-1"),
    )
    state = _engine(RelationshipExtractor()).process(
        event,
        SubjectState.empty("user-1"),
    )

    assert state.subject_id == "user-1"
    assert any(field.startswith("relationships.edge.") for field in state.entries)
    assert "relationships.edge.小明.研究项目" in state.entries
    assert not any(field.startswith("relationships.edge.") for field in SubjectState.empty("user-2").entries)

    other_mind_event = Event.user_message(
        SubjectScope(MindScope("mind-2"), SubjectRef(SubjectKind.USER, "user-1")),
        "我和小明在研究项目中合作过",
        event_id=UUID(int=1803),
    )
    other_state = _engine(RelationshipExtractor()).process(
        other_mind_event,
        SubjectState.empty("user-1", mind_id="mind-2"),
    )
    assert other_state.subject_scope.mind != state.subject_scope.mind


def test_four_narrative_layers_and_explicit_unknown_are_structured():
    messages = (
        ("任务：准备论文", NarrativeLayer.TASK),
        ("这段时间我一直在准备论文", NarrativeLayer.PERIOD),
        ("我从学生变成研究者", NarrativeLayer.IDENTITY),
        ("我和小明一起完成了研究项目", NarrativeLayer.RELATIONSHIP),
        ("这个转折的原因我不知道", NarrativeLayer.TASK),
    )
    extractor = NarrativeExtractor()

    for index, (text, layer) in enumerate(messages, start=1810):
        event = Event.user_message(
            "user-1",
            text,
            event_id=UUID(int=index),
            clock=FixedClock(datetime(2026, 9, 4, index - 1800, tzinfo=timezone.utc)),
        )
        contribution = extractor.process(event)[0]
        record = NarrativeRecord.from_state_value(contribution.value)
        assert record.layer is layer
        assert record.evidence_event_ids == (event.event_id,)
        if text == "这个转折的原因我不知道":
            assert record.unknowns == ("转折原因",)


def test_narrative_correction_revises_view_without_erasing_original_change():
    first_event = Event.user_message(
        "user-1",
        "任务：准备论文",
        event_id=UUID(int=1820),
    )
    first_state = _engine(NarrativeExtractor(), UserCorrectionModule()).process(
        first_event,
        SubjectState.empty("user-1"),
    )
    field = next(iter(first_state.entries))
    corrected_value = dict(first_state.entries[field].value)
    corrected_value.update(
        summary="准备论文并完成初稿",
        revision_of=corrected_value["narrative_id"],
        version=2,
    )
    correction = Event.correction(
        "user-1",
        target_field=field,
        cognition_type="inference",
        value=corrected_value,
        event_id=UUID(int=1821),
    )
    corrected_state = _engine(
        NarrativeExtractor(),
        UserCorrectionModule(),
    ).process(correction, first_state)

    current = NarrativeRecord.from_state_value(corrected_state.entries[field].value)
    assert current.summary == "准备论文并完成初稿"
    assert current.revision_of == current.narrative_id.split(":", 1)[0] + ":" + str(first_event.event_id)
    assert len(corrected_state.changes) == 2
    assert corrected_state.changes[0].contribution.value["summary"] == "任务：准备论文"


def test_structured_relationship_and_narrative_values_roundtrip_in_state():
    event = Event.user_message(
        "user-1",
        "我和小明在研究项目中合作过",
        event_id=UUID(int=1830),
    )
    state = _engine(RelationshipExtractor()).process(
        event,
        SubjectState.empty("user-1"),
    )

    assert state_from_json(state_to_json(state)) == state

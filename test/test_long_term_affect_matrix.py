from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from self_cognition.application.proactive import ProactiveIntentionService
from self_cognition.bootstrap import build_container
from self_cognition.core.affect import (
    EmotionState,
    MoodState,
    decay_emotion,
    decay_mood,
)
from self_cognition.core.dialogue import GroundingReview
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.events import EventEnvelope
from self_cognition.core.governance import UserControls
from self_cognition.core.proactivity import MotiveProposal
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.workspace import WorkspaceFixedContext, WorkspacePacket
from self_cognition.executive.dialogue.review_policy import ReviewPolicy
from self_cognition.executive.dialogue.rule_based import RuleBasedDialogueModel
from self_cognition.infrastructure.llm.router import (
    ModelRegistration,
    ModelRouter,
    RoutedProactivityModel,
)
from self_cognition.infrastructure.persistence.in_memory_event_store import (
    InMemoryEventStore,
)
from self_cognition.infrastructure.persistence.in_memory_governance_repository import (
    InMemoryGovernanceRepository,
)
from self_cognition.runtime.run_context import RunContext


T0 = datetime(2026, 9, 15, 0, 0, tzinfo=timezone.utc)


class FixedClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def now(self) -> datetime:
        return self.value

    def advance(self, delta: timedelta) -> None:
        self.value += delta


def _context(clock: FixedClock) -> RunContext:
    return RunContext(
        uuid4(),
        uuid4(),
        clock.now() + timedelta(minutes=5),
        clock=clock,
    )


def _workspace(user: str, *, bored: bool = True) -> WorkspacePacket:
    emotion = ()
    if bored:
        emotion = (
            {
                "field": "mood.current",
                "content": {
                    "emotion": "boredom",
                    "valence": "negative",
                    "arousal": 0.1,
                },
            },
        )
    return WorkspacePacket(
        user,
        0,
        (),
        task_context="long-term affect matrix",
        fixed_context=WorkspaceFixedContext(emotion=emotion),
        subject=SubjectScope.legacy_user(user),
    )


def test_long_term_positive_matrix_is_bounded_and_idempotent():
    clock = FixedClock(T0)
    store = InMemoryEventStore()
    service = ProactiveIntentionService(store)
    user = "long-term-positive"
    subject = SubjectScope.legacy_user(user)
    workspace = _workspace(user)

    first = EventEnvelope.user_message(user, "好无聊，想找人聊聊", clock=clock)
    assert service.form_boredom_social_motive(first, workspace, _context(clock)) is not None

    clock.advance(timedelta(minutes=5))
    second = EventEnvelope.user_message(user, "好无聊，想找人聊聊", clock=clock)
    assert service.form_boredom_social_motive(second, workspace, _context(clock)) is None

    clock.advance(timedelta(minutes=30))
    third = EventEnvelope.user_message(user, "好无聊，想找人聊聊", clock=clock)
    assert service.form_boredom_social_motive(third, workspace, _context(clock)) is not None

    due = service.consume_due(
        subject,
        as_of=clock.now(),
        context=_context(clock),
        workspace=workspace,
    )
    assert len(due) == 1
    assert (
        service.consume_due(
            subject,
            as_of=clock.now(),
            context=_context(clock),
            workspace=workspace,
        )
        == ()
    )
    assert len(service.mailbox(subject, as_of=clock.now())) == 1

    clock.advance(timedelta(minutes=11))
    assert (
        service.consume_due(
            subject,
            as_of=clock.now(),
            context=_context(clock),
            workspace=workspace,
        )
        == ()
    )

    restarted = ProactiveIntentionService(store)
    messages = restarted.mailbox(subject, as_of=clock.now())
    assert len(messages) == 1
    assert messages[0]["status"] == "expired"


def test_long_term_negative_matrix_silences_disabled_proactive_controls():
    clock = FixedClock(T0)
    store = InMemoryEventStore()
    governance = InMemoryGovernanceRepository()
    service = ProactiveIntentionService(store, governance=governance)
    user = "long-term-controls"
    subject = SubjectScope.legacy_user(user)
    workspace = _workspace(user)

    event = EventEnvelope.user_message(user, "好无聊，想找人聊聊", clock=clock)
    assert service.form_boredom_social_motive(event, workspace, _context(clock)) is not None
    assert len(service.active(subject, as_of=clock.now())) == 1

    governance.save_controls(
        UserControls(
            subject,
            disabled_proactive_tasks=frozenset({"social_connection"}),
        )
    )
    assert service.active(subject, as_of=clock.now()) == ()

    governance.save_controls(
        UserControls(
            subject,
            disabled_proactive_channels=frozenset({"default"}),
        )
    )
    assert service.active(subject, as_of=clock.now()) == ()

    governance.save_controls(
        UserControls(subject, proactive_frequency_per_hour=0)
    )
    assert service.active(subject, as_of=clock.now()) == ()
    assert (
        service.consume_due(
            subject,
            as_of=clock.now(),
            context=_context(clock),
            workspace=workspace,
        )
        == ()
    )


def test_long_term_mood_decay_ends_proactive_motive():
    clock = FixedClock(T0)
    mood = MoodState(
        mood_id=uuid4(),
        target="decay-user",
        valence="negative",
        scope="interaction",
        intensity=0.2,
        updated_at=T0,
        half_life_seconds=60.0,
        active_threshold=0.05,
        source_emotion_ids=(uuid4(),),
    )
    assert decay_mood(mood, T0 + timedelta(seconds=180)) is None

    emotion = EmotionState(
        emotion_id=uuid4(),
        target="decay-user",
        emotion="boredom",
        valence="negative",
        scope="interaction",
        intensity=0.2,
        assessed_at=T0,
        half_life_seconds=60.0,
        active_threshold=0.1,
    )
    assert decay_emotion(emotion, T0 + timedelta(seconds=120)) is None

    event = EventEnvelope.user_message(
        "decay-user",
        "好无聊，想找人聊聊",
        clock=clock,
    )
    service = ProactiveIntentionService(InMemoryEventStore())
    assert (
        service.form_boredom_social_motive(
            event,
            _workspace("decay-user", bored=False),
            _context(clock),
        )
        is None
    )


@pytest.mark.parametrize(
    ("review", "bored", "accepted", "modulated"),
    (
        (
            GroundingReview(True, "supported"),
            False,
            True,
            False,
        ),
        (
            GroundingReview(
                False,
                "factual claim is unsupported",
                social_response=True,
                unsupported_claims=("用户喜欢茶。",),
            ),
            True,
            False,
            False,
        ),
        (
            GroundingReview(
                False,
                "social response rejected by strict review",
                social_response=True,
            ),
            True,
            True,
            True,
        ),
        (
            GroundingReview(
                False,
                "social response without boredom",
                social_response=True,
            ),
            False,
            False,
            False,
        ),
    ),
)
def test_long_term_review_boundary_matrix(
    review: GroundingReview,
    bored: bool,
    accepted: bool,
    modulated: bool,
):
    decision = ReviewPolicy().decide(review, _workspace("review-user", bored=bored))
    assert decision.accepted is accepted
    assert decision.emotion_modulated is modulated


def test_long_term_proactive_provider_cooldown_is_silent():
    router = ModelRouter(
        (ModelRegistration("proactive", "proactive-default", object()),),
        failure_cooldown=timedelta(minutes=30),
        max_failure_cooldown=timedelta(minutes=30),
    )
    router.mark_degraded("proactive-default", "proactive", "ModelTimeoutError")
    service = ProactiveIntentionService(
        InMemoryEventStore(),
        model=RoutedProactivityModel(router),
    )
    event = EventEnvelope.user_message("cooldown-user", "好无聊", clock=FixedClock(T0))
    assert (
        service.evaluate(
            event,
            _workspace("cooldown-user"),
            _context(FixedClock(T0)),
        )
        is None
    )


class FailingProactiveModel:
    def propose(self, event, workspace, context):
        del event, workspace, context
        raise ContractValidationError("invalid proactive evidence")

    def express(self, intention, workspace, context):
        del intention, workspace, context
        return "unused"


def test_after_success_proactive_failure_does_not_skip_dialogue(tmp_path):
    app = build_container(
        tmp_path / "data",
        dotenv_path=tmp_path / "missing.env",
        dialogue_model=RuleBasedDialogueModel(),
        proactive_model=FailingProactiveModel(),
    )
    event = EventEnvelope.user_message("resilient-user", "你好")
    context = RunContext(
        uuid4(),
        uuid4(),
        datetime.now(timezone.utc) + timedelta(minutes=1),
    )
    try:
        app.event_bus.publish(event, context)
        results = app.event_bus.drain()

        events = app.event_store.read_by_mind(event.subject.mind)
        active_health = app.health.check().as_dict()
    finally:
        app.lifecycle.stop()

    assert results[0].status.value == "succeeded"
    assert any(item.event_type == "assistant.message" for item in events)
    assert active_health["ready"] is False
    proactive_status = next(
        item
        for item in app.model_router.statuses()
        if item.task == "proactive"
    )
    assert proactive_status.healthy is False
    assert proactive_status.degraded_reason == "ContractValidationError"


class SocialProactivityModel:
    def propose(self, event, workspace, context):
        del workspace, context
        return MotiveProposal(
            True,
            "social_engagement",
            "长期社交需要",
            "邀请用户聊天",
            0.8,
            1,
            60,
            (str(event.event_id),),
        )

    def express(self, intention, workspace, context):
        del intention, workspace, context
        return "聊聊天？"


def test_long_term_model_social_proposals_respect_cooldown():
    clock = FixedClock(T0)
    store = InMemoryEventStore()
    service = ProactiveIntentionService(store, model=SocialProactivityModel())
    user = "model-social-user"
    subject = SubjectScope.legacy_user(user)
    workspace = _workspace(user, bored=False)

    first = EventEnvelope.user_message(user, "今天有点空", clock=clock)
    assert service.evaluate(first, workspace, _context(clock)) is not None

    clock.advance(timedelta(minutes=5))
    second = EventEnvelope.user_message(user, "今天有点空", clock=clock)
    assert service.evaluate(second, workspace, _context(clock)) is None

    clock.advance(timedelta(minutes=30))
    third = EventEnvelope.user_message(user, "今天有点空", clock=clock)
    assert service.evaluate(third, workspace, _context(clock)) is not None

    assert len(service.active(subject, as_of=clock.now())) == 1


class GoalProactivityModel:
    def __init__(self):
        self.calls = 0

    def propose(self, event, workspace, context):
        del workspace, context
        kinds = ("proactive_goal_monitoring", "monitor_long_term_goal")
        kind = kinds[min(self.calls, len(kinds) - 1)]
        self.calls += 1
        return MotiveProposal(
            True,
            kind,
            "长期目标关注",
            "持续关注长期目标",
            0.8,
            1,
            3600,
            (str(event.event_id),),
        )

    def express(self, intention, workspace, context):
        del intention, workspace, context
        return "目标进展如何？"


def test_long_term_goal_proposals_share_cooldown_family():
    clock = FixedClock(T0)
    model = GoalProactivityModel()
    service = ProactiveIntentionService(InMemoryEventStore(), model=model)
    user = "model-goal-user"
    subject = SubjectScope.legacy_user(user)
    workspace = _workspace(user, bored=False)

    first = EventEnvelope.user_message(user, "请留意长期目标", clock=clock)
    assert service.evaluate(first, workspace, _context(clock)) is not None

    clock.advance(timedelta(minutes=5))
    second = EventEnvelope.user_message(user, "请继续留意长期目标", clock=clock)
    assert service.evaluate(second, workspace, _context(clock)) is None

    clock.advance(timedelta(minutes=30))
    third = EventEnvelope.user_message(user, "再留意一下长期目标", clock=clock)
    assert service.evaluate(third, workspace, _context(clock)) is not None

    assert len(service.active(subject, as_of=clock.now())) == 2

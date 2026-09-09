from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from self_cognition.bootstrap import build_container
from self_cognition.core.affect import Motive
from self_cognition.core.dialogue import DialogueRequest
from self_cognition.core.events import EventEnvelope
from self_cognition.core.proactivity import ProactiveIntention
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.workspace import WorkspacePacket
from self_cognition.infrastructure.persistence.serialization import (
    event_from_json,
    event_to_json,
)
from self_cognition.runtime.run_context import RunContext


NOW = datetime(2026, 9, 9, 16, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class FixedClock:
    value: datetime = NOW

    def now(self) -> datetime:
        return self.value


def _context() -> RunContext:
    return RunContext(
        uuid4(),
        uuid4(),
        NOW + timedelta(minutes=5),
        clock=FixedClock(),
    )


def test_restart_preserves_state_and_intention_without_rolling_context(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    app = build_container(data, dotenv_path=tmp_path / "missing.env")
    user = SubjectScope.legacy_user("stage29-user")
    mind = SubjectScope.for_mind(user.mind.mind_id)
    for text in (
        "我喜欢晚上学习",
        "小明是我的朋友",
        "这次考试通过了，我很开心",
    ):
        result = app.process_event.process(
            EventEnvelope.user_message(user, text, clock=FixedClock()),
            _context(),
        )
        assert result.error_type is None
    source = EventEnvelope.user_message(user, "记得跟进目标", clock=FixedClock())
    assert app.process_event.process(source, _context()).error_type is None
    motive = Motive(
        uuid4(),
        mind.subject.subject_id,
        "goal",
        "跟进未完成目标",
        0.8,
        1,
        NOW,
        (source.event_id,),
    )
    intention = ProactiveIntention(
        uuid4(),
        motive,
        mind,
        "询问目标进展",
        1,
        0,
        (),
        NOW,
        NOW + timedelta(hours=1),
        idempotency_key="stage29-follow-up",
    )
    app.proactive.propose(intention, context=_context())
    expected_user_state = app.state_repository.load(user)
    expected_intentions = app.proactive.active(mind, as_of=NOW)
    app.lifecycle.stop()

    restarted = build_container(data, dotenv_path=tmp_path / "missing.env")
    try:
        assert restarted.state_repository.load(user) == expected_user_state
        assert restarted.replay.replay(user) == expected_user_state
        assert restarted.proactive.active(mind, as_of=NOW) == expected_intentions
        current = EventEnvelope.user_message(user, "现在呢？", clock=FixedClock())
        fast = restarted.scheduler.fast.run(
            current,
            expected_user_state,
            expected_intentions,
        )
        assert fast.intentions == expected_intentions
    finally:
        restarted.lifecycle.stop()


def test_legacy_workspace_and_stage20_answer_event_remain_readable(
    tmp_path: Path,
) -> None:
    legacy_workspace = WorkspacePacket("stage20-user", 0, ())
    assert legacy_workspace.workspace_version == 1
    assert legacy_workspace.subject is None

    app = build_container(tmp_path, dotenv_path=tmp_path / "missing.env")
    try:
        user = SubjectScope.legacy_user("stage20-user")
        result = app.converse.converse(
            DialogueRequest(
                EventEnvelope.user_message(user, "你好！", clock=FixedClock())
            ),
            _context(),
        )
        answer = next(
            event
            for event in app.event_store.read_by_mind(user.mind)
            if event.event_id == result.response_event_id
        )
        assert event_from_json(event_to_json(answer)) == answer
    finally:
        app.lifecycle.stop()

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest

from self_cognition.cognition.affect.affect_extractor import AffectExtractor
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.events import Event
from self_cognition.core.ids import (
    contribution_id,
    new_correlation_id,
    new_event_id,
    new_run_id,
)
from self_cognition.core.state import SubjectState
from self_cognition.core.time import SYSTEM_CLOCK
from self_cognition.core.workspace import WorkspaceBuilder
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.runtime.reducer import StateReducer
from self_cognition.runtime.run_context import RunContext


@dataclass(frozen=True, slots=True)
class FixedClock:
    value: datetime

    def now(self) -> datetime:
        return self.value


def test_contribution_id_preserves_the_existing_uuid5_contract() -> None:
    event_id = UUID(int=1)

    actual = contribution_id(
        event_id,
        "semantic.preference_extractor",
        "preferences.study_time",
    )

    assert actual == uuid5(
        NAMESPACE_URL,
        (
            f"{event_id}:semantic.preference_extractor:"
            "preferences.study_time"
        ),
    )
    with pytest.raises(ContractValidationError):
        contribution_id(event_id, "", "preferences.study_time")
    with pytest.raises(ContractValidationError):
        contribution_id(event_id, "test.module", "")


def test_runtime_id_factories_return_distinct_uuids() -> None:
    identifiers = {
        new_event_id(),
        new_event_id(),
        new_run_id(),
        new_correlation_id(),
    }

    assert len(identifiers) == 4
    assert all(isinstance(identifier, UUID) for identifier in identifiers)


def test_event_and_run_context_use_injected_clocks() -> None:
    current_time = datetime(2026, 8, 29, 12, tzinfo=timezone.utc)
    clock = FixedClock(current_time)
    event = Event.user_message("user-1", "测试时钟", clock=clock)
    active_context = RunContext(
        run_id=UUID(int=1),
        correlation_id=UUID(int=2),
        deadline=current_time + timedelta(seconds=1),
        clock=clock,
    )
    expired_context = RunContext(
        run_id=UUID(int=3),
        correlation_id=UUID(int=4),
        deadline=current_time - timedelta(seconds=1),
        clock=clock,
    )

    assert event.occurred_at == current_time
    assert active_context.is_cancelled is False
    assert expired_context.is_cancelled is True
    assert SYSTEM_CLOCK.now().utcoffset() is not None


def test_workspace_uses_its_clock_for_affect_decay() -> None:
    assessed_at = datetime(2026, 8, 29, 12, tzinfo=timezone.utc)
    event = Event.user_message(
        "user-1",
        "这次考试通过了，我很开心",
        clock=FixedClock(assessed_at),
    )
    state = CognitionEngine(
        modules=(AffectExtractor(),),
        reducer=StateReducer(),
    ).process(event, SubjectState.empty("user-1"))
    builder = WorkspaceBuilder(
        clock=FixedClock(assessed_at + timedelta(hours=1))
    )

    workspace = builder.build("我对这次考试感觉怎么样？", state)

    assert workspace.items[0].content["current_intensity"] == 0.4

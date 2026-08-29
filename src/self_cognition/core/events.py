from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.ids import new_event_id
from self_cognition.core.time import Clock, SYSTEM_CLOCK

@dataclass(frozen=True, slots=True)
class Event:
    event_id: UUID
    event_type: str
    actor_id: str
    content: str
    occurred_at: datetime

    def __post_init__(self) -> None:
        if not self.event_type.strip():
            raise ContractValidationError("event_type must not be blank")
        if not self.actor_id.strip():
            raise ContractValidationError("actor_id must not be blank")
        if not self.content.strip():
            raise ContractValidationError("content must not be blank")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ContractValidationError("occurred_at must include timezone information")

    @classmethod
    def user_message(
        cls,
        actor_id: str,
        content: str,
        *,
        clock: Clock = SYSTEM_CLOCK,
    ) -> "Event":
        return cls(
            event_id=new_event_id(),
            event_type="user.message",
            actor_id=actor_id,
            content=content,
            occurred_at=clock.now(),
        )

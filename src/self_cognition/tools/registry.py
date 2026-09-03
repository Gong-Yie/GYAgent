from dataclasses import dataclass, replace
from threading import RLock
from uuid import UUID

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.events import EventEnvelope, EventSource
from self_cognition.core.identity import (
    CapabilityExecutionStatus,
    CapabilityKind,
    CapabilityPermission,
    CapabilityRecord,
)
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.time import Clock, SYSTEM_CLOCK


@dataclass(frozen=True, slots=True)
class CapabilityRegistration:
    capability_id: str
    name: str
    kind: CapabilityKind
    permission: CapabilityPermission
    enabled: bool = True
    reason: str | None = None

    def to_record(self) -> CapabilityRecord:
        return CapabilityRecord(
            capability_id=self.capability_id,
            name=self.name,
            kind=self.kind,
            registered=self.enabled,
            permission=self.permission,
            reason=self.reason,
        )


class CapabilityRegistry:
    def __init__(
        self,
        registrations: tuple[CapabilityRegistration, ...] = (),
    ) -> None:
        self._records: dict[str, CapabilityRecord] = {}
        self._lock = RLock()
        for registration in registrations:
            self.register(registration)

    def register(self, registration: CapabilityRegistration) -> None:
        if not isinstance(registration, CapabilityRegistration):
            raise TypeError("registration must be a CapabilityRegistration")
        record = registration.to_record()
        with self._lock:
            if record.capability_id in self._records:
                raise ValueError(
                    f"capability is already registered: {record.capability_id}"
                )
            self._records[record.capability_id] = record

    def registrations(self) -> tuple[CapabilityRecord, ...]:
        with self._lock:
            return tuple(
                self._records[capability_id]
                for capability_id in sorted(self._records)
            )

    def registration_event(
        self,
        capability_id: str,
        subject: SubjectScope,
        *,
        event_id: UUID | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ) -> EventEnvelope:
        return EventEnvelope.capability_observed(
            subject,
            self._get(capability_id),
            source=EventSource.SYSTEM,
            event_id=event_id,
            clock=clock,
        )

    def record_execution(
        self,
        capability_id: str,
        subject: SubjectScope,
        *,
        succeeded: bool,
        reason: str | None = None,
        event_id: UUID | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ) -> EventEnvelope:
        if not isinstance(succeeded, bool):
            raise ContractValidationError("succeeded must be a boolean")
        with self._lock:
            current = self._require_record(capability_id)
            if not current.available:
                raise ContractValidationError(
                    "unavailable capability cannot record execution"
                )
            updated = replace(
                current,
                execution_status=(
                    CapabilityExecutionStatus.SUCCEEDED
                    if succeeded
                    else CapabilityExecutionStatus.FAILED
                ),
                reason=None if succeeded else reason,
            )
            self._records[capability_id] = updated
        return EventEnvelope.capability_observed(
            subject,
            updated,
            source=EventSource.TOOL,
            event_id=event_id,
            clock=clock,
        )

    def _get(self, capability_id: str) -> CapabilityRecord:
        with self._lock:
            return self._require_record(capability_id)

    def _require_record(self, capability_id: str) -> CapabilityRecord:
        try:
            return self._records[capability_id]
        except KeyError as error:
            raise KeyError(f"unknown capability: {capability_id}") from error

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Mapping
from uuid import UUID

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.memories import MemoryType
from self_cognition.core.scopes import SubjectScope


class AuditAction(str, Enum):
    READ = "read"
    WRITE = "write"
    APPROVE = "approve"
    EXPORT = "export"
    DISCLOSURE = "disclosure"
    DELETE = "delete"
    CONTROL = "control"


@dataclass(frozen=True, slots=True)
class AuditRecord:
    audit_id: UUID
    action: AuditAction
    subject: SubjectScope
    actor: SubjectScope | None
    target_type: str
    target_id: str
    occurred_at: datetime
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.audit_id, UUID):
            raise ContractValidationError("audit_id must be a UUID")
        if not isinstance(self.action, AuditAction):
            raise ContractValidationError("audit action is invalid")
        if not isinstance(self.subject, SubjectScope):
            raise ContractValidationError("audit subject is invalid")
        if self.actor is not None and not isinstance(self.actor, SubjectScope):
            raise ContractValidationError("audit actor is invalid")
        for name, value in (("target_type", self.target_type), ("target_id", self.target_id)):
            if not isinstance(value, str) or not value.strip():
                raise ContractValidationError(f"audit {name} must not be blank")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ContractValidationError("audit occurred_at must include timezone")
        try:
            json.dumps(dict(self.details), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ContractValidationError("audit details must be JSON serializable") from error

    def to_dict(self, *, redacted: bool = False) -> dict[str, object]:
        return {
            "schema_version": 1,
            "audit_id": str(self.audit_id),
            "action": self.action.value,
            "subject": _subject_to_dict(self.subject),
            "actor": _subject_to_dict(self.actor) if self.actor else None,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "occurred_at": self.occurred_at.isoformat(),
            "details": _redact(dict(self.details)) if redacted else dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    policy_id: UUID
    subject: SubjectScope
    retention_days: int
    memory_types: tuple[MemoryType, ...] = ()
    conversation_id: str | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, UUID):
            raise ContractValidationError("retention policy ID is invalid")
        if not isinstance(self.subject, SubjectScope):
            raise ContractValidationError("retention policy subject is invalid")
        if type(self.retention_days) is not int or self.retention_days < 0:
            raise ContractValidationError("retention days must be non-negative")
        if any(not isinstance(item, MemoryType) for item in self.memory_types):
            raise ContractValidationError("retention memory types are invalid")
        if self.conversation_id is not None and not self.conversation_id.strip():
            raise ContractValidationError("retention conversation_id must not be blank")
        if not isinstance(self.enabled, bool):
            raise ContractValidationError("retention enabled must be boolean")


@dataclass(frozen=True, slots=True)
class UserControls:
    subject: SubjectScope
    disabled_modules: frozenset[str] = frozenset()
    disabled_model_providers: frozenset[str] = frozenset()
    disabled_proactive_tasks: frozenset[str] = frozenset()
    disabled_proactive_channels: frozenset[str] = frozenset()
    proactive_frequency_per_hour: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.subject, SubjectScope):
            raise ContractValidationError("control subject is invalid")
        for name in (
            "disabled_modules",
            "disabled_model_providers",
            "disabled_proactive_tasks",
            "disabled_proactive_channels",
        ):
            values = getattr(self, name)
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ContractValidationError(f"{name} must contain text values")
        if self.proactive_frequency_per_hour is not None and (
            type(self.proactive_frequency_per_hour) is not int
            or self.proactive_frequency_per_hour < 0
        ):
            raise ContractValidationError("proactive frequency must be non-negative")

    def allows_proactive(self, task: str = "", channel: str = "default") -> bool:
        if task in self.disabled_proactive_tasks or channel in self.disabled_proactive_channels:
            return False
        return self.proactive_frequency_per_hour != 0


def _subject_to_dict(subject: SubjectScope) -> dict[str, str]:
    return {
        "mind_id": subject.mind.mind_id,
        "kind": subject.subject.kind.value,
        "subject_id": subject.subject.subject_id,
    }


def _redact(value: object, key: str = "") -> object:
    if isinstance(value, Mapping):
        return {
            str(name): "[REDACTED]"
            if _sensitive_key(str(name))
            else _redact(item, str(name))
            for name, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, key) for item in value]
    return value


def _sensitive_key(value: str) -> bool:
    normalized = value.lower()
    return any(token in normalized for token in ("secret", "token", "password", "api_key", "private_key"))

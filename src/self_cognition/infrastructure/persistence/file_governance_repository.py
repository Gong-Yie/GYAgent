from __future__ import annotations

import json
import os
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from threading import RLock
from uuid import UUID

from self_cognition.core.governance import (
    AuditAction,
    AuditRecord,
    RetentionPolicy,
    UserControls,
)
from self_cognition.core.memories import MemoryType
from self_cognition.core.scopes import MindScope, SubjectKind, SubjectRef, SubjectScope
from self_cognition.infrastructure.persistence.atomic_io import atomic_write_text


class FileGovernanceRepository:
    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._audit_path = self._directory / "audit.jsonl"
        self._controls = self._directory / "controls"
        self._retention = self._directory / "retention"
        self._lock = RLock()

    def append_audit(self, record: AuditRecord) -> None:
        with self._lock, self._audit_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def read_audit(self, subject: SubjectScope) -> tuple[AuditRecord, ...]:
        if not self._audit_path.exists():
            return ()
        result = []
        for line in self._audit_path.read_text(encoding="utf-8").splitlines():
            values = json.loads(line)
            item = AuditRecord(
                UUID(values["audit_id"]), AuditAction(values["action"]),
                _subject(values["subject"]),
                None if values["actor"] is None else _subject(values["actor"]),
                values["target_type"], values["target_id"],
                datetime.fromisoformat(values["occurred_at"]),
                values.get("details", {}),
            )
            if item.subject == subject:
                result.append(item)
        return tuple(result)

    def load_controls(self, subject: SubjectScope) -> UserControls:
        path = self._controls / f"{_digest(subject)}.json"
        if not path.exists():
            return UserControls(subject)
        values = json.loads(path.read_text(encoding="utf-8"))
        return UserControls(
            subject,
            frozenset(values.get("disabled_modules", ())),
            frozenset(values.get("disabled_model_providers", ())),
            frozenset(values.get("disabled_proactive_tasks", ())),
            frozenset(values.get("disabled_proactive_channels", ())),
            values.get("proactive_frequency_per_hour"),
        )

    def save_controls(self, controls: UserControls) -> None:
        self._controls.mkdir(parents=True, exist_ok=True)
        values = {
            "schema_version": 1,
            "subject": _subject_dict(controls.subject),
            "disabled_modules": sorted(controls.disabled_modules),
            "disabled_model_providers": sorted(controls.disabled_model_providers),
            "disabled_proactive_tasks": sorted(controls.disabled_proactive_tasks),
            "disabled_proactive_channels": sorted(controls.disabled_proactive_channels),
            "proactive_frequency_per_hour": controls.proactive_frequency_per_hour,
        }
        atomic_write_text(self._controls / f"{_digest(controls.subject)}.json", json.dumps(values, ensure_ascii=False, sort_keys=True) + "\n")

    def save_retention(self, policy: RetentionPolicy) -> None:
        self._retention.mkdir(parents=True, exist_ok=True)
        values = {
            "schema_version": 1,
            "policy_id": str(policy.policy_id),
            "subject": _subject_dict(policy.subject),
            "retention_days": policy.retention_days,
            "memory_types": [item.value for item in policy.memory_types],
            "conversation_id": policy.conversation_id,
            "enabled": policy.enabled,
        }
        atomic_write_text(self._retention / f"{policy.policy_id}.json", json.dumps(values, ensure_ascii=False, sort_keys=True) + "\n")

    def read_retention(self, subject: SubjectScope) -> tuple[RetentionPolicy, ...]:
        if not self._retention.exists():
            return ()
        result = []
        for path in sorted(self._retention.glob("*.json")):
            values = json.loads(path.read_text(encoding="utf-8"))
            if _subject(values["subject"]) != subject:
                continue
            result.append(RetentionPolicy(
                UUID(values["policy_id"]), subject, int(values["retention_days"]),
                tuple(MemoryType(item) for item in values.get("memory_types", ())),
                values.get("conversation_id"), bool(values.get("enabled", True)),
            ))
        return tuple(result)


def _subject_dict(subject: SubjectScope) -> dict[str, str]:
    return {"mind_id": subject.mind.mind_id, "kind": subject.subject.kind.value, "subject_id": subject.subject.subject_id}


def _subject(value: dict[str, str]) -> SubjectScope:
    return SubjectScope(MindScope(value["mind_id"]), SubjectRef(SubjectKind(value["kind"]), value["subject_id"]))


def _digest(subject: SubjectScope) -> str:
    return sha256(json.dumps(_subject_dict(subject), sort_keys=True).encode()).hexdigest()

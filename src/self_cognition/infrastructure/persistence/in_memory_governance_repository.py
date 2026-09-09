from __future__ import annotations

from threading import RLock

from self_cognition.core.governance import AuditRecord, RetentionPolicy, UserControls
from self_cognition.core.scopes import SubjectScope


class InMemoryGovernanceRepository:
    def __init__(self) -> None:
        self._audits: list[AuditRecord] = []
        self._controls: dict[SubjectScope, UserControls] = {}
        self._retention: dict[SubjectScope, list[RetentionPolicy]] = {}
        self._lock = RLock()

    def append_audit(self, record: AuditRecord) -> None:
        with self._lock:
            if record not in self._audits:
                self._audits.append(record)

    def read_audit(self, subject: SubjectScope) -> tuple[AuditRecord, ...]:
        with self._lock:
            return tuple(item for item in self._audits if item.subject == subject)

    def load_controls(self, subject: SubjectScope) -> UserControls:
        with self._lock:
            return self._controls.get(subject, UserControls(subject))

    def save_controls(self, controls: UserControls) -> None:
        with self._lock:
            self._controls[controls.subject] = controls

    def save_retention(self, policy: RetentionPolicy) -> None:
        with self._lock:
            items = self._retention.setdefault(policy.subject, [])
            items[:] = [item for item in items if item.policy_id != policy.policy_id]
            items.append(policy)

    def read_retention(self, subject: SubjectScope) -> tuple[RetentionPolicy, ...]:
        with self._lock:
            return tuple(self._retention.get(subject, ()))

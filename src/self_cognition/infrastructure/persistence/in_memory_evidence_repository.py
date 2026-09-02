from uuid import UUID
from threading import RLock

from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.scopes import SubjectScope


class InMemoryEvidenceRepository:
    def __init__(self) -> None:
        self._evidence: dict[tuple[str, UUID], EvidenceRef] = {}
        self._lock = RLock()

    def append(self, evidence: EvidenceRef) -> None:
        key = (evidence.scope.owner.mind.mind_id, evidence.evidence_id)
        with self._lock:
            self._evidence.setdefault(key, evidence)

    def get(
        self,
        subject: SubjectScope,
        evidence_id: UUID,
    ) -> EvidenceRef | None:
        if not isinstance(subject, SubjectScope):
            raise TypeError("subject must be a SubjectScope")
        with self._lock:
            return self._evidence.get((subject.mind.mind_id, evidence_id))

    def read_by_subject(
        self,
        subject: SubjectScope,
    ) -> tuple[EvidenceRef, ...]:
        if not isinstance(subject, SubjectScope):
            raise TypeError("subject must be a SubjectScope")
        with self._lock:
            return tuple(
                evidence
                for evidence in self._evidence.values()
                if evidence.scope.owner == subject
            )

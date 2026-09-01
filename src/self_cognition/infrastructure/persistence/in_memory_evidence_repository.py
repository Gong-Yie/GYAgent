from uuid import UUID

from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.scopes import SubjectScope


class InMemoryEvidenceRepository:
    def __init__(self) -> None:
        self._evidence: dict[tuple[str, UUID], EvidenceRef] = {}

    def append(self, evidence: EvidenceRef) -> None:
        key = (evidence.scope.owner.mind.mind_id, evidence.evidence_id)
        self._evidence.setdefault(key, evidence)

    def get(
        self,
        subject: SubjectScope,
        evidence_id: UUID,
    ) -> EvidenceRef | None:
        if not isinstance(subject, SubjectScope):
            raise TypeError("subject must be a SubjectScope")
        return self._evidence.get((subject.mind.mind_id, evidence_id))

    def read_by_subject(
        self,
        subject: SubjectScope,
    ) -> tuple[EvidenceRef, ...]:
        if not isinstance(subject, SubjectScope):
            raise TypeError("subject must be a SubjectScope")
        return tuple(
            evidence
            for evidence in self._evidence.values()
            if evidence.scope.owner == subject
        )

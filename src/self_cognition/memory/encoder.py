from self_cognition.core.ids import memory_id
from self_cognition.core.contributions import ContributionOperation
from self_cognition.core.memories import (
    MemoryCues,
    MemoryConsolidationStatus,
    MemoryLifecycleStatus,
    MemoryRecord,
    MemorySourceRef,
    MemoryType,
)
from self_cognition.core.state import StateChangeRecord, StateDecisionStatus


ENCODER_MODULE = "memory.state_change_encoder"
ENCODER_VERSION = "1"
INITIAL_SALIENCE = 0.5
INITIAL_STABILITY = 0.5
INITIAL_RETRIEVABILITY = 1.0

MEMORY_TYPE_PREFIXES = (
    ("episodic.", MemoryType.EPISODIC),
    ("procedural.", MemoryType.PROCEDURAL),
    ("relationships.", MemoryType.RELATIONSHIP),
    ("narrative.", MemoryType.NARRATIVE),
    ("profile.", MemoryType.SEMANTIC),
    ("preferences.", MemoryType.SEMANTIC),
    ("identity.", MemoryType.SEMANTIC),
    ("values.", MemoryType.SEMANTIC),
    ("semantic.", MemoryType.SEMANTIC),
)


class StateChangeMemoryEncoder:
    def encode(self, change: StateChangeRecord) -> MemoryRecord | None:
        if change.status is not StateDecisionStatus.ACCEPTED:
            return None
        if change.contribution.operation is not ContributionOperation.SET:
            return None
        memory_type = self._memory_type(change.contribution.target_field)
        if memory_type is None:
            return None

        contribution = change.contribution
        target_field = contribution.target_field
        parts = target_field.split(".")
        return MemoryRecord(
            memory_id=memory_id(contribution.contribution_id, ENCODER_VERSION),
            memory_type=memory_type,
            subject=contribution.target,
            scope=contribution.scope,
            content=contribution.value,
            evidence_refs=contribution.evidence_refs,
            confidence=contribution.confidence,
            salience=INITIAL_SALIENCE,
            stability=INITIAL_STABILITY,
            retrievability=INITIAL_RETRIEVABILITY,
            version=1,
            lifecycle_status=MemoryLifecycleStatus.ACTIVE,
            created_at=change.decided_at,
            source_module=ENCODER_MODULE,
            source_module_version=ENCODER_VERSION,
            sources=(
                MemorySourceRef(
                    contribution_id=contribution.contribution_id,
                    old_state_version=change.old_version,
                    new_state_version=change.new_version,
                    target_field=contribution.target_field,
                ),
            ),
            cues=MemoryCues(
                people=(contribution.target.subject.subject_id,),
                topics=(target_field,),
                time_keys=(change.decided_at.date().isoformat(),),
                relationships=(
                    (parts[1],)
                    if target_field.startswith("relationships.") and len(parts) > 1
                    else ()
                ),
            ),
            consolidation_status=MemoryConsolidationStatus.RAW,
            expires_at=contribution.expires_at,
        )

    @staticmethod
    def _memory_type(target_field: str) -> MemoryType | None:
        for prefix, memory_type in MEMORY_TYPE_PREFIXES:
            if target_field.startswith(prefix):
                return memory_type
        return None

    def supports(self, target_field: str) -> bool:
        return self._memory_type(target_field) is not None

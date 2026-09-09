from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from self_cognition.application.forget import ForgetService
from self_cognition.application.process_event import ProcessEventService
from self_cognition.application.results import ExportResult, ProcessEventResult
from self_cognition.application.proactive import ProactiveIntentionService
from self_cognition.core.actions import ActionDecisionPayload, ActionResultPayload
from self_cognition.core.dialogue import AssistantMessagePayload
from self_cognition.core.events import EventEnvelope
from self_cognition.core.governance import AuditAction, AuditRecord, RetentionPolicy, UserControls
from self_cognition.core.memories import MemoryRecord, MemoryType
from self_cognition.core.protocols import (
    EventStore,
    EvidenceRepository,
    GovernanceRepository,
    MemoryRepository,
    RunRepository,
    StateRepository,
)
from self_cognition.core.scopes import SubjectKind, SubjectScope
from self_cognition.cognition.registry import CognitiveModuleRegistry
from self_cognition.infrastructure.persistence.atomic_io import atomic_write_text
from self_cognition.infrastructure.persistence.serialization import event_to_dict, memory_to_dict, state_to_dict
from self_cognition.core.runs import run_to_dict
from self_cognition.runtime.run_context import RunContext
from self_cognition.core.time import SYSTEM_CLOCK


@dataclass(frozen=True, slots=True)
class Explanation:
    kind: str
    target_id: str
    reason: str
    evidence_ids: tuple[UUID, ...]
    source_event_ids: tuple[UUID, ...]


class UserControlService:
    def __init__(
        self,
        event_store: EventStore,
        evidence_repository: EvidenceRepository,
        state_repository: StateRepository,
        memory_repository: MemoryRepository,
        run_repository: RunRepository,
        governance: GovernanceRepository,
        forget: ForgetService,
        process_event: ProcessEventService,
        module_registry: CognitiveModuleRegistry,
        proactive: ProactiveIntentionService,
        export_directory: str | Path,
    ) -> None:
        self._events = event_store
        self._evidence = evidence_repository
        self._states = state_repository
        self._memories = memory_repository
        self._runs = run_repository
        self._governance = governance
        self._forget = forget
        self._process_event = process_event
        self._modules = module_registry
        self._proactive = proactive
        self._export_directory = Path(export_directory)

    def controls(self, subject: SubjectScope, *, requester: SubjectScope | None = None) -> UserControls:
        self._authorize(subject, requester)
        return self._governance.load_controls(subject)

    def explain(self, kind: str, target_id: UUID | str, subject: SubjectScope, *, requester: SubjectScope | None = None) -> Explanation:
        self._authorize(subject, requester)
        identifier = str(target_id)
        evidence_ids: list[UUID] = []
        reason = "stored evidence chain"
        if kind == "memory":
            record = self._memories.load(subject, UUID(identifier))
            if record is None:
                raise LookupError("memory does not exist")
            evidence_ids.extend(ref.evidence_id for ref in record.evidence_refs)
            evidence_ids.extend(source.contribution_id for source in record.sources)
        elif kind == "state":
            state = self._states.load(subject)
            if state is None or identifier not in state.entries:
                raise LookupError("state field does not exist")
            evidence_ids.extend(ref.evidence_id for ref in state.entries[identifier].evidence_refs)
            reason = f"state field {identifier} accepted through versioned reduction"
        else:
            event = self._event(subject, UUID(identifier))
            payload = event.payload
            refs = getattr(payload, "evidence_refs", ())
            if isinstance(payload, ActionDecisionPayload):
                refs = tuple(self._evidence_ref(subject, item) for item in payload.decision.evidence_ids)
            elif isinstance(payload, ActionResultPayload):
                refs = ()
            evidence_ids.extend(ref.evidence_id for ref in refs)
            reason = f"{kind} cites its persisted evidence references"
        source_events = tuple(item for item in evidence_ids if self._event_exists(subject, item))
        self._audit(AuditAction.READ, subject, kind, identifier, {"evidence_count": len(evidence_ids)})
        return Explanation(kind, identifier, reason, tuple(dict.fromkeys(evidence_ids)), source_events)

    def explain_answer(self, subject: SubjectScope, response_event_id: UUID, *, requester: SubjectScope | None = None) -> Explanation:
        return self.explain("answer", response_event_id, subject, requester=requester)

    def explain_memory(self, subject: SubjectScope, memory_id: UUID, *, requester: SubjectScope | None = None) -> Explanation:
        return self.explain("memory", memory_id, subject, requester=requester)

    def explain_state(self, subject: SubjectScope, field: str, *, requester: SubjectScope | None = None) -> Explanation:
        return self.explain("state", field, subject, requester=requester)

    def explain_action(self, subject: SubjectScope, action_event_id: UUID, *, requester: SubjectScope | None = None) -> Explanation:
        return self.explain("action", action_event_id, subject, requester=requester)

    def correct(
        self,
        subject: SubjectScope,
        *,
        target_field: str,
        cognition_type: str,
        value: object,
        context: RunContext,
        corrected_memory_id: UUID | None = None,
        requester: SubjectScope | None = None,
    ) -> ProcessEventResult:
        self._authorize(subject, requester)
        state = self._states.load(subject)
        causation_id = None
        if state is not None and target_field in state.entries and state.entries[target_field].evidence_refs:
            causation_id = state.entries[target_field].evidence_refs[0].evidence_id
        event = EventEnvelope.correction(
            subject,
            target_field=target_field,
            cognition_type=cognition_type,
            value=value,
            corrected_memory_id=corrected_memory_id,
            causation_id=causation_id,
            clock=context.clock,
            correlation_id=context.correlation_id,
            run_id=context.run_id,
        )
        result = self._process_event.process(event, context)
        self._audit(AuditAction.WRITE, subject, "correction", str(event.event_id), {"target_field": target_field})
        return result

    def export(self, subject: SubjectScope, *, requester: SubjectScope | None = None, export_id: UUID | None = None) -> ExportResult:
        self._authorize(subject, requester)
        events = self._events.read_by_mind(subject.mind) if subject.subject.kind is SubjectKind.MIND else self._events.read_by_subject(subject)
        memories = self._memories.read_by_mind(subject.mind) if subject.subject.kind is SubjectKind.MIND else self._memories.read_by_subject(subject)
        states = self._states_for(subject, events, memories)
        runs = self._runs.read_by_subject(subject)
        audits = self._governance.read_audit(subject)
        evidence = self._evidence_for(subject, events, memories, states)
        payload = {
            "schema_version": 1,
            "export_id": str(export_id or uuid5(NAMESPACE_URL, f"export:{subject.mind.mind_id}:{subject.subject.kind.value}:{subject.subject.subject_id}")),
            "subject": {"mind_id": subject.mind.mind_id, "kind": subject.subject.kind.value, "subject_id": subject.subject.subject_id},
            "events": [event_to_dict(item) for item in events],
            "evidence": evidence,
            "states": [state_to_dict(item) for item in states],
            "memories": [memory_to_dict(item) for item in memories],
            "relationships": [entry.value for state in states for entry in state.entries.values() if isinstance(entry.value, dict) and entry.value.get("kind") == "relationship"],
            "runs": [run_to_dict(item) for item in runs],
            "audit": [item.to_dict(redacted=True) for item in audits],
        }
        resolved_id = UUID(payload["export_id"])
        self._export_directory.mkdir(parents=True, exist_ok=True)
        path = self._export_directory / f"{resolved_id}.json"
        atomic_write_text(path, json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        self._audit(AuditAction.EXPORT, subject, "export", str(resolved_id), {"path": str(path.name)})
        return ExportResult(resolved_id, subject, path, {key: len(value) for key, value in payload.items() if isinstance(value, list)})

    def export_subject(self, subject: SubjectScope, *, requester: SubjectScope | None = None, export_id: UUID | None = None) -> ExportResult:
        return self.export(subject, requester=requester, export_id=export_id)

    def set_retention(self, subject: SubjectScope, *, retention_days: int, memory_types: tuple[MemoryType, ...] = (), conversation_id: str | None = None, now: datetime, requester: SubjectScope | None = None) -> RetentionPolicy:
        self._authorize(subject, requester)
        policy = RetentionPolicy(uuid5(NAMESPACE_URL, f"retention:{subject.mind.mind_id}:{subject.subject.subject_id}:{retention_days}:{memory_types}:{conversation_id}"), subject, retention_days, memory_types, conversation_id)
        self._governance.save_retention(policy)
        self._audit(AuditAction.CONTROL, subject, "retention", str(policy.policy_id), {"retention_days": retention_days})
        return policy

    def apply_retention(self, policy: RetentionPolicy, *, now: datetime, requester: SubjectScope | None = None):
        self._authorize(policy.subject, requester)
        cutoff = now - timedelta(days=policy.retention_days)
        from self_cognition.core.deletions import DeletionSelector
        plan = self._forget.dry_run(DeletionSelector(policy.subject, memory_types=policy.memory_types, created_to=cutoff, conversation_id=policy.conversation_id), now=now)
        return self._forget.execute(plan, now=now)

    def correct_cognition(self, *args: object, **kwargs: object) -> ProcessEventResult:
        return self.correct(*args, **kwargs)

    def forget_subject(self, subject: SubjectScope, *, now: datetime, requester: SubjectScope | None = None):
        self._authorize(subject, requester)
        from self_cognition.core.deletions import DeletionSelector

        plan = self._forget.dry_run(DeletionSelector(subject, delete_subject=True), now=now)
        return self._forget.execute(plan, now=now)

    def disable_module(self, subject: SubjectScope, module_id: str, *, requester: SubjectScope | None = None) -> UserControls:
        self._authorize(subject, requester)
        self._modules.disable(module_id)
        controls = self._update(subject, "disabled_modules", module_id, True)
        self._audit(AuditAction.CONTROL, subject, "module", module_id, {"enabled": False})
        return controls

    def disable_model_provider(self, subject: SubjectScope, provider_id: str, *, requester: SubjectScope | None = None) -> UserControls:
        self._authorize(subject, requester)
        controls = self._update(subject, "disabled_model_providers", provider_id, True)
        self._audit(AuditAction.CONTROL, subject, "model_provider", provider_id, {"enabled": False})
        return controls

    def disable_proactive_task(self, subject: SubjectScope, task_id: str, *, requester: SubjectScope | None = None) -> UserControls:
        self._authorize(subject, requester)
        controls = self._update(subject, "disabled_proactive_tasks", task_id, True)
        self._cancel_intentions(subject, lambda item: item.motive.kind == task_id, "proactive task disabled")
        return controls

    def disable_proactive_channel(self, subject: SubjectScope, channel_id: str = "default", *, requester: SubjectScope | None = None) -> UserControls:
        self._authorize(subject, requester)
        controls = self._update(subject, "disabled_proactive_channels", channel_id, True)
        if channel_id == "default":
            self._cancel_intentions(subject, lambda item: True, "proactive channel disabled")
        return controls

    def set_proactive_frequency(self, subject: SubjectScope, limit_per_hour: int | None, *, requester: SubjectScope | None = None) -> UserControls:
        self._authorize(subject, requester)
        controls = replace(self._governance.load_controls(subject), proactive_frequency_per_hour=limit_per_hour)
        self._governance.save_controls(controls)
        if limit_per_hour == 0:
            self._cancel_intentions(subject, lambda item: True, "proactive frequency disabled")
        self._audit(AuditAction.CONTROL, subject, "proactive_frequency", str(limit_per_hour), {})
        return controls

    def _update(self, subject: SubjectScope, field: str, value: str, enabled: bool) -> UserControls:
        controls = self._governance.load_controls(subject)
        values = set(getattr(controls, field))
        (values.add(value) if enabled else values.discard(value))
        updated = replace(controls, **{field: frozenset(values)})
        self._governance.save_controls(updated)
        return updated

    def _cancel_intentions(self, subject: SubjectScope, predicate, reason: str) -> None:
        for intention in self._proactive.active(subject, as_of=SYSTEM_CLOCK.now()):
            if predicate(intention):
                self._proactive.cancel(subject, intention.intention_id, reason)

    def _states_for(self, subject: SubjectScope, events: tuple[EventEnvelope, ...], memories: tuple[MemoryRecord, ...]):
        scopes = {item.subject for item in events} | {item.subject for item in memories} | {subject}
        return tuple(state for scope in scopes if (state := self._states.load(scope)) is not None)

    def _evidence_for(self, subject, events, memories, states):
        refs = {}
        for event in events:
            persisted = self._evidence.get(event.subject, event.event_id)
            if persisted is not None:
                refs[persisted.evidence_id] = persisted
            for ref in getattr(event.payload, "evidence_refs", ()):
                refs[ref.evidence_id] = ref
        for memory in memories:
            for ref in memory.evidence_refs:
                refs[ref.evidence_id] = ref
        for state in states:
            for atom in state.entries.values():
                for ref in atom.evidence_refs:
                    refs[ref.evidence_id] = ref
        return [self._evidence_dict(ref) for ref in refs.values()]

    def _event(self, subject, event_id):
        for event in self._events.read_by_subject(subject):
            if event.event_id == event_id:
                return event
        raise LookupError("event does not exist")

    def _event_exists(self, subject, event_id):
        return any(item.event_id == event_id for item in self._events.read_by_subject(subject))

    def _evidence_ref(self, subject, evidence_id):
        ref = self._evidence.get(subject, evidence_id)
        if ref is None:
            raise LookupError("evidence does not exist")
        return ref

    @staticmethod
    def _evidence_dict(ref):
        return {"evidence_id": str(ref.evidence_id), "source_kind": ref.source_kind.value, "source_ref": ref.source_ref, "locator": ref.locator, "excerpt": ref.excerpt, "observed_at": ref.observed_at.isoformat() if ref.observed_at else None, "reliability": ref.reliability}

    def _audit(self, action: AuditAction, subject: SubjectScope, target_type: str, target_id: str, details: dict[str, object]) -> None:
        self._governance.append_audit(AuditRecord(uuid5(NAMESPACE_URL, f"audit:{action.value}:{subject.mind.mind_id}:{target_type}:{target_id}:{len(self._governance.read_audit(subject))}"), action, subject, subject, target_type, target_id, SYSTEM_CLOCK.now(), details))

    @staticmethod
    def _authorize(subject, requester):
        if requester is not None and requester != subject:
            raise PermissionError("requester is not authorized for this subject")

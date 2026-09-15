from __future__ import annotations

from datetime import datetime

from self_cognition.application.results import ProcessEventResult
from self_cognition.application.user_control import UserControlService
from self_cognition.core.affect import (
    AffectAssessment,
    EmotionState,
    MoodState,
    decay_assessment,
    decay_emotion,
    decay_mood,
)
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.protocols import StateRepository
from self_cognition.core.scopes import SubjectScope
from self_cognition.runtime.run_context import RunContext


class AffectViewService:
    """Read-only projection of active emotion, affect assessment and mood."""

    def __init__(self, state_repository: StateRepository) -> None:
        self._states = state_repository

    def view(self, subject: SubjectScope, *, as_of: datetime) -> dict[str, object]:
        entries, version = self._entries(subject)
        emotions: list[dict[str, object]] = []
        mood: dict[str, object] | None = None
        for field_name, entry in sorted(entries.items()):
            if field_name.startswith("affect.reaction."):
                content = self._reaction(entry.value, as_of)
                kind = "reaction"
            elif field_name.startswith("affect.current."):
                content = decay_assessment(entry.value, as_of)
                kind = "assessment"
            elif field_name.startswith("mood."):
                content = self._mood(entry.value, as_of)
                kind = "mood"
            else:
                continue
            if not isinstance(content, dict):
                continue
            item = {
                "field": field_name,
                "kind": kind,
                "content": content,
                "confidence": getattr(entry, "confidence", None),
                "evidence_ids": [
                    str(ref.evidence_id)
                    for ref in getattr(entry, "evidence_refs", ())
                ],
            }
            if field_name.startswith("mood."):
                mood = item
            else:
                emotions.append(item)
        return {
            "as_of": as_of.isoformat(),
            "state_version": version,
            "emotions": emotions,
            "mood": mood,
        }

    def _entries(self, subject: SubjectScope) -> tuple[dict[str, object], int]:
        entries: dict[str, object] = {}
        version = 0
        state = self._states.load(subject)
        if state is not None:
            entries.update(state.entries)
            version = state.version
        mind_scope = SubjectScope.for_mind(subject.mind.mind_id)
        if mind_scope != subject:
            mind_state = self._states.load(mind_scope)
            if mind_state is not None:
                for field_name, entry in mind_state.entries.items():
                    if (
                        field_name.startswith(("affect.", "mood."))
                        and field_name not in entries
                    ):
                        entries[field_name] = entry
                version = max(version, mind_state.version)
        return entries, version

    @staticmethod
    def _reaction(value: object, as_of: datetime) -> dict[str, object] | None:
        try:
            state = EmotionState.from_state_value(value)
        except ContractValidationError:
            return None
        decayed = decay_emotion(state, as_of)
        return None if decayed is None else decayed.to_state_value()

    @staticmethod
    def _mood(value: object, as_of: datetime) -> dict[str, object] | None:
        try:
            state = MoodState.from_state_value(value)
        except ContractValidationError:
            return None
        decayed = decay_mood(state, as_of)
        return None if decayed is None else decayed.to_state_value()


AFFECT_MODULE_IDS = ("affect.fast_reaction", "affect.affect_extractor")


class AffectControlService:
    """Validated user controls for emotion, mood and affect modules."""

    def __init__(self, user_control: UserControlService) -> None:
        self._user_control = user_control

    def correct(
        self,
        subject: SubjectScope,
        *,
        field: str,
        value: object,
        context: RunContext,
        requester: SubjectScope | None = None,
    ) -> ProcessEventResult:
        validated = self._validate_value(field, value)
        return self._user_control.correct(
            subject,
            target_field=field,
            cognition_type="affect",
            value=validated,
            context=context,
            requester=requester,
        )

    def close(
        self,
        subject: SubjectScope,
        *,
        requester: SubjectScope | None = None,
    ) -> object:
        controls: object = None
        for module_id in AFFECT_MODULE_IDS:
            controls = self._user_control.disable_module(
                subject,
                module_id,
                requester=requester,
            )
        return controls

    def open(
        self,
        subject: SubjectScope,
        *,
        requester: SubjectScope | None = None,
    ) -> object:
        controls: object = None
        for module_id in AFFECT_MODULE_IDS:
            controls = self._user_control.enable_module(
                subject,
                module_id,
                requester=requester,
            )
        return controls

    def export(
        self,
        subject: SubjectScope,
        *,
        requester: SubjectScope | None = None,
    ):
        return self._user_control.export(subject, requester=requester)

    def delete(
        self,
        subject: SubjectScope,
        *,
        now,
        requester: SubjectScope | None = None,
    ):
        return self._user_control.forget_subject(
            subject,
            now=now,
            requester=requester,
        )

    @staticmethod
    def _validate_value(field: str, value: object) -> dict[str, object]:
        if field.startswith("affect.reaction."):
            return EmotionState.from_state_value(value).to_state_value()
        if field.startswith("affect.current."):
            return AffectAssessment.from_state_value(value).to_state_value()
        if field.startswith("mood."):
            return MoodState.from_state_value(value).to_state_value()
        raise ContractValidationError(
            "target field is not an emotion or mood field"
        )

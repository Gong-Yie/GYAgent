from self_cognition.core.contributions import Contribution
from self_cognition.core.events import Event
from self_cognition.core.ids import contribution_id


SOURCE_MODULE = "metacognition.conflict_extractor"
PREFERENCE_FIELD = "preferences.study_time"
UNCERTAINTY_FIELD = "metacognition.uncertainties.preferences.study_time"
CONFLICT_FIELD = "metacognition.conflicts.preferences.study_time"
UNCERTAIN_STATEMENT = "我不确定更喜欢早上还是晚上学习"
CONFLICT_STATEMENT = "我既喜欢早上学习，也喜欢晚上学习"


class ConflictMetacognitionExtractor:
    """Extracts the first explicit uncertainty and contradiction rules."""

    subscriptions = frozenset({"user.message"})

    def process(self, event: Event) -> tuple[Contribution, ...]:
        if event.content == UNCERTAIN_STATEMENT:
            return (self._contribution(event, UNCERTAINTY_FIELD, "早上或晚上"),)
        if event.content != CONFLICT_STATEMENT:
            return ()

        return (
            self._contribution(event, PREFERENCE_FIELD, "早上"),
            self._contribution(event, PREFERENCE_FIELD, "晚上"),
            self._contribution(event, CONFLICT_FIELD, "早上和晚上"),
        )

    @staticmethod
    def _contribution(
        event: Event,
        target_field: str,
        value: str,
    ) -> Contribution:
        return Contribution(
            contribution_id=contribution_id(
                event.event_id,
                SOURCE_MODULE,
                f"{target_field}:{value}",
            ),
            target_subject_id=event.actor_id,
            target_field=target_field,
            value=value,
            confidence=1.0,
            evidence_event_ids=(event.event_id,),
            source_event_id=event.event_id,
            source_module=SOURCE_MODULE,
        )

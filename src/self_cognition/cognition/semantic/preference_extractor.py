from self_cognition.core.contributions import Contribution
from self_cognition.core.events import Event
from self_cognition.core.ids import contribution_id


SOURCE_MODULE = "semantic.preference_extractor"
TARGET_FIELD = "preferences.study_time"
STUDY_TIME_PREFERENCES = {
    "喜欢晚上学习": "晚上",
    "喜欢早上学习": "早上",
}


class PreferenceExtractor:
    subscriptions = frozenset({"user.message"})

    def process(self, event: Event) -> tuple[Contribution, ...]:
        matched_values = {
            preference_value
            for phrase, preference_value in STUDY_TIME_PREFERENCES.items()
            if phrase in event.content
        }
        if len(matched_values) != 1:
            return ()
        value = matched_values.pop()

        return (
            Contribution(
                contribution_id=contribution_id(
                    event.event_id,
                    SOURCE_MODULE,
                    TARGET_FIELD,
                ),
                target_subject_id=event.actor_id,
                target_field=TARGET_FIELD,
                value=value,
                confidence=1.0,
                evidence_event_ids=(event.event_id,),
                source_event_id=event.event_id,
                source_module=SOURCE_MODULE,
            ),
        )

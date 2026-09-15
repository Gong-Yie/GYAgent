import re

from self_cognition.core.cognition import CognitionRequest
from self_cognition.core.contributions import CognitiveContribution, CognitionType
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope
from self_cognition.core.ids import contribution_id


SOURCE_MODULE = "semantic.preference_extractor"
MODULE_VERSION = "2"
TARGET_FIELD = "preferences.study_time"
STUDY_TIME_PATTERN = re.compile(r"(早上|早晨|晚上|夜里|夜间)(?:学习|读书|学|了|$)")
STUDY_TIME_VALUES = {
    "早上": "早上",
    "早晨": "早上",
    "晚上": "晚上",
    "夜里": "晚上",
    "夜间": "晚上",
}

QUESTION_OR_QUERY_PATTERN = re.compile(
    r"[?？]"
    r"|什么时候|几点|多久|是否|是不是|哪一种|哪个|哪些"
    r"|(?:请|麻烦|你能|你能否).{0,10}(?:说明|描述|介绍|告诉我).{0,30}(?:偏好|喜欢)"
    r"|(?:偏好|喜欢).{0,20}(?:是什么|有哪些|怎么样|吗|呢)"
)
UNCERTAINTY_PATTERN = re.compile(r"不确定|不清楚|不知道|拿不准|说不准")
CHANGE_PATTERN = re.compile(r"改成|改为|变成|现在是|现在喜欢|如今|最近|后来|不再")
NEGATION_PREFIXES = ("不喜欢", "不再喜欢", "不想", "并不喜欢", "不太喜欢")


class PreferenceExtractor:
    subscriptions = frozenset({"user.message"})
    module_id = SOURCE_MODULE
    module_version = MODULE_VERSION
    deterministic = True

    def run(
        self,
        request: CognitionRequest,
    ) -> tuple[CognitiveContribution, ...]:
        return self.process(request.event)

    def process(self, event: EventEnvelope) -> tuple[CognitiveContribution, ...]:
        text = event.payload.text
        matches = self._positive_matches(text)
        if not matches:
            return ()
        if QUESTION_OR_QUERY_PATTERN.search(text) or UNCERTAINTY_PATTERN.search(text):
            return ()

        value = self._current_value(text, matches)
        if value is None:
            return ()

        return (
            CognitiveContribution.set_from_event(
                event,
                contribution_id=contribution_id(
                    event.event_id,
                    SOURCE_MODULE,
                    TARGET_FIELD,
                ),
                target_field=TARGET_FIELD,
                cognition_type=CognitionType.PREFERENCE,
                value=value,
                confidence=1.0,
                evidence_refs=(EvidenceRef.for_event(event),),
                source_module=SOURCE_MODULE,
                module_version=MODULE_VERSION,
            ),
        )

    @staticmethod
    def _positive_matches(text: str) -> list[re.Match[str]]:
        matches: list[re.Match[str]] = []
        for match in STUDY_TIME_PATTERN.finditer(text):
            prefix = text[max(0, match.start() - 6) : match.start()]
            if any(prefix.endswith(marker) for marker in NEGATION_PREFIXES):
                continue
            matches.append(match)
        return matches

    @staticmethod
    def _current_value(text: str, matches: list[re.Match[str]]) -> str | None:
        values = [STUDY_TIME_VALUES[match.group(1)] for match in matches]
        if len(set(values)) == 1:
            return values[0]
        if CHANGE_PATTERN.search(text):
            return values[-1]
        return None
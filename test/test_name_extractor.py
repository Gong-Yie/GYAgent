from uuid import NAMESPACE_URL, uuid5

from self_cognition.cognition.semantic.name_extractor import NameExtractor
from self_cognition.core.contributions import Contribution
from self_cognition.core.events import Event


def test_extracts_supported_name_statement():
    event = Event.user_message("user-1", "我叫小明")

    contributions = NameExtractor().process(event)

    assert len(contributions) == 1
    contribution = contributions[0]
    assert isinstance(contribution, Contribution)
    assert contribution.contribution_id == uuid5(
        NAMESPACE_URL,
        f"{event.event_id}:semantic.name_extractor:profile.name",
    )
    assert contribution.target_subject_id == "user-1"
    assert contribution.target_field == "profile.name"
    assert contribution.value == "小明"
    assert contribution.evidence_event_ids == (event.event_id,)
    assert contribution.source_event_id == event.event_id
    assert contribution.source_module == "semantic.name_extractor"


def test_ignores_unsupported_name_statement():
    event = Event.user_message("user-1", "我叫小红")

    assert NameExtractor().process(event) == ()

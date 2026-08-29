from collections.abc import Callable

import pytest

from self_cognition.cognition.semantic.name_extractor import NameExtractor
from self_cognition.cognition.semantic.preference_extractor import (
    PreferenceExtractor,
)
from self_cognition.core.contributions import Contribution
from self_cognition.core.events import Event
from self_cognition.core.protocols import CognitiveModule


@pytest.mark.parametrize(
    ("module_factory", "content", "target_field", "expected_value"),
    [
        (
            PreferenceExtractor,
            "我喜欢晚上学习",
            "preferences.study_time",
            "晚上",
        ),
        (NameExtractor, "我叫小明", "profile.name", "小明"),
    ],
)
def test_cognitive_module_contract(
    module_factory: Callable[[], CognitiveModule],
    content: str,
    target_field: str,
    expected_value: str,
):
    module = module_factory()
    event = Event.user_message("user-1", content)

    first = module.process(event)
    second = module.process(event)

    assert isinstance(module, CognitiveModule)
    assert module.subscriptions == frozenset({"user.message"})
    assert isinstance(first, tuple)
    assert first == second
    assert len(first) == 1
    assert isinstance(first[0], Contribution)
    assert first[0].target_field == target_field
    assert first[0].value == expected_value
    assert first[0].source_event_id == event.event_id


@pytest.mark.parametrize("module", [PreferenceExtractor(), NameExtractor()])
def test_cognitive_module_returns_empty_tuple_for_unrelated_event(
    module: CognitiveModule,
):
    event = Event.user_message("user-1", "今天天气很好")

    assert module.process(event) == ()

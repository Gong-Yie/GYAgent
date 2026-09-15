from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from self_cognition.cognition.semantic.llm_extractor import LLMSemanticExtractor
from self_cognition.core.events import Event
from self_cognition.infrastructure.llm.openai_responses import (
    OpenAIResponsesCognitionModel,
)
from self_cognition.runtime.run_context import RunContext
from self_cognition.settings import DotenvSecretSource


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SECRETS = DotenvSecretSource(_PROJECT_ROOT / ".env")
_LIVE_API_KEY = _SECRETS.get("OPENAI_API_KEY")
_LIVE_MODEL = _SECRETS.get("OPENAI_MODEL")
_LIVE_BASE_URL = _SECRETS.get("OPENAI_BASE_URL")


@pytest.mark.live_openai
@pytest.mark.skipif(
    not _LIVE_API_KEY or not _LIVE_MODEL,
    reason="requires OPENAI_API_KEY and OPENAI_MODEL in process env or .env",
)
def test_real_openai_preference_extraction():
    model = OpenAIResponsesCognitionModel.from_api_key(
        _LIVE_API_KEY,
        _LIVE_MODEL,
        base_url=_LIVE_BASE_URL,
        timeout_seconds=30,
        max_output_tokens=2048,
    )
    event = Event.user_message("user-1", "我喜欢晚上学习")
    context = RunContext(
        run_id=uuid4(),
        correlation_id=uuid4(),
        deadline=datetime.now(timezone.utc) + timedelta(seconds=45),
    )

    contributions = LLMSemanticExtractor(model).process(event, context)

    assert len(contributions) == 1
    assert contributions[0].target_field == "preferences.study_time"
    assert contributions[0].value == "晚上"
@pytest.mark.live_openai
@pytest.mark.skipif(
    not _LIVE_API_KEY or not _LIVE_MODEL,
    reason="requires OPENAI_API_KEY and OPENAI_MODEL in process env or .env",
)
def test_real_openai_natural_language_preference_update():
    model = OpenAIResponsesCognitionModel.from_api_key(
        _LIVE_API_KEY,
        _LIVE_MODEL,
        base_url=_LIVE_BASE_URL,
        timeout_seconds=30,
        max_output_tokens=4096,
    )
    event = Event.user_message(
        "user-1",
        "我以前喜欢晚上学习，但最近改成早上了",
    )
    context = RunContext(
        run_id=uuid4(),
        correlation_id=uuid4(),
        deadline=datetime.now(timezone.utc) + timedelta(seconds=45),
    )

    contributions = LLMSemanticExtractor(model).process(event, context)

    assert len(contributions) == 1
    assert contributions[0].target_field == "preferences.study_time"
    assert contributions[0].value == "早上"
import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from self_cognition.cognition.semantic.llm_extractor import LLMSemanticExtractor
from self_cognition.core.events import Event
from self_cognition.infrastructure.llm.openai_responses import (
    OpenAIResponsesCognitionModel,
)
from self_cognition.runtime.run_context import RunContext


@pytest.mark.live_openai
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_MODEL"),
    reason="requires OPENAI_API_KEY and OPENAI_MODEL",
)
def test_real_openai_preference_extraction():
    model = OpenAIResponsesCognitionModel.from_api_key(
        os.environ["OPENAI_API_KEY"],
        os.environ["OPENAI_MODEL"],
        timeout_seconds=30,
        max_output_tokens=256,
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

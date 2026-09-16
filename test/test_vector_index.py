from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from self_cognition.bootstrap import build_container
from self_cognition.core.events import EventEnvelope
from self_cognition.core.scopes import MindScope, SubjectScope
from self_cognition.core.time import SYSTEM_CLOCK
from self_cognition.runtime.run_context import RunContext
from self_cognition.settings import ApplicationSettings


def _process(container, subject: SubjectScope, text: str) -> None:
    context = RunContext(
        uuid4(),
        uuid4(),
        SYSTEM_CLOCK.now() + timedelta(minutes=5),
    )
    event = EventEnvelope.user_message(
        subject,
        text,
        run_id=context.run_id,
        correlation_id=context.correlation_id,
    )
    result = container.process_event.process(event, context)
    assert result.status.value == "succeeded"


def test_vector_index_rebuild_delete_search_and_deletion_propagation(
    tmp_path: Path,
) -> None:
    container = build_container(
        tmp_path,
        settings=ApplicationSettings(data_dir=tmp_path, worker_enabled=False),
        dotenv_path=tmp_path / "missing.env",
    )
    subject = SubjectScope.legacy_user("vector-user")
    _process(container, subject, "我喜欢晚上学习")
    mind = MindScope(subject.mind.mind_id)
    memory = container.memory_repository.read_by_subject(subject)[0]

    rebuilt = container.vector_index.rebuild_all()
    assert len(rebuilt) == 1
    assert container.vector_index.verify()

    matches = container.vector_index.search(mind, "晚上学习", limit=5)
    assert any(match.memory_id == memory.memory_id for match in matches)
    assert all(match.source_ref == f"memory:{match.memory_id}" for match in matches)

    container.vector_index.delete_all()
    assert container.vector_index.load(mind) is None
    assert not container.vector_index.verify()
    assert len(container.vector_index.rebuild_all()) == 1
    assert container.vector_index.verify()

    container.memory_repository.delete(subject, (memory.memory_id,))
    assert not container.vector_index.verify()
    matches_after = container.vector_index.search(mind, "晚上学习", limit=5)
    assert all(match.memory_id != memory.memory_id for match in matches_after)
    assert container.vector_index.verify()

    other_mind = MindScope("other-mind")
    assert container.vector_index.search(other_mind, "晚上学习") == ()
    assert container.vector_index.load(other_mind) is None

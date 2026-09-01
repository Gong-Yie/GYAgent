import hashlib
import json

import pytest

from self_cognition.core.errors import (
    ContractValidationError,
    MalformedSerializedDataError,
    ScopeMismatchError,
)
from self_cognition.core.indexes import WorkspaceIndex
from self_cognition.core.scopes import (
    DEFAULT_MIND_ID,
    ConversationScope,
    DataScope,
    DisclosureScope,
    MindScope,
    SubjectKind,
    SubjectRef,
    SubjectScope,
)
from self_cognition.core.state import SubjectState
from self_cognition.infrastructure.persistence.file_state_repository import (
    FileStateRepository,
)
from self_cognition.infrastructure.persistence.in_memory_state_repository import (
    InMemoryStateRepository,
)
from self_cognition.infrastructure.persistence.serialization import (
    state_from_json,
    state_to_json,
)


def make_subject(
    mind_id: str,
    subject_id: str,
    kind: SubjectKind = SubjectKind.USER,
) -> SubjectScope:
    return SubjectScope(MindScope(mind_id), SubjectRef(kind, subject_id))


def make_stored_state(scope: SubjectScope) -> SubjectState:
    return SubjectState(
        subject_id=scope.subject.subject_id,
        version=1,
        entries={},
        applied_contribution_ids=frozenset(),
        conflicts=frozenset(),
        mind_id=scope.mind.mind_id,
        subject_kind=scope.subject.kind,
    )


def test_disclosure_scopes_describe_intent_without_splitting_the_mind():
    owner = make_subject("mind-1", "user-a")
    user_b = make_subject("mind-1", "user-b")
    mind = make_subject("mind-1", "mind-1", SubjectKind.MIND)
    conversation = ConversationScope("conversation-1", group_id="group-1")

    private = DataScope(owner, DisclosureScope.PRIVATE)
    shared_conversation = DataScope(
        owner,
        DisclosureScope.CONVERSATION,
        conversation,
    )
    shared_group = DataScope(owner, DisclosureScope.GROUP, conversation)
    internal = DataScope(mind, DisclosureScope.MIND)

    assert private.matches_disclosure_intent(owner) is True
    assert private.matches_disclosure_intent(user_b) is False
    assert shared_conversation.matches_disclosure_intent(
        user_b,
        conversation,
    ) is True
    assert shared_group.matches_disclosure_intent(
        user_b,
        ConversationScope("conversation-2", group_id="group-1"),
    ) is True
    assert internal.matches_disclosure_intent(mind) is True
    assert internal.matches_disclosure_intent(user_b) is False


def test_cross_mind_scope_is_rejected_and_invalid_contexts_are_rejected():
    owner = make_subject("mind-1", "user-a")
    foreign_user = make_subject("mind-2", "user-a")

    with pytest.raises(ScopeMismatchError):
        DataScope(owner, DisclosureScope.PRIVATE).matches_disclosure_intent(
            foreign_user
        )
    with pytest.raises(ContractValidationError):
        DataScope(owner, DisclosureScope.CONVERSATION)
    with pytest.raises(ContractValidationError):
        DataScope(
            owner,
            DisclosureScope.GROUP,
            ConversationScope("conversation-1"),
        )


@pytest.mark.parametrize("repository_kind", ["memory", "file"])
def test_state_repositories_isolate_minds_and_subject_kinds(
    repository_kind: str,
    tmp_path,
):
    repository = (
        InMemoryStateRepository()
        if repository_kind == "memory"
        else FileStateRepository(tmp_path / "states")
    )
    first_user = make_subject("mind-1", "shared-id")
    second_user = make_subject("mind-2", "shared-id")
    group = make_subject("mind-1", "shared-id", SubjectKind.GROUP)

    for scope in (first_user, second_user, group):
        repository.save(make_stored_state(scope), expected_version=0)

    assert repository.load(first_user) == make_stored_state(first_user)
    assert repository.load(second_user) == make_stored_state(second_user)
    assert repository.load(group) == make_stored_state(group)
    assert repository.load("shared-id") is None


def test_legacy_state_maps_to_default_mind_and_keeps_its_file_name(tmp_path):
    state = make_stored_state(SubjectScope.legacy_user("user-1"))
    legacy_data = json.loads(state_to_json(state))
    legacy_data["schema_version"] = 1
    del legacy_data["mind_id"]
    del legacy_data["subject_kind"]
    del legacy_data["changes"]
    legacy_payload = json.dumps(legacy_data, ensure_ascii=False)

    restored = state_from_json(legacy_payload)

    assert restored.mind_id == DEFAULT_MIND_ID
    assert restored.subject_kind is SubjectKind.USER
    state_directory = tmp_path / "states"
    state_directory.mkdir()
    digest = hashlib.sha256(b"user-1").hexdigest()
    (state_directory / f"{digest}.json").write_text(
        legacy_payload,
        encoding="utf-8",
    )
    assert FileStateRepository(state_directory).load("user-1") == restored


def test_scoped_state_json_rejects_blank_mind_id():
    data = json.loads(
        state_to_json(make_stored_state(make_subject("mind-1", "user-1")))
    )
    data["mind_id"] = "  "

    with pytest.raises(MalformedSerializedDataError):
        state_from_json(json.dumps(data))


def test_workspace_index_rejects_same_subject_from_another_mind():
    first = make_stored_state(make_subject("mind-1", "user-1"))
    second = make_stored_state(make_subject("mind-2", "user-1"))

    assert WorkspaceIndex.build(first).is_compatible(second) is False

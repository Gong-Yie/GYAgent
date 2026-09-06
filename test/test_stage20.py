import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4, uuid5

import pytest

from self_cognition.application.results import ProcessEventStatus
from self_cognition.bootstrap import ApplicationContainer, build_container
from self_cognition.core.deletions import DeletionSelector, DeletionStatus
from self_cognition.core.dialogue import (
    AssistantMessagePayload,
    ClaimStance,
    DialogueClaim,
    DialogueContextPayload,
    DialogueDraft,
    DialogueModelOutput,
    DialogueRequest,
    DisclosureDecision,
    draft_from_dict,
    draft_to_dict,
    parse_model_json,
)
from self_cognition.core.errors import ContractValidationError, ModelOutputError
from self_cognition.core.events import EventEnvelope, SelfModelObservationPayload
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.identity import SelfModelAspect
from self_cognition.core.scopes import (
    ConversationScope,
    DisclosureScope,
    MindScope,
    SubjectKind,
    SubjectRef,
    SubjectScope,
)
from self_cognition.core.workspace import (
    RetrievalBudget,
    WorkspacePacket,
    estimate_tokens,
    workspace_model_context,
)
from self_cognition.executive.dialogue.fake import RuleDialogueAdapter
from self_cognition.executive.dialogue.grounding import validate_grounding
from self_cognition.executive.dialogue.rule_based import RuleBasedDialogueModel
from self_cognition.infrastructure.llm.dialogue_responses import (
    OpenAIResponsesDialogueModel,
)
from self_cognition.infrastructure.persistence.in_memory_event_store import (
    InMemoryEventStore,
)
from self_cognition.infrastructure.persistence.serialization import (
    event_from_json,
    event_to_json,
)
from self_cognition.runtime.run_context import RunContext

NOW = datetime(2026, 9, 6, 8, tzinfo=timezone.utc)
QUESTION = "我喜欢什么时候学习？"


@dataclass
class FixedClock:
    value: datetime = NOW

    def now(self) -> datetime:
        return self.value


def context() -> RunContext:
    return RunContext(uuid4(), uuid4(), NOW + timedelta(hours=1), clock=FixedClock())


def subject(name: str = "alice", mind: str = "mind-20") -> SubjectScope:
    return SubjectScope(MindScope(mind), SubjectRef(SubjectKind.USER, name))


def message(
    text: str = QUESTION, name: str = "alice", mind: str = "mind-20"
) -> EventEnvelope:
    return EventEnvelope.user_message(subject(name, mind), text, clock=FixedClock())


def seed(container: ApplicationContainer, event: EventEnvelope) -> EventEnvelope:
    result = container.process_event.process(event, context())
    assert result.status is ProcessEventStatus.SUCCEEDED
    return next(
        stored
        for stored in container.event_store.read_by_subject(event.subject)
        if stored.event_id == event.event_id
    )


class CountingModel(RuleDialogueAdapter):
    def __init__(self, problem: str = "") -> None:
        super().__init__(RuleBasedDialogueModel())
        self.problem = problem
        self.calls: list[str] = []
        self.workspaces: list[WorkspacePacket] = []

    def generate(
        self, workspace: WorkspacePacket, run: RunContext
    ) -> DialogueModelOutput:
        self.calls.append("generate")
        self.workspaces.append(workspace)
        if self.problem == "crash":
            raise KeyboardInterrupt("simulated process interruption")
        if self.problem == "no_calls":
            raise AssertionError("persisted requests must not invoke the model")
        output = super().generate(workspace, run)
        if self.problem == "malformed":
            return replace(output, raw_output="invalid-original-model-output")
        if self.problem == "cancel":
            run.cancel()
        if self.problem == "timeout":
            run.clock.value = run.deadline
        if self.problem in {"foreign_citation", "unsupported", "hidden_claim"}:
            draft = draft_from_dict(parse_model_json(output.raw_output))
            claim = draft.claims[0]
            if self.problem == "foreign_citation":
                draft = replace(
                    draft, claims=(replace(claim, evidence_ids=(uuid4(),)),)
                )
            elif self.problem == "unsupported":
                text = "你喜欢凌晨三点学习。"
                draft = replace(draft, text=text, claims=(replace(claim, text=text),))
            else:
                draft = replace(draft, text="你已经获得诺贝尔奖。", claims=())
            return replace(
                output, raw_output=json.dumps(draft_to_dict(draft), ensure_ascii=False)
            )
        return output

    def review(
        self,
        workspace: WorkspacePacket,
        draft: DialogueDraft,
        run: RunContext,
    ) -> DialogueModelOutput:
        self.calls.append("review")
        if self.problem == "bad_review":
            return DialogueModelOutput(
                "fake", "bad-review", '{"supported":"yes","reason":"bad"}'
            )
        return super().review(workspace, draft, run)


class SharedModel(CountingModel):
    def generate(
        self, workspace: WorkspacePacket, run: RunContext
    ) -> DialogueModelOutput:
        self.calls.append("generate")
        self.workspaces.append(workspace)
        claims = tuple(
            DialogueClaim(
                f"{item.subject.subject.subject_id}: {item.content}。",
                tuple(ref.evidence_id for ref in item.evidence_refs),
                ClaimStance.SUPPORTED,
            )
            for item in workspace.items
            if item.target_field in {"preferences.study_time", "identity.role"}
        )
        draft = DialogueDraft(
            "".join(claim.text for claim in claims),
            claims,
            DisclosureDecision(
                "disclose",
                DisclosureScope.GROUP,
                "Fake value decision: disclose relevant private facts in this group.",
                tuple(
                    dict.fromkeys(
                        value for claim in claims for value in claim.evidence_ids
                    )
                ),
                True,
            ),
        )
        return DialogueModelOutput(
            "shared-fake", f"shared-{run.run_id}", json.dumps(draft_to_dict(draft))
        )

    def review(
        self,
        workspace: WorkspacePacket,
        draft: DialogueDraft,
        run: RunContext,
    ) -> DialogueModelOutput:
        self.calls.append("review")
        return DialogueModelOutput(
            "review-fake",
            f"review-{run.run_id}",
            '{"supported":true,"reason":"fixture supported"}',
        )


def shared_setup(
    tmp_path: Path,
) -> tuple[ApplicationContainer, SharedModel, EventEnvelope, EventEnvelope]:
    model = SharedModel()
    container = build_container(tmp_path, dialogue_model=model)
    seed(container, message("我喜欢晚上学习；alice-private-input"))
    bob = seed(container, message("我喜欢早上学习；bob-private-input", "bob"))
    seed(
        container,
        message("我喜欢早上学习；foreign-mind-secret", "foreign", "other-mind"),
    )
    seed(
        container,
        EventEnvelope.self_model_observation(
            SubjectScope.for_mind("mind-20"),
            SelfModelObservationPayload(
                SelfModelAspect.IDENTITY, "role", "研究助手", 1.0
            ),
            clock=FixedClock(),
        ),
    )
    request = EventEnvelope.user_message(
        subject(),
        "请说明 alice 与 bob 的晚上和早上学习偏好及你的身份",
        conversation=ConversationScope("chat-20", "group-20"),
        disclosure=DisclosureScope.GROUP,
        clock=FixedClock(),
    )
    return container, model, bob, request


def test_rule_dialogue_persists_and_reuses_without_replaying_models(
    tmp_path: Path,
) -> None:
    model = CountingModel()
    container = build_container(tmp_path, dialogue_model=model)
    original = seed(container, message("我喜欢晚上学习"))
    request = DialogueRequest(message())
    result = container.converse.converse(request, context())
    assert result.status is ProcessEventStatus.SUCCEEDED
    assert result.response.text == "你喜欢晚上学习。"
    assert result.evidence_refs == (EvidenceRef.for_event(original),)
    assert model.calls == ["generate", "review"]
    events = container.event_store.read_by_mind(subject().mind)
    answer = next(
        event for event in events if event.event_id == result.response_event_id
    )
    assert answer.actor == SubjectScope.for_mind("mind-20").subject
    assert answer.subject == SubjectScope.for_mind("mind-20")
    assert answer.payload.recipient == subject()
    assert answer.payload.review.supported
    for event in events:
        assert event_from_json(event_to_json(event)) == event
    assert container.replay.replay(subject()) == container.state_repository.load(
        subject()
    )
    assert container.replay.replay(answer.subject).version == 0
    no_calls = CountingModel("no_calls")
    restarted = build_container(tmp_path, dialogue_model=no_calls)
    reused = restarted.converse.converse(request, context())
    assert reused.reused and reused.response_event_id == result.response_event_id
    assert reused.response == result.response
    assert no_calls.calls == []
    with pytest.raises(ContractValidationError, match="different input"):
        restarted.converse.converse(
            DialogueRequest(
                replace(request.event, payload=message("不同问题").payload)
            ),
            context(),
        )


def test_shared_workspace_preserves_ownership_budget_and_disclosure(
    tmp_path: Path,
) -> None:
    container, model, _, event = shared_setup(tmp_path)
    result = container.converse.converse(
        DialogueRequest(event, max_tokens=6000), context()
    )
    assert result.status is ProcessEventStatus.SUCCEEDED
    packet = model.workspaces[0]
    owners = {item.subject.subject.subject_id for item in packet.items}
    assert {"alice", "bob", "mind-20"} <= owners
    assert "foreign" not in owners
    assert "foreign-mind-secret" not in json.dumps(workspace_model_context(packet))
    assert packet.used_tokens == estimate_tokens(workspace_model_context(packet))
    assert packet.used_tokens <= packet.budget.max_tokens
    assert all(item.subject.mind == subject().mind for item in packet.items)
    assert all(
        item.state_version is not None and item.data_scope is not None
        for item in packet.items
    )
    assert result.response.disclosure.overrides_intent
    assert result.response.disclosure.scope is DisclosureScope.GROUP
    assert model.calls == ["generate", "review"]
    low_budget = container.workspace_builder.build_shared(
        event.payload.text,
        tuple(
            container.state_repository.load(owner)
            for owner in {item.subject for item in packet.items}
        ),
        subject(),
        input_evidence=packet.input_evidence,
        budget=RetrievalBudget(max_tokens=1024, max_items=1),
    )
    assert len(low_budget.items) <= 1 and low_budget.used_tokens <= 1024
    assert any(not decision.selected for decision in low_budget.decisions)
    with pytest.raises(ContractValidationError, match="mind boundaries"):
        container.workspace_builder.build_shared(
            QUESTION,
            (container.state_repository.load(subject("foreign", "other-mind")),),
            subject(),
            input_evidence=packet.input_evidence,
        )


@pytest.mark.parametrize(
    "question,expected_calls",
    [("你好！", ["generate"]), ("没有证据的未知问题", ["generate", "review"])],
)
def test_smalltalk_and_unknown_have_bounded_calls(
    tmp_path: Path, question: str, expected_calls: list[str]
) -> None:
    model = CountingModel()
    container = build_container(tmp_path, dialogue_model=model)
    result = container.converse.converse(DialogueRequest(message(question)), context())
    assert result.status is ProcessEventStatus.SUCCEEDED
    assert model.calls == expected_calls
    assert not result.evidence_refs
    if len(expected_calls) == 2:
        assert result.response.claims[0].stance is ClaimStance.UNKNOWN


@pytest.mark.parametrize(
    "problem,error_type,calls",
    [
        ("malformed", "ModelOutputError", ["generate"]),
        ("foreign_citation", "ModelOutputError", ["generate"]),
        ("unsupported", "GroundingRejected", ["generate", "review"]),
        ("hidden_claim", "GroundingRejected", ["generate", "review"]),
        ("bad_review", "ModelOutputError", ["generate", "review"]),
        ("cancel", "RunCancelledError", ["generate"]),
        ("timeout", "ModelTimeoutError", ["generate"]),
    ],
)
def test_invalid_or_interrupted_output_is_persisted_but_not_published(
    tmp_path: Path,
    problem: str,
    error_type: str,
    calls: list[str],
) -> None:
    model = CountingModel(problem)
    container = build_container(tmp_path, dialogue_model=model)
    seed(container, message("我喜欢晚上学习"))
    request = DialogueRequest(message())
    result = container.converse.converse(request, context())
    assert result.error_type == error_type
    assert result.response is None and result.response_event_id is None
    assert model.calls == calls
    events = container.event_store.read_by_subject(SubjectScope.for_mind("mind-20"))
    assert not any(event.event_type == "assistant.message" for event in events)
    assert any(event.event_type == "dialogue.failed" for event in events)
    raw = [
        event.payload.raw_output
        for event in events
        if event.event_type == "model.response"
    ]
    assert len(raw) == len(calls)
    if problem == "malformed":
        assert raw == ["invalid-original-model-output"]
    repeated = container.converse.converse(request, context())
    assert repeated.reused and repeated.error_type == error_type
    assert model.calls == calls


def test_low_confidence_cannot_be_relabelled_as_certain(tmp_path: Path) -> None:
    model = CountingModel()
    container = build_container(tmp_path, dialogue_model=model)
    seed(container, message("我喜欢晚上学习"))
    result = container.converse.converse(DialogueRequest(message()), context())
    packet = model.workspaces[0]
    uncertain = replace(
        packet, items=tuple(replace(item, confidence=0.4) for item in packet.items)
    )
    with pytest.raises(ModelOutputError, match="uncertain evidence"):
        validate_grounding(result.response, uncertain)
    cautious = replace(
        result.response,
        claims=tuple(
            replace(claim, stance=ClaimStance.UNCERTAIN)
            for claim in result.response.claims
        ),
    )
    assert validate_grounding(cautious, uncertain) == result.evidence_refs


def test_budget_rejection_happens_before_model_call(tmp_path: Path) -> None:
    model = CountingModel()
    container = build_container(tmp_path, dialogue_model=model)
    result = container.converse.converse(
        DialogueRequest(message(), max_tokens=1), context()
    )
    assert result.status is ProcessEventStatus.FAILED
    assert result.error_type == "ContractValidationError"
    assert model.calls == []


def test_crashed_generation_is_explicitly_terminated_on_resume(tmp_path: Path) -> None:
    model = CountingModel("crash")
    container = build_container(tmp_path, dialogue_model=model)
    request = DialogueRequest(message())
    with pytest.raises(KeyboardInterrupt):
        container.converse.converse(request, context())
    resumed_model = CountingModel("no_calls")
    restarted = build_container(tmp_path, dialogue_model=resumed_model)
    result = restarted.converse.converse(request, context())
    assert result.error_type == "InterruptedDialogue"
    assert result.response is None and resumed_model.calls == []


def test_concurrent_duplicate_requests_only_generate_once(tmp_path: Path) -> None:
    model = CountingModel()
    container = build_container(tmp_path, dialogue_model=model)
    request = DialogueRequest(message("你好"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(container.converse.converse, request, context())
            for _ in range(2)
        ]
        results = [future.result() for future in futures]
    assert all(result.status is ProcessEventStatus.SUCCEEDED for result in results)
    assert results[0].response_event_id == results[1].response_event_id
    assert sum(result.reused for result in results) == 1
    assert model.calls == ["generate"]


def test_answer_write_failure_does_not_regenerate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = CountingModel()
    container = build_container(tmp_path, dialogue_model=model)
    original_append = container.event_store.append

    def fail_answer(event: EventEnvelope) -> None:
        if event.event_type == "assistant.message":
            raise OSError("simulated answer write failure")
        original_append(event)

    monkeypatch.setattr(container.event_store, "append", fail_answer)
    request = DialogueRequest(message("你好"))
    result = container.converse.converse(request, context())
    assert result.error_type == "OSError"
    assert model.calls == ["generate"]
    assert container.converse.converse(request, context()).reused
    assert model.calls == ["generate"]


def test_deletion_removes_dependent_answers_without_other_sources_or_regeneration(
    tmp_path: Path,
) -> None:
    container, model, bob, event = shared_setup(tmp_path)
    request = DialogueRequest(event, max_tokens=6000)
    result = container.converse.converse(request, context())
    assert result.status is ProcessEventStatus.SUCCEEDED
    before = container.event_store.read_by_mind(subject().mind)
    answer = next(item for item in before if item.event_id == result.response_event_id)
    started = next(item for item in before if item.event_id == answer.causation_id)
    plan = container.forget.dry_run(
        DeletionSelector(bob.subject, delete_subject=True), now=NOW
    )
    assert result.response_event_id in {
        value for impact in plan.effective_impacts for value in impact.event_ids
    }
    assert container.forget.execute(plan, now=NOW).status is DeletionStatus.COMPLETED
    after = container.event_store.read_by_mind(subject().mind)
    assert not any(
        item.event_id in {answer.event_id, started.event_id} for item in after
    )
    assert not any(
        item.event_type == "model.response" and item.causation_id == started.event_id
        for item in after
    )
    assert any(item.event_id == event.event_id for item in after)
    assert (
        container.state_repository.load(subject()).get("preferences.study_time").value
        == "晚上"
    )
    assert (
        container.state_repository.load(SubjectScope.for_mind("mind-20"))
        .get("identity.role")
        .value
        == "研究助手"
    )
    assert "bob-private-input" not in (tmp_path / "events" / "events.jsonl").read_text(
        encoding="utf-8"
    )
    no_calls = CountingModel("no_calls")
    restarted = build_container(tmp_path, dialogue_model=no_calls)
    repeated = restarted.converse.converse(request, context())
    assert repeated.status is ProcessEventStatus.FAILED and no_calls.calls == []
    assert restarted.replay.replay(subject()) == restarted.state_repository.load(
        subject()
    )
    for store in (container.event_store, InMemoryEventStore()):
        if isinstance(store, InMemoryEventStore):
            store.append_many(before)
            store.redact(started.subject, (started.event_id,), uuid4())
        late = replace(answer, event_id=uuid4())
        store.append(late)
        assert not any(
            item.event_id == late.event_id
            for item in store.read_by_subject(late.subject)
        )


class ResponsesClient:
    def __init__(self) -> None:
        self.responses = self
        self.api_key = "system-config-secret"
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        if kwargs["text"]["format"]["name"] == "dialogue_grounding":
            output = {"supported": True, "reason": "offline fixture checked"}
        else:
            packet = json.loads(kwargs["input"])
            items = [
                item
                for item in packet["items"]
                if item["target_field"] == "preferences.study_time"
            ]
            item = items[0]
            text = f"你喜欢{item['content']}学习。"
            draft = DialogueDraft(
                text,
                (
                    DialogueClaim(
                        text,
                        tuple(
                            UUID(ref["evidence_id"]) for ref in item["evidence_refs"]
                        ),
                        ClaimStance.SUPPORTED,
                    ),
                ),
                DisclosureDecision(
                    "disclose", DisclosureScope.PRIVATE, "offline decision fixture", ()
                ),
            )
            output = draft_to_dict(draft)
        return SimpleNamespace(
            id=f"response-{len(self.calls)}",
            output_text=json.dumps(output, ensure_ascii=False),
            status="completed",
        )

    def close(self) -> None:
        pass


def test_responses_adapter_uses_fresh_bounded_calls_and_same_service(
    tmp_path: Path,
) -> None:
    client = ResponsesClient()
    model = OpenAIResponsesDialogueModel(client, "configured-model")
    container = build_container(tmp_path, dialogue_model=model)
    seed(container, message("我喜欢晚上学习；我主动提供的密钥=user-provided-secret"))
    for question in (QUESTION, "请说说晚上学习偏好，本轮独有标记"):
        result = container.converse.converse(
            DialogueRequest(message(question)), context()
        )
        assert result.status is ProcessEventStatus.SUCCEEDED
        assert result.response.text == "你喜欢晚上学习。"
    assert len(client.calls) == 4
    for call in client.calls:
        assert call["store"] is False and call["text"]["format"]["strict"] is True
        assert "previous_response_id" not in call and "conversation" not in call
        assert "api_key" not in call["input"]
        assert "system-config-secret" not in call["input"]
        assert "user-provided-secret" in call["input"]
        assert 0 < call["timeout"] <= 30
    assert QUESTION not in client.calls[2]["input"]
    assert "answer" not in json.loads(client.calls[2]["input"])


@pytest.mark.parametrize("change", ["delete", "update", "mutate"])
def test_context_changes_stop_publication_before_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    model = CountingModel()
    container = build_container(tmp_path, dialogue_model=model)
    seed(
        container,
        message("我开始准备研究项目" if change == "mutate" else "我喜欢晚上学习"),
    )
    original_generate = model.generate

    def generate_and_change(
        packet: WorkspacePacket,
        run: RunContext,
    ) -> DialogueModelOutput:
        output = original_generate(packet, run)
        if change == "delete":
            plan = container.forget.dry_run(
                DeletionSelector(subject(), delete_subject=True),
                now=NOW,
            )
            container.forget.execute(plan, now=NOW)
        elif change == "update":
            seed(container, message("我喜欢早上学习"))
        else:
            item = next(item for item in packet.items if isinstance(item.content, dict))
            item.content["summary"] = "injected-summary"
        return output

    monkeypatch.setattr(model, "generate", generate_and_change)
    question = "我的项目经历如何发展？" if change == "mutate" else QUESTION
    result = container.converse.converse(DialogueRequest(message(question)), context())
    assert result.status is ProcessEventStatus.FAILED
    assert result.response is None and model.calls == ["generate"]
    assert not any(
        event.event_type == "assistant.message"
        for event in container.event_store.read_by_mind(subject().mind)
    )
    if change == "mutate":
        assert "injected-summary" not in repr(
            container.state_repository.load(subject())
        )

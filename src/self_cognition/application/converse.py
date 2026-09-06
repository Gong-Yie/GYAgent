import json
import logging
from dataclasses import replace
from threading import RLock
from uuid import UUID, uuid5

from self_cognition.application.process_event import ProcessEventService
from self_cognition.application.results import ConverseResult, ProcessEventStatus
from self_cognition.core.dialogue import (
    AssistantMessagePayload,
    DialogueContextPayload,
    DialogueFailurePayload,
    DialogueModel,
    DialogueModelOutput,
    DialogueRequest,
    draft_from_dict,
    parse_model_json,
    review_from_dict,
)
from self_cognition.core.errors import (
    ContractValidationError,
    ModelOutputError,
    ModelTimeoutError,
    RunCancelledError,
)
from self_cognition.core.events import EventEnvelope, EventSource
from self_cognition.core.evidence import EvidenceRef, EvidenceSourceKind
from self_cognition.core.protocols import (
    EventStore,
    EvidenceRepository,
    StateRepository,
)
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.state import SubjectState
from self_cognition.core.workspace import (
    RetrievalBudget,
    WorkspaceBuilder,
    WorkspacePacket,
    WorkspaceRunInfo,
    workspace_model_context,
)
from self_cognition.executive.dialogue.grounding import (
    is_plain_smalltalk,
    validate_grounding,
)
from self_cognition.runtime.run_context import RunContext

logger = logging.getLogger(__name__)


class ConverseService:
    def __init__(
        self,
        process_event: ProcessEventService,
        event_store: EventStore,
        evidence_repository: EvidenceRepository,
        state_repository: StateRepository,
        workspace_builder: WorkspaceBuilder,
        model: DialogueModel,
    ) -> None:
        self._process_event = process_event
        self._events = event_store
        self._evidence = evidence_repository
        self._states = state_repository
        self._builder = workspace_builder
        self._model = model
        self._lock = RLock()

    def converse(self, request: DialogueRequest, context: RunContext) -> ConverseResult:
        with self._lock:
            return self._converse(request, context)

    def _converse(
        self, request: DialogueRequest, context: RunContext
    ) -> ConverseResult:
        origin = request.event
        history = self._events.read_by_mind(origin.subject.mind)
        stored = next(
            (event for event in history if event.event_id == origin.event_id), None
        )
        if stored is not None:
            if replace(stored, run_id=None, correlation_id=None) != replace(
                origin,
                run_id=None,
                correlation_id=None,
            ):
                raise ContractValidationError(
                    "request ID already belongs to a different input"
                )
            terminal = next(
                (
                    event
                    for event in history
                    if isinstance(
                        event.payload, (AssistantMessagePayload, DialogueFailurePayload)
                    )
                    and event.payload.request_event_id == origin.event_id
                    and event.payload.recipient == origin.subject
                ),
                None,
            )
            if terminal is not None:
                return self._result(terminal, reused=True)
            origin = stored
        started = next(
            (
                event
                for event in history
                if isinstance(event.payload, DialogueContextPayload)
                and event.payload.request_event_id == origin.event_id
            ),
            None,
        )
        if started is not None:
            return self._failure(
                origin,
                started,
                context,
                "resume",
                "InterruptedDialogue",
                started.payload.old_version,
                started.payload.new_version,
            )
        if stored is not None:
            context = replace(
                context,
                run_id=stored.run_id or context.run_id,
                correlation_id=stored.correlation_id or context.correlation_id,
            )
        processed = self._process_event.process(origin, context)
        if not processed.event_saved:
            return ConverseResult(
                processed.status,
                context.run_id,
                context.correlation_id,
                processed.old_version,
                processed.new_version,
                processed.state_changed,
                False,
                error_type=processed.error_type,
            )
        origin = next(
            event
            for event in self._events.read_by_subject(origin.subject)
            if event.event_id == origin.event_id
        )
        if (
            processed.status is not ProcessEventStatus.SUCCEEDED
            or processed.state is None
        ):
            return self._failure(
                origin,
                origin,
                context,
                "cognition",
                processed.error_type or "CognitionFailed",
                processed.old_version,
                processed.new_version,
            )
        stage = "workspace"
        cause = origin
        try:
            self._ensure_active(context)
            workspace = self._workspace(request, origin, processed.state, context)
            refs = self._workspace_refs(workspace)
            self._require_live(workspace)
            payload = DialogueContextPayload(
                origin.event_id,
                origin.subject,
                json.dumps(
                    workspace_model_context(workspace),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                refs,
                processed.old_version,
                processed.new_version,
            )
            cause = self._event(origin, origin, "dialogue.started", payload, context)
            self._persist(cause)
            stage = "generate"
            output = self._model.generate(workspace, context)
            self._save_output(cause, output, context)
            self._ensure_active(context)
            self._require_unchanged(workspace, payload.workspace_json)
            draft = draft_from_dict(parse_model_json(output.raw_output))
            cited = validate_grounding(draft, workspace)
            review = None
            if not is_plain_smalltalk(workspace.task_context, draft):
                stage = "grounding"
                self._require_live(workspace)
                output = self._model.review(workspace, draft, context)
                self._save_output(cause, output, context)
                self._ensure_active(context)
                self._require_unchanged(workspace, payload.workspace_json)
                review = review_from_dict(parse_model_json(output.raw_output))
                if not review.supported:
                    return self._failure(
                        origin,
                        cause,
                        context,
                        stage,
                        "GroundingRejected",
                        processed.old_version,
                        processed.new_version,
                    )
            stage = "persist_answer"
            self._require_live(workspace)
            self._ensure_active(context)
            answer = self._event(
                origin,
                cause,
                "assistant.message",
                AssistantMessagePayload(
                    origin.event_id,
                    origin.subject,
                    draft,
                    cited,
                    review,
                    processed.old_version,
                    processed.new_version,
                ),
                context,
            )
            self._persist(answer)
            return self._result(answer)
        except Exception as error:
            logger.warning(
                "dialogue failed stage=%s error_type=%s", stage, type(error).__name__
            )
            return self._failure(
                origin,
                cause,
                context,
                stage,
                type(error).__name__,
                processed.old_version,
                processed.new_version,
            )

    def _workspace(
        self,
        request: DialogueRequest,
        origin: EventEnvelope,
        current: SubjectState,
        context: RunContext,
    ) -> WorkspacePacket:
        owners = {
            origin.subject,
            SubjectScope.for_mind(origin.subject.mind.mind_id),
            *(
                event.subject
                for event in self._events.read_by_mind(origin.subject.mind)
            ),
        }
        states = {origin.subject: current}
        for owner in owners - {origin.subject}:
            state = self._states.load(owner)
            if state is not None:
                states[owner] = state
        return self._builder.build_shared(
            origin.payload.text,
            tuple(states.values()),
            origin.subject,
            input_evidence=EvidenceRef.for_event(origin),
            conversation=origin.scope.conversation,
            budget=RetrievalBudget(request.max_tokens, request.max_items),
            run_info=WorkspaceRunInfo(
                context.run_id,
                context.correlation_id,
                context.deadline,
                context.is_cancelled,
            ),
            as_of=context.clock.now(),
        )

    def _require_live(self, workspace: WorkspacePacket) -> None:
        current = {
            event.event_id: event
            for event in self._events.read_by_mind(workspace.subject.mind)
        }
        for ref in self._workspace_refs(workspace):
            if ref.source_kind is EvidenceSourceKind.SYSTEM_PRIOR:
                continue
            event = current.get(ref.evidence_id)
            if event is None or event.subject != ref.scope.owner:
                raise ContractValidationError(
                    "workspace evidence was deleted or is unavailable"
                )
            if event.event_type == "assistant.message":
                raise ModelOutputError(
                    "assistant assertions are not independent factual evidence"
                )
        versions = {(item.subject, item.state_version) for item in workspace.items}
        versions.add((workspace.subject, workspace.state_version))
        for owner, version in versions:
            state = self._states.load(owner)
            if (state.version if state is not None else 0) != version:
                raise ContractValidationError("workspace changed during dialogue")

    @staticmethod
    def _require_unchanged(workspace: WorkspacePacket, snapshot: str) -> None:
        if (
            json.dumps(
                workspace_model_context(workspace), ensure_ascii=False, sort_keys=True
            )
            != snapshot
        ):
            raise ModelOutputError("model changed the read-only workspace")

    @staticmethod
    def _workspace_refs(workspace: WorkspacePacket) -> tuple[EvidenceRef, ...]:
        refs = {ref.evidence_id: ref for ref in workspace.evidence_refs}
        if workspace.input_evidence is not None:
            refs[workspace.input_evidence.evidence_id] = workspace.input_evidence
        return tuple(refs.values())

    def _save_output(
        self,
        cause: EventEnvelope,
        output: DialogueModelOutput,
        context: RunContext,
    ) -> None:
        self._persist(
            EventEnvelope.model_response(
                cause,
                model=output.model,
                response_id=output.response_id,
                raw_output=output.raw_output,
                clock=context.clock,
                run_id=context.run_id,
                correlation_id=context.correlation_id,
            )
        )
        if output.error_type is not None:
            raise ModelOutputError("provider returned invalid or incomplete output")

    def _persist(self, event: EventEnvelope) -> None:
        self._events.append(event)
        if not any(
            stored == event for stored in self._events.read_by_subject(event.subject)
        ):
            raise ContractValidationError(
                "deleted or conflicting dialogue event cannot be written"
            )
        self._evidence.append(EvidenceRef.for_event(event))

    def _failure(
        self,
        origin: EventEnvelope,
        cause: EventEnvelope,
        context: RunContext,
        stage: str,
        error_type: str,
        old_version: int | None,
        new_version: int | None,
    ) -> ConverseResult:
        if not any(
            stored.event_id == cause.event_id
            for stored in self._events.read_by_subject(cause.subject)
        ):
            cause = origin
        event = self._event(
            origin,
            cause,
            "dialogue.failed",
            DialogueFailurePayload(
                origin.event_id,
                origin.subject,
                stage,
                error_type,
                old_version,
                new_version,
            ),
            context,
        )
        try:
            self._persist(event)
        except (OSError, ContractValidationError):
            logger.error(
                "dialogue terminal persistence unavailable run_id=%s", context.run_id
            )
            return ConverseResult(
                ProcessEventStatus.FAILED,
                context.run_id,
                context.correlation_id,
                old_version,
                new_version,
                None if old_version is None else old_version != new_version,
                True,
                error_type="DialoguePersistenceFailed",
            )
        return self._result(event)

    @staticmethod
    def _event(
        origin: EventEnvelope,
        cause: EventEnvelope,
        event_type: str,
        payload: (
            DialogueContextPayload | AssistantMessagePayload | DialogueFailurePayload
        ),
        context: RunContext,
    ) -> EventEnvelope:
        owner = SubjectScope.for_mind(origin.subject.mind.mind_id)
        is_answer = isinstance(payload, AssistantMessagePayload)
        scope = replace(origin.scope, owner=owner)
        if is_answer:
            scope = replace(scope, disclosure=payload.answer.disclosure.scope)
        now = context.clock.now()
        return EventEnvelope(
            uuid5(origin.event_id, event_type),
            event_type,
            owner.subject if is_answer else None,
            owner,
            payload,
            now,
            now,
            EventSource.MODEL if is_answer else EventSource.SYSTEM,
            scope,
            cause.event_id,
            context.correlation_id,
            context.run_id,
        )

    @staticmethod
    def _ensure_active(context: RunContext) -> None:
        if context.cancelled:
            raise RunCancelledError("dialogue cancelled")
        if context.clock.now() >= context.deadline:
            raise ModelTimeoutError("dialogue deadline reached")

    @staticmethod
    def _result(event: EventEnvelope, *, reused: bool = False) -> ConverseResult:
        payload = event.payload
        success = isinstance(payload, AssistantMessagePayload)
        error_type = None if success else payload.error_type
        status = ProcessEventStatus.SUCCEEDED if success else ProcessEventStatus.FAILED
        if error_type == "RunCancelledError":
            status = ProcessEventStatus.CANCELLED
        return ConverseResult(
            status,
            event.run_id,
            event.correlation_id,
            payload.old_version,
            payload.new_version,
            (
                None
                if payload.old_version is None
                else payload.old_version != payload.new_version
            ),
            True,
            payload.answer if success else None,
            payload.evidence_refs if success else (),
            event.event_id if success else None,
            error_type,
            reused,
        )

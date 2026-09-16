import logging
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path

from self_cognition.application.affect import (
    AffectControlService,
    AffectViewService,
)
from self_cognition.application.execute_action import ActionService
from self_cognition.application.process_event import ProcessEventService
from self_cognition.application.converse import ConverseService
from self_cognition.application.pursue_goal import PursueGoalService
from self_cognition.application.proactive import ProactiveIntentionService
from self_cognition.application.user_control import UserControlService
from self_cognition.core.actions import ActionModel
from self_cognition.core.dialogue import DialogueModel, DialogueRequest
from self_cognition.core.events import EventEnvelope
from self_cognition.core.plans import PlanningModel
from self_cognition.executive.dialogue.fake import RuleDialogueAdapter
from self_cognition.application.replay import ReplayService
from self_cognition.application.forget import ForgetService
from self_cognition.blackboard.reducer import StateReducer
from self_cognition.blackboard.service import CognitiveSpaceService
from self_cognition.cognition.affect.affect_extractor import AffectExtractor
from self_cognition.cognition.affect.fast_reaction import FastAffectExtractor
from self_cognition.cognition.registry import (
    CognitiveModuleRegistry,
    ModuleRegistration,
)
from self_cognition.cognition.episodic.memory_extractor import (
    EpisodicMemoryExtractor,
)
from self_cognition.cognition.procedural.execution_extractor import (
    ProceduralExecutionExtractor,
)
from self_cognition.cognition.identity.identity_value_extractor import (
    IdentityValueExtractor,
)
from self_cognition.cognition.identity.self_model import (
    SelfModelCognitionModule,
)
from self_cognition.cognition.metacognition.conflict_extractor import (
    ConflictMetacognitionExtractor,
)
from self_cognition.cognition.metacognition.correction import UserCorrectionModule
from self_cognition.cognition.narrative.narrative_extractor import (
    NarrativeExtractor,
)
from self_cognition.cognition.relationship.relationship_extractor import (
    RelationshipExtractor,
)
from self_cognition.core.protocols import (
    CognitionModel,
    EvidenceRepository,
    DeletionRepository,
    EventStore,
    MemoryRepository,
    RunRepository,
    GovernanceRepository,
    StateRepository,
)
from self_cognition.core.proactivity import ProactivityModel
from self_cognition.core.state import SubjectState
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.workspace import WorkspaceBuilder
from self_cognition.cognition.semantic.concept_pattern_extractor import (
    ConceptPatternExtractor,
)
from self_cognition.cognition.semantic.name_extractor import NameExtractor
from self_cognition.cognition.semantic.preference_extractor import (
    PreferenceExtractor,
)
from self_cognition.cognition.semantic.llm_extractor import LLMSemanticExtractor
from self_cognition.executive.dialogue.rule_based import RuleBasedDialogueModel
from self_cognition.executive.action.fake import RuleActionModel
from self_cognition.executive.planning.fake import RulePlanningModel
from self_cognition.executive.planning.validator import PlanValidator
from self_cognition.infrastructure.persistence.file_event_store import FileEventStore
from self_cognition.infrastructure.persistence.file_deletion_repository import (
    FileDeletionRepository,
)
from self_cognition.infrastructure.persistence.file_layout import FileDataLayout
from self_cognition.infrastructure.persistence.file_memory_repository import (
    FileMemoryRepository,
)
from self_cognition.infrastructure.persistence.file_process_journal import (
    FileProcessJournal,
)
from self_cognition.infrastructure.persistence.file_provenance_store import (
    FileProvenanceStore,
)
from self_cognition.infrastructure.persistence.file_vector_index_store import (
    FileVectorIndexStore,
)
from self_cognition.infrastructure.persistence.file_processing_recovery import (
    FileProcessingRecovery,
)
from self_cognition.infrastructure.persistence.file_state_repository import (
    FileStateRepository,
)
from self_cognition.infrastructure.persistence.file_run_repository import (
    FileRunRepository,
)
from self_cognition.infrastructure.persistence.file_governance_repository import (
    FileGovernanceRepository,
)
from self_cognition.infrastructure.persistence.in_memory_evidence_repository import (
    InMemoryEvidenceRepository,
)
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.runtime.event_bus import RetryPolicy, SingleMachineEventBus
from self_cognition.runtime.recovery import RunRecoveryService
from self_cognition.runtime.run_service import RunLifecycle
from self_cognition.indexes.provenance import ProvenanceGraphService
from self_cognition.indexes.vector import VectorIndexService
from self_cognition.runtime.scheduler import DualLoopScheduler
from self_cognition.runtime.run_context import RunContext
from self_cognition.workers.scheduler import SchedulerWorker
from self_cognition.core.ids import new_correlation_id, new_run_id
from self_cognition.runtime.health import HealthService
from self_cognition.observability.metrics import MetricsRegistry
from self_cognition.observability.tracing import TraceRecorder
from self_cognition.infrastructure.llm.model_config import (
    ModelEnvironmentConfig,
    load_model_config,
)
from self_cognition.infrastructure.llm.router import (
    ModelRegistration,
    ModelRouter,
    RoutedActionModel,
    RoutedDialogueModel,
    RoutedPlanningModel,
    RoutedProactivityModel,
)
from self_cognition.infrastructure.llm.action_responses import (
    OpenAIResponsesActionModel,
)
from self_cognition.infrastructure.llm.dialogue_responses import (
    OpenAIResponsesDialogueModel,
)
from self_cognition.infrastructure.llm.openai_responses import (
    OpenAIResponsesCognitionModel,
)
from self_cognition.infrastructure.llm.proactive_responses import (
    OpenAIResponsesProactivityModel,
)
from self_cognition.infrastructure.llm.planning_responses import (
    OpenAIResponsesPlanningModel,
)
from self_cognition.executive.orchestrator import (
    ExecutiveOrchestrator,
    default_fast_handler,
)
from self_cognition.lifecycle import ApplicationLifecycle
from self_cognition.memory.encoder import StateChangeMemoryEncoder
from self_cognition.memory.behavior import (
    MemoryConsolidationService,
    MemoryRetrievalService,
)
from self_cognition.memory.service import (
    MemoryAccessService,
    MemoryEncodingService,
    MemoryLifecycleService,
)
from self_cognition.workspace.retrieval import HybridWorkspaceRetriever
from self_cognition.settings import (
    ApplicationSettings,
    DotenvSecretSource,
    load_settings,
)
from self_cognition.core.time import SYSTEM_CLOCK
from self_cognition.tools.registry import CapabilityRegistry
from self_cognition.tools.executor import (
    FileReadToolExecutor,
    ToolExecutionPolicy,
    ToolExecutor,
    ToolRouterExecutor,
    WorkspaceShellExecutor,
    WorkspaceWebSearchExecutor,
)



logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ApplicationContainer:
    settings: ApplicationSettings
    secret_source: DotenvSecretSource
    event_store: EventStore
    evidence_repository: EvidenceRepository
    state_repository: StateRepository
    affect_view: AffectViewService
    affect_control: AffectControlService
    memory_repository: MemoryRepository
    provenance: ProvenanceGraphService
    vector_index: VectorIndexService
    memory_encoding: MemoryEncodingService
    memory_access: MemoryAccessService
    memory_retrieval: MemoryRetrievalService
    memory_consolidation: MemoryConsolidationService
    memory_lifecycle: MemoryLifecycleService
    deletion_repository: DeletionRepository
    forget: ForgetService
    process_event: ProcessEventService
    converse: ConverseService
    pursue_goal: PursueGoalService
    action: ActionService
    event_bus: SingleMachineEventBus
    replay: ReplayService
    workspace_builder: WorkspaceBuilder
    dialogue_model: RuleBasedDialogueModel | DialogueModel
    module_registry: CognitiveModuleRegistry
    capability_registry: CapabilityRegistry
    planning_model: PlanningModel
    action_model: ActionModel
    tool_executor: ToolExecutor | None
    run_repository: RunRepository
    run_lifecycle: RunLifecycle
    run_recovery: RunRecoveryService
    proactive: ProactiveIntentionService
    scheduler: DualLoopScheduler
    orchestrator: ExecutiveOrchestrator
    lifecycle: ApplicationLifecycle
    user_control: UserControlService
    governance: GovernanceRepository
    metrics: MetricsRegistry
    traces: TraceRecorder
    model_router: ModelRouter
    health: HealthService


def build_container(
    data_dir: str | Path | None = None,
    *,
    settings: ApplicationSettings | None = None,
    dotenv_path: str | Path = ".env",
    module_registrations: tuple[ModuleRegistration, ...] | None = None,
    dialogue_model: RuleBasedDialogueModel | DialogueModel | None = None,
    planning_model: PlanningModel | None = None,
    action_model: ActionModel | None = None,
    tool_executor: ToolExecutor | None = None,
    metacognition_model: CognitionModel | None = None,
    affect_model: CognitionModel | None = None,
    semantic_model: CognitionModel | None = None,
    proactive_model: ProactivityModel | None = None,
) -> ApplicationContainer:
    resolved_settings = settings or load_settings(dotenv_path)
    if data_dir is not None:
        resolved_settings = replace(resolved_settings, data_dir=Path(data_dir))
    dialogue_model_explicit = dialogue_model is not None
    planning_model_explicit = planning_model is not None
    action_model_explicit = action_model is not None
    secret_source = DotenvSecretSource(Path(dotenv_path))
    openai_configuration = _openai_configuration(secret_source)
    model_environment = _load_model_environment(secret_source)
    configured_tasks = (
        frozenset(model_environment.routes)
        if model_environment is not None
        else frozenset()
    )
    if openai_configuration is not None:
        api_key, model, base_url = openai_configuration
        if dialogue_model is None:
            dialogue_model = OpenAIResponsesDialogueModel.from_api_key(
                api_key,
                model,
                base_url=base_url,
                max_output_tokens=resolved_settings.dialogue_max_output_tokens,
                temperature=resolved_settings.model_temperature,
            )
        if planning_model is None:
            planning_model = OpenAIResponsesPlanningModel.from_api_key(
                api_key,
                model,
                base_url=base_url,
                max_output_tokens=resolved_settings.cognition_max_output_tokens,
                temperature=resolved_settings.model_temperature,
            )
        if action_model is None:
            action_model = OpenAIResponsesActionModel.from_api_key(
                api_key,
                model,
                base_url=base_url,
                max_output_tokens=resolved_settings.cognition_max_output_tokens,
                temperature=resolved_settings.model_temperature,
            )
        if openai_configuration is not None and proactive_model is None:
            proactive_model = OpenAIResponsesProactivityModel.from_api_key(
                api_key,
                model,
                base_url=base_url,
                max_output_tokens=resolved_settings.cognition_max_output_tokens,
                temperature=resolved_settings.model_temperature,
            )
        if module_registrations is None and metacognition_model is None:
            metacognition_model = OpenAIResponsesCognitionModel.from_api_key(
                api_key,
                model,
                base_url=base_url,
                max_output_tokens=resolved_settings.cognition_max_output_tokens,
                assessment_kind="metacognition",
                temperature=resolved_settings.model_temperature,
            )
        if module_registrations is None and affect_model is None:
            affect_model = OpenAIResponsesCognitionModel.from_api_key(
                api_key,
                model,
                base_url=base_url,
                max_output_tokens=resolved_settings.cognition_max_output_tokens,
                assessment_kind="affect",
                temperature=resolved_settings.model_temperature,
            )
        if module_registrations is None and semantic_model is None:
            semantic_model = OpenAIResponsesCognitionModel.from_api_key(
                api_key,
                model,
                base_url=base_url,
                max_output_tokens=resolved_settings.cognition_max_output_tokens,
                assessment_kind="semantic",
                temperature=resolved_settings.model_temperature,
            )
    layout = FileDataLayout(resolved_settings.data_dir).ensure()
    metrics = MetricsRegistry()
    traces = TraceRecorder()
    event_store = FileEventStore(
        layout.event_log,
        layout.deletions / "event_tombstones.jsonl",
    )
    run_repository = FileRunRepository(layout.runs)
    governance = FileGovernanceRepository(layout.governance)
    run_lifecycle = RunLifecycle(run_repository)
    run_recovery = RunRecoveryService(run_repository, event_store)
    run_recovery.recover(SYSTEM_CLOCK.now())
    evidence_repository = InMemoryEvidenceRepository()
    state_repository = FileStateRepository(layout.states)
    affect_view = AffectViewService(state_repository)
    memory_repository = FileMemoryRepository(
        layout.memories,
        layout.indexes / "memories",
        layout.memory_access,
    )
    provenance = ProvenanceGraphService(
        event_store,
        memory_repository,
        FileProvenanceStore(layout.indexes / "provenance"),
    )
    vector_index = VectorIndexService(
        memory_repository,
        event_store,
        FileVectorIndexStore(layout.indexes / "vector"),
    )
    memory_encoding = MemoryEncodingService(
        memory_repository,
        StateChangeMemoryEncoder(),
    )
    memory_access = MemoryAccessService(memory_repository)
    memory_retrieval = MemoryRetrievalService(memory_repository)
    memory_consolidation = MemoryConsolidationService(memory_repository)
    memory_lifecycle = MemoryLifecycleService(memory_repository)
    deletion_repository = FileDeletionRepository(layout.deletions)
    process_journal = FileProcessJournal(layout.processing)
    FileProcessingRecovery(layout.event_log, process_journal).reconcile()
    module_registry = _module_registry(
        module_registrations
        or _default_module_registrations(
            metacognition_model,
            affect_model,
            semantic_model,
        ),
        resolved_settings.enabled_modules,
    )
    workspace_builder = WorkspaceBuilder(
        HybridWorkspaceRetriever(memory_repository)
    )
    engine = CognitionEngine(
        modules=module_registry.all_modules(),
        cognitive_space=CognitiveSpaceService(StateReducer()),
        module_registry=module_registry,
        workspace_builder=workspace_builder,
    )
    process_event = ProcessEventService(
        event_store=event_store,
        evidence_repository=evidence_repository,
        state_repository=state_repository,
        engine=engine,
        process_journal=process_journal,
        memory_encoding=memory_encoding,
        run_lifecycle=run_lifecycle,
        metrics=metrics,
        governance=governance,
    )
    model_router = ModelRouter(
        disabled_provider_loader=lambda subject: governance.load_controls(
            subject
        ).disabled_model_providers,
        failure_cooldown=timedelta(
            seconds=resolved_settings.model_failure_cooldown_seconds
        ),
        max_failure_cooldown=timedelta(
            seconds=resolved_settings.model_max_failure_cooldown_seconds
        ),
        clock=SYSTEM_CLOCK,
    )
    _register_configured_models(
        model_router,
        model_environment,
        secret_source,
        resolved_settings,
    )
    selected_proactive_model: ProactivityModel | None = None
    if proactive_model is not None:
        model_router.register(
            ModelRegistration("proactive", "proactive-default", proactive_model)
        )
    if proactive_model is not None or "proactive" in configured_tasks:
        selected_proactive_model = RoutedProactivityModel(model_router)
    proactive = ProactiveIntentionService(
        event_store,
        evidence_repository,
        governance,
        metrics,
        selected_proactive_model,
    )
    wake_subjects: set[SubjectScope] = {
        event.subject
        for event in event_store.read_all()
        if event.event_type in {"user.message", "proactive.intention", "motive.formed"}
    }
    last_reassessment = {
        event.subject: event.recorded_at
        for event in event_store.read_all()
        if event.event_type == "proactivity.reassessment"
    }

    def wake_due_intentions() -> None:
        now = SYSTEM_CLOCK.now()
        for subject in tuple(wake_subjects):
            try:
                context = RunContext(
                    new_run_id(),
                    new_correlation_id(),
                    now + timedelta(seconds=30),
                )
                state = state_repository.load(subject) or SubjectState.empty(
                    subject.subject.subject_id,
                    mind_id=subject.mind.mind_id,
                    subject_kind=subject.subject.kind,
                )
                workspace = workspace_builder.build(
                    f"时间唤醒：{now.isoformat()}", state
                )
                proactive.consume_due(
                    subject,
                    as_of=now,
                    context=context,
                    workspace=workspace,
                )
                active = proactive.active(subject, as_of=now)
                if not active:
                    continue
                previous = last_reassessment.get(subject)
                if previous is not None and now - previous < timedelta(minutes=1):
                    continue
                event = EventEnvelope.proactivity_reassessment(
                    subject,
                    "time.tick",
                    previous_assessment_at=previous,
                    clock=context.clock,
                    correlation_id=context.correlation_id,
                    run_id=context.run_id,
                )
                event_store.append(event)
                last_reassessment[subject] = event.recorded_at
                proactive.evaluate(event, workspace, context)
            except Exception as error:
                if metrics is not None:
                    metrics.increment("proactive.scheduler.failures")
                logger.warning(
                    "scheduler proactive task failed subject_id=%s error_type=%s",
                    subject.subject.subject_id,
                    type(error).__name__,
                )
                continue

    scheduler = DualLoopScheduler(
        lambda event, context: converse.converse(DialogueRequest(event), context),
        metrics=metrics,
        traces=traces,
    )
    orchestrator = ExecutiveOrchestrator(
        state_repository,
        proactive,
        scheduler,
        fast_handler=default_fast_handler,
    )
    def after_success(event: EventEnvelope, context: RunContext) -> None:
        wake_subjects.add(event.subject)
        observed = None
        try:
            observed = proactive.observe_event(event, context=context)
            state = state_repository.load(event.subject)
            if state is None:
                state = SubjectState.empty(
                    event.subject.subject.subject_id,
                    mind_id=event.subject.mind.mind_id,
                    subject_kind=event.subject.subject.kind,
                )
            if observed is None and getattr(event.payload, "text", None):
                workspace = workspace_builder.build(event.payload.text, state)
                try:
                    proactive.evaluate(event, workspace, context)
                except Exception as error:
                    if metrics is not None:
                        metrics.increment(
                            "proactive.after_success.model_failures"
                        )
                    logger.warning(
                        "proactive evaluation failed event_id=%s error_type=%s",
                        event.event_id,
                        type(error).__name__,
                    )
                try:
                    proactive.form_boredom_social_motive(
                        event,
                        workspace,
                        context,
                    )
                except Exception as error:
                    if metrics is not None:
                        metrics.increment(
                            "proactive.after_success.deterministic_failures"
                        )
                    logger.warning(
                        "proactive motive formation failed event_id=%s error_type=%s",
                        event.event_id,
                        type(error).__name__,
                    )
        except Exception as error:
            if metrics is not None:
                metrics.increment("proactive.after_success.failures")
            logger.warning(
                "proactive after_success setup failed event_id=%s error_type=%s",
                event.event_id,
                type(error).__name__,
            )
        if event.event_type == "user.message":
            converse.converse(DialogueRequest(event), context)

    event_bus = SingleMachineEventBus(
        event_store,
        process_journal,
        process_event,
        max_workers=resolved_settings.worker_max_workers,
        retry_policy=RetryPolicy(
            lease_timeout=timedelta(
                seconds=resolved_settings.worker_lease_timeout_seconds
            )
        ),
        metrics=metrics,
        after_success=after_success,
    )
    replay = ReplayService(event_store=event_store, engine=engine)
    forget = ForgetService(
        event_store=event_store,
        evidence_repository=evidence_repository,
        state_repository=state_repository,
        memory_repository=memory_repository,
        deletion_repository=deletion_repository,
        replay=replay,
        process_journal=process_journal,
        run_repository=run_repository,
        governance=governance,
    )
    forget.recover(now=SYSTEM_CLOCK.now())
    selected_dialogue_model = dialogue_model or RuleBasedDialogueModel()
    dialogue_adapter = (
        RuleDialogueAdapter(selected_dialogue_model)
        if isinstance(selected_dialogue_model, RuleBasedDialogueModel)
        else selected_dialogue_model
    )
    capability_registry = CapabilityRegistry()
    if tool_executor is None:
        policy = ToolExecutionPolicy(
            (Path.cwd() / "workspace",),
            network_enabled=True,
        )
        shell = WorkspaceShellExecutor(policy)
        search = WorkspaceWebSearchExecutor(policy)
        tool_executor = ToolRouterExecutor(
            {shell.tool_id: shell, search.tool_id: search}
        )
        capability_registry.register(shell.registration)
        capability_registry.register(search.registration)
    if isinstance(tool_executor, FileReadToolExecutor):
        capability_registry.register(tool_executor.registration)
    selected_planning_model = planning_model or RulePlanningModel()
    selected_action_model = action_model or RuleActionModel()
    dialogue_default_cost = (
        2.0
        if (
            "dialogue" in configured_tasks
            and not dialogue_model_explicit
            and openai_configuration is None
        )
        else 0.0
    )
    planning_default_cost = (
        2.0
        if (
            "planning" in configured_tasks
            and not planning_model_explicit
            and openai_configuration is None
        )
        else 0.0
    )
    action_default_cost = (
        2.0
        if (
            "action" in configured_tasks
            and not action_model_explicit
            and openai_configuration is None
        )
        else 0.0
    )
    model_router.register(
        ModelRegistration(
            "dialogue",
            "dialogue-default",
            dialogue_adapter,
            cost_per_call=dialogue_default_cost,
        )
    )
    model_router.register(
        ModelRegistration(
            "planning",
            "planning-default",
            selected_planning_model,
            cost_per_call=planning_default_cost,
        )
    )
    model_router.register(
        ModelRegistration(
            "action",
            "action-default",
            selected_action_model,
            cost_per_call=action_default_cost,
        )
    )
    converse = ConverseService(
        process_event,
        event_store,
        evidence_repository,
        state_repository,
        workspace_builder,
        RoutedDialogueModel(model_router),
    )
    pursue_goal = PursueGoalService(
        process_event,
        event_store,
        state_repository,
        workspace_builder,
        RoutedPlanningModel(model_router),
        capability_registry,
        PlanValidator(),
    )
    action = ActionService(
        event_store,
        state_repository,
        workspace_builder,
        pursue_goal,
        capability_registry,
        RoutedActionModel(model_router),
        executor=tool_executor,
        run_lifecycle=run_lifecycle,
        process_event=process_event,
        governance=governance,
    )
    user_control = UserControlService(
        event_store,
        evidence_repository,
        state_repository,
        memory_repository,
        run_repository,
        governance,
        forget,
        process_event,
        module_registry,
        proactive,
        layout.exports,
    )
    affect_control = AffectControlService(user_control)
    wake_worker = SchedulerWorker(
        (wake_due_intentions,),
        interval_seconds=resolved_settings.worker_poll_interval_seconds,
    )
    lifecycle = ApplicationLifecycle(
        event_bus,
        worker_enabled=resolved_settings.worker_enabled,
        worker_poll_interval_seconds=(
            resolved_settings.worker_poll_interval_seconds
        ),
        resources=(
            event_bus,
            event_store,
            evidence_repository,
            state_repository,
            selected_dialogue_model,
            selected_planning_model,
            selected_action_model,
            scheduler,
            *((wake_worker,) if resolved_settings.worker_enabled else ()),
        ),
    )
    health = HealthService(
        data_dir=layout.root,
        event_bus=event_bus,
        lifecycle=lifecycle,
        module_registry=module_registry,
        capability_registry=capability_registry,
        model_router=model_router,
    )
    return ApplicationContainer(
        settings=resolved_settings,
        secret_source=secret_source,
        event_store=event_store,
        evidence_repository=evidence_repository,
        state_repository=state_repository,
        affect_view=affect_view,
        affect_control=affect_control,
        memory_repository=memory_repository,
        provenance=provenance,
        vector_index=vector_index,
        memory_encoding=memory_encoding,
        memory_access=memory_access,
        memory_retrieval=memory_retrieval,
        memory_consolidation=memory_consolidation,
        memory_lifecycle=memory_lifecycle,
        deletion_repository=deletion_repository,
        forget=forget,
        process_event=process_event,
        converse=converse,
        pursue_goal=pursue_goal,
        action=action,
        event_bus=event_bus,
        replay=replay,
        workspace_builder=workspace_builder,
        dialogue_model=selected_dialogue_model,
        module_registry=module_registry,
        capability_registry=capability_registry,
        planning_model=selected_planning_model,
        action_model=selected_action_model,
        tool_executor=tool_executor,
        run_repository=run_repository,
        run_lifecycle=run_lifecycle,
        run_recovery=run_recovery,
        proactive=proactive,
        scheduler=scheduler,
        orchestrator=orchestrator,
        lifecycle=lifecycle,
        user_control=user_control,
        governance=governance,
        metrics=metrics,
        traces=traces,
        model_router=model_router,
        health=health,
    )


def _load_model_environment(
    secret_source: DotenvSecretSource,
) -> ModelEnvironmentConfig | None:
    configured_path = secret_source.get("SC_MODELS_CONFIG")
    if configured_path:
        path = Path(configured_path)
        if not path.exists():
            raise FileNotFoundError(
                f"SC_MODELS_CONFIG does not exist: {path}"
            )
        return load_model_config(path, environment=secret_source.get("SC_ENV"))
    default_path = Path("config/models.json")
    if default_path.exists():
        return load_model_config(
            default_path,
            environment=secret_source.get("SC_ENV"),
        )
    return None


def _register_configured_models(
    model_router: ModelRouter,
    model_environment: ModelEnvironmentConfig | None,
    secret_source: DotenvSecretSource,
    settings: ApplicationSettings,
) -> None:
    if model_environment is None:
        return
    for task, provider_ids in sorted(model_environment.routes.items()):
        for index, provider_id in enumerate(provider_ids):
            provider = model_environment.providers[provider_id]
            api_key = secret_source.get(provider.api_key_env)
            if api_key is None:
                raise ValueError(
                    f"missing credential for configured provider {provider_id}"
                )
            model = _configured_model_for_task(
                task,
                provider,
                api_key,
                settings,
            )
            model_router.register(
                ModelRegistration(
                    task,
                    f"configured:{provider_id}",
                    model,
                    cost_per_call=0.5 + index,
                )
            )


def _configured_model_for_task(
    task: str,
    provider,
    api_key: str,
    settings: ApplicationSettings,
) -> object:
    base_url = provider.base_url
    if task == "dialogue":
        return OpenAIResponsesDialogueModel.from_api_key(
            api_key,
            provider.model,
            base_url=base_url,
            max_output_tokens=settings.dialogue_max_output_tokens,
            temperature=settings.model_temperature,
        )
    if task == "planning":
        return OpenAIResponsesPlanningModel.from_api_key(
            api_key,
            provider.model,
            base_url=base_url,
            max_output_tokens=settings.cognition_max_output_tokens,
            temperature=settings.model_temperature,
        )
    if task == "action":
        return OpenAIResponsesActionModel.from_api_key(
            api_key,
            provider.model,
            base_url=base_url,
            max_output_tokens=settings.cognition_max_output_tokens,
            temperature=settings.model_temperature,
        )
    if task == "proactive":
        return OpenAIResponsesProactivityModel.from_api_key(
            api_key,
            provider.model,
            base_url=base_url,
            max_output_tokens=settings.cognition_max_output_tokens,
            temperature=settings.model_temperature,
        )
    raise ValueError(f"unsupported configured model task: {task}")


def _default_module_registrations(
    metacognition_model: CognitionModel | None = None,
    affect_model: CognitionModel | None = None,
    semantic_model: CognitionModel | None = None,
) -> tuple[ModuleRegistration, ...]:
    return (
        ModuleRegistration(
            (
                "semantic.llm_extractor"
                if semantic_model is not None
                else "semantic.preference_extractor"
            ),
            "semantic",
            (
                LLMSemanticExtractor.module_version
                if semantic_model is not None
                else PreferenceExtractor.module_version
            ),
            LLMSemanticExtractor(semantic_model)
            if semantic_model is not None
            else PreferenceExtractor(),
        ),
        ModuleRegistration(
            "semantic.name_extractor",
            "semantic",
            "1",
            NameExtractor(),
        ),
        ModuleRegistration(
            "semantic.concept_pattern_extractor",
            "semantic",
            "1",
            ConceptPatternExtractor(),
        ),
        ModuleRegistration(
            "episodic.memory_extractor",
            "episodic",
            "1",
            EpisodicMemoryExtractor(),
        ),
        ModuleRegistration(
            "procedural.execution_extractor",
            "episodic",
            "1",
            ProceduralExecutionExtractor(),
        ),
        ModuleRegistration(
            "relationship.relationship_extractor",
            "relationship",
            "1",
            RelationshipExtractor(),
        ),
        ModuleRegistration(
            "metacognition.user_correction",
            "metacognition",
            "1",
            UserCorrectionModule(),
        ),
        ModuleRegistration(
            "metacognition.conflict_extractor",
            "metacognition",
            "2" if metacognition_model is not None else "1",
            ConflictMetacognitionExtractor(metacognition_model),
        ),
        ModuleRegistration(
            "identity.identity_value_extractor",
            "identity",
            "1",
            IdentityValueExtractor(),
        ),
        ModuleRegistration(
            "identity.self_model",
            "identity",
            "1",
            SelfModelCognitionModule(),
        ),
        ModuleRegistration(
            "affect.fast_reaction",
            "affect",
            "1",
            FastAffectExtractor(),
        ),
        ModuleRegistration(
            "affect.affect_extractor",
            "affect",
            "2" if affect_model is not None else "1",
            AffectExtractor(affect_model),
        ),
        ModuleRegistration(
            "narrative.narrative_extractor",
            "narrative",
            "1",
            NarrativeExtractor(),
        ),
    )


def _module_registry(
    registrations: tuple[ModuleRegistration, ...],
    enabled_module_ids: frozenset[str] | None,
) -> CognitiveModuleRegistry:
    known_ids = {registration.module_id for registration in registrations}
    if enabled_module_ids is not None:
        unknown_ids = enabled_module_ids - known_ids
        if unknown_ids:
            raise ValueError(
                "unknown enabled cognitive modules: "
                + ", ".join(sorted(unknown_ids))
            )
    configured = tuple(
        replace(
            registration,
            enabled=(
                enabled_module_ids is None
                or registration.module_id in enabled_module_ids
            ),
        )
        for registration in registrations
    )
    return CognitiveModuleRegistry(configured)


def _openai_configuration(
    secret_source: DotenvSecretSource,
) -> tuple[str, str, str | None] | None:
    api_key, model, base_url = tuple(
        (value.strip() or None) if value is not None else None
        for value in (
            secret_source.get("OPENAI_API_KEY"),
            secret_source.get("OPENAI_MODEL"),
            secret_source.get("OPENAI_BASE_URL"),
        )
    )
    if api_key is None and model is None and base_url is None:
        return None
    if api_key is None or model is None:
        raise ValueError(
            "OPENAI_API_KEY and OPENAI_MODEL must be configured together"
        )
    return api_key, model, base_url

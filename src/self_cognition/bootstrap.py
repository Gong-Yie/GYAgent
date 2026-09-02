from dataclasses import dataclass, replace
from pathlib import Path

from self_cognition.application.process_event import ProcessEventService
from self_cognition.application.replay import ReplayService
from self_cognition.blackboard.reducer import StateReducer
from self_cognition.blackboard.service import CognitiveSpaceService
from self_cognition.cognition.affect.affect_extractor import AffectExtractor
from self_cognition.cognition.registry import (
    CognitiveModuleRegistry,
    ModuleRegistration,
)
from self_cognition.cognition.episodic.memory_extractor import (
    EpisodicMemoryExtractor,
)
from self_cognition.cognition.identity.identity_value_extractor import (
    IdentityValueExtractor,
)
from self_cognition.cognition.metacognition.conflict_extractor import (
    ConflictMetacognitionExtractor,
)
from self_cognition.cognition.narrative.narrative_extractor import (
    NarrativeExtractor,
)
from self_cognition.cognition.relationship.relationship_extractor import (
    RelationshipExtractor,
)
from self_cognition.core.protocols import (
    EvidenceRepository,
    EventStore,
    StateRepository,
)
from self_cognition.core.workspace import WorkspaceBuilder
from self_cognition.cognition.semantic.name_extractor import NameExtractor
from self_cognition.cognition.semantic.preference_extractor import (
    PreferenceExtractor,
)
from self_cognition.executive.dialogue.rule_based import RuleBasedDialogueModel
from self_cognition.infrastructure.persistence.file_event_store import FileEventStore
from self_cognition.infrastructure.persistence.file_layout import FileDataLayout
from self_cognition.infrastructure.persistence.file_process_journal import (
    FileProcessJournal,
)
from self_cognition.infrastructure.persistence.file_processing_recovery import (
    FileProcessingRecovery,
)
from self_cognition.infrastructure.persistence.file_state_repository import (
    FileStateRepository,
)
from self_cognition.infrastructure.persistence.in_memory_evidence_repository import (
    InMemoryEvidenceRepository,
)
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.runtime.event_bus import SingleMachineEventBus
from self_cognition.lifecycle import ApplicationLifecycle
from self_cognition.settings import (
    ApplicationSettings,
    DotenvSecretSource,
    load_settings,
)


@dataclass(frozen=True, slots=True)
class ApplicationContainer:
    settings: ApplicationSettings
    secret_source: DotenvSecretSource
    event_store: EventStore
    evidence_repository: EvidenceRepository
    state_repository: StateRepository
    process_event: ProcessEventService
    event_bus: SingleMachineEventBus
    replay: ReplayService
    workspace_builder: WorkspaceBuilder
    dialogue_model: RuleBasedDialogueModel
    module_registry: CognitiveModuleRegistry
    lifecycle: ApplicationLifecycle


def build_container(
    data_dir: str | Path | None = None,
    *,
    settings: ApplicationSettings | None = None,
    dotenv_path: str | Path = ".env",
    module_registrations: tuple[ModuleRegistration, ...] | None = None,
    dialogue_model: RuleBasedDialogueModel | None = None,
) -> ApplicationContainer:
    resolved_settings = settings or load_settings(dotenv_path)
    if data_dir is not None:
        resolved_settings = replace(resolved_settings, data_dir=Path(data_dir))
    layout = FileDataLayout(resolved_settings.data_dir).ensure()
    event_store = FileEventStore(layout.event_log)
    evidence_repository = InMemoryEvidenceRepository()
    state_repository = FileStateRepository(layout.states)
    process_journal = FileProcessJournal(layout.processing)
    FileProcessingRecovery(layout.event_log, process_journal).reconcile()
    module_registry = _module_registry(
        module_registrations or _default_module_registrations(),
        resolved_settings.enabled_modules,
    )
    engine = CognitionEngine(
        modules=module_registry.all_modules(),
        cognitive_space=CognitiveSpaceService(StateReducer()),
        module_registry=module_registry,
    )
    process_event = ProcessEventService(
        event_store=event_store,
        evidence_repository=evidence_repository,
        state_repository=state_repository,
        engine=engine,
        process_journal=process_journal,
    )
    event_bus = SingleMachineEventBus(
        event_store,
        process_journal,
        process_event,
        max_workers=resolved_settings.worker_max_workers,
    )
    selected_dialogue_model = dialogue_model or RuleBasedDialogueModel()
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
        ),
    )
    return ApplicationContainer(
        settings=resolved_settings,
        secret_source=DotenvSecretSource(Path(dotenv_path)),
        event_store=event_store,
        evidence_repository=evidence_repository,
        state_repository=state_repository,
        process_event=process_event,
        event_bus=event_bus,
        replay=ReplayService(event_store=event_store, engine=engine),
        workspace_builder=WorkspaceBuilder(),
        dialogue_model=selected_dialogue_model,
        module_registry=module_registry,
        lifecycle=lifecycle,
    )


def _default_module_registrations() -> tuple[ModuleRegistration, ...]:
    return (
        ModuleRegistration(
            "semantic.preference_extractor",
            "semantic",
            "1",
            PreferenceExtractor(),
        ),
        ModuleRegistration(
            "semantic.name_extractor",
            "semantic",
            "1",
            NameExtractor(),
        ),
        ModuleRegistration(
            "episodic.memory_extractor",
            "episodic",
            "1",
            EpisodicMemoryExtractor(),
        ),
        ModuleRegistration(
            "relationship.relationship_extractor",
            "relationship",
            "1",
            RelationshipExtractor(),
        ),
        ModuleRegistration(
            "metacognition.conflict_extractor",
            "metacognition",
            "1",
            ConflictMetacognitionExtractor(),
        ),
        ModuleRegistration(
            "identity.identity_value_extractor",
            "identity",
            "1",
            IdentityValueExtractor(),
        ),
        ModuleRegistration(
            "affect.affect_extractor",
            "affect",
            "1",
            AffectExtractor(),
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

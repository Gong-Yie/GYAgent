from dataclasses import dataclass
from pathlib import Path

from self_cognition.application.process_event import ProcessEventService
from self_cognition.application.replay import ReplayService
from self_cognition.cognition.affect.affect_extractor import AffectExtractor
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
from self_cognition.core.protocols import EventStore, StateRepository
from self_cognition.core.workspace import WorkspaceBuilder
from self_cognition.cognition.semantic.name_extractor import NameExtractor
from self_cognition.cognition.semantic.preference_extractor import (
    PreferenceExtractor,
)
from self_cognition.executive.dialogue.rule_based import RuleBasedDialogueModel
from self_cognition.infrastructure.persistence.file_event_store import FileEventStore
from self_cognition.infrastructure.persistence.file_state_repository import (
    FileStateRepository,
)
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.runtime.reducer import StateReducer


@dataclass(frozen=True, slots=True)
class ApplicationContainer:
    event_store: EventStore
    state_repository: StateRepository
    process_event: ProcessEventService
    replay: ReplayService
    workspace_builder: WorkspaceBuilder
    dialogue_model: RuleBasedDialogueModel


def build_container(data_dir: str | Path = "data") -> ApplicationContainer:
    root = Path(data_dir)
    event_store = FileEventStore(root / "events.jsonl")
    state_repository = FileStateRepository(root / "states")
    engine = CognitionEngine(
        modules=(
            PreferenceExtractor(),
            NameExtractor(),
            EpisodicMemoryExtractor(),
            RelationshipExtractor(),
            ConflictMetacognitionExtractor(),
            IdentityValueExtractor(),
            AffectExtractor(),
            NarrativeExtractor(),
        ),
        reducer=StateReducer(),
    )
    return ApplicationContainer(
        event_store=event_store,
        state_repository=state_repository,
        process_event=ProcessEventService(
            event_store=event_store,
            state_repository=state_repository,
            engine=engine,
        ),
        replay=ReplayService(event_store=event_store, engine=engine),
        workspace_builder=WorkspaceBuilder(),
        dialogue_model=RuleBasedDialogueModel(),
    )

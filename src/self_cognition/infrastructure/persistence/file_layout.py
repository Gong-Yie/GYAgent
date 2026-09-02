from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FileDataLayout:
    """Canonical directories for file-backed authoritative and derived data."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @property
    def events(self) -> Path:
        return self.root / "events"

    @property
    def evidence(self) -> Path:
        return self.root / "evidence"

    @property
    def states(self) -> Path:
        return self.root / "states"

    @property
    def memories(self) -> Path:
        return self.root / "memories"

    @property
    def memory_access(self) -> Path:
        return self.root / "memory_access"

    @property
    def deletions(self) -> Path:
        return self.root / "deletions"

    @property
    def processing(self) -> Path:
        return self.root / "processing"

    @property
    def indexes(self) -> Path:
        return self.root / "indexes"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @property
    def backups(self) -> Path:
        return self.root / "backups"

    @property
    def blobs(self) -> Path:
        return self.root / "blobs"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def event_log(self) -> Path:
        """Return the new event path while reading the pre-layout legacy path."""

        canonical = self.events / "events.jsonl"
        legacy = self.root / "events.jsonl"
        if canonical.exists() or not legacy.exists():
            return canonical
        return legacy

    def ensure(self) -> "FileDataLayout":
        for directory in self.directories:
            directory.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def directories(self) -> tuple[Path, ...]:
        return (
            self.events,
            self.evidence,
            self.states,
            self.memories,
            self.memory_access,
            self.deletions,
            self.processing,
            self.indexes,
            self.runs,
            self.exports,
            self.backups,
            self.blobs,
            self.cache,
            self.logs,
        )

import json
import os
from threading import RLock
from pathlib import Path
from uuid import UUID

from self_cognition.core.actions import action_dependency_ids
from self_cognition.core.errors import MalformedSerializedDataError
from self_cognition.core.events import EventEnvelope
from self_cognition.core.dialogue import dialogue_dependency_ids
from self_cognition.core.plans import planning_dependency_ids
from self_cognition.core.proactivity import proactive_dependency_ids
from self_cognition.core.scopes import MindScope, SubjectScope
from self_cognition.infrastructure.persistence.serialization import (
    event_from_json,
    event_to_json,
)
from self_cognition.infrastructure.persistence.atomic_io import atomic_write_text
from self_cognition.infrastructure.persistence.file_lock import BlockingFileLock


class FileEventStore:
    """Append-only event log on the local filesystem.

    The log is the authoritative store, so an instance must observe records that
    another process appended after it was built: a background worker clears
    outbox entries for events a sibling process wrote, and a deletion performed
    elsewhere has to take effect here without a restart. Each instance therefore
    keeps a byte offset into the log and parses only the appended tail, under the
    same file lock writers use, so the offset always lands on a record boundary.
    """

    def __init__(
        self,
        path: str | Path,
        tombstone_path: str | Path | None = None,
    ) -> None:
        self._path = Path(path)
        self._append_lock_path = self._path.parent / f"{self._path.name}.lock"
        self._tombstone_path = Path(
            tombstone_path or self._path.parent / "event_tombstones.jsonl"
        )
        self._lock = RLock()
        self._tombstones = self._read_tombstones()
        self._tombstone_size = self._file_size(self._tombstone_path)
        self._events: list[EventEnvelope] = []
        self._event_ids: set[UUID] = set()
        self._offset = 0
        self._record_count = 0
        with BlockingFileLock(self._append_lock_path):
            self._reload_locked()

    def append(self, event: EventEnvelope) -> None:
        with self._lock:
            self._refresh_tombstones_unlocked()
            with BlockingFileLock(self._append_lock_path):
                self._ingest_locked()
                if self._is_duplicate_unlocked(event):
                    return
                self._append_locked(event)

    def append_many(self, events: tuple[EventEnvelope, ...]) -> None:
        with self._lock:
            self._refresh_tombstones_unlocked()
            with BlockingFileLock(self._append_lock_path):
                self._ingest_locked()
                pending = tuple(
                    event for event in events if not self._is_duplicate_unlocked(event)
                )
                if not pending:
                    return
                if len({event.event_id for event in pending}) != len(pending):
                    raise ValueError("event batch contains duplicate IDs")
                for event in pending:
                    self._append_locked(event)

    def read_by_subject(
        self,
        subject: SubjectScope,
    ) -> tuple[EventEnvelope, ...]:
        if not isinstance(subject, SubjectScope):
            raise TypeError("subject must be a SubjectScope")
        with self._lock:
            self._refresh_unlocked()
            return tuple(
                event for event in self._events if event.subject == subject
            )

    def read_all(self) -> tuple[EventEnvelope, ...]:
        with self._lock:
            self._refresh_unlocked()
            return tuple(self._events)

    def read_by_mind(self, mind: MindScope) -> tuple[EventEnvelope, ...]:
        if not isinstance(mind, MindScope):
            raise TypeError("mind must be a MindScope")
        with self._lock:
            self._refresh_unlocked()
            return tuple(event for event in self._events if event.subject.mind == mind)

    def redact(
        self,
        subject: SubjectScope,
        event_ids: tuple[UUID, ...],
        plan_id: UUID,
    ) -> None:
        targets = set(event_ids)
        with self._lock:
            # The rewrite replaces the whole log, so it has to see every event a
            # sibling process appended, and it has to happen under the append
            # lock: otherwise the rebuild would silently drop their records, and
            # a reader holding the lock could observe the file mid-replacement.
            with BlockingFileLock(self._append_lock_path):
                self._refresh_tombstones_unlocked()
                self._ingest_locked()
                if any(
                    event.event_id in targets and event.subject != subject
                    for event in self._events
                ):
                    raise ValueError(
                        "cannot redact an event owned by another subject"
                    )
                new_tombstones = targets - self._tombstones
                if new_tombstones:
                    self._append_tombstones(plan_id, new_tombstones)
                    self._tombstones.update(new_tombstones)
                    self._tombstone_size = self._file_size(self._tombstone_path)
                self._events = [
                    event for event in self._events if event.event_id not in targets
                ]
                self._event_ids.difference_update(targets)
                self._record_count = len(self._events)
                payload = "".join(
                    event_to_json(event) + "\n" for event in self._events
                )
                atomic_write_text(self._path, payload)
                self._offset = len(payload.encode("utf-8"))

    def _append_locked(self, event: EventEnvelope) -> None:
        """Write one record. The caller holds the append lock.

        Holding the lock across the write is what makes ``_offset`` exact: no
        sibling process can slip a record in between this instance reading its
        offset and writing at it, which would otherwise leave the offset inside
        a record and break the next tail parse.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        record = event_to_json(event) + "\n"
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(record)
            handle.flush()
            os.fsync(handle.fileno())
        self._offset += len(record.encode("utf-8"))
        self._record_count += 1
        self._events.append(event)
        self._event_ids.add(event.event_id)

    def _refresh_unlocked(self) -> None:
        """Pick up events and tombstones another process appended meanwhile.

        Only the appended tail is parsed. Re-reading the whole log on every
        observed external append would cost more the longer the log grows, and a
        sharded deployment observes an external append on most reads.

        Tombstones are re-read only once the log itself has changed, because
        redact always rewrites the log right after recording them. Probing for a
        tombstone file that does not exist is several times the cost of a hit,
        so keeping that off the common read path matters.
        """
        if self._file_size(self._path) == self._offset:
            return
        with BlockingFileLock(self._append_lock_path):
            self._refresh_tombstones_unlocked()
            self._ingest_locked()

    def _ingest_locked(self) -> None:
        """Absorb everything appended past the current offset.

        The caller holds the append lock, so the tail it reads ends on a record
        boundary and cannot be extended underneath it.
        """
        if self._file_size(self._path) < self._offset:
            # redact rewrites the log, so a byte offset no longer means anything.
            self._reload_locked()
            return
        data = self._read_bytes(self._offset)
        if not data:
            return
        self._absorb(self._decode_records(data, self._record_count + 1), len(data))

    def _reload_locked(self) -> None:
        """Re-read the whole log. The caller holds the append lock."""
        data = self._read_bytes(0)
        self._events = []
        self._event_ids = set()
        self._offset = 0
        self._record_count = 0
        self._absorb(self._decode_records(data), len(data))

    def _absorb(
        self,
        records: tuple[EventEnvelope, ...],
        byte_count: int,
    ) -> None:
        for event in records:
            if event.event_id in self._event_ids or event.event_id in self._tombstones:
                continue
            self._events.append(event)
            self._event_ids.add(event.event_id)
        self._offset += byte_count
        self._record_count += len(records)

    def _refresh_tombstones_unlocked(self) -> None:
        size = self._file_size(self._tombstone_path)
        if size == self._tombstone_size:
            return
        # Measured before the read, so a concurrent append can only make the
        # recorded size too small, which costs one redundant re-read.
        self._tombstone_size = size
        self._tombstones = self._read_tombstones()
        withdrawn = {
            event.event_id
            for event in self._events
            if event.event_id in self._tombstones
        }
        if not withdrawn:
            return
        self._events = [
            event for event in self._events if event.event_id not in withdrawn
        ]
        self._event_ids.difference_update(withdrawn)

    def _read_bytes(self, offset: int) -> bytes:
        """Read from a byte offset to the end. The caller holds the append lock."""
        try:
            with self._path.open("rb") as handle:
                handle.seek(offset)
                return handle.read()
        except FileNotFoundError:
            return b""

    @staticmethod
    def _decode_records(
        data: bytes,
        start_line: int = 1,
    ) -> tuple[EventEnvelope, ...]:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise MalformedSerializedDataError(
                "event log is not valid UTF-8"
            ) from error
        # Records are written as one JSON document plus "\n", and JSON escapes
        # every control character, so splitting on "\n" is exact.
        lines = text.split("\n")
        if lines:
            lines.pop()
        events: list[EventEnvelope] = []
        for index, line in enumerate(lines):
            line_number = start_line + index
            if not line.strip():
                raise MalformedSerializedDataError(
                    f"blank event record at line {line_number}"
                )
            try:
                events.append(event_from_json(line))
            except MalformedSerializedDataError as error:
                raise MalformedSerializedDataError(
                    f"invalid event record at line {line_number}"
                ) from error
        return tuple(events)

    def _read_tombstones(self) -> set[UUID]:
        try:
            data = self._tombstone_path.read_bytes()
        except FileNotFoundError:
            return set()
        # A writer may be mid-append, and a final line without its terminator is
        # an unfinished write rather than corruption; complete lines still have
        # to parse.
        lines = data.split(b"\n")
        if lines:
            lines.pop()
        tombstones: set[UUID] = set()
        for line_number, raw in enumerate(lines, start=1):
            try:
                values = json.loads(raw)
                tombstones.add(UUID(values["event_id"]))
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
                UnicodeDecodeError,
            ) as error:
                raise MalformedSerializedDataError(
                    f"invalid event tombstone at line {line_number}"
                ) from error
        return tombstones

    def _append_tombstones(self, plan_id: UUID, event_ids: set[UUID]) -> None:
        self._tombstone_path.parent.mkdir(parents=True, exist_ok=True)
        with self._tombstone_path.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            for event_id in sorted(event_ids, key=lambda value: value.int):
                handle.write(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "plan_id": str(plan_id),
                            "event_id": str(event_id),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())

    def _is_duplicate_unlocked(self, event: EventEnvelope) -> bool:
        return (
            event.event_id in self._event_ids
            or event.event_id in self._tombstones
            or bool(
                (
                    action_dependency_ids(event)
                    | dialogue_dependency_ids(event)
                    | planning_dependency_ids(event)
                    | proactive_dependency_ids(event)
                )
                & self._tombstones
            )
        )

    @staticmethod
    def _file_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

from pathlib import Path
from uuid import UUID

from self_cognition.core.errors import MalformedSerializedDataError
from self_cognition.core.events import (
    EventEnvelope,
    ProcessingFailurePayload,
)
from self_cognition.core.processing import ProcessingRecord, ProcessingStatus
from self_cognition.core.scopes import SubjectScope
from self_cognition.infrastructure.persistence.file_process_journal import (
    FileProcessJournal,
)
from self_cognition.infrastructure.persistence.serialization import event_from_json


class FileProcessingRecovery:
    def __init__(
        self,
        event_log: str | Path,
        journal: FileProcessJournal,
    ) -> None:
        self._event_log = Path(event_log)
        self._journal = journal

    def reconcile(self) -> tuple[ProcessingRecord, ...]:
        events = self._read_events()
        reductions = self._terminal_events(events, "state.reduced")
        failures = self._terminal_events(events, "processing.failed")
        records = []
        for event in events:
            if event.event_type != "user.message":
                continue
            key = (event.event_id, event.subject)
            reduction = reductions.get(key)
            failure = failures.get(key)
            if reduction is not None:
                record = self._journal.recover(
                    event,
                    ProcessingStatus.COMPLETED,
                    reduction.recorded_at,
                )
            elif failure is not None:
                payload = failure.payload
                if not isinstance(payload, ProcessingFailurePayload):
                    raise MalformedSerializedDataError(
                        "processing.failed payload has an invalid type"
                    )
                record = self._journal.recover(
                    event,
                    ProcessingStatus.FAILED,
                    failure.recorded_at,
                    error_type=payload.error_type,
                )
            else:
                record = self._journal.recover(
                    event,
                    ProcessingStatus.PENDING,
                    event.recorded_at,
                )
            records.append(record)
        return tuple(records)

    @staticmethod
    def _terminal_events(
        events: tuple[EventEnvelope, ...],
        event_type: str,
    ) -> dict[tuple[UUID, SubjectScope], EventEnvelope]:
        return {
            (event.causation_id, event.subject): event
            for event in events
            if event.event_type == event_type and event.causation_id is not None
        }

    def _read_events(self) -> tuple[EventEnvelope, ...]:
        if not self._event_log.exists():
            return ()
        events = []
        try:
            with self._event_log.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
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
        except UnicodeError as error:
            raise MalformedSerializedDataError(
                "event log is not valid UTF-8"
            ) from error
        return tuple(events)

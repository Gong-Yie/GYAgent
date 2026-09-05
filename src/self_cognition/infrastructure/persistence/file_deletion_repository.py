import json
from datetime import datetime
from pathlib import Path
from uuid import UUID

from self_cognition.core.deletions import (
    DeletionImpact,
    InvalidatedModuleResult,
    DeletionPlan,
    DeletionSelector,
    DeletionStatus,
)
from self_cognition.core.errors import (
    ContractValidationError,
    MalformedSerializedDataError,
)
from self_cognition.core.memories import MemoryType
from self_cognition.core.scopes import MindScope, SubjectKind, SubjectRef, SubjectScope
from self_cognition.infrastructure.persistence.atomic_io import atomic_write_text
from self_cognition.infrastructure.persistence.file_lock import FileLock


class FileDeletionRepository:
    def __init__(self, directory: str | Path) -> None:
        root = Path(directory)
        self._current = root / "current"
        self._history = root / "history"
        self._locks = root / "locks"
        for path in (self._current, self._history, self._locks):
            path.mkdir(parents=True, exist_ok=True)

    def save(self, plan: DeletionPlan) -> None:
        with FileLock(self._lock_path(plan.plan_id)):
            current = self.get(plan.plan_id)
            self._validate_transition(current, plan)
            if current == plan:
                return
            payload = _plan_to_json(plan)
            atomic_write_text(self._current_path(plan.plan_id), payload)
            history_path = self._history_path(plan.plan_id)
            history = (
                history_path.read_text(encoding="utf-8")
                if history_path.exists()
                else ""
            )
            atomic_write_text(history_path, history + payload + "\n")

    def get(self, plan_id: UUID) -> DeletionPlan | None:
        path = self._current_path(plan_id)
        if not path.exists():
            return None
        return _plan_from_json(path.read_text(encoding="utf-8"))

    def read_by_status(
        self,
        status: DeletionStatus,
    ) -> tuple[DeletionPlan, ...]:
        return tuple(
            plan
            for path in sorted(self._current.glob("*.json"))
            if (plan := _plan_from_json(path.read_text(encoding="utf-8"))).status
            is status
        )

    @staticmethod
    def _validate_transition(
        current: DeletionPlan | None,
        new: DeletionPlan,
    ) -> None:
        if current is None:
            if new.status is not DeletionStatus.PLANNED:
                raise ValueError("new deletion must start as planned")
            return
        if current.digest != new.digest:
            raise ValueError("deletion plan contents cannot change")
        allowed = {
            DeletionStatus.PLANNED: {DeletionStatus.EXECUTING},
            DeletionStatus.EXECUTING: {
                DeletionStatus.EXECUTING,
                DeletionStatus.COMPLETED,
                DeletionStatus.FAILED,
            },
            DeletionStatus.FAILED: {DeletionStatus.EXECUTING},
            DeletionStatus.COMPLETED: {DeletionStatus.COMPLETED},
        }
        if new.status not in allowed[current.status]:
            raise ValueError("invalid deletion status transition")

    def _current_path(self, plan_id: UUID) -> Path:
        return self._current / f"{plan_id}.json"

    def _history_path(self, plan_id: UUID) -> Path:
        return self._history / f"{plan_id}.jsonl"

    def _lock_path(self, plan_id: UUID) -> Path:
        return self._locks / f"{plan_id}.lock"


def _plan_to_json(plan: DeletionPlan) -> str:
    selector = plan.selector
    payload = {
        "schema_version": 2 if plan.impacts else 1,
        "plan_id": str(plan.plan_id),
        "digest": plan.digest,
        "selector": {
            "mind_id": selector.subject.mind.mind_id,
            "subject_kind": selector.subject.subject.kind.value,
            "subject_id": selector.subject.subject.subject_id,
            "memory_id": str(selector.memory_id) if selector.memory_id else None,
            "memory_types": [value.value for value in selector.memory_types],
            "created_from": (
                selector.created_from.isoformat() if selector.created_from else None
            ),
            "created_to": (
                selector.created_to.isoformat() if selector.created_to else None
            ),
            "conversation_id": selector.conversation_id,
            "delete_subject": selector.delete_subject,
        },
        "memory_ids": [str(value) for value in plan.memory_ids],
        "event_ids": [str(value) for value in plan.event_ids],
        "created_at": plan.created_at.isoformat(),
        "reason": plan.reason,
        "status": plan.status.value,
        "status_updated_at": (
            plan.status_updated_at.isoformat()
            if plan.status_updated_at is not None
            else None
        ),
        "failure_type": plan.failure_type,
        "cache_result": plan.cache_result,
        "export_result": plan.export_result,
    }
    if plan.impacts:
        payload["impacts"] = [item.to_dict() for item in plan.impacts]
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _plan_from_json(payload: str) -> DeletionPlan:
    try:
        values = json.loads(payload)
        if values.get("schema_version") not in (1, 2):
            raise ValueError("unsupported deletion schema")
        selector_values = values["selector"]
        selector = DeletionSelector(
            subject=SubjectScope(
                MindScope(selector_values["mind_id"]),
                SubjectRef(
                    SubjectKind(selector_values["subject_kind"]),
                    selector_values["subject_id"],
                ),
            ),
            memory_id=(
                UUID(selector_values["memory_id"])
                if selector_values["memory_id"] is not None
                else None
            ),
            memory_types=tuple(
                MemoryType(value) for value in selector_values["memory_types"]
            ),
            created_from=_optional_datetime(selector_values["created_from"]),
            created_to=_optional_datetime(selector_values["created_to"]),
            conversation_id=selector_values["conversation_id"],
            delete_subject=selector_values["delete_subject"],
        )
        plan = DeletionPlan(
            plan_id=UUID(values["plan_id"]),
            selector=selector,
            memory_ids=tuple(UUID(value) for value in values["memory_ids"]),
            event_ids=tuple(UUID(value) for value in values["event_ids"]),
            created_at=datetime.fromisoformat(values["created_at"]),
            reason=values["reason"],
            status=DeletionStatus(values["status"]),
            status_updated_at=_optional_datetime(values["status_updated_at"]),
            failure_type=values["failure_type"],
            cache_result=values["cache_result"],
            export_result=values["export_result"],
            impacts=(
                tuple(
                    DeletionImpact(
                        subject=SubjectScope(
                            MindScope(item["mind_id"]),
                            SubjectRef(
                                SubjectKind(item["subject_kind"]), item["subject_id"]
                            ),
                        ),
                        event_ids=tuple(UUID(value) for value in item["event_ids"]),
                        memory_ids=tuple(UUID(value) for value in item["memory_ids"]),
                        invalidated_results=tuple(
                            InvalidatedModuleResult(
                                event_id=UUID(result["event_id"]),
                                cause_id=UUID(result["cause_id"]),
                                module_id=result["module_id"],
                                module_version=result["module_version"],
                                deterministic=result["deterministic"],
                            )
                            for result in item["invalidated_results"]
                        ),
                    )
                    for item in values["impacts"]
                )
                if values["schema_version"] == 2
                else ()
            ),
        )
        if values["digest"] != plan.digest:
            raise ValueError("deletion plan digest does not match contents")
        return plan
    except (
        ContractValidationError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise MalformedSerializedDataError("invalid deletion plan") from error


def _optional_datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Mapping
from uuid import UUID

from self_cognition.core.errors import ContractValidationError
from self_cognition.core.scopes import (
    MindScope,
    SubjectKind,
    SubjectRef,
    SubjectScope,
)


PROVENANCE_SCHEMA_VERSION = 1


class ProvenanceNodeKind(str, Enum):
    EVENT = "event"
    CONTRIBUTION = "contribution"
    MEMORY = "memory"
    EMOTION = "emotion"
    MOOD = "mood"


class ProvenanceEdgeKind(str, Enum):
    CAUSED_BY = "caused_by"
    DERIVED_FROM = "derived_from"
    SUPERSEDES = "supersedes"


@dataclass(frozen=True, slots=True)
class ProvenanceNode:
    node_id: UUID
    kind: ProvenanceNodeKind
    subject: SubjectScope
    recorded_at: datetime
    attributes: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, UUID):
            raise ContractValidationError("provenance node_id must be a UUID")
        if not isinstance(self.kind, ProvenanceNodeKind):
            raise ContractValidationError("provenance node kind is invalid")
        if not isinstance(self.subject, SubjectScope):
            raise ContractValidationError("provenance node subject is invalid")
        _require_aware(self.recorded_at, "provenance node recorded_at")
        if not isinstance(self.attributes, Mapping):
            raise ContractValidationError("provenance node attributes must be a mapping")

    def to_state_value(self) -> dict[str, object]:
        return {
            "node_id": str(self.node_id),
            "kind": self.kind.value,
            "subject": _subject_to_value(self.subject),
            "recorded_at": self.recorded_at.isoformat(),
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_state_value(cls, value: object) -> "ProvenanceNode":
        data = _require_mapping(value, "provenance node")
        return cls(
            node_id=_require_uuid(data.get("node_id"), "provenance node.node_id"),
            kind=_require_enum(
                ProvenanceNodeKind,
                data.get("kind"),
                "provenance node.kind",
            ),
            subject=_subject_from_value(data.get("subject")),
            recorded_at=_require_datetime(
                data.get("recorded_at"),
                "provenance node.recorded_at",
            ),
            attributes=_require_mapping(
                data.get("attributes"),
                "provenance node.attributes",
            ),
        )


@dataclass(frozen=True, slots=True)
class ProvenanceEdge:
    edge_id: UUID
    kind: ProvenanceEdgeKind
    source_id: UUID
    target_id: UUID
    subject: SubjectScope
    recorded_at: datetime
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.edge_id, UUID):
            raise ContractValidationError("provenance edge_id must be a UUID")
        if not isinstance(self.kind, ProvenanceEdgeKind):
            raise ContractValidationError("provenance edge kind is invalid")
        if not isinstance(self.source_id, UUID) or not isinstance(self.target_id, UUID):
            raise ContractValidationError("provenance edge endpoints must be UUIDs")
        if not isinstance(self.subject, SubjectScope):
            raise ContractValidationError("provenance edge subject is invalid")
        _require_aware(self.recorded_at, "provenance edge recorded_at")
        if not isinstance(self.attributes, Mapping):
            raise ContractValidationError("provenance edge attributes must be a mapping")

    def to_state_value(self) -> dict[str, object]:
        return {
            "edge_id": str(self.edge_id),
            "kind": self.kind.value,
            "source_id": str(self.source_id),
            "target_id": str(self.target_id),
            "subject": _subject_to_value(self.subject),
            "recorded_at": self.recorded_at.isoformat(),
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_state_value(cls, value: object) -> "ProvenanceEdge":
        data = _require_mapping(value, "provenance edge")
        return cls(
            edge_id=_require_uuid(data.get("edge_id"), "provenance edge.edge_id"),
            kind=_require_enum(
                ProvenanceEdgeKind,
                data.get("kind"),
                "provenance edge.kind",
            ),
            source_id=_require_uuid(
                data.get("source_id"),
                "provenance edge.source_id",
            ),
            target_id=_require_uuid(
                data.get("target_id"),
                "provenance edge.target_id",
            ),
            subject=_subject_from_value(data.get("subject")),
            recorded_at=_require_datetime(
                data.get("recorded_at"),
                "provenance edge.recorded_at",
            ),
            attributes=_require_mapping(
                data.get("attributes"),
                "provenance edge.attributes",
            ),
        )


@dataclass(frozen=True, slots=True)
class ProvenanceGraph:
    mind_id: str
    nodes: tuple[ProvenanceNode, ...]
    edges: tuple[ProvenanceEdge, ...]
    schema_version: int = PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.mind_id, str) or not self.mind_id.strip():
            raise ContractValidationError("provenance graph mind_id must not be blank")
        if self.schema_version != PROVENANCE_SCHEMA_VERSION:
            raise ContractValidationError("provenance graph schema version is invalid")
        node_by_id = {node.node_id: node for node in self.nodes}
        if len(node_by_id) != len(self.nodes):
            raise ContractValidationError("provenance graph node IDs must be unique")
        edge_ids = {edge.edge_id for edge in self.edges}
        if len(edge_ids) != len(self.edges):
            raise ContractValidationError("provenance graph edge IDs must be unique")
        for node in self.nodes:
            if node.subject.mind.mind_id != self.mind_id:
                raise ContractValidationError("provenance node crosses minds")
        for edge in self.edges:
            if edge.subject.mind.mind_id != self.mind_id:
                raise ContractValidationError("provenance edge crosses minds")
            if edge.source_id not in node_by_id or edge.target_id not in node_by_id:
                raise ContractValidationError(
                    "provenance edge endpoints must exist in the graph"
                )

    def node(self, node_id: UUID) -> ProvenanceNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def outgoing(self, node_id: UUID) -> tuple[ProvenanceEdge, ...]:
        return tuple(edge for edge in self.edges if edge.source_id == node_id)

    def incoming(self, node_id: UUID) -> tuple[ProvenanceEdge, ...]:
        return tuple(edge for edge in self.edges if edge.target_id == node_id)

    def provenance_chain(self, node_id: UUID) -> tuple[ProvenanceNode, ...]:
        self.node(node_id)
        seen: set[UUID] = set()
        ordered: list[ProvenanceNode] = []
        pending = [node_id]
        while pending:
            current = pending.pop(0)
            if current in seen:
                continue
            seen.add(current)
            ordered.append(self.node(current))
            for edge in sorted(
                self.outgoing(current),
                key=lambda item: (item.kind.value, item.edge_id.int),
            ):
                if edge.target_id not in seen:
                    pending.append(edge.target_id)
        return tuple(ordered)

    def source_events(self, node_id: UUID) -> tuple[ProvenanceNode, ...]:
        return tuple(
            node
            for node in self.provenance_chain(node_id)
            if node.kind is ProvenanceNodeKind.EVENT
        )

    def to_state_value(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mind_id": self.mind_id,
            "nodes": [node.to_state_value() for node in self.nodes],
            "edges": [edge.to_state_value() for edge in self.edges],
        }

    @classmethod
    def from_state_value(cls, value: object) -> "ProvenanceGraph":
        data = _require_mapping(value, "provenance graph")
        nodes = data.get("nodes")
        edges = data.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ContractValidationError("provenance graph nodes and edges must be arrays")
        return cls(
            mind_id=_require_text(data.get("mind_id"), "provenance graph.mind_id"),
            nodes=tuple(ProvenanceNode.from_state_value(item) for item in nodes),
            edges=tuple(ProvenanceEdge.from_state_value(item) for item in edges),
            schema_version=_require_int(
                data.get("schema_version"),
                "provenance graph.schema_version",
            ),
        )


def _subject_to_value(subject: SubjectScope) -> dict[str, str]:
    return {
        "mind_id": subject.mind.mind_id,
        "subject_kind": subject.subject.kind.value,
        "subject_id": subject.subject.subject_id,
    }


def _subject_from_value(value: object) -> SubjectScope:
    data = _require_mapping(value, "provenance subject")
    try:
        return SubjectScope(
            MindScope(_require_text(data.get("mind_id"), "provenance subject.mind_id")),
            SubjectRef(
                SubjectKind(
                    _require_text(
                        data.get("subject_kind"),
                        "provenance subject.subject_kind",
                    )
                ),
                _require_text(
                    data.get("subject_id"),
                    "provenance subject.subject_id",
                ),
            ),
        )
    except ValueError as error:
        raise ContractValidationError("provenance subject kind is invalid") from error


def _require_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ContractValidationError(f"{name} must be an object")
    return value


def _require_uuid(value: object, name: str) -> UUID:
    try:
        return UUID(str(value))
    except (TypeError, ValueError) as error:
        raise ContractValidationError(f"{name} must be a UUID") from error


def _require_datetime(value: object, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError) as error:
        raise ContractValidationError(f"{name} must be an ISO datetime") from error
    _require_aware(parsed, name)
    return parsed


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{name} must be non-blank text")
    return value


def _require_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractValidationError(f"{name} must be an integer")
    return value


def _require_enum(enum_type, value: object, name: str):
    try:
        return enum_type(str(value))
    except ValueError as error:
        raise ContractValidationError(f"{name} is invalid") from error


def _require_aware(value: datetime, name: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ContractValidationError(f"{name} must include a timezone")

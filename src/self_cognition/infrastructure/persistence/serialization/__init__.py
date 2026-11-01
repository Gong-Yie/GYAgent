"""Serialization codecs for persisted aggregates.

The historical import path
``self_cognition.infrastructure.persistence.serialization`` stays stable; the
implementation lives in one module per aggregate so that event, memory and
state persistence can evolve without sharing a single 2k-line module.
"""

from self_cognition.core.events import EVENT_SCHEMA_VERSION

from ._events import event_from_dict, event_from_json, event_to_dict, event_to_json
from ._memories import (
    MEMORY_ACCESS_SCHEMA_VERSION,
    MEMORY_SCHEMA_VERSION,
    memory_access_from_dict,
    memory_access_from_json,
    memory_access_to_dict,
    memory_access_to_json,
    memory_from_dict,
    memory_from_json,
    memory_to_dict,
    memory_to_json,
)
from ._state import (
    LEGACY_STATE_TIME,
    STATE_SCHEMA_VERSION,
    state_from_dict,
    state_from_json,
    state_to_dict,
    state_to_json,
)

__all__ = [
    "EVENT_SCHEMA_VERSION",
    "LEGACY_STATE_TIME",
    "MEMORY_ACCESS_SCHEMA_VERSION",
    "MEMORY_SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "event_from_dict",
    "event_from_json",
    "event_to_dict",
    "event_to_json",
    "memory_access_from_dict",
    "memory_access_from_json",
    "memory_access_to_dict",
    "memory_access_to_json",
    "memory_from_dict",
    "memory_from_json",
    "memory_to_dict",
    "memory_to_json",
    "state_from_dict",
    "state_from_json",
    "state_to_dict",
    "state_to_json",
]

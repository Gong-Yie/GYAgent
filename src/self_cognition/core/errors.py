class CognitionError(Exception):
    """Base exception for self-cognition domain errors."""


class ContractValidationError(CognitionError):
    """Raised when a core data contract receives invalid data."""


class SubjectMismatchError(CognitionError):
    """Raised when data is applied to a different subject."""


class ScopeMismatchError(CognitionError):
    """Raised when data is accessed from a different mind scope."""


class VersionConflictError(CognitionError):
    """Raised when a state save uses a stale or invalid version."""


class RunCancelledError(CognitionError):
    """Raised internally when a run is cancelled at a safe boundary."""


class SerializationError(CognitionError):
    """Raised when persisted data cannot be decoded or encoded safely."""


class MalformedSerializedDataError(SerializationError):
    """Raised when serialized data has invalid structure or values."""


class UnsupportedSchemaVersionError(SerializationError):
    """Raised when serialized data uses an unsupported schema version."""


class FileLockUnavailableError(CognitionError):
    """Raised when an exclusive file lock is already held."""


class ModelOutputError(CognitionError):
    """Raised when a model response violates the cognition output contract."""


class ModelTimeoutError(CognitionError):
    """Raised when a cognition model exceeds its allowed response time."""

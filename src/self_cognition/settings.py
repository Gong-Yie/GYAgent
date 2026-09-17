from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


SETTINGS_SCHEMA_VERSION = 1
MAX_MODEL_OUTPUT_TOKENS = 393216
_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class ApplicationSettings:
    schema_version: int = SETTINGS_SCHEMA_VERSION
    data_dir: Path = Path("data")
    enabled_modules: frozenset[str] | None = None
    worker_enabled: bool = False
    worker_poll_interval_seconds: float = 0.1
    worker_max_workers: int = 4
    worker_lease_timeout_seconds: float = 300.0
    cognition_max_output_tokens: int = 2048
    dialogue_max_output_tokens: int = 4096
    model_temperature: float = 0.0
    model_timeout_seconds: float = 120.0
    model_failure_cooldown_seconds: float = 30.0
    model_max_failure_cooldown_seconds: float = 900.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        if self.schema_version != SETTINGS_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported settings schema version: {self.schema_version}"
            )
        if self.worker_poll_interval_seconds <= 0:
            raise ValueError("worker poll interval must be positive")
        if self.worker_max_workers < 1:
            raise ValueError("worker max workers must be positive")
        if self.worker_lease_timeout_seconds <= 0:
            raise ValueError("worker lease timeout must be positive")
        if self.cognition_max_output_tokens < 1:
            raise ValueError("cognition max output tokens must be positive")
        if self.dialogue_max_output_tokens < 1:
            raise ValueError("dialogue max output tokens must be positive")
        if self.cognition_max_output_tokens > MAX_MODEL_OUTPUT_TOKENS:
            raise ValueError(
                "cognition max output tokens must not exceed "
                f"{MAX_MODEL_OUTPUT_TOKENS}"
            )
        if self.dialogue_max_output_tokens > MAX_MODEL_OUTPUT_TOKENS:
            raise ValueError(
                "dialogue max output tokens must not exceed "
                f"{MAX_MODEL_OUTPUT_TOKENS}"
            )
        if not 0.0 <= self.model_temperature <= 2.0:
            raise ValueError("model temperature must be between 0 and 2")
        if self.model_timeout_seconds <= 0:
            raise ValueError("model timeout must be positive")
        if self.model_failure_cooldown_seconds <= 0:
            raise ValueError("model failure cooldown must be positive")
        if (
            self.model_max_failure_cooldown_seconds
            < self.model_failure_cooldown_seconds
        ):
            raise ValueError(
                "model maximum failure cooldown must not be shorter than base"
            )


@dataclass(frozen=True, slots=True)
class DotenvSecretSource:
    dotenv_path: Path = Path(".env")
    environ: Mapping[str, str] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def get(self, name: str) -> str | None:
        if _KEY_PATTERN.fullmatch(name) is None:
            raise ValueError("secret name must be an environment variable name")
        process_values = os.environ if self.environ is None else self.environ
        value = process_values.get(name)
        if value is None:
            value = _read_dotenv(self.dotenv_path).get(name)
        return value or None


def load_settings(
    dotenv_path: str | Path = ".env",
    *,
    environ: Mapping[str, str] | None = None,
) -> ApplicationSettings:
    values = _read_dotenv(Path(dotenv_path))
    values.update(dict(os.environ if environ is None else environ))
    enabled_modules = _enabled_modules(values.get("SC_ENABLED_MODULES"))
    return ApplicationSettings(
        schema_version=_integer(values, "SC_CONFIG_VERSION", 1),
        data_dir=Path(values.get("SC_DATA_DIR", "data")),
        enabled_modules=enabled_modules,
        worker_enabled=_boolean(values, "SC_WORKER_ENABLED", False),
        worker_poll_interval_seconds=_floating_point(
            values,
            "SC_WORKER_POLL_INTERVAL_SECONDS",
            0.1,
        ),
        worker_max_workers=_integer(values, "SC_WORKER_MAX_WORKERS", 4),
        worker_lease_timeout_seconds=_floating_point(
            values, "SC_WORKER_LEASE_TIMEOUT_SECONDS", 300.0
        ),
        cognition_max_output_tokens=_integer(
            values, "SC_COGNITION_MAX_OUTPUT_TOKENS", 2048
        ),
        dialogue_max_output_tokens=_integer(
            values, "SC_DIALOGUE_MAX_OUTPUT_TOKENS", 4096
        ),
        model_temperature=_floating_point(
            values, "SC_MODEL_TEMPERATURE", 0.0
        ),
        model_timeout_seconds=_floating_point(
            values, "SC_MODEL_TIMEOUT_SECONDS", 120.0
        ),
        model_failure_cooldown_seconds=_floating_point(
            values, "SC_MODEL_FAILURE_COOLDOWN_SECONDS", 30.0
        ),
        model_max_failure_cooldown_seconds=_floating_point(
            values, "SC_MODEL_MAX_FAILURE_COOLDOWN_SECONDS", 900.0
        ),
    )


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"invalid .env line {line_number}")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if _KEY_PATTERN.fullmatch(key) is None:
            raise ValueError(f"invalid .env key at line {line_number}")
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def _enabled_modules(raw_value: str | None) -> frozenset[str] | None:
    if raw_value is None or raw_value.strip().lower() == "all":
        return None
    return frozenset(
        module_id.strip()
        for module_id in raw_value.split(",")
        if module_id.strip()
    )


def _boolean(values: Mapping[str, str], key: str, default: bool) -> bool:
    raw_value = values.get(key)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be a boolean")


def _integer(values: Mapping[str, str], key: str, default: int) -> int:
    raw_value = values.get(key)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(f"{key} must be an integer") from error


def _floating_point(
    values: Mapping[str, str],
    key: str,
    default: float,
) -> float:
    raw_value = values.get(key)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as error:
        raise ValueError(f"{key} must be a number") from error

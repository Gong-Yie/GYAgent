from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


MODEL_CONFIG_SCHEMA_VERSION = 1
MODEL_TASKS = frozenset(
    {
        "dialogue",
        "planning",
        "action",
        "proactive",
    }
)
_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FORBIDDEN_PROVIDER_KEYS = frozenset({"api_key", "key", "token", "secret", "password"})


@dataclass(frozen=True, slots=True)
class ModelProviderConfig:
    provider_id: str
    api_key_env: str
    model: str
    base_url: str | None = None


@dataclass(frozen=True, slots=True)
class ModelEnvironmentConfig:
    environment: str
    default_environment: str
    providers: Mapping[str, ModelProviderConfig]
    routes: Mapping[str, tuple[str, ...]]


def load_model_config(
    path: str | Path,
    *,
    environment: str | None = None,
) -> ModelEnvironmentConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"model config does not exist: {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("model config must be an object")
    if data.get("schema_version") != MODEL_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported model config schema version")
    default_environment = data.get("default_environment")
    environments = data.get("environments")
    if not isinstance(default_environment, str) or not default_environment.strip():
        raise ValueError("model config default_environment must be non-blank")
    if not isinstance(environments, dict) or not environments:
        raise ValueError("model config environments must be a non-empty object")
    selected = environment or default_environment
    raw_environment = environments.get(selected)
    if not isinstance(raw_environment, dict):
        raise ValueError(f"model config environment is unknown: {selected}")
    raw_providers = raw_environment.get("providers")
    raw_routes = raw_environment.get("routes")
    if not isinstance(raw_providers, dict) or not raw_providers:
        raise ValueError("model config providers must be a non-empty object")
    if not isinstance(raw_routes, dict) or not raw_routes:
        raise ValueError("model config routes must be a non-empty object")

    providers: dict[str, ModelProviderConfig] = {}
    for provider_id, raw_provider in raw_providers.items():
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise ValueError("model config provider ID must be non-blank")
        if not isinstance(raw_provider, dict):
            raise ValueError(f"provider {provider_id} must be an object")
        forbidden = _FORBIDDEN_PROVIDER_KEYS & {
            str(key).lower() for key in raw_provider
        }
        if forbidden:
            raise ValueError(
                f"provider {provider_id} must reference secrets by api_key_env, "
                "not store secret values"
            )
        allowed = {"api_key_env", "model", "base_url"}
        if set(raw_provider) - allowed:
            raise ValueError(f"provider {provider_id} has unsupported fields")
        api_key_env = raw_provider.get("api_key_env")
        model = raw_provider.get("model")
        base_url = raw_provider.get("base_url")
        if not isinstance(api_key_env, str) or _ENV_NAME_PATTERN.fullmatch(
            api_key_env
        ) is None:
            raise ValueError(f"provider {provider_id} api_key_env is invalid")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"provider {provider_id} model must be non-blank")
        if base_url is not None and (
            not isinstance(base_url, str) or not base_url.strip()
        ):
            raise ValueError(f"provider {provider_id} base_url is invalid")
        providers[provider_id] = ModelProviderConfig(
            provider_id=provider_id,
            api_key_env=api_key_env,
            model=model.strip(),
            base_url=base_url.strip() if isinstance(base_url, str) else None,
        )

    routes: dict[str, tuple[str, ...]] = {}
    for task, raw_provider_ids in raw_routes.items():
        if task not in MODEL_TASKS:
            raise ValueError(f"model config task is unsupported: {task}")
        if not isinstance(raw_provider_ids, list) or not raw_provider_ids:
            raise ValueError(f"model route {task} must be a non-empty array")
        provider_ids = tuple(str(item) for item in raw_provider_ids)
        if len(set(provider_ids)) != len(provider_ids):
            raise ValueError(f"model route {task} contains duplicate providers")
        unknown = [item for item in provider_ids if item not in providers]
        if unknown:
            raise ValueError(
                f"model route {task} references unknown providers: {unknown}"
            )
        routes[task] = provider_ids

    return ModelEnvironmentConfig(
        environment=selected,
        default_environment=default_environment,
        providers=providers,
        routes=routes,
    )

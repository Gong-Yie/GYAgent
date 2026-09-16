from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_cognition.bootstrap import build_container
from self_cognition.infrastructure.llm.model_config import load_model_config
from self_cognition.settings import ApplicationSettings, MAX_MODEL_OUTPUT_TOKENS


def _config_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "default_environment": "development",
        "environments": {
            "development": {
                "providers": {
                    "primary": {
                        "api_key_env": "TEST_PRIMARY_KEY",
                        "model": "model-a",
                        "base_url": "https://a.example.test/v1",
                    },
                    "backup": {
                        "api_key_env": "TEST_BACKUP_KEY",
                        "model": "model-b",
                        "base_url": "https://b.example.test/v1",
                    },
                },
                "routes": {
                    "dialogue": ["primary", "backup"],
                    "planning": ["primary", "backup"],
                    "action": ["primary", "backup"],
                    "proactive": ["primary", "backup"],
                },
            },
            "production": {
                "providers": {
                    "primary": {
                        "api_key_env": "PROD_PRIMARY_KEY",
                        "model": "model-prod",
                    }
                },
                "routes": {"dialogue": ["primary"]},
            },
        },
    }


def test_model_config_selects_environment_and_providers(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(json.dumps(_config_payload()), encoding="utf-8")

    config = load_model_config(path, environment="production")

    assert config.environment == "production"
    assert tuple(config.providers) == ("primary",)
    assert config.routes["dialogue"] == ("primary",)


def test_model_config_rejects_stored_secrets_and_bad_schema(
    tmp_path: Path,
) -> None:
    payload = _config_payload()
    payload["environments"]["development"]["providers"]["primary"]["api_key"] = "secret"
    path = tmp_path / "models.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="api_key_env"):
        load_model_config(path)

    payload = _config_payload()
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        load_model_config(path)


def test_build_container_registers_configured_provider_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("openai")
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps(_config_payload()), encoding="utf-8")
    monkeypatch.setenv("SC_MODELS_CONFIG", str(config_path))
    monkeypatch.setenv("SC_ENV", "development")
    monkeypatch.setenv("TEST_PRIMARY_KEY", "primary-key")
    monkeypatch.setenv("TEST_BACKUP_KEY", "backup-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    container = build_container(
        tmp_path,
        settings=ApplicationSettings(data_dir=tmp_path, worker_enabled=False),
        dotenv_path=tmp_path / "missing.env",
    )

    statuses = container.model_router.statuses()
    registered = {
        (item.task, item.provider_id, item.cost_per_call)
        for item in statuses
    }
    for task in ("dialogue", "planning", "action", "proactive"):
        assert (task, "configured:primary", 0.5) in registered
        assert (task, "configured:backup", 1.5) in registered

    selected = container.model_router.select("dialogue")
    assert selected.provider_id == "configured:primary"
    assert container.proactive._model is not None
    assert (
        container.model_router.select("proactive").provider_id
        == "configured:primary"
    )

    container.model_router.mark_degraded(
        "configured:primary",
        "dialogue",
        "ModelTimeoutError",
    )
    assert (
        container.model_router.select("dialogue").provider_id
        == "configured:backup"
    )


def test_application_settings_rejects_output_tokens_above_maximum() -> None:
    assert (
        ApplicationSettings(
            cognition_max_output_tokens=MAX_MODEL_OUTPUT_TOKENS,
        ).cognition_max_output_tokens
        == MAX_MODEL_OUTPUT_TOKENS
    )
    with pytest.raises(ValueError, match="must not exceed"):
        ApplicationSettings(
            cognition_max_output_tokens=MAX_MODEL_OUTPUT_TOKENS + 1,
        )
    with pytest.raises(ValueError, match="must not exceed"):
        ApplicationSettings(
            dialogue_max_output_tokens=MAX_MODEL_OUTPUT_TOKENS + 1,
        )

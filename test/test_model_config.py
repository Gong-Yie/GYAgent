from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from self_cognition.bootstrap import build_container
from self_cognition.infrastructure.llm.model_config import load_model_config
from self_cognition.infrastructure.llm.router import (
    ModelRegistration,
    ModelRouter,
)
from self_cognition.infrastructure.persistence.file_model_health_store import (
    FileModelHealthStore,
)
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

def test_model_config_migrates_legacy_flat_payload(tmp_path: Path) -> None:
    development = _config_payload()["environments"]["development"]
    payload = {
        "default_environment": "development",
        "providers": development["providers"],
        "routes": development["routes"],
    }
    path = tmp_path / "models.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_model_config(path)

    assert config.environment == "development"
    assert tuple(config.providers) == ("primary", "backup")
    assert config.routes["dialogue"] == ("primary", "backup")

def test_model_router_restores_bounded_provider_health(tmp_path: Path) -> None:
    store = FileModelHealthStore(tmp_path / "model_health.json")

    def router() -> ModelRouter:
        return ModelRouter(
            (
                ModelRegistration(
                    "dialogue",
                    "primary",
                    object(),
                    cost_per_call=0.5,
                ),
                ModelRegistration(
                    "dialogue",
                    "backup",
                    object(),
                    cost_per_call=1.5,
                ),
            ),
            failure_cooldown=timedelta(seconds=30),
            health_snapshot_sink=store.save,
        )

    first = router()
    first.mark_degraded("primary", "dialogue", "ModelTimeoutError")

    second = router()
    second.restore_health(store.load())

    statuses = {item.provider_id: item for item in second.statuses()}
    assert statuses["primary"].healthy is False
    assert statuses["primary"].degraded_reason == "ModelTimeoutError"
    assert second.select("dialogue").provider_id == "backup"

    second.mark_healthy("primary", "dialogue")
    third = router()
    third.restore_health(store.load())
    recovered = {item.provider_id: item for item in third.statuses()}
    assert recovered["primary"].healthy is True
    assert third.select("dialogue").provider_id == "primary"


def test_model_router_does_not_persist_routine_health(
    tmp_path: Path,
) -> None:
    store = FileModelHealthStore(tmp_path / "model_health.json")
    model_router = ModelRouter(
        (ModelRegistration("dialogue", "primary", object()),),
        health_snapshot_sink=store.save,
    )

    model_router.mark_healthy("primary", "dialogue")

    assert not store.path.exists()


def test_model_health_store_ignores_corrupt_snapshot(tmp_path: Path) -> None:
    store = FileModelHealthStore(tmp_path / "model_health.json")
    store.path.write_text("{not-json", encoding="utf-8")

    assert store.load() == ()

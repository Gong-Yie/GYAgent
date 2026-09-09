import sys
from datetime import timedelta
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from self_cognition.blackboard.reducer import StateReducer
from self_cognition.blackboard.service import CognitiveSpaceService
from self_cognition.bootstrap import build_container
from self_cognition.cognition.registry import (
    CognitiveModuleRegistry,
    ModuleHealth,
    ModuleRegistration,
)
from self_cognition.core.ids import new_correlation_id, new_run_id
from self_cognition.core.events import EventEnvelope
from self_cognition.core.state import SubjectState
from self_cognition.core.time import SYSTEM_CLOCK
from self_cognition.lifecycle import ApplicationLifecycle
from self_cognition.infrastructure.llm.action_responses import (
    OpenAIResponsesActionModel,
)
from self_cognition.infrastructure.llm.dialogue_responses import (
    OpenAIResponsesDialogueModel,
)
from self_cognition.infrastructure.llm.openai_responses import (
    OpenAIResponsesCognitionModel,
)
from self_cognition.infrastructure.llm.planning_responses import (
    OpenAIResponsesPlanningModel,
)
from self_cognition.runtime.engine import CognitionEngine
from self_cognition.runtime.run_context import RunContext
from self_cognition.settings import DotenvSecretSource, load_settings


class FailingModule:
    subscriptions = frozenset({"user.message"})

    def process(self, event: EventEnvelope) -> tuple[object, ...]:
        raise LookupError("module failed")


class RecordingBus:
    def __init__(self) -> None:
        self.drained = Event()

    def drain(self) -> tuple[object, ...]:
        self.drained.set()
        return ()


class RecordingResource:
    def __init__(self, calls: list[str], name: str) -> None:
        self._calls = calls
        self._name = name

    def start(self) -> None:
        self._calls.append(f"start:{self._name}")

    def close(self) -> None:
        self._calls.append(f"close:{self._name}")


def test_dotenv_loads_defaults_and_process_environment_overrides(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            (
                "SC_CONFIG_VERSION=1",
                "SC_DATA_DIR=from-dotenv",
                "SC_ENABLED_MODULES=semantic.name_extractor",
                "SC_WORKER_ENABLED=true",
                "SC_WORKER_POLL_INTERVAL_SECONDS=0.25",
                "SC_WORKER_MAX_WORKERS=2",
                "OPENAI_API_KEY=must-not-enter-settings",
                "OPENAI_MODEL=shared-model",
                "OPENAI_BASE_URL=https://models.example.test/v1",
            )
        ),
        encoding="utf-8",
    )

    settings = load_settings(
        dotenv,
        environ={"SC_DATA_DIR": "from-environment"},
    )

    assert settings.data_dir == Path("from-environment")
    assert settings.enabled_modules == frozenset({"semantic.name_extractor"})
    assert settings.worker_enabled is True
    assert settings.worker_poll_interval_seconds == 0.25
    assert settings.worker_max_workers == 2
    assert "must-not-enter-settings" not in repr(settings)
    assert not hasattr(settings, "openai_api_key")
    secret_source = DotenvSecretSource(
        dotenv,
        environ={"OPENAI_API_KEY": "environment-secret"},
    )
    assert secret_source.get("OPENAI_API_KEY") == "environment-secret"
    assert "environment-secret" not in repr(secret_source)
    assert "must-not-enter-settings" not in repr(secret_source)
    assert DotenvSecretSource(dotenv, environ={}).get("OPENAI_API_KEY") == (
        "must-not-enter-settings"
    )
    assert DotenvSecretSource(dotenv, environ={}).get("OPENAI_BASE_URL") == (
        "https://models.example.test/v1"
    )
    assert DotenvSecretSource(dotenv, environ={}).get("OPENAI_MODEL") == (
        "shared-model"
    )


@pytest.mark.parametrize(
    "model_class",
    (
        OpenAIResponsesCognitionModel,
        OpenAIResponsesDialogueModel,
        OpenAIResponsesPlanningModel,
        OpenAIResponsesActionModel,
    ),
)
def test_all_openai_agent_factories_forward_base_url(
    monkeypatch,
    model_class,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeOpenAIClient:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=FakeOpenAIClient),
    )

    model_class.from_api_key(
        "test-key",
        "test-model",
        base_url="https://models.example.test/v1",
    )

    assert calls == [
        {
            "api_key": "test-key",
            "base_url": "https://models.example.test/v1",
            "max_retries": 0,
        }
    ]


def test_build_container_uses_one_openai_model_for_all_default_agents(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            (
                "OPENAI_API_KEY=test-key",
                "OPENAI_MODEL=shared-model",
                "OPENAI_BASE_URL=https://models.example.test/v1",
            )
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    class FakeOpenAIClient:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

        def close(self) -> None:
            return None

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=FakeOpenAIClient),
    )

    container = build_container(tmp_path / "data", dotenv_path=dotenv)
    modules = {
        module.module_id: module
        for module in container.module_registry.all_modules()
    }
    configured_models = (
        container.dialogue_model,
        container.planning_model,
        container.action_model,
        modules["metacognition.conflict_extractor"]._model,
        modules["affect.affect_extractor"]._model,
    )

    assert all(model._model == "shared-model" for model in configured_models)
    assert calls == [
        {
            "api_key": "test-key",
            "base_url": "https://models.example.test/v1",
            "max_retries": 0,
        }
    ] * 5


def test_build_container_rejects_partial_openai_configuration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENAI_MODEL=shared-model\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be configured together"):
        build_container(tmp_path / "data", dotenv_path=dotenv)


def test_dotenv_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("SC_CONFIG_VERSION=2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported settings schema"):
        load_settings(dotenv, environ={})


def test_default_registry_has_twelve_modules_in_seven_categories(tmp_path: Path) -> None:
    container = build_container(tmp_path, dotenv_path=tmp_path / "missing.env")

    statuses = container.module_registry.statuses()

    assert len(statuses) == 12
    assert len({status.category for status in statuses}) == 7
    assert all(status.health is ModuleHealth.HEALTHY for status in statuses)


def test_registry_disables_modules_and_records_degradation() -> None:
    module = FailingModule()
    registry = CognitiveModuleRegistry(
        (ModuleRegistration("test.failing", "semantic", "1", module),)
    )
    engine = CognitionEngine(
        (module,),
        CognitiveSpaceService(StateReducer()),
        module_registry=registry,
    )

    with pytest.raises(LookupError):
        engine.process(
            EventEnvelope.user_message("user-1", "test"),
            SubjectState.empty("user-1"),
        )
    degraded = registry.statuses()[0]
    assert degraded.health is ModuleHealth.DEGRADED
    assert degraded.degraded_reason == "LookupError"

    registry.disable("test.failing")
    unchanged = engine.process(
        EventEnvelope.user_message("user-1", "test"),
        SubjectState.empty("user-1"),
    )
    assert unchanged == SubjectState.empty("user-1")
    assert registry.statuses()[0].health is ModuleHealth.DISABLED


def test_registry_enable_affects_future_engine_processing() -> None:
    module = FailingModule()
    registry = CognitiveModuleRegistry(
        (
            ModuleRegistration(
                "test.failing",
                "semantic",
                "1",
                module,
                enabled=False,
            ),
        )
    )
    engine = CognitionEngine(
        (module,),
        CognitiveSpaceService(StateReducer()),
        module_registry=registry,
    )

    registry.enable("test.failing")
    with pytest.raises(LookupError):
        engine.process(
            EventEnvelope.user_message("user-1", "test"),
            SubjectState.empty("user-1"),
        )


def test_lifecycle_starts_worker_and_closes_resources_in_reverse_order() -> None:
    bus = RecordingBus()
    calls: list[str] = []
    lifecycle = ApplicationLifecycle(
        bus,
        worker_enabled=True,
        worker_poll_interval_seconds=0.01,
        resources=(
            RecordingResource(calls, "first"),
            RecordingResource(calls, "second"),
        ),
    )

    lifecycle.start()
    assert lifecycle.wait_ready(1)
    assert bus.drained.wait(1)
    lifecycle.stop(1)

    assert lifecycle.is_ready is False
    assert lifecycle.worker_error_type is None
    assert calls == [
        "start:first",
        "start:second",
        "close:second",
        "close:first",
    ]


def test_stopped_container_rejects_new_events(tmp_path: Path) -> None:
    container = build_container(tmp_path, dotenv_path=tmp_path / "missing.env")
    container.lifecycle.start()
    container.lifecycle.stop(1)
    context = RunContext(
        new_run_id(),
        new_correlation_id(),
        SYSTEM_CLOCK.now() + timedelta(seconds=30),
    )

    with pytest.raises(RuntimeError, match="not accepting"):
        container.event_bus.publish(
            EventEnvelope.user_message("user-1", "test"),
            context,
        )

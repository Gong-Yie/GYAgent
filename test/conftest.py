from pathlib import Path

import pytest

from self_cognition import settings as settings_module


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_DOTENV = (_PROJECT_ROOT / ".env").resolve()
_MODEL_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "SC_WORKER_ENABLED",
)


@pytest.fixture(autouse=True)
def _isolate_default_config(request, monkeypatch):
    if request.node.get_closest_marker("live_openai") is not None:
        return
    for name in _MODEL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    original_read_dotenv = settings_module._read_dotenv

    def isolated_read_dotenv(path):
        try:
            resolved = Path(path).resolve()
        except OSError:
            resolved = None
        if resolved == _PROJECT_DOTENV:
            return {}
        return original_read_dotenv(path)

    monkeypatch.setattr(settings_module, "_read_dotenv", isolated_read_dotenv)
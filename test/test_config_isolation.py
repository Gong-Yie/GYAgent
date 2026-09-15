from pathlib import Path

from self_cognition.bootstrap import build_container
from self_cognition.executive.dialogue.rule_based import RuleBasedDialogueModel
from self_cognition.settings import DotenvSecretSource


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_container_ignores_project_dotenv(tmp_path):
    container = build_container(tmp_path)
    try:
        assert isinstance(container.dialogue_model, RuleBasedDialogueModel)
    finally:
        container.lifecycle.stop()


def test_project_dotenv_secret_is_hidden_from_default_tests():
    source = DotenvSecretSource(_PROJECT_ROOT / ".env")

    assert source.get("OPENAI_API_KEY") is None
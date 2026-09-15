import json
from pathlib import Path

from self_cognition.bootstrap import build_container
from self_cognition.core.scopes import SubjectScope
from self_cognition.interfaces.cli import main
from self_cognition.interfaces.http.server import _handle


def test_cli_and_http_expose_active_emotion_and_mood(tmp_path, capsys) -> None:
    app = build_container(
        tmp_path / "data",
        dotenv_path=tmp_path / "missing.env",
    )
    subject = SubjectScope.legacy_user("emotion-view-user")
    try:
        assert (
            main(
                ["chat", subject.subject.subject_id, "我现在有点无聊"],
                container=app,
            )
            == 0
        )
        capsys.readouterr()

        assert (
            main(
                ["emotion", subject.subject.subject_id],
                container=app,
            )
            == 0
        )
        cli_payload = json.loads(capsys.readouterr().out)
        assert cli_payload["state_version"] >= 1
        fields = {item["field"] for item in cli_payload["emotions"]}
        assert "affect.reaction.interaction" in fields
        reaction = next(
            item
            for item in cli_payload["emotions"]
            if item["field"] == "affect.reaction.interaction"
        )
        assert reaction["content"]["emotion"] == "boredom"
        assert cli_payload["mood"]["field"] == "mood.current"

        http_payload = _handle(
            app,
            "GET",
            "/emotion",
            {},
            {"subject_id": subject.subject.subject_id},
        )
        assert http_payload["emotions"]
        assert http_payload["mood"] is not None

        mood_payload = _handle(
            app,
            "GET",
            "/mood",
            {},
            {"subject_id": subject.subject.subject_id},
        )
        assert mood_payload["mood"]["field"] == "mood.current"

        affect_payload = _handle(
            app,
            "GET",
            "/affect",
            {},
            {"subject_id": subject.subject.subject_id},
        )
        assert {item["field"] for item in affect_payload["emotions"]} == {
            item["field"] for item in http_payload["emotions"]
        }
    finally:
        app.lifecycle.stop()


def test_webui_exposes_emotion_entry() -> None:
    root = Path(__file__).resolve().parents[1] / "webui"
    html = (root / "index.html").read_text(encoding="utf-8")
    script = (root / "app.js").read_text(encoding="utf-8")

    assert 'data-view="emotion"' in html
    assert "renderEmotion" in script
    assert "/emotion?subject_id=" in script

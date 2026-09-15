import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from self_cognition.bootstrap import build_container
from self_cognition.core.affect import EmotionState
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

        assert (
            main(
                ["emotion", subject.subject.subject_id, "--action", "close"],
                container=app,
            )
            == 0
        )
        cli_closed = json.loads(capsys.readouterr().out)
        assert "affect.fast_reaction" in cli_closed["disabled_modules"]
        assert "affect.affect_extractor" in cli_closed["disabled_modules"]

        assert (
            main(
                ["emotion", subject.subject.subject_id, "--action", "open"],
                container=app,
            )
            == 0
        )
        cli_open = json.loads(capsys.readouterr().out)
        assert "affect.fast_reaction" not in cli_open["disabled_modules"]
        assert "affect.affect_extractor" not in cli_open["disabled_modules"]

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

        corrected = EmotionState(
            emotion_id=uuid4(),
            target=subject.subject.subject_id,
            emotion="calm",
            valence="positive",
            scope="interaction",
            intensity=0.9,
            assessed_at=datetime.now(timezone.utc),
            half_life_seconds=3600.0,
            arousal=0.2,
            control=0.8,
            certainty=0.7,
            cause="http correction",
        )
        corrected_response = _handle(
            app,
            "POST",
            "/emotion/correct",
            {},
            {
                "subject_id": subject.subject.subject_id,
                "target_field": "affect.reaction.interaction",
                "value": corrected.to_state_value(),
            },
        )
        assert corrected_response["status"] == "succeeded"

        closed_response = _handle(
            app,
            "POST",
            "/emotion/close",
            {},
            {"subject_id": subject.subject.subject_id},
        )
        assert "affect.fast_reaction" in closed_response["controls"]["disabled_modules"]
        opened_response = _handle(
            app,
            "POST",
            "/emotion/open",
            {},
            {"subject_id": subject.subject.subject_id},
        )
        assert "affect.fast_reaction" not in opened_response["controls"]["disabled_modules"]

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
    assert "data-open-emotion" in script

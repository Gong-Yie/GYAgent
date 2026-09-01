import json

from self_cognition.interfaces.cli import main


def test_cli_persists_state_and_answers_after_a_new_process_setup(tmp_path, capsys):
    data_dir = tmp_path / "data"

    first_exit_code = main(
        [
            "user-1",
            "我喜欢晚上学习",
            "--data-dir",
            str(data_dir),
        ]
    )
    first_output = json.loads(capsys.readouterr().out)

    second_exit_code = main(
        [
            "user-1",
            "我喜欢什么时候学习？",
            "--data-dir",
            str(data_dir),
        ]
    )
    second_output = json.loads(capsys.readouterr().out)

    assert first_exit_code == 0
    assert first_output["status"] == "succeeded"
    assert first_output["state_changed"] is True
    assert first_output["new_version"] == 1

    assert second_exit_code == 0
    assert second_output["status"] == "succeeded"
    assert second_output["state_changed"] is False
    assert second_output["old_version"] == 1
    assert second_output["new_version"] == 1
    assert second_output["response"] == "你喜欢晚上学习。"
    assert len(second_output["evidence_refs"]) == 1


def test_cli_returns_nonzero_for_a_failed_persistent_dependency(tmp_path, capsys):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "events.jsonl").write_text("{broken}\n", encoding="utf-8")

    exit_code = main(["user-1", "测试", "--data-dir", str(data_dir)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert json.loads(captured.err)["status"] == "failed"
    assert captured.out == ""


def test_cli_reloads_and_expresses_a_persisted_preference_conflict(
    tmp_path,
    capsys,
):
    data_dir = tmp_path / "data"

    first_exit_code = main(
        [
            "user-1",
            "我既喜欢早上学习，也喜欢晚上学习",
            "--data-dir",
            str(data_dir),
        ]
    )
    first_output = json.loads(capsys.readouterr().out)
    second_exit_code = main(
        [
            "user-1",
            "我喜欢什么时候学习？",
            "--data-dir",
            str(data_dir),
        ]
    )
    second_output = json.loads(capsys.readouterr().out)

    assert first_exit_code == 0
    assert first_output["state_changed"] is True
    assert second_exit_code == 0
    assert second_output["response"] == (
        "你的学习时间偏好存在冲突：同时提到了早上和晚上。"
    )
    assert len(second_output["evidence_refs"]) == 1


def test_cli_requires_confirmation_to_change_persisted_identity(
    tmp_path,
    capsys,
):
    data_dir = tmp_path / "data"

    for message in (
        "我的角色是研究助手",
        "我的角色是学习助手",
    ):
        assert main(["user-1", message, "--data-dir", str(data_dir)]) == 0
        capsys.readouterr()

    assert main(
        ["user-1", "我的角色是什么？", "--data-dir", str(data_dir)]
    ) == 0
    unconfirmed_output = json.loads(capsys.readouterr().out)

    assert unconfirmed_output["response"] == "你的角色是研究助手。"

    assert main(
        [
            "user-1",
            "我确认将角色改为学习助手",
            "--data-dir",
            str(data_dir),
        ]
    ) == 0
    capsys.readouterr()
    assert main(
        ["user-1", "我的角色是什么？", "--data-dir", str(data_dir)]
    ) == 0
    confirmed_output = json.loads(capsys.readouterr().out)

    assert confirmed_output["response"] == "你的角色是学习助手。"
    assert len(confirmed_output["evidence_refs"]) == 2


def test_cli_reloads_scoped_affect_and_answers_with_evidence(tmp_path, capsys):
    data_dir = tmp_path / "data"

    assert main(
        [
            "user-1",
            "这次考试通过了，我很开心",
            "--data-dir",
            str(data_dir),
        ]
    ) == 0
    capsys.readouterr()
    assert main(
        [
            "user-1",
            "我对这次考试感觉怎么样？",
            "--data-dir",
            str(data_dir),
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["response"].startswith("你对这次考试感到开心")
    assert len(output["evidence_refs"]) == 1


def test_cli_persists_and_answers_an_ordered_project_narrative(tmp_path, capsys):
    data_dir = tmp_path / "data"
    for message in ("我完成了研究项目", "我开始准备研究项目"):
        assert main(
            ["user-1", message, "--data-dir", str(data_dir)]
        ) == 0
        capsys.readouterr()

    assert main(
        [
            "user-1",
            "我的项目经历如何发展？",
            "--data-dir",
            str(data_dir),
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["response"] == (
        "你的项目叙事是：先是开始准备研究项目，后来完成研究项目。"
    )
    assert len(output["evidence_refs"]) == 2

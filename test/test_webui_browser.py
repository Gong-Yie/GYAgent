from __future__ import annotations

import subprocess
from pathlib import Path
from threading import Thread

import pytest

from self_cognition.bootstrap import build_container
from self_cognition.interfaces.http import create_server
from self_cognition.settings import ApplicationSettings


EDGE_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
)


def _browser_path() -> Path | None:
    for candidate in EDGE_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


@pytest.mark.skipif(
    _browser_path() is None,
    reason="no supported headless browser is installed",
)
def test_headless_browser_renders_health_view(tmp_path: Path) -> None:
    browser = _browser_path()
    assert browser is not None
    app = build_container(
        tmp_path / "data",
        settings=ApplicationSettings(
            data_dir=tmp_path / "data",
            worker_enabled=False,
        ),
        dotenv_path=tmp_path / "missing.env",
    )
    server = create_server(app, static_directory=Path("webui"))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = (
            f"http://127.0.0.1:{server.server_port}/?view=health"
        )
        completed = subprocess.run(
            [
                str(browser),
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                f"--user-data-dir={tmp_path / 'browser-profile'}",
                "--virtual-time-budget=5000",
                "--dump-dom",
                url,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        if completed.returncode != 0:
            pytest.skip(
                "headless browser could not start: "
                + completed.stderr[-300:]
            )
        dom = completed.stdout
        assert "健康与降级历史" in dom
        assert "最近失败链" in dom
        assert "读取中" not in dom
    finally:
        server.shutdown()
        server.server_close()
        app.lifecycle.stop()

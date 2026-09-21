import os
import tempfile
import time
from pathlib import Path


REPLACE_ATTEMPTS = 20
REPLACE_RETRY_SECONDS = 0.05


def atomic_write_text(
    path: str | Path,
    content: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Write text durably, replacing the target only after fsync succeeds."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _replace_with_retry(source: Path, target: Path) -> None:
    """Replace ``target``, waiting out a concurrent reader holding it open.

    On Windows ``os.replace`` fails with ``PermissionError`` (WinError 5) while
    another process has the destination open, which is routine for a store that
    several workers write to. The replacement is atomic, so retrying leaves the
    target either wholly old or wholly new; only the bounded wait is added.
    """
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_RETRY_SECONDS)

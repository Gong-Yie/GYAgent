from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


def create_private_temp_dir(
    prefix: str,
    directory: str | Path | None = None,
) -> Path:
    """Create a uniquely named temporary directory under a controlled root.

    POSIX keeps owner-only mode bits.  Windows ignores POSIX mode bits and
    inherits the parent directory ACL, so callers must pass a private
    ``directory`` (or rely on the user's default temporary directory).
    """
    root = Path(directory) if directory is not None else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    mode = 0o777 if os.name == "nt" else 0o700
    for _ in range(10):
        path = root / f"{prefix}{uuid4().hex[:8]}"
        try:
            path.mkdir(mode=mode)
        except FileExistsError:
            continue
        return path
    raise FileExistsError(f"could not create temporary directory under {root}")


@contextmanager
def temporary_directory(
    prefix: str,
    directory: str | Path | None = None,
) -> Iterator[Path]:
    path = create_private_temp_dir(prefix, directory)
    try:
        yield path
    finally:
        shutil.rmtree(path)
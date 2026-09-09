"""Create, restore, or migrate self-cognition data without overwriting targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from self_cognition.infrastructure.persistence.backup import (  # noqa: E402
    BackupResult,
    create_backup,
    migrate_data,
    restore_backup,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    backup = commands.add_parser("backup")
    backup.add_argument("source", type=Path)
    backup.add_argument("archive", type=Path)
    backup.add_argument("--config", action="append", type=Path, default=[])
    restore = commands.add_parser("restore")
    restore.add_argument("archive", type=Path)
    restore.add_argument("target", type=Path)
    migrate = commands.add_parser("migrate")
    migrate.add_argument("source", type=Path)
    migrate.add_argument("target", type=Path)
    migrate.add_argument("--config", action="append", type=Path, default=[])
    return parser


def _report(result: BackupResult) -> None:
    print(
        json.dumps(
            {
                "path": str(result.path),
                "file_count": result.file_count,
                "snapshot": result.snapshot.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "backup":
        result = create_backup(
            arguments.source,
            arguments.archive,
            non_sensitive_config=tuple(arguments.config),
        )
    elif arguments.command == "restore":
        result = restore_backup(arguments.archive, arguments.target)
    else:
        result = migrate_data(
            arguments.source,
            arguments.target,
            non_sensitive_config=tuple(arguments.config),
        )
    _report(result)


if __name__ == "__main__":
    main()

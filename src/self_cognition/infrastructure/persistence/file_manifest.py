import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from self_cognition.infrastructure.persistence.atomic_io import atomic_write_text


@dataclass(frozen=True, slots=True)
class FileManifestEntry:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class FileManifest:
    entries: tuple[FileManifestEntry, ...]

    def to_json(self) -> str:
        payload = {
            "schema_version": 1,
            "entries": [
                {
                    "path": entry.path,
                    "size": entry.size,
                    "sha256": entry.sha256,
                }
                for entry in self.entries
            ],
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


def build_manifest(root: str | Path) -> FileManifest:
    base = Path(root)
    entries: list[FileManifestEntry] = []
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        relative = path.relative_to(base).as_posix()
        if relative.endswith(".manifest.json"):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(
            FileManifestEntry(
                path=relative,
                size=path.stat().st_size,
                sha256=digest,
            )
        )
    return FileManifest(tuple(entries))


def write_manifest(root: str | Path, path: str | Path) -> FileManifest:
    manifest = build_manifest(root)
    atomic_write_text(path, manifest.to_json() + "\n")
    return manifest

"""File manifests and checksums for generated replication artifacts."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

TEXT_MANIFEST_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}


def manifest_digest(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_MANIFEST_SUFFIXES:
        # Normalize line endings so Windows/Linux checkouts hash text artifacts the same way.
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        hash_mode = "lf_normalized"
    else:
        hash_mode = "raw"
    return {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "hash_mode": hash_mode,
    }


def write_manifest_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "relative_path",
        "stage",
        "bytes",
        "sha256",
        "hash_mode",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

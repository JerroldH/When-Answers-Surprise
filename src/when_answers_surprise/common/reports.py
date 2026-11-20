"""Small text reports used by staging and audit scripts."""

from __future__ import annotations

from pathlib import Path


def write_audit_report(path: Path, title: str, audit: dict[str, object]) -> None:
    """Write a stable markdown audit report from scalar diagnostics."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"- {key}: {value}" for key, value in audit.items())
    path.write_text(f"# {title}\n\n{rows}\n", encoding="utf-8")


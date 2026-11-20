"""Canonical local paths for the public replication package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved repository paths used by scripts and documentation builders."""

    root: Path

    @classmethod
    def from_file(cls, file: str | Path, *, parents_to_root: int) -> "ProjectPaths":
        return cls(Path(file).resolve().parents[parents_to_root])

    @classmethod
    def discover(cls) -> "ProjectPaths":
        """Find the repo root when a helper is called from an arbitrary cwd."""

        here = Path.cwd().resolve()
        for candidate in [here, *here.parents]:
            if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
                return cls(candidate)
        raise FileNotFoundError("Could not find repository root from current directory.")

    @property
    def src(self) -> Path:
        return self.root / "src"

    @property
    def data_root(self) -> Path:
        return self.root / "data" / "wrds_sp500_2018_2025"

    @property
    def analysis_root(self) -> Path:
        return self.data_root / "analysis"

    @property
    def reports_root(self) -> Path:
        return self.root / "reports"

    @property
    def wrds_config_root(self) -> Path:
        return self.root / "configs" / "wrds_exports"


def display_path(path: Path | None, *, root: Path | None = None) -> str | None:
    """Render paths relative to the project when possible."""

    if path is None:
        return None
    resolved = path.resolve()
    base = root.resolve() if root else ProjectPaths.discover().root
    try:
        # Relative paths keep manifests portable across machines.
        return resolved.relative_to(base).as_posix()
    except ValueError:
        return str(resolved)

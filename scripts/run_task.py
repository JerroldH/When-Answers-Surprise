"""Run named replication tasks from scripts/tasks.yaml."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = PROJECT_ROOT / "scripts" / "tasks.yaml"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    tasks = load_tasks(TASKS_PATH)
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: python scripts/run_task.py --list")
        print("       python scripts/run_task.py <task_id> [task args...]")
        print("\nUse --list to show task ids. Arguments after <task_id> pass through unchanged.")
        return 0
    if argv[0] == "--list":
        print_task_list(tasks)
        return 0
    task_id = argv[0]
    task_args = argv[1:]
    if task_id not in tasks:
        available = ", ".join(sorted(tasks))
        raise SystemExit(f"Unknown task '{task_id}'. Available tasks: {available}")

    script = PROJECT_ROOT / str(tasks[task_id]["script"])
    command = [sys.executable, str(script), *task_args]
    env = os.environ.copy()
    src = str(PROJECT_ROOT / "src")
    # ponytail: scripts run from a fresh checkout before editable install; this keeps that path working.
    env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    return completed.returncode


def load_tasks(path: Path) -> dict[str, dict[str, Any]]:
    """Load stable task ids; scripts are implementation details behind this map."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, dict):
        raise ValueError(f"{path} must contain a top-level tasks mapping.")
    return tasks


def print_task_list(tasks: dict[str, dict[str, Any]]) -> None:
    """Print the ownership-level task view, not raw file order."""

    for task_id, meta in sorted(tasks.items(), key=lambda item: (item[1].get("group", ""), item[0])):
        print(f"{task_id:35} {meta.get('group', ''):15} {meta.get('description', '')}")


if __name__ == "__main__":
    raise SystemExit(main())

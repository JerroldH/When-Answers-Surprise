from __future__ import annotations

import os
from typing import Any


class ConnectionConfigError(RuntimeError):
    """Raised when a WRDS connection cannot be configured safely."""


def username_from_env(username_env: str) -> str:
    username = os.environ.get(username_env)
    if not username:
        raise ConnectionConfigError(
            f"WRDS username environment variable is not set: {username_env}"
        )
    return username


def connect_wrds(username_env: str = "WRDS_USER") -> Any:
    username = username_from_env(username_env)
    try:
        import wrds
    except ImportError as exc:
        raise ConnectionConfigError(
            "The official `wrds` package is not installed. Run `python -m pip install -e .`."
        ) from exc
    return wrds.Connection(wrds_username=username)


def close_connection(db: Any) -> None:
    close = getattr(db, "close", None)
    if callable(close):
        close()

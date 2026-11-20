from __future__ import annotations

import re


IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.]+$")
SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def validate_identifier(value: str, *, label: str = "identifier") -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string.")
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} may only contain letters, numbers, underscores, and dots: {value!r}"
        )
    return value


def validate_safe_name(value: str, *, label: str = "name") -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string.")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a plain file-safe name: {value!r}")
    if not SAFE_NAME_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} may only contain letters, numbers, underscores, dashes, and dots: {value!r}"
        )
    return value

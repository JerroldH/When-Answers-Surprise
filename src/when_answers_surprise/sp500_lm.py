"""Compatibility facade for :mod:`when_answers_surprise.empirical.pipeline`."""

from __future__ import annotations

import sys as _sys

from when_answers_surprise.empirical import pipeline as _impl

# Preserve old imports and monkeypatch behavior during the public-package transition.
globals().update(_impl.__dict__)
_sys.modules[__name__] = _impl

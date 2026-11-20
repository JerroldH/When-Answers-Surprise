"""Compatibility facade for :mod:`when_answers_surprise.data.ibes_staging`."""

from __future__ import annotations

import sys as _sys

from when_answers_surprise.data import ibes_staging as _impl

globals().update(_impl.__dict__)
_sys.modules[__name__] = _impl

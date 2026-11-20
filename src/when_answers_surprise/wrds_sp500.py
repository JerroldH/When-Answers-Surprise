"""Compatibility facade for :mod:`when_answers_surprise.data.wrds_sp500`."""

from __future__ import annotations

import sys as _sys

from when_answers_surprise.data import wrds_sp500 as _impl

globals().update(_impl.__dict__)
_sys.modules[__name__] = _impl

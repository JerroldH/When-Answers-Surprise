"""Compatibility facade for :mod:`when_answers_surprise.empirical.learner_audit`."""

from __future__ import annotations

import sys as _sys

from when_answers_surprise.empirical import learner_audit as _impl

globals().update(_impl.__dict__)
_sys.modules[__name__] = _impl

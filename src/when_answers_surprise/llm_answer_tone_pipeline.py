"""Compatibility facade for :mod:`when_answers_surprise.llm.answer_pipeline`."""

from __future__ import annotations

import sys as _sys

from when_answers_surprise.llm import answer_pipeline as _impl

globals().update(_impl.__dict__)
_sys.modules[__name__] = _impl

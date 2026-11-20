"""Numerical helpers shared by diagnostics and publication builders."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def safe_corr(left: Sequence[float] | pd.Series, right: Sequence[float] | pd.Series) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 2 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return float("nan")
    return float(frame["left"].corr(frame["right"]))


def round_numeric(data: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    out = data.copy()
    numeric = out.select_dtypes(include=["number"]).columns
    out[numeric] = out[numeric].round(digits)
    return out


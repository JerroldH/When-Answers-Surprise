import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from when_answers_surprise.empirical.pipeline import aligned_event_window  # noqa: E402


def test_horizon_is_exactly_h_trading_rows_starting_at_event_day() -> None:
    values = pd.Series(range(30), index=pd.date_range("2024-01-01", periods=30, freq="B"))

    one_day = aligned_event_window(values, event_index=5, horizon=1)
    twenty_days = aligned_event_window(values, event_index=5, horizon=20)

    assert len(one_day) == 1
    assert len(twenty_days) == 20
    assert one_day.index[0] == values.index[5]
    assert twenty_days.index[-1] == values.index[24]


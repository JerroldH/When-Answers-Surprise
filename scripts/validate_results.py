"""Check the paper anchors without manuscript sources or private data."""
import math,json
from pathlib import Path
import pandas as pd
RESULTS = Path(__file__).resolve().parents[1] / "paper_results"

def _close(actual: float, expected: float, tolerance: float = 0.005) -> None:
    if not math.isclose(actual, expected, abs_tol=tolerance):
        raise AssertionError(f"Expected {expected}, found {actual}")


def validate_results() -> None:
    primary = pd.read_csv(RESULTS / "holdout_primary.csv").set_index("specification")
    horizons = pd.read_csv(RESULTS / "holdout_horizons.csv")
    interactions = pd.read_csv(RESULTS / "temporal_interactions.csv").set_index(
        "specification"
    )
    pairing = pd.read_csv(RESULTS / "development_context_pairing.csv").set_index(
        "variant"
    )
    decomposition = pd.read_csv(
        RESULTS / "development_answer_decomposition.csv"
    ).set_index(["specification", "term"])
    timing = pd.read_csv(RESULTS / "holdout_timing_robustness.csv")
    daily = pd.read_csv(RESULTS / "holdout_daily_pre_event.csv")
    numbers = json.loads((RESULTS / "manuscript_numbers.json").read_text())

    if set(primary.index) != {"P0", "Q2", "A3"}:
        raise AssertionError("Primary table must contain exactly P0, Q2, and A3")
    if len(horizons) != 15 or set(horizons["horizon"]) != {1, 3, 5, 10, 20}:
        raise AssertionError("Horizon table must contain 3 stages x 5 horizons")
    if set(horizons["period"].astype(str)) != {"2025"}:
        raise AssertionError("Horizon data must be labeled as the 2025 holdout")
    if set(horizons["unit"]) != {"bps_per_1sd"}:
        raise AssertionError("Horizon data have an unexpected return unit")

    _close(primary.loc["A3", "coefficient_bps"], 61.47)
    _close(primary.loc["A3", "adjusted_se_bps"], 18.72)
    _close(primary.loc["A3", "ci_low_bps"], 24.78)
    _close(primary.loc["A3", "ci_high_bps"], 98.16)
    _close(primary.loc["P0", "coefficient_bps"], 99.04)
    _close(primary.loc["Q2", "coefficient_bps"], 113.85)
    if set(primary["n_calls"].astype(int)) != {1901}:
        raise AssertionError("Primary holdout N must be 1,901")
    _close(numbers["a3_pre_event_joint_p_value"], 0.435)
    _close(interactions.loc["A3", "coefficient_bps"], 13.21)
    _close(interactions.loc["A3", "p_value"], 0.404)
    _close(pairing.loc["A3", "matched_r2"], 0.0463035, tolerance=1e-7)
    _close(pairing.loc["A3", "expected_deranged_r2"], 0.0259782, tolerance=1e-7)
    _close(
        decomposition.loc[
            ("U3_plus_D03_common_sd", "D03_common_z"), "beta_bps"
        ],
        78.10245,
        tolerance=1e-5,
    )
    _close(
        decomposition.loc[
            ("expected_plus_residual_common_sd", "expected_A3_common_z"),
            "beta_bps",
        ],
        43.25297,
        tolerance=1e-5,
    )
    _close(
        decomposition.loc[
            ("expected_plus_residual_common_sd", "residual_A3_common_z"),
            "beta_bps",
        ],
        54.10346,
        tolerance=1e-5,
    )
    _close(
        decomposition.loc[
            ("expected_plus_residual_common_sd", "residual_minus_expected"),
            "p_value",
        ],
        0.604999,
        tolerance=1e-6,
    )
    exclusion = timing.loc[timing["specification"].eq("exclude_market_hours")].iloc[0]
    timing_joint = timing.loc[
        timing["result_type"].eq("joint_timing_equality_test")
    ].iloc[0]
    _close(exclusion["beta_bps"], 64.06287, tolerance=1e-5)
    if int(exclusion["n_calls"]) != 1_351:
        raise AssertionError("Market-hours exclusion must leave 1,351 calls")
    _close(timing_joint["p_value"], 0.872944, tolerance=1e-6)
    if len(daily) != 20 or set(daily["event_day"].astype(int)) != set(range(-20, 0)):
        raise AssertionError("Daily pre-event interface must contain days -20 through -1")
    _close(daily["family_joint_p_value"].iloc[0], 0.245877, tolerance=1e-6)

    expected_a3 = {1: 61.47, 3: 64.11, 5: 69.64, 10: 73.94, 20: 54.98}
    a3 = horizons.loc[horizons["specification"].eq("A3")].set_index("horizon")
    for horizon, expected in expected_a3.items():
        _close(a3.loc[horizon, "coefficient_bps"], expected)

    horizon_one = horizons.loc[horizons["horizon"].eq(1)].set_index(
        "specification"
    )
    for specification in ("P0", "Q2", "A3"):
        _close(
            horizon_one.loc[specification, "coefficient_bps"],
            primary.loc[specification, "coefficient_bps"],
            tolerance=1e-8,
        )

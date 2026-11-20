"""Joint horizon inference for revision-v2 response curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2

from when_answers_surprise.empirical.pipeline import HORIZONS
from when_answers_surprise.empirical.pre_event_audit import normal_two_sided_p


@dataclass(frozen=True)
class JointSystemResult:
    coefficients: pd.DataFrame
    covariance: np.ndarray
    coefficient_index: pd.DataFrame
    n_obs: int


def fit_joint_horizon_system(
    data: pd.DataFrame,
    *,
    outcomes: Sequence[str],
    treatments: Sequence[str],
    controls: Sequence[str],
    firm_fe: str = "permno",
    quarter_fe: str = "event_calendar_quarter",
    cluster_cols: tuple[str, str] = ("permno", "event_trading_date"),
) -> JointSystemResult:
    """Fit common-design OLS outcomes and their joint two-way cluster covariance."""

    regressors = list(dict.fromkeys([*treatments, *controls]))
    needed = list(
        dict.fromkeys([*outcomes, *regressors, firm_fe, quarter_fe, *cluster_cols])
    )
    clean = data[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    design = clean[regressors].copy()
    if "match_quality" in design:
        design["match_quality"] = design["match_quality"].astype(str)
        design = pd.get_dummies(
            design, columns=["match_quality"], drop_first=True, dtype=float
        )
    firm = pd.get_dummies(clean[firm_fe].astype(str), prefix="firm", drop_first=True, dtype=float)
    quarter = pd.get_dummies(
        clean[quarter_fe].astype(str), prefix="qtr", drop_first=True, dtype=float
    )
    x_frame = pd.concat([design.astype(float), firm, quarter], axis=1)
    x_frame.insert(0, "intercept", 1.0)
    x = x_frame.to_numpy(dtype=float)
    xtx_inv = np.linalg.pinv(x.T @ x)

    betas: dict[str, np.ndarray] = {}
    residuals: dict[str, np.ndarray] = {}
    for outcome in outcomes:
        y = clean[outcome].astype(float).to_numpy()
        beta = xtx_inv @ x.T @ y
        betas[outcome] = beta
        residuals[outcome] = y - x @ beta

    index_rows = []
    beta_vector = []
    treatment_positions = [list(x_frame.columns).index(name) for name in treatments]
    for outcome in outcomes:
        for treatment, position in zip(treatments, treatment_positions, strict=True):
            index_rows.append(
                {
                    "position": len(index_rows),
                    "outcome": outcome,
                    "horizon": _horizon_from_outcome(outcome),
                    "treatment": treatment,
                }
            )
            beta_vector.append(float(betas[outcome][position]))
    index = pd.DataFrame(index_rows)
    residual_matrix = np.column_stack([residuals[outcome] for outcome in outcomes])
    covariance = joint_treatment_cluster_covariance(
        x,
        residual_matrix,
        xtx_inv,
        treatment_positions,
        clean[cluster_cols[0]],
        clean[cluster_cols[1]],
    )

    beta_array = np.asarray(beta_vector)
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    coefficients = index.copy()
    coefficients["beta"] = beta_array
    coefficients["standard_error"] = se
    coefficients["t_stat"] = np.divide(
        beta_array, se, out=np.full_like(beta_array, np.nan), where=se > 0
    )
    coefficients["p_value"] = coefficients["t_stat"].map(normal_two_sided_p)
    coefficients["n_obs"] = len(clean)
    return JointSystemResult(coefficients, covariance, index, len(clean))


def joint_treatment_cluster_covariance(
    x: np.ndarray,
    residuals: np.ndarray,
    xtx_inv: np.ndarray,
    treatment_positions: Sequence[int],
    cluster_a: pd.Series,
    cluster_b: pd.Series,
) -> np.ndarray:
    """Compute all outcome-treatment covariance blocks in three cluster passes."""

    left = one_way_joint_treatment_covariance(
        x, residuals, xtx_inv, treatment_positions, cluster_a
    )
    right = one_way_joint_treatment_covariance(
        x, residuals, xtx_inv, treatment_positions, cluster_b
    )
    pair = cluster_a.astype(str).reset_index(drop=True) + "||" + cluster_b.astype(
        str
    ).reset_index(drop=True)
    intersection = one_way_joint_treatment_covariance(
        x, residuals, xtx_inv, treatment_positions, pair
    )
    covariance = left + right - intersection
    return (covariance + covariance.T) / 2.0


def one_way_joint_treatment_covariance(
    x: np.ndarray,
    residuals: np.ndarray,
    xtx_inv: np.ndarray,
    treatment_positions: Sequence[int],
    clusters: pd.Series,
) -> np.ndarray:
    values = pd.Series(clusters).reset_index(drop=True)
    projection = xtx_inv[np.asarray(treatment_positions, dtype=int), :]
    cluster_scores = []
    for value in values.dropna().unique():
        mask = values.eq(value).to_numpy()
        raw_score = x[mask].T @ residuals[mask]
        projected = projection @ raw_score
        cluster_scores.append(projected.T.reshape(-1))
    width = residuals.shape[1] * len(treatment_positions)
    scores = np.vstack(cluster_scores) if cluster_scores else np.zeros((0, width))
    n_obs, n_params = x.shape
    n_clusters = max(values.nunique(dropna=True), 1)
    correction = (n_clusters / max(n_clusters - 1, 1)) * (
        (n_obs - 1) / max(n_obs - n_params, 1)
    )
    return correction * (scores.T @ scores)


def two_way_cluster_cross_covariance(
    x: np.ndarray,
    residual_left: np.ndarray,
    residual_right: np.ndarray,
    xtx_inv: np.ndarray,
    cluster_a: pd.Series,
    cluster_b: pd.Series,
) -> np.ndarray:
    left = one_way_cluster_cross_covariance(
        x, residual_left, residual_right, xtx_inv, cluster_a
    )
    right = one_way_cluster_cross_covariance(
        x, residual_left, residual_right, xtx_inv, cluster_b
    )
    pair = cluster_a.astype(str).reset_index(drop=True) + "||" + cluster_b.astype(
        str
    ).reset_index(drop=True)
    intersection = one_way_cluster_cross_covariance(
        x, residual_left, residual_right, xtx_inv, pair
    )
    return left + right - intersection


def one_way_cluster_cross_covariance(
    x: np.ndarray,
    residual_left: np.ndarray,
    residual_right: np.ndarray,
    xtx_inv: np.ndarray,
    clusters: pd.Series,
) -> np.ndarray:
    values = pd.Series(clusters).reset_index(drop=True)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for value in values.dropna().unique():
        mask = values.eq(value).to_numpy()
        left_score = x[mask].T @ residual_left[mask]
        right_score = x[mask].T @ residual_right[mask]
        meat += np.outer(left_score, right_score)
    n_obs, n_params = x.shape
    n_clusters = max(values.nunique(dropna=True), 1)
    correction = (n_clusters / max(n_clusters - 1, 1)) * (
        (n_obs - 1) / max(n_obs - n_params, 1)
    )
    return correction * xtx_inv @ meat @ xtx_inv


def curve_joint_tests(
    coefficients: pd.DataFrame,
    covariance: np.ndarray,
    coefficient_index: pd.DataFrame,
) -> pd.DataFrame:
    """Test each stage's complete response curve against joint zero."""

    rows = []
    for treatment, group in coefficient_index.groupby("treatment", sort=False):
        positions = group.sort_values("horizon")["position"].astype(int).to_numpy()
        beta = coefficients.set_index("position").loc[positions, "beta"].to_numpy(dtype=float)
        cov = covariance[np.ix_(positions, positions)]
        statistic = float(beta @ np.linalg.pinv(cov) @ beta)
        rows.append(
            {
                "treatment": treatment,
                "test": "all_horizons_zero",
                "wald_chi2": statistic,
                "df": len(positions),
                "p_value": float(chi2.sf(statistic, len(positions))),
            }
        )
    return pd.DataFrame(rows)


def simultaneous_bands(
    coefficients: pd.DataFrame,
    covariance: np.ndarray,
    coefficient_index: pd.DataFrame,
    *,
    seed: int = 20260715,
    draws: int = 10_000,
    family: str = "stage",
) -> pd.DataFrame:
    """Construct Gaussian max-|t| bands from the supplied joint covariance."""

    rng = np.random.default_rng(seed)
    result = coefficients.copy()
    result["simultaneous_critical_value"] = np.nan
    result["simultaneous_ci_low"] = np.nan
    result["simultaneous_ci_high"] = np.nan
    groups = (
        [("all_stage_horizon", coefficient_index)]
        if family == "all"
        else list(coefficient_index.groupby("treatment", sort=False))
    )
    for _, group in groups:
        positions = group["position"].astype(int).to_numpy()
        cov = covariance[np.ix_(positions, positions)]
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        valid = se > 0
        correlation = np.eye(len(positions))
        if valid.any():
            denominator = np.outer(se, se)
            correlation = np.divide(
                cov,
                denominator,
                out=np.eye(len(positions)),
                where=denominator > 0,
            )
            correlation = nearest_psd_correlation(correlation)
        simulated = rng.multivariate_normal(np.zeros(len(positions)), correlation, size=draws)
        critical = float(np.quantile(np.max(np.abs(simulated), axis=1), 0.95))
        for position, standard_error in zip(positions, se, strict=True):
            beta = float(result.loc[result["position"].eq(position), "beta"].iloc[0])
            mask = result["position"].eq(position)
            result.loc[mask, "simultaneous_critical_value"] = critical
            result.loc[mask, "simultaneous_ci_low"] = beta - critical * standard_error
            result.loc[mask, "simultaneous_ci_high"] = beta + critical * standard_error
    return result


def adjusted_inference(
    conditional: JointSystemResult,
    first_stage_covariance: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Add first-stage refit covariance to the conditional response covariance."""

    if first_stage_covariance.shape != conditional.covariance.shape:
        raise ValueError(
            "First-stage covariance shape does not match conditional system: "
            f"{first_stage_covariance.shape} != {conditional.covariance.shape}"
        )
    total = conditional.covariance + first_stage_covariance
    total = (total + total.T) / 2.0
    out = conditional.coefficients.copy()
    out = out.rename(
        columns={
            "standard_error": "conditional_standard_error",
            "t_stat": "conditional_t_stat",
            "p_value": "conditional_p_value",
        }
    )
    se = np.sqrt(np.clip(np.diag(total), 0.0, None))
    out["bootstrap_adjusted_standard_error"] = se
    out["bootstrap_adjusted_t_stat"] = np.divide(
        out["beta"].to_numpy(),
        se,
        out=np.full(len(out), np.nan),
        where=se > 0,
    )
    out["bootstrap_adjusted_p_value"] = out["bootstrap_adjusted_t_stat"].map(
        normal_two_sided_p
    )
    out["bootstrap_adjusted_ci_low"] = out["beta"] - 1.96 * se
    out["bootstrap_adjusted_ci_high"] = out["beta"] + 1.96 * se
    return out, total


def nearest_psd_correlation(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    clipped = vectors @ np.diag(np.clip(values, 1e-10, None)) @ vectors.T
    scale = np.sqrt(np.clip(np.diag(clipped), 1e-10, None))
    return clipped / np.outer(scale, scale)


def _index_position(index: pd.DataFrame, outcome: str, treatment: str) -> int:
    row = index.loc[index["outcome"].eq(outcome) & index["treatment"].eq(treatment)]
    if len(row) != 1:
        raise ValueError(f"Joint coefficient index is not unique for {outcome}/{treatment}")
    return int(row.iloc[0]["position"])


def _horizon_from_outcome(outcome: str) -> int | float:
    if outcome.startswith("CAR_0_"):
        return int(outcome.rsplit("_", 1)[1])
    if outcome.startswith("AR_m"):
        return -int(outcome.removeprefix("AR_m"))
    if outcome.startswith("CAR_m"):
        return -int(outcome.rsplit("_m", 1)[1])
    return np.nan


def main_joint_system(data: pd.DataFrame, *, controls: Sequence[str]) -> JointSystemResult:
    return fit_joint_horizon_system(
        data,
        outcomes=[f"CAR_0_{horizon}" for horizon in HORIZONS],
        treatments=[
            "prepared_innovation_z",
            "question_innovation_z",
            "answer_innovation_z",
        ],
        controls=controls,
    )

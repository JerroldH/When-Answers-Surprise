"""ICAIF tables from the frozen aggregate results."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper_results"
OUT = ROOT / "outputs" / "tables"

def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _p(value: float) -> str:
    if value < 0.001:
        return "$<.001$"
    return f"{value:.3f}".replace("0.", ".")


def _plain_p(value: float) -> str:
    """Format a manuscript p-value without math delimiters."""
    if value < 0.001:
        return "<.001"
    return f"{value:.3f}".replace("0.", ".")


def _p_expression(value: float) -> str:
    """Format a complete inline p-value expression."""
    value_text = _plain_p(value)
    relation = "" if value_text.startswith("<") else "="
    return f"$p{relation}{value_text}$"


def _icaif_tables(
    primary: pd.DataFrame,
    context_pairing: pd.DataFrame,
    llm_ladders: pd.DataFrame,
    numbers: dict,
) -> dict[str, str]:
    p = primary.set_index("specification")
    rows = []
    for specification in ("P0", "Q2", "A3"):
        row = p.loc[specification]
        label = {"P0": "Prepared residual", "Q2": "Question residual", "A3": "Full-context answer residual"}[specification]
        rows.append(
            f"{label} & {row.coefficient_bps:.2f} & "
            f"{row.adjusted_se_bps:.2f} & [{row.ci_low_bps:.2f}, {row.ci_high_bps:.2f}] & "
            f"{_p(float(row.adjusted_p_value))} \\\\"
        )
    timing = numbers["holdout_timing"]
    daily = numbers["daily_pre_event_a3"]
    no_market = timing["exclude_market_hours"]
    timing_rows = [
        (
            "Exclude market-hours calls",
            f"{no_market['n_calls']:,}",
            f"{no_market['coefficient_bps']:.2f} bps",
            f"SE {no_market['standard_error_bps']:.2f}",
            _p(no_market["p_value"]),
        ),
        (
            "Equal slopes across timing groups",
            "1,901",
            rf"$\chi^2({timing['slope_equality_joint_df']})={timing['slope_equality_joint_chi2']:.2f}$",
            "three timing buckets",
            _p(timing["slope_equality_joint_p_value"]),
        ),
        (
            "Twenty daily pre-event coefficients",
            "1,901",
            rf"$\chi^2({daily['joint_df']})={daily['joint_chi2']:.2f}$",
            r"AR$[-20,-1]$",
            _p(daily["joint_p_value"]),
        ),
    ]
    table1 = r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{5.0pt}
\caption{2025 holdout estimates and event-time checks.}
\label{tab:icaif-primary}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrr@{}}
\toprule
\multicolumn{5}{@{}l}{\textbf{Panel A: 2025 holdout estimates}} \\
\addlinespace[1pt]
Tone residual & Coef. & Adj. SE & 95\% CI & $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}

\vspace{2pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrllr@{}}
\toprule
\multicolumn{5}{@{}l}{\textbf{Panel B: Event-time alignment checks}} \\
\addlinespace[1pt]
Check & $N$ & Estimate / statistic & SE / reference & $p$ \\
\midrule
""" + "\n".join(" & ".join(row) + r" \\" for row in timing_rows) + r"""
\bottomrule
\end{tabular*}

\vspace{2pt}
\begin{minipage}{\textwidth}
\footnotesize\textit{Notes:} Coefficients are basis points per one development-sample
SD. Panel A standard errors include 200 first-stage refits. Panel B standard
errors are clustered by firm and event date. AR$[-20,-1]$ denotes the daily
abnormal returns from trading day $-20$ through day $-1$.
\end{minipage}
\end{table*}
"""

    pairing = context_pairing.set_index("variant")
    pairing_a0 = pairing.loc["A0"]
    pairing_a3 = pairing.loc["A3"]
    decomposition = numbers["development_answer_decomposition"]
    d03 = decomposition["U3_plus_D03_common_sd|D03_common_z"]
    raw = decomposition["raw_tone_common_sd|raw_answer_tone_common_z"]
    expected = decomposition["expected_A3_common_sd|expected_A3_common_z"]
    residual = decomposition["residual_A3_common_sd|residual_A3_common_z"]
    joint_expected = decomposition[
        "expected_plus_residual_common_sd|expected_A3_common_z"
    ]
    joint_residual = decomposition[
        "expected_plus_residual_common_sd|residual_A3_common_z"
    ]
    difference = decomposition[
        "expected_plus_residual_common_sd|residual_minus_expected"
    ]
    pairing_rows = [
        (
            "A0: Pre-call",
            f"{pairing_a0['matched_r2']:.3f}",
            f"{pairing_a0['expected_deranged_r2']:.3f}",
            "0.00",
            "all tied",
            "n/a",
        ),
        (
            "A3: Full context",
            f"{pairing_a3['matched_r2']:.3f}",
            f"{pairing_a3['expected_deranged_r2']:.3f}",
            f"{pairing_a3['deranged_mse_increase_percent']:.2f}",
            f"{int(pairing_a3['positive_pairing_gain_calls']):,} / {int(pairing_a3['n_calls']):,}",
            _p(float(pairing_a3["one_sided_sign_p_value"])),
        ),
    ]
    decomposition_rows = [
        (
            "Predicted-tone change after adding context",
            f"{d03['coefficient_bps']:.2f}",
            f"{d03['standard_error_bps']:.2f}",
            _p(d03["p_value"]),
        ),
        (
            "Raw answer tone",
            f"{raw['coefficient_bps']:.2f}",
            f"{raw['standard_error_bps']:.2f}",
            _p(raw["p_value"]),
        ),
        (
            "Full-context answer residual",
            f"{residual['coefficient_bps']:.2f}",
            f"{residual['standard_error_bps']:.2f}",
            _p(residual["p_value"]),
        ),
    ]
    llm_a3 = llm_ladders.loc[llm_ladders["variant"].eq("A3")].set_index("scorer")
    llm_rows = []
    for scorer in ("Gemini", "DeepSeek"):
        row = llm_a3.loc[scorer]
        llm_rows.append(
            (
                scorer,
                f"{row['oof_r2']:.3f}",
                f"{row['max_placebo_abs_t']:.2f}",
                f"{row['car1_bps']:.1f} ({row['car1_t_stat']:.2f})",
                f"{row['car20_bps']:.1f} ({row['car20_t_stat']:.2f})",
            )
        )
    table2 = r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{5.0pt}
\caption{Context tests, development results, and alternative tone scores.}
\label{tab:icaif-credibility}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrr@{}}
\toprule
\multicolumn{6}{@{}l}{\textbf{Panel A: Does each prediction fit its own answer?}} \\
\addlinespace[1pt]
Context model & Own-turn $R^2$ & Mismatched-turn $R^2$ & MSE increase (\%) & Calls improved & $p$ \\
\midrule
""" + "\n".join(" & ".join(row) + r" \\" for row in pairing_rows) + r"""
\bottomrule
\end{tabular*}

\vspace{2pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrr@{}}
\toprule
\multicolumn{4}{@{}l}{\textbf{Panel B: How does added dialogue context enter the return decomposition?}} \\
\addlinespace[1pt]
Tone measure & Coef. (bps) & SE & $p$ \\
\midrule
""" + "\n".join(" & ".join(row) + r" \\" for row in decomposition_rows) + r"""
\bottomrule
\end{tabular*}

\vspace{2pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrr@{}}
\toprule
\multicolumn{5}{@{}l}{\textbf{Panel C: 2025 results with alternative tone scores}} \\
\addlinespace[1pt]
Scorer & Full-context OOF $R^2$ & A3 placebo $|t|$ & $\CAR^{(1)}$ bps ($t$) & $\CAR^{(20)}$ bps ($t$) \\
\midrule
""" + "\n".join(" & ".join(row) + r" \\" for row in llm_rows) + r"""
\bottomrule
\end{tabular*}

\vspace{2pt}
\begin{minipage}{\textwidth}
\footnotesize\textit{Notes:} Panel A uses 182,809 turns from 9,740 calls and
averages over all within-call mismatches. A0 has the same prediction for every
answer in a call, so its $p$ value is undefined. The A3 $p$ value is from a
one-sided sign test across calls. Panel B uses 9,583 calls and
scales all measures by the same development-sample tone SD. The first row is
estimated jointly with the A3 residual; the next two rows are separate
regressions. Entering the A3 prediction and residual jointly gives """ + f"{joint_expected['coefficient_bps']:.2f} bps\n" + f"(SE {joint_expected['standard_error_bps']:.2f}, {_p_expression(joint_expected['p_value'])}) and {joint_residual['coefficient_bps']:.2f} bps (SE {joint_residual['standard_error_bps']:.2f}, {_p_expression(joint_residual['p_value'])}); their {difference['coefficient_bps']:.2f}-bps\n" + f"difference has {_p_expression(difference['p_value'])}. " + r"""Panel C covers 34,001 turns, 1,954 first-stage calls,
and 1,901 response calls. Its placebo column reports the largest absolute
pre-event $t$ statistic for A3. Standard errors are clustered by firm and event
date. Parentheses contain $t$ statistics. OOF denotes out-of-fold.
\end{minipage}
\end{table*}
"""
    return {"table1_primary.tex": table1, "table2_credibility.tex": table2}


def _build_camera_ready_holdout() -> None:
    """Render the post-review comparison without modifying the frozen contract."""
    import hashlib
    import json

    source = RESULTS / "camera_ready_lm_holdout"
    path = source / "holdout_lm_response_diagnostic.csv"
    audit = json.loads((source / "diagnostic_manifest.json").read_text())
    record = next(r for r in audit["outputs"] if r["path"] == path.name)
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("Camera-ready diagnostic CSV checksum mismatch")
    frame = pd.read_csv(path)
    if len(frame) != 7 or not frame.n_calls.eq(1901).all() or not audit["same_sample_all"]:
        raise ValueError("Camera-ready comparison must retain the 1,901-call intersection")
    rows = []
    for row in frame.itertuples(index=False):
        if row.specification == "A3 + Delta":
            label = r"A3 residual" if row.term == "A3_common" else r"Prediction change $\Delta$"
        else:
            label = "Raw answer tone" if row.specification == "Raw" else f"{row.specification} residual"
        rows.append(f"{label} & {row.beta_bps:.2f} & {row.se_bps:.2f} & {_p(row.p_value)} " + r"\\")
    table = r"""% Generated from paper_results/camera_ready_lm_holdout.
\begin{table}[t]
\centering
\small
\caption{2025 LM answer-tone comparison on a common scale.}
\label{tab:icaif-holdout-ladder}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrr@{}}
\toprule
Tone measure & Coef. (bps) & SE & $p$ \\
\midrule
\multicolumn{4}{@{}l}{\textbf{A. Separate regressions}} \\ \addlinespace[2pt]
""" + "\n".join(rows[:5]) + r"""
\addlinespace[3pt]\midrule
\multicolumn{4}{@{}l}{\textbf{B. Joint regression}} \\ \addlinespace[2pt]
""" + "\n".join(rows[5:]) + r"""
\bottomrule
\end{tabular*}
\vspace{4pt}
\begin{minipage}{\columnwidth}
\footnotesize\textit{Notes:} Post-review diagnostic; 1,901 calls; outcome $\CAR^{(1)}$.
All models retain the original controls, fixed effects, and prepared/question
residuals. Coefficients use one development raw-tone SD; Table~\ref{tab:icaif-primary} uses residual
SDs. SEs are firm/date-clustered, conditional on tone predictions.
$\Delta$ is the A0-to-A3 prediction change (Equation~\ref{eq:icaif-a3-a0-change}).
\end{minipage}
\end{table}
"""
    _write(OUT / "table3_holdout_ladder.tex", table)


def build():
    tables = _icaif_tables(pd.read_csv(RESULTS/"holdout_primary.csv"), pd.read_csv(RESULTS/"development_context_pairing.csv"), pd.read_csv(RESULTS/"llm_appendix/holdout_ladders.csv"), json.loads((RESULTS/"manuscript_numbers.json").read_text()))
    for name,text in tables.items():
        _write(OUT/name,text)
    _build_camera_ready_holdout()

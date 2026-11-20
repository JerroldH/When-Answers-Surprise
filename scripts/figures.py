"""Build the ICAIF statistical figures from frozen paper results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / 'outputs' / 'figures'
RESULTS = ROOT / "paper_results"

NAVY = "#24374B"
BLUE = "#4C78A8"
TEAL = "#4E9F91"
CORAL = "#D66B5D"
GOLD = "#B98A3D"
GRAY = "#687887"
LIGHT = "#DCE3E8"


def _save(fig: plt.Figure, stem: str, *, tight: bool = True) -> None:
    for suffix in ("pdf", "svg", "png"):
        output_path = FIGURE_DIR / f"{stem}.{suffix}"
        metadata = {
            "pdf": {"CreationDate": None, "ModDate": None},
            "svg": {"Date": None},
            "png": None,
        }[suffix]
        fig.savefig(
            output_path,
            dpi=600 if suffix == "png" else 300,
            bbox_inches="tight" if tight else None,
            pad_inches=0.035,
            facecolor="white",
            metadata=metadata,
        )
        if suffix == "svg":
            svg_text = output_path.read_text(encoding="utf-8")
            clean_svg = "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n"
            output_path.write_text(clean_svg, encoding="utf-8")
    plt.close(fig)


def build_answer_evidence() -> None:
    horizons = pd.read_csv(RESULTS / "holdout_horizons.csv")
    primary = pd.read_csv(RESULTS / "holdout_primary.csv").set_index("specification")
    daily_pre = pd.read_csv(RESULTS / "holdout_daily_pre_event.csv")
    if set(horizons["period"].astype(str)) != {"2025"}:
        raise ValueError("ICAIF result figure requires frozen 2025 horizon data")

    a3 = (
        horizons.loc[horizons["specification"].eq("A3")]
        .sort_values("horizon")
        .copy()
    )
    h1 = a3["horizon"].eq(1)
    if int(h1.sum()) != 1:
        raise ValueError("ICAIF figure requires one full-context answer row at h=1")
    figure_coefficient = float(a3.loc[h1, "coefficient_bps"].iloc[0])
    table_coefficient = float(primary.loc["A3", "coefficient_bps"])
    if abs(figure_coefficient - table_coefficient) > 1e-8:
        raise ValueError("ICAIF h=1 coefficient disagrees with the primary table")
    if set(daily_pre["event_day"].astype(int)) != set(range(-20, 0)):
        raise ValueError("ICAIF timing figure requires all full-context answer daily leads")
    daily_pre = daily_pre.sort_values("event_day")

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55), gridspec_kw={"width_ratios": [1.1, 0.9]}, facecolor="white")
    ax = axes[0]
    ax.errorbar(
        a3["horizon"],
        a3["coefficient_bps"],
        yerr=[a3["coefficient_bps"] - a3["ci_low_bps"], a3["ci_high_bps"] - a3["coefficient_bps"]],
        color=CORAL,
        marker="D",
        markerfacecolor="white",
        markeredgewidth=1.4,
        linestyle="-",
        linewidth=1.7,
        capsize=3,
    )
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.set_title("A. 2025 holdout response", loc="left", fontsize=9.2, fontweight="bold", color=NAVY)
    ax.set_xlabel("Trading days in cumulative window", fontsize=8)
    ax.set_ylabel("Full-context answer coefficient\n(bps per 1 SD)", fontsize=8)
    ax.set_xticks([1, 3, 5, 10, 20])
    ax.set_ylim(-20, 145)
    ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax.tick_params(labelsize=7.2)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.fill_between(
        daily_pre["event_day"],
        daily_pre["simultaneous_ci_low_bps"],
        daily_pre["simultaneous_ci_high_bps"],
        color=TEAL,
        alpha=0.11,
        linewidth=0,
        label="95% simultaneous band",
    )
    ax.errorbar(
        daily_pre["event_day"],
        daily_pre["beta_bps"],
        yerr=[
            daily_pre["beta_bps"] - daily_pre["pointwise_ci_low_bps"],
            daily_pre["pointwise_ci_high_bps"] - daily_pre["beta_bps"],
        ],
        color=TEAL,
        marker="o",
        markersize=3.2,
        markerfacecolor="white",
        linewidth=1.2,
        capsize=2,
        label="Pointwise 95% CI",
    )
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.set_xlabel("Trading day relative to event", fontsize=8)
    ax.set_ylabel("Full-context answer coefficient\n(bps per 1 SD)", fontsize=8)
    ax.set_title("B. Pre-event timing check", loc="left", fontsize=9.2, fontweight="bold", color=NAVY)
    ax.set_xlim(-20.7, -0.3)
    ax.set_xticks([-20, -15, -10, -5, -1])
    ax.set_ylim(-42, 42)
    ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax.tick_params(labelsize=7.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.98,
        0.96,
        "20-day joint p = .246\nHolm-adjusted p = .492",
        transform=ax.transAxes,
        fontsize=6.8,
        color=GRAY,
        ha="right",
        va="top",
    )

    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.84, wspace=0.42)
    _save(fig, "figure2_answer_evidence")


def build_core_evidence() -> None:
    pairing = pd.read_csv(RESULTS / "development_context_pairing.csv").set_index("variant")
    lm = pd.read_csv(RESULTS / "holdout_lm_ladders.csv")
    llm = pd.read_csv(RESULTS / "llm_appendix" / "holdout_ladders.csv")

    if not {"A0", "A3"}.issubset(pairing.index):
        raise ValueError("Figure 3 requires A0 and A3 pairing results")
    if pairing.loc[["A0", "A3"], "n_calls"].astype(int).nunique() != 1:
        raise ValueError("Figure 3 pairing rows must use the same calls")
    if pairing.loc[["A0", "A3"], "n_turns"].astype(int).nunique() != 1:
        raise ValueError("Figure 3 pairing rows must use the same turns")

    lm = lm.loc[
        lm["sample"].eq("holdout_2025")
        & lm["component"].eq("answer")
        & lm["variant"].isin(["A0", "A1", "A2", "A3"])
    ].copy()
    llm = llm.loc[
        llm["sample"].eq("holdout_2025")
        & llm["component"].eq("answer")
        & llm["variant"].isin(["A0", "A1", "A2", "A3"])
        & llm["scorer"].isin(["Gemini", "DeepSeek"])
    ].copy()
    ladders = pd.concat(
        [
            lm[["scorer", "variant", "n_turns", "n_calls", "n_features", "oof_r2"]],
            llm[["scorer", "variant", "n_turns", "n_calls", "n_features", "oof_r2"]],
        ],
        ignore_index=True,
    )
    variants = ["A0", "A1", "A2", "A3"]
    scorers = ["LM dictionary", "Gemini", "DeepSeek"]
    expected = {(scorer, variant) for scorer in scorers for variant in variants}
    if set(zip(ladders["scorer"], ladders["variant"], strict=True)) != expected:
        raise ValueError("Figure 3 requires a complete LM/Gemini/DeepSeek A0--A3 grid")
    if set(ladders["n_turns"].astype(int)) != {34_001}:
        raise ValueError("Figure 3 ladders must use the same 34,001 holdout turns")
    if set(ladders["n_calls"].astype(int)) != {1_954}:
        raise ValueError("Figure 3 ladders must use the same 1,954 holdout calls")
    expected_features = {"A0": 29, "A1": 38, "A2": 42, "A3": 58}
    for _, row in ladders.iterrows():
        if int(row["n_features"]) != expected_features[row["variant"]]:
            raise ValueError("Figure 3 ladder feature contracts do not match")

    # Reserve space beside each axis for direct labels at the line endpoints.
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8, "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(7.05, 2.4), facecolor="white")
        left = fig.add_axes([.075, .23, .29, .61])
        right = fig.add_axes([.60, .23, .27, .61])
        for column, color, marker, linestyle, label in [
            ("matched_r2", NAVY, "D", "-", "Own answer"),
            ("expected_deranged_r2", CORAL, "s", "--", "Other answers"),
        ]:
            values = pairing.loc[["A0", "A3"], column].astype(float).to_numpy()
            left.plot([0, 1], values, marker=marker, linestyle=linestyle,
                      color=color, markerfacecolor="white", linewidth=1.25, markersize=4)
            left.text(1.09, values[-1], label, ha="left", va="center",
                      fontsize=7.2, color=color)
        left.set(xlim=(-.08, 1.06), ylim=(0, .052), xticks=[0, 1],
                 xticklabels=["A0\nPre-call", "A3\nFull context"],
                 yticks=[0, .01, .02, .03, .04, .05], ylabel="Out-of-fold R²")
        left.set_title("A. Within-call matching", loc="left", fontsize=9,
                       fontweight="bold", pad=10)
        for scorer, label, color, marker, linestyle in [
            ("LM dictionary", "LM", NAVY, "D", "-"),
            ("Gemini", "Gemini", TEAL, "s", "--"),
            ("DeepSeek", "DeepSeek", CORAL, "o", ":"),
        ]:
            values = (ladders.loc[ladders["scorer"].eq(scorer)]
                      .set_index("variant").loc[variants, "oof_r2"].astype(float))
            gains = 100.0 * (values - values.iloc[0])
            right.plot(range(4), gains, marker=marker, linestyle=linestyle,
                       color=color, markerfacecolor="white", linewidth=1.25, markersize=4)
            right.text(3.18, gains.iloc[-1], label, va="center", ha="left",
                       color=color, fontsize=7.4)
        right.set(xticks=range(4), xticklabels=variants, xlim=(-.1, 3.12),
                  ylim=(-.10, 4.1), xlabel="Answer context", ylabel="Δ out-of-fold R² (pp)")
        right.set_title("B. Context gains", loc="left", fontsize=9,
                        fontweight="bold", pad=10)
        for axis in (left, right):
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color=LIGHT, linewidth=.6)
            axis.set_axisbelow(True)
            axis.tick_params(length=3, width=.7, labelsize=7.2)
        _save(fig, "figure3_core_evidence", tight=False)


def build() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "when-answers-surprise-icaif",
        }
    )
    # Figure 1 is maintained in figure1_economic_timeline.drawio; export via draw.io.
    # The camera-ready ICAIF variant differs from the older shared/FinNLP figure.
    # Do not overwrite it from this statistical-figure builder.
    build_answer_evidence()
    build_core_evidence()
    print("Built ICAIF statistical figures from paper_results. Figure 1 is draw.io")

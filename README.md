# When Answers Surprise

*Sequential Tone Residualization and Abnormal Returns in Earnings Calls*

Jerrold Huang and Nathan Zhang · ACM ICAIF 2026

[Paper (standalone author version, 23-page PDF)](docs/when-answers-surprise.pdf) ·
[Oral presentation (PDF, 15 slides + appendix)](docs/icaif-2026-slides.pdf) ·
[ACM publication](https://doi.org/10.1145/3858063.3858890)

How much of a manager’s tone could you have anticipated before hearing the answer?
We estimate expected tone from what was already available, including the analyst’s
question and the earlier conversation. Slides 5–6 walk through an answer that
sounds positive but falls short of what the question led the model to expect.

In a held-out sample of 1,901 earnings calls from 2025, a one-standard-deviation increase
in call-level answer surprise was associated with a 0.615 percentage point higher
one-day market-adjusted return (95% CI 0.248 to 0.982 percentage points).

## Rebuild the ICAIF proceedings exhibits

The figure and table numbers below refer to the ACM proceedings version.
The standalone paper includes additional examples and technical appendices.
From this repository's root, with Python 3.12 and uv:

```sh
uv sync --locked --python 3.12 --extra dev --extra publication
uv run --no-sync python scripts/reproduce.py
uv run --no-sync python -m pytest -q
```

Installing packages may need internet access. The exhibit build itself needs no
network, WRDS account, LLM key, LaTeX installation, or data outside this tree.
It verifies the frozen inputs in `checksums.sha256`, checks the published numeric
anchors, and writes:

| Exhibit | Location under `outputs/` | Source |
| --- | --- | --- |
| Fig. 1: research design | `figures/figure1_economic_timeline.pdf` | Camera-ready vector PDF, copied unchanged; editable Draw.io source in `design/` |
| Fig. 2: answer response and event timing | `figures/figure2_answer_evidence.*` | 2025 horizon, uncertainty and pre-event estimates |
| Fig. 3: context pairing and tone measures | `figures/figure3_core_evidence.*` | Development context comparison and 2025 tone ladders |
| Tables 1, 2, 3 | `tables/table{1_primary,2_credibility,3_holdout_ladder}.tex` | Camera-ready table generators and frozen aggregates |

The statistical figures are generated as PDF, SVG and PNG. The tables are LaTeX
fragments for the paper. Table 1 scales each residual by its development-sample
standard deviation and includes first-stage estimation uncertainty. Table 3
compares tone measures on a common raw-tone scale with conditional clustered
standard errors.

## Re-estimation with licensed inputs

The [unified-panel pipeline](docs/holdout_2025_runbook.md) re-estimates the regressions
from licensed inputs. On the same 1,901 holdout calls, the answer coefficient is
**61.35 bps** in the refit and **61.47 bps** in the published paper.

Real-data scoring requires the complete study-version LM master dictionary;
a missing, truncated or different CSV stops the run. This route does not call
an LLM. Re-estimating the alternative scores separately requires provider
access and permission to process the inputs.

Table 1 of the standalone paper illustrates three exchanges with short, masked
transcript excerpts. The CSVs in `paper_results/` contain aggregate estimates.
Full transcripts, security-level returns, event panels, turn-level scores, API
responses and credentials are excluded from the package.

Source snapshot: `codex/camera-ready`, commit
`76ec9d2d104a2a31cf3ce2aa0a90b825abf83597`. The private research repository
and its commit history are not part of this package.

## License

The replication code is available under the [MIT License](LICENSE).
The authors’ original content in the paper and slides is available under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Credit Jerrold Huang and Nathan Zhang when reusing it. Third-party materials,
including quoted transcript text, and the underlying commercial data remain
subject to their respective rights and terms.


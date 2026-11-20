# When Answers Surprise

Replication materials for Jerrold Huang and Nathan Zhang, *When Answers Surprise:
Sequential Tone Residualization and Abnormal Returns in Earnings Calls*
([ICAIF 2026](https://doi.org/10.1145/3858063.3858890)).

## Rebuild the paper exhibits

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
fragments for the paper, not standalone documents. For orientation, Table 1
reports a **61.47 bps** full-context answer coefficient across **1,901** holdout
calls (adjusted SE 18.72). Its scale is one development-sample residual SD.
Table 3 instead uses a common raw-tone SD and conditional clustered SEs. The
two tables should not be expected to have identical coefficients.

## Re-estimation with licensed inputs

The [unified-panel pipeline](docs/holdout_2025_runbook.md) re-estimates the regressions
from licensed inputs. On the same 1,901 holdout calls, the answer coefficient is
**61.35 bps** in the refit and **61.47 bps** in the published paper.

Real-data scoring requires the complete study-version LM master dictionary;
a missing, truncated or different CSV stops the run. This route does not call
an LLM. Re-estimating the alternative scores separately requires provider
access and permission to process the inputs.

No transcripts, excerpts, security-level returns, event panels, turn-level
scores, API responses or credentials are distributed. The CSVs in
`paper_results/` contain aggregate estimates only. Licensed inputs remain
subject to their providers' terms.

Source snapshot: `codex/camera-ready`, commit
`76ec9d2d104a2a31cf3ce2aa0a90b825abf83597`. The private research repository
and its commit history are not part of this package.


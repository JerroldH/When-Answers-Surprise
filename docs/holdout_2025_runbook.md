# Re-estimate the 2025 holdout

Use one event panel for the development fits and the 2025 holdout. Run these
commands from the repository root (Bash, Python 3.12). No LLM calls are made.

## Prepare the inputs

Install the locked environment:

```bash
uv sync --locked --python 3.12 --extra dev --extra publication
```

If you already have the licensed exports and strict historical staging files,
skip the WRDS steps below. Otherwise, with WRDS access configured, run:

```bash
uv run --no-sync python scripts/run_task.py export_raw_wrds
uv run --no-sync python scripts/run_task.py build_staging_dag
uv run --no-sync python scripts/run_task.py export_holdout_2025_wrds
```

The historical root needs `derived/transcript_ibes_panel.parquet`, the staged
CIQ components, membership and security master, and `raw/wrds_exports/`.
Use the study's legacy CRSP daily data through 2023 and CIZ from 2024 onwards.
Historical IBES exports must include 2023 consensus snapshots (by STATPERS)
and actuals (by fiscal period end), so early-2024 calls can match to 2023
earnings. The holdout raw directory contains the 2024 and 2025 exports.

Obtain the complete `Loughran-McDonald_MasterDictionary_1993-2025.csv`
separately and put it in the dictionary directory. The scorer verifies its
checksum. See the [data contract](data_contract.md) for row keys.

## Run the refit

Choose a **new private output directory outside the checkout**. The run writes
licensed event and turn records there. It does not require prior model results.

```bash
uv run --no-sync python scripts/run_task.py run_strict_lm_panel \
  --historical-root data/wrds_sp500_2018_2025 \
  --holdout-raw data/wrds_holdout_2025_ciz/raw/wrds_exports \
  --dictionary-dir data/external \
  --output-dir /path/to/new/private/lm_run
```

The pipeline matches calls, tunes Ridge on earlier years, and computes residual
scales from the 2020--2024 out-of-fold response sample. It freezes the 2025
residuals before merging event-window returns. The development scales are in
`protocol/holdout_protocol.json`. The coefficients in
`response/holdout_2025_primary_coefficients.csv` are decimal returns; multiply
by 10,000 for basis points per development-sample residual SD.

## Check the result

With the study's local data vintage, the refit uses 9,591 development calls and
the same 1,901 holdout calls (499 firms, 173 event dates) as the paper.

| CAR[0,1], bps | Published Table 1 | Unified-panel refit |
| --- | ---: | ---: |
| P0: prepared remarks | 99.04 | 97.61 |
| Q2: analyst questions | 113.85 | 114.06 |
| A3: management answers | 61.47 | 61.35 |

The refit reports conditional standard errors; it does not repeat the paper's
200-refit uncertainty adjustment. To rebuild the published tables and figures
from their frozen aggregates instead, use `python scripts/reproduce.py`.

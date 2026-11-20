# Data Contract

## Local Roots

- `data/wrds_sp500_2018_2025/raw/wrds_exports/`: licensed raw WRDS exports.
- `data/wrds_sp500_2018_2025/staging/`: cleaned intermediate staging tables.
- `data/wrds_sp500_2018_2025/derived/`: transcript/IBES event panel.
- `data/wrds_sp500_2018_2025/analysis/`: empirical outputs and diagnostics.

## Protected Inputs

Raw WRDS, Capital IQ, IBES, CRSP, and Compustat data are protected and excluded
from Git. The repository provides export configs and code, not licensed data.

## Audit Expectations

Staging and matching tasks should report row counts, duplicate keys, match
rates, and output paths. Crosswalk or matching steps must not silently change
the unit of observation.

## Empirical Keys

- Event-level response sample: unique at `transcriptid`.
- Turn-level first-stage frames: unique at `transcriptid, turn_index`.
- Time-forward first-stage folds: test-year rows train only on earlier years.


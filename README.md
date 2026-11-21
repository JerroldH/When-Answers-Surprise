# When Answers Surprise: Sequential Tone Residualization and Abnormal Returns in Earnings Calls

Jerrold Huang and Nathan Zhang · ACM International Conference on AI in Finance (ICAIF), 2026

[Paper](docs/when-answers-surprise.pdf) · [Slides](docs/icaif-2026-slides.pdf) · [ACM proceedings](https://doi.org/10.1145/3858063.3858890) · [Citation](CITATION.cff)

An optimistic answer on an earnings call can still fall short of what the analyst’s question led listeners to expect. We measure *answer surprise* as the difference between a manager’s observed tone and the tone a model predicts from information available before the answer. We train the model on earlier years and compare predictions as we add the prepared remarks, earlier exchanges, the current question, and the analyst’s history.

Across 1,901 earnings calls in the 2025 holdout, a one-standard-deviation increase in call-level answer surprise was associated with one-day market-adjusted returns that were 61.5 basis points higher (95% CI, 24.8–98.2 basis points).

## Reproduce the figures and tables

From the repository root, use Python 3.12 and uv to rebuild Figures 1–3 and Tables 1–3 of the ACM proceedings paper.

```bash
uv sync --locked --python 3.12 --extra dev --extra publication
uv run --no-sync python scripts/reproduce.py
```

The script checks the bundled inputs against [their checksums](checksums.sha256) and verifies the published estimates. It writes the figures to `outputs/figures/` and the LaTeX tables to `outputs/tables/`. The [aggregate estimates](paper_results/) and [research design diagram](design/) are included in the repository.

Run the tests in the same environment.

```bash
uv run --no-sync python -m pytest -q
```

## Re-estimate the models

The [replication guide](docs/holdout_2025_runbook.md) describes how to construct the event panel and refit the dictionary-based models using licensed data and the Loughran–McDonald master dictionary. The guide compares the refit with the published estimates on the same 1,901 calls. The [data documentation](docs/data_contract.md) defines the required inputs and observation keys.

## License

The [MIT License](LICENSE) covers the replication code. The authors’ original content in the paper and slides is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Third-party materials, including transcript quotations and commercial data, retain their respective terms.

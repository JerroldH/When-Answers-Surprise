import hashlib
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper_results"


def test_holdout_protocol_preserves_original_digest():
    path = ROOT / "configs/wrds_holdout_2025/holdout_protocol.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == path.with_suffix(".sha256").read_text().strip()


def test_alternative_scores_and_post_review_scale():
    llm = pd.read_csv(RESULTS / "llm_appendix/holdout_ladders.csv").query("variant == 'A3'").set_index("scorer")
    assert set(llm.index) == {"Gemini", "DeepSeek"}
    for name, expected in (("Gemini", 100.8), ("DeepSeek", 112.0)):
        assert llm.loc[name, "car1_bps"] == pytest.approx(expected, abs=0.05)
    diag = pd.read_csv(RESULTS / "camera_ready_lm_holdout/holdout_lm_response_diagnostic.csv")
    assert set(diag.n_calls) == {1901}
    assert set(diag.unit) == {"bps_per_development_raw_tone_sd"}
    delta = diag.loc[diag.term.eq("delta_common")].squeeze()
    assert delta.p_value == pytest.approx(0.232, abs=0.0005)


import hashlib

import pytest

from when_answers_surprise import semantic_scoring as lm


def test_production_dictionary_must_be_present_and_complete(tmp_path):
    path = tmp_path / lm.LM_FILENAME
    with pytest.raises(FileNotFoundError, match="official"):
        lm.load_lm_dictionary(tmp_path)

    for damaged in (b"", b"Word\nGAIN\n", b"Word,Positive,Negative\nGAIN,2025,0\n"):
        path.write_bytes(damaged)
        with pytest.raises(ValueError, match="Incomplete or different LM dictionary"):
            lm.load_lm_dictionary(tmp_path)


def test_verified_dictionary_preserves_tone_under_lf_and_crlf(tmp_path, monkeypatch):
    header = "Word," + ",".join(lm.LM_CATEGORIES)
    data = (header + "\nLOSS,2025,0,0,0,0,0,0,0\nGAIN,0,2025,0,0,0,0,0,0\n").encode()
    monkeypatch.setattr(lm, "LM_SHA256", hashlib.sha256(data).hexdigest())
    path = tmp_path / lm.LM_FILENAME
    for sep in (b"\n", b"\r\n"):
        path.write_bytes(data.replace(b"\n", sep))
        dictionary = lm.load_lm_dictionary(tmp_path)
        assert dictionary.source_path == path
        assert lm.score_lm_text("gain gain loss flat", dictionary)["lm_net_tone"] == 250.0

    # Even a syntactically valid CSV missing one vocabulary row must not pass.
    path.write_bytes(data[:data.index(b"GAIN")])
    with pytest.raises(ValueError, match="Incomplete or different LM dictionary"):
        lm.load_lm_dictionary(tmp_path)

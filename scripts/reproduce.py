"""Rebuild the ICAIF tables and figures using only frozen aggregate inputs."""
import hashlib
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import figures
import tables
from validate_results import validate_results

ROOT = Path(__file__).resolve().parents[1]


def verify_inputs():
    records = dict(line.split("  ", 1)[::-1] for line in
                   (ROOT / "checksums.sha256").read_text().splitlines())
    files = {p.relative_to(ROOT).as_posix() for name in ("paper_results", "design")
             for p in (ROOT / name).rglob("*") if p.is_file()}
    if files != records.keys():
        raise ValueError(f"Frozen input set changed: {sorted(files ^ records.keys())}")
    for name, digest in records.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    return len(records)


if __name__ == "__main__":
    n = verify_inputs()
    validate_results()
    tables.build()
    figures.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figures.build()
    shutil.copyfile(ROOT / "design/figure1_economic_timeline.pdf",
                    figures.FIGURE_DIR / "figure1_economic_timeline.pdf")
    print(f"Verified {n} inputs. ICAIF Tables 1-3 and Figures 1-3: {ROOT / 'outputs'}")


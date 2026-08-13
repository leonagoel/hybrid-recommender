from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "example_small.csv"
RUNNER = ROOT / "scripts" / "run_pipeline_example.py"


def test_example_small_csv_exists_and_is_valid_fixture():
    assert DATASET.exists(), "example_small.csv should exist in the datasets directory"

    stat = DATASET.stat()
    assert 20 <= sum(1 for _ in DATASET.open("r", encoding="utf-8")) - 1 <= 100
    assert stat.st_size < 50 * 1024, "example_small.csv should stay under 50 KB"

    import pandas as pd

    df = pd.read_csv(DATASET)
    required = {"title", "description", "category", "user_id", "rating"}
    assert required.issubset(df.columns)
    assert df["rating"].notna().all()
    assert df["title"].astype(str).str.len().gt(0).all()


def test_example_runner_exits_successfully():
    result = subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Top recommendations:" in result.stdout
    assert "hybrid=" in result.stdout

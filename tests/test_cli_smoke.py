import os
from pathlib import Path

from click.testing import CliRunner

from src.cli import cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DATASET = PROJECT_ROOT / "datasets" / "sample_products.csv"


def test_cli_help_lists_commands():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "build" in result.output
    assert "evaluate" in result.output
    assert "recommend" in result.output
    assert "run-demo" in result.output


def test_run_demo_invokes_pipeline(monkeypatch):
    runner = CliRunner()

    def fake_build_pipeline(dataset_path):
        class FakeRow:
            def __init__(self, title):
                self.title = title

            def __getitem__(self, item):
                if item == "title":
                    return self.title
                return None

        class FakeIlocIndex:
            def __getitem__(self, item):
                return FakeRow("Demo Item")

            def __call__(self):
                return FakeRow("Demo Item")

        class FakeItemFrame:
            def __init__(self):
                self.empty = False
                self.iloc = FakeIlocIndex()

            def __getitem__(self, item):
                return ["Demo Item"]

            def __len__(self):
                return 1

        class FakeModel:
            def recommend(self, title, top_n=5):
                return [{"title": "Demo Item", "hybrid_score": 0.9}]

        return (
            {"user_id": ["u1"], "title": ["Demo Item"]},
            FakeItemFrame(),
            object(),
            object(),
            FakeModel(),
        )

    monkeypatch.setattr("src.cli.build_pipeline", fake_build_pipeline)
    monkeypatch.setattr("src.cli.save_models", lambda *args, **kwargs: None)

    result = runner.invoke(cli, ["run-demo"])

    assert result.exit_code == 0
    assert "Demo" in result.output or "Sample recommendations" in result.output


def test_invalid_dataset_path_returns_non_zero_exit_code():
    runner = CliRunner()
    result = runner.invoke(cli, ["build", "--dataset", "missing.csv"])

    assert result.exit_code != 0
    assert "Dataset not found" in result.output


def test_data_path_environment_is_used(monkeypatch):
    runner = CliRunner()
    monkeypatch.setenv("DATA_PATH", str(EXAMPLE_DATASET))

    result = runner.invoke(cli, ["build", "--force"])

    assert result.exit_code == 0
    assert "Using dataset" in result.output

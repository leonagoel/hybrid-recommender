"""Command-line interface for the Hybrid Recommender repository."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import click
import joblib

from src.data.dataset_manager import DatasetManager
from src.evaluation.evaluation import run_evaluation
from src.model.collaborative_model import CollaborativeRecommender
from src.model.content_model import ContentRecommender
from src.model.hybrid_model import HybridRecommender


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_DATASETS = [
    PROJECT_ROOT / "datasets" / "sample_products.csv",
    PROJECT_ROOT / "datasets" / "ratings.csv",
    PROJECT_ROOT / "datasets" / "books.csv",
]
MODEL_METADATA = MODEL_DIR / "metadata.json"


def resolve_dataset_path(dataset_path: str | None = None) -> Path:
    if dataset_path:
        path = Path(dataset_path)
        if not path.exists():
            raise click.ClickException(f"Dataset not found: {dataset_path}")
        if path.is_dir():
            raise click.ClickException(f"Dataset path is a directory, not a file: {dataset_path}")
        return path

    env_path = os.getenv("DATA_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    for candidate in DEFAULT_DATASETS:
        if candidate.exists():
            return candidate

    raise click.ClickException(
        "No dataset path provided and no default dataset was found in datasets/. "
        "Set DATA_PATH or provide --dataset."
    )


def dataset_summary(interaction_df: Any, item_df: Any) -> str:
    users = interaction_df["user_id"].nunique()
    items = interaction_df["title"].nunique()
    rows = len(interaction_df)
    titles = len(item_df)
    return (
        f"dataset rows={rows}, users={users}, unique titles={items}, "
        f"item records={titles}"
    )


def ensure_model_dir() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def metadata_for_dataset(dataset_path: Path) -> dict[str, Any]:
    return {
        "dataset_path": str(dataset_path.resolve()),
        "dataset_mtime": dataset_path.stat().st_mtime,
    }


def metadata_matches(dataset_path: Path, metadata: dict[str, Any]) -> bool:
    return (
        metadata.get("dataset_path") == str(dataset_path.resolve())
        and metadata.get("dataset_mtime") == dataset_path.stat().st_mtime
    )


def save_models(
    dataset_path: Path,
    content_model: ContentRecommender,
    collab_model: CollaborativeRecommender | None,
    hybrid_model: HybridRecommender,
    item_df: Any,
    interaction_df: Any,
) -> None:
    ensure_model_dir()
    joblib.dump(content_model, MODEL_DIR / "content_model.joblib")
    if collab_model is not None:
        joblib.dump(collab_model, MODEL_DIR / "collab_model.joblib")
    else:
        if (MODEL_DIR / "collab_model.joblib").exists():
            (MODEL_DIR / "collab_model.joblib").unlink()
    joblib.dump(hybrid_model, MODEL_DIR / "hybrid_model.joblib")
    joblib.dump(item_df, MODEL_DIR / "item_df.joblib")
    joblib.dump(interaction_df, MODEL_DIR / "interaction_df.joblib")
    MODEL_METADATA.write_text(json.dumps(metadata_for_dataset(dataset_path), indent=2))


def load_models(dataset_path: Path) -> tuple[ContentRecommender | None, CollaborativeRecommender | None, HybridRecommender | None]:
    if not MODEL_METADATA.exists():
        return None, None, None

    try:
        metadata = json.loads(MODEL_METADATA.read_text())
    except Exception:
        return None, None, None

    if not metadata_matches(dataset_path, metadata):
        return None, None, None

    content_model = None
    collab_model = None
    hybrid_model = None

    try:
        content_path = MODEL_DIR / "content_model.joblib"
        hybrid_path = MODEL_DIR / "hybrid_model.joblib"
        if content_path.exists() and hybrid_path.exists():
            content_model = joblib.load(content_path)
            hybrid_model = joblib.load(hybrid_path)
            collab_path = MODEL_DIR / "collab_model.joblib"
            if collab_path.exists():
                collab_model = joblib.load(collab_path)
    except Exception:
        return None, None, None

    return content_model, collab_model, hybrid_model


def build_pipeline(dataset_path: Path) -> tuple[Any, Any, ContentRecommender, CollaborativeRecommender | None, HybridRecommender]:
    try:
        dm = DatasetManager()
        dm.load_csv(str(dataset_path))
        interaction_df, item_df = dm.merge_all()
    except Exception as exc:  # pragma: no cover - exercised via Click error handling
        raise click.ClickException(f"Unable to read dataset '{dataset_path}': {exc}") from exc

    click.echo("Building ContentRecommender...")
    content_model = ContentRecommender(item_df)

    collab_model = None
    if interaction_df["user_id"].nunique() > 1 and interaction_df["title"].nunique() > 1:
        click.echo("Building CollaborativeRecommender...")
        collab_model = CollaborativeRecommender(interaction_df)
    else:
        click.echo("Skipping CollaborativeRecommender: not enough users or items")

    click.echo("Building HybridRecommender...")
    hybrid_model = HybridRecommender(content_model, collab_model, item_df)

    return interaction_df, item_df, content_model, collab_model, hybrid_model


def get_recommendations(hybrid_model: HybridRecommender, title: str, top_n: int) -> list[dict[str, Any]]:
    if not title or not str(title).strip():
        raise click.ClickException("Recommendation title must be a non-empty string.")
    if top_n <= 0:
        raise click.ClickException("Top-N must be a positive integer.")
    try:
        recs = hybrid_model.recommend(title=title, top_n=top_n)
    except Exception as exc:
        raise click.ClickException(f"Unable to generate recommendations: {exc}") from exc
    if not recs:
        return []
    return recs


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """CLI for build, evaluate, recommend, and demo workflows."""


@cli.result_callback()
def _handle_exceptions(result: Any, **kwargs: Any) -> None:
    if isinstance(result, int):
        raise SystemExit(result)


@cli.command()
@click.option("-d", "--dataset", type=click.Path(exists=False, dir_okay=False), help="Path to the dataset CSV file.")
@click.option("--force", is_flag=True, help="Rebuild models even when cached models exist.")
def build(dataset: str | None, force: bool) -> None:
    """Build recommender models from a local dataset."""
    try:
        dataset_path = resolve_dataset_path(dataset)
    except click.ClickException as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Using dataset: {dataset_path}")

    if not force:
        content_model, collab_model, hybrid_model = load_models(dataset_path)
        if hybrid_model is not None:
            click.echo("Loaded persisted models from models/; build skipped.")
            return

    try:
        interaction_df, item_df, content_model, collab_model, hybrid_model = build_pipeline(dataset_path)
        save_models(dataset_path, content_model, collab_model, hybrid_model, item_df, interaction_df)
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f"Model build failed: {exc}") from exc

    click.echo("Build complete.")
    click.echo(dataset_summary(interaction_df, item_df))
    click.echo(f"Models persisted under: {MODEL_DIR}")


@cli.command()
@click.option("-d", "--dataset", type=click.Path(exists=False, dir_okay=False), help="Path to the dataset CSV file.")
@click.option("-k", "--k", type=int, default=10, show_default=True, help="Cutoff for evaluation metrics.")
def evaluate(dataset: str | None, k: int) -> None:
    """Evaluate the recommender on the given dataset."""
    if k <= 0:
        raise click.ClickException("Evaluation cutoff k must be a positive integer.")

    try:
        dataset_path = resolve_dataset_path(dataset)
    except click.ClickException as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Evaluating dataset: {dataset_path}")

    try:
        results = run_evaluation(k=k, data_path=str(dataset_path))
    except Exception as exc:
        raise click.ClickException(f"Evaluation failed: {exc}") from exc

    if not results:
        raise click.ClickException("Evaluation could not be executed. Check the dataset or dataset format.")

    for mode, metrics in results.items():
        click.echo(f"{mode.capitalize():<15} | " + ", ".join(f"{name}={value:.4f}" for name, value in metrics.items()))


@cli.command()
@click.option("-t", "--title", required=True, help="Item title to use as query.")
@click.option("-n", "--top-n", type=int, default=10, show_default=True, help="Number of recommendations to return.")
@click.option("-d", "--dataset", type=click.Path(exists=False, dir_okay=False), help="Path to the dataset CSV file.")
def recommend(title: str, top_n: int, dataset: str | None) -> None:
    """Generate top-N recommendations for a given title."""
    if top_n <= 0:
        raise click.ClickException("Top-N must be a positive integer.")

    try:
        dataset_path = resolve_dataset_path(dataset)
    except click.ClickException as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Using dataset: {dataset_path}")

    try:
        content_model, collab_model, hybrid_model = load_models(dataset_path)
        if hybrid_model is None:
            click.echo("No cached models found. Building models now...")
            interaction_df, item_df, content_model, collab_model, hybrid_model = build_pipeline(dataset_path)
            save_models(dataset_path, content_model, collab_model, hybrid_model, item_df, interaction_df)

        recs = get_recommendations(hybrid_model, title, top_n)
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f"Recommendation failed: {exc}") from exc

    if not recs:
        raise click.ClickException(f"No recommendations produced for title: {title}")

    click.echo(f"Top {top_n} recommendations for '{title}':")
    for index, rec in enumerate(recs, start=1):
        title_text = rec.get("title", "<unknown>")
        score = rec.get("hybrid_score") or rec.get("content_score") or rec.get("collab_score") or 0.0
        click.echo(f"{index}. {title_text} (score={float(score):.4f})")


@cli.command(name="run-demo")
def run_demo() -> None:
    """Run a lightweight example pipeline using the repository's default dataset."""
    try:
        dataset_path = resolve_dataset_path(None)
    except click.ClickException as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Running demo with dataset: {dataset_path}")

    try:
        interaction_df, item_df, content_model, collab_model, hybrid_model = build_pipeline(dataset_path)
        save_models(dataset_path, content_model, collab_model, hybrid_model, item_df, interaction_df)
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f"Demo pipeline failed: {exc}") from exc

    if item_df.empty:
        raise click.ClickException("Dataset is empty; cannot run demo.")

    example_title = str(item_df.iloc[0]["title"])
    click.echo(f"Example query title: {example_title}")
    try:
        recs = hybrid_model.recommend(example_title, top_n=5)
    except Exception as exc:
        raise click.ClickException(f"Demo recommendation failed: {exc}") from exc

    click.echo("Sample recommendations:")
    for index, rec in enumerate(recs, start=1):
        click.echo(f"{index}. {rec.get('title', '<unknown>')} (score={float(rec.get('hybrid_score', 0.0)):.4f})")

    click.echo("\nDemo evaluation:")
    try:
        results = run_evaluation(k=10, data_path=str(dataset_path))
        if results:
            for mode, metrics in results.items():
                click.echo(f"{mode.capitalize():<15} | " + ", ".join(f"{name}={value:.4f}" for name, value in metrics.items()))
    except Exception as exc:
        raise click.ClickException(f"Demo evaluation failed: {exc}") from exc


if __name__ == "__main__":
    cli(prog_name="python -m src.cli")

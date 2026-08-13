# Tiny example dataset

This directory includes a compact synthetic dataset for local smoke tests and contributor onboarding. The file `example_small.csv` is intentionally tiny and deterministic so it can be used in CI, local validation, and quick recommender demos without requiring network access or external services.

## Purpose

`example_small.csv` is meant to exercise the repository's real recommendation pipeline with a minimal but realistic interaction dataset:

- multiple users
- multiple categories
- repeated user-item interactions across items
- enough content overlap for content-based recommendations to be meaningful
- enough overlap for collaborative filtering to produce stable results

This dataset is not a production dataset. It is designed to be small enough for fast execution while still exercising the actual recommender stack.

## Required columns

The dataset includes the standard columns used by the repository's adapter and models:

- `item_id`
- `title`
- `description`
- `category`
- `user_id`
- `rating`
- `views`
- `purchases`

The adapter will normalize these fields and fill missing defaults as needed.

## How to run the example

From the repository root:

```bash
python scripts/run_pipeline_example.py
```

This script loads the CSV using a path derived from the repo root, runs the repository's existing preprocessing and adapter flow, builds the content, collaborative, and hybrid recommenders, and prints a top-5 recommendation list for a deterministic sample item.

## Expected behavior

The script should finish successfully without any external credentials, network calls, or Supabase configuration. It should print the selected source item and a ranked list of the top recommendations. Results are expected to be stable across repeated runs because the input data and recommender setup are deterministic.

## Why this matters for CI and onboarding

This fixture is useful because:

- it is fast enough for local smoke tests and CI runners
- it exercises the real recommender pipeline without extra setup
- it demonstrates the repository's actual APIs without requiring production services
- it helps contributors verify that data ingestion, preprocessing, and recommendation generation still work correctly after changes

## Offline-safe notes

This dataset and the example runner require no external service, API key, LLM provider, or database connection.

def _build_test_data(
    data_path: str | None = None,
    random_seed: int = 42,
):
    """Build minimal models and test pairs for benchmark scripts.

    Uses a fixed ``random_seed`` so that repeated calls with the same dataset
    produce the same test pairs, making benchmark comparisons stable.

    Args:
        data_path (str | None, optional): File path to source dataset CSV. 
            Defaults to None (which falls back to the DATA_PATH environment variable 
            or "data/products.csv").
        random_seed (int, optional): Seed value used to maintain reproducible 
            sampling of test pairs. Defaults to 42.

    Returns:
        tuple: A 4-element tuple containing:
            - content_model (ContentRecommender or None): Initialized content filtering model.
            - collab_model (_Collab or None): Dummy structural SVD model wrapper.
            - df (pd.DataFrame or None): Cleaned and prepared Pandas DataFrame.
            - test_pairs (list): Collection of evaluation pairs for benchmarking metrics.
    """
    from src.model.content_model import ContentRecommender

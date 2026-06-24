"""
backend/nlp_engine.py — NLP processing utilities for the Hybrid Recommender.

Issue #1595: Added VADER custom lexicon support for slang and informal language.
The lexicon is loaded from config/vader_lexicon_custom.json at runtime and
version-controlled so it can be reviewed and updated via PRs.
"""

import json
import logging
import os
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

logger = logging.getLogger(__name__)

# Path to the version-controlled custom VADER lexicon (Issue #1595)
_LEXICON_PATH = Path(__file__).parent.parent / "config" / "vader_lexicon_custom.json"


def load_vader_with_custom_lexicon():
    """
    Issue #1595 — Load VADER SentimentIntensityAnalyzer with a custom slang lexicon.

    Reads polarity scores from config/vader_lexicon_custom.json and applies them
    to the VADER analyser via make_custom_heuristics().  If the lexicon file is
    missing or malformed, falls back to the standard VADER lexicon with a warning.

    Returns:
        vaderSentiment.SentimentIntensityAnalyzer: Configured analyser instance.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        logger.warning(
            "vaderSentiment not installed; custom lexicon loading skipped. "
            "Install it with: pip install vaderSentiment"
        )
        return None

    analyzer = SentimentIntensityAnalyzer()

    if not _LEXICON_PATH.exists():
        logger.warning(
            "Custom VADER lexicon not found at %s. "
            "Using default VADER lexicon.",
            _LEXICON_PATH,
        )
        return analyzer

    try:
        with open(_LEXICON_PATH, "r", encoding="utf-8") as f:
            custom_entries: dict = json.load(f)

        if not isinstance(custom_entries, dict):
            raise ValueError("Lexicon JSON must be a flat object {word: polarity_score}.")

        # VADER's lexicon is a plain dict attribute; we update it in-place.
        analyzer.lexicon.update(
            {word: float(score) for word, score in custom_entries.items()}
        )
        logger.info(
            "Loaded %d custom VADER lexicon entries from %s",
            len(custom_entries),
            _LEXICON_PATH,
        )
    except Exception as exc:
        logger.error(
            "Failed to load custom VADER lexicon from %s: %s. "
            "Falling back to default VADER lexicon.",
            _LEXICON_PATH,
            exc,
        )

    return analyzer


class NLPEngine:
    """
    Processes textual metadata to compute item-to-item content similarities.

    Attributes:
        vectorizer (TfidfVectorizer): The TF-IDF model instantiation used for text parsing.
    """

    def __init__(self):
        """Initializes the NLPEngine with empty text vectorization layers."""
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def compute_tfidf_matrix(self, metadata_list: list):
        """
        Transforms a raw collection of text metadata into an alternate numerical TF-IDF matrix.

        Args:
            metadata_list (list): A list of strings representing item details or descriptions.

        Returns:
            scipy.sparse._csr.csr_matrix: A sparse matrix containing TF-IDF weights.
        """
        # Original function logic goes here
        pass

    def calculate_similarity(self, tfidf_matrix) -> float:
        """
        Calculates the pairwise cosine similarity score between item matrices.

        Args:
            tfidf_matrix (csr_matrix): A sparse matrix tracking word importance values.

        Returns:
            ndarray: A square similarity matrix representing pairwise text matching profiles.
        """
        # Original function logic goes here
        pass
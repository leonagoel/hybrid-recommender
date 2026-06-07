"""
Streamlit UI for the Hybrid Recommender System.

Reuses existing model classes directly — no backend server required.

Run with:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# ── Dynamic Path Mapping Fix (#490) ──────────────────────────────────────────
CURRENT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CURRENT_DIR.parent.parent  # Steps out of src/api to project root

# Ensure internal source packages can be imported without directory errors
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_adapter import adapt_data, read_file
from src.model.content_model import ContentRecommender
from src.model.collaborative_model import CollaborativeRecommender
from src.model.hybrid_model import HybridRecommender
from src.model.causal_config import CausalConfig
from src.model.llm_explainer import get_explainer


# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid Recommender",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Hybrid Recommender System")
st.caption("Content-Based · Collaborative · Sentiment — all in one engine")

# ── Session state initialisation ─────────────────────────────────────────────
for key in ("content_model", "collab_model", "hybrid_model", "adapted_df", "meta", "uploaded_file_name", "explainer"):
    if key not in st.session_state:
        st.session_state[key] = None

# Initialize LLM explainer once
if st.session_state.explainer is None:
    st.session_state.explainer = get_explainer()

# ── Sidebar — settings ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    top_n = st.slider(
        "Top-N Recommendations",
        min_value=5, max_value=20, value=10, step=1,
    )
    diversity = st.slider(
        "🌈 Recommendation Diversity",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Increase recommendation variety by reducing similar recommendations."
    )

    serendipity = st.slider(
        "✨ Recommendation Serendipity",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Discover more unexpected recommendations."
    )

    enable_llm_explanations = st.checkbox(
        "🤖 Enable LLM Explanations",
        value=True,
        help="Generate AI-powered explanations for recommendations"
    )

    # ── Causal Inference settings ─────────────────────────────────────────
    st.subheader("🔬 Causal Debiasing")
    st.caption(
        "IPS reweighting reduces popularity and category bias. "
        "Higher λ = stronger correction."
    )

    # Master toggle — stored in session state so Build Models can read it
    enable_causal = st.toggle(
        "Enable Causal Debiasing",
        value=False,
        help=(
            "When ON, items that were over-exposed in training data "
            "(popular / dominant-category) are downweighted so niche "
            "but genuinely relevant items surface higher."
        ),
    )

    # λ slider — only meaningful when causal is on, but always rendered
    # so the value is preserved when the user toggles back on
    causal_lambda = st.slider(
        "λ — Correction Strength",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        disabled=not enable_causal,
        help="0 = no correction (pure correlation). 1 = full IPS reweighting.",
    )

    causal_clip = st.slider(
        "IPS Clip Max",
        min_value=1.0, max_value=10.0, value=5.0, step=0.5,
        disabled=not enable_causal,
        help="Maximum IPS weight cap. Prevents rare items from dominating.",
    )

    st.subheader("⚖️ Hybrid Weights")
    st.caption("Weights are auto-normalised to sum to 1 by the model.")

    alpha = st.slider("α — Content-Based",  min_value=0.0, max_value=1.0, value=0.40, step=0.05)
    beta  = st.slider("β — Collaborative",  min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    gamma = st.slider("γ — Sentiment",      min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    
    # Live Normalized Weight Preview
    weights = {
        "Content-Based": alpha,
        "Collaborative": beta,
        "Sentiment": gamma,
    }

    total_weight = sum(weights.values())

    st.markdown("### Live Normalized Weight Preview")

    if total_weight <= 0:
        st.warning("All weights are set to zero. Please increase at least one weight to see the normalized distribution.")
    else:
        normalized_weights = {
            name: value / total_weight
            for name, value in weights.items()
        }

        for name, value in normalized_weights.items():
            st.write(f"**{name}:** {value:.2f}")
            st.progress(value)

        st.success(
            f"Total Normalized Weight: {sum(normalized_weights.values()):.2f}"
        )

    total_weight = alpha + beta + gamma
    if total_weight == 0:
        effective_weights = {
            "α Content": 1 / 3,
            "β Collaborative": 1 / 3,
            "γ Sentiment": 1 / 3,
        }
        st.warning("All raw weights are zero, so the model will use equal weights.")
    else:
        effective_weights = {
            "α Content": alpha / total_weight,
            "β Collaborative": beta / total_weight,
            "γ Sentiment": gamma / total_weight,
        }

    st.caption("Effective weights used by the model")
    weight_cols = st.columns(3)
    for col, (label, value) in zip(weight_cols, effective_weights.items()):
        col.metric(label, f"{value:.3f}")
        col.progress(value)

    if st.button("Apply Weights", key="apply_weights_btn"):
        if st.session_state.hybrid_model is not None:
            st.session_state.hybrid_model.set_weights(alpha, beta, gamma)
            st.success("Weights updated!")
        else:
            st.warning("Build the models first.")


# ── Step 1: Upload dataset ───────────────────────────────────────────────────
st.header("1️⃣  Upload Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    if st.session_state.uploaded_file_name != uploaded_file.name:
        # New file — re-adapt data and reset models
        try:
            raw_df = read_file(uploaded_file, file_format='csv')
            adapted_df, meta = adapt_data(raw_df)

            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.adapted_df    = adapted_df
            st.session_state.meta          = meta
            st.session_state.content_model = None
            st.session_state.collab_model  = None
            st.session_state.hybrid_model  = None

        except Exception as e:
            st.error(f"Failed to read dataset: {e}")

    # Always show status for the currently loaded file (same or new)
    if st.session_state.adapted_df is not None:
        adapted_df = st.session_state.adapted_df
        meta       = st.session_state.meta

        st.success(f"✅ Dataset loaded — {len(adapted_df):,} rows detected.")

        with st.expander("Preview adapted data"):
            st.dataframe(adapted_df.head(10))

        with st.expander("Detected columns"):
            detected = {k: v for k, v in meta.items() if k.endswith("_col") and v is not None}
            st.json(detected)


# ── Step 2: Build models ─────────────────────────────────────────────────────
st.header("2️⃣  Build Models")

if st.session_state.adapted_df is None:
    st.info("Upload a dataset above to enable model building.")
else:
    if st.button("🔨 Build Models"):
        adapted_df = st.session_state.adapted_df
        meta       = st.session_state.meta

        with st.spinner("Building models — this may take a moment for large datasets…"):
            try:
                # Content model (always built)
                content_model = ContentRecommender(adapted_df)

                # Collaborative model — requires more than one unique user
                collab_model = None
                if meta["has_user_data"] and adapted_df["user_id"].nunique() > 1:
                    collab_model = CollaborativeRecommender(adapted_df)

                # Build CausalConfig from sidebar settings
                causal_cfg = (
                    CausalConfig(enabled=True, blend_lambda=causal_lambda, clip_max=causal_clip)
                    if enable_causal
                    else CausalConfig.disabled()
                )

                # Hybrid model — pass causal_config for structured configuration
                hybrid_model = HybridRecommender(
                    content_model, collab_model, adapted_df,
                    causal_config=causal_cfg,
                )
                hybrid_model.set_weights(alpha, beta, gamma)

                st.session_state.content_model = content_model
                st.session_state.collab_model  = collab_model
                st.session_state.hybrid_model  = hybrid_model

                if collab_model is not None:
                    st.success("✅ Content model and Collaborative model trained. Hybrid mode active.")
                else:
                    st.success("✅ Content model trained. Collaborative model skipped (dataset needs more than one unique user).")

                # Show causal diagnostics immediately after build
                if enable_causal and hybrid_model._debiaser is not None:
                    with st.expander("🔬 Causal Debiaser Diagnostics", expanded=False):
                        summary = hybrid_model._debiaser.summary()
                        st.json(summary)

            except Exception as e:
                st.error(f"Model build failed: {e}")


# ── Step 3: Get recommendations ──────────────────────────────────────────────
st.header("3️⃣  Get Recommendations")

if st.session_state.hybrid_model is None:
    st.info("Build models above to enable recommendations.")
else:
    adapted_df   = st.session_state.adapted_df
    hybrid_model = st.session_state.hybrid_model
    collab_model = st.session_state.collab_model

    query = st.text_input(
        "Enter an item name or User ID",
        placeholder="e.g. Item Name or user_id",
    )

    submitted = st.button("🚀 Get Recommendations")

    if submitted:
        if not query.strip():
            st.warning("Please enter an item name or User ID.")
        else:
            query = query.strip()

            # ── Determine input type: User ID or item name ────────────────
            is_user_id = query in adapted_df["user_id"].astype(str).values

            try:
                if is_user_id and collab_model is not None:
                    # ── Collaborative path: personalised for the user ──────
                    badge       = "🤝 COLLABORATIVE"
                    badge_color = "blue"

                    recs_raw = collab_model.predict_for_user(query, top_n=top_n)

                    if not recs_raw:
                        st.warning(
                            f"No collaborative recommendations found for User ID **'{query}'**. "
                            "The user may have no interaction history."
                        )
                        st.stop()

                    # Normalise to a common display format
                    recs = [
                        {
                            "title":         r["title"],
                            "hybrid_score":  round(r["predicted_score"], 4),
                            "content_score": "—",
                            "collab_score":  round(r["predicted_score"], 4),
                            "sentiment_score": "—",
                            "rating":        "—",
                            "category":      "",
                            "description":   "",
                            "top_reviews":   [],
                        }
                        for r in recs_raw
                    ]
                    
                    query_item_for_explanation = f"User {query}"

                else:
                    # ── Item name path: hybrid / content recommendations ───
                    title_series = adapted_df["title"].astype(str)

                    # Exact match first (case-insensitive)
                    exact = title_series[title_series.str.lower() == query.lower()]

                    if exact.empty:
                        # Fall back to partial match
                        fuzzy = title_series[
                            title_series.str.lower().str.contains(query.lower(), na=False)
                        ]
                        if fuzzy.empty:
                            st.warning(
                                f"No item found matching **'{query}'**. "
                                "Try a different name or check the spelling."
                            )
                            st.stop()
                        item_title = fuzzy.iloc[0]
                        st.info(f"Exact match not found. Using closest match: **{item_title}**")
                    else:
                        item_title = exact.iloc[0]
                        recs = hybrid_model.recommend(item_title,top_n=top_n,explain=True,diversity=diversity,serendipity=serendipity,)

                    if collab_model is None:
                            badge       = "📄 CONTENT-BASED"
                            badge_color = "green"
                    else:
                            badge       = "🔀 HYBRID"
                            badge_color = "violet"
                        
                    query_item_for_explanation = item_title

                # ── Generate LLM explanations if enabled ──────────────────
                if enable_llm_explanations and st.session_state.explainer and recs:
                    for rec in recs:
                        try:
                            explanation = st.session_state.explainer.explain_recommendation(
                                recommended_item=rec.get("title", "Unknown"),
                                query_item=query_item_for_explanation,
                                scores={
                                    "hybrid": rec.get("hybrid_score"),
                                    "content": rec.get("content_score"),
                                    "collab": rec.get("collab_score"),
                                    "sentiment": rec.get("sentiment_score"),
                                },
                                description=rec.get("description", ""),
                                top_reviews=rec.get("top_reviews", []),
                                category=rec.get("category", ""),
                            )
                            rec["llm_explanation"] = explanation
                        except Exception as e:
                            rec["llm_explanation"] = f"Error: {str(e)}"
                else:
                    for rec in recs:
                        rec["llm_explanation"] = None

                # ── Render results ────────────────────────────────────────
                if not recs:
                    st.warning(
                        "No recommendations returned. "
                        "Try a different input or rebuild the models."
                    )
                else:
                    st.markdown(f"### Results &nbsp; :{badge_color}[{badge}]")
                    st.caption(f"Showing top {len(recs)} recommendations")
                    st.markdown("---")

                    for i, rec in enumerate(recs, start=1):
                        title    = rec.get("title", "Unknown")
                        category = rec.get("category", "")

                        # Show causal score column only when debiasing was active
                        causal_active = (
                            st.session_state.hybrid_model is not None
                            and st.session_state.hybrid_model.use_causal_debiasing
                        )

                        if causal_active:
                            col_rank, col_title, col_hybrid, col_content, col_collab, col_causal, col_rating = st.columns(
                                [0.4, 2.2, 0.9, 0.9, 0.9, 0.9, 0.9]
                            )
                        else:
                            col_rank, col_title, col_hybrid, col_content, col_collab, col_rating = st.columns(
                                [0.4, 2.5, 1.0, 1.0, 1.0, 1.0]
                            )

                        col_rank.markdown(f"**#{i}**")

                        title_label = f"**{title}**"
                        if category:
                            title_label += f"  \n`{category}`"
                        col_title.markdown(title_label)

                        col_hybrid.metric("Hybrid",  rec.get("hybrid_score",   "—"))
                        col_content.metric("Content", rec.get("content_score",  "—"))
                        col_collab.metric("Collab",  rec.get("collab_score",   "—"))
                        if causal_active:
                            # Show original (pre-debiasing) score as delta so the
                            # user can see how much the causal layer changed the rank
                            original = rec.get("original_score")
                            causal   = rec.get("causal_score", rec.get("hybrid_score"))
                            delta    = round(causal - original, 4) if original is not None else None
                            col_causal.metric(
                                "🔬 Causal",
                                causal,
                                delta=delta,
                                delta_color="normal",
                                help="Debiased score. Delta = causal − original.",
                            )
                        col_rating.metric("Rating",  rec.get("rating",         "—"))

                        # Display LLM explanation in a new row
                        explanation = rec.get("llm_explanation")
                        if explanation and explanation != "None":
                            st.write(f"**💡 Why this match:** {explanation}")
                        else:
                            st.write("*Explanation not available*")
                        # Explainable AI Dashboard
                        exp = rec.get("explanation")
                        if exp and exp.get("human_explanation"):
                            st.info(exp["human_explanation"])

                        if exp:
                            with st.expander("📊 Explainable AI Dashboard", expanded=False):

                                st.subheader("Model Weights")
                                st.json(exp.get("active_weights", {}))

                                st.subheader("Component Scores")
                                st.json(exp.get("component_scores", {}))

                                st.subheader("Weighted Contributions")
                                st.json(exp.get("weighted_components", {}))

                                st.subheader("Signals")
                                st.json(exp.get("signals", {}))

                                terms = exp.get("top_content_terms", [])
                                if terms:
                                    st.subheader("Top Content Terms")
                                    st.write(", ".join(map(str, terms)))

                        st.divider()

            except Exception as e:
                st.error(f"Recommendation failed: {e}")
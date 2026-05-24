"""
Streamlit UI for the Hybrid Recommender System.

Reuses existing model classes directly — no backend server required.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

from data_adapter import adapt_data, read_file
from content_model import ContentRecommender
from collaborative_model import CollaborativeRecommender
from hybrid_model import HybridRecommender
from llm_explainer import get_explainer


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

    
    enable_llm_explanations = st.checkbox(
        "🤖 Enable LLM Explanations",
        value=True,
        help="Generate AI-powered explanations for recommendations"
    )

    st.subheader("⚖️ Hybrid Weights")
    st.caption("Weights are auto-normalised to sum to 1 by the model.")

    alpha = st.slider("α — Content-Based",  min_value=0.0, max_value=1.0, value=0.40, step=0.05)
    beta  = st.slider("β — Collaborative",  min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    gamma = st.slider("γ — Sentiment",      min_value=0.0, max_value=1.0, value=0.25, step=0.05)

    if st.button("Apply Weights", width='stretch'):
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
            st.dataframe(adapted_df.head(10), width='stretch')

        with st.expander("Detected columns"):
            detected = {k: v for k, v in meta.items() if k.endswith("_col") and v is not None}
            st.json(detected)


# ── Step 2: Build models ─────────────────────────────────────────────────────
st.header("2️⃣  Build Models")

if st.session_state.adapted_df is None:
    st.info("Upload a dataset above to enable model building.")
else:
    if st.button("🔨 Build Models", width='stretch'):
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

                # Hybrid model
                hybrid_model = HybridRecommender(content_model, collab_model, adapted_df)
                hybrid_model.set_weights(alpha, beta, gamma)

                st.session_state.content_model = content_model
                st.session_state.collab_model  = collab_model
                st.session_state.hybrid_model  = hybrid_model

                if collab_model is not None:
                    st.success("✅ Content model and Collaborative model trained. Hybrid mode active.")
                else:
                    st.success("✅ Content model trained. "
                               "Collaborative model skipped (dataset needs more than one unique user).")

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

    submitted = st.button("🚀 Get Recommendations", width='content')

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

                    recs = hybrid_model.recommend(item_title, top_n=top_n)

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
                        col_rating.metric("Rating",  rec.get("rating",         "—"))

                        # Display LLM explanation in a new row
                        explanation = rec.get("llm_explanation")
                        if explanation and explanation != "None":
                            st.write(f"**💡 Why this match:** {explanation}")
                        else:
                            st.write("*Explanation not available*")

                        st.divider()

            except Exception as e:
                st.error(f"Recommendation failed: {e}")
import streamlit as st
import pandas as pd
from data_adapter import adapt_data
from content_model import ContentRecommender
from collaborative_model import CollaborativeRecommender
from hybrid_model import HybridRecommender

st.title("📊 Hybrid Recommender System")

uploaded_file = st.file_uploader("Upload your dataset (CSV)")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### Raw Data")
    st.dataframe(df.head())

    # Apply Data Adapter
    adapted_df, meta = adapt_data(df)

    st.write("### Detected Columns")
    st.json(meta)

    st.write("### Adapted Data")
    st.dataframe(adapted_df.head())

    st.markdown("---")
    st.subheader("⚖️ Hybrid Weights")
    st.caption("Weights are auto-normalised to sum to 1 by the model.")

    # Sorting options
    sort_option = st.selectbox(
        "Sort Products By",
        ["Relevance", "Price: Low to High", "Price: High to Low", "Rating"]
    )

    # Apply sorting
    sorted_df = adapted_df.copy()

    if sort_option == "Price: Low to High" and "price" in sorted_df.columns:
        sorted_df = sorted_df.sort_values(by="price", ascending=True)

    elif sort_option == "Price: High to Low" and "price" in sorted_df.columns:
        sorted_df = sorted_df.sort_values(by="price", ascending=False)

    elif sort_option == "Rating" and "rating" in sorted_df.columns:
        sorted_df = sorted_df.sort_values(by="rating", ascending=False)

    st.write("### Product Listings")
    st.dataframe(sorted_df.head(20))

    # Hybrid weights
    alpha = st.slider(
        "α — Content-Based",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.05
    )

    beta = st.slider(
        "β — Collaborative",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05
    )

    gamma = st.slider(
        "γ — Sentiment",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05
    )

    # Build models
    content_model = ContentRecommender(adapted_df)

    collab_model = None
    if meta.get('has_user_data'):
        collab_model = CollaborativeRecommender(adapted_df)

    hybrid_model = HybridRecommender(
        content_model,
        collab_model,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )

    item_list = adapted_df['title'].dropna().unique()

    # Limit selectbox options to prevent browser OOM with huge datasets
    if len(item_list) > 100:
        item_list = item_list[:100]

    selected_item = st.selectbox("Select Item", item_list)

    if st.button("Recommend"):
        recs = hybrid_model.recommend(selected_item)

        st.write("### Recommendations")

        if isinstance(recs, pd.DataFrame):
            st.dataframe(recs)
        else:
            st.write(recs)
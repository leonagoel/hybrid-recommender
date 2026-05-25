"""
Streamlit UI for the Hybrid Recommender System.

Follows the layered architecture: Streamlit → FastAPI Backend → Supabase Database.
All models and data processing are delegated to the backend API.

Run the backend first:
    cd backend && python main.py

Then run Streamlit:
    streamlit run app.py
"""

import streamlit as st
import requests
import os
from typing import Optional, Dict, Any, List

# ── Backend Configuration ────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def api_call(method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
    """Make HTTP request to backend API."""
    url = f"{BACKEND_URL}{endpoint}"
    try:
        if method.upper() == "GET":
            response = requests.get(url, **kwargs)
        elif method.upper() == "POST":
            response = requests.post(url, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to backend at {BACKEND_URL}. Ensure it's running: `cd backend && python main.py`")
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Backend error: {e.response.json().get('detail', str(e))}")
        st.stop()


# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid Recommender",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Hybrid Recommender System")
st.caption("Content-Based · Collaborative · Sentiment — all in one engine")

# ── Session state initialisation ─────────────────────────────────────────────
if "backend_status" not in st.session_state:
    st.session_state.backend_status = None
if "weights" not in st.session_state:
    st.session_state.weights = None


def get_backend_status():
    """Check backend status and cache it."""
    try:
        return api_call("GET", "/api/status")
    except:
        return {"ready": False, "message": "Backend unavailable"}


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
    
    # Live Normalized Weight Preview
    weights_dict = {
        "Content-Based": alpha,
        "Collaborative": beta,
        "Sentiment": gamma,
    }

    total_weight = sum(weights_dict.values())

    st.markdown("### Live Normalized Weight Preview")

    if total_weight <= 0:
        st.warning("All weights are set to zero. Please increase at least one weight.")
    else:
        normalized_weights = {
            name: value / total_weight
            for name, value in weights_dict.items()
        }

        for name, value in normalized_weights.items():
            st.write(f"**{name}:** {value:.2f}")
            st.progress(value)

        st.success(
            f"Total Normalized Weight: {sum(normalized_weights.values()):.2f}"
        )

        if st.button("📤 Apply Weights", width='stretch'):
            try:
                api_call("POST", "/api/weights", json={
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma
                })
                st.success("✅ Weights updated!")
            except Exception as e:
                st.error(f"Failed to update weights: {e}")

    st.divider()
    
    # Backend status indicator
    def get_backend_status():
        """Check backend status safely."""
        try:
            response = api_call("GET", "/api/status")

            if not isinstance(response, dict):
                return {
                    "ready": False,
                    "message": "Invalid backend response"
                }

            return response

        except Exception as e:
            return {
            "ready": False,
            "message": str(e)
            }




# ── Step 1: Upload dataset ───────────────────────────────────────────────────
st.header("1️⃣  Upload Dataset")
st.caption("Upload a CSV file to train the recommendation models on the backend.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Upload to Backend", key="upload_btn", width='stretch'):
            with st.spinner(f"Uploading {uploaded_file.name}…"):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
                    result = api_call("POST", "/api/upload", files=files)
                    st.success(f"✅ Uploaded {result.get('rows_processed', 0)} rows successfully!")
                except Exception as e:
                    st.error(f"Upload failed: {e}")
    
    with col2:
        st.write("")  # Spacer
        st.text(f"File: {uploaded_file.name}")


# ── Step 2: Build models ─────────────────────────────────────────────────────
st.header("2️⃣  Build Models")
st.caption("Train the hybrid recommendation models on the backend.")

if st.button("🔨 Build Models", width='stretch', key="build_models_btn"):
    with st.spinner("Building models on backend — this may take a moment…"):
        try:
            result = api_call("POST", "/api/build")
            st.session_state.backend_status = get_backend_status()
            
            if result.get("status") == "success":
                st.success(f"✅ Models built in {result.get('build_time', 'N/A')}s")
                st.info(f"📊 Dataset size: {result.get('item_count', 'N/A')} items")
            else:
                st.error(f"Build failed: {result.get('message', 'Unknown error')}")
        except Exception as e:
            st.error(f"Build request failed: {e}")




# ── Step 3: Get recommendations ──────────────────────────────────────────────
st.header("3️⃣  Get Recommendations")

status = get_backend_status()

if not status.get("ready"):
    st.info("⏳ Backend models not yet built. Complete Steps 1-2 above.")
else:
    query = st.text_input(
        "Enter an item name",
        placeholder="e.g., Product Name or Harry Potter",
    )

    col1, col2 = st.columns([3, 1])
    
    with col1:
        submitted = st.button("🚀 Get Recommendations", width='stretch', key="recommend_btn")
    
    with col2:
        st.write("")  # Spacer

    if submitted:
        if not query.strip():
            st.warning("Please enter an item name.")
        else:
            query = query.strip()
            
            with st.spinner(f"Getting recommendations for '{query}'…"):
                try:
                    # Call backend recommendation endpoint
                    params = {
                        "title": query,
                        "top_n": top_n,
                        "explain": True,
                        "llm_explain": enable_llm_explanations,
                    }
                    payload = api_call("GET", "/api/recommend", params=params)
                    
                    recs = payload.get("recommendations", [])
                    query_item = payload.get("query_item", query)
                    weights = payload.get("weights", {})

                    if not recs:
                        st.warning(
                            f"No recommendations found for **'{query}'**. "
                            "Try a different item name."
                        )
                    else:
                        # Determine recommendation type
                        rec_type = payload.get("type", "hybrid").upper()
                        badge_colors = {
                            "CONTENT-BASED": "green",
                            "COLLABORATIVE": "blue",
                            "HYBRID": "violet",
                        }
                        badge_color = badge_colors.get(rec_type, "gray")
                        
                        # Display results
                        st.markdown(f"### Results &nbsp; :{badge_color}[{rec_type}]")
                        st.caption(f"Showing top {len(recs)} recommendations for '{query_item}'")
                        
                        # Display weights
                        col1, col2, col3 = st.columns(3)
                        col1.metric("α Content", f"{weights.get('alpha', 0):.2%}")
                        col2.metric("β Collaborative", f"{weights.get('beta', 0):.2%}")
                        col3.metric("γ Sentiment", f"{weights.get('gamma', 0):.2%}")
                        
                        st.markdown("---")

                        # Display each recommendation
                        for i, rec in enumerate(recs, start=1):
                            title = rec.get("title", "Unknown")
                            
                            col_rank, col_title, col_score = st.columns([0.3, 2.5, 1.2])

                            col_rank.markdown(f"**#{i}**")
                            col_title.markdown(f"**{title}**")
                            col_score.metric(
                                "Score",
                                f"{rec.get('hybrid_score', 0):.3f}",
                            )

                            # Display detailed scores if available
                            scores_cols = st.columns(3)
                            scores_cols[0].write(f"📄 Content: {rec.get('content_score', '—')}")
                            scores_cols[1].write(f"🤝 Collab: {rec.get('collab_score', '—')}")
                            scores_cols[2].write(f"💭 Sentiment: {rec.get('sentiment_score', '—')}")

                            # Display LLM explanation if available
                            explanation = rec.get("llm_explanation")
                            if explanation and explanation != "None":
                                st.info(f"💡 **Why this match:** {explanation}")

                            st.divider()

                except Exception as e:
                    st.error(f"Recommendation failed: {str(e)}")



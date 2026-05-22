# Testing the LLM Explanation Feature

This guide provides step-by-step instructions to test the newly added LLM explanation feature for the Hybrid Recommender System.

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure your .env file has the Google API key
echo "GOOGLE_API_KEY=your-key-here" >> .env
```

## Quick Start Testing

### 1. Run Unit Tests (Recommended First)

Test the LLM explainer module without needing a full app:

```bash
# Run only LLM tests
pytest tests/test_llm_explainer.py -v

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/test_llm_explainer.py -v --cov=llm_explainer
```

**Expected Output:**
```
tests/test_llm_explainer.py::TestLLMExplainerInit::test_explainer_initialization PASSED
tests/test_llm_explainer.py::TestFallbackExplanations::test_fallback_content_explanation PASSED
tests/test_llm_explainer.py::TestExplainMultiple::test_explain_multiple_recommendations PASSED
... (more tests)
================================ 16 passed in 2.34s ================================
```

### 2. Test Pipeline with LLM (Full Integration)

```bash
python test_pipeline.py
```

This will:
- Load the dataset ✓
- Run NLP sentiment analysis ✓
- Build all models (content, collaborative, hybrid) ✓
- Get recommendations ✓
- **NEW**: Generate LLM explanations for recommendations ✓

### 3. Interactive Streamlit Testing (Visual)

```bash
# Activate virtual environment (if needed)
source .venv/bin/activate

# Run the Streamlit app
streamlit run app.py
```

Then in the UI:
1. ✅ Upload a dataset (e.g., `datasets/sample_products.csv`)
2. ✅ Click **"Build Models"**
3. ✅ Toggle **"🤖 Enable LLM Explanations"** in sidebar
4. ✅ Enter a product name and click **"Get Recommendations"**
5. ✅ Scroll down to see explanations like:
   ```
   💡 Why this match: This item shares similar content features 
   and characteristics with your query...
   ```

### 4. FastAPI Backend Testing

```bash
# Start the backend server
cd backend
python -m uvicorn main:app --reload --port 8000
```

Test the LLM endpoint:

```bash
# With LLM explanations
curl "http://localhost:8000/api/recommend/Harry%20Potter?top_n=5&llm_explain=true"

# Without LLM explanations
curl "http://localhost:8000/api/recommend/Harry%20Potter?top_n=5"
```

Expected JSON response:
```json
{
  "query_item": "Harry Potter",
  "recommendations": [
    {
      "title": "Harry Potter Collection",
      "hybrid_score": 0.92,
      "llm_explanation": "This item shares similar content features with your query..."
    }
  ],
  "llm_explain": true
}
```

---

## Testing Checklist

### Unit Testing
- [ ] All LLM explainer tests pass
- [ ] Fallback explanations generate correctly
- [ ] Batch explanations work for multiple items
- [ ] Singleton pattern works (`get_explainer()`)

### Integration Testing
- [ ] Pipeline test completes successfully
- [ ] LLM explanations are added to recommendations
- [ ] Fallback kicks in when LLM is unavailable
- [ ] No runtime errors with empty datasets

### UI Testing (Streamlit)
- [ ] Toggle checkbox works
- [ ] Explanations appear below ratings
- [ ] Explanations are complete (not truncated)
- [ ] Works with both hybrid and collaborative recommendations
- [ ] Error messages display if API fails

### API Testing (FastAPI)
- [ ] `/api/recommend/{item}?llm_explain=true` returns explanations
- [ ] `/api/recommend/{item}?llm_explain=false` skips explanations
- [ ] Explanations in JSON response are valid
- [ ] Graceful error handling when LLM unavailable

---

## Test Scenarios

### Scenario 1: Full Happy Path
```bash
# 1. Run unit tests
pytest tests/test_llm_explainer.py -v

# 2. Run pipeline
python test_pipeline.py

# 3. Start Streamlit
streamlit run app.py

# 4. Upload dataset → Build Models → Enable LLM → Get Recommendations
```

### Scenario 2: LLM Unavailable
```bash
# Test with invalid API key
GOOGLE_API_KEY="invalid-key" streamlit run app.py

# Should show fallback explanations instead
# Expected: Explanations still appear with fallback text
```

### Scenario 3: Edge Cases
```python
# Test with empty recommendations
# Test with missing descriptions
# Test with special characters in item names
# Test with very long descriptions
```

---

## Debugging Tips

### If explanations don't appear:

1. **Check API Key:**
   ```bash
   echo $GOOGLE_API_KEY
   # Should show your key (or empty for fallback)
   ```

2. **Check logs:**
   ```bash
   # Streamlit logs
   streamlit run app.py 2>&1 | grep -i "explanation"
   
   # Python logs
   python -c "from llm_explainer import get_explainer; e = get_explainer(); print(e.client)"
   ```

3. **Test explainer directly:**
   ```python
   from llm_explainer import get_explainer
   explainer = get_explainer()
   explanation = explainer.explain_recommendation(
       recommended_item="Test",
       query_item="Query",
       scores={"hybrid": 0.85}
   )
   print(explanation)  # Should print explanation or None
   ```

### Common Issues:

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: google-generativeai` | Run `pip install google-generativeai` |
| Explanations say "None" | Check `GOOGLE_API_KEY` or API quota |
| Streamlit slow | Reduce `top_n` recommendations or disable LLM temporarily |
| API timeouts | Increase timeout or use fallback explanations |

---

## Performance Testing

```bash
# Time the recommendation + explanation generation
python -c "
import time
from data_adapter import read_file, adapt_data
from content_model import ContentRecommender
from hybrid_model import HybridRecommender
from llm_explainer import get_explainer

# Load data
adapted_df, meta = adapt_data(read_file('datasets/sample_products.csv'))
cm = ContentRecommender(adapted_df)
hm = HybridRecommender(cm, None, adapted_df)
explainer = get_explainer()

# Time recommendation
start = time.time()
recs = hm.recommend(adapted_df['title'].iloc[0], top_n=5)
print(f'Recommendations: {time.time() - start:.2f}s')

# Time explanations
start = time.time()
results = explainer.explain_multiple(recs, adapted_df['title'].iloc[0])
print(f'Explanations: {time.time() - start:.2f}s')
"
```

---

## Continuous Integration (CI)

All tests should pass before merging:

```bash
#!/bin/bash
set -e

echo "Running unit tests..."
pytest tests/ -v

echo "Running pipeline test..."
python test_pipeline.py

echo "Checking code style..."
flake8 llm_explainer.py --max-line-length=120

echo "All checks passed! ✅"
```

---

## Summary

✅ **Unit Tests:** `pytest tests/test_llm_explainer.py -v`
✅ **Integration:** `python test_pipeline.py`
✅ **UI Testing:** `streamlit run app.py`
✅ **API Testing:** `curl http://localhost:8000/api/recommend/...?llm_explain=true`

For detailed test results and debugging, check the logs output by each testing method.

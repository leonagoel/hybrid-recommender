# Quick Testing Reference

## Run All Tests

```bash
# Install dependencies
pip install -r requirements.txt

# Unit tests only
pytest tests/test_llm_explainer.py -v

# Full test suite
pytest tests/ -v

# Pipeline test
python test_pipeline.py
```

## Interactive Testing

```bash
# Start Streamlit app
streamlit run app.py

# Then:
# 1. Upload dataset
# 2. Build models
# 3. Toggle "🤖 Enable LLM Explanations" 
# 4. Search for an item
# 5. See explanations below scores
```

## API Testing

```bash
# Start backend
cd backend && python -m uvicorn main:app --reload

# Test with LLM explanations
curl "http://localhost:8000/api/recommend/Harry%20Potter?llm_explain=true"

# Test without LLM
curl "http://localhost:8000/api/recommend/Harry%20Potter?llm_explain=false"
```

## Python Testing

```python
# Test LLM explainer directly
from llm_explainer import get_explainer

explainer = get_explainer()
explanation = explainer.explain_recommendation(
    recommended_item="Item A",
    query_item="Item B",
    scores={"hybrid": 0.85},
    category="Electronics"
)
print(explanation)
```

## Expected Results

### Unit Tests
- 16 tests should pass
- All test categories should pass:
  - Initialization
  - Fallback explanations
  - LLM explanations
  - Batch processing

### Pipeline Test
```
✅ All pipeline tests passed!
```

### Streamlit UI
- Explanations appear below rating metrics
- Format: "💡 Why this match: [explanation]"
- Both LLM and fallback explanations work

### API Response
```json
{
  "recommendations": [
    {
      "title": "...",
      "llm_explanation": "..."
    }
  ]
}
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Tests fail | Run `pip install -r requirements.txt` |
| No explanations | Check `GOOGLE_API_KEY` in `.env` |
| Slow response | Fallback explanations are faster |
| Import error | Ensure you're in virtual environment |

---

See `TESTING.md` for detailed instructions.

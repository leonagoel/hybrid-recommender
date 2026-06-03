## What changed
- Fixed the legacy synchronous recommendation endpoints so they no longer return `null` responses.
- Cleaned up unreachable/invalid code paths in the similar-items endpoint to avoid undefined-variable crashes.
- Ensured the recommendation payload returns a complete JSON response (including model weights).

## Why
The `GET /api/recommend` and `GET /api/recommend/{item_title}` endpoints were effectively broken and returning HTTP 200 with a `null` body, which prevents the frontend HTTP fallback from rendering recommendations. There were also dead/unreachable lines referencing undefined variables that could cause runtime errors.

## How to test
```bash
pip install -r requirements.txt
python -m py_compile backend/main.py

# Start the backend (use the repo’s usual start command if different)
python backend/main.py
# or
uvicorn backend.main:app --reload

# In another terminal, after building models (if required in your env):
curl "http://localhost:8000/api/recommend/Harry%20Potter?top_n=5"
curl "http://localhost:8000/api/recommend?title=Harry%20Potter&top_n=5"
curl "http://localhost:8000/api/similar/1?top_n=3"
```

## Screenshots (if UI change)
N/A (backend-only change).

## Checklist
- [x] I have read the [CONTRIBUTING.md](CONTRIBUTING.md)
- [x] My code follows PEP8 style (`flake8 .`)
- [x] I have tested my changes locally
- [ ] I have added/updated tests where applicable
- [x] I can explain every line of code I've written
- [x] I have NOT used AI-generated code without understanding and attributing it

## Related issue
Closes #203

## AI assistance disclosure
- [ ] I did not use AI assistance for this PR
- [x] I used AI assistance for: debugging the root cause from the issue report and drafting a minimal fix; all changes were reviewed and understood before committing.

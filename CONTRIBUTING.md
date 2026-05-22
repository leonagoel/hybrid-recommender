# Contributing to Hybrid Recommender System

Welcome! This project is part of **GSSoC 2026 (GirlScript Summer of Code)**. We're glad you're here. This guide will help you make your first contribution smoothly.

---

## Table of Contents

- [Before You Start](#before-you-start)
- [Prerequisites](#Prerequisites)
- [How to Contribute](#how-to-contribute)
- [Branch Naming](#branch-naming)
- [Commit Message Format](#commit-message-format)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Code Style](#code-style)
- [Running the Project Locally](#running-the-project-locally)
- [GSSoC Label Guide](#gssoc-label-guide)
- [Common Mistakes to Avoid](#Common-Mistakes-to-Avoid)

---

## Before You Start

1. **Find an issue** — Go to the [Issues tab](../../issues) and look for ones labelled `gssoc:approved` + `good first issue` if you are a beginner.
2. **Comment to claim** — Comment `"I'd like to work on this"` on the issue and **wait for it to be assigned to you** before starting any code.
3. **Do not submit a PR for an unassigned issue** — it will be closed without review.

---

## Prerequisites
- Python 3.8+
- Git installed
- GitHub account
- Basic knowledge of Python

---

## How to Contribute

```bash
# 1. Fork the repository (click Fork on GitHub)

# 2. Clone your fork
git clone https://github.com/<your-username>/hybrid-recommender.git
cd hybrid-recommender

# 3. Add the original repo as upstream
git remote add upstream https://github.com/leonagoel/hybrid-recommender.git

# 4. Create a new branch (see naming guide below)
git checkout -b feat/add-streamlit-ui

# 5. Install dependencies
pip install -r requirements.txt

# 6. Make your changes and test them locally

# 7. Commit your changes
git add .
git commit -m "feat: add streamlit UI for hybrid recommendations"

# 8. Push to your fork
git push origin feat/add-streamlit-ui

# 9. Open a Pull Request on GitHub against the main branch
```

---

## Branch Naming

Use this format: `type/short-description`

| Type | When to use |
|------|-------------|
| `feat/` | Adding a new feature |
| `fix/` | Fixing a bug |
| `docs/` | Documentation changes only |
| `test/` | Adding or updating tests |
| `refactor/` | Code cleanup with no feature/fix |
| `perf/` | Performance improvements |

Examples:
- `feat/add-streamlit-ui`
- `fix/cold-start-fallback`
- `docs/update-setup-guide`
- `test/unit-tests-content-model`

---

## Commit Message Format

```
type: short description (max 72 chars)
```

Examples:
```
feat: add configurable alpha/beta/gamma weights via CLI
fix: handle empty interaction_df in CollaborativeRecommender
docs: add Jupyter notebook demo for hybrid recommendations
test: add unit tests for bayesian_rating function
refactor: extract weight normalization into helper function
```

- Use **present tense** ("add" not "added")
- Keep it **short and clear**
- Reference the issue number if applicable: `fix: handle cold start (#12)`

---

## Pull Request Guidelines

Every PR description **must include**:

```
## What changed
<!-- Describe what you added/fixed/changed -->

## Why
<!-- Explain the problem this solves or feature this adds -->

## How to test
<!-- Step-by-step instructions to test your changes locally -->

## Screenshots (if UI change)
<!-- Add before/after screenshots or screen recording -->

## Related issue
Closes #<issue-number>
```

**Rules:**
- One issue per PR — don't bundle unrelated changes
- All code must be tested locally before submitting
- If the project has tests, update them to cover your change
- Respond to review comments within **48 hours** or the PR may be closed
- Do not merge your own PR — wait for a mentor/PA to review it

---

## Code Style

This project uses **Python** and follows **PEP8**.

```bash
# Check for PEP8 issues
pip install flake8
flake8 . --max-line-length=100

# Auto-format (optional but recommended)
pip install black
black .
```

Key rules:
- Use `snake_case` for variable and function names
- Use `PascalCase` for class names
- Keep functions small and focused — one responsibility per function
- Add docstrings to all classes and functions
- No hardcoded credentials — use `.env` file (see `.env.example`)

---

## Running the Project Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment variables
cp .env.example .env
# Fill in your Supabase credentials in .env

# 3. Start the backend server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 4. Open http://localhost:8000 in your browser

# 5. Run the smoke test (no Supabase needed for this)
python test_pipeline.py
```

---

## GSSoC Label Guide

For your PR to count on the GSSoC leaderboard, mentors will add these labels after review:

| Label | Meaning |
|-------|---------|
| `gssoc:approved` | ✅ Required — PR counts toward leaderboard |
| `level:beginner` | 3 points |
| `level:intermediate` | 7 points |
| `level:advanced` | 10 points |
| `quality:clean` | Score multiplier for well-written code |
| `quality:exceptional` | Higher multiplier for outstanding contributions |
| `mentor:username` | Mentor who reviewed the PR |

**You do not add these labels yourself** — mentors add them upon review.

---

## Questions?

- Open a [GitHub Discussion](../../discussions) or comment on the issue
- Join the **GSSoC Discord** and find the project channel
- Tag **@leonagoel** if you are blocked for more than 48 hours

---

## Common Mistakes to Avoid
- Don't work on unassigned issues
- Don't submit multiple PRs for same issue  
- Don't forget to sync fork before starting
- Always test locally before PR

Happy contributing! 🚀

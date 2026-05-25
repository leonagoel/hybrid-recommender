# Contributing to Hybrid Recommender 🤝

Welcome! This project is part of **GSSoC 2026** and we're thrilled to have you here.
Please read this guide fully before raising an issue or submitting a PR.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Branch Naming Convention](#branch-naming-convention)
- [Commit Message Format](#commit-message-format)
- [How to Raise an Issue](#how-to-raise-an-issue)
- [PR Submission Checklist](#pr-submission-checklist)
- [GSSoC-Specific Notes](#gssoc-specific-notes)
- [Code Style](#code-style)
- [Need Help?](#need-help)

---

## Getting Started

### 1. Fork & Clone

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/<your-username>/hybrid-recommender.git
cd hybrid-recommender
```

### 2. Add Upstream Remote

```bash
git remote add upstream https://github.com/leonagoel/hybrid-recommender.git
git remote -v  # should show both origin and upstream
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Fill in your Supabase URL and anon key in .env
```

### 5. Run the App

```bash
uvicorn backend.main:app --reload
# Open http://localhost:8000 in your browser
```

---

## Branch Naming Convention

Always create a new branch for each issue. Never commit directly to `main`.

| Type          | Pattern                 | Example                    |
| ------------- | ----------------------- | -------------------------- |
| New feature   | `feat/<short-desc>`     | `feat/add-pagination`      |
| Bug fix       | `fix/<short-desc>`      | `fix/search-crash`         |
| Documentation | `docs/<short-desc>`     | `docs/add-contributing`    |
| Refactor      | `refactor/<short-desc>` | `refactor/split-app-js`    |
| Tests         | `test/<short-desc>`     | `test/recommendation-unit` |
| CI / Tooling  | `ci/<short-desc>`       | `ci/update-flake8`         |

```bash
# Always branch off a fresh main
git checkout main
git pull upstream main
git checkout -b feat/your-feature-name
```

---

## Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

<type>: <short summary in present tense, under 72 chars>
[optional body — explain WHY, not WHAT]
[optional footer — e.g. Resolves #123]

**Types:** `feat` · `fix` · `docs` · `refactor` · `test` · `ci` · `chore`

**Good examples:**
feat: add pagination to product listing API
fix: prevent search crash on empty query
docs: enhance CONTRIBUTING.md with GSSoC guide
refactor: split app.js into focused ES modules
test: add unit tests for hybrid scoring logic

**Rules:**

- Use present tense — "add" not "added"
- No capital letter at the start of the summary
- No full stop at the end
- Always reference the issue: `Resolves #123` in the commit body or PR

---

## How to Raise an Issue

1. **Search first** — check [existing issues](https://github.com/leonagoel/hybrid-recommender/issues) to avoid duplicates
2. Use the correct template:
   - **Bug Report** — for something broken
   - **Feature Request** — for new functionality or improvements
3. Fill in **all sections** of the template fully (Problem Statement, Proposed Solution, Acceptance Criteria)
4. Add the most relevant label: `type:bug` / `type:feature` / `type:docs`
5. For GSSoC: wait for `gssoc:approved` label before starting any work
6. Comment `"I'd like to work on this!"` and **wait to be assigned** before starting
7. PRs submitted without prior assignment may be closed without review

    [ ] I branched off a fresh main (not an old or stale branch)
    [ ] My branch name follows the naming convention above
    [ ] My commit messages follow the Conventional Commits format
    [ ] I linked the issue with Closes #issue-number in the PR body
    [ ] I filled in all PR template sections (What changed / Why / How to test)
    [ ] I tested my changes locally end-to-end
    [ ] No new PEP8 / flake8 errors (run: flake8 backend/ --max-line-length=79)
    [ ] No unrelated files modified
    [ ] No .env, pycache, node_modules, or .DS_Store committed
    [ ] No console.log() left in production JS (use console.warn/error only)

## PR Submission Checklist

Before opening a PR, tick every box:
[ ] I branched off a fresh main (not an old or stale branch)
[ ] My branch name follows the naming convention above
[ ] My commit messages follow the Conventional Commits format
[ ] I linked the issue with Closes #<issue-number> in the PR body
[ ] I filled in all PR template sections (What changed / Why / How to test)
[ ] I tested my changes locally end-to-end
[ ] No new PEP8 / flake8 errors (run: flake8 backend/ --max-line-length=79)
[ ] No unrelated files modified
[ ] No .env, pycache, node_modules, or .DS_Store committed
[ ] No console.log() left in production JS (use console.warn/error only)

**PR title format** (must match Conventional Commits):
feat: add pagination to product listing
fix: resolve search crash on empty input
docs: enhance CONTRIBUTING.md with GSSoC guide
refactor: split frontend app.js into ES modules

---

## GSSoC-Specific Notes

### How the Contribution Flow Works

| Step | What Happens                            | Who               |
| ---- | --------------------------------------- | ----------------- |
| 1    | Issue raised and `gssoc:approved` added | Mentor            |
| 2    | You comment and get **assigned**        | You + Mentor      |
| 3    | You fork, branch, build, and open PR    | You               |
| 4    | CI runs lint + smoke test               | Automated         |
| 5    | Mentor reviews and merges               | @leonagoel        |
| 6    | Points credited at **4 AM IST daily**   | GSSoC leaderboard |

### Labels Explained

| Label                  | Meaning                                   |
| ---------------------- | ----------------------------------------- |
| `gssoc:approved`       | Approved for GSSoC — safe to start        |
| `level:beginner`       | ~10 GSSoC points                          |
| `level:intermediate`   | ~25 GSSoC points                          |
| `level:advanced`       | ~45–50 GSSoC points                       |
| `status:review-needed` | PR submitted, awaiting mentor review      |
| `mentor:leonagoel`     | Assigned to project mentor @leonagoel     |
| `type:refactor`        | Code restructuring without feature change |
| `type:feature`         | New functionality                         |
| `type:bug`             | Bug fix                                   |
| `type:docs`            | Documentation only                        |

### GSSoC Rules

- One contributor per issue — first person assigned gets it
- Respond to review comments within **48 hours** or PR may be reassigned
- Do not submit multiple PRs for the same issue
- Do not copy code from other contributors' open (unmerged) PRs
- Do not open a PR from your `main` branch — always use a feature branch

---

## Code Style

### Python — Backend (`backend/`)

- Strictly follow **PEP8** — CI runs `flake8` and will fail your PR if violated
- Max line length: **79 characters**
- Use **type hints** on all function signatures
- Write **Google-style docstrings** on all public functions

```python
# ✅ Good
def get_recommendations(title: str, k: int = 10) -> list[dict]:
    """Return top-k hybrid recommendations for a product title.

    Args:
        title: Product title to base recommendations on.
        k: Number of recommendations to return.

    Returns:
        List of recommendation dicts containing score, title, category.
    """
    ...

# ❌ Bad — no types, no docstring, line too long
def get_recommendations(title, k=10):
    return recommender.get_hybrid_recommendations_for_title_with_weights(title, k, alpha=0.4, beta=0.4, gamma=0.2)
```

### JavaScript — Frontend (`frontend/`)

- **ES Modules only** (`import`/`export`) — no CommonJS `require()`
- Follow the module structure from `frontend/js/` (see PR #214):
  - `state.js` — global state only
  - `auth.js` — auth logic only
  - `search.js` — search logic only
  - `recommendations.js` — recommendation logic only
  - `ui.js` — DOM helpers only
  - `app.js` — entry point / wiring only
- Always null-check DOM queries: `document.getElementById('x')?.addEventListener(...)`
- No inline styles — use CSS classes
- Escape all user-facing strings before DOM insertion

### General

- No commented-out dead code in PRs
- No `console.log()` left in production code
- No hardcoded secrets, API keys, or credentials — use `.env`

---

## Need Help?

- Open a [GitHub Discussion](https://github.com/leonagoel/hybrid-recommender/discussions) for general questions
- Tag `@leonagoel` in your issue or PR comment if you're blocked for more than 24 hours
- For GSSoC programme queries: [gssoc.girlscript.tech](https://gssoc.girlscript.tech)

---

_Happy Contributing! Every PR — big or small — makes this project better._ 🚀

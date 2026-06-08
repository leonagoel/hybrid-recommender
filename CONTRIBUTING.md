<div align="center">

![Typing SVG](https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=28&duration=3000&pause=800&color=6C63FF&center=true&vCenter=true&multiline=true&width=700&height=80&lines=Contributing+to+hybrid-recommender)

![Typing SVG](https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=16&duration=2500&pause=600&color=A78BFA&center=true&vCenter=true&width=700&lines=collaborative+filtering+%2B+content-based+approaches;built+by+the+community%2C+for+the+community;every+contribution+improves+the+signal+%F0%9F%8E%AF)

</div>

---

We welcome contributions from everyone — whether you're fine-tuning a model, squashing a bug, or improving a docstring. This guide covers everything you need to get started.

---

## Table of Contents

- [Before You Begin](#before-you-begin)
- [How the Project is Structured](#how-the-project-is-structured)
- [Setting Up Locally](#setting-up-locally)
- [Finding Something to Work On](#finding-something-to-work-on)
- [Contribution Limits](#contribution-limits)
- [Making Your Contribution](#making-your-contribution)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Getting Help](#getting-help)

---

## Before You Begin

- Search [existing issues](https://github.com/leonagoel/hybrid-recommender/issues) before opening a new one
- For large changes (new features, architecture changes), open a discussion first
- Read through the [README](README.md) to understand how the recommender works
- Be respectful — we're all here to learn and build

---

## How the Project is Structured

```
hybrid-recommender/
├── backend/           # Python — recommendation logic, APIs
│   ├── models/        # Collaborative + content-based models
│   ├── api/           # REST endpoints
│   └── utils/         # Shared utilities
├── frontend/          # JavaScript — user interface
│   ├── src/
│   └── public/
├── datasets/          # Sample and test datasets
├── scripts/           # Data processing and setup scripts
├── tests/             # Test suites (Python + JS)
├── supabase/          # DB schema and edge functions
└── .github/workflows/ # Automation and CI
```

---

## Setting Up Locally

### Prerequisites
- Python 3.9+
- Node.js 18+
- Git

### Steps

```bash
# 1. Fork the repo, then clone your fork
git clone https://github.com/YOUR-USERNAME/hybrid-recommender.git
cd hybrid-recommender

# 2. Add upstream remote
git remote add upstream https://github.com/leonagoel/hybrid-recommender.git

# 3. Backend setup
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 4. Frontend setup
cd ../frontend
npm install

# 5. Environment variables
cp .env.example .env.local
# Fill in your credentials in .env.local

# 6. Run both
# Terminal 1 — backend
cd ../backend && uvicorn main:app --reload

# Terminal 2 — frontend
cd ../frontend && npm run dev
```

### Pre-commit Hooks

After cloning, install the pre-commit hooks to automatically lint and format your code before each commit:

```bash
pip install pre-commit   # or: pip install -r requirements.txt
pre-commit install
```

The hooks will run Ruff linting/formatting, check for trailing whitespace, fix end-of-files, validate YAML/JSON, and more. You can also run them manually at any time:

```bash
pre-commit run --all-files
```

---

## Finding Something to Work On

| Label | What it means |
|-------|--------------|
| `good first issue` | Self-contained, well-scoped — great starting point |
| `level:beginner` | Minimal context needed, clear expected output |
| `level:intermediate` | Requires understanding of the codebase |
| `level:advanced` | ML model changes, architecture work |
| `ml/ai` | Touches recommendation logic |
| `frontend` | UI/UX work |
| `bug` | Something is broken |
| `documentation` | Docs, comments, guides |

Browse open issues → [github.com/leonagoel/hybrid-recommender/issues](https://github.com/leonagoel/hybrid-recommender/issues)

Leave a comment on an issue to claim it before starting work.

---

## Contribution Limits

To keep the project manageable and fair for everyone:

- **Max 3 open issues** assigned to you at a time
- **Max 3 open PRs** from you at a time
- Close or complete existing work before picking up new items
- If you claimed an issue but can't continue, leave a comment so someone else can take over

---

## Making Your Contribution

```bash
# 1. Sync your fork with upstream
git fetch upstream
git checkout main
git merge upstream/main

# 2. Create a branch
git checkout -b fix/cold-start-bug
# or
git checkout -b feat/content-based-filter
# or
git checkout -b docs/setup-guide

# 3. Make your changes, then commit
git add .
git commit -m "fix: handle cold start for new users"

# 4. Push
git push origin your-branch-name
```

### Commit Message Format

```
<type>: <short description>

Types: feat | fix | docs | refactor | test | chore
```

Examples:
- `feat: add cosine similarity scoring to content model`
- `fix: correct NaN handling in collaborative filter`
- `docs: add dataset preparation guide`
- `test: add unit tests for recommendation pipeline`

---

## Pull Request Guidelines

### PR Description Template

```markdown
## What does this PR do?
<!-- A clear summary of the change -->

## Related issue
Closes #

## Type of change
- [ ] Bug fix
- [ ] New feature
- [ ] Refactor
- [ ] Documentation
- [ ] Tests

## How to test this
<!-- Steps to verify your change works -->

## Screenshots (if UI change)
```

### Before you submit

- [ ] Synced with latest `main`
- [ ] No merge conflicts
- [ ] Tested locally (both backend and frontend if applicable)
- [ ] No hardcoded credentials or API keys
- [ ] Relevant tests added or updated
- [ ] Docstrings/comments added for new functions

### Review process
- At least one maintainer review is required before merging
- Address all review comments before re-requesting review
- Be responsive — PRs inactive for 7 days may be closed

---

## Code Standards

### Python (backend)

- Follow [PEP 8](https://pep8.org/)
- Use type hints for function signatures
- Write docstrings for all public functions

```python
# Good
def compute_similarity(user_id: int, item_id: int) -> float:
    """
    Compute cosine similarity between user and item vectors.

    Args:
        user_id: ID of the user
        item_id: ID of the item

    Returns:
        Similarity score between 0 and 1
    """
    ...

# Avoid
def sim(u, i):
    # does stuff
    ...
```

### JavaScript (frontend)

- Use `const`/`let`, not `var`
- Prefer arrow functions
- Handle errors explicitly — no silent failures

```javascript
// Good
const fetchRecommendations = async (userId) => {
  try {
    const data = await api.get(`/recommend/${userId}`);
    return data;
  } catch (error) {
    console.error('Recommendation fetch failed:', error);
    throw error;
  }
};

// Avoid
function getRec(id) {
  return api.get('/recommend/' + id);
}
```

---

## Testing

### Python tests

```bash
cd backend
pytest tests/
```

### Frontend tests

```bash
cd frontend
npm run test
```

### What to test
- New functions should have at least one unit test
- Bug fixes should include a test that would have caught the bug
- Don't break existing tests — run the full suite before submitting

---

## Getting Help

- 💬 **Discussions** — questions, ideas, design decisions → [Discussion Board](https://github.com/leonagoel/hybrid-recommender/discussions)
- 🐛 **Issues** — bugs and feature requests → [Issues](https://github.com/leonagoel/hybrid-recommender/issues)
- 📖 **Docs** — project overview → [README](README.md)

If you're stuck on something, open a discussion rather than sitting on it — the community is here to help.

---

## Security

Never commit:
- API keys or tokens
- Database connection strings with credentials
- `.env` files or any secrets

If you discover a security vulnerability, please open a **private** issue rather than a public one.

---

*Thank you for contributing. Every improvement — big or small — makes the recommender better for everyone.*

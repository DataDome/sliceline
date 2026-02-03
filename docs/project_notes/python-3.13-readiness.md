# Python 3.13 Readiness Tracking

## Current Status: BLOCKED - Waiting for optbinning release

**Last Updated**: 2026-01-27

## Blocker

### optbinning Python 3.13 Support
- **PR**: https://github.com/guillermo-navas-palencia/optbinning/pull/366
- **Status**: Open (actively being worked on)
- **Last Activity**: 2026-01-26
- **Dependencies**: Waiting for ropwr#16 to be merged first
- **Key Changes**:
  - Python 3.13 compatibility
  - Updated ortools constraint to `>=9.4,<9.17,!=9.12` (excludes buggy 9.12)
  - CI improvements

## Sliceline Python 3.13 Readiness

### Core Package ✅
- No blockers
- All dependencies support Python 3.13

### Dev Dependencies ⏳
- **optbinning**: Waiting for release with Python 3.13 support
- Used only in example notebooks (california_housing.ipynb, titanic.ipynb)

## Action Plan When Ready

### Step 1: Monitor optbinning Release
Check PyPI for new optbinning release with Python 3.13 support:
```bash
# Check latest version
pip index versions optbinning

# Or visit: https://pypi.org/project/optbinning/#history
```

### Step 2: Update sliceline
Once optbinning releases with Python 3.13 support:

1. **Update pyproject.toml** (3 changes):
   ```toml
   # Add Python 3.13 classifier
   classifiers = [
       ...
       "Programming Language :: Python :: 3.13",
   ]

   # Update python version constraint
   [project]
   requires-python = ">=3.10"  # No change needed, already open-ended

   # Update optbinning if needed
   [project.optional-dependencies]
   dev = [
       "optbinning>=0.X.X",  # Update to version with Python 3.13
       ...
   ]
   ```

2. **Update .github/workflows/push-pull.yml**:
   ```yaml
   strategy:
     matrix:
       python-version: ["3.10", "3.11", "3.12", "3.13"]  # Add 3.13
   ```

3. **Update CONTRIBUTING.md**:
   ```markdown
   Sliceline is intended to work with **Python 3.10 or above** (including Python 3.13).
   ```

4. **Test locally**:
   ```bash
   # Create Python 3.13 environment
   uv venv --python 3.13
   uv sync --all-extras
   uv run pytest tests/ -v

   # Test notebooks
   make execute-notebooks
   ```

5. **Create PR**:
   ```bash
   git checkout -b feature/python-3.13-support
   git add pyproject.toml .github/workflows/push-pull.yml CONTRIBUTING.md
   git commit -S -m "feat: add Python 3.13 support

- Add Python 3.13 classifier to pyproject.toml
- Add Python 3.13 to CI test matrix
- Update optbinning to version X.X.X with Python 3.13 support
- Update documentation

Closes #XXX

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
   git push -u origin feature/python-3.13-support
   # Create PR manually or with gh cli
   ```

## Monitoring Strategy

### Weekly Check (until released)
```bash
# Check optbinning PR status
gh pr view 366 --repo guillermo-navas-palencia/optbinning

# Check if merged and released
pip index versions optbinning | grep -A 5 "optbinning"
```

### Subscribe to Notifications
- Watch optbinning releases: https://github.com/guillermo-navas-palencia/optbinning/releases
- Watch PyPI RSS: https://pypi.org/rss/project/optbinning/releases.xml

## Notes

- optbinning is only used in dev dependencies for example notebooks
- Core sliceline package has no Python 3.13 blockers
- The PR excludes ortools 9.12 due to bugs
- Expected optbinning constraint: `ortools>=9.4,<9.17,!=9.12`

## Timeline Estimate

Based on PR activity:
- **Optimistic**: 2-4 weeks (if ropwr#16 merges quickly)
- **Realistic**: 1-2 months
- **Pessimistic**: 3+ months

Will update this document as status changes.

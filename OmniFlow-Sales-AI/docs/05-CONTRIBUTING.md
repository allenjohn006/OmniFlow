# 🤝 Contributing Guide

## Welcome!

Thank you for your interest in contributing to OmniFlow Sales AI. This document provides guidelines for contributing code, documentation, and bug reports.

---

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork locally
git clone https://github.com/YOUR_USERNAME/OmniFlow-Sales-AI.git
cd OmniFlow-Sales-AI

# Add upstream remote
git remote add upstream https://github.com/original-owner/OmniFlow-Sales-AI.git
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-you're-fixing
```

**Branch Naming Conventions**:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `perf/` - Performance improvements
- `test/` - Test additions/improvements

### 3. Set Up Development Environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\Activate.ps1  # Windows

pip install -r requirements.txt
pip install black flake8 pytest  # Development tools

# Run tests to verify setup
pytest
```

---

## Code Standards

### Python Style Guide: PEP 8

All code must follow PEP 8 with these tools:

#### Black (Code Formatter)
```bash
# Format all files
black .

# Format specific file
black src/training.py
```

#### Flake8 (Linter)
```bash
# Check all Python files
flake8

# Check specific directory
flake8 src/
```

#### Common Standards

**Line Length**: 100 characters maximum
**Imports**: Organized by `isort` standards
```python
# Standard library
import json
import logging
from pathlib import Path

# Third party
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# Local
from src.preprocessing import build_training_frame
```

**Naming Conventions**:
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_single_underscore`

**Docstring Format**: Google-style docstrings
```python
def build_training_frame(data_source, target_col: str = "sales") -> pd.DataFrame:
    """Build ML-ready training frame with feature engineering.
    
    Performs data loading, feature extraction, and validation for model training.
    Handles missing values, categorical encoding, and dtype normalization.
    
    Args:
        data_source: Path (str), bytes, or DataFrame with raw data
        target_col: Target column name (default: "sales")
    
    Returns:
        DataFrame with engineered features and validated types
    
    Raises:
        FileNotFoundError: If required auxiliary files missing
        ValueError: If required columns missing from input
    
    Example:
        >>> df = build_training_frame("data/raw/combined.csv")
        >>> X, y = df.drop('sales', axis=1), df['sales']
    """
```

---

## Git Commit Standards

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type**: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`
**Scope**: The module affected (e.g., `drift`, `training`, `api`)
**Subject**: Brief description (50 chars max, lowercase)

**Examples**:
```git
feat(drift): add auto-retrain with guardrail thresholds
fix(preprocessing): explicit float64 casting for pandas 2.2 compatibility
docs(readme): add system architecture section
perf(training): optimize feature encoding with categorical dtype
test(drift): add tests for drift detection edge cases
```

### Good Commit Practices

- ✅ One logical change per commit
- ✅ Test before committing
- ✅ Reference issues: `Closes #123` in footer
- ✅ Write present tense: "Add feature" not "Added feature"
- ❌ Don't mix formatting with logic changes
- ❌ Don't commit print statements or debug code

---

## Testing

### Test Structure

Tests live in `tests/` directory (mirror of `src/`):
```
tests/
├── test_training.py
├── test_preprocessing.py
├── test_drift.py
├── test_retrain.py
└── fixtures/
    └── sample_data.csv
```

### Writing Tests

```python
import pytest
from src.preprocessing import build_training_frame

class TestPreprocessing:
    """Tests for preprocessing module."""
    
    def test_build_training_frame_basic(self):
        """Test basic frame building with valid data."""
        df = build_training_frame("tests/fixtures/sample_data.csv")
        
        assert df is not None
        assert len(df) > 0
        assert "family" in df.columns
        assert "lag_7" in df.columns
    
    def test_missing_required_columns(self):
        """Test error handling for missing columns."""
        with pytest.raises(ValueError):
            build_training_frame("tests/fixtures/incomplete_data.csv")
    
    def test_dtype_casting(self):
        """Test that numerical features are float64."""
        df = build_training_frame("tests/fixtures/sample_data.csv")
        
        assert df["store_nbr"].dtype == "float64"
        assert df["dcoilwtico"].dtype == "float64"
```

### Run Tests

```bash
# Run all tests
pytest

# Run specific file
pytest tests/test_preprocessing.py

# Run with coverage
pytest --cov=src

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

### Test Coverage

Aim for >80% coverage:
```bash
pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

---

## Documentation Standards

### README.md
- Concise overview (what is this?)
- Quick start guide
- Feature highlights
- Architecture diagram (ASCII or link)
- License

### Code Comments
```python
# Use when "why" is non-obvious, not "what"
# Good:
if drift_ratio > 0.75:
    # Trigger retrain if majority of features drift
    # (prevents model degradation with distributional shift)
    retrain_model()

# Bad:
# Check if drift_ratio > 0.75
if drift_ratio > 0.75:
    retrain_model()
```

### Docstrings
- Every function/class gets a docstring
- Include args, return, raises, example
- Update docstring if function signature changes

### README for Modules
Optional README.md in subdirectories:
```
src/
├── README.md  ← Document module purpose & key functions
└── training.py
```

---

## Pull Request Process

### 1. Create Pull Request

```bash
# Push your branch
git push origin feature/your-feature-name

# Open PR on GitHub
# Title: `feat(scope): brief description`
# Description: Include motivation, changes, testing
```

### 2. PR Description Template

```markdown
## 📝 Description
Brief description of changes

## 🎯 Motivation and Context
Why is this change needed?

## 📋 Type of Change
- [ ] 🐛 Bug fix
- [ ] ✨ New feature
- [ ] 📚 Documentation update
- [ ] ♻️ Refactoring
- [ ] 🚀 Performance improvement

## ✅ Testing
- [ ] Unit tests added
- [ ] Manual testing done
- [ ] No breaking changes

## 📸 Screenshots/Results (if applicable)
Before/after or test results

## ✔️ Checklist
- [ ] Code follows PEP 8 / black formatting
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests pass locally

## 🔗 Related Issues
Closes #123
```

### 3. Code Review

- **Respond to feedback** within 48 hours
- **Don't take criticism personally** - it's about code quality
- **Discuss disagreements** respectfully
- **Update code** based on feedback
- **Request re-review** after changes

### 4. Merge

- Requires 1 approval from maintainer
- All CI checks must pass
- Branch deleted after merge

---

## Common Contribution Types

### Adding a New Feature

1. **Create feature branch**:
   ```bash
   git checkout -b feature/new-feature
   ```

2. **Write failing test first** (TDD):
   ```python
   def test_new_feature():
       result = new_feature(input)
       assert result == expected
   ```

3. **Implement feature** to make test pass

4. **Add docstring and comments**

5. **Update relevant documentation**

6. **Create PR with motivation**

### Fixing a Bug

1. **Create issue** describing the bug

2. **Create fix branch**:
   ```bash
   git checkout -b fix/issue-description
   ```

3. **Add regression test** (fails before fix, passes after)

4. **Implement fix**

5. **Verify test passes** and no new issues

6. **Create PR** referencing the issue

### Improving Performance

1. **Document baseline** performance

2. **Profile before/after**:
   ```python
   import cProfile
   cProfile.run('function_to_optimize()')
   ```

3. **Implement optimization**

4. **Benchmark improvement**:
   ```bash
   pytest benchmark/ -v
   ```

5. **Document changes and results**

### Improving Documentation

```bash
git checkout -b docs/topic
# Edit .md files in docs/ or README.md
# Run spell check: aspell check file.md
git commit -m "docs(scope): description"
git push origin docs/topic
```

---

## Reporting Issues

### Bug Report Template

```markdown
**Describe the bug**
Clear description of what the bug is

**Steps to Reproduce**
1. Upload file...
2. Click button...
3. See error...

**Expected behavior**
What should happen?

**Actual behavior**
What actually happened?

**Environment**
- OS: Windows/macOS/Linux
- Python version: 3.9.x
- Package versions: see pip freeze

**Logs/Error Messages**
```
Paste full traceback here
```

**Screenshots**
If applicable

**Workaround**
Any temporary workaround?
```

### Feature Request Template

```markdown
**Use Case**
Describe the problem you're trying to solve

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Other approaches tried?

**Additional Context**
Any other information?
```

---

## Code Review Checklist

When reviewing PRs, check:

- [ ] **Correctness**: Does the code do what it's supposed to?
- [ ] **Testing**: Are there adequate tests? Do they pass?
- [ ] **Performance**: Any obvious inefficiencies?
- [ ] **Security**: Any potential vulnerabilities?
- [ ] **Documentation**: Docstrings, comments, README updated?
- [ ] **Style**: Follows PEP 8 and project conventions?
- [ ] **No Regressions**: Existing tests still pass?
- [ ] **Scope**: Does PR stay focused (not too many unrelated changes)?

---

## Development Tips

### Local Testing Before PR

```bash
# Format code
black .

# Check style
flake8

# Run tests
pytest --cov=src

# Type checking
mypy src/  # (if mypy installed)
```

### Staying Updated

```bash
# Sync with latest upstream
git fetch upstream
git rebase upstream/main

# Resolve conflicts if any
# Then force push:
git push origin feature/branch --force
```

### Debugging in IDE

**VS Code launch.json**:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal"
    },
    {
      "name": "FastAPI",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": ["api.main:app", "--reload"],
      "jinja": true
    }
  ]
}
```

---

## Community

- **Questions?** Open a Discussion on GitHub
- **Chat?** Join our Discord (link)
- **Ideas?** Share in Issues or Discussions
- **Found a bug?** Report it with the template above

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

## Recognition

Contributors are recognized in:
- GitHub contributors page
- CONTRIBUTORS.md file
- Release notes for accepted changes

Thank you for contributing! 🎉


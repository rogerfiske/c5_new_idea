# Contributing to Quantum State Prediction Experiment

Thank you for contributing to this project! This document provides guidelines for writing code, documentation, and tests.

## Core Principle: Clarity for Non-Programmers

**The primary user of this codebase is a non-programmer.** All code, documentation, and instructions must be written with exceptional clarity and detailed explanations.

## Coding Standards

### 1. Extensive Comments

**Every code block must explain the "why," not just the "what."**

#### Good Example ✅
```python
# Calculate the normalized amplitudes for quantum state superposition
# We divide by sqrt(k) where k=5 (number of active positions) to ensure
# the quantum state has unit norm (sum of squared amplitudes = 1)
amplitude = 1.0 / np.sqrt(len(active_positions))
```

#### Bad Example ❌
```python
# Calculate amplitude
amplitude = 1.0 / np.sqrt(len(active_positions))
```

### 2. Type Hints

**All function signatures must include type hints** for parameters and return values.

```python
def impute_quantum_state(
    binary_vector: np.ndarray,
    method: str,
    normalize: bool = True
) -> np.ndarray:
    """
    Impute quantum state from binary vector using specified method.

    Args:
        binary_vector: 39-dimensional binary array (0s and 1s)
        method: Imputation method name ("basis_embedding", "amplitude_embedding", etc.)
        normalize: Whether to normalize the output state vector

    Returns:
        Quantum state vector as numpy array

    Raises:
        ValueError: If binary_vector is not 39-dimensional
        ValueError: If method is not recognized
    """
    pass
```

### 3. Modularity

- **Functions should do one thing well** and be no longer than 50 lines
- **Classes should have a single responsibility**
- **Break complex operations into smaller, named helper functions**

#### Example
```python
def train_lgbm_ranker(X_train, y_train, params):
    """Train LightGBM ranker model."""
    # Break into clear steps
    model = _initialize_lgbm_model(params)
    model = _fit_with_validation(model, X_train, y_train)
    metrics = _evaluate_training(model, X_train, y_train)
    return model, metrics
```

### 4. Centralized Configuration

**Never hard-code file paths, magic numbers, or hyperparameters.**

```python
# ✅ Good: Use centralized config
from src.config import DATA_RAW, DATASET_PATH
df = pd.read_csv(DATASET_PATH)

# ❌ Bad: Hard-coded path
df = pd.read_csv("data/raw/c5_Matrix.csv")
```

For model hyperparameters, use config dictionaries:

```python
# src/modeling/configs.py
LGBM_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "num_leaves": 31,
    "learning_rate": 0.05
}
```

### 5. Logging Over Print

**Use Python's logging module** for all output except Jupyter notebooks.

```python
import logging

# Configure logger at module level
logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.debug("Detailed debug information")
logger.info("Processing experiment exp01_basis_embedding")
logger.warning("Imputation resulted in 10% zero amplitudes")
logger.error("Failed to load dataset: file not found")
```

### 6. Error Handling

**All functions must handle expected errors gracefully** with informative messages.

```python
def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load the c5_Matrix.csv dataset with validation.

    Args:
        file_path: Path to the CSV file

    Returns:
        Validated pandas DataFrame

    Raises:
        FileNotFoundError: If dataset file does not exist
        ValueError: If dataset format is invalid
    """
    # Check file exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. "
            f"USER ACTION REQUIRED: Copy c5_Matrix.csv to data/raw/ directory. "
            f"See README.md section 'Data Setup' for instructions."
        )

    # Load and validate
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

    # Validate structure
    if len(df.columns) < 39:
        raise ValueError(
            f"Expected at least 39 QV columns, found {len(df.columns)}. "
            f"Check that the dataset file is correct."
        )

    logger.info(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    return df
```

### 7. Docstrings

**All modules, classes, and functions must have docstrings** using Google style format.

```python
def calculate_ndcg(y_true: np.ndarray, y_pred: np.ndarray, k: int = 20) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at rank k.

    NDCG measures the quality of a ranking by comparing the predicted
    ranking to the ideal ranking based on true relevance scores.

    Args:
        y_true: Ground truth relevance scores (39-dim binary vector)
        y_pred: Predicted relevance scores (39-dim probability vector)
        k: Rank cutoff for evaluation (default: 20)

    Returns:
        NDCG score between 0.0 (worst) and 1.0 (perfect)

    Example:
        >>> y_true = np.array([0, 1, 0, 1, 0])
        >>> y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        >>> calculate_ndcg(y_true, y_pred, k=3)
        0.95

    References:
        - Järvelin, K., & Kekäläinen, J. (2002). "Cumulative gain-based
          evaluation of IR techniques." ACM TOIS.
    """
    pass
```

## File Organization

### Module Structure
```
src/
├── __init__.py           # Package initialization
├── config.py             # Centralized configuration
├── data_loader.py        # Data loading utilities
├── evaluation.py         # Evaluation metrics
├── main.py               # Main execution script
├── imputation/           # Imputation methods module
│   ├── __init__.py
│   ├── base_imputer.py   # Abstract base class
│   ├── basis_embedding.py
│   ├── amplitude_embedding.py
│   ├── angle_encoding.py
│   ├── density_matrix.py
│   └── graph_cycle_encoding.py
└── modeling/             # Modeling module
    ├── __init__.py
    ├── pipeline.py       # Training/prediction orchestration
    ├── rankers/
    │   ├── __init__.py
    │   ├── frequency_ranker.py
    │   ├── lgbm_ranker.py
    │   ├── set_transformer_model.py
    │   └── gnn_ranker.py
    └── ensembles/
        ├── __init__.py
        ├── rrf.py
        └── weighted_average.py
```

## Testing Standards

### Test Coverage
- **All public functions must have unit tests**
- **Each imputation method must have integration tests**
- **Each ranker model must have integration tests**
- **Critical workflows must have end-to-end tests**

### Test Structure
```python
# tests/unit/imputation/test_amplitude_embedding.py
import pytest
import numpy as np
from src.imputation.amplitude_embedding import AmplitudeEmbedding

class TestAmplitudeEmbedding:
    """Test suite for Amplitude Embedding imputation method."""

    def test_impute_with_five_active_positions(self):
        """Test imputation with standard 5 active positions."""
        # Arrange
        binary_vector = np.zeros(39)
        binary_vector[[4, 7, 9, 29, 37]] = 1  # 5 active positions
        imputer = AmplitudeEmbedding()

        # Act
        quantum_state = imputer.impute(binary_vector)

        # Assert
        assert quantum_state.shape == (39,)
        # Check normalization: sum of squared amplitudes = 1
        assert np.isclose(np.sum(quantum_state ** 2), 1.0)
        # Check only active positions have non-zero amplitudes
        assert np.all(quantum_state[[4, 7, 9, 29, 37]] != 0)
        assert np.all(quantum_state[[0, 1, 2, 3, 5, 6]] == 0)

    def test_impute_with_invalid_dimension(self):
        """Test that invalid input dimension raises ValueError."""
        # Arrange
        binary_vector = np.zeros(40)  # Wrong dimension
        imputer = AmplitudeEmbedding()

        # Act & Assert
        with pytest.raises(ValueError, match="Expected 39-dimensional"):
            imputer.impute(binary_vector)
```

### Fixtures
```python
# tests/fixtures/sample_data.py
import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_binary_vector():
    """Fixture providing a standard binary vector for testing."""
    vec = np.zeros(39)
    vec[[4, 7, 9, 29, 37]] = 1  # Event 1 pattern
    return vec

@pytest.fixture
def sample_dataset():
    """Fixture providing a small sample dataset for testing."""
    data = {
        'QS_1': [5], 'QS_2': [8], 'QS_3': [10], 'QS_4': [30], 'QS_5': [38],
    }
    # Add QV columns
    for i in range(1, 40):
        data[f'QV_{i}'] = [0]
    df = pd.DataFrame(data)
    # Set QV values for event 1
    df.loc[0, ['QV_5', 'QV_8', 'QV_10', 'QV_30', 'QV_38']] = 1
    return df
```

## Git Workflow

### Branching Strategy
- `main` branch: Production-ready code
- `develop` branch: Integration branch for features
- Feature branches: `feature/epic-X-story-Y-description`
- Bugfix branches: `bugfix/issue-description`

### Commit Messages
Use clear, descriptive commit messages:

```
✅ Good:
"Add Amplitude Embedding imputation method (Story 2.3)"
"Fix NDCG calculation for edge case with no positive labels"
"Update README with data setup instructions"

❌ Bad:
"fix bug"
"update code"
"stuff"
```

### Commit Workflow
```bash
# 1. Check status
git status

# 2. Add files
git add src/imputation/amplitude_embedding.py
git add tests/unit/imputation/test_amplitude_embedding.py

# 3. Commit with descriptive message
git commit -m "Add Amplitude Embedding imputation method (Story 2.3)

- Implement superposition over active positions
- Normalize amplitudes to ensure unit norm
- Add comprehensive unit tests
- Add docstrings and type hints"

# 4. Push to remote
git push origin feature/epic-2-story-3-amplitude-embedding
```

## Documentation Standards

### Inline Documentation
- **Every complex algorithm needs a comment block** explaining the approach
- **Every magic number needs a comment** explaining its significance
- **Every assumption needs documentation** so future maintainers understand constraints

### README Updates
- Update README.md whenever:
  - New dependencies are added
  - Setup instructions change
  - New usage patterns are introduced

### Change Logs
- Update relevant change logs in PRD and architecture docs when:
  - Major features are added
  - Architecture changes occur
  - Breaking changes are introduced

## Jupyter Notebook Standards

### Notebook Structure
```markdown
# 1. Title and Overview
Description of what this notebook does

# 2. Imports and Setup
All import statements and configuration

# 3. Load Data
Load and validate dataset

# 4. Analysis/Experiment
Main content with clear section headers

# 5. Results
Summarize findings

# 6. Next Steps
Document follow-up actions or questions
```

### Notebook Best Practices
- **Run "Restart and Run All" before committing** to ensure reproducibility
- **Clear output cells before committing** if outputs are large (>1MB)
- **Add markdown cells liberally** to explain what each code cell does
- **Use meaningful variable names** (not `df1`, `df2`, `result`)

## Code Review Checklist

Before submitting code for review, verify:

- [ ] All functions have type hints
- [ ] All functions have Google-style docstrings
- [ ] All complex logic has explanatory comments
- [ ] No hard-coded paths or magic numbers
- [ ] Logging used instead of print statements
- [ ] Error handling is comprehensive with clear messages
- [ ] Unit tests written and passing
- [ ] pytest runs successfully with no failures
- [ ] Code follows PEP 8 style guidelines
- [ ] README updated if needed
- [ ] Commit messages are clear and descriptive

## Questions or Issues?

If you have questions about these guidelines or encounter issues:

1. Check existing documentation in `docs/`
2. Review similar existing code for patterns
3. Ask for clarification from the project lead

---

**Remember**: The goal is to make this codebase accessible and maintainable for non-programmers. When in doubt, over-explain rather than under-explain.

**Last Updated**: 2025-10-12
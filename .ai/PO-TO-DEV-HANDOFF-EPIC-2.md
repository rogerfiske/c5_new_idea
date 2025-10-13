# PO-TO-DEV HANDOFF: EPIC 2 - QUANTUM-INSPIRED IMPUTATION FRAMEWORK

**Document Version**: 1.0
**Date**: 2025-10-13
**Prepared By**: Sarah (Product Owner)
**For**: James (Developer Agent)
**Project**: Quantum State Prediction Experiment
**PRD Version**: 1.3
**Architecture Version**: 1.2

---

## EXECUTIVE SUMMARY

Epic 2 implements the core quantum-inspired imputation framework, including an abstract base class and 5 concrete imputation strategies. This epic is critical as it creates the feature engineering foundation that all ranking models (Epic 3) will depend on.

**Epic Status**: Ready to begin
**Prerequisite**: Epic 1 complete ✅
**Stories**: 7 stories (2.1-2.7)
**Estimated Complexity**: High (quantum-inspired mathematical concepts, abstract design patterns)
**Critical Success Factor**: Abstract base class (Story 2.1) must define clear interface contract to ensure consistency across all 5 imputation methods

---

## EPIC 2 GOALS

From PRD Section 3 - Epic 2:

> **Goal**: Build a flexible framework for applying quantum-inspired imputation methods and implement the initial set of strategies.

**Key Objectives**:
1. Design abstract base class with standardized interface for all imputation methods
2. Implement 5 quantum-inspired imputation strategies with consistent interfaces
3. Create utility script to apply any imputation strategy to raw dataset
4. Ensure all implementations are heavily documented (NFR2: non-programmer friendly)
5. Achieve comprehensive test coverage for all imputation methods

---

## VALIDATION FINDINGS & RECOMMENDATIONS

### From Epic 1 Validation Report

**✅ Strengths to Leverage**:
- Testing infrastructure (Story 1.4) is robust - use pytest patterns established in `tests/unit/test_data_loader.py`
- Data loading utilities (`src/data_loader.py`) ready for use - import and leverage existing validation functions
- EDA insights (notebooks/1_EDA.ipynb) inform imputation design:
  - Non-uniform position distribution (positions 1-10 most frequent)
  - Strong co-occurrence patterns detected
  - High state diversity (9,741 unique states, 84% diversity)

**⚠️ Critical Recommendations**:

1. **Define Imputation Output Contract (Story 2.1)**: The abstract base class MUST specify exact output format:
   - Input: pandas DataFrame from `data_loader.load_dataset()`
   - Output: Specify data structure (DataFrame? NumPy array? Dimensions?)
   - Use Python type hints for all method signatures
   - Document expected output dimensions and data types

2. **Test Coverage Target**: Aim for 80%+ function coverage for all imputation modules
   - Add to Story 2.1 acceptance criteria
   - Each strategy (Stories 2.2-2.6) must have unit tests matching `test_data_loader.py` pattern

3. **Parameter Validation**: All imputation methods should validate inputs using patterns from `data_loader.validate_dataset_structure()`

---

## STORY-BY-STORY BREAKDOWN

### Story 2.1: Base Imputation Class (Abstract Interface)

**Priority**: CRITICAL (blocks all other stories)
**Complexity**: Medium-High
**Estimated Lines**: 150-200 lines with docstrings

#### Requirements (PRD Lines 87-92)

From PRD:
> Design and implement a base Python class for imputation strategies that takes raw data and returns a feature-engineered representation.

**Detailed Specifications**:

1. **File Location**: `src/imputation/base_imputer.py`

2. **Class Design**:
```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

class BaseImputer(ABC):
    """
    Abstract base class for quantum-inspired imputation strategies.

    This class defines the standard interface that all imputation methods
    must implement. Each imputation strategy transforms raw binary quantum
    state data into feature-engineered representations suitable for ranking
    model training.

    Attributes:
        name (str): Human-readable name of the imputation strategy
        config (Dict[str, Any]): Configuration parameters for the strategy
        fitted_ (bool): Whether the imputer has been fitted to data

    Examples:
        >>> class MyImputer(BaseImputer):
        ...     def _fit(self, X):
        ...         # Learn strategy-specific parameters
        ...         pass
        ...     def _transform(self, X):
        ...         # Apply transformation
        ...         return transformed_data
        >>> imputer = MyImputer(name="my_strategy")
        >>> imputer.fit(training_data)
        >>> imputed = imputer.transform(new_data)
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the base imputer."""
        pass

    @abstractmethod
    def _fit(self, X: pd.DataFrame) -> None:
        """
        Learn imputation parameters from training data.

        Args:
            X: Input DataFrame with columns [event-ID, QV_1, ..., QV_39]

        Raises:
            ValueError: If input data format is invalid
        """
        pass

    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using learned parameters.

        Args:
            X: Input DataFrame with columns [event-ID, QV_1, ..., QV_39]

        Returns:
            Transformed feature matrix of shape (n_samples, n_features)
            where n_features depends on the imputation strategy

        Raises:
            ValueError: If imputer not fitted or invalid input
        """
        pass

    def fit(self, X: pd.DataFrame) -> "BaseImputer":
        """Public fit method with validation."""
        pass

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Public transform method with validation."""
        pass

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        pass

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate input DataFrame matches expected format."""
        pass
```

3. **Input Validation Requirements**:
   - Check DataFrame has 40 columns (event-ID + QV_1-39)
   - Verify QV columns are binary (0 or 1)
   - Verify no missing values
   - Use patterns from `data_loader.validate_dataset_structure()`

4. **Output Specification** (CRITICAL):
   - **Return Type**: `np.ndarray` of shape `(n_samples, n_features)`
   - **n_features**: Varies by strategy, should be documented in subclass docstring
   - **Data Type**: `np.float64` (to support continuous-valued features)
   - **No NaN/Inf**: Validate output has no invalid values

5. **Configuration Management**:
   - Accept optional `config` dict for strategy-specific parameters
   - Store configuration for reproducibility
   - Include config in `__repr__()` for debugging

6. **Error Handling**:
   - Raise `ValueError` for invalid input data
   - Raise `RuntimeError` if transform() called before fit()
   - User-friendly error messages (NFR2)

#### Acceptance Criteria

- [ ] Base class defined in `src/imputation/base_imputer.py`
- [ ] Abstract methods `_fit()` and `_transform()` defined with type hints
- [ ] Public methods `fit()`, `transform()`, `fit_transform()` implemented
- [ ] Input validation method `_validate_input()` reuses `data_loader` validation patterns
- [ ] Output contract clearly specified: `np.ndarray` of shape `(n_samples, n_features)`
- [ ] Comprehensive docstrings with Examples section (NFR2)
- [ ] Unit tests in `tests/unit/test_base_imputer.py`:
  - Test abstract methods cannot be instantiated directly
  - Test input validation catches invalid data
  - Test fit-before-transform enforcement
- [ ] Test coverage ≥ 80% for base class

#### Technical Notes

- Import existing `data_loader` functions: `from src.data_loader import validate_dataset_structure`
- Use Python's `logging` module (not print statements)
- Follow coding standards from `architecture.md` section 6

---

### Story 2.2: Basis Embedding Strategy

**Priority**: High
**Complexity**: Medium
**Estimated Lines**: 120-150 lines with docstrings
**Dependencies**: Story 2.1 complete

#### Requirements (PRD Lines 94-99)

From PRD:
> Implement the **Basis Embedding** strategy.

**Quantum-Inspired Concept**:
Basis Embedding maps each binary quantum state vector directly to a computational basis state representation. Each active position (QV=1) corresponds to a basis vector in a Hilbert space.

**Mathematical Specification**:
- Input: Binary vector with 5 active positions out of 39 possible
- Output: Feature representation capturing active position indices and patterns
- Example: Event with active positions [1, 5, 8, 30, 38] → Feature vector encoding these positions

#### Detailed Specifications

1. **File Location**: `src/imputation/basis_embedding.py`

2. **Class Implementation**:
```python
class BasisEmbedding(BaseImputer):
    """
    Basis Embedding imputation strategy.

    Maps binary quantum state vectors to computational basis state representations.
    Each active position (QV=1) is treated as a basis vector in a 39-dimensional
    Hilbert space.

    This strategy creates features based on:
    1. Active position indices (which of the 39 positions are active)
    2. Position frequency statistics
    3. One-hot encoding of active positions

    Parameters:
        include_frequency_features (bool): Whether to include position frequency stats

    Output Dimensions:
        n_features = 39 (one-hot) + 39 (frequency) = 78 features per sample

    Examples:
        >>> from src.data_loader import load_dataset
        >>> from src.imputation.basis_embedding import BasisEmbedding
        >>> df = load_dataset()
        >>> imputer = BasisEmbedding(name="basis_embedding")
        >>> imputer.fit(df)
        >>> features = imputer.transform(df)
        >>> features.shape
        (11581, 78)
    """

    def __init__(self, name: str = "basis_embedding",
                 include_frequency_features: bool = True):
        super().__init__(name=name, config={
            "include_frequency_features": include_frequency_features
        })
        self.include_frequency_features = include_frequency_features
        self.position_frequencies_ = None

    def _fit(self, X: pd.DataFrame) -> None:
        """
        Learn position frequency statistics from training data.

        Calculates how often each of the 39 positions is active across
        all training samples. This will be used to create frequency-based
        features during transformation.
        """
        # Calculate position frequencies
        qv_columns = [f'QV_{i}' for i in range(1, 40)]
        self.position_frequencies_ = X[qv_columns].sum(axis=0) / len(X)
        # Convert to numpy array for efficient computation
        self.position_frequencies_ = self.position_frequencies_.values

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform binary vectors to basis embedding features.

        Creates feature matrix with:
        - Columns 0-38: One-hot encoding (copy of QV_1-39)
        - Columns 39-77: Position frequency features (optional)

        Returns:
            Feature matrix of shape (n_samples, 78)
        """
        qv_columns = [f'QV_{i}' for i in range(1, 40)]
        qv_data = X[qv_columns].values

        features = [qv_data]  # Start with one-hot encoding

        if self.include_frequency_features:
            # Multiply each position by its frequency
            frequency_features = qv_data * self.position_frequencies_
            features.append(frequency_features)

        result = np.hstack(features)
        return result
```

3. **Feature Engineering Strategy**:
   - **One-Hot Component**: Direct copy of QV_1-39 columns (preserves active positions)
   - **Frequency Component**: Weight each position by its training set frequency (captures statistical patterns from EDA)
   - **Output**: 78 features (or 39 if frequency features disabled)

4. **Testing Requirements**:
   - File: `tests/unit/test_basis_embedding.py`
   - Test fixture: Use `tests/fixtures/sample_valid_dataset.csv`
   - Test cases:
     - Test fit() learns position frequencies
     - Test transform() produces expected output shape (n_samples, 78)
     - Test output has no NaN/Inf
     - Test with/without frequency features
     - Test error handling for unfitted imputer

#### Acceptance Criteria

- [ ] `BasisEmbedding` class inherits from `BaseImputer`
- [ ] Implements `_fit()` and `_transform()` abstract methods
- [ ] Produces output of shape `(n_samples, 78)` with default config
- [ ] Code includes detailed inline comments explaining quantum concept (NFR2)
- [ ] Comprehensive docstrings with mathematical explanation
- [ ] Unit tests pass with coverage ≥ 80%
- [ ] Can successfully fit and transform `tests/fixtures/sample_valid_dataset.csv`

---

### Story 2.3: Amplitude Embedding Strategy

**Priority**: High
**Complexity**: Medium
**Estimated Lines**: 130-160 lines with docstrings
**Dependencies**: Story 2.1 complete

#### Requirements (PRD Lines 100-103)

From PRD:
> Implement the **Amplitude Embedding** strategy (superposition over active positions).

**Quantum-Inspired Concept**:
Amplitude Embedding represents quantum states as superpositions where amplitudes are distributed across active positions. This captures the probabilistic nature of quantum superposition.

**Mathematical Specification**:
- Input: Binary vector with 5 active positions
- Transform: Create amplitude vector where active positions have non-zero amplitudes summing to 1 (normalized)
- Output: Feature representation capturing amplitude distributions and interference patterns

#### Detailed Specifications

1. **File Location**: `src/imputation/amplitude_embedding.py`

2. **Key Concepts**:
   - **Uniform Superposition**: Equal amplitude (1/√5 ≈ 0.447) for each of 5 active positions
   - **Weighted Superposition**: Amplitudes proportional to position frequencies from training data
   - **Amplitude Squared Features**: |amplitude|² represents probability (quantum mechanics)

3. **Feature Engineering Strategy**:
   - **Amplitude Vector** (39 dims): Amplitudes for each position (0 for inactive, non-zero for active)
   - **Probability Vector** (39 dims): Amplitude squared (|a|²) for each position
   - **Phase Features** (optional): Encode relative phases between active positions
   - **Output**: 78+ features depending on configuration

4. **Implementation Pattern**:
```python
class AmplitudeEmbedding(BaseImputer):
    """
    Amplitude Embedding imputation strategy.

    Represents quantum states as superpositions with amplitudes distributed
    across active positions. This captures the probabilistic nature of
    quantum superposition states.

    Parameters:
        normalization (str): "uniform" or "weighted"
            - uniform: Equal amplitudes (1/√5) for all active positions
            - weighted: Amplitudes proportional to position frequencies
        include_probability_features (bool): Include |amplitude|² features

    Output Dimensions:
        n_features = 39 (amplitudes) + 39 (probabilities) = 78 features
    """

    def _fit(self, X: pd.DataFrame) -> None:
        """Learn position frequencies for weighted normalization."""
        # Calculate position frequencies for weighted amplitudes
        pass

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform to amplitude embedding features.

        For each event:
        1. Identify 5 active positions
        2. Assign amplitudes based on normalization strategy
        3. Create amplitude vector (39 dims)
        4. Compute probability vector (amplitude squared)
        5. Concatenate features
        """
        pass
```

#### Acceptance Criteria

- [ ] `AmplitudeEmbedding` class inherits from `BaseImputer`
- [ ] Implements both uniform and weighted superposition modes
- [ ] Amplitudes properly normalized (sum of squared amplitudes = 1)
- [ ] Produces output of expected shape
- [ ] Docstrings explain superposition concept in accessible language (NFR2)
- [ ] Unit tests cover both normalization strategies
- [ ] Test coverage ≥ 80%

---

### Story 2.4: Angle/Rotation Encoding Strategy

**Priority**: High
**Complexity**: Medium
**Estimated Lines**: 120-150 lines with docstrings
**Dependencies**: Story 2.1 complete

#### Requirements (PRD Lines 104-107)

From PRD:
> Implement the **Angle/Rotation Encoding** strategy.

**Quantum-Inspired Concept**:
Angle Encoding maps binary states to rotation angles on the Bloch sphere (quantum state representation). Each position corresponds to a rotation angle, and active positions determine the overall quantum state orientation.

**Mathematical Specification**:
- Input: Binary vector with 5 active positions
- Transform: Map positions to angles θ ∈ [0, 2π]
- Features: Trigonometric functions (sin, cos) of angles capture rotational symmetries
- Output: Feature vector encoding angular relationships

#### Detailed Specifications

1. **File Location**: `src/imputation/angle_encoding.py`

2. **Angular Mapping Strategy**:
   - Map position indices to angles: `θᵢ = 2π * (i-1) / 39` for QV_i
   - For active positions, compute sin(θ), cos(θ), tan(θ/2)
   - Aggregate angular features across 5 active positions per event

3. **Feature Engineering**:
   - **Direct Angle Features** (39 dims): θᵢ for each position (0 if inactive, angle if active)
   - **Trigonometric Features** (78 dims): sin(θᵢ), cos(θᵢ) for each position
   - **Aggregated Features** (6 dims): Sum/mean of sin and cos across active positions
   - **Output**: ~123 features

4. **Cyclic Symmetry**: Leverage C₃₉ cyclic group structure (positions wrap around)

#### Acceptance Criteria

- [ ] `AngleEncoding` class inherits from `BaseImputer`
- [ ] Angular mapping properly implements rotational encoding
- [ ] Trigonometric features (sin, cos) computed correctly
- [ ] Handles cyclic symmetry of C₃₉ group
- [ ] Docstrings explain rotation concept with visual diagrams in comments (NFR2)
- [ ] Unit tests verify angular computations
- [ ] Test coverage ≥ 80%

---

### Story 2.5: Density Matrix Embedding Strategy

**Priority**: High
**Complexity**: High (most mathematically complex)
**Estimated Lines**: 150-200 lines with docstrings
**Dependencies**: Story 2.1 complete

#### Requirements (PRD Lines 108-112)

From PRD:
> Implement the **Density Matrix Embedding** strategy (for mixed states).

**Quantum-Inspired Concept**:
Density Matrix Embedding represents quantum states as mixed states using density matrices (ρ). This captures statistical ensembles and partial information about quantum systems.

**Mathematical Specification**:
- Density Matrix: ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ| (sum over pure states weighted by probabilities)
- For binary states: Construct density matrix from active position patterns
- Properties: ρ is Hermitian, positive semi-definite, Tr(ρ) = 1
- Output: Features extracted from density matrix (eigenvalues, traces, purity)

#### Detailed Specifications

1. **File Location**: `src/imputation/density_matrix.py`

2. **Density Matrix Construction**:
   - Create 39×39 density matrix for each event
   - Diagonal elements: Probabilities for each position (1/5 for active, 0 for inactive)
   - Off-diagonal elements: Coherences between active positions (1/25 for active pairs)

3. **Feature Extraction from Density Matrix**:
   - **Diagonal Elements** (39 dims): Main diagonal of ρ
   - **Purity**: Tr(ρ²) (measure of "mixedness")
   - **Eigenvalues** (5-10 largest): Spectral features
   - **Von Neumann Entropy**: -Tr(ρ log ρ) (optional, computationally expensive)
   - **Output**: ~50-60 features

4. **Computational Efficiency**:
   - Full 39×39 matrix is sparse (only 5 active positions)
   - Consider sparse matrix representation
   - Eigenvalue computation may be slow (use `np.linalg.eigh` for Hermitian matrices)

#### Acceptance Criteria

- [ ] `DensityMatrixEmbedding` class inherits from `BaseImputer`
- [ ] Correctly constructs density matrix satisfying quantum properties:
  - Hermitian: ρ = ρ†
  - Positive semi-definite: all eigenvalues ≥ 0
  - Normalized: Tr(ρ) = 1
- [ ] Extracts physically meaningful features (purity, eigenvalues)
- [ ] Handles computational complexity efficiently
- [ ] Extensive docstrings explaining density matrix concept (NFR2)
- [ ] Unit tests verify matrix properties
- [ ] Test coverage ≥ 80%

#### Technical Notes

- This is the most complex imputation method mathematically
- Consider implementing simplified version first, then add advanced features
- May require `scipy.linalg` for efficient matrix operations
- Document computational complexity in docstrings (relevant for RunPod decisions in Epic 5)

---

### Story 2.6: Graph/Cycle Encoding Strategy

**Priority**: High
**Complexity**: Medium-High
**Estimated Lines**: 140-180 lines with docstrings
**Dependencies**: Story 2.1 complete

#### Requirements (PRD Lines 113-116)

From PRD:
> Implement the **Graph/Cycle Encoding** (circular convolution / DFT) strategy. Include ring features (circular distances & DFT harmonics).

**Quantum-Inspired Concept**:
Graph/Cycle Encoding treats the 39 positions as nodes in a cyclic graph (C₃₉ ring). Uses Discrete Fourier Transform (DFT) to capture periodic patterns and harmonic structure in the cyclic group.

**Mathematical Specification**:
- Graph Structure: 39 nodes in a cycle (position i connects to i+1 mod 39)
- DFT: Transform binary vector to frequency domain using FFT
- Circular Distances: Distance between active positions on the ring
- Output: DFT harmonics + graph-based features

#### Detailed Specifications

1. **File Location**: `src/imputation/graph_cycle_encoding.py`

2. **Graph Features**:
   - **Circular Distances**: Shortest path distances between active positions on ring
   - **Clustering Coefficient**: How "clustered" active positions are on the ring
   - **Ring Symmetry**: Detect symmetric patterns (e.g., evenly spaced active positions)

3. **DFT/FFT Features**:
   - Apply FFT to binary vector: `F = np.fft.fft(qv_vector)`
   - Extract magnitude spectrum: `|F|`
   - Extract phase spectrum: `angle(F)`
   - Low-frequency harmonics capture global patterns
   - Output: 39 magnitude features + 39 phase features = 78 DFT features

4. **Combined Feature Set**:
   - **DFT Features** (78 dims): Magnitude and phase spectra
   - **Graph Features** (~20 dims): Distances, clustering, symmetry measures
   - **Output**: ~98 features

5. **Cyclic Convolution** (optional advanced feature):
   - Convolve binary vector with circular kernels
   - Detect local patterns on the ring

#### Implementation Pattern

```python
class GraphCycleEncoding(BaseImputer):
    """
    Graph/Cycle Encoding imputation strategy.

    Treats the 39 quantum positions as nodes in a cyclic graph (C₃₉ ring).
    Uses Discrete Fourier Transform (DFT) to capture periodic patterns
    and harmonic structure in the cyclic group.

    Parameters:
        include_dft_features (bool): Include DFT magnitude and phase
        include_graph_features (bool): Include graph-based distances and clustering
        n_harmonics (int): Number of DFT harmonics to keep (default: 20)

    Output Dimensions:
        n_features = 2*n_harmonics (DFT) + ~20 (graph) ≈ 60 features
    """

    def _fit(self, X: pd.DataFrame) -> None:
        """Learn global DFT statistics (optional)."""
        pass

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform to graph/cycle encoding features.

        For each event:
        1. Apply FFT to binary vector
        2. Extract magnitude and phase of top harmonics
        3. Compute circular distances between active positions
        4. Calculate graph clustering metrics
        5. Concatenate all features
        """
        pass
```

#### Acceptance Criteria

- [ ] `GraphCycleEncoding` class inherits from `BaseImputer`
- [ ] Correctly implements FFT/DFT using `np.fft.fft`
- [ ] Computes circular distances on C₃₉ ring
- [ ] Extracts meaningful graph features (clustering, symmetry)
- [ ] Docstrings explain DFT and graph concepts (NFR2)
- [ ] Unit tests verify DFT properties (Parseval's theorem: energy conservation)
- [ ] Test coverage ≥ 80%

#### Technical Notes

- Use `np.fft.fft` for efficient DFT computation (O(n log n))
- FFT output is complex-valued; extract magnitude and phase separately
- Consider dimensionality reduction (keep only top k harmonics)
- This strategy leverages EDA finding: cyclic patterns in data

---

### Story 2.7: Imputation Utility Script

**Priority**: High
**Complexity**: Medium
**Estimated Lines**: 180-220 lines with docstrings
**Dependencies**: Stories 2.1-2.6 complete

#### Requirements (PRD Lines 117-124)

From PRD:
> Implement a script to apply any chosen imputation strategy to the raw dataset and save the output.

**Purpose**: Provide unified interface to apply any of the 5 imputation methods to raw data, with validation and logging.

#### Detailed Specifications

1. **File Location**: `src/imputation/apply_imputation.py`

2. **Functionality**:
   - Accept command-line arguments: `--strategy`, `--input-path`, `--output-path`, `--config`
   - Load raw data using `data_loader.load_dataset()`
   - Instantiate selected imputation strategy
   - Fit and transform data
   - Validate imputation output (no NaN/Inf)
   - Save imputed data with metadata
   - Log execution details

3. **Supported Strategies**:
```python
AVAILABLE_STRATEGIES = {
    "basis_embedding": BasisEmbedding,
    "amplitude_embedding": AmplitudeEmbedding,
    "angle_encoding": AngleEncoding,
    "density_matrix": DensityMatrixEmbedding,
    "graph_cycle": GraphCycleEncoding
}
```

4. **Command-Line Interface**:
```bash
# Example usage
python src/imputation/apply_imputation.py \
    --strategy basis_embedding \
    --input data/raw/c5_Matrix.csv \
    --output data/processed/imputed_basis_embedding_v1.parquet \
    --config '{"include_frequency_features": true}'
```

5. **Output Format**:
   - **File Format**: Parquet (efficient for large numerical data)
   - **Columns**: `event-ID` + feature columns `feat_0`, `feat_1`, ..., `feat_N`
   - **Metadata File**: Save JSON sidecar with imputation metadata

6. **Metadata JSON** (saved as `{output_path}.meta.json`):
```json
{
    "strategy": "basis_embedding",
    "timestamp": "2025-10-13T15:30:00Z",
    "input_file": "data/raw/c5_Matrix.csv",
    "input_shape": [11581, 40],
    "output_file": "data/processed/imputed_basis_embedding_v1.parquet",
    "output_shape": [11581, 78],
    "config": {"include_frequency_features": true},
    "execution_time_seconds": 2.45,
    "validation_passed": true
}
```

7. **Validation Checks**:
```python
def validate_imputation_output(features: np.ndarray) -> Dict[str, Any]:
    """
    Validate imputation output quality.

    Checks:
    - No NaN values
    - No Inf values
    - Expected dimensions
    - Reasonable value ranges

    Returns:
        Dictionary with validation results
    """
    validation_result = {
        "has_nan": np.isnan(features).any(),
        "has_inf": np.isinf(features).any(),
        "shape": features.shape,
        "value_range": (features.min(), features.max()),
        "passed": True
    }

    if validation_result["has_nan"] or validation_result["has_inf"]:
        validation_result["passed"] = False

    return validation_result
```

8. **Error Handling** (NFR2: user-friendly):
   - Invalid strategy name: List available strategies
   - File not found: Provide clear path to data setup instructions
   - Imputation failure: Log error with strategy-specific details
   - Validation failure: Report what went wrong (NaN, Inf, dimensions)

9. **Logging**:
```python
import logging

logger = logging.getLogger(__name__)

# Log execution flow
logger.info(f"Loading data from {input_path}")
logger.info(f"Applying {strategy} imputation strategy")
logger.info(f"Imputation complete: {output_shape}")
logger.info(f"Validation passed: {validation_result['passed']}")
logger.info(f"Saved to {output_path}")
```

#### Also Create: Main Function for Library Usage

```python
def apply_imputation(
    strategy: str,
    input_path: Path,
    output_path: Path,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Apply imputation strategy to dataset (programmatic interface).

    This function can be called from other Python scripts or notebooks
    without using command-line interface.

    Args:
        strategy: Name of imputation strategy
        input_path: Path to raw CSV file
        output_path: Path to save imputed data
        config: Optional configuration for the strategy

    Returns:
        Metadata dictionary with execution details

    Examples:
        >>> from src.imputation.apply_imputation import apply_imputation
        >>> metadata = apply_imputation(
        ...     strategy="basis_embedding",
        ...     input_path=Path("data/raw/c5_Matrix.csv"),
        ...     output_path=Path("data/processed/imputed_basis.parquet")
        ... )
        >>> print(f"Imputation took {metadata['execution_time_seconds']:.2f}s")
    """
    pass
```

#### Acceptance Criteria

- [ ] Script accepts command-line arguments: `--strategy`, `--input-path`, `--output-path`, `--config`
- [ ] Supports all 5 imputation strategies from Stories 2.2-2.6
- [ ] Validates imputation output (NaN, Inf, dimensions)
- [ ] Saves imputed data to Parquet format
- [ ] Saves metadata JSON sidecar file
- [ ] Provides clear error messages for all failure modes (NFR2)
- [ ] Logs execution details using `logging` module
- [ ] Includes programmatic interface (`apply_imputation()` function)
- [ ] Comprehensive docstrings with usage examples (NFR2)
- [ ] Unit tests in `tests/unit/test_apply_imputation.py`:
  - Test each strategy can be applied
  - Test validation catches NaN/Inf
  - Test error handling for invalid strategy names
  - Test metadata file is created
- [ ] Integration test in `tests/integration/test_imputation_pipeline.py`:
  - End-to-end test: raw data → imputation → validated output
  - Test round-trip: save and reload imputed data
- [ ] Test coverage ≥ 80%

---

## EPIC-LEVEL ACCEPTANCE CRITERIA

All stories (2.1-2.7) must be complete before Epic 2 is considered done:

- [ ] **Story 2.1**: Abstract base class with clear interface contract ✅
- [ ] **Story 2.2**: Basis Embedding implemented and tested ✅
- [ ] **Story 2.3**: Amplitude Embedding implemented and tested ✅
- [ ] **Story 2.4**: Angle Encoding implemented and tested ✅
- [ ] **Story 2.5**: Density Matrix Embedding implemented and tested ✅
- [ ] **Story 2.6**: Graph/Cycle Encoding implemented and tested ✅
- [ ] **Story 2.7**: Utility script implemented and tested ✅

**Epic-Level Validation**:
- [ ] All 5 strategies produce valid output on full dataset (11,581 events)
- [ ] All strategies have ≥ 80% test coverage
- [ ] All code follows architecture.md section 6 coding standards
- [ ] All docstrings are comprehensive (NFR2)
- [ ] Integration test passes: raw data → each imputation → valid output
- [ ] README.md updated with imputation usage examples
- [ ] Git commits follow established pattern from Epic 1

---

## DEPENDENCIES & SEQUENCING

### Critical Path

```
Story 2.1 (Base Class)
    ↓
Stories 2.2-2.6 (5 Imputation Strategies) ← Can be done in parallel
    ↓
Story 2.7 (Utility Script)
```

**Recommended Order**:
1. **Story 2.1** (CRITICAL - blocks everything else)
2. **Story 2.2** (Basis Embedding - simplest, good starting point)
3. **Story 2.3** (Amplitude Embedding - moderate complexity)
4. **Story 2.4** (Angle Encoding - moderate complexity)
5. **Story 2.6** (Graph/Cycle - requires FFT understanding)
6. **Story 2.5** (Density Matrix - most complex, save for last)
7. **Story 2.7** (Utility Script - integrates all strategies)

### External Dependencies

- **NumPy**: Required for all strategies (FFT, matrix operations)
- **SciPy**: May be needed for Story 2.5 (density matrix eigenvalues)
- **Pandas**: Already used in Epic 1
- No new external dependencies required (all in standard data science stack)

---

## TESTING STRATEGY

### Test Coverage Targets

- **Overall Epic 2 Coverage**: ≥ 80% function coverage
- **Per-Story Coverage**: ≥ 80% for each story
- **Critical Modules**: Base class and utility script should aim for 90%+

### Test Structure

```
tests/
├── unit/
│   ├── test_base_imputer.py          # Story 2.1
│   ├── test_basis_embedding.py       # Story 2.2
│   ├── test_amplitude_embedding.py   # Story 2.3
│   ├── test_angle_encoding.py        # Story 2.4
│   ├── test_density_matrix.py        # Story 2.5
│   ├── test_graph_cycle.py           # Story 2.6
│   └── test_apply_imputation.py      # Story 2.7
├── integration/
│   └── test_imputation_pipeline.py   # End-to-end imputation tests
└── fixtures/
    └── sample_valid_dataset.csv      # Reuse from Epic 1
```

### Test Patterns to Follow

Use patterns established in `tests/unit/test_data_loader.py`:
- Fixtures for reusable test data
- Comprehensive docstrings explaining what each test validates
- Test both success and failure cases
- Use `pytest.raises()` for exception testing
- Use `pytest.mark.unit` marker for unit tests

### Example Test Structure

```python
import pytest
import numpy as np
from src.imputation.basis_embedding import BasisEmbedding
from src.data_loader import load_dataset

@pytest.fixture
def sample_data(valid_dataset_path):
    """Load sample dataset for testing."""
    return load_dataset(valid_dataset_path)

@pytest.mark.unit
def test_basis_embedding_fit_transform(sample_data):
    """
    Test that BasisEmbedding.fit_transform produces expected output.

    Verifies:
    - Output is numpy array
    - Output shape is (n_samples, 78)
    - No NaN or Inf values
    """
    imputer = BasisEmbedding()
    features = imputer.fit_transform(sample_data)

    assert isinstance(features, np.ndarray)
    assert features.shape == (len(sample_data), 78)
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()

@pytest.mark.unit
def test_basis_embedding_not_fitted_error(sample_data):
    """
    Test that transform() raises error when called before fit().
    """
    imputer = BasisEmbedding()

    with pytest.raises(RuntimeError) as exc_info:
        imputer.transform(sample_data)

    assert "fit" in str(exc_info.value).lower()
```

---

## DOCUMENTATION REQUIREMENTS (NFR2)

### Comprehensive Docstrings Required

Every class, method, and function must have docstrings including:
1. **Brief Description**: What does this do?
2. **Parameters**: Type hints + descriptions
3. **Returns**: Type + description
4. **Raises**: Exceptions that can be raised
5. **Examples**: Usage examples with expected output
6. **Notes** (for complex concepts): Mathematical background, quantum-inspired concepts explained in accessible language

### Inline Comments Required

For complex algorithms (especially Stories 2.4-2.6):
- Explain **why** each step is done (not just what)
- Reference quantum mechanics concepts when relevant
- Break down mathematical operations into understandable chunks

### Example: Good Documentation

```python
def _compute_density_matrix(self, qv_vector: np.ndarray) -> np.ndarray:
    """
    Compute density matrix for a single quantum state.

    In quantum mechanics, a density matrix ρ represents a mixed state
    (statistical ensemble of pure states). For our binary quantum states:

    - Diagonal elements: Probability of each position being active
    - Off-diagonal elements: Quantum coherence between position pairs

    The density matrix satisfies:
    1. Hermitian: ρ = ρ† (conjugate transpose)
    2. Positive semi-definite: all eigenvalues ≥ 0
    3. Normalized: Tr(ρ) = 1 (trace equals 1)

    Args:
        qv_vector: Binary vector of shape (39,) with exactly 5 ones

    Returns:
        Density matrix of shape (39, 39) satisfying quantum properties

    Examples:
        >>> qv = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, ...])  # 5 ones
        >>> rho = self._compute_density_matrix(qv)
        >>> rho.shape
        (39, 39)
        >>> np.allclose(rho, rho.T.conj())  # Check Hermitian
        True
        >>> np.trace(rho)  # Check normalized
        1.0

    Notes:
        This implementation uses a simplified density matrix where
        coherences are equal for all active position pairs. More
        sophisticated approaches could weight coherences by position
        frequencies or distances on the C₃₉ ring.
    """
    n_positions = len(qv_vector)
    active_positions = np.where(qv_vector == 1)[0]
    n_active = len(active_positions)

    # Initialize 39x39 matrix with zeros
    rho = np.zeros((n_positions, n_positions), dtype=np.float64)

    # Set diagonal elements: probability 1/n_active for each active position
    # This ensures Tr(ρ) = n_active * (1/n_active) = 1
    for i in active_positions:
        rho[i, i] = 1.0 / n_active

    # Set off-diagonal elements: coherence between active position pairs
    # Coherence magnitude is 1/(n_active²) to maintain normalization
    for i in active_positions:
        for j in active_positions:
            if i != j:
                rho[i, j] = 1.0 / (n_active ** 2)

    return rho
```

---

## REFERENCE MATERIALS

### Epic 1 Assets to Leverage

1. **Data Loading**: `src/data_loader.py`
   - Use `load_dataset()` to load raw data
   - Use `validate_dataset_structure()` patterns for input validation

2. **Testing Patterns**: `tests/unit/test_data_loader.py`
   - 365 lines of well-structured test examples
   - Fixture patterns, docstrings, comprehensive coverage

3. **EDA Insights**: `notebooks/1_EDA.ipynb`
   - Position frequency analysis informs Basis Embedding
   - Co-occurrence patterns inform Density Matrix Embedding
   - Temporal stability suggests strategies should learn from training data

4. **Configuration Management**: `src/config.py`
   - Use centralized path constants
   - Add new paths if needed (e.g., `IMPUTED_DATA_DIR`)

### Architecture Documentation

- **Section 4**: Project structure (`src/imputation/` directory)
- **Section 6**: Coding standards (type hints, logging, modularity)
- **Section 3**: Tech stack (NumPy 1.26+, SciPy if needed)

### PRD Context

- **Section 1**: Goals (G1: flexible imputation framework)
- **Section 2**: Requirements (NFR2: non-programmer friendly, NFR3: modularity)
- **Section 3**: Epic 2 full specification (PRD lines 84-124)

---

## SUCCESS METRICS

### Code Quality Metrics

- [ ] Test coverage ≥ 80% across all Epic 2 modules
- [ ] All functions have comprehensive docstrings with examples
- [ ] Zero linting errors (if linter configured)
- [ ] All type hints present in function signatures

### Functional Metrics

- [ ] All 5 imputation strategies successfully process full dataset (11,581 events)
- [ ] Output validation passes (no NaN, no Inf)
- [ ] Execution time logged for each strategy (informs RunPod decisions in Epic 5)

### Documentation Metrics

- [ ] Every complex algorithm has inline comments explaining "why"
- [ ] README.md updated with "Using Imputation Framework" section
- [ ] At least 1 example notebook demonstrating imputation usage

---

## EXPECTED DELIVERABLES

Upon Epic 2 completion, the following files should exist:

### Source Code
- `src/imputation/base_imputer.py` (Story 2.1)
- `src/imputation/basis_embedding.py` (Story 2.2)
- `src/imputation/amplitude_embedding.py` (Story 2.3)
- `src/imputation/angle_encoding.py` (Story 2.4)
- `src/imputation/density_matrix.py` (Story 2.5)
- `src/imputation/graph_cycle_encoding.py` (Story 2.6)
- `src/imputation/apply_imputation.py` (Story 2.7)
- `src/imputation/__init__.py` (exports all strategies)

### Tests
- `tests/unit/test_base_imputer.py`
- `tests/unit/test_basis_embedding.py`
- `tests/unit/test_amplitude_embedding.py`
- `tests/unit/test_angle_encoding.py`
- `tests/unit/test_density_matrix.py`
- `tests/unit/test_graph_cycle.py`
- `tests/unit/test_apply_imputation.py`
- `tests/integration/test_imputation_pipeline.py`

### Documentation
- Updated `README.md` with imputation usage section
- Optional: `notebooks/2_Imputation_Demo.ipynb` demonstrating all 5 strategies

### Generated Artifacts (from testing)
- `data/processed/imputed_basis_embedding_test.parquet` (from integration tests)
- Corresponding `.meta.json` metadata files

### Git Commits
- Minimum 7 commits (one per story)
- Commit messages follow Epic 1 pattern: descriptive, includes story number

---

## POTENTIAL CHALLENGES & MITIGATION

### Challenge 1: Quantum Concepts Complexity

**Risk**: Quantum-inspired mathematical concepts may be difficult to implement correctly.

**Mitigation**:
- Start with simplest strategy (Basis Embedding) to establish patterns
- Reference EDA insights to validate feature engineering makes sense
- Write tests first (TDD) to clarify expected behavior
- Use inline comments extensively to explain quantum concepts

### Challenge 2: Output Dimensionality Inconsistency

**Risk**: Different strategies produce different output dimensions, complicating downstream use.

**Mitigation**:
- **Story 2.1 (Base Class)** must clearly document that output dimensions vary by strategy
- Each strategy docstring must explicitly state output dimensions
- Metadata JSON includes `output_shape` for traceability
- Epic 3 (ranking models) will handle variable input dimensions

### Challenge 3: Computational Performance

**Risk**: Some strategies (especially Density Matrix with eigenvalue computation) may be slow.

**Mitigation**:
- Log execution time in Story 2.7 utility script
- Use efficient NumPy operations (vectorized, not loops)
- Consider sparse matrix representations where applicable
- Performance profiling deferred to Epic 5 (experiments track timing)

### Challenge 4: Test Complexity for Mathematical Operations

**Risk**: Difficult to write tests for complex mathematical transformations (DFT, density matrices).

**Mitigation**:
- Test fundamental properties (e.g., density matrix is Hermitian)
- Use small hand-crafted examples with known outputs
- Test invariants (e.g., Tr(ρ) = 1, energy conservation in DFT)
- Don't test exact values, test properties and ranges

---

## COMMUNICATION PROTOCOL

### When to Ask PO for Clarification

1. **Ambiguous Requirements**: If imputation strategy specification is unclear
2. **Output Format Decisions**: If uncertain about feature engineering approach
3. **Scope Questions**: If unsure whether a feature is MVP or post-MVP

### When to Proceed Autonomously

1. **Implementation Details**: Algorithm implementation choices within acceptance criteria
2. **Test Structure**: How to organize tests (follow Epic 1 patterns)
3. **Code Organization**: Module structure within `src/imputation/`

### Progress Reporting

After each story completion:
1. Run `pytest` and report test coverage
2. Demonstrate strategy on sample dataset
3. Commit code with descriptive message
4. Report any challenges encountered

---

## FINAL CHECKLIST FOR EPIC 2 COMPLETION

Before declaring Epic 2 complete, verify:

- [ ] All 7 stories (2.1-2.7) meet acceptance criteria
- [ ] `pytest` runs successfully with ≥ 80% coverage for Epic 2 modules
- [ ] All 5 imputation strategies successfully process full dataset
- [ ] README.md updated with imputation usage instructions
- [ ] All code follows architecture.md coding standards
- [ ] Git commits are descriptive and reference story numbers
- [ ] No critical TODOs or FIXMEs left in code
- [ ] Can successfully run: `python src/imputation/apply_imputation.py --strategy basis_embedding --input data/raw/c5_Matrix.csv --output data/processed/test_output.parquet`
- [ ] Epic 2 completion validated by PO (run `*execute-checklist-po` if available)

---

## APPENDIX: Quick Reference

### Key File Paths
- Raw data: `data/raw/c5_Matrix.csv`
- Processed data: `data/processed/`
- Source code: `src/imputation/`
- Tests: `tests/unit/`, `tests/integration/`

### Key Functions to Import
```python
from src.data_loader import load_dataset, validate_dataset_structure
from src.config import DATA_RAW, DATA_PROCESSED
```

### Pytest Commands
```bash
pytest                                    # Run all tests
pytest tests/unit/test_basis_embedding.py # Run specific test file
pytest -v                                 # Verbose output
pytest --cov=src/imputation               # Coverage report for imputation module
pytest -m unit                            # Run only unit tests
```

### Git Workflow
```bash
git add src/imputation/basis_embedding.py tests/unit/test_basis_embedding.py
git commit -m "Implement Basis Embedding imputation strategy (Story 2.2)

- Created BasisEmbedding class inheriting from BaseImputer
- Implements one-hot encoding + position frequency features
- Output shape: (n_samples, 78)
- Unit tests: 8 test cases, 85% coverage
- All tests passing

Story 2.2 acceptance criteria met.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

**END OF HANDOFF DOCUMENT**

This document provides all specifications needed to autonomously implement Epic 2. For questions or clarifications, contact Sarah (PO).

**Next PO Review Point**: After Story 2.1 completion (critical to validate base class design before proceeding with concrete strategies).

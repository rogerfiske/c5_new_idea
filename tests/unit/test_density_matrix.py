"""
Unit Tests for Density Matrix Embedding Imputation Strategy

This test suite validates the Density Matrix Embedding imputation method, which
represents quantum states as density matrices (mixed states).

Test Coverage:
- Initialization with different configurations
- Quantum properties (Hermitian, positive semi-definite, normalized)
- Diagonal elements
- Eigenvalue computation
- Purity Tr(ρ²)
- Von Neumann entropy
- Output shapes and types
- Feature name generation
- Integration tests

Author: BMad Dev Agent (James)
Date: 2025-10-13
Story: Epic 2, Story 2.5 - Density Matrix Embedding Strategy
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.imputation.density_matrix import DensityMatrixEmbedding
from src.data_loader import load_dataset


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def fixture_dir():
    """Provide path to test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def valid_dataset_path(fixture_dir):
    """Provide path to valid sample dataset."""
    return fixture_dir / "sample_valid_dataset.csv"


@pytest.fixture
def valid_dataset(valid_dataset_path):
    """Load valid sample dataset."""
    return pd.read_csv(valid_dataset_path)


# ============================================================================
# Tests for Initialization
# ============================================================================

def test_density_matrix_initialization_default():
    """
    Test DensityMatrixEmbedding initializes with correct default parameters.

    Verifies:
    - Default name is "density_matrix"
    - n_eigenvalues defaults to 5
    - include_diagonal defaults to True
    - include_purity defaults to True
    - include_entropy defaults to False
    - Not fitted initially
    - Config stored correctly
    """
    imputer = DensityMatrixEmbedding()

    assert imputer.name == "density_matrix"
    assert imputer.n_eigenvalues == 5
    assert imputer.include_diagonal is True
    assert imputer.include_purity is True
    assert imputer.include_entropy is False
    assert imputer.fitted_ is False
    assert imputer.config == {
        "n_eigenvalues": 5,
        "include_diagonal": True,
        "include_purity": True,
        "include_entropy": False
    }


def test_density_matrix_initialization_custom():
    """
    Test DensityMatrixEmbedding can be initialized with custom parameters.

    Verifies:
    - Custom name is stored
    - n_eigenvalues can be customized
    - include_diagonal can be set to False
    - include_purity can be set to False
    - include_entropy can be set to True
    - Config reflects custom settings
    """
    imputer = DensityMatrixEmbedding(
        name="custom_density",
        n_eigenvalues=10,
        include_diagonal=False,
        include_purity=False,
        include_entropy=True
    )

    assert imputer.name == "custom_density"
    assert imputer.n_eigenvalues == 10
    assert imputer.include_diagonal is False
    assert imputer.include_purity is False
    assert imputer.include_entropy is True


def test_invalid_n_eigenvalues_raises_error():
    """
    Test that invalid n_eigenvalues values raise ValueError.

    Verifies:
    - n_eigenvalues = 0 raises error
    - n_eigenvalues = 40 raises error
    - n_eigenvalues = -1 raises error
    """
    with pytest.raises(ValueError):
        DensityMatrixEmbedding(n_eigenvalues=0)

    with pytest.raises(ValueError):
        DensityMatrixEmbedding(n_eigenvalues=40)

    with pytest.raises(ValueError):
        DensityMatrixEmbedding(n_eigenvalues=-1)


# ============================================================================
# Tests for _fit() Method
# ============================================================================

def test_fit_no_learning_required(valid_dataset):
    """
    Test that fit() completes successfully even though no learning occurs.

    Verifies:
    - fit() runs without error
    - fitted_ flag is set
    - No parameters are learned (deterministic construction)
    """
    imputer = DensityMatrixEmbedding()

    imputer.fit(valid_dataset)

    assert imputer.fitted_ is True


# ============================================================================
# Tests for _transform() - Output Shapes
# ============================================================================

def test_transform_output_shape_default(valid_dataset):
    """
    Test transform() with default settings.

    Verifies:
    - Output shape is (n_samples, 45)
    - 45 = 39 (diagonal) + 5 (eigenvalues) + 1 (purity)
    """
    imputer = DensityMatrixEmbedding()
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    expected_features = 39 + 5 + 1  # diagonal + eigenvalues + purity
    assert features.shape == (len(valid_dataset), expected_features)


def test_transform_output_shape_with_entropy(valid_dataset):
    """
    Test transform() with entropy included.

    Verifies:
    - Output shape is (n_samples, 46)
    - 46 = 39 (diagonal) + 5 (eigenvalues) + 1 (purity) + 1 (entropy)
    """
    imputer = DensityMatrixEmbedding(include_entropy=True)
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    expected_features = 39 + 5 + 1 + 1
    assert features.shape == (len(valid_dataset), expected_features)


def test_transform_output_shape_minimal(valid_dataset):
    """
    Test transform() with minimal settings (only eigenvalues).

    Verifies:
    - Output shape is (n_samples, 5)
    """
    imputer = DensityMatrixEmbedding(
        include_diagonal=False,
        include_purity=False,
        n_eigenvalues=5
    )
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    assert features.shape == (len(valid_dataset), 5)


def test_transform_output_shape_all_eigenvalues(valid_dataset):
    """
    Test transform() with all 39 eigenvalues.

    Verifies:
    - Output shape is (n_samples, 80)
    - 80 = 39 (diagonal) + 39 (eigenvalues) + 1 (purity) + 1 (entropy)
    """
    imputer = DensityMatrixEmbedding(
        n_eigenvalues=39,
        include_entropy=True
    )
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    expected_features = 39 + 39 + 1 + 1
    assert features.shape == (len(valid_dataset), expected_features)


# ============================================================================
# Tests for _transform() - Output Types and Values
# ============================================================================

def test_transform_output_type(valid_dataset):
    """
    Test that transform() returns numpy array with correct dtype.

    Verifies:
    - Returns numpy.ndarray
    - Data type is float
    """
    imputer = DensityMatrixEmbedding()
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    assert isinstance(features, np.ndarray)
    assert np.issubdtype(features.dtype, np.floating)


def test_transform_no_nan_or_inf(valid_dataset):
    """
    Test that transform() output contains no NaN or Inf values.

    Verifies:
    - No NaN values
    - No Inf values
    - All values are finite
    """
    imputer = DensityMatrixEmbedding()
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    assert not np.isnan(features).any(), "Output contains NaN values"
    assert not np.isinf(features).any(), "Output contains Inf values"
    assert np.all(np.isfinite(features)), "Output contains non-finite values"


# ============================================================================
# Tests for Diagonal Elements
# ============================================================================

def test_diagonal_elements_correct_values(valid_dataset):
    """
    Test that diagonal elements are correctly computed.

    Verifies:
    - Diagonal elements are 1/5 for active positions
    - Diagonal elements are 0 for inactive positions
    - Values match QV columns scaled by 1/5
    """
    imputer = DensityMatrixEmbedding(
        include_purity=False,
        n_eigenvalues=5
    )
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    # First 39 columns are diagonal elements
    diagonal = features[:, :39]

    # Extract QV columns
    qv_columns = [f'QV_{i}' for i in range(1, 40)]
    qv_data = valid_dataset[qv_columns].values

    # Expected diagonal: qv_data / 5
    expected_diagonal = qv_data.astype(np.float64) / 5.0

    np.testing.assert_array_almost_equal(diagonal, expected_diagonal, decimal=10)


def test_diagonal_elements_value_range(valid_dataset):
    """
    Test that diagonal elements are in valid range [0, 0.2].

    Verifies:
    - All values >= 0
    - All values <= 0.2 (which is 1/5)
    """
    imputer = DensityMatrixEmbedding()
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    diagonal = features[:, :39]

    assert np.all(diagonal >= 0)
    assert np.all(diagonal <= 0.2)


# ============================================================================
# Tests for Eigenvalues
# ============================================================================

def test_eigenvalues_correct_structure(valid_dataset):
    """
    Test that eigenvalues have correct structure.

    Verifies:
    - First 5 eigenvalues are 1/5 = 0.2
    - Remaining eigenvalues are 0
    - This matches equal superposition over 5 positions
    """
    imputer = DensityMatrixEmbedding(
        include_diagonal=False,
        include_purity=False,
        n_eigenvalues=39  # Get all eigenvalues
    )
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    # First 5 eigenvalues should be 0.2
    np.testing.assert_array_almost_equal(features[:, :5], 0.2, decimal=10)

    # Remaining 34 eigenvalues should be 0
    np.testing.assert_array_almost_equal(features[:, 5:], 0.0, decimal=10)


def test_eigenvalues_nonnegative(valid_dataset):
    """
    Test that all eigenvalues are non-negative.

    Verifies:
    - Eigenvalues >= 0 (positive semi-definite requirement)
    """
    imputer = DensityMatrixEmbedding(
        include_diagonal=False,
        include_purity=False,
        n_eigenvalues=39
    )
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    assert np.all(features >= 0), "All eigenvalues must be non-negative"


def test_eigenvalues_sum_to_one(valid_dataset):
    """
    Test that eigenvalues sum to 1 (trace = 1 requirement).

    Verifies:
    - Sum of all eigenvalues = 1.0
    - This validates trace normalization
    """
    imputer = DensityMatrixEmbedding(
        include_diagonal=False,
        include_purity=False,
        n_eigenvalues=39
    )
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    eigenvalue_sums = features.sum(axis=1)
    np.testing.assert_array_almost_equal(eigenvalue_sums, 1.0, decimal=10)


# ============================================================================
# Tests for Purity
# ============================================================================

def test_purity_correct_value(valid_dataset):
    """
    Test that purity is correctly computed.

    Verifies:
    - Purity = 0.2 (which is 1/5 for equal superposition)
    - This is constant for all samples
    """
    imputer = DensityMatrixEmbedding(
        include_diagonal=False,
        n_eigenvalues=5
    )
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    # Last column is purity
    purity = features[:, -1]

    np.testing.assert_array_almost_equal(purity, 0.2, decimal=10)


def test_purity_value_range(valid_dataset):
    """
    Test that purity is in valid range [0, 1].

    Verifies:
    - 0 <= purity <= 1
    - purity = 1 for pure state, < 1 for mixed state
    """
    imputer = DensityMatrixEmbedding()
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    # Purity is second-to-last column (before eigenvalues if no entropy)
    purity = features[:, 44]  # Position 44 in default config

    assert np.all(purity >= 0) and np.all(purity <= 1)


# ============================================================================
# Tests for Von Neumann Entropy
# ============================================================================

def test_entropy_correct_value(valid_dataset):
    """
    Test that von Neumann entropy is correctly computed.

    Verifies:
    - Entropy = log(5) ≈ 1.609 for equal superposition
    - This is constant for all samples
    """
    imputer = DensityMatrixEmbedding(
        include_diagonal=False,
        n_eigenvalues=5,
        include_purity=False,
        include_entropy=True
    )
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    # Last column is entropy
    entropy = features[:, -1]

    expected_entropy = np.log(5)  # log(5) ≈ 1.609
    np.testing.assert_array_almost_equal(entropy, expected_entropy, decimal=6)


def test_entropy_nonnegative(valid_dataset):
    """
    Test that entropy is non-negative.

    Verifies:
    - Entropy >= 0 (always for physical states)
    """
    imputer = DensityMatrixEmbedding(include_entropy=True)
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    # Entropy is last column
    entropy = features[:, -1]

    assert np.all(entropy >= 0), "Entropy must be non-negative"


# ============================================================================
# Tests for Quantum Properties
# ============================================================================

def test_verify_quantum_properties_trace(valid_dataset):
    """
    Test that density matrix has trace = 1.

    Verifies:
    - Tr(ρ) = Σᵢ ρᵢᵢ = 1 (normalization)
    """
    imputer = DensityMatrixEmbedding()
    imputer.fit(valid_dataset)

    qv_columns = [f'QV_{i}' for i in range(1, 40)]
    qv_data = valid_dataset[qv_columns].values

    props = imputer.verify_quantum_properties(qv_data)

    np.testing.assert_almost_equal(props["trace"], 1.0, decimal=10)


def test_verify_quantum_properties_purity(valid_dataset):
    """
    Test that density matrix has correct purity.

    Verifies:
    - Tr(ρ²) = 0.2 for our equal superposition
    """
    imputer = DensityMatrixEmbedding()
    imputer.fit(valid_dataset)

    qv_columns = [f'QV_{i}' for i in range(1, 40)]
    qv_data = valid_dataset[qv_columns].values

    props = imputer.verify_quantum_properties(qv_data)

    np.testing.assert_almost_equal(props["purity"], 0.2, decimal=10)


def test_verify_quantum_properties_hermitian(valid_dataset):
    """
    Test that density matrix is Hermitian.

    Verifies:
    - ρ = ρ† (by construction for real symmetric matrices)
    """
    imputer = DensityMatrixEmbedding()
    imputer.fit(valid_dataset)

    qv_columns = [f'QV_{i}' for i in range(1, 40)]
    qv_data = valid_dataset[qv_columns].values

    props = imputer.verify_quantum_properties(qv_data)

    assert props["hermitian"] is True


def test_verify_quantum_properties_positive_semidefinite(valid_dataset):
    """
    Test that density matrix is positive semi-definite.

    Verifies:
    - All eigenvalues >= 0
    """
    imputer = DensityMatrixEmbedding()
    imputer.fit(valid_dataset)

    qv_columns = [f'QV_{i}' for i in range(1, 40)]
    qv_data = valid_dataset[qv_columns].values

    props = imputer.verify_quantum_properties(qv_data)

    assert props["positive_semidefinite"] is True


# ============================================================================
# Tests for get_feature_names()
# ============================================================================

def test_get_feature_names_default(valid_dataset):
    """
    Test get_feature_names() with default settings.

    Verifies:
    - Returns 45 names
    - Names follow expected pattern
    """
    imputer = DensityMatrixEmbedding()
    imputer.fit(valid_dataset)
    names = imputer.get_feature_names()

    assert len(names) == 45

    # Check diagonal names
    assert names[0] == 'rho_11'
    assert names[38] == 'rho_3939'

    # Check eigenvalue names
    assert names[39] == 'eigenvalue_1'
    assert names[43] == 'eigenvalue_5'

    # Check purity
    assert names[44] == 'purity'


def test_get_feature_names_with_entropy(valid_dataset):
    """
    Test get_feature_names() with entropy included.

    Verifies:
    - Returns 46 names
    - Entropy name is included
    """
    imputer = DensityMatrixEmbedding(include_entropy=True)
    imputer.fit(valid_dataset)
    names = imputer.get_feature_names()

    assert len(names) == 46
    assert names[-1] == 'von_neumann_entropy'


def test_get_feature_names_minimal(valid_dataset):
    """
    Test get_feature_names() with minimal settings.

    Verifies:
    - Returns 5 names (only eigenvalues)
    """
    imputer = DensityMatrixEmbedding(
        include_diagonal=False,
        include_purity=False,
        n_eigenvalues=5
    )
    imputer.fit(valid_dataset)
    names = imputer.get_feature_names()

    assert len(names) == 5
    assert all('eigenvalue' in name for name in names)


# ============================================================================
# Tests for fit_transform()
# ============================================================================

def test_fit_transform_works(valid_dataset):
    """
    Test that fit_transform() works correctly.

    Verifies:
    - Can fit and transform in one call
    - Returns correct output shape
    - Sets fitted_ flag
    """
    imputer = DensityMatrixEmbedding()
    features = imputer.fit_transform(valid_dataset)

    assert imputer.fitted_ is True
    assert features.shape == (len(valid_dataset), 45)


def test_fit_transform_equivalent_to_separate_calls(valid_dataset):
    """
    Test that fit_transform(X) produces same output as fit(X).transform(X).

    Verifies:
    - Both workflows produce identical results
    """
    # Approach 1: fit_transform
    imputer1 = DensityMatrixEmbedding()
    features1 = imputer1.fit_transform(valid_dataset)

    # Approach 2: fit then transform
    imputer2 = DensityMatrixEmbedding()
    imputer2.fit(valid_dataset)
    features2 = imputer2.transform(valid_dataset)

    # Should be identical
    np.testing.assert_array_equal(features1, features2)


# ============================================================================
# Tests for Error Handling
# ============================================================================

def test_transform_before_fit_raises_error(valid_dataset):
    """
    Test that calling transform() before fit() raises RuntimeError.

    Verifies:
    - RuntimeError is raised
    - Error message mentions fitting
    """
    imputer = DensityMatrixEmbedding()

    with pytest.raises(RuntimeError) as exc_info:
        imputer.transform(valid_dataset)

    assert "fit" in str(exc_info.value).lower()


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_workflow_all_configurations(valid_dataset):
    """
    Integration test: Test multiple configuration combinations.

    Verifies:
    - Various configurations work end-to-end
    - Each produces expected output shape
    """
    configs = [
        (True, True, False, 5, 45),     # Default
        (True, True, True, 5, 46),      # With entropy
        (False, False, False, 5, 5),    # Only eigenvalues
        (True, True, True, 39, 80),     # All features
    ]

    for diag, purity, entropy, n_eig, expected_features in configs:
        imputer = DensityMatrixEmbedding(
            include_diagonal=diag,
            include_purity=purity,
            include_entropy=entropy,
            n_eigenvalues=n_eig
        )
        features = imputer.fit_transform(valid_dataset)

        assert features.shape == (len(valid_dataset), expected_features)
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()


def test_consistency_across_multiple_transforms(valid_dataset):
    """
    Test that multiple transform calls produce identical results.

    Verifies:
    - Transform is deterministic (no randomness)
    - Can be called multiple times with same result
    """
    imputer = DensityMatrixEmbedding()
    imputer.fit(valid_dataset)

    # Transform multiple times
    features1 = imputer.transform(valid_dataset)
    features2 = imputer.transform(valid_dataset)
    features3 = imputer.transform(valid_dataset)

    # All should be identical
    np.testing.assert_array_equal(features1, features2)
    np.testing.assert_array_equal(features2, features3)


def test_works_with_single_sample(valid_dataset):
    """
    Test that imputer works with single sample (edge case).

    Verifies:
    - Can transform single row
    - Output shape is (1, n_features)
    """
    imputer = DensityMatrixEmbedding()
    imputer.fit(valid_dataset)

    # Transform single row
    single_sample = valid_dataset.iloc[[0]]
    features = imputer.transform(single_sample)

    assert features.shape == (1, 45)
    assert not np.isnan(features).any()

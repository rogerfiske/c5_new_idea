"""
Unit Tests for Basis Embedding Imputation Strategy

This test suite validates the Basis Embedding imputation method, which maps
binary quantum state vectors to computational basis state representations.

Test Coverage:
- Fit learns position frequencies correctly
- Transform produces correct output shape
- Output has no NaN/Inf values
- Works with and without frequency features
- Error handling for unfitted imputer
- Integration with base class validation

Author: BMad Dev Agent (James)
Date: 2025-10-13
Story: Epic 2, Story 2.2 - Basis Embedding Strategy
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.imputation.basis_embedding import BasisEmbedding
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

def test_basis_embedding_initialization_default():
    """
    Test BasisEmbedding initializes with correct default parameters.

    Verifies:
    - Default name is "basis_embedding"
    - include_frequency_features defaults to True
    - Not fitted initially
    - Config stored correctly
    """
    imputer = BasisEmbedding()

    assert imputer.name == "basis_embedding"
    assert imputer.include_frequency_features is True
    assert imputer.fitted_ is False
    assert imputer.config == {"include_frequency_features": True}
    assert imputer.position_frequencies_ is None


def test_basis_embedding_initialization_custom():
    """
    Test BasisEmbedding can be initialized with custom parameters.

    Verifies:
    - Custom name is stored
    - include_frequency_features can be set to False
    - Config reflects custom settings
    """
    imputer = BasisEmbedding(
        name="custom_basis",
        include_frequency_features=False
    )

    assert imputer.name == "custom_basis"
    assert imputer.include_frequency_features is False
    assert imputer.config == {"include_frequency_features": False}


# ============================================================================
# Tests for _fit() Method
# ============================================================================

def test_fit_learns_position_frequencies(valid_dataset):
    """
    Test that fit() correctly learns position frequencies from training data.

    Verifies:
    - position_frequencies_ is set after fit()
    - Has correct shape (39,)
    - Values are in valid range [0, 1]
    - Frequencies sum to 5 (since exactly 5 positions active per row)
    """
    imputer = BasisEmbedding()
    imputer.fit(valid_dataset)

    # Should have learned frequencies
    assert imputer.position_frequencies_ is not None

    # Should have 39 frequencies (one per position)
    assert imputer.position_frequencies_.shape == (39,)

    # Frequencies should be between 0 and 1
    assert np.all(imputer.position_frequencies_ >= 0)
    assert np.all(imputer.position_frequencies_ <= 1)

    # Sum of frequencies should equal 5 (each row has 5 active positions)
    # Sum = (total active positions) / n_samples = (5 * n_samples) / n_samples = 5
    expected_sum = 5.0
    actual_sum = imputer.position_frequencies_.sum()
    np.testing.assert_almost_equal(actual_sum, expected_sum, decimal=10)


def test_fit_frequencies_match_manual_calculation(valid_dataset):
    """
    Test that learned frequencies match manual calculation.

    Verifies:
    - Frequencies are calculated correctly
    - Formula: frequency[i] = (count of QV_i=1) / n_samples
    """
    imputer = BasisEmbedding()
    imputer.fit(valid_dataset)

    # Manually calculate frequencies for QV_1
    qv_1_count = valid_dataset['QV_1'].sum()
    expected_freq_1 = qv_1_count / len(valid_dataset)
    actual_freq_1 = imputer.position_frequencies_[0]

    np.testing.assert_almost_equal(actual_freq_1, expected_freq_1, decimal=10)

    # Manually calculate for QV_10
    qv_10_count = valid_dataset['QV_10'].sum()
    expected_freq_10 = qv_10_count / len(valid_dataset)
    actual_freq_10 = imputer.position_frequencies_[9]

    np.testing.assert_almost_equal(actual_freq_10, expected_freq_10, decimal=10)


# ============================================================================
# Tests for _transform() Method
# ============================================================================

def test_transform_output_shape_with_frequency_features(valid_dataset):
    """
    Test that transform() produces correct output shape with frequency features.

    Verifies:
    - Output shape is (n_samples, 78) with default settings
    - 78 = 39 (one-hot) + 39 (frequency)
    """
    imputer = BasisEmbedding(include_frequency_features=True)
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    assert features.shape == (len(valid_dataset), 78)


def test_transform_output_shape_without_frequency_features(valid_dataset):
    """
    Test that transform() produces correct output shape without frequency features.

    Verifies:
    - Output shape is (n_samples, 39) when include_frequency_features=False
    """
    imputer = BasisEmbedding(include_frequency_features=False)
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    assert features.shape == (len(valid_dataset), 39)


def test_transform_output_type(valid_dataset):
    """
    Test that transform() returns numpy array with correct dtype.

    Verifies:
    - Returns numpy.ndarray
    - Data type is float (not int, even though inputs are binary)
    """
    imputer = BasisEmbedding()
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
    imputer = BasisEmbedding()
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    assert not np.isnan(features).any(), "Output contains NaN values"
    assert not np.isinf(features).any(), "Output contains Inf values"
    assert np.all(np.isfinite(features)), "Output contains non-finite values"


def test_transform_onehot_features_match_input(valid_dataset):
    """
    Test that one-hot features (first 39 columns) match input QV columns.

    Verifies:
    - First 39 columns of output are identical to QV_1-39 input
    - This is true for both with/without frequency features
    """
    imputer = BasisEmbedding()
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    # Extract QV columns from input
    qv_columns = [f'QV_{i}' for i in range(1, 40)]
    qv_data = valid_dataset[qv_columns].values

    # First 39 columns should match input
    onehot_features = features[:, :39]
    np.testing.assert_array_equal(onehot_features, qv_data)


def test_transform_frequency_features_correct(valid_dataset):
    """
    Test that frequency features (columns 39-77) are correctly weighted.

    Verifies:
    - Frequency features = one-hot × learned_frequencies
    - Formula is applied correctly
    """
    imputer = BasisEmbedding(include_frequency_features=True)
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    # Extract QV columns from input
    qv_columns = [f'QV_{i}' for i in range(1, 40)]
    qv_data = valid_dataset[qv_columns].values

    # Expected frequency features: qv_data × position_frequencies
    expected_freq_features = qv_data * imputer.position_frequencies_

    # Actual frequency features (columns 39-77)
    actual_freq_features = features[:, 39:78]

    np.testing.assert_array_almost_equal(
        actual_freq_features,
        expected_freq_features,
        decimal=10
    )


def test_transform_without_frequency_only_onehot(valid_dataset):
    """
    Test that without frequency features, output is just one-hot encoding.

    Verifies:
    - When include_frequency_features=False, output is exactly QV columns
    """
    imputer = BasisEmbedding(include_frequency_features=False)
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    # Extract QV columns from input
    qv_columns = [f'QV_{i}' for i in range(1, 40)]
    qv_data = valid_dataset[qv_columns].values

    # Should be exactly the same
    np.testing.assert_array_equal(features, qv_data)


# ============================================================================
# Tests for fit_transform() Method
# ============================================================================

def test_fit_transform_works(valid_dataset):
    """
    Test that fit_transform() works correctly.

    Verifies:
    - Can fit and transform in one call
    - Returns correct output shape
    - Sets fitted_ flag
    """
    imputer = BasisEmbedding()
    features = imputer.fit_transform(valid_dataset)

    assert imputer.fitted_ is True
    assert features.shape == (len(valid_dataset), 78)


def test_fit_transform_equivalent_to_separate_calls(valid_dataset):
    """
    Test that fit_transform(X) produces same output as fit(X).transform(X).

    Verifies:
    - Both workflows produce identical results
    """
    # Approach 1: fit_transform
    imputer1 = BasisEmbedding()
    features1 = imputer1.fit_transform(valid_dataset)

    # Approach 2: fit then transform
    imputer2 = BasisEmbedding()
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
    imputer = BasisEmbedding()

    with pytest.raises(RuntimeError) as exc_info:
        imputer.transform(valid_dataset)

    assert "fit" in str(exc_info.value).lower()


def test_invalid_input_raises_error():
    """
    Test that invalid input data raises ValueError.

    Verifies:
    - Non-DataFrame input is rejected
    - Wrong number of columns is rejected
    """
    imputer = BasisEmbedding()

    # Test with list (not DataFrame)
    with pytest.raises(ValueError):
        imputer.fit([1, 2, 3])

    # Test with DataFrame with wrong columns
    df_wrong = pd.DataFrame(np.random.rand(10, 10))
    with pytest.raises(ValueError):
        imputer.fit(df_wrong)


# ============================================================================
# Tests for get_feature_names() Method
# ============================================================================

def test_get_feature_names_with_frequency(valid_dataset):
    """
    Test get_feature_names() returns correct names with frequency features.

    Verifies:
    - Returns 78 names when include_frequency_features=True
    - Names follow expected pattern
    """
    imputer = BasisEmbedding(include_frequency_features=True)
    imputer.fit(valid_dataset)
    names = imputer.get_feature_names()

    assert len(names) == 78

    # Check first few one-hot names
    assert names[0] == 'qv_1_onehot'
    assert names[1] == 'qv_2_onehot'
    assert names[38] == 'qv_39_onehot'

    # Check first few frequency names
    assert names[39] == 'qv_1_freq'
    assert names[40] == 'qv_2_freq'
    assert names[77] == 'qv_39_freq'


def test_get_feature_names_without_frequency(valid_dataset):
    """
    Test get_feature_names() returns correct names without frequency features.

    Verifies:
    - Returns 39 names when include_frequency_features=False
    - All names are one-hot type
    """
    imputer = BasisEmbedding(include_frequency_features=False)
    imputer.fit(valid_dataset)
    names = imputer.get_feature_names()

    assert len(names) == 39

    # All should be one-hot names
    assert names[0] == 'qv_1_onehot'
    assert names[38] == 'qv_39_onehot'
    assert 'freq' not in ''.join(names)


# ============================================================================
# Tests for Value Ranges
# ============================================================================

def test_output_value_ranges(valid_dataset):
    """
    Test that output values are in expected ranges.

    Verifies:
    - One-hot features are binary (0 or 1)
    - Frequency features are in [0, 1] range
    """
    imputer = BasisEmbedding()
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    # One-hot features (first 39 columns) should be 0 or 1
    onehot_features = features[:, :39]
    assert np.all((onehot_features == 0) | (onehot_features == 1))

    # Frequency features (columns 39-77) should be in [0, 1]
    freq_features = features[:, 39:78]
    assert np.all(freq_features >= 0)
    assert np.all(freq_features <= 1)


def test_active_position_count_preserved(valid_dataset):
    """
    Test that each row still has exactly 5 active positions in one-hot features.

    Verifies:
    - Sum of first 39 columns equals 5 for each row
    - Quantum state constraint is preserved
    """
    imputer = BasisEmbedding()
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    # One-hot features (first 39 columns)
    onehot_features = features[:, :39]

    # Each row should have exactly 5 active positions
    row_sums = onehot_features.sum(axis=1)
    assert np.all(row_sums == 5), "Some rows don't have exactly 5 active positions"


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_workflow_with_sample_data(valid_dataset):
    """
    Integration test: Full workflow from initialization to transformation.

    Verifies:
    - Complete workflow works end-to-end
    - Can handle both training and test data
    """
    # Split data (simple first/last split)
    train_data = valid_dataset.iloc[:7]
    test_data = valid_dataset.iloc[7:]

    # Create imputer
    imputer = BasisEmbedding()

    # Fit on training data
    imputer.fit(train_data)
    assert imputer.fitted_ is True

    # Transform training data
    train_features = imputer.transform(train_data)
    assert train_features.shape == (7, 78)

    # Transform test data (different size)
    test_features = imputer.transform(test_data)
    assert test_features.shape == (3, 78)

    # Verify no NaN or Inf
    assert not np.isnan(train_features).any()
    assert not np.isnan(test_features).any()


def test_consistency_across_multiple_transforms(valid_dataset):
    """
    Test that multiple transform calls produce identical results.

    Verifies:
    - Transform is deterministic (no randomness)
    - Can be called multiple times with same result
    """
    imputer = BasisEmbedding()
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
    imputer = BasisEmbedding()
    imputer.fit(valid_dataset)

    # Transform single row
    single_sample = valid_dataset.iloc[[0]]
    features = imputer.transform(single_sample)

    assert features.shape == (1, 78)
    assert not np.isnan(features).any()
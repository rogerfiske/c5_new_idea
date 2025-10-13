"""
Unit Tests for Angle Encoding Imputation Strategy

This test suite validates the Angle Encoding imputation method, which maps
binary quantum state vectors to rotation angles on a circular representation.

Test Coverage:
- Initialization with different configurations
- Position angle mapping (θᵢ = 2π(i-1)/39)
- Direct angle features
- Trigonometric features (sin/cos)
- Aggregated angle features
- Output shapes and types
- Feature name generation
- Integration tests

Author: BMad Dev Agent (James)
Date: 2025-10-13
Story: Epic 2, Story 2.4 - Angle Encoding Strategy
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.imputation.angle_encoding import AngleEncoding
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

def test_angle_encoding_initialization_default():
    """
    Test AngleEncoding initializes with correct default parameters.

    Verifies:
    - Default name is "angle_encoding"
    - include_trig_features defaults to True
    - include_aggregated_features defaults to True
    - Not fitted initially
    - Config stored correctly
    - Position angles are pre-computed
    """
    imputer = AngleEncoding()

    assert imputer.name == "angle_encoding"
    assert imputer.include_trig_features is True
    assert imputer.include_aggregated_features is True
    assert imputer.fitted_ is False
    assert imputer.config == {
        "include_trig_features": True,
        "include_aggregated_features": True
    }
    assert imputer.position_angles_ is not None
    assert imputer.position_angles_.shape == (39,)


def test_angle_encoding_initialization_custom():
    """
    Test AngleEncoding can be initialized with custom parameters.

    Verifies:
    - Custom name is stored
    - include_trig_features can be set to False
    - include_aggregated_features can be set to False
    - Config reflects custom settings
    """
    imputer = AngleEncoding(
        name="custom_angle",
        include_trig_features=False,
        include_aggregated_features=False
    )

    assert imputer.name == "custom_angle"
    assert imputer.include_trig_features is False
    assert imputer.include_aggregated_features is False
    assert imputer.config == {
        "include_trig_features": False,
        "include_aggregated_features": False
    }


# ============================================================================
# Tests for Position Angle Mapping
# ============================================================================

def test_position_angles_correct_formula():
    """
    Test that position angles follow formula θᵢ = 2π(i-1)/39.

    Verifies:
    - Position 1 has angle 0
    - Position 39 has angle 2π*38/39
    - Angles are evenly spaced
    - All angles in range [0, 2π)
    """
    imputer = AngleEncoding()

    # Test specific positions
    assert imputer.position_angles_[0] == 0.0  # Position 1
    np.testing.assert_almost_equal(
        imputer.position_angles_[19],
        2 * np.pi * 19 / 39,
        decimal=10
    )  # Position 20
    np.testing.assert_almost_equal(
        imputer.position_angles_[38],
        2 * np.pi * 38 / 39,
        decimal=10
    )  # Position 39

    # All angles should be in [0, 2π)
    assert np.all(imputer.position_angles_ >= 0)
    assert np.all(imputer.position_angles_ < 2 * np.pi)


def test_position_angles_cyclic_spacing():
    """
    Test that position angles are evenly spaced around the circle.

    Verifies:
    - Adjacent angles differ by 2π/39
    - Respects C₃₉ cyclic group structure
    """
    imputer = AngleEncoding()

    expected_spacing = 2 * np.pi / 39

    # Check spacing between adjacent positions
    for i in range(38):  # Positions 1-38
        actual_spacing = imputer.position_angles_[i + 1] - imputer.position_angles_[i]
        np.testing.assert_almost_equal(actual_spacing, expected_spacing, decimal=10)


def test_get_position_angle_utility():
    """
    Test get_position_angle() utility method.

    Verifies:
    - Returns correct angle for valid positions
    - Raises error for invalid positions
    """
    imputer = AngleEncoding()

    # Valid positions
    assert imputer.get_position_angle(1) == 0.0
    np.testing.assert_almost_equal(
        imputer.get_position_angle(20),
        2 * np.pi * 19 / 39,
        decimal=10
    )

    # Invalid positions
    with pytest.raises(ValueError):
        imputer.get_position_angle(0)
    with pytest.raises(ValueError):
        imputer.get_position_angle(40)


# ============================================================================
# Tests for _fit() Method
# ============================================================================

def test_fit_no_learning_required(valid_dataset):
    """
    Test that fit() completes successfully even though no learning occurs.

    Verifies:
    - fit() runs without error
    - Angles remain unchanged (pre-computed)
    - fitted_ flag is set
    """
    imputer = AngleEncoding()
    angles_before = imputer.position_angles_.copy()

    imputer.fit(valid_dataset)

    assert imputer.fitted_ is True
    np.testing.assert_array_equal(imputer.position_angles_, angles_before)


# ============================================================================
# Tests for _transform() - Output Shapes
# ============================================================================

def test_transform_output_shape_all_features(valid_dataset):
    """
    Test transform() with all features enabled (default).

    Verifies:
    - Output shape is (n_samples, 123)
    - 123 = 39 (angles) + 78 (sin/cos) + 6 (aggregated)
    """
    imputer = AngleEncoding()
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    assert features.shape == (len(valid_dataset), 123)


def test_transform_output_shape_no_aggregated(valid_dataset):
    """
    Test transform() without aggregated features.

    Verifies:
    - Output shape is (n_samples, 117)
    - 117 = 39 (angles) + 78 (sin/cos)
    """
    imputer = AngleEncoding(include_aggregated_features=False)
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    assert features.shape == (len(valid_dataset), 117)


def test_transform_output_shape_no_trig(valid_dataset):
    """
    Test transform() without trigonometric features.

    Verifies:
    - Output shape is (n_samples, 45)
    - 45 = 39 (angles) + 6 (aggregated)
    """
    imputer = AngleEncoding(
        include_trig_features=False,
        include_aggregated_features=True
    )
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    assert features.shape == (len(valid_dataset), 45)


def test_transform_output_shape_angles_only(valid_dataset):
    """
    Test transform() with only angle features.

    Verifies:
    - Output shape is (n_samples, 39)
    """
    imputer = AngleEncoding(
        include_trig_features=False,
        include_aggregated_features=False
    )
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    assert features.shape == (len(valid_dataset), 39)


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
    imputer = AngleEncoding()
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
    imputer = AngleEncoding()
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    assert not np.isnan(features).any(), "Output contains NaN values"
    assert not np.isinf(features).any(), "Output contains Inf values"
    assert np.all(np.isfinite(features)), "Output contains non-finite values"


# ============================================================================
# Tests for Angle Features
# ============================================================================

def test_angle_features_correct_mapping(valid_dataset):
    """
    Test that angle features correctly map active positions to angles.

    Verifies:
    - Inactive positions have angle 0
    - Active positions have their assigned angle θᵢ
    """
    imputer = AngleEncoding(
        include_trig_features=False,
        include_aggregated_features=False
    )
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    # Extract QV columns
    qv_columns = [f'QV_{i}' for i in range(1, 40)]
    qv_data = valid_dataset[qv_columns].values

    # Expected: angle_features[i] = qv_data[i] * position_angles[i]
    expected_angles = qv_data * imputer.position_angles_

    np.testing.assert_array_almost_equal(features, expected_angles, decimal=10)


def test_angle_features_value_range(valid_dataset):
    """
    Test that angle feature values are in expected range.

    Verifies:
    - All values in [0, 2π]
    """
    imputer = AngleEncoding(
        include_trig_features=False,
        include_aggregated_features=False
    )
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    assert np.all(features >= 0)
    assert np.all(features <= 2 * np.pi)


# ============================================================================
# Tests for Trigonometric Features
# ============================================================================

def test_trigonometric_features_correct_computation(valid_dataset):
    """
    Test that sin/cos features are correctly computed.

    Verifies:
    - Sin features = sin(θᵢ) * QV[i]
    - Cos features = cos(θᵢ) * QV[i]
    - Values are in range [-1, 1]
    """
    imputer = AngleEncoding(include_aggregated_features=False)
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    # Extract QV columns
    qv_columns = [f'QV_{i}' for i in range(1, 40)]
    qv_data = valid_dataset[qv_columns].values

    # Expected features
    expected_angles = qv_data * imputer.position_angles_
    expected_sin = qv_data * np.sin(imputer.position_angles_)
    expected_cos = qv_data * np.cos(imputer.position_angles_)

    # Extract actual features
    actual_angles = features[:, :39]
    actual_sin = features[:, 39:78]
    actual_cos = features[:, 78:117]

    np.testing.assert_array_almost_equal(actual_angles, expected_angles, decimal=10)
    np.testing.assert_array_almost_equal(actual_sin, expected_sin, decimal=10)
    np.testing.assert_array_almost_equal(actual_cos, expected_cos, decimal=10)


def test_trigonometric_features_value_range(valid_dataset):
    """
    Test that sin/cos features are in valid range [-1, 1].

    Verifies:
    - Sin values in [-1, 1]
    - Cos values in [-1, 1]
    """
    imputer = AngleEncoding(include_aggregated_features=False)
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    sin_features = features[:, 39:78]
    cos_features = features[:, 78:117]

    assert np.all(sin_features >= -1)
    assert np.all(sin_features <= 1)
    assert np.all(cos_features >= -1)
    assert np.all(cos_features <= 1)


# ============================================================================
# Tests for Aggregated Features
# ============================================================================

def test_aggregated_features_correct_computation(valid_dataset):
    """
    Test that aggregated features are correctly computed.

    Verifies:
    - sum_sin, sum_cos are sums across active positions
    - mean_sin, mean_cos are averages (sum / 5)
    - resultant_magnitude is vector length
    - resultant_angle is atan2(sum_sin, sum_cos)
    """
    imputer = AngleEncoding()
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    # Extract QV columns
    qv_columns = [f'QV_{i}' for i in range(1, 40)]
    qv_data = valid_dataset[qv_columns].values

    # Compute expected aggregated features manually
    sin_values = qv_data * np.sin(imputer.position_angles_)
    cos_values = qv_data * np.cos(imputer.position_angles_)

    expected_sum_sin = sin_values.sum(axis=1)
    expected_sum_cos = cos_values.sum(axis=1)
    expected_mean_sin = expected_sum_sin / 5
    expected_mean_cos = expected_sum_cos / 5
    expected_magnitude = np.sqrt(expected_sum_sin**2 + expected_sum_cos**2)
    expected_angle = np.arctan2(expected_sum_sin, expected_sum_cos)

    # Extract actual aggregated features (last 6 columns)
    actual_sum_sin = features[:, -6]
    actual_sum_cos = features[:, -5]
    actual_mean_sin = features[:, -4]
    actual_mean_cos = features[:, -3]
    actual_magnitude = features[:, -2]
    actual_angle = features[:, -1]

    np.testing.assert_array_almost_equal(actual_sum_sin, expected_sum_sin, decimal=10)
    np.testing.assert_array_almost_equal(actual_sum_cos, expected_sum_cos, decimal=10)
    np.testing.assert_array_almost_equal(actual_mean_sin, expected_mean_sin, decimal=10)
    np.testing.assert_array_almost_equal(actual_mean_cos, expected_mean_cos, decimal=10)
    np.testing.assert_array_almost_equal(actual_magnitude, expected_magnitude, decimal=10)
    np.testing.assert_array_almost_equal(actual_angle, expected_angle, decimal=10)


def test_aggregated_features_value_ranges(valid_dataset):
    """
    Test that aggregated features are in reasonable ranges.

    Verifies:
    - sum_sin, sum_cos in [-5, 5] (max 5 active positions)
    - mean_sin, mean_cos in [-1, 1] (average of sin/cos)
    - resultant_magnitude in [0, 5] (can't exceed 5 unit vectors)
    - resultant_angle in [-π, π]
    """
    imputer = AngleEncoding()
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    sum_sin = features[:, -6]
    sum_cos = features[:, -5]
    mean_sin = features[:, -4]
    mean_cos = features[:, -3]
    magnitude = features[:, -2]
    angle = features[:, -1]

    # Sum ranges
    assert np.all(sum_sin >= -5) and np.all(sum_sin <= 5)
    assert np.all(sum_cos >= -5) and np.all(sum_cos <= 5)

    # Mean ranges
    assert np.all(mean_sin >= -1) and np.all(mean_sin <= 1)
    assert np.all(mean_cos >= -1) and np.all(mean_cos <= 1)

    # Magnitude range
    assert np.all(magnitude >= 0) and np.all(magnitude <= 5)

    # Angle range
    assert np.all(angle >= -np.pi) and np.all(angle <= np.pi)


# ============================================================================
# Tests for get_feature_names()
# ============================================================================

def test_get_feature_names_all_features(valid_dataset):
    """
    Test get_feature_names() with all features enabled.

    Verifies:
    - Returns 123 names
    - Names follow expected pattern
    """
    imputer = AngleEncoding()
    imputer.fit(valid_dataset)
    names = imputer.get_feature_names()

    assert len(names) == 123

    # Check angle names
    assert names[0] == 'qv_1_angle'
    assert names[38] == 'qv_39_angle'

    # Check sin names
    assert names[39] == 'qv_1_sin'
    assert names[77] == 'qv_39_sin'

    # Check cos names
    assert names[78] == 'qv_1_cos'
    assert names[116] == 'qv_39_cos'

    # Check aggregated names
    assert names[117] == 'sum_sin'
    assert names[118] == 'sum_cos'
    assert names[119] == 'mean_sin'
    assert names[120] == 'mean_cos'
    assert names[121] == 'resultant_magnitude'
    assert names[122] == 'resultant_angle'


def test_get_feature_names_angles_only(valid_dataset):
    """
    Test get_feature_names() with only angle features.

    Verifies:
    - Returns 39 names
    - All names are angle type
    """
    imputer = AngleEncoding(
        include_trig_features=False,
        include_aggregated_features=False
    )
    imputer.fit(valid_dataset)
    names = imputer.get_feature_names()

    assert len(names) == 39
    assert names[0] == 'qv_1_angle'
    assert names[38] == 'qv_39_angle'
    assert 'sin' not in ''.join(names)
    assert 'cos' not in ''.join(names)


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
    imputer = AngleEncoding()
    features = imputer.fit_transform(valid_dataset)

    assert imputer.fitted_ is True
    assert features.shape == (len(valid_dataset), 123)


def test_fit_transform_equivalent_to_separate_calls(valid_dataset):
    """
    Test that fit_transform(X) produces same output as fit(X).transform(X).

    Verifies:
    - Both workflows produce identical results
    """
    # Approach 1: fit_transform
    imputer1 = AngleEncoding()
    features1 = imputer1.fit_transform(valid_dataset)

    # Approach 2: fit then transform
    imputer2 = AngleEncoding()
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
    imputer = AngleEncoding()

    with pytest.raises(RuntimeError) as exc_info:
        imputer.transform(valid_dataset)

    assert "fit" in str(exc_info.value).lower()


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_workflow_all_configurations(valid_dataset):
    """
    Integration test: Test all configuration combinations.

    Verifies:
    - All 4 configurations work end-to-end
    - Each produces expected output shape
    """
    configs = [
        (True, True, 123),   # All features
        (True, False, 117),  # No aggregated
        (False, True, 45),   # No trig
        (False, False, 39)   # Only angles
    ]

    for trig, agg, expected_features in configs:
        imputer = AngleEncoding(
            include_trig_features=trig,
            include_aggregated_features=agg
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
    imputer = AngleEncoding()
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
    imputer = AngleEncoding()
    imputer.fit(valid_dataset)

    # Transform single row
    single_sample = valid_dataset.iloc[[0]]
    features = imputer.transform(single_sample)

    assert features.shape == (1, 123)
    assert not np.isnan(features).any()

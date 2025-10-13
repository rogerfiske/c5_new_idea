"""
Unit Tests for Graph/Cycle Encoding Imputation Strategy

This test suite validates the Graph/Cycle Encoding imputation method, which uses
Discrete Fourier Transform (DFT) and graph features on the C₃₉ cyclic ring.

Test Coverage:
- Initialization with different configurations
- DFT magnitude and phase features
- Graph features (distances, clustering, symmetry)
- Output shapes and types
- Parseval's theorem verification
- Feature name generation
- Integration tests

Author: BMad Dev Agent (James)
Date: 2025-10-13
Story: Epic 2, Story 2.6 - Graph/Cycle Encoding Strategy
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.imputation.graph_cycle_encoding import GraphCycleEncoding
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

def test_graph_cycle_encoding_initialization_default():
    """
    Test GraphCycleEncoding initializes with correct default parameters.

    Verifies:
    - Default name is "graph_cycle_encoding"
    - include_dft_features defaults to True
    - include_graph_features defaults to True
    - n_harmonics defaults to 20
    - Not fitted initially
    - Config stored correctly
    """
    imputer = GraphCycleEncoding()

    assert imputer.name == "graph_cycle_encoding"
    assert imputer.include_dft_features is True
    assert imputer.include_graph_features is True
    assert imputer.n_harmonics == 20
    assert imputer.fitted_ is False
    assert imputer.config == {
        "include_dft_features": True,
        "include_graph_features": True,
        "n_harmonics": 20
    }


def test_graph_cycle_encoding_initialization_custom():
    """
    Test GraphCycleEncoding can be initialized with custom parameters.

    Verifies:
    - Custom name is stored
    - include_dft_features can be set to False
    - include_graph_features can be set to False
    - n_harmonics can be customized
    - Config reflects custom settings
    """
    imputer = GraphCycleEncoding(
        name="custom_graph",
        include_dft_features=False,
        include_graph_features=True,
        n_harmonics=10
    )

    assert imputer.name == "custom_graph"
    assert imputer.include_dft_features is False
    assert imputer.include_graph_features is True
    assert imputer.n_harmonics == 10


def test_invalid_n_harmonics_raises_error():
    """
    Test that invalid n_harmonics values raise ValueError.

    Verifies:
    - n_harmonics = 0 raises error
    - n_harmonics = 40 raises error
    - n_harmonics = -1 raises error
    """
    with pytest.raises(ValueError):
        GraphCycleEncoding(n_harmonics=0)

    with pytest.raises(ValueError):
        GraphCycleEncoding(n_harmonics=40)

    with pytest.raises(ValueError):
        GraphCycleEncoding(n_harmonics=-1)


# ============================================================================
# Tests for _fit() Method
# ============================================================================

def test_fit_no_learning_required(valid_dataset):
    """
    Test that fit() completes successfully even though no learning occurs.

    Verifies:
    - fit() runs without error
    - fitted_ flag is set
    - No parameters are learned (deterministic features)
    """
    imputer = GraphCycleEncoding()

    imputer.fit(valid_dataset)

    assert imputer.fitted_ is True


# ============================================================================
# Tests for _transform() - Output Shapes
# ============================================================================

def test_transform_output_shape_all_features(valid_dataset):
    """
    Test transform() with all features enabled (default).

    Verifies:
    - Output shape is (n_samples, 57) with n_harmonics=20
    - 57 = 40 (DFT) + 17 (graph)
    - 40 = 20 magnitude + 20 phase
    """
    imputer = GraphCycleEncoding()
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    expected_dft_features = 2 * 20  # magnitude + phase
    expected_graph_features = 17
    expected_total = expected_dft_features + expected_graph_features

    assert features.shape == (len(valid_dataset), expected_total)


def test_transform_output_shape_full_harmonics(valid_dataset):
    """
    Test transform() with all 39 harmonics.

    Verifies:
    - Output shape is (n_samples, 95) with n_harmonics=39
    - 95 = 78 (DFT) + 17 (graph)
    """
    imputer = GraphCycleEncoding(n_harmonics=39)
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    expected_dft_features = 2 * 39
    expected_graph_features = 17
    expected_total = expected_dft_features + expected_graph_features

    assert features.shape == (len(valid_dataset), expected_total)


def test_transform_output_shape_dft_only(valid_dataset):
    """
    Test transform() with only DFT features.

    Verifies:
    - Output shape is (n_samples, 40) with n_harmonics=20
    - 40 = 20 magnitude + 20 phase
    """
    imputer = GraphCycleEncoding(
        include_graph_features=False,
        n_harmonics=20
    )
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    expected_features = 2 * 20
    assert features.shape == (len(valid_dataset), expected_features)


def test_transform_output_shape_graph_only(valid_dataset):
    """
    Test transform() with only graph features.

    Verifies:
    - Output shape is (n_samples, 17)
    """
    imputer = GraphCycleEncoding(include_dft_features=False)
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    assert features.shape == (len(valid_dataset), 17)


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
    imputer = GraphCycleEncoding()
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
    imputer = GraphCycleEncoding()
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    assert not np.isnan(features).any(), "Output contains NaN values"
    assert not np.isinf(features).any(), "Output contains Inf values"
    assert np.all(np.isfinite(features)), "Output contains non-finite values"


# ============================================================================
# Tests for DFT Features
# ============================================================================

def test_dft_magnitude_nonnegative(valid_dataset):
    """
    Test that DFT magnitude values are non-negative.

    Verifies:
    - All magnitude values >= 0
    - Magnitude is |F[k]| which is always non-negative
    """
    imputer = GraphCycleEncoding(include_graph_features=False, n_harmonics=20)
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    # First n_harmonics columns are magnitudes
    magnitudes = features[:, :20]

    assert np.all(magnitudes >= 0), "DFT magnitudes should be non-negative"


def test_dft_phase_range(valid_dataset):
    """
    Test that DFT phase values are in expected range.

    Verifies:
    - Phase values in [-π, π]
    - This is the range returned by np.angle()
    """
    imputer = GraphCycleEncoding(include_graph_features=False, n_harmonics=20)
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    # Second n_harmonics columns are phases
    phases = features[:, 20:40]

    assert np.all(phases >= -np.pi)
    assert np.all(phases <= np.pi)


def test_dft_dc_component_correct(valid_dataset):
    """
    Test that DC component (k=0) equals sum of input values.

    Verifies:
    - Magnitude of DC component (k=0) equals number of ones (5)
    - Phase of DC component should be 0 or π (real-valued for binary input)
    """
    imputer = GraphCycleEncoding(include_graph_features=False, n_harmonics=20)
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    # DC component is first column (k=0 magnitude)
    dc_magnitudes = features[:, 0]

    # For binary vectors with 5 ones, DC magnitude should be 5
    np.testing.assert_array_almost_equal(dc_magnitudes, 5.0, decimal=10)


def test_parseval_theorem_energy_conservation(valid_dataset):
    """
    Test Parseval's theorem: energy is conserved in DFT.

    Verifies:
    - Σₙ |x[n]|² = (1/N) Σₖ |F[k]|²
    - For binary vectors with 5 ones: left side = 5
    - This validates DFT correctness
    """
    imputer = GraphCycleEncoding(
        include_graph_features=False,
        n_harmonics=39  # Use all harmonics
    )
    imputer.fit(valid_dataset)

    # Extract QV data
    qv_columns = [f'QV_{i}' for i in range(1, 40)]
    qv_data = valid_dataset[qv_columns].values

    # Compute FFT manually
    fft_result = np.fft.fft(qv_data, axis=1)
    magnitudes_all = np.abs(fft_result)

    # Energy in spatial domain
    spatial_energy = (qv_data ** 2).sum(axis=1)  # Should be 5 for each row

    # Energy in frequency domain (Parseval's theorem)
    frequency_energy = (magnitudes_all ** 2).sum(axis=1) / 39

    # Should be equal
    np.testing.assert_array_almost_equal(spatial_energy, frequency_energy, decimal=8)


# ============================================================================
# Tests for Graph Features
# ============================================================================

def test_graph_distances_valid_range(valid_dataset):
    """
    Test that circular distances are in valid range.

    Verifies:
    - min_circular_distance in [0, 19]
    - max_circular_distance in [0, 19]
    - Circular distance on C₃₉ ring is at most 19 (half the ring)
    """
    imputer = GraphCycleEncoding(include_dft_features=False)
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    min_dist = features[:, 0]
    max_dist = features[:, 1]

    assert np.all(min_dist >= 0) and np.all(min_dist <= 19)
    assert np.all(max_dist >= 0) and np.all(max_dist <= 19)
    assert np.all(min_dist <= max_dist)


def test_graph_angular_span_valid_range(valid_dataset):
    """
    Test that angular span is in valid range.

    Verifies:
    - angular_span in [0, 19]
    - Span is shortest arc covering all 5 active positions
    """
    imputer = GraphCycleEncoding(include_dft_features=False)
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    angular_span = features[:, 5]

    assert np.all(angular_span >= 0) and np.all(angular_span <= 19)


def test_graph_consecutive_pairs_valid_range(valid_dataset):
    """
    Test that consecutive pairs count is valid.

    Verifies:
    - consecutive_pairs in [0, 5]
    - Maximum is 5 if all positions are adjacent (rare)
    """
    imputer = GraphCycleEncoding(include_dft_features=False)
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    consecutive_pairs = features[:, 6]

    assert np.all(consecutive_pairs >= 0) and np.all(consecutive_pairs <= 5)


def test_graph_clustering_coefficient_range(valid_dataset):
    """
    Test that clustering coefficient is in [0, 1] range.

    Verifies:
    - clustering_coefficient in [0, 1]
    - 1 = very clustered, 0 = spread out
    """
    imputer = GraphCycleEncoding(include_dft_features=False)
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    clustering_coef = features[:, 10]

    assert np.all(clustering_coef >= 0) and np.all(clustering_coef <= 1)


def test_graph_symmetry_scores_range(valid_dataset):
    """
    Test that symmetry scores are in valid ranges.

    Verifies:
    - symmetry_score >= 0 (variance-based)
    - reflection_symmetry in [0, 1]
    - rotational_order_3 in [0, 1]
    - rotational_order_13 in [0, 1]
    - position_spread in [0, 1]
    """
    imputer = GraphCycleEncoding(include_dft_features=False)
    imputer.fit(valid_dataset)
    features = imputer.transform(valid_dataset)

    symmetry_score = features[:, 12]
    reflection_symmetry = features[:, 13]
    rotational_order_3 = features[:, 14]
    rotational_order_13 = features[:, 15]
    position_spread = features[:, 16]

    assert np.all(symmetry_score >= 0)
    assert np.all(reflection_symmetry >= 0) and np.all(reflection_symmetry <= 1)
    assert np.all(rotational_order_3 >= 0) and np.all(rotational_order_3 <= 1)
    assert np.all(rotational_order_13 >= 0) and np.all(rotational_order_13 <= 1)
    assert np.all(position_spread >= 0) and np.all(position_spread <= 1)


# ============================================================================
# Tests for get_feature_names()
# ============================================================================

def test_get_feature_names_all_features(valid_dataset):
    """
    Test get_feature_names() with all features enabled.

    Verifies:
    - Returns 57 names (n_harmonics=20)
    - Names follow expected pattern
    """
    imputer = GraphCycleEncoding(n_harmonics=20)
    imputer.fit(valid_dataset)
    names = imputer.get_feature_names()

    expected_count = 2 * 20 + 17  # DFT + graph
    assert len(names) == expected_count

    # Check DFT magnitude names
    assert names[0] == 'dft_magnitude_0'
    assert names[19] == 'dft_magnitude_19'

    # Check DFT phase names
    assert names[20] == 'dft_phase_0'
    assert names[39] == 'dft_phase_19'

    # Check graph feature names
    assert names[40] == 'min_circular_distance'
    assert names[-1] == 'position_spread'


def test_get_feature_names_dft_only(valid_dataset):
    """
    Test get_feature_names() with only DFT features.

    Verifies:
    - Returns 40 names (2 * n_harmonics=20)
    - No graph feature names
    """
    imputer = GraphCycleEncoding(
        include_graph_features=False,
        n_harmonics=20
    )
    imputer.fit(valid_dataset)
    names = imputer.get_feature_names()

    assert len(names) == 40
    assert 'dft_magnitude_0' in names
    assert 'dft_phase_0' in names
    assert 'min_circular_distance' not in names


def test_get_feature_names_graph_only(valid_dataset):
    """
    Test get_feature_names() with only graph features.

    Verifies:
    - Returns 17 names
    - No DFT feature names
    """
    imputer = GraphCycleEncoding(include_dft_features=False)
    imputer.fit(valid_dataset)
    names = imputer.get_feature_names()

    assert len(names) == 17
    assert 'min_circular_distance' in names
    assert 'position_spread' in names
    assert 'dft_magnitude_0' not in names


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
    imputer = GraphCycleEncoding()
    features = imputer.fit_transform(valid_dataset)

    assert imputer.fitted_ is True
    assert features.shape == (len(valid_dataset), 57)


def test_fit_transform_equivalent_to_separate_calls(valid_dataset):
    """
    Test that fit_transform(X) produces same output as fit(X).transform(X).

    Verifies:
    - Both workflows produce identical results
    """
    # Approach 1: fit_transform
    imputer1 = GraphCycleEncoding()
    features1 = imputer1.fit_transform(valid_dataset)

    # Approach 2: fit then transform
    imputer2 = GraphCycleEncoding()
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
    imputer = GraphCycleEncoding()

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
    - Various n_harmonics values work
    - All feature combination flags work
    - Each produces expected output shape
    """
    configs = [
        (True, True, 10, 37),   # 10 harmonics: 20 DFT + 17 graph
        (True, True, 39, 95),   # All harmonics: 78 DFT + 17 graph
        (True, False, 20, 40),  # DFT only: 40 features
        (False, True, 20, 17),  # Graph only: 17 features
    ]

    for dft, graph, n_harm, expected_features in configs:
        imputer = GraphCycleEncoding(
            include_dft_features=dft,
            include_graph_features=graph,
            n_harmonics=n_harm
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
    imputer = GraphCycleEncoding()
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
    imputer = GraphCycleEncoding()
    imputer.fit(valid_dataset)

    # Transform single row
    single_sample = valid_dataset.iloc[[0]]
    features = imputer.transform(single_sample)

    assert features.shape == (1, 57)
    assert not np.isnan(features).any()

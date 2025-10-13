"""
Unit Tests for Base Imputer Class

This test suite validates the abstract base class for imputation strategies.
It ensures that the base class enforces correct interface contracts and
validates inputs/outputs properly.

Test Coverage:
- Abstract class cannot be instantiated directly
- Input validation catches invalid data formats
- Fit-before-transform enforcement
- Output validation catches NaN/Inf
- Method chaining works correctly

Author: BMad Dev Agent (James)
Date: 2025-10-13
Story: Epic 2, Story 2.1 - Base Imputation Class
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Import the base class we're testing
from src.imputation.base_imputer import BaseImputer


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def fixture_dir():
    """
    Provide the path to the test fixtures directory.

    Returns:
        Path: Absolute path to tests/fixtures/
    """
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def valid_dataset_path(fixture_dir):
    """
    Provide path to a valid sample dataset.

    Returns:
        Path: Path to sample_valid_dataset.csv
    """
    return fixture_dir / "sample_valid_dataset.csv"


@pytest.fixture
def valid_dataset(valid_dataset_path):
    """
    Load valid sample dataset for testing.

    Returns:
        pd.DataFrame: Valid dataset with 10 rows, 40 columns
    """
    return pd.read_csv(valid_dataset_path)


@pytest.fixture
def concrete_imputer():
    """
    Create a concrete implementation of BaseImputer for testing.

    This fixture provides a minimal concrete implementation that can be
    used to test the base class functionality without abstract method errors.

    Returns:
        ConcreteImputer: A testable implementation of BaseImputer
    """
    class ConcreteImputer(BaseImputer):
        """Simple concrete implementation for testing."""

        def _fit(self, X: pd.DataFrame) -> None:
            """Store mean values as a simple learned parameter."""
            qv_columns = [f'QV_{i}' for i in range(1, 40)]
            self.mean_values_ = X[qv_columns].mean().values

        def _transform(self, X: pd.DataFrame) -> np.ndarray:
            """Return QV columns multiplied by mean values."""
            qv_columns = [f'QV_{i}' for i in range(1, 40)]
            qv_data = X[qv_columns].values
            # Simple transformation: multiply by learned means
            features = qv_data * self.mean_values_
            return features

    return ConcreteImputer(name="test_imputer", config={"test_param": 42})


# ============================================================================
# Tests for Abstract Class Enforcement
# ============================================================================

def test_cannot_instantiate_abstract_base_class():
    """
    Test that BaseImputer cannot be instantiated directly.

    This test verifies that the abstract base class properly enforces
    the interface contract - concrete implementations must provide
    _fit() and _transform() methods.
    """
    # Attempting to instantiate BaseImputer directly should raise TypeError
    with pytest.raises(TypeError) as exc_info:
        BaseImputer(name="test")

    # Error message should mention abstract methods
    error_msg = str(exc_info.value)
    assert "abstract" in error_msg.lower()


def test_concrete_implementation_requires_both_methods():
    """
    Test that concrete classes must implement both _fit() and _transform().

    This verifies that partial implementations are not allowed - both
    abstract methods must be overridden.
    """
    # Implementation with only _fit() should fail
    class PartialImputer(BaseImputer):
        def _fit(self, X: pd.DataFrame) -> None:
            pass

    with pytest.raises(TypeError):
        PartialImputer(name="partial")


# ============================================================================
# Tests for Input Validation
# ============================================================================

def test_validate_input_rejects_non_dataframe(concrete_imputer):
    """
    Test that input validation rejects non-DataFrame inputs.

    Verifies:
    - Rejects lists, arrays, dicts, etc.
    - Error message is user-friendly
    """
    # Try to fit with a list instead of DataFrame
    with pytest.raises(ValueError) as exc_info:
        concrete_imputer.fit([1, 2, 3])

    assert "pandas DataFrame" in str(exc_info.value)
    assert "pd.read_csv()" in str(exc_info.value)  # User-friendly suggestion


def test_validate_input_rejects_wrong_column_count(concrete_imputer):
    """
    Test that input validation rejects DataFrames with wrong number of columns.

    Verifies:
    - Must have exactly 40 columns
    - Error message explains expected format
    """
    # Create DataFrame with only 10 columns
    df_wrong_columns = pd.DataFrame(np.random.randint(0, 2, (10, 10)))

    with pytest.raises(ValueError) as exc_info:
        concrete_imputer.fit(df_wrong_columns)

    assert "40 columns" in str(exc_info.value)
    assert "event-ID" in str(exc_info.value)


def test_validate_input_rejects_wrong_column_names(concrete_imputer):
    """
    Test that input validation rejects DataFrames with incorrect column names.

    Verifies:
    - Column names must match ['event-ID', 'QV_1', ..., 'QV_39']
    - Error message shows expected format
    """
    # Create DataFrame with 40 columns but wrong names
    df_wrong_names = pd.DataFrame(
        np.random.randint(0, 2, (10, 40)),
        columns=[f'col_{i}' for i in range(40)]
    )

    with pytest.raises(ValueError) as exc_info:
        concrete_imputer.fit(df_wrong_names)

    assert "column" in str(exc_info.value).lower()
    assert "QV_" in str(exc_info.value)


def test_validate_input_rejects_missing_values(concrete_imputer, valid_dataset):
    """
    Test that input validation rejects data with missing values (NaN).

    Verifies:
    - Detects NaN in any column
    - Error message reports number of missing values
    """
    # Create dataset with missing value
    df_with_nan = valid_dataset.copy()
    df_with_nan.iloc[0, 1] = np.nan

    with pytest.raises(ValueError) as exc_info:
        concrete_imputer.fit(df_with_nan)

    assert "missing" in str(exc_info.value).lower()
    assert "NaN" in str(exc_info.value)


def test_validate_input_rejects_non_binary_qv_values(concrete_imputer, valid_dataset):
    """
    Test that input validation rejects QV columns with non-binary values.

    Verifies:
    - QV values must be 0 or 1
    - Error message mentions binary constraint
    """
    # Create dataset with value = 2 in QV column
    df_non_binary = valid_dataset.copy()
    df_non_binary.iloc[0, 1] = 2  # QV_1 = 2 (invalid)

    with pytest.raises(ValueError) as exc_info:
        concrete_imputer.fit(df_non_binary)

    assert "binary" in str(exc_info.value).lower()
    assert "0 or 1" in str(exc_info.value)


def test_validate_input_rejects_invalid_qv_sum(concrete_imputer, valid_dataset):
    """
    Test that input validation rejects rows where QV sum ≠ 5.

    Verifies:
    - Each row must have exactly 5 active positions
    - Error message explains quantum state constraint
    """
    # Create dataset with row where sum ≠ 5
    df_invalid_sum = valid_dataset.copy()
    # Set all QV columns to 0 for first row (sum = 0, not 5)
    qv_columns = [f'QV_{i}' for i in range(1, 40)]
    df_invalid_sum.loc[0, qv_columns] = 0

    with pytest.raises(ValueError) as exc_info:
        concrete_imputer.fit(df_invalid_sum)

    assert "5 active positions" in str(exc_info.value)
    assert "quantum state constraint" in str(exc_info.value).lower()


def test_validate_input_passes_for_valid_data(concrete_imputer, valid_dataset):
    """
    Test that input validation passes for correctly formatted data.

    Verifies:
    - Valid dataset passes all checks
    - No exceptions raised
    """
    # This should not raise any exceptions
    concrete_imputer._validate_input(valid_dataset)


# ============================================================================
# Tests for Fit-Before-Transform Enforcement
# ============================================================================

def test_transform_raises_error_if_not_fitted(concrete_imputer, valid_dataset):
    """
    Test that transform() raises error when called before fit().

    This verifies the fit-before-transform contract is enforced.
    """
    # Transform without fitting should raise RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        concrete_imputer.transform(valid_dataset)

    # Error message should mention fitting
    assert "fitted" in str(exc_info.value).lower()
    assert "fit()" in str(exc_info.value)


def test_fitted_flag_set_after_fit(concrete_imputer, valid_dataset):
    """
    Test that fitted_ flag is set to True after successful fit().

    Verifies:
    - fitted_ is False initially
    - fitted_ is True after fit()
    """
    # Initially not fitted
    assert concrete_imputer.fitted_ is False

    # Fit the imputer
    concrete_imputer.fit(valid_dataset)

    # Now should be fitted
    assert concrete_imputer.fitted_ is True


def test_transform_works_after_fit(concrete_imputer, valid_dataset):
    """
    Test that transform() works correctly after fit() is called.

    Verifies:
    - fit() followed by transform() succeeds
    - Returns numpy array
    - Output has correct number of samples
    """
    # Fit then transform should work
    concrete_imputer.fit(valid_dataset)
    features = concrete_imputer.transform(valid_dataset)

    # Should return numpy array
    assert isinstance(features, np.ndarray)

    # Should have same number of samples
    assert features.shape[0] == len(valid_dataset)


# ============================================================================
# Tests for fit_transform() Method
# ============================================================================

def test_fit_transform_combines_fit_and_transform(concrete_imputer, valid_dataset):
    """
    Test that fit_transform() correctly combines fit() and transform().

    Verifies:
    - fit_transform() returns same result as fit() + transform()
    - Sets fitted_ flag
    """
    # Use fit_transform
    features = concrete_imputer.fit_transform(valid_dataset)

    # Should return numpy array
    assert isinstance(features, np.ndarray)

    # Should be fitted
    assert concrete_imputer.fitted_ is True

    # Should have correct number of samples
    assert features.shape[0] == len(valid_dataset)


def test_fit_transform_equivalent_to_separate_calls(valid_dataset):
    """
    Test that fit_transform(X) produces same output as fit(X).transform(X).

    Verifies:
    - Both workflows produce identical results
    """
    # Create two identical imputers
    class SimpleImputer(BaseImputer):
        def _fit(self, X: pd.DataFrame) -> None:
            qv_columns = [f'QV_{i}' for i in range(1, 40)]
            self.sum_values_ = X[qv_columns].sum().values

        def _transform(self, X: pd.DataFrame) -> np.ndarray:
            qv_columns = [f'QV_{i}' for i in range(1, 40)]
            return X[qv_columns].values * self.sum_values_

    imputer1 = SimpleImputer(name="test1")
    imputer2 = SimpleImputer(name="test2")

    # Approach 1: fit_transform
    features1 = imputer1.fit_transform(valid_dataset)

    # Approach 2: fit then transform
    imputer2.fit(valid_dataset)
    features2 = imputer2.transform(valid_dataset)

    # Should be identical
    np.testing.assert_array_equal(features1, features2)


# ============================================================================
# Tests for Output Validation
# ============================================================================

def test_output_validation_rejects_nan(valid_dataset):
    """
    Test that output validation catches NaN values.

    Verifies:
    - Transform that produces NaN raises ValueError
    - Error message mentions NaN
    """
    class NaNImputer(BaseImputer):
        def _fit(self, X: pd.DataFrame) -> None:
            pass

        def _transform(self, X: pd.DataFrame) -> np.ndarray:
            # Return array with NaN
            features = np.ones((len(X), 10))
            features[0, 0] = np.nan
            return features

    imputer = NaNImputer(name="nan_test")
    imputer.fit(valid_dataset)

    with pytest.raises(ValueError) as exc_info:
        imputer.transform(valid_dataset)

    assert "NaN" in str(exc_info.value)


def test_output_validation_rejects_inf(valid_dataset):
    """
    Test that output validation catches Inf values.

    Verifies:
    - Transform that produces Inf raises ValueError
    - Error message mentions Inf
    """
    class InfImputer(BaseImputer):
        def _fit(self, X: pd.DataFrame) -> None:
            pass

        def _transform(self, X: pd.DataFrame) -> np.ndarray:
            # Return array with Inf
            features = np.ones((len(X), 10))
            features[0, 0] = np.inf
            return features

    imputer = InfImputer(name="inf_test")
    imputer.fit(valid_dataset)

    with pytest.raises(ValueError) as exc_info:
        imputer.transform(valid_dataset)

    assert "Inf" in str(exc_info.value)


def test_output_validation_rejects_non_array(valid_dataset):
    """
    Test that output validation rejects non-numpy-array outputs.

    Verifies:
    - _transform() must return numpy array
    - Error message mentions numpy array
    """
    class ListImputer(BaseImputer):
        def _fit(self, X: pd.DataFrame) -> None:
            pass

        def _transform(self, X: pd.DataFrame) -> np.ndarray:
            # Return list instead of numpy array (wrong!)
            return [[1, 2, 3]] * len(X)

    imputer = ListImputer(name="list_test")
    imputer.fit(valid_dataset)

    with pytest.raises(ValueError) as exc_info:
        imputer.transform(valid_dataset)

    assert "numpy array" in str(exc_info.value)


# ============================================================================
# Tests for Method Chaining
# ============================================================================

def test_fit_returns_self_for_chaining(concrete_imputer, valid_dataset):
    """
    Test that fit() returns self to enable method chaining.

    Verifies:
    - fit() returns the imputer instance
    - Enables patterns like: imputer.fit(X).transform(Y)
    """
    result = concrete_imputer.fit(valid_dataset)

    # Should return self
    assert result is concrete_imputer

    # Can chain transform() call
    features = result.transform(valid_dataset)
    assert isinstance(features, np.ndarray)


# ============================================================================
# Tests for Configuration and Attributes
# ============================================================================

def test_imputer_stores_name_and_config():
    """
    Test that imputer correctly stores name and config attributes.

    Verifies:
    - name attribute is set
    - config attribute is set
    - Default config is empty dict
    """
    class DummyImputer(BaseImputer):
        def _fit(self, X): pass
        def _transform(self, X): return np.ones((len(X), 10))

    # With config
    imputer1 = DummyImputer(name="test", config={"param": 42})
    assert imputer1.name == "test"
    assert imputer1.config == {"param": 42}

    # Without config (should default to empty dict)
    imputer2 = DummyImputer(name="test2")
    assert imputer2.config == {}


def test_repr_includes_key_info(concrete_imputer):
    """
    Test that __repr__() provides useful debugging information.

    Verifies:
    - Includes class name
    - Includes name attribute
    - Includes fitted status
    - Includes config
    """
    repr_str = repr(concrete_imputer)

    assert "ConcreteImputer" in repr_str
    assert "test_imputer" in repr_str
    assert "not fitted" in repr_str.lower()
    assert "config" in repr_str.lower()


# ============================================================================
# Integration Test
# ============================================================================

def test_full_workflow_with_real_data(valid_dataset):
    """
    Integration test: Full workflow from instantiation to transform.

    This test verifies the complete workflow:
    1. Create concrete imputer
    2. Fit on training data
    3. Transform data
    4. Validate output

    This ensures all components work together correctly.
    """
    # Create concrete implementation
    class TestImputer(BaseImputer):
        def _fit(self, X: pd.DataFrame) -> None:
            qv_columns = [f'QV_{i}' for i in range(1, 40)]
            self.frequencies_ = X[qv_columns].sum() / len(X)

        def _transform(self, X: pd.DataFrame) -> np.ndarray:
            qv_columns = [f'QV_{i}' for i in range(1, 40)]
            qv_data = X[qv_columns].values
            # Weight by frequencies
            features = qv_data * self.frequencies_.values
            return features

    # Instantiate
    imputer = TestImputer(name="integration_test", config={"test": True})
    assert not imputer.fitted_

    # Fit
    imputer.fit(valid_dataset)
    assert imputer.fitted_
    assert hasattr(imputer, 'frequencies_')

    # Transform
    features = imputer.transform(valid_dataset)
    assert isinstance(features, np.ndarray)
    assert features.shape == (len(valid_dataset), 39)
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()

    # Verify can transform again (idempotent)
    features2 = imputer.transform(valid_dataset)
    np.testing.assert_array_equal(features, features2)

"""
Unit Tests for Data Loader Module

This test suite validates the data loading and validation functions
for the Quantum State Prediction Experiment dataset.

Test Coverage:
- load_dataset(): Success and file-not-found cases
- validate_dataset_structure(): Valid and invalid structures
- validate_data_integrity(): Valid and invalid data
- clean_dataset(): Data type conversions
- save_processed_data(): File saving operations

Author: BMad Dev Agent (James)
Date: 2025-10-13
Story: Epic 1, Story 1.2 - Data Loading and Validation
"""

import pytest
import pandas as pd
from pathlib import Path

# Import the functions we're testing
from src.data_loader import (
    load_dataset,
    validate_dataset_structure,
    validate_data_integrity,
    clean_dataset,
    save_processed_data
)


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
def invalid_columns_path(fixture_dir):
    """
    Provide path to a dataset with wrong number of columns.

    Returns:
        Path: Path to sample_invalid_columns.csv
    """
    return fixture_dir / "sample_invalid_columns.csv"


@pytest.fixture
def invalid_values_path(fixture_dir):
    """
    Provide path to a dataset with invalid QV values.

    Returns:
        Path: Path to sample_invalid_values.csv
    """
    return fixture_dir / "sample_invalid_values.csv"


@pytest.fixture
def missing_values_path(fixture_dir):
    """
    Provide path to a dataset with missing values.

    Returns:
        Path: Path to sample_missing_values.csv
    """
    return fixture_dir / "sample_missing_values.csv"


# ============================================================================
# Tests for load_dataset()
# ============================================================================

def test_load_dataset_success(valid_dataset_path):
    """
    Test that load_dataset() successfully loads a valid CSV file.

    This test verifies:
    - Function returns a pandas DataFrame
    - DataFrame has data (not empty)
    - Expected columns are present
    """
    # Load the dataset
    df = load_dataset(valid_dataset_path)

    # Verify it's a DataFrame
    assert isinstance(df, pd.DataFrame), "load_dataset should return a DataFrame"

    # Verify it has data
    assert len(df) > 0, "Loaded dataset should not be empty"

    # Verify key columns exist
    assert "event-ID" in df.columns, "Dataset should have event-ID column"
    assert "QV_1" in df.columns, "Dataset should have QV_1 column"


def test_load_dataset_file_not_found():
    """
    Test that load_dataset() raises FileNotFoundError for missing files.

    This test verifies:
    - Correct exception type is raised
    - Error message is user-friendly (mentions USER ACTION REQUIRED)
    """
    # Try to load a file that doesn't exist
    nonexistent_path = Path("data/raw/nonexistent_file.csv")

    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError) as exc_info:
        load_dataset(nonexistent_path)

    # Verify the error message is user-friendly
    error_message = str(exc_info.value)
    assert "USER ACTION REQUIRED" in error_message, \
        "Error message should guide the user (NFR2 compliance)"


# ============================================================================
# Tests for validate_dataset_structure()
# ============================================================================

def test_validate_structure_valid(valid_dataset_path):
    """
    Test that validate_dataset_structure() passes for a valid dataset.

    This test verifies:
    - valid=True for a correctly formatted dataset
    - No validation errors
    - Correct row and column counts reported
    """
    df = load_dataset(valid_dataset_path)
    result = validate_dataset_structure(df)

    # Should pass validation
    assert result["valid"] is True, "Valid dataset should pass structure validation"
    assert len(result["validation_errors"]) == 0, "No errors should be reported"

    # Check reported dimensions
    assert result["row_count"] == 10, "Sample dataset has 10 rows"
    assert result["column_count"] == 40, "Dataset should have 40 columns"
    assert result["missing_values"] == 0, "Valid dataset has no missing values"


def test_validate_structure_wrong_columns(invalid_columns_path):
    """
    Test that validate_dataset_structure() detects wrong column count.

    This test verifies:
    - valid=False for dataset with wrong number of columns
    - Validation errors are reported
    - Error message mentions column count mismatch
    """
    df = load_dataset(invalid_columns_path)
    result = validate_dataset_structure(df)

    # Should fail validation
    assert result["valid"] is False, "Dataset with wrong columns should fail"
    assert len(result["validation_errors"]) > 0, "Errors should be reported"

    # Check that column count error is mentioned
    errors_text = " ".join(result["validation_errors"])
    assert "column" in errors_text.lower(), "Error should mention column issues"


def test_validate_structure_missing_values(missing_values_path):
    """
    Test that validate_dataset_structure() detects missing values.

    This test verifies:
    - Missing values are detected
    - Error message mentions the count of missing values
    """
    df = load_dataset(missing_values_path)
    result = validate_dataset_structure(df)

    # Should detect missing values
    assert result["missing_values"] > 0, "Should detect NaN values"
    assert result["valid"] is False, "Dataset with missing values should fail"


# ============================================================================
# Tests for validate_data_integrity()
# ============================================================================

def test_validate_integrity_valid(valid_dataset_path):
    """
    Test that validate_data_integrity() passes for valid data.

    This test verifies:
    - valid=True for data that satisfies all business rules
    - Event IDs are sequential
    - QV values are binary
    - QV sums equal 5
    """
    df = load_dataset(valid_dataset_path)
    result = validate_data_integrity(df)

    # Should pass all integrity checks
    assert result["valid"] is True, "Valid data should pass integrity validation"
    assert result["event_id_sequential"] is True, "Event IDs should be sequential"
    assert result["event_id_duplicates"] is False, "No duplicate IDs"
    assert result["qv_values_binary"] is True, "QV values should be binary"
    assert result["qv_sum_valid"] is True, "QV sums should equal 5"
    assert len(result["integrity_errors"]) == 0, "No errors should be reported"


def test_validate_integrity_qv_sum_invalid(invalid_values_path):
    """
    Test that validate_data_integrity() detects invalid QV sums.

    This test verifies:
    - QV sum validation detects rows where sum ≠ 5
    - Error message explains the quantum state constraint
    """
    df = load_dataset(invalid_values_path)
    result = validate_data_integrity(df)

    # Should fail validation
    assert result["valid"] is False, "Data with invalid QV sums should fail"
    assert result["qv_sum_valid"] is False, "Should detect QV sum violations"

    # Check error message mentions the constraint
    errors_text = " ".join(result["integrity_errors"])
    assert "5 active positions" in errors_text or "sum" in errors_text.lower(), \
        "Error should explain the QV sum=5 constraint"


def test_validate_integrity_qv_out_of_range(invalid_values_path):
    """
    Test that validate_data_integrity() detects non-binary QV values.

    This test verifies:
    - Binary validation detects values other than 0 or 1
    - Error message mentions binary constraint
    """
    df = load_dataset(invalid_values_path)
    result = validate_data_integrity(df)

    # Should detect non-binary values (row 2 has a '2' in QV_1)
    assert result["valid"] is False, "Data with non-binary values should fail"
    assert result["qv_values_binary"] is False, "Should detect non-binary values"


# ============================================================================
# Tests for clean_dataset()
# ============================================================================

def test_clean_dataset():
    """
    Test that clean_dataset() properly cleans data formatting issues.

    This test verifies:
    - Float columns are converted to int where appropriate
    - Returns a DataFrame
    - Original data is not modified (returns a copy)
    """
    # Create a test DataFrame with float types
    test_data = {
        "event-ID": [1.0, 2.0, 3.0],
        "QV_1": [1.0, 0.0, 1.0],
        "QV_2": [0.0, 1.0, 0.0]
    }
    df = pd.DataFrame(test_data)

    # Clean the dataset
    df_clean = clean_dataset(df)

    # Verify it's a DataFrame
    assert isinstance(df_clean, pd.DataFrame), "Should return a DataFrame"

    # Verify data types were converted
    assert df_clean["event-ID"].dtype == int, "event-ID should be converted to int"
    assert df_clean["QV_1"].dtype == int, "QV columns should be converted to int"

    # Verify original DataFrame was not modified (should be a copy)
    assert df["event-ID"].dtype == float, "Original DataFrame should be unchanged"


# ============================================================================
# Tests for save_processed_data()
# ============================================================================

def test_save_processed_data(valid_dataset_path, tmp_path):
    """
    Test that save_processed_data() successfully saves a dataset.

    This test verifies:
    - File is created at the specified location
    - Saved file can be read back
    - Data is preserved correctly

    Uses pytest's tmp_path fixture to avoid creating permanent test files.
    """
    # Load a valid dataset
    df = load_dataset(valid_dataset_path)

    # Save to a temporary location
    output_path = tmp_path / "test_output.csv"
    save_processed_data(df, output_path)

    # Verify file was created
    assert output_path.exists(), "Output file should be created"

    # Verify we can read it back
    df_loaded = pd.read_csv(output_path)
    assert len(df_loaded) == len(df), "Saved data should have same number of rows"
    assert list(df_loaded.columns) == list(df.columns), \
        "Saved data should have same columns"


# ============================================================================
# Integration Test
# ============================================================================

def test_full_pipeline(valid_dataset_path, tmp_path):
    """
    Integration test: Full pipeline from load to save.

    This test verifies the complete workflow:
    1. Load dataset
    2. Validate structure
    3. Validate integrity
    4. Clean dataset
    5. Save processed data

    This ensures all functions work together correctly.
    """
    # Step 1: Load
    df = load_dataset(valid_dataset_path)
    assert len(df) > 0, "Dataset should load"

    # Step 2: Validate structure
    structure_result = validate_dataset_structure(df)
    assert structure_result["valid"], "Structure should be valid"

    # Step 3: Validate integrity
    integrity_result = validate_data_integrity(df)
    assert integrity_result["valid"], "Integrity should be valid"

    # Step 4: Clean
    df_clean = clean_dataset(df)
    assert len(df_clean) == len(df), "Cleaning should preserve row count"

    # Step 5: Save
    output_path = tmp_path / "validated_dataset.csv"
    save_processed_data(df_clean, output_path)
    assert output_path.exists(), "Output file should exist"

    # Verify round-trip
    df_reloaded = load_dataset(output_path)
    assert len(df_reloaded) == len(df), "Round-trip should preserve data"
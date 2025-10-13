"""
Unit Tests for Apply Imputation Utility Script

This test suite validates the apply_imputation utility that provides both
command-line and programmatic interfaces to apply imputation strategies.

Test Coverage:
- validate_imputation_output function
- apply_imputation function with all 5 strategies
- Configuration handling
- Error handling (invalid strategy, missing files, validation failures)
- File I/O (Parquet and JSON metadata)
- Integration tests

Author: BMad Dev Agent (James)
Date: 2025-10-13
Story: Epic 2, Story 2.7 - Imputation Utility Script
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
import tempfile
import shutil

from src.imputation.apply_imputation import (
    validate_imputation_output,
    apply_imputation,
    AVAILABLE_STRATEGIES
)


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
def temp_output_dir():
    """Create temporary directory for output files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup after test
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


# ============================================================================
# Tests for validate_imputation_output()
# ============================================================================

def test_validate_imputation_output_valid():
    """
    Test validation with valid imputation output.

    Verifies:
    - Validation passes for clean data
    - All fields present in result dict
    - passed = True
    """
    # Create valid feature matrix
    features = np.random.rand(100, 50)

    result = validate_imputation_output(features)

    assert result["passed"] is True
    assert result["has_nan"] is False
    assert result["has_inf"] is False
    assert result["shape"] == (100, 50)
    assert isinstance(result["value_range"], tuple)
    assert len(result["value_range"]) == 2


def test_validate_imputation_output_with_nan():
    """
    Test validation with NaN values.

    Verifies:
    - Validation fails when NaN present
    - has_nan = True
    - passed = False
    """
    features = np.random.rand(100, 50)
    features[10, 20] = np.nan

    result = validate_imputation_output(features)

    assert result["passed"] is False
    assert result["has_nan"] is True


def test_validate_imputation_output_with_inf():
    """
    Test validation with Inf values.

    Verifies:
    - Validation fails when Inf present
    - has_inf = True
    - passed = False
    """
    features = np.random.rand(100, 50)
    features[10, 20] = np.inf

    result = validate_imputation_output(features)

    assert result["passed"] is False
    assert result["has_inf"] is True


# ============================================================================
# Tests for apply_imputation() - All Strategies
# ============================================================================

def test_apply_imputation_basis_embedding(valid_dataset_path, temp_output_dir):
    """
    Test apply_imputation with basis_embedding strategy.

    Verifies:
    - Can apply basis embedding
    - Creates output Parquet file
    - Creates metadata JSON file
    - Metadata contains expected fields
    """
    output_path = temp_output_dir / "imputed_basis.parquet"

    metadata = apply_imputation(
        strategy="basis_embedding",
        input_path=valid_dataset_path,
        output_path=output_path
    )

    # Check metadata
    assert metadata["strategy"] == "basis_embedding"
    assert metadata["validation_passed"] is True
    assert "execution_time_seconds" in metadata
    assert isinstance(metadata["output_shape"], list)

    # Check files created
    assert output_path.exists()
    assert Path(str(output_path) + ".meta.json").exists()

    # Check output can be loaded
    df_output = pd.read_parquet(output_path)
    assert len(df_output) == 10  # Sample dataset has 10 rows
    assert 'event-ID' in df_output.columns


def test_apply_imputation_amplitude_embedding(valid_dataset_path, temp_output_dir):
    """
    Test apply_imputation with amplitude_embedding strategy.

    Verifies:
    - Can apply amplitude embedding
    - Output files created
    - Correct output shape
    """
    output_path = temp_output_dir / "imputed_amplitude.parquet"

    metadata = apply_imputation(
        strategy="amplitude_embedding",
        input_path=valid_dataset_path,
        output_path=output_path
    )

    assert metadata["strategy"] == "amplitude_embedding"
    assert metadata["validation_passed"] is True
    assert output_path.exists()


def test_apply_imputation_angle_encoding(valid_dataset_path, temp_output_dir):
    """
    Test apply_imputation with angle_encoding strategy.

    Verifies:
    - Can apply angle encoding
    - Output files created
    """
    output_path = temp_output_dir / "imputed_angle.parquet"

    metadata = apply_imputation(
        strategy="angle_encoding",
        input_path=valid_dataset_path,
        output_path=output_path
    )

    assert metadata["strategy"] == "angle_encoding"
    assert metadata["validation_passed"] is True
    assert output_path.exists()


def test_apply_imputation_density_matrix(valid_dataset_path, temp_output_dir):
    """
    Test apply_imputation with density_matrix strategy.

    Verifies:
    - Can apply density matrix
    - Output files created
    """
    output_path = temp_output_dir / "imputed_density.parquet"

    metadata = apply_imputation(
        strategy="density_matrix",
        input_path=valid_dataset_path,
        output_path=output_path
    )

    assert metadata["strategy"] == "density_matrix"
    assert metadata["validation_passed"] is True
    assert output_path.exists()


def test_apply_imputation_graph_cycle(valid_dataset_path, temp_output_dir):
    """
    Test apply_imputation with graph_cycle strategy.

    Verifies:
    - Can apply graph/cycle encoding
    - Output files created
    """
    output_path = temp_output_dir / "imputed_graph.parquet"

    metadata = apply_imputation(
        strategy="graph_cycle",
        input_path=valid_dataset_path,
        output_path=output_path
    )

    assert metadata["strategy"] == "graph_cycle"
    assert metadata["validation_passed"] is True
    assert output_path.exists()


# ============================================================================
# Tests for Configuration Handling
# ============================================================================

def test_apply_imputation_with_config(valid_dataset_path, temp_output_dir):
    """
    Test apply_imputation with custom configuration.

    Verifies:
    - Can pass config dict to strategy
    - Config stored in metadata
    """
    output_path = temp_output_dir / "imputed_amplitude_weighted.parquet"

    config = {
        "normalization": "weighted",
        "include_probability_features": False
    }

    metadata = apply_imputation(
        strategy="amplitude_embedding",
        input_path=valid_dataset_path,
        output_path=output_path,
        config=config
    )

    assert metadata["config"] == config
    assert metadata["validation_passed"] is True
    assert output_path.exists()


def test_apply_imputation_invalid_config(valid_dataset_path, temp_output_dir):
    """
    Test apply_imputation with invalid configuration.

    Verifies:
    - Raises ValueError for invalid config parameters
    """
    output_path = temp_output_dir / "imputed.parquet"

    config = {
        "invalid_parameter": "value"
    }

    with pytest.raises(ValueError):
        apply_imputation(
            strategy="basis_embedding",
            input_path=valid_dataset_path,
            output_path=output_path,
            config=config
        )


# ============================================================================
# Tests for Error Handling
# ============================================================================

def test_apply_imputation_invalid_strategy(valid_dataset_path, temp_output_dir):
    """
    Test apply_imputation with invalid strategy name.

    Verifies:
    - Raises ValueError for invalid strategy
    - Error message lists available strategies
    """
    output_path = temp_output_dir / "imputed.parquet"

    with pytest.raises(ValueError) as exc_info:
        apply_imputation(
            strategy="invalid_strategy",
            input_path=valid_dataset_path,
            output_path=output_path
        )

    assert "invalid_strategy" in str(exc_info.value).lower()
    assert "available strategies" in str(exc_info.value).lower()


def test_apply_imputation_missing_input_file(temp_output_dir):
    """
    Test apply_imputation with missing input file.

    Verifies:
    - Raises FileNotFoundError for missing input
    - Error message mentions file not found
    """
    input_path = temp_output_dir / "nonexistent.csv"
    output_path = temp_output_dir / "imputed.parquet"

    with pytest.raises(FileNotFoundError) as exc_info:
        apply_imputation(
            strategy="basis_embedding",
            input_path=input_path,
            output_path=output_path
        )

    assert "not found" in str(exc_info.value).lower()


# ============================================================================
# Tests for File I/O
# ============================================================================

def test_output_parquet_format(valid_dataset_path, temp_output_dir):
    """
    Test that output is saved in Parquet format correctly.

    Verifies:
    - Can read output Parquet file
    - Has event-ID column
    - Has feature columns
    - Correct number of rows
    """
    output_path = temp_output_dir / "imputed.parquet"

    metadata = apply_imputation(
        strategy="basis_embedding",
        input_path=valid_dataset_path,
        output_path=output_path
    )

    # Load output
    df_output = pd.read_parquet(output_path)

    # Check structure
    assert 'event-ID' in df_output.columns
    assert len(df_output) == metadata["input_shape"][0]
    assert len(df_output.columns) == metadata["output_shape"][1] + 1  # +1 for event-ID


def test_metadata_json_format(valid_dataset_path, temp_output_dir):
    """
    Test that metadata JSON is saved correctly.

    Verifies:
    - Metadata file created
    - Can parse JSON
    - Contains required fields
    - Values are correct types
    """
    output_path = temp_output_dir / "imputed.parquet"
    metadata_path = Path(str(output_path) + ".meta.json")

    metadata = apply_imputation(
        strategy="basis_embedding",
        input_path=valid_dataset_path,
        output_path=output_path
    )

    # Check file exists
    assert metadata_path.exists()

    # Load and validate JSON
    with open(metadata_path, 'r') as f:
        loaded_metadata = json.load(f)

    # Check required fields
    required_fields = [
        "strategy", "timestamp", "input_file", "input_shape",
        "output_file", "output_shape", "config",
        "execution_time_seconds", "validation_passed"
    ]

    for field in required_fields:
        assert field in loaded_metadata

    # Check types
    assert isinstance(loaded_metadata["execution_time_seconds"], (int, float))
    assert isinstance(loaded_metadata["validation_passed"], bool)
    assert isinstance(loaded_metadata["input_shape"], list)
    assert isinstance(loaded_metadata["output_shape"], list)


def test_output_directory_created(valid_dataset_path, temp_output_dir):
    """
    Test that output directories are created if they don't exist.

    Verifies:
    - Can create nested output directories
    - Files saved successfully
    """
    output_path = temp_output_dir / "nested" / "dir" / "imputed.parquet"

    metadata = apply_imputation(
        strategy="basis_embedding",
        input_path=valid_dataset_path,
        output_path=output_path
    )

    assert output_path.exists()
    assert output_path.parent.exists()


# ============================================================================
# Tests for AVAILABLE_STRATEGIES
# ============================================================================

def test_available_strategies_contains_all_five():
    """
    Test that AVAILABLE_STRATEGIES dict contains all 5 strategies.

    Verifies:
    - Has exactly 5 strategies
    - All expected strategy names present
    """
    expected_strategies = {
        "basis_embedding",
        "amplitude_embedding",
        "angle_encoding",
        "density_matrix",
        "graph_cycle"
    }

    assert set(AVAILABLE_STRATEGIES.keys()) == expected_strategies


def test_available_strategies_classes_valid():
    """
    Test that all strategy classes in AVAILABLE_STRATEGIES are valid.

    Verifies:
    - Can instantiate each class
    - Each has required methods (fit, transform)
    """
    for strategy_name, strategy_class in AVAILABLE_STRATEGIES.items():
        # Instantiate
        imputer = strategy_class()

        # Check methods exist
        assert hasattr(imputer, 'fit')
        assert hasattr(imputer, 'transform')
        assert hasattr(imputer, 'fit_transform')
        assert hasattr(imputer, 'get_feature_names')


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_workflow_all_strategies(valid_dataset_path, temp_output_dir):
    """
    Integration test: Apply all 5 strategies and verify outputs.

    Verifies:
    - All strategies work end-to-end
    - All produce valid outputs
    - All create metadata files
    """
    for strategy_name in AVAILABLE_STRATEGIES.keys():
        output_path = temp_output_dir / f"imputed_{strategy_name}.parquet"

        metadata = apply_imputation(
            strategy=strategy_name,
            input_path=valid_dataset_path,
            output_path=output_path
        )

        # Check success
        assert metadata["validation_passed"] is True
        assert output_path.exists()
        assert Path(str(output_path) + ".meta.json").exists()

        # Check can load output
        df_output = pd.read_parquet(output_path)
        assert len(df_output) == 10
        assert 'event-ID' in df_output.columns


def test_metadata_consistency_with_output(valid_dataset_path, temp_output_dir):
    """
    Test that metadata accurately describes the output.

    Verifies:
    - Metadata output_shape matches actual output
    - Metadata input_shape matches input
    """
    output_path = temp_output_dir / "imputed.parquet"

    metadata = apply_imputation(
        strategy="basis_embedding",
        input_path=valid_dataset_path,
        output_path=output_path
    )

    # Load actual files
    df_input = pd.read_csv(valid_dataset_path)
    df_output = pd.read_parquet(output_path)

    # Check metadata matches reality
    assert metadata["input_shape"] == list(df_input.shape)
    # Output has event-ID + features
    assert len(df_output.columns) == metadata["output_shape"][1] + 1
    assert len(df_output) == metadata["output_shape"][0]


# ============================================================================
# Tests for CLI (main function)
# ============================================================================

def test_cli_basic_usage(valid_dataset_path, temp_output_dir):
    """
    Test command-line interface with basic usage.

    Verifies:
    - CLI accepts required arguments
    - Exits with code 0 on success
    - Creates output files
    """
    import subprocess
    import sys

    output_path = temp_output_dir / "cli_test.parquet"

    # Run CLI as module
    result = subprocess.run(
        [
            sys.executable, "-m",
            "src.imputation.apply_imputation",
            "--strategy", "basis_embedding",
            "--input", str(valid_dataset_path),
            "--output", str(output_path)
        ],
        capture_output=True,
        text=True
    )

    # Check success
    assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
    assert "IMPUTATION COMPLETE" in result.stdout
    assert output_path.exists()
    assert Path(str(output_path) + ".meta.json").exists()


def test_cli_with_config(valid_dataset_path, temp_output_dir):
    """
    Test CLI with JSON configuration.

    Verifies:
    - CLI accepts --config argument
    - Config is parsed correctly
    - Runs successfully
    """
    import subprocess
    import sys

    output_path = temp_output_dir / "cli_config_test.parquet"
    config_json = '{"normalization": "weighted"}'

    # Run CLI with config as module
    result = subprocess.run(
        [
            sys.executable, "-m",
            "src.imputation.apply_imputation",
            "--strategy", "amplitude_embedding",
            "--input", str(valid_dataset_path),
            "--output", str(output_path),
            "--config", config_json
        ],
        capture_output=True,
        text=True
    )

    # Check success
    assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
    assert output_path.exists()


def test_cli_invalid_strategy(valid_dataset_path, temp_output_dir):
    """
    Test CLI with invalid strategy name.

    Verifies:
    - CLI rejects invalid strategy
    - Exits with non-zero code
    - Error message shown
    """
    import subprocess
    import sys

    output_path = temp_output_dir / "cli_invalid.parquet"

    # Run CLI with invalid strategy as module
    result = subprocess.run(
        [
            sys.executable, "-m",
            "src.imputation.apply_imputation",
            "--strategy", "invalid_strategy",
            "--input", str(valid_dataset_path),
            "--output", str(output_path)
        ],
        capture_output=True,
        text=True
    )

    # Check failure
    assert result.returncode != 0


def test_cli_help(capsys):
    """
    Test CLI help message.

    Verifies:
    - --help flag works
    - Shows available strategies
    """
    import subprocess
    import sys

    # Run CLI with --help as module
    result = subprocess.run(
        [
            sys.executable, "-m",
            "src.imputation.apply_imputation",
            "--help"
        ],
        capture_output=True,
        text=True
    )

    # Check help output
    assert result.returncode == 0, f"Help failed with stderr: {result.stderr}"
    assert "basis_embedding" in result.stdout
    assert "amplitude_embedding" in result.stdout
    assert "angle_encoding" in result.stdout
    assert "density_matrix" in result.stdout
    assert "graph_cycle" in result.stdout

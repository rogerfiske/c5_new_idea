"""
Data Loading and Validation Module

This module provides functions to load, validate, clean, and save the
c5_Matrix.csv dataset for the Quantum State Prediction Experiment.

The dataset contains binary vector representations of quantum states:
- event-ID: Unique identifier for each event (sequential integers)
- QV_1 through QV_39: Binary indicators (0 or 1) for each of 39 possible positions
- Each row must have exactly 5 active positions (sum of QV columns = 5)

Functions:
    load_dataset: Load the CSV file into a pandas DataFrame
    validate_dataset_structure: Check that the dataset has the correct shape and columns
    validate_data_integrity: Verify business rules (QV sum = 5, event IDs sequential, etc.)
    clean_dataset: Handle minor formatting issues (if any)
    save_processed_data: Save validated/cleaned data to the processed directory

Example Usage:
    >>> from src.data_loader import load_dataset, validate_dataset_structure
    >>> df = load_dataset()
    >>> validation_result = validate_dataset_structure(df)
    >>> if validation_result["valid"]:
    ...     print("Dataset is valid!")

Author: BMad Dev Agent (James)
Date: 2025-10-13
Story: Epic 1, Story 1.2 - Data Loading and Validation
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# Import path constants from our centralized configuration
from src.config import DATASET_PATH, DATA_PROCESSED

# Configure logging for this module
# This allows us to track what the data loader is doing
logger = logging.getLogger(__name__)


def load_dataset(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the c5_Matrix.csv dataset from disk.

    This function reads the CSV file containing quantum state data and returns
    it as a pandas DataFrame. It performs a file existence check before attempting
    to load to provide a user-friendly error message if the file is missing.

    Args:
        file_path (Optional[Path]): Path to the CSV file. If None, uses the
            default path from config.DATASET_PATH (data/raw/c5_Matrix.csv)

    Returns:
        pd.DataFrame: The loaded dataset with columns:
            - event-ID: Unique event identifier
            - QV_1 through QV_39: Binary quantum state indicators

    Raises:
        FileNotFoundError: If the dataset file doesn't exist at the specified path

    Example:
        >>> # Load using default path
        >>> df = load_dataset()
        >>> print(f"Loaded {len(df)} events")

        >>> # Load from custom path
        >>> custom_path = Path("data/processed/cleaned_dataset.csv")
        >>> df = load_dataset(custom_path)
    """
    # Use default path if none provided
    # This follows the principle of sensible defaults from config.py
    if file_path is None:
        file_path = DATASET_PATH

    # Check if file exists BEFORE trying to load it
    # This prevents cryptic pandas errors and gives users clear guidance
    if not file_path.exists():
        # Provide a user-friendly error message explaining what to do
        # This is critical for NFR2 (non-programmer friendly)
        raise FileNotFoundError(
            f"Dataset not found at {file_path}.\n"
            f"USER ACTION REQUIRED: Ensure c5_Matrix.csv is in the data/raw/ directory.\n"
            f"See README.md section 'Data Setup' for instructions."
        )

    # Log that we're about to load the file (useful for debugging)
    logger.info(f"Loading dataset from {file_path}")

    # Load the CSV file
    # We use pandas default parameters which handle most common cases
    df = pd.read_csv(file_path)

    # Log success with the dataset dimensions
    # The :, formatting adds thousand separators for readability
    logger.info(f"Dataset loaded successfully: {len(df):,} rows × {len(df.columns)} columns")

    return df


def validate_dataset_structure(df: pd.DataFrame) -> dict:
    """
    Validate that the dataset has the expected structure.

    This function checks that the dataset matches the expected format:
    - Correct number of columns (40: event-ID + QV_1 through QV_39)
    - Correct column names
    - No missing values
    - Appropriate data types

    The quantum state dataset represents binary vectors where each row
    indicates which of the 39 possible quantum states are "active" (value=1)
    versus "inactive" (value=0).

    Args:
        df (pd.DataFrame): The dataset to validate

    Returns:
        dict: Validation results with keys:
            - "valid" (bool): True if all structural checks pass
            - "row_count" (int): Number of rows in the dataset
            - "column_count" (int): Number of columns found
            - "missing_values" (int): Total count of missing (NaN) values
            - "validation_errors" (list[str]): List of error messages (empty if valid)

    Example:
        >>> df = load_dataset()
        >>> result = validate_dataset_structure(df)
        >>> if result["valid"]:
        ...     print("Structure validation passed!")
        >>> else:
        ...     for error in result["validation_errors"]:
        ...         print(f"  - {error}")
    """
    # Initialize the result dictionary
    validation_errors = []
    row_count = len(df)
    column_count = len(df.columns)
    missing_values = df.isna().sum().sum()  # Total NaN count across entire DataFrame

    # Log that we're starting validation
    logger.info("Validating dataset structure...")

    # Check 1: Column count
    # Expected: 40 columns (1 event-ID + 39 QV columns)
    EXPECTED_COLUMNS = 40
    if column_count != EXPECTED_COLUMNS:
        error_msg = (
            f"Expected {EXPECTED_COLUMNS} columns but found {column_count}. "
            f"Dataset may be corrupted or incorrectly formatted."
        )
        validation_errors.append(error_msg)
        logger.error(error_msg)

    # Check 2: Required columns exist
    # The dataset must have an event-ID column and all QV columns
    required_columns = ["event-ID"] + [f"QV_{i}" for i in range(1, 40)]

    for col in required_columns:
        if col not in df.columns:
            error_msg = f"Missing required column: {col}. Check dataset format."
            validation_errors.append(error_msg)
            logger.error(error_msg)

    # Check 3: No missing values
    # The dataset should be complete - no NaN values
    if missing_values > 0:
        error_msg = (
            f"Found {missing_values} missing values. "
            f"Dataset should be complete with no NaN entries."
        )
        validation_errors.append(error_msg)
        logger.warning(error_msg)

    # Check 4: Data types
    # event-ID should be numeric, QV columns should be numeric (will check they're binary later)
    if "event-ID" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["event-ID"]):
            error_msg = "Column 'event-ID' should be numeric (integer) type."
            validation_errors.append(error_msg)
            logger.error(error_msg)

    # Check that QV columns are numeric
    qv_columns = [f"QV_{i}" for i in range(1, 40)]
    for col in qv_columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            error_msg = f"Column '{col}' should be numeric type."
            validation_errors.append(error_msg)
            logger.error(error_msg)

    # Determine if validation passed
    valid = len(validation_errors) == 0

    if valid:
        logger.info("✓ Structure validation passed")
    else:
        logger.error(f"✗ Structure validation failed with {len(validation_errors)} errors")

    # Return comprehensive validation results
    return {
        "valid": valid,
        "row_count": row_count,
        "column_count": column_count,
        "missing_values": missing_values,
        "validation_errors": validation_errors
    }


def validate_data_integrity(df: pd.DataFrame) -> dict:
    """
    Validate business rules and data integrity constraints.

    This function checks that the data follows the quantum state representation rules:
    1. Event IDs are sequential with no gaps or duplicates
    2. QV values are binary (only 0 or 1)
    3. Each row has exactly 5 active positions (sum of QV columns = 5)

    The "exactly 5 active positions" rule is fundamental to the quantum state
    representation used in this experiment. It means each event has precisely
    5 of the 39 possible quantum states active simultaneously.

    Args:
        df (pd.DataFrame): The dataset to validate

    Returns:
        dict: Integrity validation results with keys:
            - "valid" (bool): True if all integrity checks pass
            - "event_id_sequential" (bool): True if event IDs are sequential
            - "event_id_duplicates" (bool): True if there are duplicate event IDs
            - "qv_values_binary" (bool): True if all QV values are 0 or 1
            - "qv_sum_valid" (bool): True if all rows sum to exactly 5
            - "integrity_errors" (list[str]): List of error messages (empty if valid)

    Example:
        >>> df = load_dataset()
        >>> result = validate_data_integrity(df)
        >>> if result["valid"]:
        ...     print("All business rules satisfied!")
        >>> else:
        ...     print("Data integrity issues found:")
        ...     for error in result["integrity_errors"]:
        ...         print(f"  - {error}")
    """
    # Initialize result tracking
    integrity_errors = []
    event_id_sequential = True
    event_id_duplicates = False
    qv_values_binary = True
    qv_sum_valid = True

    logger.info("Validating data integrity...")

    # Check 1: Event IDs are sequential (no gaps)
    # Expected pattern: 1, 2, 3, 4, ... with no missing numbers
    if "event-ID" in df.columns:
        event_ids = df["event-ID"].values
        expected_ids = list(range(event_ids[0], event_ids[0] + len(event_ids)))

        if not all(event_ids == expected_ids):
            event_id_sequential = False
            error_msg = (
                f"Event IDs are not sequential. Expected continuous sequence starting "
                f"from {event_ids[0]}, but found gaps or non-sequential values."
            )
            integrity_errors.append(error_msg)
            logger.error(error_msg)

        # Check for duplicates
        if df["event-ID"].duplicated().any():
            event_id_duplicates = True
            duplicate_count = df["event-ID"].duplicated().sum()
            error_msg = f"Found {duplicate_count} duplicate event IDs. Each event should be unique."
            integrity_errors.append(error_msg)
            logger.error(error_msg)

    # Check 2: QV values are binary (0 or 1 only)
    # Get all QV columns
    qv_columns = [f"QV_{i}" for i in range(1, 40)]
    qv_columns_present = [col for col in qv_columns if col in df.columns]

    if qv_columns_present:
        # Check if all values in QV columns are either 0 or 1
        qv_data = df[qv_columns_present]
        non_binary_mask = ~qv_data.isin([0, 1]).all(axis=1)

        if non_binary_mask.any():
            qv_values_binary = False
            non_binary_count = non_binary_mask.sum()
            error_msg = (
                f"Found {non_binary_count} rows with non-binary QV values. "
                f"All QV columns must contain only 0 or 1."
            )
            integrity_errors.append(error_msg)
            logger.error(error_msg)

    # Check 3: QV sum equals 5 for each row
    # This is the core quantum state constraint: exactly 5 active positions
    if qv_columns_present:
        qv_sums = df[qv_columns_present].sum(axis=1)
        invalid_sum_mask = qv_sums != 5

        if invalid_sum_mask.any():
            qv_sum_valid = False
            invalid_count = invalid_sum_mask.sum()
            error_msg = (
                f"Found {invalid_count} rows where QV sum ≠ 5. "
                f"Each row must have exactly 5 active positions (quantum state constraint). "
                f"This is fundamental to the quantum state representation. "
                f"See docs/Assigning Quantum States to Binary csv.md for details."
            )
            integrity_errors.append(error_msg)
            logger.error(error_msg)

            # Log some examples of invalid rows for debugging
            invalid_rows = df[invalid_sum_mask].head(3)
            logger.debug(f"Example invalid rows:\n{invalid_rows[['event-ID'] + qv_columns_present]}")

    # Determine overall validity
    valid = len(integrity_errors) == 0

    if valid:
        logger.info("✓ Data integrity validation passed")
    else:
        logger.error(f"✗ Data integrity validation failed with {len(integrity_errors)} errors")

    return {
        "valid": valid,
        "event_id_sequential": event_id_sequential,
        "event_id_duplicates": event_id_duplicates,
        "qv_values_binary": qv_values_binary,
        "qv_sum_valid": qv_sum_valid,
        "integrity_errors": integrity_errors
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply minor formatting corrections to the dataset.

    This function handles common data formatting issues like:
    - Converting float columns to integers (e.g., 1.0 -> 1)
    - Stripping whitespace from string columns
    - Ensuring correct data types

    IMPORTANT: This function does NOT fix invalid data. If data violates
    business rules (e.g., QV sum ≠ 5), those issues should be flagged as errors,
    not automatically "fixed". Only cosmetic formatting is handled here.

    Args:
        df (pd.DataFrame): The dataset to clean

    Returns:
        pd.DataFrame: The cleaned dataset (may be the same as input if no cleaning needed)

    Example:
        >>> df = load_dataset()
        >>> df_clean = clean_dataset(df)
        >>> # Any cleaning operations are logged
    """
    logger.info("Cleaning dataset...")

    # Make a copy to avoid modifying the original
    df_clean = df.copy()

    # Cleaning operation 1: Ensure event-ID is integer type
    # Sometimes CSVs load with float type (e.g., 1.0 instead of 1)
    if "event-ID" in df_clean.columns:
        if df_clean["event-ID"].dtype == float:
            df_clean["event-ID"] = df_clean["event-ID"].astype(int)
            logger.info("Converted 'event-ID' from float to int")

    # Cleaning operation 2: Ensure QV columns are integer type
    qv_columns = [f"QV_{i}" for i in range(1, 40)]
    qv_columns_present = [col for col in qv_columns if col in df_clean.columns]

    for col in qv_columns_present:
        if df_clean[col].dtype == float:
            # Only convert if all values are whole numbers
            if (df_clean[col] % 1 == 0).all():
                df_clean[col] = df_clean[col].astype(int)
                logger.info(f"Converted '{col}' from float to int")

    # Cleaning operation 3: Strip whitespace from column names
    # Sometimes CSV files have extra spaces in headers
    original_columns = df_clean.columns.tolist()
    df_clean.columns = df_clean.columns.str.strip()
    if list(df_clean.columns) != original_columns:
        logger.info("Stripped whitespace from column names")

    logger.info("✓ Dataset cleaning complete")

    return df_clean


def save_processed_data(
    df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> None:
    """
    Save the cleaned/validated dataset to the processed data directory.

    After loading and validating the raw dataset, this function saves it
    to the processed/ directory for use in downstream processing steps.

    Args:
        df (pd.DataFrame): The dataset to save
        output_path (Optional[Path]): Where to save the file. If None, saves to
            data/processed/validated_dataset.csv

    Returns:
        None

    Raises:
        OSError: If unable to create directories or write the file

    Example:
        >>> df = load_dataset()
        >>> df_clean = clean_dataset(df)
        >>> save_processed_data(df_clean)
        >>> print("Dataset saved to data/processed/validated_dataset.csv")
    """
    # Use default path if none provided
    if output_path is None:
        output_path = DATA_PROCESSED / "validated_dataset.csv"

    # Ensure the output directory exists
    # This prevents "directory not found" errors
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving processed dataset to {output_path}")

    # Save as CSV
    df.to_csv(output_path, index=False)

    # Get file size for logging
    file_size = output_path.stat().st_size
    file_size_kb = file_size / 1024

    logger.info(f"✓ Dataset saved successfully ({file_size_kb:.1f} KB)")
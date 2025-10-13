"""
Apply Imputation Utility Script

This module provides both command-line and programmatic interfaces to apply
any of the 5 quantum-inspired imputation strategies to the c5_Matrix dataset.

Supports:
- Basis Embedding
- Amplitude Embedding
- Angle Encoding
- Density Matrix Embedding
- Graph/Cycle Encoding

Usage:
    Command-line:
        python src/imputation/apply_imputation.py \\
            --strategy basis_embedding \\
            --input data/raw/c5_Matrix.csv \\
            --output data/processed/imputed_basis.parquet

    Programmatic:
        from src.imputation.apply_imputation import apply_imputation
        metadata = apply_imputation(
            strategy="basis_embedding",
            input_path=Path("data/raw/c5_Matrix.csv"),
            output_path=Path("data/processed/imputed_basis.parquet")
        )

Author: BMad Dev Agent (James)
Date: 2025-10-13
Story: Epic 2, Story 2.7 - Imputation Utility Script
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.data_loader import load_dataset
from src.imputation.basis_embedding import BasisEmbedding
from src.imputation.amplitude_embedding import AmplitudeEmbedding
from src.imputation.angle_encoding import AngleEncoding
from src.imputation.density_matrix import DensityMatrixEmbedding
from src.imputation.graph_cycle_encoding import GraphCycleEncoding

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Available imputation strategies
AVAILABLE_STRATEGIES = {
    "basis_embedding": BasisEmbedding,
    "amplitude_embedding": AmplitudeEmbedding,
    "angle_encoding": AngleEncoding,
    "density_matrix": DensityMatrixEmbedding,
    "graph_cycle": GraphCycleEncoding
}


def validate_imputation_output(features: np.ndarray) -> Dict[str, Any]:
    """
    Validate imputation output quality.

    Checks for common issues:
    - NaN values (computation errors)
    - Inf values (numerical overflow)
    - Expected dimensions (must be 2D array)
    - Reasonable value ranges (no extreme values)

    Args:
        features: Imputed feature matrix of shape (n_samples, n_features)

    Returns:
        Dictionary with validation results:
        - has_nan: bool, True if any NaN values present
        - has_inf: bool, True if any Inf values present
        - shape: tuple, output shape
        - value_range: tuple, (min, max) values
        - passed: bool, True if all checks passed

    Examples:
        >>> features = imputer.fit_transform(df)
        >>> result = validate_imputation_output(features)
        >>> if not result["passed"]:
        ...     print(f"Validation failed: {result}")
    """
    validation_result = {
        "has_nan": bool(np.isnan(features).any()),
        "has_inf": bool(np.isinf(features).any()),
        "shape": features.shape,
        "value_range": (float(features.min()), float(features.max())),
        "passed": True
    }

    # Check for invalid values
    if validation_result["has_nan"]:
        logger.error("Validation failed: Output contains NaN values")
        validation_result["passed"] = False

    if validation_result["has_inf"]:
        logger.error("Validation failed: Output contains Inf values")
        validation_result["passed"] = False

    # Check dimensions
    if len(features.shape) != 2:
        logger.error(f"Validation failed: Expected 2D array, got shape {features.shape}")
        validation_result["passed"] = False

    return validation_result


def apply_imputation(
    strategy: str,
    input_path: Path,
    output_path: Path,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Apply imputation strategy to dataset (programmatic interface).

    This function provides a Python API to apply any of the 5 imputation
    strategies without using the command-line interface. It handles the
    complete workflow: load data, fit imputer, transform, validate, save.

    Args:
        strategy: Name of imputation strategy, one of:
            - "basis_embedding"
            - "amplitude_embedding"
            - "angle_encoding"
            - "density_matrix"
            - "graph_cycle"
        input_path: Path to input CSV file (raw c5_Matrix data)
        output_path: Path to save imputed data (Parquet format)
        config: Optional configuration dict for the strategy.
                Parameters depend on the strategy chosen.

    Returns:
        Metadata dictionary with execution details:
        - strategy: str, strategy name
        - timestamp: str, ISO format timestamp
        - input_file: str, input file path
        - input_shape: list, [n_samples, n_columns]
        - output_file: str, output file path
        - output_shape: list, [n_samples, n_features]
        - config: dict, strategy configuration
        - execution_time_seconds: float, time taken
        - validation_passed: bool, whether output validation passed

    Raises:
        ValueError: If strategy name is invalid
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If imputation or validation fails

    Examples:
        >>> from pathlib import Path
        >>> from src.imputation.apply_imputation import apply_imputation
        >>>
        >>> # Apply basis embedding with default config
        >>> metadata = apply_imputation(
        ...     strategy="basis_embedding",
        ...     input_path=Path("data/raw/c5_Matrix.csv"),
        ...     output_path=Path("data/processed/imputed_basis.parquet")
        ... )
        >>> print(f"Took {metadata['execution_time_seconds']:.2f}s")
        >>>
        >>> # Apply amplitude embedding with custom config
        >>> metadata = apply_imputation(
        ...     strategy="amplitude_embedding",
        ...     input_path=Path("data/raw/c5_Matrix.csv"),
        ...     output_path=Path("data/processed/imputed_amplitude.parquet"),
        ...     config={"normalization": "weighted"}
        ... )

    Notes:
        - Input file must be valid c5_Matrix format (validated by data_loader)
        - Output saved in Parquet format for efficiency
        - Metadata saved as JSON sidecar: {output_path}.meta.json
        - Logging to console for progress tracking
    """
    start_time = time.time()

    # Validate strategy name
    if strategy not in AVAILABLE_STRATEGIES:
        raise ValueError(
            f"Invalid strategy '{strategy}'. "
            f"Available strategies: {list(AVAILABLE_STRATEGIES.keys())}"
        )

    # Validate input path
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\\n"
            f"Please ensure the c5_Matrix.csv file exists at the specified location."
        )

    logger.info(f"Starting imputation with strategy: {strategy}")
    logger.info(f"Input file: {input_path}")
    logger.info(f"Output file: {output_path}")

    # Load dataset
    logger.info("Loading dataset...")
    try:
        df = load_dataset(input_path)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Instantiate imputer
    logger.info(f"Instantiating {strategy} imputer...")
    imputer_class = AVAILABLE_STRATEGIES[strategy]

    if config is None:
        imputer = imputer_class()
    else:
        try:
            imputer = imputer_class(**config)
        except TypeError as e:
            logger.error(f"Invalid config for {strategy}: {e}")
            raise ValueError(f"Invalid configuration parameters: {e}")

    # Fit and transform
    logger.info("Fitting imputer to data...")
    try:
        imputer.fit(df)
        logger.info("Transforming data...")
        features = imputer.transform(df)
        logger.info(f"Transformation complete: output shape {features.shape}")
    except Exception as e:
        logger.error(f"Imputation failed: {e}")
        raise RuntimeError(f"Imputation failed: {e}")

    # Validate output
    logger.info("Validating imputation output...")
    validation_result = validate_imputation_output(features)

    if not validation_result["passed"]:
        logger.error("Validation failed!")
        raise RuntimeError(
            f"Imputation validation failed: {validation_result}"
        )

    logger.info("Validation passed ✓")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save imputed data
    logger.info(f"Saving imputed data to {output_path}...")
    try:
        # Create DataFrame with event-ID and features
        feature_names = imputer.get_feature_names()
        output_df = pd.DataFrame(features, columns=feature_names)
        output_df.insert(0, 'event-ID', df['event-ID'].values)

        # Save as Parquet
        output_df.to_parquet(output_path, index=False)
        logger.info(f"Saved imputed data: {len(output_df)} rows, {len(output_df.columns)} columns")
    except Exception as e:
        logger.error(f"Failed to save output: {e}")
        raise

    # Calculate execution time
    execution_time = time.time() - start_time

    # Create metadata
    metadata = {
        "strategy": strategy,
        "timestamp": pd.Timestamp.now().isoformat(),
        "input_file": str(input_path),
        "input_shape": list(df.shape),
        "output_file": str(output_path),
        "output_shape": list(features.shape),
        "config": config if config is not None else {},
        "execution_time_seconds": round(execution_time, 3),
        "validation_passed": validation_result["passed"],
        "validation_details": {
            "value_range": validation_result["value_range"],
            "has_nan": validation_result["has_nan"],
            "has_inf": validation_result["has_inf"]
        }
    }

    # Save metadata
    metadata_path = Path(str(output_path) + ".meta.json")
    logger.info(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Imputation complete! Execution time: {execution_time:.2f}s")

    return metadata


def main():
    """
    Command-line interface for applying imputation strategies.

    Parses command-line arguments and calls apply_imputation().

    Arguments:
        --strategy: Imputation strategy name (required)
        --input: Path to input CSV file (required)
        --output: Path to output Parquet file (required)
        --config: JSON string with strategy configuration (optional)

    Examples:
        # Basic usage
        python src/imputation/apply_imputation.py \\
            --strategy basis_embedding \\
            --input data/raw/c5_Matrix.csv \\
            --output data/processed/imputed_basis.parquet

        # With configuration
        python src/imputation/apply_imputation.py \\
            --strategy amplitude_embedding \\
            --input data/raw/c5_Matrix.csv \\
            --output data/processed/imputed_amplitude.parquet \\
            --config '{"normalization": "weighted", "include_probability_features": true}'

    Exit codes:
        0: Success
        1: Error (invalid arguments, file not found, imputation failed, etc.)
    """
    parser = argparse.ArgumentParser(
        description="Apply quantum-inspired imputation strategy to c5_Matrix dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available strategies:
  basis_embedding       - Direct basis state mapping
  amplitude_embedding   - Quantum superposition with Born rule
  angle_encoding        - Bloch sphere rotation angles
  density_matrix        - Mixed state density matrices
  graph_cycle           - DFT + cyclic graph features

Examples:
  # Apply basis embedding (default config)
  python src/imputation/apply_imputation.py \\
      --strategy basis_embedding \\
      --input data/raw/c5_Matrix.csv \\
      --output data/processed/imputed_basis.parquet

  # Apply amplitude embedding with weighted normalization
  python src/imputation/apply_imputation.py \\
      --strategy amplitude_embedding \\
      --input data/raw/c5_Matrix.csv \\
      --output data/processed/imputed_amplitude.parquet \\
      --config '{"normalization": "weighted"}'

  # Apply graph/cycle encoding with 39 harmonics
  python src/imputation/apply_imputation.py \\
      --strategy graph_cycle \\
      --input data/raw/c5_Matrix.csv \\
      --output data/processed/imputed_graph.parquet \\
      --config '{"n_harmonics": 39}'
        """
    )

    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        choices=list(AVAILABLE_STRATEGIES.keys()),
        help='Imputation strategy to apply'
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file (c5_Matrix.csv)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output Parquet file'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='JSON string with strategy configuration (optional)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Parse config if provided
    config = None
    if args.config:
        try:
            config = json.loads(args.config)
            logger.info(f"Using configuration: {config}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in --config argument: {e}")
            sys.exit(1)

    # Convert paths
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Apply imputation
    try:
        metadata = apply_imputation(
            strategy=args.strategy,
            input_path=input_path,
            output_path=output_path,
            config=config
        )

        # Print summary
        print("\\n" + "=" * 60)
        print("IMPUTATION COMPLETE")
        print("=" * 60)
        print(f"Strategy:        {metadata['strategy']}")
        print(f"Input shape:     {metadata['input_shape']}")
        print(f"Output shape:    {metadata['output_shape']}")
        print(f"Execution time:  {metadata['execution_time_seconds']:.2f}s")
        print(f"Output file:     {metadata['output_file']}")
        print(f"Metadata file:   {metadata['output_file']}.meta.json")
        print("=" * 60)

        sys.exit(0)

    except Exception as e:
        logger.error(f"Imputation failed: {e}")
        print(f"\\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

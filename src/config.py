"""
Project Configuration
Centralized path management for the Quantum State Prediction Experiment.

This module provides all file paths and directory constants used throughout
the project. By centralizing paths here, we ensure:
1. No hard-coded paths scattered across the codebase
2. Easy modification if directory structure changes
3. Clear documentation of where all files are stored

Usage Example:
    from src.config import DATASET_PATH, EXPERIMENTS_DIR
    df = pd.read_csv(DATASET_PATH)
    experiment_dir = EXPERIMENTS_DIR / "exp01_basis_embedding"
"""

import os
from pathlib import Path

# ============================================================================
# Root Directories
# ============================================================================

# Project root: the parent directory of this src/ folder
PROJECT_ROOT = Path(__file__).parent.parent

# Main data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"

# Experiments directory (Epic 5: 5 sequential imputation experiments)
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Individual experiment directories
EXP01_BASIS_EMBEDDING = EXPERIMENTS_DIR / "exp01_basis_embedding"
EXP02_AMPLITUDE_EMBEDDING = EXPERIMENTS_DIR / "exp02_amplitude_embedding"
EXP03_ANGLE_ENCODING = EXPERIMENTS_DIR / "exp03_angle_encoding"
EXP04_DENSITY_MATRIX = EXPERIMENTS_DIR / "exp04_density_matrix"
EXP05_GRAPH_CYCLE = EXPERIMENTS_DIR / "exp05_graph_cycle"

# Production directory (Epic 7: Final production run with best method)
PRODUCTION_DIR = PROJECT_ROOT / "production"
PRODUCTION_DATA = PRODUCTION_DIR / "data"
PRODUCTION_MODELS = PRODUCTION_DIR / "models"
PRODUCTION_PREDICTIONS = PRODUCTION_DIR / "predictions"
PRODUCTION_REPORTS = PRODUCTION_DIR / "reports"

# Development models directory (not experiment-specific)
MODELS_DIR = PROJECT_ROOT / "models"

# Reports and analysis
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_FIGURES = REPORTS_DIR / "figures"

# Jupyter notebooks
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Tests directory
TESTS_DIR = PROJECT_ROOT / "tests"

# ============================================================================
# Dataset Files
# ============================================================================

# Primary dataset filename
DATASET_FILENAME = "c5_Matrix.csv"

# Full path to the raw dataset
# USER ACTION REQUIRED: Place c5_Matrix.csv in data/raw/ directory
DATASET_PATH = DATA_RAW / DATASET_FILENAME

# ============================================================================
# Experiment Subdirectories
# ============================================================================

def get_experiment_paths(experiment_id: str) -> dict:
    """
    Get all standard paths for a given experiment.

    Each experiment follows the same directory structure:
    - data/: Imputed data output
    - models/: Trained model artifacts
    - reports/: Holdout test summaries and metrics
    - logs/: Execution logs

    Args:
        experiment_id: Experiment identifier (e.g., "exp01_basis_embedding")

    Returns:
        Dictionary with keys: "root", "data", "models", "reports", "logs"

    Example:
        >>> paths = get_experiment_paths("exp01_basis_embedding")
        >>> print(paths["models"])
        PosixPath('experiments/exp01_basis_embedding/models')
    """
    exp_root = EXPERIMENTS_DIR / experiment_id
    return {
        "root": exp_root,
        "data": exp_root / "data",
        "models": exp_root / "models",
        "reports": exp_root / "reports",
        "logs": exp_root / "logs"
    }

# ============================================================================
# File Path Helpers
# ============================================================================

def get_imputed_data_path(experiment_id: str) -> Path:
    """
    Get the path to the imputed data file for an experiment.

    Args:
        experiment_id: Experiment identifier

    Returns:
        Path to imputed_data.parquet file

    Example:
        >>> path = get_imputed_data_path("exp01_basis_embedding")
        >>> print(path)
        experiments/exp01_basis_embedding/data/imputed_data.parquet
    """
    return EXPERIMENTS_DIR / experiment_id / "data" / "imputed_data.parquet"

def get_model_path(experiment_id: str, model_name: str) -> Path:
    """
    Get the path to a trained model file for an experiment.

    Args:
        experiment_id: Experiment identifier
        model_name: Model filename (e.g., "lgbm_ranker.joblib")

    Returns:
        Path to model file

    Example:
        >>> path = get_model_path("exp01_basis_embedding", "lgbm_ranker.joblib")
        >>> print(path)
        experiments/exp01_basis_embedding/models/lgbm_ranker.joblib
    """
    return EXPERIMENTS_DIR / experiment_id / "models" / model_name

def get_report_path(experiment_id: str, report_name: str) -> Path:
    """
    Get the path to a report file for an experiment.

    Args:
        experiment_id: Experiment identifier
        report_name: Report filename (e.g., "holdout_test_summary.txt")

    Returns:
        Path to report file

    Example:
        >>> path = get_report_path("exp01_basis_embedding", "holdout_test_summary.txt")
        >>> print(path)
        experiments/exp01_basis_embedding/reports/holdout_test_summary.txt
    """
    return EXPERIMENTS_DIR / experiment_id / "reports" / report_name

# ============================================================================
# Directory Creation
# ============================================================================

def ensure_directories() -> None:
    """
    Create all required directories if they don't exist.

    This function should be called at the start of any script that writes
    files to ensure the directory structure is in place.

    This is idempotent - safe to call multiple times.

    Example:
        >>> from src.config import ensure_directories
        >>> ensure_directories()
        >>> # All directories now exist and are ready for use
    """
    # Core directories
    directories = [
        DATA_RAW,
        DATA_PROCESSED,
        MODELS_DIR,
        REPORTS_DIR,
        REPORTS_FIGURES,
        NOTEBOOKS_DIR,
        TESTS_DIR,
    ]

    # Experiment directories (all 5 experiments)
    for exp_id in [
        "exp01_basis_embedding",
        "exp02_amplitude_embedding",
        "exp03_angle_encoding",
        "exp04_density_matrix",
        "exp05_graph_cycle"
    ]:
        paths = get_experiment_paths(exp_id)
        directories.extend(paths.values())

    # Production directories
    directories.extend([
        PRODUCTION_DATA,
        PRODUCTION_MODELS,
        PRODUCTION_PREDICTIONS,
        PRODUCTION_REPORTS,
    ])

    # Create all directories
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Validation
# ============================================================================

def validate_dataset_exists() -> bool:
    """
    Check if the c5_Matrix.csv dataset file exists in the expected location.

    Returns:
        True if dataset exists, False otherwise

    Example:
        >>> from src.config import validate_dataset_exists, DATASET_PATH
        >>> if not validate_dataset_exists():
        ...     print(f"Dataset not found at {DATASET_PATH}")
        ...     print("USER ACTION REQUIRED: Copy c5_Matrix.csv to data/raw/")
    """
    return DATASET_PATH.exists()

# ============================================================================
# Configuration Summary
# ============================================================================

def print_config_summary() -> None:
    """
    Print a summary of all configured paths for debugging/verification.

    Useful for verifying the configuration is correct and all paths
    are set up as expected.

    Example:
        >>> from src.config import print_config_summary
        >>> print_config_summary()
        === Quantum State Prediction - Configuration Summary ===
        Project Root: /path/to/project
        Dataset Path: /path/to/project/data/raw/c5_Matrix.csv
        ...
    """
    print("=" * 60)
    print("Quantum State Prediction - Configuration Summary")
    print("=" * 60)
    print(f"Project Root:       {PROJECT_ROOT}")
    print(f"Dataset Path:       {DATASET_PATH}")
    print(f"Dataset Exists:     {validate_dataset_exists()}")
    print(f"Experiments Dir:    {EXPERIMENTS_DIR}")
    print(f"Production Dir:     {PRODUCTION_DIR}")
    print(f"Reports Dir:        {REPORTS_DIR}")
    print(f"Models Dir:         {MODELS_DIR}")
    print(f"Notebooks Dir:      {NOTEBOOKS_DIR}")
    print("=" * 60)

# ============================================================================
# Module Test
# ============================================================================

if __name__ == "__main__":
    # When run directly, print configuration and create directories
    print_config_summary()
    print("\nCreating all directories...")
    ensure_directories()
    print("[OK] All directories created successfully")

    if not validate_dataset_exists():
        print("\n[WARNING] Dataset file not found!")
        print(f"   Expected location: {DATASET_PATH}")
        print("   USER ACTION REQUIRED: Copy c5_Matrix.csv to data/raw/")
        print("   See README.md section 'Data Setup' for instructions.")
    else:
        print(f"\n[OK] Dataset file found at {DATASET_PATH}")
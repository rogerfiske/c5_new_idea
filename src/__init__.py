"""
Quantum State Prediction Experiment - Source Package

This package contains all source code for the quantum state prediction project.

Modules:
    - config: Centralized configuration and path management
    - data_loader: Dataset loading and validation utilities
    - evaluation: Evaluation metrics and reporting
    - main: Main execution script and CLI
    - imputation: Quantum-inspired imputation methods (5 methods)
    - modeling: Ranking models and training pipelines
"""

__version__ = "1.0.0"
__author__ = "Quantum Prediction Team"

# Import commonly used items for convenience
from . import config

__all__ = ["config"]
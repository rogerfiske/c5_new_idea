"""
Quantum-Inspired Imputation Methods Module

This module contains 5 quantum-inspired methods for imputing quantum states
from binary vectors in the c5_Matrix dataset.

Methods:
1. Basis Embedding (basis_embedding.py)
   - Direct mapping to computational basis states

2. Amplitude Embedding (amplitude_embedding.py)
   - Superposition over active positions with normalized amplitudes

3. Angle Encoding (angle_encoding.py)
   - Rotation-based quantum encoding using gate operations

4. Density Matrix Embedding (density_matrix.py)
   - Mixed state representations for probabilistic modeling

5. Graph/Cycle Encoding (graph_cycle_encoding.py)
   - Cyclic graph structure modeling for C₃₉ ring

Each imputation method will be implemented in Epic 2 of the project.

Usage Example:
    from src.imputation.amplitude_embedding import AmplitudeEmbedding

    imputer = AmplitudeEmbedding()
    quantum_state = imputer.impute(binary_vector)
"""

from src.imputation.base_imputer import BaseImputer

__all__ = [
    # Base class (Epic 2, Story 2.1)
    "BaseImputer",

    # Five imputation methods (to be implemented in Epic 2, Stories 2.2-2.6)
    # "BasisEmbedding",
    # "AmplitudeEmbedding",
    # "AngleEncoding",
    # "DensityMatrixEmbedding",
    # "GraphCycleEncoding",
]
"""
Basis Embedding Imputation Strategy

This module implements the Basis Embedding quantum-inspired imputation method,
which maps binary quantum state vectors to computational basis state representations.
Each active position (QV=1) is treated as a basis vector in a 39-dimensional
Hilbert space.

The quantum-inspired concept:
In quantum mechanics, a basis state |i⟩ represents a pure state where only
position i is active. A quantum state with multiple active positions can be
viewed as a superposition of basis states. This implementation creates features
that capture:
1. Which positions are active (one-hot encoding)
2. Statistical frequency patterns learned from training data

Author: BMad Dev Agent (James)
Date: 2025-10-13
Story: Epic 2, Story 2.2 - Basis Embedding Strategy
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

from src.imputation.base_imputer import BaseImputer

# Set up logging
logger = logging.getLogger(__name__)


class BasisEmbedding(BaseImputer):
    """
    Basis Embedding imputation strategy.

    Maps binary quantum state vectors to computational basis state representations.
    Each active position (QV=1) is treated as a basis vector in a 39-dimensional
    Hilbert space.

    Quantum Mechanics Background:
    ------------------------------
    In quantum computing, a basis state |i⟩ represents a state where only one
    computational basis position is active. For example, |3⟩ means position 3
    is active (value 1) and all other positions are inactive (value 0).

    A general quantum state can be written as a superposition:
        |ψ⟩ = α₁|1⟩ + α₂|2⟩ + ... + α₃₉|39⟩

    Our binary quantum states have exactly 5 active positions, representing
    a special kind of superposition where 5 basis states are "on" simultaneously.

    Feature Engineering Strategy:
    -----------------------------
    This implementation creates two types of features:

    1. **One-Hot Encoding** (39 features):
       Direct copy of QV_1 through QV_39. This preserves which positions are
       active in the original quantum state. For example, if QV_3=1, then
       feature 2 (0-indexed) will be 1.

    2. **Frequency-Weighted Features** (39 features, optional):
       Each position is weighted by how frequently it appears as active in
       the training data. This captures statistical patterns:
       - Positions that are rarely active get low weights
       - Positions that are frequently active get high weights
       - Helps ranking models learn which positions are "important"

    Parameters:
    -----------
    name : str, optional (default="basis_embedding")
        Human-readable name for this imputation strategy

    include_frequency_features : bool, optional (default=True)
        Whether to include frequency-weighted features in the output.
        - If True: output has 78 features (39 one-hot + 39 frequency)
        - If False: output has 39 features (39 one-hot only)

    Output Dimensions:
    ------------------
    - With frequency features (default): (n_samples, 78)
    - Without frequency features: (n_samples, 39)

    Attributes:
    -----------
    position_frequencies_ : np.ndarray, shape (39,)
        Learned frequency of each position in training data.
        position_frequencies_[i] = fraction of training samples where position i+1 is active
        Only set after calling fit().

    Examples:
    ---------
    >>> from src.data_loader import load_dataset
    >>> from src.imputation.basis_embedding import BasisEmbedding
    >>>
    >>> # Load dataset
    >>> df = load_dataset()
    >>>
    >>> # Create imputer with default settings
    >>> imputer = BasisEmbedding()
    >>>
    >>> # Fit and transform
    >>> features = imputer.fit_transform(df)
    >>> print(features.shape)  # (11581, 78)
    >>>
    >>> # Without frequency features
    >>> imputer_simple = BasisEmbedding(include_frequency_features=False)
    >>> features_simple = imputer_simple.fit_transform(df)
    >>> print(features_simple.shape)  # (11581, 39)

    Notes:
    ------
    - This is the simplest imputation strategy in Epic 2
    - Good baseline for comparing more complex strategies
    - Low computational cost (just array multiplication)
    - Leverages EDA insight: position frequencies are non-uniform
      (positions 1-10 are most frequent in the dataset)
    """

    def __init__(
        self,
        name: str = "basis_embedding",
        include_frequency_features: bool = True
    ):
        """
        Initialize Basis Embedding imputer.

        Args:
            name: Human-readable name for this imputation strategy
            include_frequency_features: Whether to include frequency-weighted features

        Examples:
            >>> # Default: includes frequency features
            >>> imputer1 = BasisEmbedding()
            >>>
            >>> # Without frequency features
            >>> imputer2 = BasisEmbedding(include_frequency_features=False)
            >>>
            >>> # Custom name
            >>> imputer3 = BasisEmbedding(name="my_basis_embedding")
        """
        # Initialize parent class with config
        super().__init__(
            name=name,
            config={"include_frequency_features": include_frequency_features}
        )

        # Store parameter as instance attribute
        self.include_frequency_features = include_frequency_features

        # Learned parameter (set during fit())
        self.position_frequencies_ = None

        logger.debug(
            f"Initialized BasisEmbedding with "
            f"include_frequency_features={include_frequency_features}"
        )

    def _fit(self, X: pd.DataFrame) -> None:
        """
        Learn position frequency statistics from training data.

        This method calculates how often each of the 39 positions is active
        across all training samples. These frequencies capture the statistical
        distribution of active positions in the dataset.

        The learned frequencies will be used during transformation to create
        frequency-weighted features that help models learn which positions
        are more "important" (appear more frequently).

        Args:
            X: Training DataFrame with columns [event-ID, QV_1, ..., QV_39]
               Shape: (n_samples, 40)

        Side Effects:
            Sets self.position_frequencies_ to numpy array of shape (39,)
            containing the fraction of training samples where each position is active.

        Mathematical Details:
            position_frequencies_[i] = (number of samples where QV_{i+1}=1) / n_samples

            For example, if QV_5 is active in 3000 out of 11581 samples:
            position_frequencies_[4] = 3000 / 11581 ≈ 0.259

        Notes:
            - This is called automatically by the public fit() method
            - Input validation is performed by the parent class
            - Frequencies are stored as numpy array for efficient computation
        """
        # Define QV column names
        qv_columns = [f'QV_{i}' for i in range(1, 40)]

        # Calculate position frequencies
        # Sum each column (counts active positions), then divide by number of samples
        position_counts = X[qv_columns].sum(axis=0)
        self.position_frequencies_ = position_counts.values / len(X)

        # Log statistics for debugging
        logger.info(
            f"Learned position frequencies from {len(X)} samples. "
            f"Frequency range: [{self.position_frequencies_.min():.4f}, "
            f"{self.position_frequencies_.max():.4f}]"
        )
        logger.debug(
            f"Top 5 most frequent positions: "
            f"{np.argsort(self.position_frequencies_)[-5:][::-1] + 1}"
        )

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform binary quantum state vectors to basis embedding features.

        This method creates feature representations by:
        1. Extracting the one-hot encoded QV columns (which positions are active)
        2. Optionally weighting each position by its training frequency

        Args:
            X: Input DataFrame with columns [event-ID, QV_1, ..., QV_39]
               Shape: (n_samples, 40)

        Returns:
            Feature matrix of shape:
            - (n_samples, 78) if include_frequency_features=True
            - (n_samples, 39) if include_frequency_features=False

            Data type: np.float64

        Feature Description:
            If include_frequency_features=True:
            - Columns 0-38: One-hot encoding (copy of QV_1 through QV_39)
            - Columns 39-77: Frequency-weighted features (QV * learned_frequency)

            If include_frequency_features=False:
            - Columns 0-38: One-hot encoding only

        Mathematical Details:
            One-hot features: f_i = QV_{i+1} (binary: 0 or 1)

            Frequency features: f_{39+i} = QV_{i+1} × position_frequencies_[i]

            For example, if QV_5=1 and position_frequencies_[4]=0.259:
            - f_4 = 1 (one-hot)
            - f_43 = 1 × 0.259 = 0.259 (frequency-weighted)

        Notes:
            - This is called automatically by the public transform() method
            - Input validation is performed by the parent class
            - Output validation (NaN/Inf check) is performed by parent class
        """
        # Extract QV columns
        qv_columns = [f'QV_{i}' for i in range(1, 40)]
        qv_data = X[qv_columns].values  # Shape: (n_samples, 39)

        # Start with one-hot encoding
        features_list = [qv_data]

        # Add frequency-weighted features if requested
        if self.include_frequency_features:
            # Multiply each position by its learned frequency
            # Broadcasting: (n_samples, 39) * (39,) → (n_samples, 39)
            frequency_features = qv_data * self.position_frequencies_
            features_list.append(frequency_features)

        # Concatenate features horizontally
        # If include_frequency_features=True: (n_samples, 39) + (n_samples, 39) → (n_samples, 78)
        # If include_frequency_features=False: just (n_samples, 39)
        result = np.hstack(features_list)

        logger.debug(
            f"Transformed {len(X)} samples to shape {result.shape}. "
            f"Feature types: {'one-hot + frequency' if self.include_frequency_features else 'one-hot only'}"
        )

        return result

    def get_feature_names(self) -> list:
        """
        Get human-readable names for output features.

        This method provides descriptive names for each output feature,
        which is useful for model interpretation and debugging.

        Returns:
            List of feature names, length 78 or 39 depending on configuration

        Examples:
            >>> imputer = BasisEmbedding()
            >>> imputer.fit(df)
            >>> names = imputer.get_feature_names()
            >>> print(names[:5])
            ['qv_1_onehot', 'qv_2_onehot', 'qv_3_onehot', 'qv_4_onehot', 'qv_5_onehot']
            >>> print(names[39:44])
            ['qv_1_freq', 'qv_2_freq', 'qv_3_freq', 'qv_4_freq', 'qv_5_freq']

        Notes:
            - Not required by base class, but useful for downstream analysis
            - Can be called after fitting to get feature names
        """
        # One-hot feature names
        onehot_names = [f'qv_{i}_onehot' for i in range(1, 40)]

        if self.include_frequency_features:
            # Frequency feature names
            freq_names = [f'qv_{i}_freq' for i in range(1, 40)]
            return onehot_names + freq_names
        else:
            return onehot_names
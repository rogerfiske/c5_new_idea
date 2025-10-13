"""
Amplitude Embedding Imputation Strategy

This module implements the Amplitude Embedding quantum-inspired imputation method,
which represents quantum states as superpositions where amplitudes are distributed
across active positions. This captures the probabilistic nature of quantum
superposition states.

The quantum-inspired concept:
In quantum mechanics, a superposition state is written as:
    |ψ⟩ = α₁|1⟩ + α₂|2⟩ + ... + α₃₉|39⟩

where αᵢ are complex amplitudes satisfying |α₁|² + |α₂|² + ... + |α₃₉|² = 1.
The squared magnitude |αᵢ|² represents the probability of measuring position i.

For our binary quantum states with exactly 5 active positions, we create amplitude
vectors where the 5 active positions have non-zero amplitudes that are normalized.

Author: BMad Dev Agent (James)
Date: 2025-10-13
Story: Epic 2, Story 2.3 - Amplitude Embedding Strategy
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Literal

from src.imputation.base_imputer import BaseImputer

# Set up logging
logger = logging.getLogger(__name__)


class AmplitudeEmbedding(BaseImputer):
    """
    Amplitude Embedding imputation strategy.

    Represents quantum states as superpositions with amplitudes distributed
    across active positions. This captures the probabilistic nature of quantum
    superposition states.

    Quantum Mechanics Background:
    ------------------------------
    In quantum mechanics, a pure quantum state |ψ⟩ is represented as a
    superposition of basis states with complex amplitudes:

        |ψ⟩ = α₁|1⟩ + α₂|2⟩ + ... + α₃₉|39⟩

    Key properties:
    1. Amplitudes αᵢ are complex numbers (we use real for simplicity)
    2. Normalization: Σᵢ |αᵢ|² = 1 (sum of squared amplitudes equals 1)
    3. Born rule: |αᵢ|² = probability of measuring position i

    For our binary quantum states:
    - 5 positions are active (QV=1), 34 are inactive (QV=0)
    - Active positions get non-zero amplitudes
    - Inactive positions get zero amplitudes
    - Amplitudes are normalized so squared sum = 1

    Normalization Strategies:
    -------------------------
    **Uniform Normalization** (default):
    - All active positions get equal amplitude: αᵢ = 1/√5 ≈ 0.447
    - This represents a "uniform superposition" over 5 basis states
    - Reflects maximum uncertainty: all active positions equally likely
    - Example: If positions [1,5,8,30,38] are active:
      α₁ = α₅ = α₈ = α₃₀ = α₃₈ = 1/√5 ≈ 0.447
      All other αᵢ = 0

    **Weighted Normalization**:
    - Amplitudes proportional to position frequencies from training data
    - More frequent positions get larger amplitudes
    - Normalized so squared sum = 1
    - Example: If QV_1 appears in 30% of training, QV_5 in 10%:
      α₁ will be larger than α₅ (after normalization)

    Feature Engineering Strategy:
    -----------------------------
    This implementation creates two types of features:

    1. **Amplitude Features** (39 features):
       The amplitude αᵢ for each position. Zero for inactive positions,
       non-zero (normalized) for active positions.

    2. **Probability Features** (39 features):
       The squared amplitude |αᵢ|² for each position. This represents
       the "measurement probability" in quantum mechanics. Also called
       the Born rule probabilities.

    Parameters:
    -----------
    name : str, optional (default="amplitude_embedding")
        Human-readable name for this imputation strategy

    normalization : {"uniform", "weighted"}, optional (default="uniform")
        Normalization strategy for amplitudes:
        - "uniform": Equal amplitudes (1/√n_active) for all active positions
        - "weighted": Amplitudes proportional to position frequencies

    include_probability_features : bool, optional (default=True)
        Whether to include probability (amplitude squared) features.
        - If True: output has 78 features (39 amplitudes + 39 probabilities)
        - If False: output has 39 features (39 amplitudes only)

    Output Dimensions:
    ------------------
    - With probability features (default): (n_samples, 78)
    - Without probability features: (n_samples, 39)

    Attributes:
    -----------
    position_frequencies_ : np.ndarray, shape (39,)
        Learned frequency of each position (used for weighted normalization).
        Only set after calling fit().

    Examples:
    ---------
    >>> from src.data_loader import load_dataset
    >>> from src.imputation.amplitude_embedding import AmplitudeEmbedding
    >>>
    >>> # Load dataset
    >>> df = load_dataset()
    >>>
    >>> # Uniform superposition (default)
    >>> imputer = AmplitudeEmbedding()
    >>> features = imputer.fit_transform(df)
    >>> print(features.shape)  # (11581, 78)
    >>>
    >>> # Weighted by training frequencies
    >>> imputer_weighted = AmplitudeEmbedding(normalization="weighted")
    >>> features_weighted = imputer_weighted.fit_transform(df)
    >>> print(features_weighted.shape)  # (11581, 78)
    >>>
    >>> # Without probability features
    >>> imputer_simple = AmplitudeEmbedding(include_probability_features=False)
    >>> features_simple = imputer_simple.fit_transform(df)
    >>> print(features_simple.shape)  # (11581, 39)

    Notes:
    ------
    - Amplitudes are normalized: sum of squared amplitudes = 1.0 for each row
    - Uniform normalization: amplitude = 1/√5 ≈ 0.447 for active positions
    - Weighted normalization uses position frequencies from training data
    - This captures quantum superposition concept in classical features
    """

    def __init__(
        self,
        name: str = "amplitude_embedding",
        normalization: Literal["uniform", "weighted"] = "uniform",
        include_probability_features: bool = True
    ):
        """
        Initialize Amplitude Embedding imputer.

        Args:
            name: Human-readable name for this imputation strategy
            normalization: "uniform" or "weighted" amplitude normalization
            include_probability_features: Whether to include |amplitude|² features

        Raises:
            ValueError: If normalization is not "uniform" or "weighted"

        Examples:
            >>> # Default: uniform normalization with probabilities
            >>> imputer1 = AmplitudeEmbedding()
            >>>
            >>> # Weighted normalization
            >>> imputer2 = AmplitudeEmbedding(normalization="weighted")
            >>>
            >>> # Without probability features
            >>> imputer3 = AmplitudeEmbedding(include_probability_features=False)
        """
        # Validate normalization parameter
        if normalization not in ["uniform", "weighted"]:
            raise ValueError(
                f"normalization must be 'uniform' or 'weighted', got '{normalization}'"
            )

        # Initialize parent class with config
        super().__init__(
            name=name,
            config={
                "normalization": normalization,
                "include_probability_features": include_probability_features
            }
        )

        # Store parameters as instance attributes
        self.normalization = normalization
        self.include_probability_features = include_probability_features

        # Learned parameter (set during fit(), used for weighted normalization)
        self.position_frequencies_ = None

        logger.debug(
            f"Initialized AmplitudeEmbedding with normalization={normalization}, "
            f"include_probability_features={include_probability_features}"
        )

    def _fit(self, X: pd.DataFrame) -> None:
        """
        Learn position frequency statistics for weighted normalization.

        For uniform normalization, this method doesn't need to learn anything
        (amplitudes are always 1/√5). However, for weighted normalization,
        we learn position frequencies from training data to weight amplitudes.

        Args:
            X: Training DataFrame with columns [event-ID, QV_1, ..., QV_39]
               Shape: (n_samples, 40)

        Side Effects:
            Sets self.position_frequencies_ to numpy array of shape (39,)
            containing position frequencies (only used if normalization="weighted")

        Notes:
            - For uniform normalization: frequencies are computed but not used
            - For weighted normalization: frequencies determine amplitude weights
            - Called automatically by the public fit() method
        """
        # Calculate position frequencies (same as Basis Embedding)
        qv_columns = [f'QV_{i}' for i in range(1, 40)]
        position_counts = X[qv_columns].sum(axis=0)
        self.position_frequencies_ = position_counts.values / len(X)

        logger.info(
            f"Learned position frequencies from {len(X)} samples. "
            f"Normalization mode: {self.normalization}"
        )

        if self.normalization == "weighted":
            logger.debug(
                f"Frequency range for weighting: "
                f"[{self.position_frequencies_.min():.4f}, "
                f"{self.position_frequencies_.max():.4f}]"
            )

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform binary quantum states to amplitude embedding features.

        This method creates amplitude vectors where:
        1. Active positions (QV=1) get non-zero amplitudes
        2. Inactive positions (QV=0) get zero amplitudes
        3. Amplitudes are normalized so sum of squares = 1 for each row
        4. Optionally compute probability features (amplitude squared)

        Args:
            X: Input DataFrame with columns [event-ID, QV_1, ..., QV_39]
               Shape: (n_samples, 40)

        Returns:
            Feature matrix of shape:
            - (n_samples, 78) if include_probability_features=True
            - (n_samples, 39) if include_probability_features=False
            Data type: np.float64

        Algorithm:
            For each sample:
            1. Identify which positions are active (QV=1)
            2. Assign raw amplitudes based on normalization mode:
               - uniform: all active get value 1.0
               - weighted: active get position_frequencies_[i]
            3. Normalize so sum of squared amplitudes = 1:
               normalized_amp[i] = raw_amp[i] / sqrt(sum(raw_amp²))
            4. Compute probabilities: prob[i] = normalized_amp[i]²
            5. Concatenate amplitude and probability features

        Mathematical Details:
            Uniform normalization for n_active=5:
                α = 1/√5 ≈ 0.447 for each active position
                Σᵢ α² = 5 × (1/√5)² = 5 × (1/5) = 1.0 ✓

            Weighted normalization:
                raw_αᵢ = frequency[i] × QV[i]
                α = raw_α / ||raw_α|| where ||·|| is L2 norm
                Ensures Σᵢ α² = 1.0

        Notes:
            - Output validation (NaN/Inf) performed by parent class
            - Each row is independently normalized
        """
        # Extract QV columns
        qv_columns = [f'QV_{i}' for i in range(1, 40)]
        qv_data = X[qv_columns].values  # Shape: (n_samples, 39)

        # Create amplitude features based on normalization mode
        if self.normalization == "uniform":
            # Uniform: all active positions get equal raw amplitude (1.0)
            raw_amplitudes = qv_data.astype(np.float64)
        else:  # weighted
            # Weighted: active positions weighted by frequency
            # Broadcasting: (n_samples, 39) * (39,) → (n_samples, 39)
            raw_amplitudes = qv_data * self.position_frequencies_

        # Normalize amplitudes so sum of squares = 1 for each row
        # L2 norm per row: sqrt(sum of squared amplitudes)
        amplitude_norms = np.sqrt((raw_amplitudes ** 2).sum(axis=1, keepdims=True))

        # Avoid division by zero (shouldn't happen with valid data, but defensive)
        amplitude_norms = np.where(amplitude_norms > 0, amplitude_norms, 1.0)

        # Normalized amplitudes
        amplitudes = raw_amplitudes / amplitude_norms  # Shape: (n_samples, 39)

        # Start with amplitude features
        features_list = [amplitudes]

        # Add probability features if requested
        if self.include_probability_features:
            # Probability = amplitude squared (Born rule)
            probabilities = amplitudes ** 2  # Shape: (n_samples, 39)
            features_list.append(probabilities)

        # Concatenate features horizontally
        result = np.hstack(features_list)

        logger.debug(
            f"Transformed {len(X)} samples to shape {result.shape}. "
            f"Normalization: {self.normalization}, "
            f"Features: {'amplitude + probability' if self.include_probability_features else 'amplitude only'}"
        )

        return result

    def get_feature_names(self) -> list:
        """
        Get human-readable names for output features.

        Returns:
            List of feature names, length 78 or 39 depending on configuration

        Examples:
            >>> imputer = AmplitudeEmbedding()
            >>> imputer.fit(df)
            >>> names = imputer.get_feature_names()
            >>> print(names[:5])
            ['qv_1_amplitude', 'qv_2_amplitude', ..., 'qv_5_amplitude']
            >>> print(names[39:44])
            ['qv_1_probability', 'qv_2_probability', ..., 'qv_5_probability']
        """
        # Amplitude feature names
        amplitude_names = [f'qv_{i}_amplitude' for i in range(1, 40)]

        if self.include_probability_features:
            # Probability feature names
            probability_names = [f'qv_{i}_probability' for i in range(1, 40)]
            return amplitude_names + probability_names
        else:
            return amplitude_names

    def verify_normalization(self, features: np.ndarray) -> dict:
        """
        Verify that amplitude normalization is correct.

        This is a diagnostic method to check that amplitudes are properly
        normalized (sum of squared amplitudes = 1 for each row).

        Args:
            features: Output from transform(), shape (n_samples, n_features)

        Returns:
            Dictionary with normalization statistics:
            - "mean_norm": Average sum of squared amplitudes (should be ~1.0)
            - "max_deviation": Maximum deviation from 1.0
            - "all_normalized": True if all rows are normalized (within tolerance)

        Examples:
            >>> imputer = AmplitudeEmbedding()
            >>> features = imputer.fit_transform(df)
            >>> stats = imputer.verify_normalization(features)
            >>> print(stats["all_normalized"])  # Should be True
            True
        """
        # Extract amplitude features (first 39 columns)
        amplitudes = features[:, :39]

        # Compute sum of squared amplitudes for each row
        squared_sums = (amplitudes ** 2).sum(axis=1)

        # Statistics
        mean_norm = squared_sums.mean()
        max_deviation = np.abs(squared_sums - 1.0).max()
        all_normalized = np.allclose(squared_sums, 1.0, atol=1e-10)

        return {
            "mean_norm": float(mean_norm),
            "max_deviation": float(max_deviation),
            "all_normalized": all_normalized
        }

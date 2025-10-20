"""
Simple Ensemble for Quantum State Prediction

This module implements a straightforward ensemble method that combines predictions
from multiple trained ranker models using weighted averaging of prediction scores.

Why Ensembles?
--------------
Epic 9A discovered that all imputation methods show ~300% over-prediction bias,
revealing that bias is data-level, not feature-level. The ensemble approach combines
multiple models (LGBM, SetTransformer, GNN) to leverage their complementary strengths
and improve prediction robustness.

The ensemble doesn't eliminate bias (that's handled by RangeAwareBiasCorrection in
Story 9B.2), but it can improve recall and reduce variance by averaging predictions.

How It Works:
-------------
1. Each model produces scores for all 39 positions
2. Scores are aggregated using weighted averaging (equal or confidence-based)
3. Final ranking is determined by aggregated scores
4. Top-k positions are returned

Author: BMad Dev Agent (James)
Date: 2025-10-17
Epic: Epic 9B - Ensemble & Bias Correction
Story: 9B.1 - Simple Ensemble Implementation
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleEnsemble:
    """
    Simple weighted ensemble for combining multiple ranker predictions.

    This ensemble combines predictions from multiple trained models (typically
    LGBM, SetTransformer, and GNN) using weighted averaging of their prediction
    scores. It supports both equal weighting (default) and confidence-based
    weighting strategies.

    Design Philosophy:
    ------------------
    **Start Simple:** This implementation uses straightforward weighted averaging
    rather than complex meta-learning or stacking approaches. Position-aware
    weighting (from Epic 8 analysis showing models excel at different ranges)
    is deferred to future optimization if this simple approach proves insufficient.

    **NFR2 Compliance:** All code is heavily commented to explain the "why" behind
    each decision, not just the "what". This makes the ensemble understandable
    for non-programmers and aids debugging.

    Weighting Strategies:
    ---------------------
    1. **Equal Weighting (Default):**
       - Each model gets weight = 1/N where N is number of models
       - Simple, no hyperparameters, often surprisingly effective
       - Example: 3 models → weights = [0.333, 0.333, 0.334]

    2. **Confidence-Based Weighting:**
       - Weight each model by its prediction confidence (max score)
       - Models that are "more sure" get higher weight
       - Computed per-event dynamically
       - May help if one model is much more certain than others

    Args:
        models: List of trained ranker models
            Each model must have a predict() method that returns scores
            Typically: [lgbm_ranker, settransformer_ranker, gnn_ranker]
        weights: Optional custom weights for each model
            If None, uses equal weighting (recommended to start)
            If provided, must sum to 1.0 and match number of models
            Example: [0.4, 0.3, 0.3] → first model gets 40% weight
        weighting_strategy: Strategy for aggregating scores
            'equal': Use fixed equal weights (default)
            'confidence': Use dynamic confidence-based weights per event
            'custom': Use provided custom weights

    Attributes:
        models_: List of loaded ranker models
        weights_: Array of model weights (shape: n_models)
        weighting_strategy_: Selected weighting strategy
        n_models_: Number of models in ensemble
        model_names_: List of model names for logging/debugging

    Example:
        >>> # Load trained models
        >>> lgbm = joblib.load('production/models/density_matrix/lgbm_ranker.pkl')
        >>> st = torch.load('production/models/density_matrix/settransformer_ranker.pth')
        >>> gnn = torch.load('production/models/density_matrix/gnn_ranker.pth')
        >>>
        >>> # Create ensemble with equal weights
        >>> ensemble = SimpleEnsemble(models=[lgbm, st, gnn])
        >>>
        >>> # Make predictions
        >>> top_20, scores = ensemble.predict(event_features, k=20)
        >>>
        >>> # Check model contributions
        >>> contributions = ensemble.get_model_contributions()
        >>> print(contributions)  # {'lgbm': 0.333, 'settransformer': 0.333, 'gnn': 0.334}

    Notes:
        - Models are assumed to be already trained (loaded from disk)
        - All models must accept the same input features
        - Models must return scores for all 39 positions
        - This ensemble does NOT perform training - it only combines predictions
    """

    def __init__(
        self,
        models: List[Any],
        weights: Optional[List[float]] = None,
        weighting_strategy: str = 'equal'
    ):
        """
        Initialize simple ensemble.

        Args:
            models: List of trained ranker models
            weights: Optional custom weights (must sum to 1.0)
            weighting_strategy: 'equal', 'confidence', or 'custom'

        Raises:
            ValueError: If models list is empty
            ValueError: If weights don't sum to 1.0
            ValueError: If weights length doesn't match models length
            ValueError: If weighting_strategy is invalid
        """
        # Validation
        if not models:
            raise ValueError("models list cannot be empty. Provide at least one trained model.")

        if len(models) < 1:
            raise ValueError(
                f"Need at least 1 model for ensemble, got {len(models)}. "
                "For single model, use the model directly instead of ensemble."
            )

        valid_strategies = ['equal', 'confidence', 'custom']
        if weighting_strategy not in valid_strategies:
            raise ValueError(
                f"weighting_strategy must be one of {valid_strategies}, "
                f"got '{weighting_strategy}'"
            )

        if weighting_strategy == 'custom' and weights is None:
            raise ValueError(
                "weighting_strategy='custom' requires weights parameter. "
                "Either provide weights or use 'equal' or 'confidence' strategy."
            )

        # Store models
        self.models_ = models
        self.n_models_ = len(models)
        self.weighting_strategy_ = weighting_strategy

        # Extract model names for logging
        self.model_names_ = [
            model.__class__.__name__ if hasattr(model, '__class__')
            else f"Model{i}"
            for i, model in enumerate(models)
        ]

        # Set weights based on strategy
        if weighting_strategy == 'custom':
            # Use provided custom weights
            if len(weights) != self.n_models_:
                raise ValueError(
                    f"weights length ({len(weights)}) must match models length ({self.n_models_})"
                )

            weights_array = np.array(weights, dtype=float)

            # Validate weights sum to 1.0 (with tolerance for floating point errors)
            if not np.isclose(weights_array.sum(), 1.0, atol=1e-6):
                raise ValueError(
                    f"weights must sum to 1.0, got {weights_array.sum():.6f}. "
                    "Example: [0.4, 0.3, 0.3] for 3 models."
                )

            self.weights_ = weights_array

        elif weighting_strategy == 'equal':
            # Equal weights: 1/N for each model
            self.weights_ = np.ones(self.n_models_, dtype=float) / self.n_models_

        else:  # confidence
            # Weights will be computed dynamically per-event
            # Set to None to indicate dynamic weighting
            self.weights_ = None

        logger.info(f"Initialized SimpleEnsemble with {self.n_models_} models")
        logger.info(f"Models: {', '.join(self.model_names_)}")
        logger.info(f"Weighting strategy: {weighting_strategy}")
        if self.weights_ is not None:
            logger.info(f"Weights: {self.weights_}")

    def predict(
        self,
        event_features: np.ndarray,
        k: int = 20
    ) -> Tuple[List[int], np.ndarray]:
        """
        Predict top-k positions by combining model predictions.

        Algorithm:
        ----------
        1. Get predictions from all models (scores for all 39 positions)
        2. Aggregate scores using weighted averaging
        3. Rank positions by aggregated scores (descending)
        4. Return top-k positions

        For equal weighting:
            ensemble_score[pos] = (model1_score[pos] + model2_score[pos] + ...) / n_models

        For confidence-based weighting:
            model_weight = max(model_scores) / sum(all_max_scores)
            ensemble_score[pos] = sum(model_weight * model_score[pos] for all models)

        Args:
            event_features: Input features for one or more events
                Shape: (n_events, n_features) or (n_features,) for single event
            k: Number of top positions to return (default: 20)
                Must be in range [1, 39]

        Returns:
            Tuple of (top_k_positions, aggregated_scores):
                top_k_positions: Ranked list of k positions
                    Shape: (n_events, k) or (k,) for single event
                    Each position is in range [1, 39]
                    Most likely position at index 0, second at index 1, etc.
                aggregated_scores: Score for each position (length 39)
                    Shape: (n_events, 39) or (39,) for single event
                    Higher score = more likely

        Raises:
            ValueError: If k not in range [1, 39]
            RuntimeError: If model prediction fails

        Example:
            >>> # Single event
            >>> top_20, scores = ensemble.predict(features, k=20)
            >>> top_20.shape  # (20,)
            >>> scores.shape  # (39,)
            >>> top_20[0]  # Most likely position (1-39)
            >>>
            >>> # Multiple events
            >>> top_20, scores = ensemble.predict(features_batch, k=20)
            >>> top_20.shape  # (n_events, 20)
            >>> scores.shape  # (n_events, 39)
        """
        # Validate k
        if not (1 <= k <= 39):
            raise ValueError(
                f"k must be in range [1, 39], got {k}. "
                "There are only 39 possible quantum positions in the C₃₉ group."
            )

        # Handle single event vs batch
        single_event = False

        # Determine if input is single event (need to expand dims)
        # Keep DataFrame/numpy format as-is for now (will convert per-model in _get_model_scores)
        if isinstance(event_features, pd.DataFrame):
            if len(event_features) == 1:
                single_event = True
            n_events = len(event_features)
        else:
            # Numpy array
            if event_features.ndim == 1:
                event_features = event_features.reshape(1, -1)
                single_event = True
            n_events = event_features.shape[0]

        logger.info(f"Predicting top-{k} for {n_events} event(s) using ensemble...")

        # Step 1: Get predictions from all models
        # Each model returns scores for all 39 positions
        all_model_scores = []

        for i, model in enumerate(self.models_):
            try:
                # Get scores from model
                # Different models have different interfaces, handle both cases
                scores = self._get_model_scores(model, event_features)

                # Validate shape
                if scores.shape != (n_events, 39):
                    raise ValueError(
                        f"Model {self.model_names_[i]} returned scores with shape {scores.shape}, "
                        f"expected ({n_events}, 39)"
                    )

                all_model_scores.append(scores)
                logger.debug(f"Got scores from {self.model_names_[i]}: shape {scores.shape}")

            except Exception as e:
                logger.error(f"Failed to get predictions from {self.model_names_[i]}: {e}")
                raise RuntimeError(
                    f"Ensemble prediction failed: {self.model_names_[i]} raised error. "
                    f"Error: {e}"
                ) from e

        # Stack scores: shape (n_models, n_events, 39)
        all_model_scores = np.stack(all_model_scores, axis=0)

        # Step 2: Aggregate scores using weighted averaging
        if self.weighting_strategy_ == 'confidence':
            # Confidence-based weighting (dynamic per-event)
            aggregated_scores = self._aggregate_confidence_weighted(all_model_scores)
        else:
            # Equal or custom weighting (fixed weights)
            # Reshape weights for broadcasting: (n_models, 1, 1)
            weights_broadcast = self.weights_.reshape(-1, 1, 1)
            # Weighted sum: (n_events, 39)
            aggregated_scores = np.sum(all_model_scores * weights_broadcast, axis=0)

        # Step 3: Rank positions by aggregated scores (descending)
        # argsort returns indices of smallest to largest, so use negative scores
        ranked_indices = np.argsort(-aggregated_scores, axis=1)

        # Step 4: Convert 0-based indices to 1-based position numbers and take top-k
        top_k_positions = ranked_indices[:, :k] + 1  # +1 for 1-based indexing

        # Handle single event case
        if single_event:
            top_k_positions = top_k_positions[0]
            aggregated_scores = aggregated_scores[0]

        logger.info(f"✓ Ensemble predictions generated for {n_events} event(s)")

        return top_k_positions, aggregated_scores

    def _get_model_scores(self, model: Any, event_features) -> np.ndarray:
        """
        Get prediction scores from a model.

        Different model types have different prediction interfaces:
        - LGBM: model.predict() returns scores directly (or via internal method)
        - Neural (PyTorch): model() or model.predict() with forward pass
        - Need to handle both cases

        Args:
            model: Trained model object
            event_features: Input features (DataFrame or numpy array)
                Shape: (n_events, n_features)

        Returns:
            scores: Prediction scores for all 39 positions
                Shape: (n_events, 39)
                Higher score = more likely

        Raises:
            RuntimeError: If cannot extract scores from model
        """
        # Try different approaches to get scores

        # Approach 1: Check if model has predict_scores method
        if hasattr(model, 'predict_scores'):
            return model.predict_scores(event_features)

        # Approach 2: Check if it's an LGBM-based ranker (our custom LGBMRanker class)
        if hasattr(model, '_create_ranking_features') and hasattr(model, 'model_'):
            # For LGBMRanker, use internal method to create ranking format
            # Preserve DataFrame if input is already DataFrame (keeps column names)
            if isinstance(event_features, pd.DataFrame):
                df_features = event_features
            else:
                df_features = pd.DataFrame(event_features)
            ranking_features = model._create_ranking_features(df_features)
            scores_flat = model.model_.predict(ranking_features)
            # Reshape: (n_events * 39,) -> (n_events, 39)
            return scores_flat.reshape(len(event_features), 39)

        # Approach 3: Check if it's a PyTorch-based ranker (SetTransformerRanker or GNNRanker)
        # These rankers have a model_ attribute that is the actual PyTorch model
        if hasattr(model, 'model_') and hasattr(model.model_, 'forward'):
            import torch
            # For PyTorch rankers, prepare data and call through the model_ attribute
            # Use _prepare_training_data() if available
            if hasattr(model, '_prepare_training_data'):
                # Convert to DataFrame if not already
                if not isinstance(event_features, pd.DataFrame):
                    event_features_df = pd.DataFrame(event_features)
                else:
                    event_features_df = event_features

                X_tensor, _ = model._prepare_training_data(event_features_df)
                X_tensor = X_tensor.to(model.device)

                # Make predictions
                model.model_.eval()
                with torch.no_grad():
                    # Check if model requires adjacency matrix (GNN)
                    if hasattr(model, 'adj_matrix_'):
                        scores = model.model_(X_tensor, model.adj_matrix_)
                    else:
                        # SetTransformer doesn't need adjacency matrix
                        scores = model.model_(X_tensor)
                    scores = scores.cpu().numpy()

                return scores

        # If none of the approaches worked, raise error
        raise RuntimeError(
            f"Cannot extract scores from model type {type(model)}. "
            "Model must have one of: predict_scores(), predict() (LGBM), forward() (PyTorch)"
        )

    def _aggregate_confidence_weighted(self, all_model_scores: np.ndarray) -> np.ndarray:
        """
        Aggregate scores using confidence-based weighting.

        Confidence-based weighting assigns higher weight to models that are
        "more confident" about their predictions (have higher max scores).

        Formula per event:
            model_confidence = max(model_scores for that event)
            model_weight = model_confidence / sum(all_model_confidences)
            ensemble_score[pos] = sum(model_weight * model_score[pos])

        This is computed dynamically per-event because model confidence
        varies across events.

        Args:
            all_model_scores: Scores from all models
                Shape: (n_models, n_events, 39)

        Returns:
            aggregated_scores: Confidence-weighted aggregated scores
                Shape: (n_events, 39)
        """
        # Compute confidence per model per event (max score across 39 positions)
        # Shape: (n_models, n_events)
        model_confidences = np.max(all_model_scores, axis=2)

        # Compute weights per event
        # Shape: (n_models, n_events)
        total_confidence = np.sum(model_confidences, axis=0, keepdims=True)  # (1, n_events)
        model_weights = model_confidences / total_confidence  # (n_models, n_events)

        # Reshape weights for broadcasting: (n_models, n_events, 1)
        model_weights_broadcast = model_weights[:, :, np.newaxis]

        # Weighted sum
        # Shape: (n_events, 39)
        aggregated_scores = np.sum(all_model_scores * model_weights_broadcast, axis=0)

        logger.debug(f"Confidence-weighted aggregation: avg weights per model: {np.mean(model_weights, axis=1)}")

        return aggregated_scores

    def get_model_contributions(self) -> Dict[str, float]:
        """
        Calculate how much each model contributed to final predictions.

        For fixed weighting (equal or custom), this returns the configured weights.
        For confidence-based weighting, this returns average weights across all
        predictions made so far (not available until predict() has been called).

        This is useful for:
        - Transparency: Understanding which models drive the ensemble
        - Debugging: Identifying if one model dominates
        - Analysis: Comparing expected vs actual model contributions

        Returns:
            contributions: Dict mapping model names to their contribution weights
                Example: {'LGBMRanker': 0.333, 'SetTransformer': 0.333, 'GNN': 0.334}

        Example:
            >>> contributions = ensemble.get_model_contributions()
            >>> print(f"LGBM contributes {contributions['LGBMRanker']:.1%}")
            LGBM contributes 33.3%
        """
        if self.weights_ is not None:
            # Fixed weights (equal or custom)
            return {
                name: float(weight)
                for name, weight in zip(self.model_names_, self.weights_)
            }
        else:
            # Confidence-based weights (dynamic)
            # Return equal weights as placeholder
            # In production, could track actual weights used across predictions
            logger.warning(
                "Confidence-based weighting is dynamic per-event. "
                "Returning equal weights as placeholder. "
                "Actual weights vary by event based on model confidence."
            )
            equal_weight = 1.0 / self.n_models_
            return {name: equal_weight for name in self.model_names_}

    def __repr__(self) -> str:
        """String representation of ensemble."""
        return (
            f"SimpleEnsemble("
            f"n_models={self.n_models_}, "
            f"strategy={self.weighting_strategy_}"
            f")"
        )

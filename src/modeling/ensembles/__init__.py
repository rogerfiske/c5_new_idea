"""
Ensemble Methods Sub-Module

This sub-module contains ensemble methods for combining predictions
from multiple ranking models.

Ensemble Strategies (Epic 6 - Optional):
1. Reciprocal Rank Fusion (rrf.py)
   - Combines rankings from multiple models using RRF algorithm

2. Weighted Average (weighted_average.py)
   - Weighted combination of model predictions based on performance

Usage Example:
    from src.modeling.ensembles.rrf import ReciprocalRankFusion

    ensemble = ReciprocalRankFusion(models=[model1, model2, model3])
    combined_predictions = ensemble.predict(X_test)
"""

__all__ = [
    # To be populated if Epic 6 is implemented
    # "ReciprocalRankFusion",
    # "WeightedAverageEnsemble",
]
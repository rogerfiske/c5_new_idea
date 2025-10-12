"""
Modeling Module

This module contains ranking models and training/prediction pipelines.

Components:
- rankers/: Individual ranking model implementations (4 model families)
- ensembles/: Ensemble methods for combining multiple rankers
- pipeline.py: Training and prediction orchestration

Ranking Models (Epic 3):
1. Frequency Baselines (frequency_ranker.py)
2. LightGBM Ranker (lgbm_ranker.py)
3. Set Transformer (set_transformer_model.py)
4. Graph Neural Network (gnn_ranker.py)

Usage Example:
    from src.modeling.rankers.lgbm_ranker import LGBMRanker

    ranker = LGBMRanker()
    ranker.train(X_train, y_train)
    predictions = ranker.predict(X_test)
"""

__all__ = [
    # Will be populated as models are implemented in Epic 3
]
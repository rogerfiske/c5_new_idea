"""
Ranking Models Sub-Module

This sub-module contains implementations of 4 ranking model families
for predicting the next quantum state.

Models:
1. Frequency Baselines (frequency_ranker.py)
   - Simple statistical baselines using historical frequencies

2. LightGBM/XGBoost Ranker (lgbm_ranker.py)
   - Gradient-boosted decision tree ranking models

3. Set Transformer (set_transformer_model.py)
   - Deep learning attention-based model for set prediction

4. Graph Neural Network (gnn_ranker.py)
   - GNN operating on the C₃₉ cyclic graph structure

All rankers will be implemented in Epic 3 (Stories 3.1-3.5).
"""

__all__ = [
    # To be populated as models are implemented in Epic 3
    # "FrequencyRanker",
    # "LGBMRanker",
    # "SetTransformer",
    # "GNNRanker",
]

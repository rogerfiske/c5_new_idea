"""
Unit Tests for SimpleEnsemble

Tests the SimpleEnsemble class to ensure it correctly combines predictions
from multiple models using various weighting strategies.

Author: BMad Dev Agent (James)
Date: 2025-10-17
Epic: Epic 9B - Ensemble & Bias Correction
Story: 9B.1 - Simple Ensemble Implementation
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from src.modeling.ensembles.simple_ensemble import SimpleEnsemble


class TestSimpleEnsembleInit:
    """Test SimpleEnsemble initialization and validation."""

    def test_init_with_single_model(self):
        """Test ensemble initialization with one model."""
        model = Mock()
        ensemble = SimpleEnsemble(models=[model])
        assert ensemble.n_models_ == 1
        assert ensemble.weighting_strategy_ == 'equal'
        assert np.allclose(ensemble.weights_, [1.0])

    def test_init_with_multiple_models(self):
        """Test ensemble initialization with multiple models."""
        models = [Mock(), Mock(), Mock()]
        ensemble = SimpleEnsemble(models=models)
        assert ensemble.n_models_ == 3
        assert np.allclose(ensemble.weights_, [1/3, 1/3, 1/3])

    def test_init_empty_models_raises_error(self):
        """Test that empty models list raises ValueError."""
        with pytest.raises(ValueError, match="models list cannot be empty"):
            SimpleEnsemble(models=[])

    def test_init_equal_weighting(self):
        """Test equal weighting strategy."""
        models = [Mock(), Mock()]
        ensemble = SimpleEnsemble(models=models, weighting_strategy='equal')
        assert ensemble.weighting_strategy_ == 'equal'
        assert np.allclose(ensemble.weights_, [0.5, 0.5])

    def test_init_custom_weighting_valid(self):
        """Test custom weighting with valid weights."""
        models = [Mock(), Mock(), Mock()]
        weights = [0.5, 0.3, 0.2]
        ensemble = SimpleEnsemble(
            models=models,
            weights=weights,
            weighting_strategy='custom'
        )
        assert ensemble.weighting_strategy_ == 'custom'
        assert np.allclose(ensemble.weights_, weights)

    def test_init_custom_weighting_invalid_sum(self):
        """Test custom weighting with weights not summing to 1.0."""
        models = [Mock(), Mock()]
        weights = [0.6, 0.6]  # Sum = 1.2
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            SimpleEnsemble(
                models=models,
                weights=weights,
                weighting_strategy='custom'
            )

    def test_init_custom_weighting_wrong_length(self):
        """Test custom weighting with mismatched weights length."""
        models = [Mock(), Mock(), Mock()]
        weights = [0.5, 0.5]  # 2 weights for 3 models
        with pytest.raises(ValueError, match="weights length"):
            SimpleEnsemble(
                models=models,
                weights=weights,
                weighting_strategy='custom'
            )

    def test_init_custom_strategy_without_weights(self):
        """Test custom strategy without providing weights."""
        models = [Mock(), Mock()]
        with pytest.raises(ValueError, match="requires weights parameter"):
            SimpleEnsemble(models=models, weighting_strategy='custom')

    def test_init_invalid_strategy(self):
        """Test invalid weighting strategy."""
        models = [Mock()]
        with pytest.raises(ValueError, match="weighting_strategy must be one of"):
            SimpleEnsemble(models=models, weighting_strategy='invalid')

    def test_init_confidence_weighting(self):
        """Test confidence weighting strategy initialization."""
        models = [Mock(), Mock()]
        ensemble = SimpleEnsemble(models=models, weighting_strategy='confidence')
        assert ensemble.weighting_strategy_ == 'confidence'
        assert ensemble.weights_ is None  # Dynamic weights


class TestSimpleEnsemblePredict:
    """Test SimpleEnsemble prediction functionality."""

    def create_mock_model(self, scores):
        """
        Create a mock model that returns specified scores.

        Args:
            scores: numpy array of shape (n_events, 39) with prediction scores

        Returns:
            Mock model object
        """
        model = Mock()
        model.predict_scores = Mock(return_value=scores)
        model.__class__.__name__ = 'MockModel'
        return model

    def test_predict_single_event_equal_weights(self):
        """Test prediction on single event with equal weights."""
        # Create mock models with known scores
        # Model 1: high scores for positions 1-5
        scores1 = np.zeros(39)
        scores1[:5] = [10, 9, 8, 7, 6]

        # Model 2: high scores for positions 6-10
        scores2 = np.zeros(39)
        scores2[5:10] = [10, 9, 8, 7, 6]

        model1 = self.create_mock_model(scores1.reshape(1, 39))
        model2 = self.create_mock_model(scores2.reshape(1, 39))

        ensemble = SimpleEnsemble(models=[model1, model2])

        # Single event (1D array)
        event_features = np.random.randn(50)  # 50 features
        top_k, scores = ensemble.predict(event_features, k=5)

        # Check shapes
        assert top_k.shape == (5,)
        assert scores.shape == (39,)

        # Check that top positions include both model preferences
        # With equal weights, both models contribute equally
        assert len(top_k) == 5
        assert all(1 <= pos <= 39 for pos in top_k)

    def test_predict_multiple_events(self):
        """Test prediction on multiple events."""
        # Create mock model
        n_events = 3
        scores = np.random.randn(n_events, 39)
        model = self.create_mock_model(scores)

        ensemble = SimpleEnsemble(models=[model])

        event_features = np.random.randn(n_events, 50)
        top_k, agg_scores = ensemble.predict(event_features, k=20)

        # Check shapes
        assert top_k.shape == (n_events, 20)
        assert agg_scores.shape == (n_events, 39)

        # Check positions are valid (1-39)
        assert np.all((top_k >= 1) & (top_k <= 39))

    def test_predict_custom_weights(self):
        """Test prediction with custom weights."""
        # Model 1: high score for position 1
        scores1 = np.zeros(39)
        scores1[0] = 10

        # Model 2: high score for position 2
        scores2 = np.zeros(39)
        scores2[1] = 10

        model1 = self.create_mock_model(scores1.reshape(1, 39))
        model2 = self.create_mock_model(scores2.reshape(1, 39))

        # Give model1 80% weight, model2 20% weight
        ensemble = SimpleEnsemble(
            models=[model1, model2],
            weights=[0.8, 0.2],
            weighting_strategy='custom'
        )

        event_features = np.random.randn(50)
        top_k, scores = ensemble.predict(event_features, k=2)

        # Position 1 should be ranked first (80% * 10 = 8 vs 20% * 10 = 2)
        assert top_k[0] == 1

    def test_predict_invalid_k(self):
        """Test prediction with invalid k values."""
        model = self.create_mock_model(np.random.randn(1, 39))
        ensemble = SimpleEnsemble(models=[model])

        event_features = np.random.randn(50)

        # k too small
        with pytest.raises(ValueError, match="k must be in range"):
            ensemble.predict(event_features, k=0)

        # k too large
        with pytest.raises(ValueError, match="k must be in range"):
            ensemble.predict(event_features, k=40)

    def test_predict_model_failure_raises_error(self):
        """Test that model prediction failure is caught and reported."""
        # Create mock model that raises an error
        model = Mock()
        model.predict_scores = Mock(side_effect=RuntimeError("Model failed"))
        model.__class__.__name__ = 'FailingModel'

        ensemble = SimpleEnsemble(models=[model])

        event_features = np.random.randn(50)

        with pytest.raises(RuntimeError, match="Ensemble prediction failed"):
            ensemble.predict(event_features, k=20)


class TestSimpleEnsembleModelContributions:
    """Test get_model_contributions functionality."""

    def test_contributions_equal_weights(self):
        """Test model contributions with equal weighting."""
        models = [Mock(), Mock(), Mock()]
        for i, model in enumerate(models):
            model.__class__.__name__ = f'Model{i}'

        ensemble = SimpleEnsemble(models=models)
        contributions = ensemble.get_model_contributions()

        assert len(contributions) == 3
        assert all(np.isclose(weight, 1/3) for weight in contributions.values())

    def test_contributions_custom_weights(self):
        """Test model contributions with custom weighting."""
        models = [Mock(), Mock()]
        models[0].__class__.__name__ = 'Model0'
        models[1].__class__.__name__ = 'Model1'

        weights = [0.7, 0.3]
        ensemble = SimpleEnsemble(
            models=models,
            weights=weights,
            weighting_strategy='custom'
        )
        contributions = ensemble.get_model_contributions()

        assert len(contributions) == 2
        assert np.isclose(contributions['Model0'], 0.7)
        assert np.isclose(contributions['Model1'], 0.3)

    def test_contributions_confidence_weights(self):
        """Test model contributions with confidence weighting (dynamic)."""
        models = [Mock(), Mock()]
        models[0].__class__.__name__ = 'Model0'
        models[1].__class__.__name__ = 'Model1'
        ensemble = SimpleEnsemble(models=models, weighting_strategy='confidence')

        # For confidence weighting, should return equal weights as placeholder
        contributions = ensemble.get_model_contributions()
        assert len(contributions) == 2
        assert all(np.isclose(weight, 0.5) for weight in contributions.values())


class TestSimpleEnsembleRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ returns meaningful string."""
        models = [Mock(), Mock(), Mock()]
        ensemble = SimpleEnsemble(models=models)

        repr_str = repr(ensemble)
        assert 'SimpleEnsemble' in repr_str
        assert 'n_models=3' in repr_str
        assert 'strategy=equal' in repr_str


def test_integration_ensemble_three_models():
    """
    Integration test: Ensemble of 3 models with different preferences.

    This test simulates the real-world scenario of combining LGBM, SetTransformer,
    and GNN predictions where each model has different position preferences.
    """
    # Model 1 (LGBM): prefers LOW range (1-13)
    scores1 = np.random.randn(1, 39)
    scores1[0, 0:13] += 5  # Boost LOW range

    # Model 2 (SetTransformer): prefers MID range (14-26)
    scores2 = np.random.randn(1, 39)
    scores2[0, 13:26] += 5  # Boost MID range

    # Model 3 (GNN): prefers HIGH range (27-39)
    scores3 = np.random.randn(1, 39)
    scores3[0, 26:39] += 5  # Boost HIGH range

    # Create mock models
    model1 = Mock()
    model1.predict_scores = Mock(return_value=scores1)
    model1.__class__.__name__ = 'LGBM'

    model2 = Mock()
    model2.predict_scores = Mock(return_value=scores2)
    model2.__class__.__name__ = 'SetTransformer'

    model3 = Mock()
    model3.predict_scores = Mock(return_value=scores3)
    model3.__class__.__name__ = 'GNN'

    # Create ensemble
    ensemble = SimpleEnsemble(models=[model1, model2, model3])

    # Make prediction
    event_features = np.random.randn(50)
    top_20, scores = ensemble.predict(event_features, k=20)

    # Verify results
    assert len(top_20) == 20
    assert scores.shape == (39,)

    # With equal weighting, top-20 should include positions from all ranges
    low_range_count = np.sum((top_20 >= 1) & (top_20 <= 13))
    mid_range_count = np.sum((top_20 >= 14) & (top_20 <= 26))
    high_range_count = np.sum((top_20 >= 27) & (top_20 <= 39))

    # All ranges should be represented (since we boosted each range equally)
    assert low_range_count > 0, "LOW range should be represented in top-20"
    assert mid_range_count > 0, "MID range should be represented in top-20"
    assert high_range_count > 0, "HIGH range should be represented in top-20"

    # Get contributions
    contributions = ensemble.get_model_contributions()
    assert len(contributions) == 3
    assert all(np.isclose(weight, 1/3) for weight in contributions.values())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

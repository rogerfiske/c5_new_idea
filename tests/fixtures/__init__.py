"""
Test Fixtures Package

Shared test fixtures and sample data used across multiple test modules.

Common Fixtures:
- sample_binary_vector: Standard 39-dim binary vector for testing
- sample_dataset: Small sample of c5_Matrix.csv for testing
- mock_imputed_data: Pre-imputed data for ranker testing
- sample_trained_models: Pre-trained models for evaluation testing

Usage in Tests:
    import pytest
    from tests.fixtures.sample_data import sample_binary_vector

    def test_imputation(sample_binary_vector):
        # Use the fixture in your test
        result = impute(sample_binary_vector)
        assert result.shape == (39,)
"""

__all__ = []
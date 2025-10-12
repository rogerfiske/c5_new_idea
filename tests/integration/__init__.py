"""
Integration Tests Package

Integration tests for complete workflows and multi-component interactions.

Test Categories:
1. End-to-end imputation workflows (raw data → imputed features)
2. End-to-end training workflows (features → trained model)
3. End-to-end prediction workflows (model + holdout data → predictions)
4. Complete experiment runs (data → imputation → training → evaluation)

These tests verify that components work correctly together and
that the full pipeline produces valid outputs.
"""

__all__ = []
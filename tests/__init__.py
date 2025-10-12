"""
Test Suite for Quantum State Prediction Experiment

This package contains all tests for the project.

Test Structure:
- unit/: Unit tests for individual functions and classes
- integration/: Integration tests for complete workflows
- fixtures/: Shared test fixtures and sample data

Test Requirements (from PRD Story 1.4):
1. All public functions must have unit tests
2. Each imputation method must have integration tests
3. Each ranker model must have integration tests
4. Critical workflows must have end-to-end tests

Running Tests:
    # Run all tests
    pytest tests/

    # Run unit tests only
    pytest tests/unit/

    # Run integration tests only
    pytest tests/integration/

    # Run with coverage report
    pytest tests/ --cov=src --cov-report=html
"""

__all__ = []
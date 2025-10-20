# Position Difficulty Analysis Report - Story 8.2

**Date:** 2025-10-16
**Epic:** Epic 8 - Model Performance Analysis
**Story:** 8.2 - Performance Breakdown by Position

---

## Executive Summary

This report analyzes the prediction difficulty of each position (1-39) across all 3 models 
to identify which positions are consistently hard or easy to predict.

**Key Finding:** Position difficulty reveals model biases and can guide future improvements.

**PO-Defined Threshold:** Positions with <50% overall hit rate are considered "difficult"

---

## Top 5 Hardest Positions to Predict

| Rank | Position | Overall Hit Rate | Appearances | LGBM Hit Rate | SetTransformer Hit Rate | GNN Hit Rate |
|------|----------|------------------|-------------|---------------|------------------------|--------------|
| 1 | 29 | 91.7% | 375 | 75.2% | 100.0% | 100.0% |
| 2 | 31 | 93.7% | 363 | 81.0% | 100.0% | 100.0% |
| 3 | 32 | 93.7% | 459 | 81.0% | 100.0% | 100.0% |
| 4 | 7 | 95.4% | 411 | 86.1% | 100.0% | 100.0% |
| 5 | 9 | 95.8% | 384 | 87.5% | 100.0% | 100.0% |

**Analysis:**
- **Position 29**: Hit rate 91.7% across all models, appeared 375 times
  → SetTransformer performs best at 100.0%
- **Position 31**: Hit rate 93.7% across all models, appeared 363 times
  → SetTransformer performs best at 100.0%
- **Position 32**: Hit rate 93.7% across all models, appeared 459 times
  → SetTransformer performs best at 100.0%
- **Position 7**: Hit rate 95.4% across all models, appeared 411 times
  → SetTransformer performs best at 100.0%
- **Position 9**: Hit rate 95.8% across all models, appeared 384 times
  → SetTransformer performs best at 100.0%

---

## Top 5 Easiest Positions to Predict

| Rank | Position | Overall Hit Rate | Appearances | LGBM Hit Rate | SetTransformer Hit Rate | GNN Hit Rate |
|------|----------|------------------|-------------|---------------|------------------------|--------------|
| 1 | 27 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2 | 1 | 100.0% | 100.0% | 100.0% | 100.0% |
| 3 | 34 | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | 35 | 100.0% | 100.0% | 100.0% | 100.0% |
| 5 | 36 | 100.0% | 100.0% | 100.0% | 100.0% |

---

## Model Comparison by Position

### Model-Specific Strengths

#### LGBM
Positions where LGBM outperforms other models:


#### SETTRANSFORMER
Positions where SETTRANSFORMER outperforms other models:


#### GNN
Positions where GNN outperforms other models:


---

## Position Frequency Analysis

Analyzing relationship between position frequency and prediction difficulty:

**Correlation between frequency and hit rate:** -0.068

**Interpretation:** Position frequency has weak correlation with prediction difficulty

---

## Recommendations

Based on position-level analysis:

1. **Position 30 Analysis** (missed by all models in Draw #11,582):
   - Overall hit rate on holdout: 100.0%
   - This position is NOT particularly difficult on holdout data
   - Real-world miss may be due to pattern shift in future events

2. **Focus on Hardest Positions**: Improve feature engineering for low-performing positions

3. **Leverage Model Strengths**: Consider ensemble strategies where different models handle different positions

4. **Next Steps**: Proceed to Story 8.3 (Boundary Analysis) to check for range-based patterns

---

## Detailed Data

Complete position performance matrix saved to:
- `production/reports/position_analysis/position_performance_matrix.csv`

---

**END OF REPORT**

# Prediction Boundary Analysis Report - Story 8.3

**Date:** 2025-10-16
**Epic:** Epic 8 - Model Performance Analysis
**Story:** 8.3 - Prediction Boundary Analysis

---

## Executive Summary

This report analyzes prediction patterns across position ranges (low 1-13, mid 14-26, high 27-39) 
to identify systematic biases related to the C₃₉ cyclic group structure.

**C₃₉ Ring Structure:** Positions form a cyclic group with 39 elements, potentially creating 
boundary effects at transitions between ranges.

---

## Performance by Position Range

### Recall by Model and Range

| Model | Low (1-13) | Mid (14-26) | High (27-39) |
|-------|------------|-------------|--------------|
| LGBM | 96.5% | 97.6% | 93.6% |
| SETTRANSFORMER | 100.0% | 100.0% | 100.0% |
| GNN | 100.0% | 100.0% | 100.0% |

---

## Prediction Bias by Range

Bias = Predicted - Actual (positive = over-predicting, negative = under-predicting)

| Model | Low (1-13) Bias | Mid (14-26) Bias | High (27-39) Bias |
|-------|-----------------|------------------|-------------------|
| LGBM | +411.5% | +334.9% | +159.3% |
| SETTRANSFORMER | +307.7% | +325.5% | +268.0% |
| GNN | +289.3% | +301.5% | +308.9% |

---

## Boundary Anomalies Detected

**3 anomalies detected:**

### LOW Range
- **Pattern:** over-predicting
- **Severity:** 336.1% bias
- **Models Affected:** lgbm, settransformer, gnn
- **Details:** Average bias: +336.1%

### MID Range
- **Pattern:** over-predicting
- **Severity:** 320.6% bias
- **Models Affected:** lgbm, settransformer, gnn
- **Details:** Average bias: +320.6%

### HIGH Range
- **Pattern:** over-predicting
- **Severity:** 245.4% bias
- **Models Affected:** lgbm, settransformer, gnn
- **Details:** Average bias: +245.4%

---

## Boundary Crossing Analysis

Analyzing prediction accuracy at range boundaries:

### Low/Mid (13-14) Boundary
Positions: 13, 14

| Model | Actual Count | Correct | Accuracy |
|-------|--------------|---------|----------|
| LGBM | 228 | 216 | 94.7% |
| SETTRANSFORMER | 228 | 228 | 100.0% |
| GNN | 228 | 228 | 100.0% |

### Mid/High (26-27) Boundary
Positions: 26, 27

| Model | Actual Count | Correct | Accuracy |
|-------|--------------|---------|----------|
| LGBM | 267 | 254 | 95.1% |
| SETTRANSFORMER | 267 | 267 | 100.0% |
| GNN | 267 | 267 | 100.0% |

---

## Key Findings

### Position Distribution in Actual Outcomes
- **LOW** (1-13): 4920 positions (32.8%)
- **MID** (1-13): 4953 positions (33.0%)
- **HIGH** (1-13): 5127 positions (34.2%)

### Range Prediction Quality
- **LOW** range: 98.8% average recall
- **MID** range: 99.2% average recall
- **HIGH** range: 97.9% average recall

---

## Recommendations

Based on boundary analysis:

1. **Address Identified Biases:** 3 anomalies require attention
   - LOW range: over-predicting by 336.1%
   - MID range: over-predicting by 320.6%
   - HIGH range: over-predicting by 245.4%

2. **Feature Engineering:** Review features for biased ranges to balance predictions

3. **Boundary Positions:** Check if positions 13/14 and 26/27 need special attention

4. **Epic 8 Complete:** Proceed to PO review with comprehensive performance analysis

---

## Detailed Data

Complete range performance data saved to:
- `production/reports/boundary_analysis/range_performance.csv`

---

**END OF REPORT**

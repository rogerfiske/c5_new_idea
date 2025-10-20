# Extended Holdout Testing Report - Story 8.1

**Date:** 2025-10-16
**Epic:** Epic 8 - Model Performance Analysis
**Story:** 8.1 - Extended Holdout Testing

---

## Executive Summary

This report analyzes the performance of all 3 production models (LGBM, SetTransformer, GNN) 
on holdout test sets of varying sizes (100, 500, 1000 events). The goal is to identify 
performance trends and understand model strengths/weaknesses.

### Models Tested
- **LGBM**: Gradient Boosting (LightGBM)
- **SetTransformer**: Attention-based Set Transformer
- **GNN**: Graph Neural Network (GAT)

### Holdout Sizes
- **100 events**: Small sample (most recent)
- **500 events**: Medium sample (most recent)
- **1,000 events**: Full Epic 5 holdout (most recent)

---

## Performance Trends by Sample Size

### Recall@20 by Model and Sample Size

| Model | 100 Events | 500 Events | 1,000 Events | Trend |
|-------|------------|------------|--------------|-------|
| LGBM | 95.80% | 95.72% | 95.88% | 📈 Improving |
| SETTRANSFORMER | 100.00% | 100.00% | 100.00% | ➡️ Stable |
| GNN | 100.00% | 100.00% | 100.00% | ➡️ Stable |

---

## Wrong Predictions Distribution

Breakdown of prediction errors (0-5 wrong out of 5 positions).

### Holdout Size: 100 Events

#### LGBM - 100 Events
```
0 wrong:   79 events (79.00%)
1 wrong:   21 events (21.00%)
2 wrong:    0 events ( 0.00%)
3 wrong:    0 events ( 0.00%)
4 wrong:    0 events ( 0.00%)
5 wrong:    0 events ( 0.00%)

Average wrong predictions: 0.21
Recall@20: 95.80%
```

#### SETTRANSFORMER - 100 Events
```
0 wrong:  100 events (100.00%)
1 wrong:    0 events ( 0.00%)
2 wrong:    0 events ( 0.00%)
3 wrong:    0 events ( 0.00%)
4 wrong:    0 events ( 0.00%)
5 wrong:    0 events ( 0.00%)

Average wrong predictions: 0.00
Recall@20: 100.00%
```

#### GNN - 100 Events
```
0 wrong:  100 events (100.00%)
1 wrong:    0 events ( 0.00%)
2 wrong:    0 events ( 0.00%)
3 wrong:    0 events ( 0.00%)
4 wrong:    0 events ( 0.00%)
5 wrong:    0 events ( 0.00%)

Average wrong predictions: 0.00
Recall@20: 100.00%
```

---

### Holdout Size: 500 Events

#### LGBM - 500 Events
```
0 wrong:  393 events (78.60%)
1 wrong:  107 events (21.40%)
2 wrong:    0 events ( 0.00%)
3 wrong:    0 events ( 0.00%)
4 wrong:    0 events ( 0.00%)
5 wrong:    0 events ( 0.00%)

Average wrong predictions: 0.21
Recall@20: 95.72%
```

#### SETTRANSFORMER - 500 Events
```
0 wrong:  500 events (100.00%)
1 wrong:    0 events ( 0.00%)
2 wrong:    0 events ( 0.00%)
3 wrong:    0 events ( 0.00%)
4 wrong:    0 events ( 0.00%)
5 wrong:    0 events ( 0.00%)

Average wrong predictions: 0.00
Recall@20: 100.00%
```

#### GNN - 500 Events
```
0 wrong:  500 events (100.00%)
1 wrong:    0 events ( 0.00%)
2 wrong:    0 events ( 0.00%)
3 wrong:    0 events ( 0.00%)
4 wrong:    0 events ( 0.00%)
5 wrong:    0 events ( 0.00%)

Average wrong predictions: 0.00
Recall@20: 100.00%
```

---

### Holdout Size: 1000 Events

#### LGBM - 1000 Events
```
0 wrong:  796 events (79.60%)
1 wrong:  202 events (20.20%)
2 wrong:    2 events ( 0.20%)
3 wrong:    0 events ( 0.00%)
4 wrong:    0 events ( 0.00%)
5 wrong:    0 events ( 0.00%)

Average wrong predictions: 0.21
Recall@20: 95.88%
```

#### SETTRANSFORMER - 1000 Events
```
0 wrong: 1000 events (100.00%)
1 wrong:    0 events ( 0.00%)
2 wrong:    0 events ( 0.00%)
3 wrong:    0 events ( 0.00%)
4 wrong:    0 events ( 0.00%)
5 wrong:    0 events ( 0.00%)

Average wrong predictions: 0.00
Recall@20: 100.00%
```

#### GNN - 1000 Events
```
0 wrong: 1000 events (100.00%)
1 wrong:    0 events ( 0.00%)
2 wrong:    0 events ( 0.00%)
3 wrong:    0 events ( 0.00%)
4 wrong:    0 events ( 0.00%)
5 wrong:    0 events ( 0.00%)

Average wrong predictions: 0.00
Recall@20: 100.00%
```

---

## Model Comparison

### Best Model by Sample Size

| Sample Size | Best Model | Recall@20 | Runner-Up |
|-------------|------------|-----------|-----------|
| 100 events | SETTRANSFORMER | 100.00% | GNN (100.00%) |
| 500 events | SETTRANSFORMER | 100.00% | GNN (100.00%) |
| 1000 events | SETTRANSFORMER | 100.00% | GNN (100.00%) |

---

## Key Findings

### Performance Consistency

- **LGBM**: Mean Recall@20 = 95.80%, 
  Std Dev = 0.07%, CV = 0.07%
- **SETTRANSFORMER**: Mean Recall@20 = 100.00%, 
  Std Dev = 0.00%, CV = 0.00%
- **GNN**: Mean Recall@20 = 100.00%, 
  Std Dev = 0.00%, CV = 0.00%

### Sample Size Sensitivity

Performance differences between 100-event and 1,000-event holdout tests:

- **LGBM**: +0.08% difference → Low (stable across sizes)
- **SETTRANSFORMER**: +0.00% difference → Low (stable across sizes)
- **GNN**: +0.00% difference → Low (stable across sizes)

---

## Recommendations

Based on extended holdout testing:

1. **Best Overall Model**: SETTRANSFORMER (highest Recall@20 on full holdout)
2. **Most Consistent Model**: Analyze coefficient of variation above
3. **Next Steps**: Proceed to Story 8.2 (Position-level analysis) to understand 
   why certain positions are harder to predict.

---

## Detailed Results

Detailed per-event metrics saved to:

### LGBM
- `production/reports/extended_holdout/lgbm/holdout_100_events_summary.json`
- `production/reports/extended_holdout/lgbm/holdout_100_events_per_event_metrics.csv`
- `production/reports/extended_holdout/lgbm/holdout_500_events_summary.json`
- `production/reports/extended_holdout/lgbm/holdout_500_events_per_event_metrics.csv`
- `production/reports/extended_holdout/lgbm/holdout_1000_events_summary.json`
- `production/reports/extended_holdout/lgbm/holdout_1000_events_per_event_metrics.csv`

### SETTRANSFORMER
- `production/reports/extended_holdout/settransformer/holdout_100_events_summary.json`
- `production/reports/extended_holdout/settransformer/holdout_100_events_per_event_metrics.csv`
- `production/reports/extended_holdout/settransformer/holdout_500_events_summary.json`
- `production/reports/extended_holdout/settransformer/holdout_500_events_per_event_metrics.csv`
- `production/reports/extended_holdout/settransformer/holdout_1000_events_summary.json`
- `production/reports/extended_holdout/settransformer/holdout_1000_events_per_event_metrics.csv`

### GNN
- `production/reports/extended_holdout/gnn/holdout_100_events_summary.json`
- `production/reports/extended_holdout/gnn/holdout_100_events_per_event_metrics.csv`
- `production/reports/extended_holdout/gnn/holdout_500_events_summary.json`
- `production/reports/extended_holdout/gnn/holdout_500_events_per_event_metrics.csv`
- `production/reports/extended_holdout/gnn/holdout_1000_events_summary.json`
- `production/reports/extended_holdout/gnn/holdout_1000_events_per_event_metrics.csv`

---

**END OF REPORT**

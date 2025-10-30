# RunPod Package v4.0 - Complete Directory Structure

## Critical Fix: Pre-Imputed Architecture
This version eliminates data leakage by using PRE-COMPUTED imputed datasets instead of on-the-fly imputation.

## Package Structure

```
runpod_holdout_v4/
├── README.md                          # Quick start instructions
├── SETUP_AND_RUN.md                   # Detailed setup guide
├── requirements.txt                   # Python dependencies
├── run_holdout_test.sh                # Single command to run test
│
├── data/
│   ├── raw/
│   │   └── c5_Matrix.csv              # Raw data (11589 events)
│   └── imputed/                       # PRE-COMPUTED imputed datasets
│       ├── amplitude_imputed.csv       # 11589 events × 377 features
│       ├── angle_imputed.csv           # 11589 events × 322 features
│       ├── graph_imputed.csv           # 11589 events × 277 features
│       └── density_imputed.csv         # 11589 events × 259 features
│
├── src/
│   ├── __init__.py
│   ├── imputation/
│   │   ├── __init__.py
│   │   ├── base_imputer.py
│   │   ├── amplitude_embedding.py
│   │   ├── angle_encoding.py
│   │   ├── graph_cycle_encoding.py
│   │   └── density_matrix.py
│   │
│   ├── modeling/
│   │   ├── __init__.py
│   │   └── rankers/
│   │       ├── __init__.py
│   │       ├── lgbm_ranker.py
│   │       ├── lgbm_ranker_position_aware.py
│   │       └── position_feature_extractor.py
│   │
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py
│
├── production/
│   ├── true_holdout_test_v4.0_PRE_IMPUTED.py  # Main test script
│   └── analyze_results.py                      # Results analysis
│
└── output/                                     # Pre-created output directories
    ├── reports/                                # Results will be saved here
    │   └── holdout_100/                        # 100-event test results
    │       └── .gitkeep                        # Ensures directory exists
    ├── logs/                                   # Execution logs
    │   └── .gitkeep
    └── checkpoints/                            # Intermediate checkpoints
        └── .gitkeep
```

## What's Different from Previous Versions

### v2.0 (RunPod, FAILED - Data Leakage):
```python
# WRONG: On-the-fly imputation using test event's QV values
test_data = raw_data[raw_data['event-ID'] == test_event_id].copy()  # Has QV!
test_features = imputer.transform(test_data)  # LEAKAGE!
```

### v4.0 (CORRECT - Pre-Imputed):
```python
# RIGHT: Load pre-computed features
imputed_data = pd.read_csv('data/imputed/amplitude_imputed.csv')
train = imputed_data[imputed_data['event-ID'] < test_event_id]
test = imputed_data[imputed_data['event-ID'] == test_event_id]
# Features created BEFORE test, no access to test QV values
```

## Benefits

1. **No Data Leakage**: Test features pre-computed, not from test event's answers
2. **Faster**: ~15 seconds saved per event (no imputation overhead)
3. **Enables Metrics**: More time for detailed per-event metrics collection
4. **Pre-created Directories**: No runtime errors from missing paths
5. **Single Command**: `bash run_holdout_test.sh --events 100`

## Output Structure

After completion, `output/reports/holdout_100/` will contain:

```
checkpoint.json                      # Main results file
├── num_events_completed: 100
├── evaluations: [...]               # Per-event results
└── test_configuration: {...}

detailed_metrics/                    # If --collect-metrics used
├── event_11490_metrics.json
├── event_11491_metrics.json
└── ...

summary_report.txt                   # Human-readable summary
wrong_count_analysis.json            # Model comparison
```

## Quick Start (3 commands)

```bash
# 1. Extract package
tar -xzf runpod_holdout_v4.0.tar.gz
cd runpod_holdout_v4/

# 2. Install dependencies (30 seconds)
pip install -r requirements.txt

# 3. Run test (100 events = ~8-10 hours)
bash run_holdout_test.sh --events 100 --collect-metrics
```

## Results Location

**ALWAYS**: `output/reports/holdout_100/checkpoint.json`

No searching needed - fixed location guaranteed!

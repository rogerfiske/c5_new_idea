#!/bin/bash
#
# Create RunPod Package v4.0 - Pre-Imputed Architecture
# Fixes data leakage by using pre-computed imputed datasets
#
# Usage: bash create_runpod_package_v4.0.sh
#

set -e  # Exit on error

echo "================================================================================"
echo "Creating RunPod Package v4.0 - Pre-Imputed Architecture"
echo "================================================================================"
echo

# Package name
PACKAGE_NAME="runpod_holdout_v4.0"
PACKAGE_DIR="${PACKAGE_NAME}"

# Clean up any existing package
if [ -d "$PACKAGE_DIR" ]; then
    echo "Removing existing package directory..."
    rm -rf "$PACKAGE_DIR"
fi

echo "Creating package directory structure..."

# Create directory structure
mkdir -p "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/data/raw"
mkdir -p "$PACKAGE_DIR/data/imputed"
mkdir -p "$PACKAGE_DIR/src/imputation"
mkdir -p "$PACKAGE_DIR/src/modeling/rankers"
mkdir -p "$PACKAGE_DIR/src/evaluation"
mkdir -p "$PACKAGE_DIR/production"
mkdir -p "$PACKAGE_DIR/output/reports/holdout_100"
mkdir -p "$PACKAGE_DIR/output/logs"
mkdir -p "$PACKAGE_DIR/output/checkpoints"

echo "  [OK] Directory structure created"

# Copy data files
echo "Copying data files..."
cp data/raw/c5_Matrix.csv "$PACKAGE_DIR/data/raw/"
cp data/imputed/amplitude_imputed.csv "$PACKAGE_DIR/data/imputed/"
cp data/imputed/angle_imputed.csv "$PACKAGE_DIR/data/imputed/"
cp data/imputed/graph_imputed.csv "$PACKAGE_DIR/data/imputed/"
cp data/imputed/density_imputed.csv "$PACKAGE_DIR/data/imputed/"
echo "  [OK] Data files copied (raw + 4 imputed datasets)"

# Copy source code
echo "Copying source code..."

# Imputation modules
cp src/imputation/__init__.py "$PACKAGE_DIR/src/imputation/"
cp src/imputation/base_imputer.py "$PACKAGE_DIR/src/imputation/"
cp src/imputation/amplitude_embedding.py "$PACKAGE_DIR/src/imputation/"
cp src/imputation/angle_encoding.py "$PACKAGE_DIR/src/imputation/"
cp src/imputation/graph_cycle_encoding.py "$PACKAGE_DIR/src/imputation/"
cp src/imputation/density_matrix.py "$PACKAGE_DIR/src/imputation/"

# Modeling modules
cp src/modeling/__init__.py "$PACKAGE_DIR/src/modeling/"
cp src/modeling/rankers/__init__.py "$PACKAGE_DIR/src/modeling/rankers/"
cp src/modeling/rankers/lgbm_ranker.py "$PACKAGE_DIR/src/modeling/rankers/"
cp src/modeling/rankers/lgbm_ranker_position_aware.py "$PACKAGE_DIR/src/modeling/rankers/"
cp src/modeling/rankers/position_feature_extractor.py "$PACKAGE_DIR/src/modeling/rankers/"

# Evaluation modules
cp src/evaluation/__init__.py "$PACKAGE_DIR/src/evaluation/"
cp src/evaluation/metrics.py "$PACKAGE_DIR/src/evaluation/"

# Create top-level __init__.py
touch "$PACKAGE_DIR/src/__init__.py"

echo "  [OK] Source code copied"

# Copy production scripts
echo "Copying production scripts..."
cp production/true_holdout_test_v4.0_PRE_IMPUTED.py "$PACKAGE_DIR/production/"
cp production/reports/runpod_100event/analyze_wrong_counts.py "$PACKAGE_DIR/production/analyze_results.py" 2>/dev/null || echo "  [WARN] analyze_wrong_counts.py not found, skipping"
echo "  [OK] Production scripts copied"

# Create .gitkeep files in output directories
touch "$PACKAGE_DIR/output/reports/holdout_100/.gitkeep"
touch "$PACKAGE_DIR/output/logs/.gitkeep"
touch "$PACKAGE_DIR/output/checkpoints/.gitkeep"

# Create requirements.txt
echo "Creating requirements.txt..."
cat > "$PACKAGE_DIR/requirements.txt" << 'EOF'
# RunPod Holdout Test v4.0 Requirements

# Core ML libraries
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0

# LightGBM for ranking
lightgbm==4.0.0

# Progress bars
tqdm==4.66.1

# Optional: System monitoring
psutil==5.9.5
EOF
echo "  [OK] requirements.txt created"

# Create run script
echo "Creating run_holdout_test.sh..."
cat > "$PACKAGE_DIR/run_holdout_test.sh" << 'EOF'
#!/bin/bash
#
# RunPod Holdout Test v4.0 - Single Command Execution
#
# Usage:
#   bash run_holdout_test.sh --events 100                    # No metrics
#   bash run_holdout_test.sh --events 100 --collect-metrics  # With metrics
#

# Default values
NUM_EVENTS=100
COLLECT_METRICS=""
OUTPUT_DIR="output/reports/holdout_100"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --events)
            NUM_EVENTS="$2"
            shift 2
            ;;
        --collect-metrics)
            COLLECT_METRICS="--collect-metrics"
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash run_holdout_test.sh --events N [--collect-metrics] [--output-dir DIR]"
            exit 1
            ;;
    esac
done

echo "================================================================================"
echo "RunPod Holdout Test v4.0 - Pre-Imputed Architecture"
echo "================================================================================"
echo "Events to test: $NUM_EVENTS"
echo "Collect metrics: $([ -n "$COLLECT_METRICS" ] && echo 'YES' || echo 'NO')"
echo "Output directory: $OUTPUT_DIR"
echo "================================================================================"
echo

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Run test
python production/true_holdout_test_v4.0_PRE_IMPUTED.py \
    --num-events $NUM_EVENTS \
    --output-dir "$OUTPUT_DIR" \
    $COLLECT_METRICS

echo
echo "================================================================================"
echo "TEST COMPLETE"
echo "================================================================================"
echo "Results saved to: $OUTPUT_DIR/checkpoint.json"
echo
echo "To view results:"
echo "  cat $OUTPUT_DIR/checkpoint.json | python -m json.tool"
echo
EOF

chmod +x "$PACKAGE_DIR/run_holdout_test.sh"
echo "  [OK] run_holdout_test.sh created"

# Create README
echo "Creating README.md..."
cat > "$PACKAGE_DIR/README.md" << 'EOF'
# RunPod Holdout Test v4.0 - Pre-Imputed Architecture

## Critical Fix: Eliminates Data Leakage

Previous versions (v2.0) performed on-the-fly imputation using test event's QV values, causing data leakage and impossible 100% perfect results.

**v4.0 uses PRE-COMPUTED imputed datasets** - no access to test event answers during feature creation.

## Quick Start (3 Steps)

### 1. Install Dependencies (~30 seconds)
```bash
pip install -r requirements.txt
```

### 2. Run Holdout Test

**100 events without metrics** (~8 hours):
```bash
bash run_holdout_test.sh --events 100
```

**100 events WITH detailed metrics** (~10-12 hours):
```bash
bash run_holdout_test.sh --events 100 --collect-metrics
```

**Quick test (10 events)** (~48 minutes):
```bash
bash run_holdout_test.sh --events 10
```

### 3. View Results

Results are ALWAYS saved to: `output/reports/holdout_100/checkpoint.json`

```bash
# Pretty-print results
cat output/reports/holdout_100/checkpoint.json | python -m json.tool

# View summary
python production/analyze_results.py output/reports/holdout_100/checkpoint.json
```

## What's Inside

- **data/raw/**: Original c5_Matrix.csv (11589 events)
- **data/imputed/**: Pre-computed imputed datasets (4 methods × 11589 events)
- **src/**: Source code (imputation, modeling, evaluation)
- **production/**: Test scripts
- **output/**: Results directory (pre-created, no path errors!)

## Architecture: Pre-Imputed vs On-the-Fly

### Previous (WRONG - Data Leakage):
```python
# Load raw data
test_data = raw[event==11490]  # Contains QV values (the answer!)

# Impute on-the-fly using test answers
test_features = imputer.transform(test_data)  # LEAKAGE!
```

### Current (CORRECT - Pre-Imputed):
```python
# Load PRE-COMPUTED features
imputed_data = pd.read_csv('data/imputed/amplitude_imputed.csv')

# Split by event-ID
train = imputed_data[imputed_data['event-ID'] < 11490]
test = imputed_data[imputed_data['event-ID'] == 11490]

# Features created BEFORE test, no access to test QV values
```

## Expected Results

With data leakage fixed, expect **realistic performance**:

- **Recall@20**: 60-80% per model (NOT 100%!)
- **Wrong count distribution**: Mix of 0-5 wrong across events
- **Model differences**: Different performance across 4 methods

## Timing

- **Per event**: ~5 minutes (4 models trained in parallel)
- **100 events**: ~8 hours without metrics, ~10-12 hours with metrics
- **Time breakdown**:
  - Data loading: <1s
  - Model training: ~2.5 min (87% of time)
  - Prediction: <1s
  - Metrics collection (if enabled): ~1-2 min

## Support

Questions? Check SETUP_AND_RUN.md for detailed instructions.
EOF
echo "  [OK] README.md created"

# Create detailed setup guide
echo "Creating SETUP_AND_RUN.md..."
cat > "$PACKAGE_DIR/SETUP_AND_RUN.md" << 'EOF'
# RunPod Setup and Execution Guide - v4.0

## Pre-Flight Checklist

Before starting, verify:
- [ ] GPU pod with Python 3.8+ installed
- [ ] At least 16GB RAM available
- [ ] At least 10GB disk space free
- [ ] Internet access for pip install

## Step-by-Step Setup

### 1. Upload Package to RunPod

Option A: Upload via web interface
- Open RunPod web terminal
- Click "Upload Files"
- Select `runpod_holdout_v4.0.tar.gz`

Option B: Download from URL
```bash
wget https://your-url/runpod_holdout_v4.0.tar.gz
```

### 2. Extract Package

```bash
tar -xzf runpod_holdout_v4.0.tar.gz
cd runpod_holdout_v4.0
```

Verify structure:
```bash
ls -la
# Should see: data/ src/ production/ output/ requirements.txt README.md
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Expected output:
```
Successfully installed numpy-1.24.3 pandas-2.0.3 scikit-learn-1.3.0 lightgbm-4.0.0 ...
```

### 4. Verify Setup

Quick verification test (1 event, ~5 min):
```bash
bash run_holdout_test.sh --events 1
```

Check results:
```bash
ls -lh output/reports/holdout_100/checkpoint.json
# Should show file with data
```

### 5. Run Full Test

**100-event test without metrics** (~8 hours):
```bash
nohup bash run_holdout_test.sh --events 100 > output/logs/run.log 2>&1 &
```

**With metrics** (~10-12 hours):
```bash
nohup bash run_holdout_test.sh --events 100 --collect-metrics > output/logs/run.log 2>&1 &
```

Monitor progress:
```bash
tail -f output/logs/run.log
```

### 6. Monitor Progress

Check completed events:
```bash
python -c "import json; d=json.load(open('output/reports/holdout_100/checkpoint.json')); print(f'Completed: {d[\"num_events_completed\"]}/100')"
```

### 7. Download Results

After completion:
```bash
# Create results archive
tar -czf results_v4.0.tar.gz output/reports/holdout_100/

# Download via RunPod web interface
# Or use scp if SSH enabled
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"
**Solution**: Make sure you're in the `runpod_holdout_v4.0` directory
```bash
pwd  # Should end with /runpod_holdout_v4.0
```

### "FileNotFoundError: data/imputed/amplitude_imputed.csv"
**Solution**: Verify data files extracted correctly
```bash
ls -lh data/imputed/
# Should show 4 CSV files (amplitude, angle, graph, density)
```

### "Process killed" or "Out of memory"
**Solution**: Reduce batch size or use GPU pod with more RAM

### Test taking too long
**Expected**: 100 events = ~8-10 hours (normal!)
- Each event: ~5 minutes
- Models train in parallel (4 models simultaneously)

## Advanced Options

### Custom Output Directory
```bash
bash run_holdout_test.sh --events 50 --output-dir output/reports/test_50
```

### Resume from Checkpoint
Script automatically resumes if checkpoint.json exists (NOT YET IMPLEMENTED in v4.0)

### Collect Selective Metrics
Modify script to collect metrics every 10th event (reduces overhead)
```python
# In true_holdout_test_v4.0_PRE_IMPUTED.py, line 344:
collect_metrics_for_event = (event_index % 10 == 0)  # Every 10th event
```

## Results Structure

After completion, `output/reports/holdout_100/` contains:

```
checkpoint.json                    # Main results (ALWAYS HERE!)
├── last_completed_event
├── num_events_completed
├── timestamp
└── evaluations: [
      {
        event_id: 11490,
        actual_winners: [3, 15, 20, 35, 39],
        models: {
          amplitude: { predictions: [...], hits: 4, recall_at_k: 0.8, wrong_count: 1 },
          angle_encoding: { ... },
          graph_cycle: { ... },
          density_matrix: { ... }
        }
      },
      ...
    ]
```

## Contact

Issues? Document in RUNPOD_ISSUES.md in project root.
EOF
echo "  [OK] SETUP_AND_RUN.md created"

# Package summary
echo
echo "================================================================================"
echo "PACKAGE CREATION COMPLETE"
echo "================================================================================"
echo "Package directory: $PACKAGE_DIR/"
echo
echo "Contents:"
echo "  - Data: c5_Matrix.csv + 4 imputed datasets"
echo "  - Source: Complete imputation + modeling + evaluation code"
echo "  - Scripts: true_holdout_test_v4.0_PRE_IMPUTED.py + run script"
echo "  - Docs: README.md + SETUP_AND_RUN.md"
echo "  - Output: Pre-created directories (no runtime path errors!)"
echo
echo "Next steps:"
echo "  1. Create tarball:"
echo "     tar -czf ${PACKAGE_NAME}.tar.gz $PACKAGE_DIR/"
echo
echo "  2. Test locally:"
echo "     cd $PACKAGE_DIR && bash run_holdout_test.sh --events 3"
echo
echo "  3. Upload to RunPod:"
echo "     Upload ${PACKAGE_NAME}.tar.gz via web interface"
echo
echo "================================================================================"

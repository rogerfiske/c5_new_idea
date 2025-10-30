"""
TRUE Holdout Test v4.0 - Using Pre-Imputed Datasets
===================================================

CRITICAL FIX: This version uses PRE-IMPUTED datasets instead of on-the-fly imputation.

The Previous Problem (v3.0 and RunPod):
- Loaded raw c5_Matrix.csv
- For test event N: extracted QV values (the answer!)
- Called imputer.transform(test_data_with_QV) = DATA LEAKAGE

The Correct Architecture (v4.0):
- Use pre-computed imputed datasets from data/imputed/*.csv
- These were created by regenerate_imputed_datasets.py AFTER events occurred
- Train on events 1 to N-1
- Test on event N's PRE-COMPUTED features
- No on-the-fly imputation = No leakage

Benefits:
- NO data leakage (features pre-computed before test)
- Faster (~15s saved per event - no imputation overhead)
- Enables detailed metrics collection
- Matches real-world workflow (you regenerate imputed data after ingestion)

Author: BMad Dev Agent (James)
Date: 2025-10-30
Version: 4.0 - Pre-Imputed Architecture
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import time
from multiprocessing import Pool
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.modeling.rankers.lgbm_ranker_position_aware import LGBMRankerPositionAware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Imputed dataset paths
IMPUTED_DATASETS = {
    'amplitude': 'data/imputed/amplitude_imputed.csv',
    'angle_encoding': 'data/imputed/angle_imputed.csv',
    'graph_cycle': 'data/imputed/graph_imputed.csv',
    'density_matrix': 'data/imputed/density_imputed.csv'
}


def load_pre_imputed_data(method_name: str) -> pd.DataFrame:
    """
    Load pre-computed imputed dataset.

    Args:
        method_name: One of ['amplitude', 'angle', 'graph', 'density']

    Returns:
        DataFrame with pre-computed features for all events
    """
    data_path = project_root / IMPUTED_DATASETS[method_name]

    if not data_path.exists():
        raise FileNotFoundError(
            f"Pre-imputed dataset not found: {data_path}\n"
            f"Please run: python production/regenerate_imputed_datasets.py"
        )

    logger.info(f"Loading pre-imputed {method_name} data: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"  Loaded {len(df)} events (event-ID: {df['event-ID'].min()} to {df['event-ID'].max()})")
    logger.info(f"  Features: {len([col for col in df.columns if col.startswith('feat_')])} feature columns")

    return df


def train_and_predict_single_method_PRE_IMPUTED(args):
    """
    Train on pre-imputed data (events 1 to N-1) and predict event N.

    KEY DIFFERENCE FROM v3.0:
    - Uses pre-imputed features (NO on-the-fly imputation)
    - Test features created BEFORE this test (by regenerate_imputed_datasets.py)
    - No access to test event's QV values during feature creation

    Args:
        args: Tuple of (method_name, imputed_data_path, train_event_ids, test_event_id, k, collect_metrics)

    Returns:
        Tuple of (method_name, top_k, timing, memory_gb, metrics)
    """
    method_name, imputed_data_path, train_event_ids, test_event_id, k, collect_metrics = args

    timing = {}
    start_time = time.time()

    # Load pre-imputed data
    load_start = time.time()
    imputed_data = pd.read_csv(imputed_data_path)
    timing['data_load'] = time.time() - load_start

    # Split train/test (by event-ID)
    split_start = time.time()
    train_data = imputed_data[imputed_data['event-ID'].isin(train_event_ids)].copy()
    test_data = imputed_data[imputed_data['event-ID'] == test_event_id].copy()

    if len(test_data) == 0:
        raise ValueError(f"Test event {test_event_id} not found in imputed dataset!")

    timing['data_split'] = time.time() - split_start

    # Train model
    train_start = time.time()
    model = LGBMRankerPositionAware(
        params={
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'num_leaves': 31,
            'min_child_samples': 20,
            'n_jobs': 4  # 4 threads per model
        },
        imputation_method=method_name
    )
    model.fit(train_data)
    timing['model_train'] = time.time() - train_start

    # Predict
    predict_start = time.time()
    top_k = model.predict_top_k(test_data, k=k)[0]
    timing['model_predict'] = time.time() - predict_start

    # Collect detailed metrics if requested
    metrics = None
    if collect_metrics:
        metrics_start = time.time()
        metrics = {}

        # Position scores for all 39 positions
        position_scores = model.predict_position_scores(test_data)[0]  # Shape: (39,)
        metrics['position_scores'] = position_scores.tolist()

        # Feature importances
        feature_importances = model.model_.feature_importances_
        feature_names = model.feature_names_

        # Top 20 most important features
        top_indices = np.argsort(feature_importances)[-20:][::-1]
        metrics['top_features'] = [
            {
                'name': feature_names[idx],
                'importance': float(feature_importances[idx])
            }
            for idx in top_indices
        ]

        timing['metrics_collection'] = time.time() - metrics_start

    timing['total'] = time.time() - start_time

    # Memory usage (approx)
    try:
        import psutil
        process = psutil.Process()
        memory_gb = process.memory_info().rss / (1024 ** 3)
    except:
        memory_gb = 0.0

    return (method_name, top_k.tolist(), timing, memory_gb, metrics)


def predict_single_event_parallel_PRE_IMPUTED(
    test_event_id: int,
    k: int = 20,
    n_processes: int = 4,
    collect_metrics: bool = False
) -> dict:
    """
    Predict a single event using TRUE holdout with pre-imputed datasets.

    Trains all 4 models in parallel, each using pre-computed features.

    Args:
        test_event_id: Event to predict
        k: Number of top predictions
        n_processes: Number of parallel processes (4 for 4 models)
        collect_metrics: Whether to collect detailed metrics

    Returns:
        Dictionary with predictions and timing for all 4 models
    """
    # Verify all imputed datasets exist
    for method_name, rel_path in IMPUTED_DATASETS.items():
        full_path = project_root / rel_path
        if not full_path.exists():
            raise FileNotFoundError(
                f"Missing pre-imputed dataset: {full_path}\n"
                f"Run: python production/regenerate_imputed_datasets.py"
            )

    # Load one dataset to get event range
    sample_data = pd.read_csv(project_root / IMPUTED_DATASETS['amplitude'])
    all_event_ids = sorted(sample_data['event-ID'].unique())

    if test_event_id not in all_event_ids:
        raise ValueError(
            f"Event {test_event_id} not in imputed datasets (range: {min(all_event_ids)} to {max(all_event_ids)})"
        )

    # Training data: all events before test_event_id
    train_event_ids = [eid for eid in all_event_ids if eid < test_event_id]

    logger.info(f"\nPredicting event {test_event_id}...")
    logger.info(f"  Training on {len(train_event_ids)} events (ID: {min(train_event_ids)} to {max(train_event_ids)})")
    logger.info(f"  Collect metrics: {collect_metrics}")

    # Prepare arguments for parallel processing
    args_list = [
        (
            method_name,
            str(project_root / rel_path),
            train_event_ids,
            test_event_id,
            k,
            collect_metrics
        )
        for method_name, rel_path in IMPUTED_DATASETS.items()
    ]

    # Train and predict in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.map(train_and_predict_single_method_PRE_IMPUTED, args_list)

    # Get actual winners from raw data
    raw_data = pd.read_csv(project_root / 'data' / 'raw' / 'c5_Matrix.csv')
    test_row = raw_data[raw_data['event-ID'] == test_event_id].iloc[0]
    actual_winners = [int(i) for i in range(1, 40) if test_row[f'QV_{i}'] == 1]

    # Organize results
    evaluation = {
        'event_id': int(test_event_id),
        'actual_winners': actual_winners,
        'models': {}
    }

    for method_name, top_k, timing, memory_gb, metrics in results:
        predicted_set = set(top_k)
        actual_set = set(actual_winners)

        # Calculate recall@k
        hits = len(predicted_set & actual_set)
        recall_at_k = hits / len(actual_set) if len(actual_set) > 0 else 0.0

        # Wrong count
        wrong_count = len(actual_set - predicted_set)

        model_result = {
            'predictions': [int(p) for p in top_k],
            'hits': int(hits),
            'recall_at_k': float(recall_at_k),
            'wrong_count': int(wrong_count),
            'timing': timing,
            'memory_gb': float(memory_gb)
        }

        if metrics is not None:
            model_result['metrics'] = metrics

        evaluation['models'][method_name] = model_result

    return evaluation


def main():
    parser = argparse.ArgumentParser(description='TRUE Holdout Test v4.0 - Pre-Imputed Datasets')
    parser.add_argument('--num-events', type=int, default=10, help='Number of events to test')
    parser.add_argument('--start-event', type=int, default=None, help='First event to test (default: last N events)')
    parser.add_argument('--collect-metrics', action='store_true', help='Collect detailed per-event metrics')
    parser.add_argument('--output-dir', type=str, default='production/reports/holdout_v4_pre_imputed', help='Output directory')

    args = parser.parse_args()

    print("=" * 80)
    print("TRUE HOLDOUT TEST v4.0 - Pre-Imputed Datasets Architecture")
    print("=" * 80)
    print()
    print("Test Configuration:")
    print(f"  Number of events: {args.num_events}")
    print(f"  Parallel processes: 4 (one per imputation method)")
    print(f"  Collect detailed metrics: {args.collect_metrics}")
    print(f"  Output directory: {args.output_dir}")
    print()

    # Load sample to get event range
    sample_data = pd.read_csv(project_root / IMPUTED_DATASETS['amplitude'])
    all_event_ids = sorted(sample_data['event-ID'].unique())
    max_event = max(all_event_ids)

    print(f"Pre-imputed data range: {min(all_event_ids)} to {max_event}")
    print()

    # Determine test events
    if args.start_event is not None:
        start_event = args.start_event
        end_event = min(start_event + args.num_events - 1, max_event)
    else:
        # Last N events
        end_event = max_event
        start_event = max(end_event - args.num_events + 1, 2)  # Need at least event 1 for training

    test_event_ids = list(range(start_event, end_event + 1))

    print(f"  Test range: Events {start_event} to {end_event}")
    print(f"  Training will use events 1 to (N-1) for each test event")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run predictions
    print("=" * 80)
    print("RUNNING PREDICTIONS...")
    print("=" * 80)

    evaluations = []

    for test_event_id in tqdm(test_event_ids, desc="Processing events"):
        try:
            evaluation = predict_single_event_parallel_PRE_IMPUTED(
                test_event_id=test_event_id,
                k=20,
                n_processes=4,
                collect_metrics=args.collect_metrics
            )
            evaluations.append(evaluation)

        except Exception as e:
            logger.error(f"Failed to predict event {test_event_id}: {e}")
            continue

    # Save checkpoint
    checkpoint = {
        'last_completed_event': test_event_ids[-1] if evaluations else None,
        'num_events_completed': len(evaluations),
        'timestamp': pd.Timestamp.now().isoformat(),
        'test_configuration': {
            'start_event': start_event,
            'end_event': end_event,
            'num_processes': 4,
            'collect_metrics': args.collect_metrics,
            'architecture': 'pre_imputed_v4.0'
        },
        'evaluations': evaluations
    }

    checkpoint_path = output_dir / 'checkpoint.json'
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    # Print wrong count distribution for each model
    for method_name in ['amplitude', 'angle_encoding', 'graph_cycle', 'density_matrix']:
        wrong_count_dist = {i: 0 for i in range(6)}

        for eval_data in evaluations:
            model_data = eval_data['models'].get(method_name, {})
            wrong_count = model_data.get('wrong_count', 5)
            if wrong_count in wrong_count_dist:
                wrong_count_dist[wrong_count] += 1

        total_events = len(evaluations)

        print(f"\nHOLDOUT TEST SUMMARY - {total_events} Events")
        print(f"{method_name.upper()} Embedding")
        print("  " + "-" * 60)

        descriptions = {
            0: "All 5 actual values in top-20",
            1: "4 of 5 actual values in top-20",
            2: "3 of 5 actual values in top-20",
            3: "2 of 5 actual values in top-20",
            4: "1 of 5 actual values in top-20",
            5: "0 of 5 actual values in top-20"
        }

        for wrong_count in range(6):
            count = wrong_count_dist[wrong_count]
            percentage = (count / total_events) * 100 if total_events > 0 else 0
            desc = descriptions[wrong_count]
            print(f"  {wrong_count} wrong: {count:3d} events ({percentage:5.2f}%)  <-- {desc}")

    print()
    print(f"\nResults saved to: {checkpoint_path}")
    print()


if __name__ == '__main__':
    main()

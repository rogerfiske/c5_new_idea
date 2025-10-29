#!/usr/bin/env python3.10
"""Analyze 100-event TRUE holdout test results."""
import json
import sys

def main():
    checkpoint_path = '/workspace/production/reports/true_holdout_100/checkpoint.json'

    try:
        with open(checkpoint_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        sys.exit(1)

    # Collect recalls
    recalls = {
        'amplitude': [],
        'angle_encoding': [],
        'graph_cycle': [],
        'density_matrix': [],
        'ensemble': []
    }

    for ev in data['evaluations']:
        for model in recalls.keys():
            recalls[model].append(ev[model]['recall_at_20'])

    # Print results
    print("\n" + "="*60)
    print("🎯 100-EVENT TRUE HOLDOUT TEST RESULTS")
    print("="*60)
    print(f"\nEvents Tested: {data['num_events_completed']}")
    print(f"Event Range: {data['evaluations'][0]['event_id']} - {data['evaluations'][-1]['event_id']}")
    print(f"\n{'Model':<20} {'Mean Recall':<12} {'Min':<8} {'Max':<8}")
    print("-" * 55)

    for model, values in recalls.items():
        mean = sum(values) / len(values) * 100
        min_val = min(values) * 100
        max_val = max(values) * 100
        print(f"{model:<20} {mean:>6.1f}%       {min_val:>6.1f}%  {max_val:>6.1f}%")

    # Count perfect scores
    perfect_ensemble = sum(1 for r in recalls['ensemble'] if r == 1.0)
    print(f"\n✅ Perfect Ensemble Predictions: {perfect_ensemble}/100 ({perfect_ensemble}%)")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()

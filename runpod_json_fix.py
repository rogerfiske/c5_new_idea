#!/usr/bin/env python3
"""
RunPod JSON Fix - Patch for numpy serialization bug
====================================================

This script patches the true_holdout_test_v4.0_PRE_IMPUTED.py file
to fix the JSON serialization error with numpy int64/float64 types.

Usage:
    python runpod_json_fix.py

The script will:
1. Read production/true_holdout_test_v4.0_PRE_IMPUTED.py
2. Add JSON converter function
3. Update checkpoint saving to use converter
4. Save the patched file

Author: BMad Dev Agent (James)
Date: 2025-10-30
"""

import os
from pathlib import Path

def apply_json_fix():
    """Apply JSON serialization fix to the test script"""

    # File path
    file_path = Path('production/true_holdout_test_v4.0_PRE_IMPUTED.py')

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        print("Make sure you're in the runpod_holdout_v4.0 directory!")
        return False

    # Read original content
    with open(file_path, 'r') as f:
        content = f.read()

    # Check if already patched
    if 'convert_to_json_serializable' in content:
        print("✓ File already patched - no changes needed")
        return True

    # Define the search pattern and replacement
    old_code = """    # Save checkpoint
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
        json.dump(checkpoint, f, indent=2)"""

    new_code = """    # Convert numpy types to Python native types for JSON serialization
    def convert_to_json_serializable(obj):
        \"\"\"Recursively convert numpy types to Python native types\"\"\"
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # Save checkpoint
    checkpoint = {
        'last_completed_event': int(test_event_ids[-1]) if evaluations else None,
        'num_events_completed': len(evaluations),
        'timestamp': pd.Timestamp.now().isoformat(),
        'test_configuration': {
            'start_event': int(start_event),
            'end_event': int(end_event),
            'num_processes': 4,
            'collect_metrics': args.collect_metrics,
            'architecture': 'pre_imputed_v4.0'
        },
        'evaluations': convert_to_json_serializable(evaluations)
    }

    checkpoint_path = output_dir / 'checkpoint.json'
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)"""

    # Apply the patch
    if old_code in content:
        patched_content = content.replace(old_code, new_code)

        # Write patched content
        with open(file_path, 'w') as f:
            f.write(patched_content)

        print("✓ JSON serialization fix applied successfully!")
        print(f"✓ File patched: {file_path}")
        print()
        print("The fix adds:")
        print("  1. convert_to_json_serializable() function")
        print("  2. Explicit int() conversions for start_event, end_event, last_completed_event")
        print("  3. Recursive numpy type conversion for all evaluation data")
        print()
        print("You can now run the holdout test without JSON errors:")
        print("  bash run_holdout_test.sh --events 1")
        return True
    else:
        print("ERROR: Could not find the code to patch!")
        print("The file structure may be different than expected.")
        return False

if __name__ == '__main__':
    print("=" * 80)
    print("RunPod JSON Serialization Fix")
    print("=" * 80)
    print()

    # Check current directory
    if not Path('production').exists():
        print("ERROR: 'production' directory not found!")
        print("Current directory:", Path.cwd())
        print()
        print("Please run this script from the runpod_holdout_v4.0 directory:")
        print("  cd runpod_holdout_v4.0")
        print("  python runpod_json_fix.py")
        exit(1)

    # Apply fix
    success = apply_json_fix()

    print()
    print("=" * 80)

    if success:
        print("SUCCESS - Ready to run holdout test!")
    else:
        print("FAILED - Please contact support")

    print("=" * 80)

"""
Story 9B.1: Evaluate Simple Ensemble on Holdout Set

This script evaluates the SimpleEnsemble (equal weighting) by combining
predictions from three Density Matrix models (LGBM, SetTransformer, GNN)
and comparing performance against individual models.

Author: BMad Dev Agent (James)
Date: 2025-10-17
Epic: Epic 9B - Ensemble & Bias Correction
Story: 9B.1 - Simple Ensemble Implementation

NFR2: Non-Programmer Friendly
This script is heavily commented to explain the ensemble evaluation process,
including model loading, prediction aggregation, and performance comparison.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import torch
import json
from typing import Dict, List, Tuple
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.modeling.ensembles.simple_ensemble import SimpleEnsemble
from src.modeling.rankers.lgbm_ranker import LGBMRanker
from src.modeling.rankers.deepsets_ranker import SetTransformerRanker
from src.modeling.rankers.gnn_ranker import GNNRanker


def load_holdout_data(data_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load the holdout dataset for evaluation.

    Why Holdout Data?
    -----------------
    The holdout set contains 1,000 events that the models have never seen during
    training. This allows us to test real-world prediction performance.

    Args:
        data_path: Path to density_holdout.csv (from data/splits/density/)

    Returns:
        df: Full dataframe with all columns
        X_test: Feature matrix (n_events, n_features)
        y_test: Target labels - winning positions per event
    """
    print(f"Loading holdout data from {data_path}...")
    df = pd.read_csv(data_path)

    # Separate features and targets
    # Exclude only target columns and metadata
    # NOTE: Keep QV columns - they're needed by LGBM models
    feature_cols = [col for col in df.columns if col not in
                   ['draw_number', 'draw_date', 'q_1', 'q_2', 'q_3', 'q_4', 'q_5',
                    'event-ID']]  # Exclude only targets and metadata
    X_test = df[feature_cols]  # Keep as DataFrame

    # Target: 5 winning positions per event
    target_cols = ['q_1', 'q_2', 'q_3', 'q_4', 'q_5']
    y_test = df[target_cols].values.astype(int)

    print(f"Loaded {len(df)} holdout events")
    print(f"Feature dimensions: {X_test.shape}")
    print(f"Target dimensions: {y_test.shape}")

    return X_test, y_test


def load_density_matrix_models() -> Dict[str, any]:
    """
    Load all three trained Density Matrix models.

    Why These Models?
    -----------------
    Epic 9A testing showed Density Matrix imputation achieved 100% holdout recall
    across all three model types (LGBM, SetTransformer, GNN). These are the best
    performing models to combine in an ensemble.

    Returns:
        models: Dict with keys 'lgbm', 'settransformer', 'gnn'
    """
    print("\n" + "="*80)
    print("Loading Density Matrix Models")
    print("="*80)

    models = {}
    model_dir = Path("production/models/density_matrix")

    # Load LGBM
    print(f"\n1. Loading LGBM from {model_dir / 'lgbm_ranker.pkl'}...")
    with open(model_dir / 'lgbm_ranker.pkl', 'rb') as f:
        models['lgbm'] = pickle.load(f)
    print(f"   [OK] LGBM loaded (file size: {(model_dir / 'lgbm_ranker.pkl').stat().st_size / 1024:.1f} KB)")

    # Load SetTransformer
    print(f"\n2. Loading SetTransformer from {model_dir / 'settransformer_ranker.pth'}...")
    models['settransformer'] = SetTransformerRanker.load_model(str(model_dir / 'settransformer_ranker.pth'))
    print(f"   [OK] SetTransformer loaded (file size: {(model_dir / 'settransformer_ranker.pth').stat().st_size / 1024 / 1024:.1f} MB)")

    # Load GNN
    print(f"\n3. Loading GNN from {model_dir / 'gnn_ranker.pth'}...")
    models['gnn'] = GNNRanker.load_model(str(model_dir / 'gnn_ranker.pth'))
    print(f"   [OK] GNN loaded (file size: {(model_dir / 'gnn_ranker.pth').stat().st_size / 1024 / 1024:.1f} MB)")

    print(f"\nAll 3 models loaded successfully!")
    return models


def evaluate_model(model, X_test: pd.DataFrame, y_test: np.ndarray,
                  model_name: str) -> Dict:
    """
    Evaluate a single model on the holdout set.

    Evaluation Metric: Recall@20
    ----------------------------
    Recall@20 measures: Of the 5 actual winning positions per event, how many
    appear in the model's top-20 predictions?

    100% Recall@20 = All 5 winning positions in top-20 for every event

    Args:
        model: Trained ranker model
        X_test: Holdout features as DataFrame (1000 events, n_features)
        y_test: Actual winning positions (1000 events, 5 positions each)
        model_name: Name for display purposes

    Returns:
        results: Dict with recall, per-event metrics, and predictions
    """
    print(f"\nEvaluating {model_name}...")

    n_events = len(X_test)
    recalls = []
    all_predictions = []

    for i in range(n_events):
        event_features = X_test.iloc[[i]]  # Keep as DataFrame (single row)
        actual_positions = y_test[i]

        # Get top-20 predictions
        predicted_top_20 = model.predict_top_k(event_features, k=20)[0]
        all_predictions.append(predicted_top_20)

        # Calculate recall: how many of 5 actual positions are in top-20?
        matches = np.isin(actual_positions, predicted_top_20)
        recall = matches.sum() / 5.0
        recalls.append(recall)

    avg_recall = np.mean(recalls)
    perfect_recall_count = sum(1 for r in recalls if r == 1.0)
    perfect_recall_pct = perfect_recall_count / n_events * 100

    print(f"  Recall@20: {avg_recall*100:.2f}%")
    print(f"  Perfect Recall Events: {perfect_recall_count}/{n_events} ({perfect_recall_pct:.1f}%)")

    return {
        'model_name': model_name,
        'avg_recall': avg_recall,
        'perfect_recall_count': perfect_recall_count,
        'perfect_recall_pct': perfect_recall_pct,
        'per_event_recalls': recalls,
        'predictions': all_predictions
    }


def evaluate_ensemble(ensemble: SimpleEnsemble, X_test: pd.DataFrame,
                     y_test: np.ndarray) -> Dict:
    """
    Evaluate the ensemble on the holdout set.

    Why Ensemble?
    -------------
    Combining multiple models can improve robustness by averaging out individual
    model biases. If one model is weak on LOW range and another is weak on HIGH
    range, the ensemble can balance these weaknesses.

    Args:
        ensemble: SimpleEnsemble instance
        X_test: Holdout features as DataFrame
        y_test: Actual winning positions

    Returns:
        results: Dict with recall, per-event metrics, and predictions
    """
    print(f"\nEvaluating SimpleEnsemble (equal weighting)...")

    n_events = len(X_test)
    recalls = []
    all_predictions = []

    for i in range(n_events):
        event_features = X_test.iloc[[i]]  # Keep as DataFrame (single row)
        actual_positions = y_test[i]

        # Get ensemble top-20 predictions
        predicted_top_20, _ = ensemble.predict(event_features, k=20)
        all_predictions.append(predicted_top_20)

        # Calculate recall
        matches = np.isin(actual_positions, predicted_top_20)
        recall = matches.sum() / 5.0
        recalls.append(recall)

    avg_recall = np.mean(recalls)
    perfect_recall_count = sum(1 for r in recalls if r == 1.0)
    perfect_recall_pct = perfect_recall_count / n_events * 100

    print(f"  Recall@20: {avg_recall*100:.2f}%")
    print(f"  Perfect Recall Events: {perfect_recall_count}/{n_events} ({perfect_recall_pct:.1f}%)")

    return {
        'model_name': 'SimpleEnsemble (equal weights)',
        'avg_recall': avg_recall,
        'perfect_recall_count': perfect_recall_count,
        'perfect_recall_pct': perfect_recall_pct,
        'per_event_recalls': recalls,
        'predictions': all_predictions
    }


def generate_evaluation_report(individual_results: List[Dict],
                               ensemble_results: Dict,
                               model_contributions: Dict,
                               output_path: str):
    """
    Generate comprehensive evaluation report in Markdown format.

    Report Structure:
    ----------------
    - Executive Summary: Ensemble vs individual models
    - Performance Comparison Table
    - Model Contributions Analysis
    - Recommendation: Deploy ensemble or stick with single model?

    Args:
        individual_results: List of dicts from evaluate_model() for each individual model
        ensemble_results: Dict from evaluate_ensemble()
        model_contributions: Dict from ensemble.get_model_contributions()
        output_path: Path to save the report
    """
    print(f"\nGenerating evaluation report...")

    # Determine best individual model
    best_individual = max(individual_results, key=lambda x: x['avg_recall'])

    # Compare ensemble to best individual
    ensemble_recall = ensemble_results['avg_recall']
    best_recall = best_individual['avg_recall']
    recall_diff = ensemble_recall - best_recall

    report_lines = [
        "# Story 9B.1: Simple Ensemble Evaluation Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Epic:** Epic 9B - Ensemble & Bias Correction",
        f"**Story:** 9B.1 - Simple Ensemble Implementation",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"### Ensemble Performance",
        f"- **Ensemble Recall@20:** {ensemble_recall*100:.2f}%",
        f"- **Best Individual Model:** {best_individual['model_name']} ({best_recall*100:.2f}%)",
        f"- **Performance Difference:** {recall_diff*100:+.2f}% ({'+improvement' if recall_diff > 0 else 'no improvement'})",
        "",
    ]

    # Add recommendation based on results
    if recall_diff >= 0:
        report_lines.extend([
            "### Recommendation: [OK] Proceed with Ensemble",
            "",
            f"The ensemble {'matches' if recall_diff == 0 else 'improves upon'} the best individual model. ",
            "Ensembles provide robustness by combining multiple perspectives, even if performance ",
            "is equivalent. Proceed to Story 9B.2 (Bias Correction) using this ensemble.",
            "",
        ])
    else:
        report_lines.extend([
            "### Recommendation: [WARNING] Consider Best Individual Model",
            "",
            f"The ensemble performs {abs(recall_diff)*100:.2f}% worse than the best individual model ",
            f"({best_individual['model_name']}). This suggests the ensemble weighting may need tuning, ",
            "or one model is so dominant that averaging dilutes its performance.",
            "",
            f"**Options:**",
            f"1. Proceed with {best_individual['model_name']} alone for Story 9B.2",
            f"2. Investigate model contributions to understand why ensemble underperforms",
            f"3. Try confidence-based weighting instead of equal weighting",
            "",
        ])

    report_lines.extend([
        "---",
        "",
        "## Holdout Performance Comparison",
        "",
        "### Performance Table",
        "",
        "| Model | Recall@20 | Perfect Recall Events | Perfect Recall % |",
        "|-------|-----------|----------------------|------------------|",
    ])

    # Add individual model rows
    for result in individual_results:
        report_lines.append(
            f"| {result['model_name']} | "
            f"{result['avg_recall']*100:.2f}% | "
            f"{result['perfect_recall_count']}/1000 | "
            f"{result['perfect_recall_pct']:.1f}% |"
        )

    # Add ensemble row
    report_lines.append(
        f"| **{ensemble_results['model_name']}** | "
        f"**{ensemble_results['avg_recall']*100:.2f}%** | "
        f"**{ensemble_results['perfect_recall_count']}/1000** | "
        f"**{ensemble_results['perfect_recall_pct']:.1f}%** |"
    )

    report_lines.extend([
        "",
        "### Key Observations",
        "",
    ])

    # Check if all models have 100% recall
    all_perfect = all(r['avg_recall'] == 1.0 for r in individual_results)
    if all_perfect and ensemble_results['avg_recall'] == 1.0:
        report_lines.extend([
            "- [OK] All models (individual and ensemble) achieve 100% holdout recall",
            "- [OK] No model is clearly inferior - ensemble maintains perfect performance",
            "- [OK] Ensemble provides redundancy without sacrificing accuracy",
            "",
        ])
    elif ensemble_results['avg_recall'] == 1.0:
        report_lines.extend([
            f"- [OK] Ensemble achieves 100% holdout recall",
            f"- [WARNING] Some individual models have lower recall - ensemble compensates for weaknesses",
            "",
        ])
    else:
        report_lines.extend([
            f"- [WARNING] Ensemble recall is below 100% ({ensemble_results['avg_recall']*100:.2f}%)",
            f"- [INFO] Investigation needed to understand why ensemble underperforms",
            "",
        ])

    report_lines.extend([
        "---",
        "",
        "## Model Contributions",
        "",
        "### Weight Distribution",
        "",
        f"Ensemble uses **equal weighting** strategy:",
        "",
    ])

    for model_name, weight in model_contributions.items():
        report_lines.append(f"- **{model_name}:** {weight:.3f} ({weight*100:.1f}%)")

    report_lines.extend([
        "",
        "### Analysis",
        "",
        f"With equal weighting, each model contributes 33.3% to the final prediction. ",
        f"This is the simplest ensemble strategy and works well when all models have ",
        f"similar performance.",
        "",
        f"**Future Optimizations (if needed):**",
        f"- Confidence-based weighting: Weight by model prediction confidence per event",
        f"- Position-aware weighting: Weight differently for LOW/MID/HIGH ranges (from Epic 8 analysis)",
        f"- Custom weights: Manually tune based on individual model strengths",
        "",
        "---",
        "",
        "## Next Steps",
        "",
    ])

    if recall_diff >= 0:
        report_lines.extend([
            "### [OK] Proceed to Story 9B.2: Bias Correction",
            "",
            "The ensemble has been validated and is ready for bias correction. Story 9B.2 will:",
            "",
            "1. Calculate bias correction factors from Epic 9A Density Matrix data",
            "2. Implement RangeAwareBiasCorrection class",
            "3. Apply correction to reduce ~300% bias to <150%",
            "4. Evaluate impact on Recall@20 and bias reduction",
            "",
        ])
    else:
        report_lines.extend([
            "### [WARNING] Decision Required: Ensemble vs Best Individual",
            "",
            f"The ensemble underperforms {best_individual['model_name']} by {abs(recall_diff)*100:.2f}%. ",
            "Before proceeding to Story 9B.2, consider:",
            "",
            f"1. **Option A:** Use {best_individual['model_name']} alone for bias correction",
            f"2. **Option B:** Investigate why ensemble underperforms (check model contributions, try confidence weighting)",
            f"3. **Option C:** Proceed with ensemble anyway (redundancy may be valuable even if slightly lower recall)",
            "",
        ])

    report_lines.extend([
        "---",
        "",
        "## Technical Details",
        "",
        f"- **Holdout Set:** 1,000 events (last 1,000 from imputed_density_matrix_full.parquet)",
        f"- **Evaluation Metric:** Recall@20 (% of 5 actual winning positions in top-20 predictions)",
        f"- **Ensemble Strategy:** Equal weighting (1/3 per model)",
        f"- **Models Combined:** LGBM, SetTransformer, GNN (all from Density Matrix imputation)",
        "",
        "---",
        "",
        f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Author:** BMad Dev Agent (James)",
        f"**Epic:** Epic 9B - Ensemble & Bias Correction",
        f"**Story:** 9B.1 - Simple Ensemble Implementation",
        "",
    ])

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"[OK] Report saved to {output_path}")


def save_evaluation_summary(individual_results: List[Dict],
                            ensemble_results: Dict,
                            output_path: str):
    """
    Save evaluation summary as JSON for programmatic access.

    Args:
        individual_results: List of dicts from evaluate_model()
        ensemble_results: Dict from evaluate_ensemble()
        output_path: Path to save JSON summary
    """
    summary = {
        'evaluation_date': datetime.now().isoformat(),
        'epic': 'Epic 9B - Ensemble & Bias Correction',
        'story': '9B.1 - Simple Ensemble Implementation',
        'holdout_size': 1000,
        'individual_models': [
            {
                'model_name': r['model_name'],
                'recall_at_20': r['avg_recall'],
                'perfect_recall_count': r['perfect_recall_count'],
                'perfect_recall_pct': r['perfect_recall_pct']
            }
            for r in individual_results
        ],
        'ensemble': {
            'model_name': ensemble_results['model_name'],
            'recall_at_20': ensemble_results['avg_recall'],
            'perfect_recall_count': ensemble_results['perfect_recall_count'],
            'perfect_recall_pct': ensemble_results['perfect_recall_pct'],
            'weighting_strategy': 'equal'
        }
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Summary saved to {output_path}")


def main():
    """
    Main execution function for Story 9B.1 evaluation.

    Workflow:
    ---------
    1. Load holdout data (1,000 events from Density Matrix imputation)
    2. Load 3 trained Density Matrix models (LGBM, SetTransformer, GNN)
    3. Evaluate each model individually on holdout
    4. Create SimpleEnsemble and evaluate on holdout
    5. Compare ensemble vs individual models
    6. Generate comprehensive evaluation report
    7. Save results for Story 9B.2 use
    """
    print("="*80)
    print("Story 9B.1: Simple Ensemble Evaluation")
    print("="*80)
    print()
    print("This script evaluates the SimpleEnsemble by combining predictions from")
    print("LGBM, SetTransformer, and GNN (Density Matrix imputation) and comparing")
    print("performance against individual models on the 1,000-event holdout set.")
    print()

    # Create output directory
    output_dir = Path("production/reports/epic9b_ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    # Step 1: Load holdout data
    data_path = "data/splits/density/density_holdout.csv"
    X_test, y_test = load_holdout_data(data_path)

    # Step 2: Load models
    models = load_density_matrix_models()

    # Step 3: Evaluate individual models
    print("\n" + "="*80)
    print("Evaluating Individual Models")
    print("="*80)

    individual_results = []
    for model_name, model in [('LGBM', models['lgbm']),
                               ('SetTransformer', models['settransformer']),
                               ('GNN', models['gnn'])]:
        result = evaluate_model(model, X_test, y_test, model_name)
        individual_results.append(result)

    # Step 4: Create and evaluate ensemble
    print("\n" + "="*80)
    print("Creating and Evaluating SimpleEnsemble")
    print("="*80)

    ensemble = SimpleEnsemble(
        models=[models['lgbm'], models['settransformer'], models['gnn']],
        weighting_strategy='equal'
    )

    print(f"\nEnsemble created with {ensemble.n_models_} models:")
    model_contributions = ensemble.get_model_contributions()
    for model_name, weight in model_contributions.items():
        print(f"  - {model_name}: {weight:.3f} ({weight*100:.1f}%)")

    ensemble_results = evaluate_ensemble(ensemble, X_test, y_test)

    # Step 5: Generate reports
    print("\n" + "="*80)
    print("Generating Evaluation Reports")
    print("="*80)

    report_path = output_dir / "simple_ensemble_evaluation.md"
    generate_evaluation_report(
        individual_results,
        ensemble_results,
        model_contributions,
        str(report_path)
    )

    summary_path = output_dir / "simple_ensemble_evaluation_summary.json"
    save_evaluation_summary(
        individual_results,
        ensemble_results,
        str(summary_path)
    )

    # Final summary
    print("\n" + "="*80)
    print("Story 9B.1 Evaluation Complete!")
    print("="*80)
    print()
    print("Results Summary:")
    print(f"  Individual Models:")
    for result in individual_results:
        print(f"    - {result['model_name']}: {result['avg_recall']*100:.2f}% Recall@20")
    print(f"  SimpleEnsemble: {ensemble_results['avg_recall']*100:.2f}% Recall@20")
    print()
    print(f"Reports saved to:")
    print(f"  - {report_path}")
    print(f"  - {summary_path}")
    print()
    print("Next step: Review evaluation report and proceed to Story 9B.2 (Bias Correction)")
    print()


if __name__ == '__main__':
    main()

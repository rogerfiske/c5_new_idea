# Quantum State Prediction Experiment Fullstack Architecture Document - v1.2

## 1. Introduction
This document outlines the complete fullstack architecture for the Quantum State Prediction Experiment. It covers the data processing backend, the modeling pipeline, and the "frontend" which consists of the generated reports and visualizations. This unified architecture ensures a consistent and reproducible experimental workflow. The project will be built from scratch (Greenfield) using a standard Python data science stack.

### Change Log
| Date | Version | Description | Author |
| --- | --- | --- | --- |
| 2025-10-10 | 1.0 | Initial architecture creation from PRD V1.0. | BMad Architect |
| 2025-10-10 | 1.1 | Updated Tech Stack and Project Structure to support specific imputation and model families. | BMad Architect |
| 2025-10-12 | 1.2 | Added experiments/ and production/ directory structures to support 5 sequential imputation method experiments (PRD Epic 5) and final production run (PRD Epic 7). Added experiment tracking framework documentation. Fixed naming: amplitude_superposition → amplitude_embedding. | Sarah (PO) |

---

## 2. High Level Architecture

### Technical Summary
The system is designed as a modular, sequential data processing pipeline in Python. Data flows from raw CSVs through a **pluggable collection of Quantum-Inspired Imputation modules**. The resulting feature sets are used to train a **suite of distinct ranking models** (Baselines, GBDTs, Deep Learning). An evaluation engine assesses performance using a strict holdout protocol and custom metrics. An ensembling layer can combine the outputs of individual rankers. The architecture is built for rapid experimentation and iteration.

### High Level Architecture Diagram
```mermaid
graph TD
    A[Data Ingestion<br>(c5_Matrix.csv)] --> B[Preprocessing];
    subgraph Imputation Framework
        direction LR
        C1[Basis Embedding]
        C2[Amplitude Superposition]
        C3[Angle Encoding]
        C4[...]
    end
    B --> C1 & C2 & C3 & C4 --> D[Feature Engineered Data];
    D --> E[Data Splitting<br>(Train / Holdout)];
    subgraph Ranker Zoo
        direction LR
        F1[Frequency Baselines]
        F2[LightGBM / XGBoost]
        F3[Set Transformer]
        F4[GNN / DFT]
    end
    E --> F1 & F2 & F3 & F4 --> G[Trained Model Artifacts];
    subgraph Ensembler
        G --> G_ens
    end
    subgraph Holdout Evaluation
        direction LR
        H[Holdout Data] --> I[Prediction Engine];
        G_ens --> I;
        I --> J[Generate 20 Likely States];
        K[Holdout Actuals] --> L[Evaluation Engine];
        J --> L;
    end
    L --> M[Detailed Metrics Log];
    L --> N[Holdout Test Summary Report];

## 3. Tech Stack

| Category | Technology | Version | Purpose | Rationale |
| --- | --- | --- | --- | --- |
| Language | Python | 3.11+ | Core programming language | Standard for data science; rich library ecosystem. |
| Environment | Conda / venv | Latest | Environment & package management | Ensures reproducibility and avoids dependency conflicts. |
| Data Manipulation | Pandas | 2.2+ | Data loading, cleaning, and manipulation | Industry standard for tabular data in Python. |
| Numerical Computing | NumPy | 1.26+ | Core numerical operations | Foundation for scientific computing and ML libraries. |
| ML (Baseline) | Scikit-learn | 1.4+ | Baseline models and evaluation metrics | Comprehensive, easy-to-use library for classical ML. |
| Gradient Boosting | LightGBM / XGBoost | Latest | High-performance GBDT ranking models | State-of-the-art for tabular data; fast and accurate. |
| Deep Learning | PyTorch / TensorFlow | Latest | Set Transformer & GNN models | Industry-leading frameworks for custom architectures. |
| Graph ML | PyTorch Geometric | Latest | Implementing GNNs on the C₃₉ ring | Simplifies the creation of graph neural networks in PyTorch. |
| Visualization | Matplotlib / Seaborn | Latest | EDA and results visualization | Standard plotting libraries for insightful graphics. |
| Experimentation | Jupyter Notebook | Latest | Interactive development and EDA | Ideal for iterative, exploratory work and reporting. |
| Testing | pytest | Latest | Unit and integration testing | Industry-standard testing framework for Python. |
| Model Storage | Joblib / Pickle | Latest | Saving and loading trained models | Simple and effective for serializing Python objects. |

---

## 4. Unified Project Structure
A clear, organized directory structure is crucial for managing this experimental project.

```
quantum_prediction_project/
├── .gitignore
├── environment.yml
├── README.md
├── CONTRIBUTING.md
│
├── data/
│   ├── raw/                        # Original c5_Matrix.csv
│   └── processed/                  # Cleaned and validated data
│
├── experiments/                    # Epic 5: 5 Imputation Method Experiments
│   ├── exp01_basis_embedding/
│   │   ├── data/                   # Imputed data for this experiment
│   │   ├── models/                 # Trained models for this experiment
│   │   ├── reports/                # Holdout test summary & metrics
│   │   └── logs/                   # Experiment execution logs
│   ├── exp02_amplitude_embedding/
│   │   ├── data/
│   │   ├── models/
│   │   ├── reports/
│   │   └── logs/
│   ├── exp03_angle_encoding/
│   │   ├── data/
│   │   ├── models/
│   │   ├── reports/
│   │   └── logs/
│   ├── exp04_density_matrix/
│   │   ├── data/
│   │   ├── models/
│   │   ├── reports/
│   │   └── logs/
│   └── exp05_graph_cycle/
│       ├── data/
│       ├── models/
│       ├── reports/
│       └── logs/
│
├── production/                     # Epic 7: Final Production Run
│   ├── data/
│   │   └── imputed_best_method.parquet
│   ├── models/
│   │   ├── best_ranker_v1.joblib
│   │   └── ensemble_config.json
│   ├── predictions/
│   │   └── final_prediction_next_event.json
│   └── reports/
│       └── final_production_report.md
│
├── notebooks/
│   ├── 1_EDA.ipynb
│   ├── 2_Imputation_Dev.ipynb
│   └── 3_Results_Analysis.ipynb
│
├── reports/                        # Cross-cutting analysis reports
│   ├── figures/
│   ├── imputation_comparison_analysis.ipynb  # Epic 5.7
│   └── holdout_summary_v1.txt
│
├── models/                         # Development/testing models (non-experiment)
│   ├── lgbm_ranker_v1.joblib
│   └── set_transformer_v1.pth
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   └── __init__.py
│
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── evaluation.py
    ├── main.py
    │
    ├── imputation/                 # MODULE FOR IMPUTATION (All 5 methods)
    │   ├── __init__.py
    │   ├── base_imputer.py
    │   ├── basis_embedding.py
    │   ├── amplitude_embedding.py
    │   ├── angle_encoding.py
    │   ├── density_matrix.py
    │   └── graph_cycle_encoding.py
    │
    └── modeling/                   # MODULE FOR MODELING
        ├── __init__.py
        ├── rankers/                # Sub-module for individual models
        │   ├── __init__.py
        │   ├── frequency_ranker.py
        │   ├── lgbm_ranker.py
        │   ├── set_transformer_model.py
        │   └── gnn_ranker.py
        ├── ensembles/              # Sub-module for ensembling
        │   ├── __init__.py
        │   ├── rrf.py
        │   └── weighted_average.py
        └── pipeline.py             # Script for training/prediction orchestration
```

---

## 4.1. Experiment Tracking Framework

### Purpose
The experiment tracking framework ensures reproducibility, traceability, and fair comparison across all 5 imputation method experiments (PRD Epic 5). Each experiment must be completely segregated to enable:
1. **Reproducibility**: Any experiment can be re-run independently
2. **Traceability**: Full provenance from raw data → imputation → model → prediction
3. **Fair Comparison**: All experiments use identical ranker models and evaluation criteria
4. **Best Method Selection**: Clear decision path from experiments → final production run

### Experiment Structure
Each experiment follows this standardized structure:
```
experiments/{experiment_id}/
├── data/
│   ├── imputed_data.parquet          # Output from imputation method
│   └── data_stats.json               # Imputation metadata & statistics
├── models/
│   ├── frequency_baseline.joblib     # All 4 ranker types from Epic 3
│   ├── lgbm_ranker.joblib
│   ├── set_transformer.pth
│   └── gnn_ranker.pth
├── reports/
│   ├── holdout_test_summary.txt      # Standard format from PRD Epic 4
│   ├── detailed_metrics.json         # Prediction probabilities, feature importance
│   └── model_comparisons.csv         # Per-ranker performance for this imputation
└── logs/
    ├── imputation.log                # Imputation execution log
    ├── training.log                  # Model training logs
    └── evaluation.log                # Holdout evaluation logs
```

### Experiment Metadata Standard
Each experiment must log the following metadata in `experiments/{experiment_id}/experiment_metadata.json`:
```json
{
  "experiment_id": "exp01_basis_embedding",
  "imputation_method": "basis_embedding",
  "execution_timestamp": "2025-10-12T14:30:00Z",
  "prd_version": "1.3",
  "architecture_version": "1.2",
  "data_source": {
    "raw_file": "data/raw/c5_Matrix.csv",
    "raw_file_hash": "sha256:abc123...",
    "rows": 5000,
    "columns": 45
  },
  "imputation_config": {
    "method": "basis_embedding",
    "parameters": {...},
    "output_dimensions": 1024
  },
  "rankers_trained": ["frequency_baseline", "lgbm_ranker", "set_transformer", "gnn_ranker"],
  "holdout_split": {
    "train_size": 4000,
    "holdout_size": 1000,
    "split_method": "sequential"
  },
  "execution_time": {
    "imputation_seconds": 120,
    "training_seconds": 3600,
    "evaluation_seconds": 300,
    "total_seconds": 4020
  },
  "status": "completed",
  "errors": []
}
```

### Comparative Analysis (Epic 5.7)
The comparative analysis in Story 5.7 will:
1. Load `experiment_metadata.json` from all 5 experiments
2. Load `reports/holdout_test_summary.txt` and `reports/detailed_metrics.json` from each
3. Generate comparative visualizations saved to `reports/imputation_comparison_analysis.ipynb`
4. Create decision matrix: `reports/method_selection_matrix.csv`
   - Rows: 5 imputation methods
   - Columns: Performance metrics (Excellent %, Good %, Average score per ranker, consistency, etc.)
5. Output formal recommendation: `reports/best_method_recommendation.md`

### Production Run Provenance (Epic 7)
The production run must reference the winning experiment:
```json
{
  "production_run_id": "prod_v1_final_prediction",
  "based_on_experiment": "exp02_amplitude_embedding",
  "selection_rationale": "Highest Excellent prediction rate (45%) and most consistent across rankers",
  "imputation_method": "amplitude_embedding",
  "best_ranker": "lgbm_ranker",
  "ensemble_used": false,
  "final_prediction_file": "production/predictions/final_prediction_next_event.json",
  "timestamp": "2025-10-15T10:00:00Z"
}
```

---

## 5. Development Workflow

### Local Development Setup
1. Clone the repository.
2. Create the Conda environment: `conda env create -f environment.yml`
3. Activate the environment: `conda activate quantum_project`
4. Launch Jupyter Lab: `jupyter lab`

### RunPod Workflow (for long-running tasks)
Training of the Set Transformer and GNN models are prime candidates for offloading to a RunPod instance due to their computational intensity. The following process will be used:

1. A dedicated Jupyter notebook will be created (e.g., `notebooks/runpod_training_task.ipynb`).
2. This notebook will contain all necessary code, imports, and data loading steps to be self-contained.
3. A `.zip` file will be prepared containing:
   - The notebook (`runpod_training_task.ipynb`).
   - The required processed data file (e.g., `imputed_dataset_v1.parquet`).
   - A `requirements.txt` file for pip.
   - A `INSTRUCTIONS.md` file detailing the exact steps to run on the RunPod instance.
4. The user will upload this `.zip` to their RunPod instance, execute the notebook, and download the resulting artifacts (e.g., the trained model file).

---

## 6. Coding Standards

- **Clarity is Key**: As the primary user is a non-programmer, all code must be exceptionally well-commented. Explain the "why" behind code blocks, not just the "what".
- **Modularity**: Functions and classes should be small and do one thing well.
- **Type Hinting**: Use Python type hints for all function signatures to improve readability and allow for static analysis.
- **Configuration**: Avoid hard-coding values. Use a central `config.py` file or pass parameters to scripts for things like file paths, model parameters, etc.
- **Logging**: Use Python's built-in logging module for output instead of `print()`. This allows for different levels of verbosity (DEBUG, INFO, ERROR).
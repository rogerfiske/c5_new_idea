# Quantum State Prediction Experiment

A data science project exploring quantum-inspired imputation methods for predicting the next event in a 39-dimensional cyclic binary state space.

## Project Overview

This project implements and compares 5 different quantum-inspired imputation methods to predict the next likely quantum state in a C₃₉ cyclic group. The system trains 4 different ranking model families on each imputed dataset, evaluates performance on a strict holdout set, and selects the best method for final production predictions.

### Imputation Methods
1. **Basis Embedding** - Direct mapping to computational basis states
2. **Amplitude Embedding** - Superposition over active positions
3. **Angle Encoding** - Rotation-based quantum encoding
4. **Density Matrix Embedding** - Mixed state representations
5. **Graph/Cycle Encoding** - Cyclic graph structure modeling

### Ranking Models
1. **Frequency Baselines** - Simple statistical baselines
2. **GBDT Models** - LightGBM/XGBoost rankers
3. **Set Transformer** - Deep learning attention-based model
4. **Graph Neural Network** - GNN on C₃₉ ring structure

## Project Structure

```
quantum_prediction_project/
├── data/                      # Data storage
│   ├── raw/                   # Original c5_Matrix.csv (USER ACTION REQUIRED)
│   └── processed/             # Cleaned data
├── experiments/               # 5 Imputation experiments (Epic 5)
│   ├── exp01_basis_embedding/
│   ├── exp02_amplitude_embedding/
│   ├── exp03_angle_encoding/
│   ├── exp04_density_matrix/
│   └── exp05_graph_cycle/
├── production/                # Final production run (Epic 7)
│   ├── data/
│   ├── models/
│   ├── predictions/
│   └── reports/
├── notebooks/                 # Jupyter notebooks for EDA
├── reports/                   # Analysis reports and figures
├── models/                    # Development models
├── tests/                     # Unit and integration tests
└── src/                       # Source code
    ├── imputation/            # 5 imputation methods
    └── modeling/              # Ranking models and pipelines
```

## Setup Instructions

### Prerequisites
- Python 3.11 or higher
- Conda or venv for environment management
- **USER ACTION REQUIRED**: c5_Matrix.csv dataset (see Data Setup below)

### Installation Steps

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd c5_new-idea
   ```

2. **Create and activate the Conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate quantum_project
   ```

   **OR** using venv:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Data Setup (USER ACTION REQUIRED)**:
   - Locate your `c5_Matrix.csv` file
   - Copy it to the `data/raw/` directory:
     ```bash
     # Windows example:
     copy "path\to\your\c5_Matrix.csv" "data\raw\c5_Matrix.csv"

     # Linux/Mac example:
     cp /path/to/your/c5_Matrix.csv data/raw/c5_Matrix.csv
     ```
   - Verify the file exists:
     ```bash
     # Windows:
     dir data\raw\c5_Matrix.csv

     # Linux/Mac:
     ls -lh data/raw/c5_Matrix.csv
     ```

4. **Verify installation**:
   ```bash
   python -c "import pandas, numpy, sklearn, lightgbm, torch; print('All dependencies installed successfully')"
   ```

5. **Run tests** (optional, after test suite is created):
   ```bash
   pytest tests/
   ```

## Usage

### Running Exploratory Data Analysis
```bash
jupyter lab
# Open notebooks/1_EDA.ipynb
```

### Running Experiments
```bash
# Run individual imputation experiment
python src/main.py --experiment exp01_basis_embedding

# Run all 5 experiments sequentially
python src/main.py --run-all-experiments
```

### Viewing Results
- Experiment reports: `experiments/{experiment_id}/reports/holdout_test_summary.txt`
- Comparative analysis: `reports/imputation_comparison_analysis.ipynb`
- Final production prediction: `production/predictions/final_prediction_next_event.json`

## Development Workflow

1. **Make changes** in `src/` or `notebooks/`
2. **Run tests**: `pytest tests/`
3. **Commit changes** following guidelines in CONTRIBUTING.md
4. **Document** updates in relevant reports

## Key Configuration

All file paths are managed centrally in `src/config.py`. Modify this file if you need to change directory locations.

## Documentation

- **Product Requirements**: `docs/prd.md` (v1.3)
- **Architecture**: `docs/architecture.md` (v1.2)
- **Quantum Imputation Methods**: `docs/Assigning Quantum States to Binary csv.md`
- **Contributing Guidelines**: `CONTRIBUTING.md`

## Important Notes for Non-Programmers

- **All code is extensively commented** to explain what each part does
- **Step-by-step instructions** are provided for every operation
- **USER ACTION REQUIRED** items are clearly marked throughout
- **If you encounter errors**, check logs in `experiments/*/logs/` or contact the development team
- **RunPod usage**: For GPU-intensive training (Set Transformer, GNN), detailed RunPod packaging and instructions will be provided when needed

## Project Status

Current Epic: **Epic 1 - Project Scaffolding** ✅ (In Progress)

### Epic Sequence
1. ✅ Epic 1: Project Scaffolding & Setup
2. ⏳ Epic 2: Implement 5 Imputation Methods
3. ⏳ Epic 3: Implement 4 Ranking Models
4. ⏳ Epic 4: Evaluation Engine
5. ⏳ Epic 5: Run 5 Sequential Experiments
6. ⏳ Epic 6: Ensembling (Optional)
7. ⏳ Epic 7: Final Production Prediction Run

## License

[Specify License Here]

## Contact

[Specify Contact Information]

---

**Last Updated**: 2025-10-12
**Project Version**: 1.0
**PRD Version**: 1.3
**Architecture Version**: 1.2

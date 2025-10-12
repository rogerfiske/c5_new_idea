# Quantum State Prediction Experiment Product Requirements Document (PRD) - v1.3

## 1. Goals and Background Context

### Goals
- **G1**: To develop a flexible framework for experimenting with various **quantum-inspired imputation strategies** (Basis Embedding, Amplitude Embedding, Angle Encoding, Density Matrix Embedding, Graph/Cycle Encoding) on a binary matrix dataset.
- **G2**: To build and evaluate predictive **ranking models** that forecast the '20 most likely' quantum states/values for a subsequent event.
- **G3**: To establish a robust evaluation pipeline using holdout testing and a custom success metric that values extremely accurate or extremely inaccurate predictions.
- **G4**: To create a highly detailed data collection and logging system to enable deep analysis of model performance and guide iterative improvements.

### Background Context
This project is an experimental endeavor to explore the intersection of quantum concepts and predictive modeling on a sequential binary dataset. The project will create various **feature maps based on quantum-inspired classical simulations** (not actual quantum computation) to feed into downstream classical machine learning rankers. The primary challenge is to engineer these features and evaluate their effectiveness in predicting future states. The unique success criteria (rewarding 0 or 5 incorrect predictions out of 5) emphasize the goal of achieving high certainty.

### Change Log
| Date | Version | Description | Author |
| --- | --- | --- | --- |
| 2025-10-10 | 1.0 | Initial PRD creation from user requirements. | BMad PM |
| 2025-10-10 | 1.1 | Incorporated specific imputation strategies and model families into Epics 2, 3, and 6. | BMad PM |
| 2025-10-10 | 1.2 | PO validation fixes: Added Story 1.4 (testing infrastructure), git initialization to Story 1.1, dataset initialization instructions, comprehensive error handling, explicit documentation requirements (NFR2), Epic 5 dependency clarification, performance tracking in Story 5.3. | Sarah (PO) |
| 2025-10-12 | 1.3 | Major restructuring for 5 Assignment/Imputation method experiments: (1) Added all 5 methods to Epic 2 (including missing Density Matrix Embedding), (2) Fixed naming consistency (Amplitude Embedding), (3) Restructured Epic 5 to run 5 sequential sub-experiments (one per imputation method) with comparative analysis, (4) Added Epic 7 for final production run using best method to achieve macro objective. | Sarah (PO) |

---

## 2. Requirements

### Functional
- **FR1**: The system must load and validate the `c5_Matrix.csv` dataset according to the rules in `C-5 Binary Matrix Dataset Description.md`.
- **FR2**: The system must provide a modular interface to apply different "quantum imputation" algorithms to the raw binary data.
- **FR3**: The system must be able to train machine learning models (including ensembles) on the imputed data.
- **FR4**: The models must predict a ranked list of the 20 most likely outcomes for the next event in the sequence.
- **FR5**: The system must implement a holdout testing function that strictly isolates training, testing, and validation data to prevent leakage.
- **FR6**: The system must calculate the custom success metric (0/5 wrong = Excellent, 1/4 wrong = Good, 2/3 wrong = Poor) for each prediction event.
- **FR7**: The system must log detailed metrics during holdout testing, including feature importance, prediction probabilities, and intermediate data states for post-analysis.
- **FR8**: The system must generate a final `HOLDOUT TEST SUMMARY` in the user-specified format.

### Non-Functional
- **NFR1**: All data processing and model training pipelines must be designed to run within Jupyter notebooks for easy transfer to RunPod if a task is estimated to exceed 1 hour of processing on the user's local PC.
- **NFR2**: The codebase must be heavily commented and documented to be understandable by a non-programmer.
- **NFR3**: The experimental framework must be modular, allowing for new imputation methods and models to be added with minimal changes to the core pipeline.
- **NFR4**: All outputs (data, models, reports) must be saved to a structured directory system for clarity and reproducibility.

---

## 3. Epic and Story Structure

### Epic 1: Project Scaffolding, Data Ingestion, and EDA
* **Goal**: Establish the project environment, load the dataset, perform initial validation, and conduct exploratory data analysis (EDA) to understand its characteristics.
* **Stories**:
    * **1.1**: Set up the Python environment, install base libraries (Pandas, NumPy, Matplotlib, Scikit-learn, Jupyter, pytest), and create the project directory structure.
        * Clone the repository: `git clone https://github.com/rogerfiske/c5_new_idea`
        * Create the Conda environment using Python 3.11+ (see architecture.md section 3 for specific library versions)
        * Install dependencies: Pandas 2.2+, NumPy 1.26+, Matplotlib (latest), Scikit-learn 1.4+, Jupyter (latest), pytest (latest)
        * Create project directory structure as specified in architecture.md section 4
        * Create .gitignore file with Python-specific ignores: `__pycache__/`, `.ipynb_checkpoints/`, `*.pyc`, `*.pyo`, `*.egg-info/`, `.pytest_cache/`, `data/`, `models/`, `.env`, `.venv/`, `venv/`
        * Create initial README.md with setup instructions
        * Create CONTRIBUTING.md documenting coding standards from architecture.md section 6
        * Create src/config.py with project path constants (DATA_RAW, DATA_PROCESSED, EXPERIMENTS_DIR, PRODUCTION_DIR, MODELS_DIR, REPORTS_DIR, DATASET_FILENAME) - see architecture.md section 6 for centralized path management approach
        * Make initial commit and push to remote: `git add . && git commit -m "Initial project scaffolding" && git push`
        * **Acceptance Criteria**: Environment activates successfully, all libraries import without errors, directory structure matches architecture doc, config.py contains all path constants, repository has initial commit
    * **1.2**: Implement a script to load `c5_Matrix.csv`, validate its shape and data types against the description file, and handle any initial cleaning.
        * **USER ACTION REQUIRED**: Before running this story, place the `c5_Matrix.csv` file in the `data/raw/` directory
        * Implement file existence check with clear error message if dataset is missing
        * Validate CSV shape matches expected dimensions from description file
        * Validate data types (binary values, proper formatting)
        * Handle common data issues: missing values, extra whitespace, encoding problems
        * Log validation results and any cleaning operations performed
        * Save cleaned data to `data/processed/` if modifications were needed
        * **Error Handling**: Provide user-friendly error messages for: file not found, incorrect format, invalid data types, dimension mismatches
        * **Acceptance Criteria**: Script successfully loads valid CSV, detects and reports format errors clearly, handles edge cases gracefully, all validation logic is well-commented (per NFR2)
    * **1.3**: Develop a Jupyter notebook to perform EDA, visualizing data patterns, distributions, and any immediately obvious sequential characteristics.
        * Analyze data distributions, check for patterns in binary sequences
        * Visualize temporal patterns and potential correlations
        * Document initial observations and hypotheses for imputation strategies
        * Include markdown cells explaining each analysis step (per NFR2)
        * **Acceptance Criteria**: Notebook runs end-to-end without errors, visualizations are clear and labeled, findings are documented in markdown cells
    * **1.4**: Set up testing infrastructure and create initial validation tests.
        * Create `tests/` directory structure: `tests/unit/`, `tests/integration/`, `tests/fixtures/`
        * Configure pytest with pytest.ini or pyproject.toml
        * Create test fixtures for sample data (small valid CSV, invalid CSV examples)
        * Write unit tests for data loading and validation (Story 1.2 functionality)
        * Write tests for edge cases: empty files, malformed CSVs, wrong dimensions
        * Document testing approach in README.md
        * **Acceptance Criteria**: pytest runs successfully, basic tests pass, test structure supports future test additions, tests are documented (per NFR2)

### Epic 2: Quantum-Inspired Imputation Framework
* **Goal**: Build a flexible framework for applying quantum-inspired imputation methods and implement the initial set of strategies.
* **Stories**:
    * **2.1**: Design and implement a base Python class for imputation strategies that takes raw data and returns a feature-engineered representation.
        * Define abstract base class with standard interface: `fit()`, `transform()`, `fit_transform()`
        * Include input validation and type checking in base class
        * Document the interface contract with detailed docstrings
        * Include example usage in docstring
        * **Acceptance Criteria**: Base class is well-documented with docstrings (per NFR2), provides clear interface for subclasses, includes input validation
    * **2.2**: Implement the **Basis Embedding** strategy.
        * Inherit from base imputation class (Story 2.1)
        * Implement strategy-specific logic with detailed inline comments
        * Include docstring explaining the quantum-inspired approach and parameters
        * Write unit tests for this strategy (add to tests/ structure from Story 1.4)
        * **Acceptance Criteria**: Strategy produces valid output, code is heavily commented (per NFR2), tests pass
    * **2.3**: Implement the **Amplitude Embedding** strategy (superposition over active positions).
        * Inherit from base imputation class, follow same documentation standards as Story 2.2
        * Write unit tests for this strategy
        * **Acceptance Criteria**: Strategy produces valid output, code is heavily commented (per NFR2), tests pass
    * **2.4**: Implement the **Angle/Rotation Encoding** strategy.
        * Inherit from base imputation class, follow same documentation standards as Story 2.2
        * Write unit tests for this strategy
        * **Acceptance Criteria**: Strategy produces valid output, code is heavily commented (per NFR2), tests pass
    * **2.5**: Implement the **Density Matrix Embedding** strategy (for mixed states).
        * Inherit from base imputation class, follow same documentation standards as Story 2.2
        * Implement mixed state representation using density matrices for probabilistic quantum states
        * Write unit tests for this strategy
        * **Acceptance Criteria**: Strategy produces valid output, code is heavily commented (per NFR2), tests pass
    * **2.6**: Implement the **Graph/Cycle Encoding** (circular convolution / DFT) strategy. Include ring features (circular distances & DFT harmonics).
        * Inherit from base imputation class, follow same documentation standards as Story 2.2
        * Write unit tests for this strategy
        * **Acceptance Criteria**: Strategy produces valid output, code is heavily commented (per NFR2), tests pass
    * **2.7**: Implement a script to apply any chosen imputation strategy to the raw dataset and save the output.
        * Accept strategy name as parameter (command-line or function argument)
        * Load raw data from `data/raw/`, apply selected imputation strategy
        * Validate imputation output (check for NaN, inf, expected dimensions)
        * Save imputed data to `data/processed/` with descriptive filename (e.g., `imputed_basis_embedding_v1.parquet`)
        * Log imputation metadata: timestamp, strategy used, parameters, output statistics
        * **Error Handling**: Handle invalid strategy names, data loading failures, imputation errors (NaN/inf in output), file save failures
        * **Acceptance Criteria**: Script handles all imputation strategies, provides clear error messages, validates outputs, code is well-commented (per NFR2)

### Epic 3: Individual Ranker Implementation
* **Goal**: Develop a pipeline for training and evaluating individual ranking models on the imputed data.
* **Stories**:
    * **3.1**: Create the data preparation script for modeling, including splitting data into training and a strict sequential holdout set.
        * Load imputed data from `data/processed/`
        * Implement strict sequential split (no data leakage): train on earlier events, holdout on later events
        * Validate split maintains temporal order and has no overlap
        * Save split datasets with clear naming: `train_split.parquet`, `holdout_split.parquet`
        * Log split statistics: train size, holdout size, date ranges if applicable
        * **Error Handling**: Handle missing imputed data files, invalid split ratios, data validation failures
        * **Acceptance Criteria**: Split maintains temporal integrity, no data leakage, code is well-documented (per NFR2)
    * **3.2**: Implement **Frequency-Based Baseline Rankers** (Cumulative, EMA, Bigram/co-occurrence).
        * Create ranker class(es) following consistent interface pattern
        * Include detailed docstrings explaining each baseline approach
        * Write unit tests for baseline rankers
        * **Acceptance Criteria**: Rankers produce valid ranked predictions, code is heavily commented (per NFR2), tests pass
    * **3.3**: Implement a **Gradient Boosting Ranker** using LightGBM or XGBoost over engineered features (counts, distances, harmonics, etc.). Include ring features (circular distances & DFT harmonics).
        * Create ranker class with hyperparameter configuration support
        * Include detailed docstrings and inline comments explaining feature engineering
        * Write unit tests for GBDT ranker
        * **Acceptance Criteria**: Ranker trains successfully, produces valid predictions, code is heavily commented (per NFR2), tests pass
    * **3.4**: Implement a **Set-Based Ranker** using a DeepSets or light Set Transformer architecture.
        * Create ranker class following consistent interface
        * Document architecture choices and hyperparameters in docstrings
        * Write unit tests for set-based ranker
        * **Error Handling**: Handle training divergence, invalid input shapes, GPU/CPU compatibility issues
        * **Acceptance Criteria**: Model trains successfully, produces valid predictions, code is heavily commented (per NFR2), tests pass
    * **3.5**: Implement a **Graph-Based Ranker** using a simple GNN or DFT over the ring C₃₉.
        * Create ranker class following consistent interface
        * Document graph construction and GNN architecture in docstrings
        * Write unit tests for graph-based ranker
        * **Error Handling**: Handle training divergence, invalid graph construction, dependency issues (PyTorch Geometric)
        * **Acceptance Criteria**: Model trains successfully, produces valid predictions, code is heavily commented (per NFR2), tests pass
    * **3.6**: Create a unified training script that can train any selected ranker model on a given imputed dataset and save the artifact.
        * Accept parameters: ranker type, imputed data path, hyperparameters, output path
        * Load data, instantiate selected ranker, train model, save artifact to `models/`
        * Log training metadata: ranker type, hyperparameters, training time, final metrics
        * Validate model artifact after saving (can be loaded successfully)
        * Include progress indicators for long-running training
        * **Error Handling**: Handle invalid ranker types, training failures, convergence issues, model save failures, insufficient memory
        * **Acceptance Criteria**: Script supports all ranker types, handles errors gracefully with clear messages, logs training progress, code is well-documented (per NFR2)

### Epic 4: Evaluation and Reporting Module
* **Goal**: Implement the formal holdout test, the custom success metric, and the final reporting format.
* **Stories**:
    * **4.1**: Refine the data splitting script to implement the strict, sequential holdout test protocol.
        * Enhance Story 3.1's split logic with additional validation for holdout testing
        * Ensure zero data leakage between train and holdout sets
        * Implement checks to verify temporal ordering
        * Document the holdout protocol clearly in docstrings
        * **Error Handling**: Detect and report any potential data leakage, invalid temporal splits
        * **Acceptance Criteria**: Holdout protocol is rigorously enforced, validation catches leakage, code is well-documented (per NFR2)
    * **4.2**: Implement the function to score predictions against actuals using the custom "wrong predictions" metric.
        * Implement metric: 0/5 wrong = Excellent, 1-2 wrong = Good, 3-4 wrong = Poor, 5/5 wrong = Excellent
        * Accept predicted top-20 rankings and actual outcomes
        * Return detailed scoring breakdown
        * Include docstrings explaining the metric rationale
        * Write unit tests with known inputs and expected outputs
        * **Error Handling**: Handle mismatched dimensions, missing values, invalid prediction formats
        * **Acceptance Criteria**: Metric calculates correctly, handles edge cases, code is well-documented (per NFR2), tests pass
    * **4.3**: Implement the detailed metrics collection system to log performance data during the holdout test.
        * Log prediction probabilities, feature importance (if available), intermediate states
        * Save detailed metrics to structured format (JSON or CSV) in `reports/`
        * Include timestamps, model identifiers, hyperparameters in logs
        * Document the logging schema clearly
        * **Error Handling**: Handle logging failures gracefully (don't crash evaluation), manage disk space issues
        * **Acceptance Criteria**: Comprehensive metrics logged, logs are structured and parseable, code is well-documented (per NFR2)
    * **4.4**: Create the script that generates the final `HOLDOUT TEST SUMMARY` text file in the specified format.
        * Load detailed metrics from Story 4.3
        * Generate formatted summary report matching user-specified format
        * Save to `reports/holdout_summary_v{n}.txt`
        * Include all required sections: model info, metrics, key findings
        * **Error Handling**: Handle missing metrics data, formatting errors, file write failures
        * **Acceptance Criteria**: Report matches specified format exactly, includes all required information, code is well-documented (per NFR2)

### Epic 5: Imputation Method Comparison Experiments (5 Sequential Sub-Experiments)
* **Goal**: Run 5 complete end-to-end experiments (one per imputation method) to determine which Assignment/Imputation method produces the best prediction results for the macro objective of "Predicting the 20 most likely values" for the next event.
* **Dependencies**: Requires completion of Epics 1-4 (setup, imputation framework, ranker models, evaluation system)
* **Critical Context**: Each of the 5 imputation methods must be tested independently with the SAME ranker models and evaluation criteria to enable fair comparison. Results must be segregated by experiment for traceability.
* **Stories**:
    * **5.1**: Create a master experiment orchestration script/notebook that runs: data ingestion → imputation (parameterized) → model training → holdout prediction → evaluation → reporting.
        * Integrate all modules from Epics 1-4 into cohesive workflow
        * Accept parameters: imputation_method (1-5), ranker_types (list), hyperparameters, experiment_id
        * Save all outputs to segregated directories: `experiments/{experiment_id}/`
        * Include checkpointing to resume interrupted experiments
        * Log progress at each pipeline stage with experiment context
        * Generate experiment summary with configuration, timing, and results
        * **Experiment Output Structure**: `experiments/{experiment_id}/{models/, data/, reports/, logs/}`
        * **Error Handling**: Handle failures gracefully, log errors to experiment-specific error log
        * **Acceptance Criteria**: Pipeline runs end-to-end successfully for any imputation method, handles errors, outputs are properly segregated, code is well-documented (per NFR2)
    * **5.2**: Run Experiment 1 - Basis Embedding (Direct Mapping to Computational Basis States).
        * Execute Story 5.1's pipeline with `imputation_method="basis_embedding"`, `experiment_id="exp01_basis_embedding"`
        * Train ALL ranker types from Epic 3 (Frequency Baselines, GBDT, Set Transformer, GNN)
        * Document execution time for each stage (for RunPod decision-making per NFR1)
        * Validate all outputs generated: imputed data, trained models, holdout predictions, evaluation reports
        * **Acceptance Criteria**: Experiment completes successfully, all artifacts in `experiments/exp01_basis_embedding/`, holdout test summary generated
    * **5.3**: Run Experiment 2 - Amplitude Embedding (Superposition Over Active Positions).
        * Execute Story 5.1's pipeline with `imputation_method="amplitude_embedding"`, `experiment_id="exp02_amplitude_embedding"`
        * Train ALL ranker types from Epic 3 (same as Experiment 1 for fair comparison)
        * Document execution time for each stage
        * Validate all outputs generated
        * **Acceptance Criteria**: Experiment completes successfully, all artifacts in `experiments/exp02_amplitude_embedding/`, holdout test summary generated
    * **5.4**: Run Experiment 3 - Angle Encoding (Rotation-Based Encoding).
        * Execute Story 5.1's pipeline with `imputation_method="angle_encoding"`, `experiment_id="exp03_angle_encoding"`
        * Train ALL ranker types from Epic 3 (same as Experiments 1-2 for fair comparison)
        * Document execution time for each stage
        * Validate all outputs generated
        * **Acceptance Criteria**: Experiment completes successfully, all artifacts in `experiments/exp03_angle_encoding/`, holdout test summary generated
    * **5.5**: Run Experiment 4 - Density Matrix Embedding (For Mixed States).
        * Execute Story 5.1's pipeline with `imputation_method="density_matrix"`, `experiment_id="exp04_density_matrix"`
        * Train ALL ranker types from Epic 3 (same as Experiments 1-3 for fair comparison)
        * Document execution time for each stage
        * Validate all outputs generated
        * **Acceptance Criteria**: Experiment completes successfully, all artifacts in `experiments/exp04_density_matrix/`, holdout test summary generated
    * **5.6**: Run Experiment 5 - Graph/Cycle Encoding (Circular Convolution / DFT with Ring Features).
        * Execute Story 5.1's pipeline with `imputation_method="graph_cycle_encoding"`, `experiment_id="exp05_graph_cycle"`
        * Train ALL ranker types from Epic 3 (same as Experiments 1-4 for fair comparison)
        * Document execution time for each stage
        * Validate all outputs generated
        * **Acceptance Criteria**: Experiment completes successfully, all artifacts in `experiments/exp05_graph_cycle/`, holdout test summary generated
    * **5.7**: Comparative Analysis - Compare all 5 experiments to identify the best imputation method.
        * Load holdout test summaries and detailed metrics from all 5 experiments
        * Create comparative visualizations: success metric scores, prediction accuracy, feature importance patterns
        * Compare performance across imputation methods for EACH ranker type
        * Identify the best-performing imputation method overall (and per ranker if different)
        * Document findings in a comprehensive comparison report: `reports/imputation_comparison_analysis.ipynb`
        * **Key Analysis Questions**: Which imputation method yields highest "Excellent" predictions (0/5 or 5/5 wrong)? Which is most consistent across ranker types? Are there patterns in which features work best?
        * **Performance Analysis**: Include timing analysis to inform RunPod usage decisions (per NFR1)
        * **Known Limitations**: Document limitations, assumptions, and technical debt
        * **Decision Output**: Formal recommendation on which imputation method to use for the final prediction run (Epic 7)
        * **Acceptance Criteria**: Analysis is thorough and well-documented (per NFR2), performance profiling included, clear recommendation provided with justification

### Epic 6: Model Ensembling
* **Goal**: Implement methods for combining the predictions from multiple individual rankers to improve performance.
* **Stories**:
    * **6.1**: Develop a script that loads predictions from multiple saved models for a given holdout set.
        * Load multiple model artifacts from `models/`
        * Generate predictions from each model on the same holdout data
        * Validate prediction formats are consistent across models
        * Save individual predictions for ensemble processing
        * **Error Handling**: Handle missing models, prediction failures, format inconsistencies
        * **Acceptance Criteria**: Successfully loads and runs multiple models, validates outputs, code is well-documented (per NFR2)
    * **6.2**: Implement an ensemble method using **Reciprocal Rank Fusion (RRF)** or Borda count to blend the rank lists.
        * Implement RRF/Borda count algorithm with detailed docstrings
        * Accept multiple ranked prediction lists, return combined ranking
        * Write unit tests for ensemble logic
        * **Acceptance Criteria**: Ensemble method produces valid combined rankings, code is well-documented (per NFR2), tests pass
    * **6.3**: Implement a **Weighted Averaging** ensemble method, including a walk-forward validation utility to tune the model weights.
        * Implement weighted averaging with configurable weights
        * Implement walk-forward validation for weight tuning
        * Document weight tuning approach in docstrings
        * Write unit tests for weighted ensemble
        * **Acceptance Criteria**: Ensemble method works with tuned weights, walk-forward validation is robust, code is well-documented (per NFR2), tests pass
    * **6.4**: Run an evaluation comparing at least two individual rankers against one ensemble method and generate a comparative report.
        * Use evaluation framework from Epic 4 for fair comparison
        * Generate comparative metrics and visualizations
        * Document findings in a comparative analysis notebook
        * Include recommendations on when to use ensemble vs individual models
        * **Acceptance Criteria**: Comparison is fair and thorough, insights are actionable, analysis is well-documented (per NFR2)

### Epic 7: Final Production Run with Best Method
* **Goal**: Execute the final production prediction run using the best-performing imputation method identified in Epic 5, apply ensemble techniques if beneficial, and generate the definitive "20 most likely values" prediction for the next event to achieve the project's macro objective.
* **Dependencies**: Requires completion of Epic 5 (Story 5.7 comparative analysis with method recommendation) and Epic 6 (ensemble methods)
* **Critical Context**: This epic represents the culmination of all experimental work. The best imputation method has been scientifically selected through rigorous comparison, and we now apply it to produce the final actionable prediction.
* **Stories**:
    * **7.1**: Prepare the production dataset using the best imputation method from Epic 5.
        * Load the recommendation from Story 5.7 (best imputation method)
        * Apply the selected imputation method to the FULL dataset (not just training split)
        * Validate imputed dataset quality (same checks as Epic 2, Story 2.7)
        * Save to `production/data/imputed_best_method.parquet`
        * Document which method was selected and why (reference Epic 5.7 analysis)
        * **Acceptance Criteria**: Production dataset created with best method, fully documented, quality validated
    * **7.2**: Train production models using the best-performing ranker(s) on the full imputed dataset.
        * Identify best-performing ranker(s) from Epic 5 experiments (may be method-specific or consistent across methods)
        * Train selected ranker(s) on FULL production dataset (not holdout split - use all available data for maximum performance)
        * Save production model artifacts to `production/models/`
        * Document model selection rationale, hyperparameters, training metrics
        * **Acceptance Criteria**: Production models trained successfully, artifacts saved, training fully documented
    * **7.3**: Apply ensemble methods (if beneficial) to combine production model predictions.
        * Evaluate whether ensemble methods from Epic 6 improve predictions for the selected imputation method
        * If ensemble improves performance: train ensemble on production dataset using methods from Epic 6
        * If ensemble does NOT improve: document decision to use single best ranker
        * Save ensemble configuration (if applicable) to `production/models/ensemble_config.json`
        * **Acceptance Criteria**: Ensemble decision made with justification, artifacts saved if ensemble used
    * **7.4**: Generate the final production prediction - "20 most likely values" for the next event.
        * Load production models and imputed production dataset
        * Generate prediction using best model/ensemble
        * Validate prediction format: ranked list of 20 values with confidence scores
        * Save prediction to `production/predictions/final_prediction_next_event.json`
        * Include metadata: timestamp, model used, imputation method, confidence metrics
        * **Acceptance Criteria**: Final prediction generated, properly formatted, fully documented with provenance
    * **7.5**: Create comprehensive production report documenting the entire experimental journey and final recommendation.
        * Summarize all 5 imputation experiments (Epic 5) with key findings
        * Document why the selected method was chosen (reference Epic 5.7)
        * Document model selection and ensemble decision
        * Present final prediction with confidence assessment
        * Include reproducibility instructions: "How to recreate this prediction"
        * Include future recommendations: "What to try next to improve results"
        * Save to `production/reports/final_production_report.md`
        * **Acceptance Criteria**: Report is comprehensive, tells the complete story, provides actionable next steps, heavily documented for non-programmer (per NFR2)
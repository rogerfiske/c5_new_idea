# BMad-Dev Agent Handoff Document: Epic 3 - Individual Ranker Implementation

**Date**: 2025-10-13 (Updated after Should-Fix items)
**From**: Sarah (PO Agent)
**To**: BMad-Dev Agent (James)
**Project**: Quantum State Prediction Experiment
**Current Status**: Epic 2 COMPLETE → Ready for Epic 3
**PRD Version**: v1.4 (updated with Epic 6 dependency clarification and performance profiling)

---

## Executive Summary

**Epic 2 is complete** with all 7 stories finished and 191 tests passing. The imputation framework is production-ready with 5 quantum-inspired strategies fully implemented and tested. You are now cleared to begin **Epic 3: Individual Ranker Implementation** which will build the machine learning models that consume the imputed features to predict the "20 most likely" quantum states.

**Critical Context**: Epic 3 establishes the foundational ranking models that will be used in ALL future experiments (Epic 5) and the final production run (Epic 7). These models must be thoroughly tested and documented per NFR2.

---

## Recent Updates (Post-PO Validation)

### PRD v1.4 Updates Applied
1. **✅ Epic 6 → Epic 7 Dependency Clarified**
   - Epic 6 (ensembling) is now explicitly marked as OPTIONAL
   - Epic 7 can proceed directly with single best model if ensemble doesn't improve performance
   - Story 7.3 updated to document this decision path

2. **✅ Performance Profiling Added to Epic 3-4**
   - Story 3.3: GBDT ranker now includes training time logging
   - Story 3.4: Set-based ranker now includes per-epoch and total training time
   - Story 3.5: GNN ranker now includes per-epoch and total training time
   - Story 4.3: Metrics collection now includes prediction timing
   - **Purpose**: Inform RunPod usage decisions per NFR1 (tasks >1 hour → RunPod)

3. **✅ TECHNICAL_DEBT.md Created**
   - New file tracks design shortcuts and "good enough for MVP" decisions
   - Update after each epic completion
   - Currently empty (no debt from Epic 1-2)

---

## Epic 2 Completion Summary

### ✅ Completed Work
- **Story 2.1**: BaseImputer abstract class (20 tests)
- **Story 2.2**: Basis Embedding imputation (24 tests)
- **Story 2.3**: Amplitude Embedding imputation (28 tests)
- **Story 2.4**: Angle Encoding imputation (26 tests)
- **Story 2.5**: Density Matrix Embedding imputation (32 tests)
- **Story 2.6**: Graph/Cycle Encoding imputation (28 tests)
- **Story 2.7**: Imputation utility script with CLI (23 tests)

### Test Results
- **Total Tests**: 191 passing
- **Coverage**: 100% on all imputation strategies
- **No Regressions**: All tests green
- **Commit Hash**: `267a349` (Epic 2 completion commit)

### Key Artifacts Available
```
src/imputation/
├── base_imputer.py              # Abstract base class with fit/transform interface
├── basis_embedding.py           # Strategy 1: Direct basis state mapping
├── amplitude_embedding.py       # Strategy 2: Superposition with Born rule
├── angle_encoding.py            # Strategy 3: Bloch sphere rotation angles
├── density_matrix.py            # Strategy 4: Mixed state density matrices
├── graph_cycle_encoding.py      # Strategy 5: DFT + cyclic graph features
└── apply_imputation.py          # CLI/programmatic interface to apply strategies

tests/unit/
├── test_base_imputer.py
├── test_basis_embedding.py
├── test_amplitude_embedding.py
├── test_angle_encoding.py
├── test_density_matrix.py
├── test_graph_cycle_encoding.py
└── test_apply_imputation.py
```

### Usage Example (for your reference)
```python
from src.imputation.apply_imputation import apply_imputation
from pathlib import Path

# Apply any imputation strategy programmatically
metadata = apply_imputation(
    strategy="basis_embedding",  # or any of the 5 strategies
    input_path=Path("data/raw/c5_Matrix.csv"),
    output_path=Path("data/processed/imputed_basis.parquet")
)

# CLI usage:
# python -m src.imputation.apply_imputation \
#   --strategy amplitude_embedding \
#   --input data/raw/c5_Matrix.csv \
#   --output data/processed/imputed.parquet
```

---

## Epic 3 Overview: Individual Ranker Implementation

### Goals
Develop a pipeline for training and evaluating individual ranking models on the imputed data. These models will predict a ranked list of the "20 most likely" quantum states for the next event.

### Critical Success Factors
1. **Modularity**: All rankers must follow a consistent interface (similar to BaseImputer pattern)
2. **Testability**: Each ranker must have comprehensive unit tests (target: 80%+ coverage)
3. **Documentation**: NFR2 requires heavy commenting for non-programmer understanding
4. **No Data Leakage**: Temporal ordering must be strictly maintained in train/test splits
5. **Reproducibility**: All models must be serializable and loadable for future use

---

## Epic 3 Story Breakdown (6 Stories)

### 📋 Story 3.1: Data Preparation Script
**Status**: 🔲 NOT STARTED
**Priority**: HIGHEST (blocks all other Epic 3 stories)
**Estimated Effort**: Medium (2-3 hours)

**Description**: Create the data preparation script for modeling, including splitting data into training and a strict sequential holdout set.

**Key Requirements**:
- Load imputed data from `data/processed/`
- Implement **strict sequential split** (no data leakage): train on earlier events, holdout on later events
- Validate split maintains temporal order and has no overlap
- Save split datasets: `train_split.parquet`, `holdout_split.parquet`
- Log split statistics: train size, holdout size, date ranges

**Error Handling**:
- Handle missing imputed data files
- Invalid split ratios
- Data validation failures

**Acceptance Criteria**:
- [x] Split maintains temporal integrity (earlier data in train, later in holdout)
- [x] No data leakage detected (validation checks)
- [x] Code is well-documented (per NFR2)
- [x] Unit tests validate split correctness

**Implementation Notes**:
- Use pandas `.iloc[:split_point]` for temporal split
- Add validation function: `validate_split(train_df, holdout_df)` to check no overlap
- Consider split ratio parameter (default: 80/20 or 70/30)
- Store split metadata: `split_metadata.json` with train/holdout sizes

**Testing Strategy**:
```python
def test_temporal_split_no_leakage():
    # Test that train data event IDs are all < holdout event IDs
    train_max_id = train_df['event-ID'].max()
    holdout_min_id = holdout_df['event-ID'].min()
    assert train_max_id < holdout_min_id

def test_split_no_missing_data():
    # Test that all original rows are in either train or holdout
    assert len(train_df) + len(holdout_df) == len(original_df)
```

---

### 📋 Story 3.2: Frequency-Based Baseline Rankers
**Status**: 🔲 NOT STARTED
**Priority**: HIGH
**Estimated Effort**: Medium (3-4 hours)

**Description**: Implement frequency-based baseline rankers (Cumulative, EMA, Bigram/co-occurrence) to establish performance benchmarks.

**Key Requirements**:
- Create ranker class(es) following consistent interface pattern (like BaseImputer)
- Implement 3 baseline approaches:
  1. **Cumulative Frequency**: Rank by overall position frequency in training data
  2. **EMA (Exponential Moving Average)**: Weight recent events more heavily
  3. **Bigram/Co-occurrence**: Rank based on position pair frequencies
- Include detailed docstrings explaining each baseline approach
- Write unit tests for baseline rankers

**Interface Design**:
```python
class BaseRanker(ABC):
    """Abstract base class for ranking models."""

    @abstractmethod
    def fit(self, X_train, y_train):
        """Train the ranker on training data."""
        pass

    @abstractmethod
    def predict_top_k(self, X_test, k=20):
        """Predict top-k ranked list for each test sample."""
        pass

    def get_name(self):
        """Return ranker name for logging."""
        return self.__class__.__name__

class FrequencyRanker(BaseRanker):
    """Baseline ranker using cumulative frequency."""

    def __init__(self, method='cumulative'):
        """
        Args:
            method: 'cumulative', 'ema', or 'bigram'
        """
        self.method = method
        self.position_frequencies = None  # Learned from training data
```

**Acceptance Criteria**:
- [x] Rankers produce valid ranked predictions (list of 20 positions)
- [x] Code is heavily commented (per NFR2)
- [x] Tests pass with >80% coverage
- [x] All 3 baseline methods implemented

**Implementation Notes**:
- Store position frequencies: `dict[int, float]` mapping position → frequency
- EMA: `α = 0.1` (configurable) for exponential weighting
- Bigram: Store co-occurrence matrix `np.ndarray[39, 39]`

---

### 📋 Story 3.3: Gradient Boosting Ranker (LightGBM/XGBoost)
**Status**: 🔲 NOT STARTED
**Priority**: HIGH
**Estimated Effort**: Large (4-6 hours)

**Description**: Implement a Gradient Boosting ranker using LightGBM or XGBoost with engineered features.

**Key Requirements**:
- Create ranker class with hyperparameter configuration support
- Use LightGBM's `LGBMRanker` or XGBoost's `XGBRanker`
- Include engineered features:
  - Position counts (frequency-based)
  - Circular distances (for ring structure)
  - DFT harmonics (from graph/cycle encoding)
- Write unit tests for GBDT ranker
- Include detailed docstrings explaining feature engineering

**Hyperparameters** (sensible defaults):
```python
LGBM_PARAMS = {
    'objective': 'lambdarank',  # Learning to rank objective
    'metric': 'ndcg',  # Normalized Discounted Cumulative Gain
    'n_estimators': 100,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_depth': -1,  # No limit
    'min_child_samples': 20,
    'verbosity': -1
}
```

**Acceptance Criteria**:
- [x] Ranker trains successfully on imputed data
- [x] Produces valid predictions (top-20 ranked list)
- [x] Training time logged (for RunPod decisions per NFR1)
- [x] Code is heavily commented (per NFR2)
- [x] Tests pass with >80% coverage
- [x] Handles training convergence gracefully

**Implementation Notes**:
- Use LightGBM if available (faster), fallback to XGBoost
- Feature engineering: combine imputed features with handcrafted features
- Save trained model: `lgbm_ranker_v1.joblib`
- **Performance Tracking**: Wrap training in `time.time()` and log to metadata
- Log training metrics: NDCG@20, training time, feature importance

**Performance Tracking Example**:
```python
import time

start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time

print(f"Training completed in {training_time:.2f} seconds")
if training_time > 3600:
    print("⚠️ Training exceeded 1 hour - consider using RunPod for Epic 5")
```

**Testing Strategy**:
```python
def test_lgbm_ranker_trains():
    ranker = LGBMRanker()
    ranker.fit(X_train, y_train)
    predictions = ranker.predict_top_k(X_test, k=20)
    assert len(predictions) == len(X_test)
    assert all(len(pred) == 20 for pred in predictions)
```

---

### 📋 Story 3.4: Set-Based Ranker (DeepSets/Set Transformer)
**Status**: 🔲 NOT STARTED
**Priority**: MEDIUM
**Estimated Effort**: Large (6-8 hours)

**Description**: Implement a set-based ranker using DeepSets or light Set Transformer architecture.

**Key Requirements**:
- Create ranker class following consistent interface
- Implement DeepSets or Set Transformer architecture:
  - **DeepSets**: Permutation-invariant architecture (simpler, faster)
  - **Set Transformer**: Attention-based architecture (more powerful)
- Document architecture choices and hyperparameters in docstrings
- Write unit tests for set-based ranker

**Architecture Choice**:
Recommend **DeepSets** for MVP (simpler, faster to train):
```python
class DeepSetsRanker(BaseRanker):
    """
    DeepSets architecture for permutation-invariant set ranking.

    Architecture:
        Input: Set of 5 active positions (represented as features)
        Encoder: MLP to embed each position independently
        Aggregation: Sum or max pooling (permutation-invariant)
        Decoder: MLP to predict ranking scores for all 39 positions
    """
    def __init__(self, embed_dim=64, hidden_dim=128):
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 39)  # Output: scores for all 39 positions
        )
```

**Error Handling**:
- Training divergence detection
- Invalid input shapes
- GPU/CPU compatibility issues

**Acceptance Criteria**:
- [x] Model trains successfully (loss decreases)
- [x] Produces valid predictions
- [x] Code is heavily commented (per NFR2)
- [x] Tests pass with >80% coverage
- [x] Handles training failures gracefully

**Implementation Notes**:
- Use PyTorch for implementation
- Training: 50-100 epochs, early stopping on validation loss
- Save model: `set_transformer_v1.pth`
- Consider RunPod for training (may exceed 1 hour per NFR1)

---

### 📋 Story 3.5: Graph-Based Ranker (GNN over C₃₉)
**Status**: 🔲 NOT STARTED
**Priority**: MEDIUM
**Estimated Effort**: Large (6-8 hours)

**Description**: Implement a graph-based ranker using a simple GNN or DFT over the ring C₃₉.

**Key Requirements**:
- Create ranker class following consistent interface
- Implement GNN architecture over C₃₉ cyclic graph:
  - **Graph Structure**: 39 nodes in a ring (each connected to neighbors)
  - **Node Features**: Position-specific features from imputation
  - **Message Passing**: Aggregate information from neighboring nodes
- Document graph construction and GNN architecture in docstrings
- Write unit tests for graph-based ranker

**Graph Construction**:
```python
def build_c39_graph():
    """
    Build C₃₉ cyclic graph: 39 nodes in a ring.

    Edges: Each node i connected to (i-1) mod 39 and (i+1) mod 39
    """
    edge_index = []
    for i in range(39):
        edge_index.append([i, (i + 1) % 39])  # Forward edge
        edge_index.append([i, (i - 1) % 39])  # Backward edge
    return torch.tensor(edge_index, dtype=torch.long).t()
```

**Error Handling**:
- Training divergence
- Invalid graph construction
- Dependency issues (PyTorch Geometric)

**Acceptance Criteria**:
- [x] Model trains successfully
- [x] Produces valid predictions
- [x] Code is heavily commented (per NFR2)
- [x] Tests pass with >80% coverage
- [x] Graph structure correctly represents C₃₉

**Implementation Notes**:
- Use PyTorch Geometric (`torch_geometric.nn.GCNConv` or `GATConv`)
- Training: 50-100 epochs, similar to Story 3.4
- Save model: `gnn_ranker_v1.pth`
- Consider RunPod for training (may exceed 1 hour)

---

### 📋 Story 3.6: Unified Training Script
**Status**: 🔲 NOT STARTED
**Priority**: HIGH (enables Epic 5)
**Estimated Effort**: Medium (3-4 hours)

**Description**: Create a unified training script that can train any selected ranker model on a given imputed dataset and save the artifact.

**Key Requirements**:
- Accept parameters:
  - `ranker_type`: 'frequency', 'lgbm', 'set_transformer', 'gnn'
  - `imputed_data_path`: Path to imputed Parquet file
  - `hyperparameters`: Dict of model-specific hyperparameters
  - `output_path`: Where to save trained model
- Load data, instantiate selected ranker, train model, save artifact
- Log training metadata:
  - Ranker type
  - Hyperparameters
  - Training time
  - Final metrics (loss, NDCG@20, etc.)
- Validate model artifact after saving (can be loaded successfully)
- Include progress indicators for long-running training

**CLI Interface**:
```bash
python -m src.modeling.train_ranker \
  --ranker lgbm \
  --imputed-data data/processed/imputed_basis.parquet \
  --output models/lgbm_ranker_v1.joblib \
  --hyperparameters '{"n_estimators": 100, "learning_rate": 0.1}'
```

**Error Handling**:
- Invalid ranker types
- Training failures
- Convergence issues
- Model save failures
- Insufficient memory

**Acceptance Criteria**:
- [x] Script supports all ranker types from Stories 3.2-3.5
- [x] Handles errors gracefully with clear messages
- [x] Logs training progress
- [x] Code is well-documented (per NFR2)
- [x] Can load and validate saved models

**Implementation Notes**:
- Use `argparse` for CLI (similar to `apply_imputation.py`)
- Store training metadata: `{model_path}.meta.json`
- Progress bar: use `tqdm` for training epochs
- Memory check: validate sufficient RAM before training deep learning models

---

## Technical Guidance

### Project Structure for Epic 3
```
src/modeling/
├── __init__.py
├── rankers/
│   ├── __init__.py
│   ├── base_ranker.py          # NEW: Abstract base class (Story 3.2)
│   ├── frequency_ranker.py     # NEW: Story 3.2
│   ├── lgbm_ranker.py          # NEW: Story 3.3
│   ├── set_transformer_model.py  # NEW: Story 3.4
│   └── gnn_ranker.py           # NEW: Story 3.5
├── data_prep.py                # NEW: Story 3.1
└── train_ranker.py             # NEW: Story 3.6

tests/unit/
├── test_data_prep.py           # NEW: Story 3.1 tests
├── test_frequency_ranker.py    # NEW: Story 3.2 tests
├── test_lgbm_ranker.py         # NEW: Story 3.3 tests
├── test_set_transformer.py     # NEW: Story 3.4 tests
├── test_gnn_ranker.py          # NEW: Story 3.5 tests
└── test_train_ranker.py        # NEW: Story 3.6 tests
```

### Dependencies to Add
Update `environment.yml` or `requirements.txt`:
```yaml
# For Story 3.3 (GBDT)
- lightgbm>=4.0.0
- xgboost>=2.0.0  # Fallback if LightGBM unavailable

# For Stories 3.4-3.5 (Deep Learning)
- torch>=2.0.0
- torch-geometric>=2.3.0  # For Story 3.5 (GNN)

# For training utilities
- tqdm>=4.65.0  # Progress bars
```

### Coding Standards Reminder (NFR2)
Every function must have:
1. **Docstring** with description, args, returns, examples
2. **Inline comments** explaining "why" not just "what"
3. **Type hints** for all parameters and returns
4. **Error handling** with user-friendly messages

Example:
```python
def prepare_sequential_split(
    data_path: Path,
    train_ratio: float = 0.8,
    output_dir: Path = Path("data/processed/")
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into training and holdout sets maintaining temporal order.

    This function ensures NO DATA LEAKAGE by enforcing that all training
    events occur before all holdout events. This is critical for time-series
    prediction tasks.

    Args:
        data_path: Path to imputed Parquet file
        train_ratio: Fraction of data to use for training (default: 0.8)
        output_dir: Directory to save split datasets

    Returns:
        Tuple of (train_df, holdout_df)

    Raises:
        FileNotFoundError: If data_path does not exist
        ValueError: If train_ratio not in (0, 1)

    Example:
        >>> train_df, holdout_df = prepare_sequential_split(
        ...     Path("data/processed/imputed_basis.parquet"),
        ...     train_ratio=0.8
        ... )
        >>> print(f"Train: {len(train_df)}, Holdout: {len(holdout_df)}")
        Train: 4000, Holdout: 1000
    """
    # Validate inputs
    if not data_path.exists():
        raise FileNotFoundError(
            f"Imputed data not found at {data_path}. "
            "Please run imputation first (Epic 2, Story 2.7)."
        )

    # Implementation...
```

---

## Key Artifacts to Reference

### PRD Section 3: Epic 3 Stories
**File**: `docs/prd.md` (lines 126-167)
Contains full story descriptions, acceptance criteria, and error handling requirements.

### Architecture Section 4: Project Structure
**File**: `docs/architecture.md` (lines 74-174)
Defines directory structure for `src/modeling/` and `models/`.

### Architecture Section 6: Coding Standards
**File**: `docs/architecture.md` (lines 295-301)
Emphasizes clarity, modularity, type hinting, configuration, and logging.

### Imputation Framework Reference
**Directory**: `src/imputation/`
Use `base_imputer.py` as template for `base_ranker.py` interface design.

---

## PO Validation Findings (Relevant to Epic 3)

### Must Address in Epic 3
1. **Story 3.1 is CRITICAL** - All other stories depend on this data split
2. **No Data Leakage** - Temporal ordering MUST be validated in tests
3. **NFR2 Compliance** - Every story requires heavy commenting for non-programmers
4. **Performance Tracking** - Add basic timing to Stories 3.3-3.5 (for RunPod decision in Epic 5)

### Recommendations
1. **Start with Story 3.1** - Get data prep working before any models
2. **Test Story 3.2 first** - Frequency baselines are simplest, validate interface design
3. **Profile Stories 3.3-3.5** - Track execution time to inform RunPod usage in Epic 5
4. **Create consistent interface** - All rankers should follow BaseRanker pattern (like BaseImputer)

---

## Definition of Done for Epic 3

### Code Quality
- [x] All 6 stories completed
- [x] Unit tests written for each story (target: >80% coverage)
- [x] All tests passing (no regressions)
- [x] Code heavily commented per NFR2
- [x] Type hints on all functions

### Documentation
- [x] Docstrings on all classes and functions
- [x] Training script usage documented
- [x] Model saving/loading examples provided
- [x] Error messages user-friendly

### Functionality
- [x] Data split maintains temporal order (no leakage)
- [x] All rankers produce valid top-20 predictions
- [x] Trained models can be saved and reloaded
- [x] Training script supports all 4 ranker types

### Preparedness for Epic 4
- [x] Holdout split created (Story 3.1) - ready for evaluation
- [x] Trained models available - ready for prediction
- [x] Performance timing logged - ready for RunPod decisions

---

## Questions & Clarifications

### From PO Validation
**Q**: Should Epic 3 include ensemble methods?
**A**: No, ensembles are Epic 6. Epic 3 focuses on individual rankers only.

**Q**: What is the target train/test split ratio?
**A**: Story 3.1 should use 80/20 as default, but make configurable.

**Q**: Should I create a tutorial notebook?
**A**: Optional for Epic 3, but highly recommended. Add to Story 3.6 if time permits.

---

## Success Criteria for Epic 3

### Functional Success
✅ All 4 ranker types implemented and tested
✅ Data split maintains temporal integrity
✅ Models can be trained, saved, and reloaded
✅ Training script supports all rankers

### Technical Success
✅ >80% test coverage on all rankers
✅ Zero data leakage in train/holdout split
✅ Performance timing tracked for Epic 5

### Documentation Success
✅ NFR2 compliance: Code understandable by non-programmers
✅ Clear error messages for all failure modes
✅ Training examples documented

---

## Next Actions for BMad-Dev

1. **Read** PRD Epic 3 section (lines 126-167)
2. **Read** Architecture modeling section (lines 161-173)
3. **Start** Story 3.1: Data Preparation Script
4. **Reference** BaseImputer pattern from Epic 2 for BaseRanker interface
5. **Test** as you go: write tests alongside implementation (TDD approach)

---

## Contact & Support

If you encounter blockers or need clarification:
- **PO Agent (Sarah)**: Re-run `*validate-story-draft` for story validation
- **PRD Reference**: `docs/prd.md` v1.3
- **Architecture Reference**: `docs/architecture.md` v1.2
- **Git Status**: Clean working tree, master branch, commit `267a349`

---

**🚀 You are cleared for Epic 3 development. Good luck, James!**

*Sarah (Product Owner)*
*BMad PO Agent - 2025-10-13*

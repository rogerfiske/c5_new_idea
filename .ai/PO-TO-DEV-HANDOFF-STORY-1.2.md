# Product Owner to Developer Agent Handoff Document - Story 1.2

**Date**: 2025-10-13
**From**: Sarah (Product Owner)
**To**: BMad Dev Agent (James)
**Project**: Quantum State Prediction Experiment
**Status**: ✅ APPROVED FOR STORY 1.2 DEVELOPMENT

---

## 📋 PROJECT STATUS UPDATE

### Story 1.1 Completion Summary
✅ **COMPLETE** - All acceptance criteria met on 2025-10-12

**Deliverables Completed:**
- ✅ Complete directory structure (140 files tracked)
- ✅ src/config.py with centralized path management (9,826 bytes)
- ✅ README.md with comprehensive setup instructions (6,058 bytes)
- ✅ CONTRIBUTING.md with coding standards (12,649 bytes)
- ✅ .gitignore for Python/data science project (1,267 bytes)
- ✅ All __init__.py files for package structure
- ✅ 2 git commits with detailed messages
- ✅ Dataset moved to data/raw/c5_Matrix.csv (973,514 bytes)
- ✅ Windows Unicode compatibility fixed

**Git History:**
```
a66fbdd - Add session summary for 2025-10-12
89bef7b - Fix Unicode encoding issues in config.py and move dataset to correct location
6e1831d - Initial project scaffolding for Quantum State Prediction Experiment
```

### Current Project Status
- **Epic 1 Progress**: 25% complete (Story 1.1 ✅, Stories 1.2-1.4 pending)
- **Overall Project**: 🟢 ON TRACK
- **Critical Blockers**: None
- **PRD Version**: v1.3
- **Architecture Version**: v1.2

---

## 🎯 IMMEDIATE NEXT TASK: EPIC 1, STORY 1.2

### Story 1.2: Data Loading and Validation
**Goal**: Implement a script to load c5_Matrix.csv, validate its structure, and handle data cleaning

**Location in PRD**: `docs/prd.md` lines 60-68

### Why This Story Matters
This story establishes the foundation for all subsequent work by:
1. **Validating dataset integrity** - Ensures data quality before modeling
2. **Establishing data contract** - Defines expected structure for downstream processes
3. **Implementing error handling patterns** - Sets standards for user-friendly errors (NFR2)
4. **Creating reusable data utilities** - Functions will be used throughout the project

---

## 📝 DETAILED TASK BREAKDOWN

### Task 1: Create src/data_loader.py

**File**: `src/data_loader.py`

**Required Functions:**

#### 1.1 `load_dataset(file_path: Path = None) -> pd.DataFrame`
**Purpose**: Load the CSV file and return a Pandas DataFrame

**Requirements**:
- Default parameter: `file_path=None` (uses `config.DATASET_PATH` if None)
- Use `src/config.py` constants (import `DATASET_PATH` from config)
- Implement file existence check BEFORE attempting to load
- Read CSV with appropriate pandas parameters (handle potential encoding issues)
- Log loading progress using Python logging module
- Return DataFrame

**Error Handling**:
```python
if not file_path.exists():
    raise FileNotFoundError(
        f"Dataset not found at {file_path}.\n"
        f"USER ACTION REQUIRED: Ensure c5_Matrix.csv is in the data/raw/ directory.\n"
        f"See README.md section 'Data Setup' for instructions."
    )
```

**NFR2 Requirements**:
- Comprehensive docstring with parameter types and return value
- Inline comments explaining why we check existence before loading
- User-friendly error messages (no cryptic pandas errors)

---

#### 1.2 `validate_dataset_structure(df: pd.DataFrame) -> dict`
**Purpose**: Validate that the dataset matches expected structure from dataset description

**Validations Required**:
1. **Column count check**: Should be 45 columns
2. **Column names check**:
   - Event_ID (int)
   - Timestamp (datetime or string - document which)
   - QS_1, QS_2, QS_3, QS_4, QS_5 (int, values 1-39)
   - QV_1 through QV_39 (int, binary 0 or 1)
3. **Row count check**: ~5000 rows (document actual count)
4. **No missing values**: Check for NaN/null values

**Return Value**:
```python
{
    "valid": True/False,
    "row_count": int,
    "column_count": int,
    "missing_values": int,
    "validation_errors": [list of error strings if any]
}
```

**Error Messages** (user-friendly):
- "Expected 45 columns but found {actual}. Dataset may be corrupted."
- "Missing required column: {column_name}. Check dataset format."
- "Found {count} missing values. Dataset should be complete."

**NFR2 Requirements**:
- Docstring explains what makes a valid dataset
- Inline comments explain each validation rule
- Return dictionary structure documented in docstring

---

#### 1.3 `validate_data_integrity(df: pd.DataFrame) -> dict`
**Purpose**: Validate business rules and data integrity

**Validations Required**:
1. **Event_ID validation**:
   - Sequential (no gaps)
   - No duplicates
   - Starts from expected value (document what that is)

2. **QS value validation** (Quantum State columns):
   - All values in range 1-39
   - Exactly 5 values per row
   - Document any patterns observed

3. **QV value validation** (Quantum Value columns):
   - All values are binary (0 or 1)
   - Sum of QV per row = 5 (exactly 5 active positions)
   - Validate this matches QS positions

4. **Timestamp validation** (if applicable):
   - Chronological order
   - No duplicates
   - Proper datetime format

**Return Value**:
```python
{
    "valid": True/False,
    "event_id_sequential": True/False,
    "qs_values_valid": True/False,
    "qv_sum_valid": True/False,
    "integrity_errors": [list of error strings if any]
}
```

**NFR2 Requirements**:
- Explain WHY these business rules matter
- Document the quantum state representation (5 active positions out of 39)
- User-friendly error messages

---

#### 1.4 `clean_dataset(df: pd.DataFrame) -> pd.DataFrame`
**Purpose**: Handle common data issues (if any found during validation)

**Cleaning Operations**:
- Strip whitespace from string columns (if applicable)
- Handle encoding issues
- Convert data types if needed (e.g., ensure ints are int, not float)
- Document any transformations applied

**Requirements**:
- Log all cleaning operations
- Return cleaned DataFrame
- Do NOT modify invalid data (raise errors instead)
- Only handle minor formatting issues

**NFR2 Requirements**:
- Explain each cleaning operation
- Log what was cleaned and why

---

#### 1.5 `save_processed_data(df: pd.DataFrame, output_path: Path = None) -> None`
**Purpose**: Save cleaned/validated data to processed directory

**Requirements**:
- Default: `output_path = config.DATA_PROCESSED / "validated_dataset.csv"`
- Ensure output directory exists (create if needed)
- Save as CSV with proper formatting
- Log save operation with file size

**NFR2 Requirements**:
- Docstring explains when to use this function
- Log confirmation message with file path

---

### Task 2: Create tests/unit/test_data_loader.py

**File**: `tests/unit/test_data_loader.py`

**Required Tests**:

#### 2.1 Test Fixtures (in tests/fixtures/)
Create sample data files for testing:
- `sample_valid_dataset.csv` - Small (10 rows) valid dataset
- `sample_invalid_columns.csv` - Missing columns
- `sample_invalid_values.csv` - QV sum != 5, QS out of range
- `sample_missing_values.csv` - Contains NaN values

#### 2.2 Unit Tests
1. `test_load_dataset_success()` - Load valid dataset
2. `test_load_dataset_file_not_found()` - Verify error message
3. `test_validate_structure_valid()` - Valid dataset passes
4. `test_validate_structure_wrong_columns()` - Detect column issues
5. `test_validate_integrity_valid()` - Business rules pass
6. `test_validate_integrity_qv_sum_invalid()` - Detect sum != 5
7. `test_validate_integrity_qs_out_of_range()` - Detect QS not in 1-39
8. `test_clean_dataset()` - Verify cleaning operations
9. `test_save_processed_data()` - Verify file saving

**Testing Standards**:
- Use pytest fixtures for sample data
- Use `tmp_path` fixture for file operations (no permanent test files)
- Assert on both return values AND log messages
- Test both happy path and error cases

**NFR2 Requirements**:
- Each test has docstring explaining what it validates
- Test names are descriptive

---

### Task 3: Test the Data Loader

**Commands to run**:
```bash
# Activate environment (if created)
# conda activate quantum_project

# Test 1: Import check
python -c "from src.data_loader import load_dataset; print('Import successful')"

# Test 2: Load and validate dataset
python -c "
from src.data_loader import load_dataset, validate_dataset_structure, validate_data_integrity
df = load_dataset()
print(f'Loaded {len(df)} rows, {len(df.columns)} columns')
structure = validate_dataset_structure(df)
integrity = validate_data_integrity(df)
print(f'Structure valid: {structure[\"valid\"]}')
print(f'Integrity valid: {integrity[\"valid\"]}')
"

# Test 3: Run pytest
pytest tests/unit/test_data_loader.py -v
```

**Expected Results**:
- All imports succeed
- Dataset loads with ~5000 rows, 45 columns
- Structure validation passes
- Integrity validation passes (or documents specific issues if found)
- All pytest tests pass

---

### Task 4: Document Findings

**Create**: `logs/data_validation_report.txt`

**Content Structure**:
```
DATA VALIDATION REPORT
======================
Date: 2025-10-13
Dataset: data/raw/c5_Matrix.csv
Dataset Size: 973,514 bytes (951 KB)

STRUCTURE VALIDATION
--------------------
Rows: {actual_count}
Columns: {actual_count}
Expected Columns: 45 (Event_ID, Timestamp, QS_1-5, QV_1-39)
Column Names Match: Yes/No
Data Types Valid: Yes/No
Missing Values: {count}

INTEGRITY VALIDATION
--------------------
Event_ID Sequential: Yes/No
Event_ID Range: {first_id} to {last_id}
QS Values in Range (1-39): Yes/No
QV Values Binary (0 or 1): Yes/No
QV Sum = 5 per row: Yes/No

FINDINGS & OBSERVATIONS
-----------------------
- [List any patterns observed]
- [Note any data quality issues]
- [Document any unexpected characteristics]

RECOMMENDATION
--------------
Dataset is READY/NOT READY for downstream processing.
[If not ready: List required corrections]
```

**Purpose**: This report will be referenced in Story 1.3 (Preprocessing Pipeline)

---

### Task 5: Git Commit

**After all tasks complete and tests pass**:

```bash
git add src/data_loader.py tests/unit/test_data_loader.py tests/fixtures/* logs/data_validation_report.txt
git commit -m "$(cat <<'EOF'
Implement data loading and validation (Story 1.2)

Features:
- src/data_loader.py with 5 core functions:
  * load_dataset() - Load CSV with error handling
  * validate_dataset_structure() - Check shape and columns
  * validate_data_integrity() - Check business rules
  * clean_dataset() - Handle data formatting issues
  * save_processed_data() - Save validated data

- Comprehensive validation:
  * 45 columns (Event_ID, Timestamp, QS_1-5, QV_1-39)
  * ~5000 rows validated
  * QS values in range 1-39
  * QV values binary (0 or 1)
  * QV sum = 5 per row (exactly 5 active positions)
  * Event_ID sequential with no duplicates

- Unit tests:
  * 9 test cases covering happy path and error cases
  * Test fixtures for various data scenarios
  * All tests passing

- Documentation:
  * Comprehensive docstrings per NFR2
  * User-friendly error messages
  * Data validation report in logs/

- Dataset Status: {VALIDATED/ISSUES FOUND - see logs/data_validation_report.txt}

Epic 1, Story 1.2 complete.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## 🔑 CRITICAL CONTEXT FOR DEVELOPER

### Dataset Location Confirmed
✅ **Dataset is present and validated**:
- Location: `data/raw/c5_Matrix.csv`
- Size: 973,514 bytes (951 KB)
- Status: Confirmed by user and previous session

### NFR2 Compliance - Non-Programmer Friendly Code
**MANDATORY for every function**:
- ✅ Comprehensive docstrings with parameter types and return values
- ✅ Inline comments explaining WHY, not just WHAT
- ✅ User-friendly error messages (no cryptic stack traces)
- ✅ Descriptive variable names (no single letters except i, j for loops)
- ✅ Example usage in docstrings where helpful

**Example Docstring Format**:
```python
def validate_dataset_structure(df: pd.DataFrame) -> dict:
    """
    Validate that the dataset matches the expected structure.

    The quantum state dataset should have exactly 45 columns:
    - Event_ID: Unique identifier for each event
    - Timestamp: When the event occurred
    - QS_1 through QS_5: The 5 active quantum state positions (values 1-39)
    - QV_1 through QV_39: Binary indicators (0 or 1) for each of 39 possible positions

    Args:
        df (pd.DataFrame): The dataset to validate

    Returns:
        dict: Validation results with keys:
            - "valid" (bool): True if structure is correct
            - "row_count" (int): Number of rows in dataset
            - "column_count" (int): Number of columns found
            - "validation_errors" (list): Any errors found (empty if valid)

    Raises:
        ValueError: If critical structure issues prevent validation

    Example:
        >>> df = load_dataset()
        >>> results = validate_dataset_structure(df)
        >>> if results["valid"]:
        ...     print("Dataset structure is correct!")
    """
```

### Testing Standards
- Use pytest as testing framework (already installed in Story 1.1)
- Create test fixtures in `tests/fixtures/` directory
- Use `tmp_path` fixture for temporary file operations
- Test both success and failure cases
- Assert on return values AND log messages where applicable

### Logging Standards
```python
import logging

logger = logging.getLogger(__name__)

# Usage:
logger.info(f"Loading dataset from {file_path}")
logger.info(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
logger.warning(f"Found {missing_count} missing values")
logger.error(f"Validation failed: {error_message}")
```

### Path Management
✅ **Always use src/config.py constants**:
```python
from src.config import DATASET_PATH, DATA_PROCESSED

# Good:
df = pd.read_csv(DATASET_PATH)

# Bad (never do this):
df = pd.read_csv("data/raw/c5_Matrix.csv")
```

---

## 📊 ACCEPTANCE CRITERIA (Story 1.2)

Story 1.2 is **COMPLETE** when:

1. ✅ `src/data_loader.py` exists with 5 required functions
2. ✅ All functions have comprehensive docstrings (per NFR2)
3. ✅ File existence check with user-friendly error message
4. ✅ Dataset structure validation (45 columns, proper names)
5. ✅ Data integrity validation (QS range, QV binary, QV sum = 5)
6. ✅ Unit tests in `tests/unit/test_data_loader.py` with 9+ test cases
7. ✅ All tests pass (`pytest tests/unit/test_data_loader.py`)
8. ✅ Data validation report created in `logs/data_validation_report.txt`
9. ✅ Git commit with detailed message
10. ✅ Code is well-commented and documented (per NFR2)

### Verification Commands
```bash
# Run these to verify completion:
python -c "from src.data_loader import load_dataset; df = load_dataset(); print(f'✅ Loaded {len(df)} rows')"
pytest tests/unit/test_data_loader.py -v
cat logs/data_validation_report.txt
git log -1 --oneline
```

---

## 🚀 READY TO PROCEED?

**Pre-Flight Checklist**:
- [x] Story 1.1 complete (confirmed)
- [x] Dataset present at data/raw/c5_Matrix.csv (confirmed)
- [x] PRD v1.3 and Architecture v1.2 aligned (confirmed)
- [x] Project structure ready (confirmed)
- [x] Git repository clean (confirmed)

**Story 1.2 Dependencies**: None - Ready to start immediately

**Estimated Effort**: 2-3 hours

**Next Story After 1.2**: Story 1.3 - Preprocessing Pipeline

---

## 📚 REFERENCE DOCUMENTS

### Required Reading
1. **PRD Story 1.2**: `docs/prd.md` lines 60-68
2. **Architecture Section 6**: `docs/architecture.md` lines 293-295 (Coding Standards)
3. **Dataset Description**: `docs/Assigning Quantum States to Binary csv.md`
4. **Previous Session Summary**: `📋 SESSION SUMMARY - 2025-10-12.txt`

### Key Files to Reference
- `src/config.py` - Path constants (created in Story 1.1)
- `README.md` - Project setup instructions
- `CONTRIBUTING.md` - Coding standards
- `.gitignore` - What not to commit

---

## 💡 TIPS FOR SUCCESS

### 1. Start with Data Exploration
Before implementing validation, load the dataset and explore it:
```python
import pandas as pd
from src.config import DATASET_PATH

df = pd.read_csv(DATASET_PATH)
print(df.head())
print(df.info())
print(df.describe())
```

### 2. Test-Driven Development
Consider writing tests first (TDD approach):
1. Write test for happy path (valid dataset)
2. Implement function to make test pass
3. Write test for error case
4. Improve function to handle error
5. Repeat

### 3. Error Message Examples
Good error messages for non-programmers:
```python
# Good:
raise ValueError(
    f"Invalid QV sum detected. Expected exactly 5 active positions per row, "
    f"but found rows with different sums. This indicates the quantum state "
    f"representation is corrupted. See docs/Assigning Quantum States to Binary csv.md "
    f"for the correct format."
)

# Bad:
raise ValueError("QV sum invalid")
```

### 4. Logging Examples
```python
logger.info(f"Loading dataset from {DATASET_PATH}")
logger.info(f"Dataset loaded successfully: {len(df):,} rows × {len(df.columns)} columns")
logger.info(f"Validating dataset structure...")
logger.info(f"✓ Structure validation passed")
logger.info(f"Validating data integrity...")
logger.info(f"✓ Integrity validation passed")
logger.info(f"Dataset is ready for downstream processing")
```

---

## 📞 QUESTIONS OR ISSUES?

If you encounter:
1. **Dataset format issues**: Document in `logs/data_validation_report.txt` and raise with user
2. **Unexpected data patterns**: Document and ask user for clarification
3. **Technical blockers**: Review architecture.md and PRD for guidance
4. **NFR2 compliance questions**: Ask "Would a non-programmer understand this?"

---

**Handoff Complete. Story 1.2 is ready for development. Good luck! 🚀**

---

**Sarah (Product Owner)**
*Quantum State Prediction Experiment - c5_new_idea*
*Date: 2025-10-13*
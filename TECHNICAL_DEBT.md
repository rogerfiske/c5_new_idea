# Technical Debt Tracker

This document tracks design shortcuts, TODOs, and "good enough for MVP" decisions made during development. Each item should include context, impact, and recommendations for future improvement.

**Last Updated**: 2025-10-13 (Epic 2 Complete)

---

## Active Technical Debt

### Epic 2: Quantum-Inspired Imputation Framework

*Currently: No significant technical debt identified. All strategies implemented with 100% test coverage.*

---

## Resolved Technical Debt

*None yet - project just started*

---

## Future Considerations (Not Yet Debt)

### 1. CI/CD Pipeline
**Status**: Not implemented
**Context**: Manual testing and deployment sufficient for MVP
**Impact**: Low - Single developer, experimental project
**Recommendation**: Add GitHub Actions workflow post-MVP if project transitions to production or multi-developer team

### 2. Formal Architecture Decision Records (ADRs)
**Status**: Not implemented
**Context**: Key decisions documented in PRD/Architecture changelogs
**Impact**: Low - Decisions are documented, just not in ADR format
**Recommendation**: Adopt ADR format if team grows beyond 2-3 developers

### 3. Automated Code Quality Gates
**Status**: Not implemented (no linting, type checking enforcement)
**Context**: Manual code review and NFR2 compliance sufficient for MVP
**Impact**: Low - Code quality maintained through reviews
**Recommendation**: Add `mypy`, `black`, `flake8` pre-commit hooks post-MVP

### 4. Performance Optimization
**Status**: Not prioritized yet
**Context**: Focus is on correctness and experimentation, not production performance
**Impact**: Low for experiments, Medium for production run (Epic 7)
**Recommendation**: Profile and optimize in Epic 5 if RunPod usage becomes cost-prohibitive

---

## Guidelines for Adding Technical Debt

When documenting technical debt, include:

1. **Title**: Brief description of the shortcut taken
2. **Epic/Story**: Where the debt was introduced
3. **Context**: Why this decision was made (time, complexity, uncertainty)
4. **Impact**: Low/Medium/High severity for future development
5. **Recommendation**: How to address it in the future
6. **Date Added**: When this debt was created

### Example Entry:

```markdown
### [Title] Hardcoded Hyperparameters in LGBM Ranker
**Epic/Story**: Epic 3, Story 3.3
**Context**: Used sensible defaults to get MVP working quickly. Hyperparameter tuning deferred to Epic 5.
**Impact**: Medium - May not achieve optimal performance in experiments
**Recommendation**: Add grid search or Optuna tuning in Epic 5.1 experiment orchestration
**Date Added**: 2025-10-XX
**Resolved**: N/A
```

---

## Technical Debt Review Schedule

- **After each Epic completion**: Review and add new debt items
- **Before Epic 5 (Experiments)**: Review all debt, prioritize anything that could impact experiment validity
- **Before Epic 7 (Production)**: Review all debt, address critical items that could impact final prediction

---

## Notes

- **Philosophy**: Document debt, don't fear it. MVP requires tradeoffs.
- **Prioritization**: Focus on correctness first, optimization later
- **Transparency**: Better to document shortcuts than pretend they don't exist

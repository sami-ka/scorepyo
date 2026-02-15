# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Philosophy

- **TDD**: Always write a failing test first before implementing any feature or fix. Confirm it fails, then implement to make it pass.
- **Clean architecture**: Keep concerns well-separated; respect existing module boundaries.

## Commands

All source code lives under `core_module/`. Install with dev dependencies:
```bash
pip install -e core_module/.[dev,doc]
```

Run tests:
```bash
pytest -rfs --cov=scorepyo --cov-report term-missing
```

Run a single test:
```bash
pytest core_module/tests/test_models.py::test_name -rfs
```

Format code (black + isort):
```bash
python codepal.py format
```

Type checking:
```bash
mypy --config-file ./core_module/pyproject.toml core_module/
```

Lint:
```bash
pylint --rcfile=./core_module/pyproject.toml .
```

All tool configs (black, isort, pylint, mypy, bandit) are in `core_module/pyproject.toml`.

## Architecture

Scorepyo builds **interpretable risk-score models** for binary classification — integer-based scoring systems where features map to points and scores map to probabilities.

### Package location
`core_module/scorepyo/` — the installable package source.

### Core pipeline components (all in `core_module/scorepyo/`)

- **`binarizers.py`** — `EBMBinarizer`: converts continuous/categorical features to binary features using Explainable Boosting Machine (interpretML). EBM piecewise-constant functions become binary thresholds; categorical features become one-hot.

- **`ranking.py`** — Multiple rankers to score feature importance: `LogOddsDensity`, `DiverseLogOddsDensity`, `LassoPathRank`, `LarsPathRank`, `OMPRank`, `FasterRiskRank`, `MRMRRank`, `BordaRank`. Used to select which binary features enter the model.

- **`calibration.py`** — `VanillaCalibrator` and `BootstrappedCalibrator`: map integer scores to probabilities via convex optimization (cvxpy) with monotonicity constraints.

- **`models.py`** — Main model classes:
  - `RiskScore`: requires pre-binarized features
  - `EBMRiskScore`: end-to-end model (binarizes internally via EBM)
  - Both inherit from `_BaseRiskScore`; expose `fit()`, `predict()`, `predict_proba()`, `summary()`

- **`_utils.py`** — Numba-accelerated AUC computation used during feature enumeration.

- **`exceptions.py`** — Custom exceptions with pandera integration for input validation.

### Data flow
```
Raw features → EBMBinarizer → binary features → Ranker (select top-k) → enumerate point assignments → Calibrator → risk score model
```

### Key dependencies
- `interpret` (EBM binarization)
- `cvxpy` (calibration optimization)
- `fasterrisk` (FasterRisk ranker)
- `mrmr-selection` (MRMRRank)
- `pandera` (DataFrame schema validation throughout)
- `polars` (lazy evaluation for point enumeration)

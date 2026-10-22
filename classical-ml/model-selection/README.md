
# Model Selection

This module covers hyperparameter tuning and practical model selection using scikit-learn.

## Topics

| Topic | Concepts |
|---|---|
| Hyperparameter Tuning | Model parameters vs hyperparameters |
| Grid Search | Exhaustive search using `GridSearchCV` |
| Randomized Search | Sampling parameter settings using `RandomizedSearchCV` |
| Practical Model Selection | Cross-validation, metrics, candidate comparison |

## Files

- `grid-search.md` / `grid-search.py`
- `randomized-search.md` / `randomized-search.py`
- `model-selection.md` / `model-selection.py`

## Key Lessons

- Select an appropriate scoring metric for the problem.
- Use cross-validation to compare candidate models.
- Keep the test set separate from tuning and model selection.
- Use pipelines to prevent preprocessing leakage.
- Search strategies and parameter ranges affect the selected model.
- The highest score does not automatically make a model best for every deployment scenario.

## Dataset

Examples use scikit-learn's Breast Cancer Wisconsin dataset.

## Run Examples

From the repository root:

```bash
python classical-ml/model-selection/grid-search.py
python classical-ml/model-selection/randomized-search.py
python classical-ml/model-selection/model-selection.py
```

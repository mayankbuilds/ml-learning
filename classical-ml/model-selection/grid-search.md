
# Grid Search

## What is Grid Search?

Grid Search evaluates every specified combination of hyperparameters using cross-validation and selects the combination with the best mean validation score.

## Important API

- `GridSearchCV`: performs exhaustive hyperparameter search.
- `param_grid`: dictionary of hyperparameters and candidate values.
- `best_params_`: best parameter combination found.
- `best_score_`: best mean cross-validation score.
- `best_estimator_`: estimator refitted using the best parameters.

## Example

```python
param_grid = {
    "max_depth": [2, 4, 6, None],
    "min_samples_split": [2, 5, 10],
}
```

This grid contains 4 × 3 = 12 combinations.

With 5-fold cross-validation, the search performs 12 × 5 = 60 model fits, plus refitting the best estimator on the full training data.

## Best Practices

- Use training data for hyperparameter search.
- Keep the final test set untouched until final evaluation.
- Choose a scoring metric appropriate for the problem.
- Use a pipeline when preprocessing is required.
- Avoid unnecessarily large grids because exhaustive search can be expensive.

## Limitation

Grid Search can become computationally expensive when many hyperparameters and candidate values are included.

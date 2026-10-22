
# Randomized Search

## What is Randomized Search?

`RandomizedSearchCV` evaluates a fixed number of sampled hyperparameter combinations using cross-validation instead of testing every combination in the search space.

## Important Parameters

- `estimator`: model to tune.
- `param_distributions`: hyperparameters and candidate values.
- `n_iter`: number of parameter settings to sample.
- `cv`: number of cross-validation folds.
- `scoring`: metric used to select the best model.
- `random_state`: makes sampling reproducible.
- `n_jobs`: number of parallel jobs; `-1` uses available processors.

## Important Attributes

- `best_params_`: best sampled hyperparameter combination.
- `best_score_`: best mean cross-validation score.
- `best_estimator_`: best estimator refitted on the training data.
- `cv_results_`: results for all sampled settings.

## Grid Search vs Randomized Search

| Grid Search | Randomized Search |
|---|---|
| Tests every specified combination | Samples a fixed number of settings |
| Cost grows with all grid combinations | Cost is controlled using `n_iter` |
| Good for small search spaces | Useful for large search spaces |
| Exhaustive over the specified grid | Does not guarantee finding the global best |

## Best Practices

- Use training data for hyperparameter search.
- Keep the test set untouched until final evaluation.
- Choose an appropriate scoring metric.
- Set `random_state` for reproducible sampling.
- Compare methods using consistent data splits and scoring.

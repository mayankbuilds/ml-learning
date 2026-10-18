# Overfitting and Underfitting

## 1. Generalization

Generalization is a model's ability to make useful predictions on unseen data drawn from the intended prediction setting.

A model should learn patterns that generalize rather than simply memorizing its training observations.

## 2. Underfitting

Underfitting occurs when a model fails to capture important patterns in the data.

Typical signs:

- Poor training performance.
- Poor validation performance.
- Model may be too simple for the task.
- Important features or relationships may be missing.

Possible remedies:

- Try a more expressive model.
- Add meaningful features.
- Reduce excessive regularization.
- Train longer when the algorithm has not converged.
- Check whether the data contains enough useful information.

## 3. Overfitting

Overfitting occurs when a model fits training-specific patterns, including noise, that do not generalize well.

Typical signs:

- Very good training performance.
- Noticeably worse validation performance.
- Performance changes substantially across data splits.

Possible remedies:

- Collect more representative training data.
- Reduce model complexity.
- Apply suitable regularization.
- Use cross-validation for model selection.
- Remove leakage and investigate unreliable features.
- Use early stopping when supported and appropriate.

These signs are diagnostic clues, not absolute rules. A performance gap can also arise from distribution differences, small datasets, or an unsuitable split.

## 4. Comparison

| Property | Underfitting | Good fit | Overfitting |
|---|---|---|---|
| Training error | Usually high | Low enough | Often very low |
| Validation error | High | Relatively low | Higher than training error |
| Learned pattern | Too simple | Generalizable | Too dependent on training data |
| Main response | Improve model/data | Validate and monitor | Improve generalization |

For error metrics such as MAE or RMSE, lower values are better. For metrics such as accuracy or F1-score, higher values are generally better.

## 5. Bias–Variance Intuition

- **High bias:** The model makes restrictive assumptions and misses important patterns.
- **High variance:** The fitted model is sensitive to the particular training sample.

A very simple model may have high bias. An overly flexible model trained on limited data may have high variance.

Model complexity, data quality, dataset size, and regularization all influence this balance.

## 6. Practical Experiment

The following experiment compares decision trees with different maximum depths on a synthetic regression dataset.

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

X, y = make_regression(
    n_samples=500,
    n_features=5,
    noise=25,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)

for depth in [1, 2, 5, 10, None]:
    model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_mse = mean_squared_error(
        y_train, model.predict(X_train)
    )
    test_mse = mean_squared_error(
        y_test, model.predict(X_test)
    )

    print(
        f"depth={depth!s:>4} | "
        f"train MSE={train_mse:8.2f} | "
        f"test MSE={test_mse:8.2f}"
    )
```

### How to Interpret the Output

- A shallow tree may have high training and test MSE: possible underfitting.
- A deeper tree can reduce training MSE.
- If training MSE keeps falling while test MSE increases, the model may be overfitting.
- The best depth is not necessarily the largest depth.

The exact values depend on the generated dataset and split. Use validation data or cross-validation to choose depth rather than selecting it from repeated test-set comparisons.

## 7. Regularization

Regularization discourages overly complex solutions or large parameter values.

Examples:

- Ridge regression uses an L2 penalty.
- Lasso regression uses an L1 penalty.
- Decision trees can be constrained using parameters such as `max_depth` and `min_samples_leaf`.

Regularization strength must be selected carefully. Too little may permit overfitting; too much may cause underfitting.

## Common Mistakes

- Assuming a complex model is always better.
- Judging generalization using training performance alone.
- Calling every train/validation gap overfitting without investigating the data.
- Repeatedly adjusting a model based on test results.
- Assuming regularization always improves performance.

## Quick Revision

- Underfitting: the model has not learned enough useful structure.
- Overfitting: the model has learned patterns that do not generalize.
- Generalization: performance on unseen data.
- Bias relates to restrictive assumptions; variance relates to sensitivity to training data.
- Use validation or cross-validation to compare models and control complexity.

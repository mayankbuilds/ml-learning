# Regression Evaluation

## 1. Overview

Regression models predict continuous numerical values, such as prices, temperatures, or demand.

Evaluation metrics quantify prediction errors and help compare candidate models.

## 2. Mean Absolute Error (MAE)

MAE is the average absolute difference between actual and predicted values.

$$
\text{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

- Lower is better.
- Expressed in the target's units.
- Less sensitive to large individual errors than MSE.

## 3. Mean Squared Error (MSE)

MSE averages squared prediction errors.

$$
\text{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

- Lower is better.
- Penalizes large errors more heavily because errors are squared.
- Expressed in squared target units.

## 4. Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.

$$
\text{RMSE}=\sqrt{\text{MSE}}
$$

- Lower is better.
- Expressed in the target's units.
- Gives relatively more weight to large errors.

## 5. R² Score

R² compares model prediction errors with the error from predicting the target mean.

- `1.0`: Perfect predictions.
- `0.0`: Equivalent to the mean-prediction baseline on the evaluated data.
- Below `0.0`: Worse than that baseline.

R² is not generally a percentage of predictions that are correct.

## 6. Scikit-Learn Metrics

```python id="w4nj0k"
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)
```

## 7. Metric Selection

- MAE: Useful for interpretable average absolute error.
- RMSE: Useful when large errors should be penalized more.
- R²: Useful for comparing performance against a mean-prediction baseline.
- Consider domain-specific error costs and the target's scale.

## 8. Key Takeaway

Use more than one metric when useful. Compare models on the same held-out data and interpret error magnitudes in the context of the target variable.

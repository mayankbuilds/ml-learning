# Regression Evaluation Metrics

Regression metrics quantify the difference between observed target values and model predictions.

Let:

- \(y_i\): observed target
- \(\hat{y}_i\): predicted target
- \(n\): number of observations

## 1. Mean Absolute Error (MAE)

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

MAE is the average absolute prediction error.

- Lower is better.
- It uses the same units as the target.
- It is less sensitive to large errors than squared-error metrics.

Use MAE when average absolute error is easy to interpret.

## 2. Mean Squared Error (MSE)

\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

MSE averages the squared prediction errors.

- Lower is better.
- Its units are the square of the target units.
- Large errors receive disproportionately more weight.

## 3. Root Mean Squared Error (RMSE)

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

RMSE is the square root of MSE.

- Lower is better.
- It uses the same units as the target.
- It penalizes large errors more heavily than MAE.

## 4. R² Score

\[
R^2 = 1 -
\frac{\sum_i(y_i-\hat{y}_i)^2}
{\sum_i(y_i-\bar{y})^2}
\]

Where \(\bar{y}\) is the mean of the observed target values.

R² compares the model's squared error with a reference based on predicting the observed target mean.

- R² = 1: perfect predictions on the evaluated observations.
- R² = 0: the model's squared error equals that of the observed-mean reference.
- R² < 0: the model performs worse than that reference under squared error.

R² is not the percentage of predictions that are correct. It can be negative on test data and should not be interpreted alone.

## 5. Compare Metrics with scikit-learn

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)

X, y = make_regression(
    n_samples=500,
    n_features=4,
    noise=25,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))
print("RMSE:", root_mean_squared_error(y_test, predictions))
print("R²:", r2_score(y_test, predictions))
```

## 6. How to Choose a Metric

| Metric | Useful when | Main limitation |
|---|---|---|
| MAE | Typical absolute error matters | Does not emphasize large errors as strongly |
| MSE | Large errors need a stronger penalty | Squared units are less intuitive |
| RMSE | Large errors matter and target units aid interpretation | Sensitive to large errors |
| R² | Comparing against a mean-based reference | Does not express error in target units |

Always evaluate models on the same data and compare metrics appropriate to the problem.

## 7. Small Numerical Example

Suppose the actual values are `[3, 5, 7]` and predictions are `[2, 5, 9]`.

Absolute errors are `[1, 0, 2]`.

\[
MAE = \frac{1+0+2}{3}=1
\]

Squared errors are `[1, 0, 4]`.

\[
MSE = \frac{1+0+4}{3}=\frac{5}{3}
\]

\[
RMSE = \sqrt{\frac{5}{3}}\approx 1.291
\]

This illustrates why MAE and RMSE can give different impressions of the same predictions.

## Common Mistakes

- Comparing metrics calculated on different test sets.
- Treating a lower MSE as directly comparable to a lower MAE without considering units and scale.
- Assuming R² must be between 0 and 1.
- Using training metrics as the only evidence of generalization.
- Ignoring target units and the real cost of prediction errors.

## Quick Revision

- MAE measures average absolute error.
- MSE averages squared errors.
- RMSE is the square root of MSE.
- R² compares squared error with a mean-based reference.
- Lower MAE, MSE, and RMSE are better; higher R² is generally better when evaluating the same task and dataset.

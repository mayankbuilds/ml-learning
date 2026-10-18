# Regularization: Ridge and Lasso

## 1. What Is Regularization?

Regularization adds a penalty to a model's objective to discourage overly complex parameter values.

For regression, regularization can reduce coefficient magnitudes and help control overfitting, especially when features are numerous or correlated.

A regularized model trades off fitting the training data against the penalty.

## 2. Ridge Regression (L2)

Ridge adds a squared-coefficient penalty to the least-squares objective:

\[
J(\boldsymbol{\beta}) =
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
+\alpha\sum_{j=1}^{p}\beta_j^2
\]

Here, \(\alpha\) controls regularization strength.

- Larger \(\alpha\) applies stronger shrinkage.
- Coefficients are generally reduced in magnitude.
- Ridge typically does not force coefficients exactly to zero.
- It can be useful when many features contribute or features are correlated.

The intercept is not penalized by default in scikit-learn's `Ridge`.

## 3. Lasso Regression (L1)

Lasso adds an absolute-coefficient penalty:

\[
J(\boldsymbol{\beta}) =
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
+\alpha\sum_{j=1}^{p}|\beta_j|
\]

- Larger \(\alpha\) applies stronger shrinkage.
- Some coefficients can become exactly zero.
- This can produce a sparse model.
- With correlated features, which feature remains selected can be unstable.

The intercept is not penalized by default in scikit-learn's `Lasso`.

## 4. Ridge vs Lasso

| Property | Ridge | Lasso |
|---|---|---|
| Penalty | Squared coefficients (L2) | Absolute coefficients (L1) |
| Shrinks coefficients | Yes | Yes |
| Can set coefficients exactly to zero | Generally no | Yes |
| Useful for | Distributed signal across features | Sparse solutions when appropriate |
| Feature selection | Not usually | Can perform implicit feature selection |

Neither method is always superior. Select regularization strength using validation or cross-validation.

## 5. Why Scaling Matters

Regularization penalizes coefficient magnitudes. If features have very different scales, their coefficient magnitudes are not directly comparable.

Scaling numerical features before fitting is therefore generally important for Ridge and Lasso.

Fit the scaler on training data only. A scikit-learn pipeline helps prevent leakage.

## 6. Implementation: Compare Linear, Ridge, and Lasso

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

X, y = make_regression(
    n_samples=500,
    n_features=20,
    n_informative=8,
    noise=25,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0),
    ),
    "Lasso": make_pipeline(
        StandardScaler(),
        Lasso(alpha=1.0, max_iter=10000),
    ),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n{name}")
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("RMSE:", root_mean_squared_error(y_test, predictions))
    print("R²:", r2_score(y_test, predictions))
```

The chosen alpha values are illustrative, not guaranteed to be optimal. For a fair model-selection process, tune them using cross-validation on the training data.

## 7. Inspecting Lasso Coefficients

```python
lasso = models["Lasso"]
lasso.fit(X_train, y_train)

coefficients = lasso.named_steps["lasso"].coef_

print("Total features:", len(coefficients))
print("Zero coefficients:", (coefficients == 0).sum())
```

This shows how many coefficients Lasso has set exactly to zero in this fitted model. It does not guarantee that the selected features are causally important or stable across different datasets.

## 8. Choosing Alpha

- Very small alpha: weak regularization; the model behaves more like ordinary Linear Regression.
- Moderate alpha: may improve generalization.
- Very large alpha: can shrink coefficients excessively and cause underfitting.

Use cross-validation to choose alpha. Keep the test set untouched until final evaluation.

## Common Mistakes

- Assuming regularization always improves test performance.
- Comparing Ridge and Lasso without appropriate scaling.
- Tuning alpha repeatedly on the test set.
- Assuming a nonzero coefficient proves feature importance or causation.
- Treating Lasso's feature selection as guaranteed to be stable.

## Quick Revision

- Ridge uses an L2 penalty.
- Lasso uses an L1 penalty.
- Alpha controls regularization strength.
- Scaling is generally important for penalized regression.
- Choose hyperparameters using validation or cross-validation.

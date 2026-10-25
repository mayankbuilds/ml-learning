# Linear Regression

## 1. Overview

Linear Regression is a supervised learning algorithm that models the relationship between input features and a continuous target using a linear function of the model parameters.

For one feature:

$$
\hat{y} = b_0 + b_1x
$$

For multiple features:

$$
\hat{y} = b_0 + b_1x_1 + \cdots + b_px_p
$$

Where:

- $\hat{y}$: predicted value
- $b_0$: intercept
- $b_i$: coefficient associated with feature $x_i$
- $p$: number of features

The fitted coefficients are commonly estimated using Ordinary Least Squares (OLS), which minimizes the sum of squared residuals.

## 2. Residuals and the Objective

A residual is the difference between an observed target and its prediction:

$$
e_i = y_i - \hat{y}_i
$$

OLS minimizes the residual sum of squares:

$$
\text{RSS} = \sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

Squaring penalizes large errors more heavily than small errors.

## 3. Interpreting Coefficients

A coefficient describes the change in the model's predicted target for a one-unit increase in its corresponding feature, **holding the other features constant**.

Example:

$$
\widehat{\text{Price}} = 10 + 0.04(\text{Area})
$$

If area is measured in square feet and price in lakh, the model predicts an increase of 0.04 lakh for each additional square foot.

This is a model-based association, not automatically a causal relationship.

For correlated features, coefficient estimates can be unstable. Coefficients should also be interpreted in the context of feature units and preprocessing.

## 4. Implementation with scikit-learn

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

X, y = make_regression(
    n_samples=500,
    n_features=3,
    noise=20,
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

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("MAE:", mean_absolute_error(y_test, predictions))
print("RMSE:", root_mean_squared_error(y_test, predictions))
print("R²:", r2_score(y_test, predictions))
```

### Important API Methods and Attributes

- `fit(X_train, y_train)`: estimates the model coefficients.
- `predict(X_test)`: generates predictions.
- `intercept_`: fitted intercept.
- `coef_`: fitted feature coefficients.

The generated dataset is for learning and experimentation, not evidence of performance on a real-world problem.

## 5. Key Assumptions and Diagnostics

Ordinary Least Squares does not require every assumption below merely to produce predictions. However, assumptions matter for interpretation and for statistical inference.

### Linearity

The expected target should be adequately represented by a linear combination of the features and coefficients.

Inspect residual plots to identify systematic patterns. Nonlinear feature transformations may help when relationships are nonlinear.

### Independence of Errors

Errors should be independent for many standard inferential procedures. Time series and grouped data may violate this assumption.

A random train/test split can also be inappropriate when observations are dependent over time or within groups.

### Constant Error Variance (Homoscedasticity)

The conditional variance of errors should be approximately constant for standard OLS inference.

A funnel-shaped residual plot can indicate heteroscedasticity.

### Residual Normality

Residuals being approximately normal is relevant to exact small-sample inference under standard assumptions. It is not required simply to calculate OLS predictions.

### Multicollinearity

Strong correlations among features can make individual coefficients unstable and difficult to interpret. Multicollinearity does not automatically make predictions inaccurate.

## 6. Limitations

- Sensitive to influential observations and outliers.
- Cannot represent complex nonlinear relationships without suitable feature transformations.
- Extrapolation beyond the observed feature range can be unreliable.
- Correlated features can make coefficient interpretation difficult.
- A good training fit does not guarantee good generalization.

## Common Mistakes

- Interpreting association as causation.
- Assuming a coefficient always describes a causal effect.
- Evaluating a model on its training data alone.
- Assuming high R² guarantees accurate predictions for every observation.
- Treating all assumptions as mandatory for the algorithm to run.

## Quick Revision

- Linear Regression predicts a continuous target.
- OLS minimizes the sum of squared residuals.
- Coefficients describe conditional changes in predictions.
- Residual diagnostics help identify model limitations.
- Use held-out data and suitable metrics to evaluate generalization.

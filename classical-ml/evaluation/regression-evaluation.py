"""Evaluate a regression model using multiple metrics."""

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# This dataset may require a download on the first run.
X, y = fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = make_pipeline(
    StandardScaler(),
    LinearRegression(),
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("MAE:", round(mean_absolute_error(y_test, predictions), 4))
print("MSE:", round(mean_squared_error(y_test, predictions), 4))
print("RMSE:", round(root_mean_squared_error(y_test, predictions), 4))
print("R²:", round(r2_score(y_test, predictions), 4))

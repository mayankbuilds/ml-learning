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
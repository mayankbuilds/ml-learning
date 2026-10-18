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
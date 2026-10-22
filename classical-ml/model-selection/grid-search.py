
"""Tune a Decision Tree using GridSearchCV."""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

param_grid = {
    "max_depth": [2, 4, 6, None],
    "min_samples_split": [2, 5, 10],
}

search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)

search.fit(X_train, y_train)

print("Combinations tested:", len(search.cv_results_["params"]))
print("Best parameters:", search.best_params_)
print(f"Best CV F1: {search.best_score_:.4f}")

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test F1: {f1_score(y_test, y_pred):.4f}")

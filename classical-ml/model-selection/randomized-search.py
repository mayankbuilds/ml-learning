
"""Tune a Decision Tree using RandomizedSearchCV."""

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

param_distributions = {
    "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
    "min_samples_split": [2, 5, 10, 15, 20],
    "min_samples_leaf": [1, 2, 4, 6, 8],
    "criterion": ["gini", "entropy", "log_loss"],
}

search = RandomizedSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring="f1",
    random_state=42,
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

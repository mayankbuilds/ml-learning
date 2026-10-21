"""Evaluate Logistic Regression using stratified cross-validation."""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000),
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy",
)

print("Fold accuracies:", scores.round(4))
print("Mean accuracy:", round(scores.mean(), 4))
print("Standard deviation:", round(scores.std(), 4))

"""Compare classifiers using identical stratified CV folds."""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    ),
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(random_state=42),
    ),
}

rows = []

for name, model in models.items():
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "f1", "roc_auc"],
        n_jobs=-1,
    )

    rows.append(
        {
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "F1": scores["test_f1"].mean(),
            "ROC-AUC": scores["test_roc_auc"].mean(),
        }
    )

results = pd.DataFrame(rows).sort_values(
    by="F1",
    ascending=False,
)

print(
    results.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "F1": "{:.4f}".format,
            "ROC-AUC": "{:.4f}".format,
        },
    )
)

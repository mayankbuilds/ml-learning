import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

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

results = []

for name, model in models.items():
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "f1", "roc_auc"],
        n_jobs=-1,
    )

    results.append({
        "Model": name,
        "CV Accuracy": scores["test_accuracy"].mean(),
        "CV F1": scores["test_f1"].mean(),
        "CV ROC-AUC": scores["test_roc_auc"].mean(),
    })

comparison = pd.DataFrame(results).sort_values(
    by="CV F1",
    ascending=False,
)

print("Cross-validation comparison:")
print(
    comparison.to_string(
        index=False,
        formatters={
            "CV Accuracy": "{:.4f}".format,
            "CV F1": "{:.4f}".format,
            "CV ROC-AUC": "{:.4f}".format,
        },
    )
)

best_name = comparison.iloc[0]["Model"]
best_model = models[best_name]

best_model.fit(X_train, y_train)
test_predictions = best_model.predict(X_test)

print(f"\nSelected model by CV F1: {best_name}")
print(
    f"Test Accuracy: "
    f"{accuracy_score(y_test, test_predictions):.4f}"
)
print(
    f"Test F1: "
    f"{f1_score(y_test, test_predictions):.4f}"
)
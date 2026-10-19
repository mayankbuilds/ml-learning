"""End-to-end preprocessing with ColumnTransformer and Pipeline."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# A small, built-in dataset avoids external downloads.
# All features are numeric; a categorical feature is added for demonstration.
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer(as_frame=True)
X = dataset.data.iloc[:, :5].copy()
y = dataset.target

X["measurement_group"] = pd.cut(
    X.iloc[:, 0],
    bins=3,
    labels=["Low", "Medium", "High"],
).astype(object)

# Add a few missing values to demonstrate imputation.
X.loc[X.index[0], X.columns[1]] = float("nan")
X.loc[X.index[1], "measurement_group"] = None

numeric_columns = X.select_dtypes(include="number").columns.tolist()
categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Numeric features:", len(numeric_columns))
print("Categorical features:", len(categorical_columns))
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "datasets" / "penguins.csv"
MODEL_PATH = Path(__file__).resolve().parent / "penguin-classifier.joblib"

df = pd.read_csv(DATA_PATH).dropna(subset=["species"])
X = df.drop(columns=["species", "year"])
y = df["species"]

numeric = ["bill_length_mm", "bill_depth_mm",
           "flipper_length_mm", "body_mass_g"]
categorical = ["island", "sex"]

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Dataset rows:", len(df))
print("Train samples:", len(X_train))
print("Test samples:", len(X_test))
print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")
print("\nClassification report:")
print(classification_report(y_test, predictions))

joblib.dump(model, MODEL_PATH)
print("Saved model:", MODEL_PATH.name)

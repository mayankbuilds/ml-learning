from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the dataset
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "datasets" / "penguins.csv"
MODEL_PATH = Path(__file__).resolve().parent / "penguins-model.joblib"

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["species"])

X = df.drop(columns=["species", "year"])
y = df["species"]

numeric_features = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
categorical_features = ["island", "sex"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model.fit(X_train, y_train)
original_predictions = model.predict(X_test)

# Save and reload the complete pipeline
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
loaded_model = joblib.load(MODEL_PATH)

loaded_predictions = loaded_model.predict(X_test)
predictions_match = np.array_equal(
    original_predictions, loaded_predictions
)

print("Dataset rows:", len(df))
print("Training samples:", len(X_train))
print("Test samples:", len(X_test))
print(f"Test accuracy: {accuracy_score(y_test, loaded_predictions):.4f}")
print("Saved model:", MODEL_PATH.name)
print("Predictions match:", predictions_match)

if not predictions_match:
    raise RuntimeError("Loaded model predictions differ from original predictions.")

print("Model persistence check passed.")

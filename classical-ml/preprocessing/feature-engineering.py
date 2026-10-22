from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "datasets" / "penguins.csv"

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["species"]).copy()

print("Original shape:", df.shape)

# 1. Feature creation: body mass in kilograms
df["body_mass_kg"] = df["body_mass_g"] / 1000

# 2. Feature creation: bill length-to-depth ratio
df["bill_length_depth_ratio"] = (
    df["bill_length_mm"] / df["bill_depth_mm"]
)

# 3. Handle invalid values created by missing or zero denominators
df["bill_length_depth_ratio"] = (
    df["bill_length_depth_ratio"].replace([float("inf"), -float("inf")], float("nan"))
)

print("\nNew features:")
print(df[["body_mass_kg", "bill_length_depth_ratio"]].head())

# 4. Separate target from input features
X = df.drop(columns=["species", "year"])
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\nTraining rows:", len(X_train))
print("Test rows:", len(X_test))
print("Target kept separate:", "species" not in X_train.columns)
print("Feature engineering demonstration passed.")
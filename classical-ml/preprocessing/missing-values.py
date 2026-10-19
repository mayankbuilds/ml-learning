"""Demonstrate missing-value imputation."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.DataFrame({
    "age": [21, np.nan, 25, 30, np.nan, 35],
    "income": [30000, 42000, np.nan, 50000, 48000, 60000],
    "city": ["Delhi", "Mumbai", None, "Delhi", "Mumbai", "Pune"],
})

print("Original data:")
print(df)
print("\nMissing values per column:")
print(df.isna().sum())

# Impute numerical columns with their medians.
numeric_columns = ["age", "income"]
numeric_imputer = SimpleImputer(strategy="median")
df[numeric_columns] = numeric_imputer.fit_transform(
    df[numeric_columns]
)

# Impute categorical values with the most frequent category.
categorical_imputer = SimpleImputer(strategy="most_frequent")
df[["city"]] = categorical_imputer.fit_transform(df[["city"]])

print("\nData after imputation:")
print(df)
print("\nMissing values remaining:")
print(df.isna().sum())

"""Compare StandardScaler and MinMaxScaler."""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.DataFrame({
    "age": [18, 25, 32, 45, 60],
    "income": [20000, 35000, 50000, 100000, 200000],
})

print("Original data:")
print(df)

# Demonstration only: fit_transform on this entire sample.
standard_scaler = StandardScaler()
standard_scaled = standard_scaler.fit_transform(df)

standard_df = pd.DataFrame(
    standard_scaled,
    columns=df.columns,
)

print("\nStandardScaler output:")
print(standard_df.round(3))
print("\nStandardized means:")
print(standard_df.mean().round(3))

minmax_scaler = MinMaxScaler()
minmax_scaled = minmax_scaler.fit_transform(df)

minmax_df = pd.DataFrame(
    minmax_scaled,
    columns=df.columns,
)

print("\nMinMaxScaler output:")
print(minmax_df.round(3))
print("\nMinMax ranges:")
print(
    (minmax_df.min().round(3), minmax_df.max().round(3))
)

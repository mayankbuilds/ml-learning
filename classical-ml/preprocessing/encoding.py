"""Demonstrate one-hot and ordinal encoding."""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

df = pd.DataFrame({
    "city": ["Delhi", "Mumbai", "Pune", "Delhi", "Mumbai"],
    "size": ["Medium", "Small", "Large", "Small", "Medium"],
})

print("Original data:")
print(df)

# Nominal feature: city.
city_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
)
city_encoded = city_encoder.fit_transform(df[["city"]])

city_df = pd.DataFrame(
    city_encoded,
    columns=city_encoder.get_feature_names_out(["city"]),
)

print("\nOne-hot encoded city:")
print(city_df)

# Ordinal feature: size.
size_encoder = OrdinalEncoder(
    categories=[["Small", "Medium", "Large"]]
)
df["size_encoded"] = size_encoder.fit_transform(df[["size"]]).astype(int)

print("\nOrdinal encoded size:")
print(df[["size", "size_encoded"]])

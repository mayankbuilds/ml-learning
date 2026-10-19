# Categorical Encoding

## 1. What Is Encoding?

Machine learning models generally require numerical inputs. Encoding converts categorical features into numerical representations.

Examples of categorical features include city, color, product type, and education level.

## 2. One-Hot Encoding

One-hot encoding creates a separate binary column for each category.

Example:

| City | City_Delhi | City_Mumbai | City_Pune |
|---|---:|---:|---:|
| Delhi | 1 | 0 | 0 |
| Mumbai | 0 | 1 | 0 |
| Pune | 0 | 0 | 1 |

Use `OneHotEncoder` for nominal categories that have no natural order.

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
)
```

`handle_unknown="ignore"` allows the encoder to process categories not seen during fitting.

## 3. Ordinal Encoding

Ordinal encoding maps categories to integer values.

For example:

- Low → 0
- Medium → 1
- High → 2

Use `OrdinalEncoder` when the categories have a meaningful order.

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(
    categories=[["Low", "Medium", "High"]]
)
```

Do not assign arbitrary numeric order to nominal categories such as cities; the numbers could imply a relationship that does not exist.

## 4. Important Considerations

- Fit encoders on training data only.
- Use `OneHotEncoder` for nominal categories in most standard tabular workflows.
- Use ordinal encoding when a meaningful category order exists.
- Consider feature dimensionality when a column has many unique categories.
- Choose encoding based on the feature's meaning and the model.

## 5. Key Takeaway

Encoding is not just converting text into numbers. The representation should preserve the meaning of the categorical feature without introducing a misleading order.

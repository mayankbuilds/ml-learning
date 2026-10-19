# Handling Missing Values

## 1. Why Missing Values Matter

Real-world datasets may contain missing entries because of incomplete forms, measurement failures, or unavailable information.

Many scikit-learn estimators cannot directly handle arbitrary missing values, so missing data may need to be imputed or otherwise handled.

## 2. Common Strategies

- **Mean imputation:** Replace missing numerical values with the training-set mean. Sensitive to outliers.
- **Median imputation:** Replace missing numerical values with the training-set median. Often more robust to outliers.
- **Most-frequent imputation:** Replace missing values with the most common observed value. Useful for categorical features.
- **Constant imputation:** Replace missing entries with a specified value such as `"Unknown"` or `0`, when appropriate.
- **Dropping rows or columns:** Consider only when the amount and pattern of missing data justify it.

## 3. Scikit-learn Implementation

Use `SimpleImputer`:

```python
from sklearn.impute import SimpleImputer

numeric_imputer = SimpleImputer(strategy="median")
categorical_imputer = SimpleImputer(
    strategy="most_frequent"
)
```

Fit the imputers on training data and use the fitted imputers to transform other datasets.

## 4. Important Considerations

- Inspect how much data is missing and which columns are affected.
- Mean imputation can be distorted by outliers.
- Median imputation is often useful for skewed numerical data.
- Most-frequent imputation may be inappropriate when the category distribution has no meaningful dominant value.
- Imputation can hide information about why values are missing; missingness indicators may help in some problems.

## 5. Key Takeaway

Choose an imputation strategy based on the feature, data distribution, and problem context. Never calculate imputation statistics from the full dataset before splitting it into training and test sets.

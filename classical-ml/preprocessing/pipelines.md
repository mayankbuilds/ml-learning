# Pipelines and ColumnTransformer

## 1. Why Pipelines?

A pipeline combines preprocessing transformations and a machine learning estimator into a single workflow.

Benefits:

- Consistent transformations during training and prediction.
- Less repetitive code.
- Easier cross-validation and hyperparameter tuning.
- Helps prevent data leakage when fitted correctly.

## 2. Pipeline

`Pipeline` chains steps together. Intermediate steps transform data, and the final step is usually an estimator.

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])
```

The example is for numerical features; use appropriate encoding for categorical features.

## 3. ColumnTransformer

`ColumnTransformer` applies different transformations to different columns.

For example:

- Numerical columns: median imputation and standardization.
- Categorical columns: most-frequent imputation and one-hot encoding.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

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
```

## 4. Complete Workflow

A complete estimator can combine the preprocessor and model:

```python
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Fit the complete pipeline on training data only. The pipeline learns its preprocessing parameters from that training data.

## 5. Key Takeaway

Use `Pipeline`
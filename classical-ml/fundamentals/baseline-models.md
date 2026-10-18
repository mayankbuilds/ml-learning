# Baseline Models

## 1. What Is a Baseline?

A baseline is a simple reference method used to judge whether a more sophisticated model provides useful improvement.

A complex model should not be considered successful merely because it produces predictions. Its performance should be compared against a meaningful reference.

## 2. Regression Baselines

A simple regression baseline can predict a constant value for every observation.

Common strategies:

- Mean: predict the mean of the training targets.
- Median: predict the median of the training targets.

The median baseline can be useful when absolute error matters and the target contains extreme values.

```python
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=500,
    n_features=4,
    noise=20,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

predictions = baseline.predict(X_test)

print("Baseline MAE:", mean_absolute_error(y_test, predictions))
print("Baseline RMSE:", root_mean_squared_error(y_test, predictions))
```

The baseline learns its constant from the training targets. It does not learn relationships between the features and target.

## 3. Classification Baselines

A common classification baseline predicts the most frequent class found in the training targets.

```python
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    weights=[0.8, 0.2],
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

predictions = baseline.predict(X_test)

print("Baseline accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))
```

In this imbalanced dataset, a model that always predicts the majority class can achieve high accuracy while failing to identify minority-class examples.

Therefore, accuracy alone may not be sufficient. Consider precision, recall, F1-score, the confusion matrix, and the consequences of false positives and false negatives.

## 4. Comparing a Baseline with a Real Model

The following example compares a majority-class baseline against Logistic Regression.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

models = {
    "Baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    ),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("F1-score:", f1_score(y_test, predictions, zero_division=0))
```

This code illustrates metric comparison. For actual model selection, compare candidates on validation data or through cross-validation, then evaluate the selected approach on the held-out test set.

## 5. What Makes a Good Baseline?

A useful baseline should be:

- Simple to understand and reproduce.
- Appropriate for the prediction task.
- Evaluated with the same data split and relevant metrics as candidate models.
- Free from information leakage.
- Strong enough to expose whether the more complex approach adds value.

The simplest possible baseline is not always the only baseline worth considering. Domain knowledge may suggest a more meaningful reference, such as predicting the previous observation for certain time-series tasks.

## 6. Interpreting Results

If a sophisticated model performs worse than a baseline, investigate:

1. Whether preprocessing is correct.
2. Whether the model is suitable for the data.
3. Whether hyperparameters are reasonable.
4. Whether the evaluation metric matches the objective.
5. Whether the dataset is too small, noisy, or imbalanced.
6. Whether the baseline or model has been evaluated incorrectly.

A baseline is a reference point, not a guarantee that every trained model must outperform it.

## Common Mistakes

- Skipping the baseline and reporting a model score without context.
- Comparing models using different data splits.
- Using accuracy alone for an imbalanced classification problem.
- Assuming a high score automatically means a useful model.
- Selecting a model using repeated test-set experiments.

## Quick Revision

- `DummyRegressor` provides simple regression baselines.
- `DummyClassifier` provides simple classification baselines.
- Regression and classification require task-appropriate metrics.
- Baselines help establish whether a model adds predictive value.
- Model comparisons should use fair validation procedures.

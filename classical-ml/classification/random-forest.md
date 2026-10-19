# Random Forest for Classification

## 1. Overview

Random Forest is an ensemble learning algorithm that combines predictions from many Decision Trees.

It introduces randomness through bootstrap samples and feature selection at splits. Combining trees often reduces the variance of an individual tree.

## 2. Core Concepts

- **Bootstrap sampling:** Training samples are randomly sampled with replacement for individual trees by default.
- **Feature randomness:** A subset of features is considered when choosing splits.
- **Ensemble prediction:** Classification combines tree predictions, typically by voting or aggregated class probabilities.

A forest can capture nonlinear relationships and feature interactions.

## 3. Implementation

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

X, y = make_classification(
    n_samples=600,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("F1-score:", f1_score(y_test, predictions))
```

## 4. Important Hyperparameters

- `n_estimators`: number of trees.
- `max_depth`: maximum depth of each tree.
- `min_samples_leaf`: minimum samples per leaf.
- `max_features`: number or proportion of features considered at each split.
- `class_weight`: can adjust the treatment of class frequencies.
- `n_jobs`: controls parallel execution.

Tune important parameters using cross-validation rather than selecting them based on test results.

## 5. Feature Importance

```python
for index, importance in enumerate(model.feature_importances_):
    print(f"Feature {index}: {importance:.3f}")
```

Impurity-based feature importance can favor variables with many possible split points. Permutation importance on held-out data is an alternative, though correlated features can complicate interpretation.

## 6. Decision Tree vs Random Forest

| Property | Decision Tree | Random Forest |
|---|---|---|
| Model structure | Single tree | Ensemble of trees |
| Interpretability | Easier for small trees | Harder to interpret globally |
| Variance | Can be high | Often reduced |
| Computation | Usually lower | Usually higher |
| Feature scaling | Generally unnecessary | Generally unnecessary |

A Random Forest is not guaranteed to outperform every individual tree or every other algorithm on every dataset.

## Advantages

- Captures nonlinear relationships.
- Often provides a strong tabular-data baseline.
- Less sensitive to individual training observations than a single deep tree.
- Supports class probabilities and feature importance.

## Limitations

- Can require more memory and computation.
- Less interpretable than a small tree.
- Can be large for deployment.
- Probability estimates may need calibration for some applications.

## Common Mistakes

- Assuming more trees always produce a meaningful performance gain.
- Ignoring class imbalance.
- Using test data repeatedly for hyperparameter tuning.
- Treating feature importance as causal evidence.

## Quick Revision

- Random Forest combines multiple randomized Decision Trees.
- `n_estimators` controls forest size.
- `min_samples_leaf` and `max_depth` help control tree complexity.
- Scaling is generally not required.

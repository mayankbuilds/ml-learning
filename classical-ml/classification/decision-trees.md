# Decision Trees for Classification

## 1. Overview

A Decision Tree predicts a class by applying a sequence of feature-based rules.

Each internal node tests a feature, each branch represents an outcome, and each leaf produces a class prediction or class probabilities.

Trees can model nonlinear decision boundaries and feature interactions without requiring feature scaling.

## 2. Splitting Criteria

A tree selects splits that improve class purity according to a criterion.

### Gini Impurity

\[
Gini=1-\sum_{k=1}^{K}p_k^2
\]

Here, \(p_k\) is the fraction of observations belonging to class \(k\) at a node.

Gini is zero when all observations at the node belong to one class.

### Entropy

\[
H=-\sum_{k=1}^{K}p_k\log_2(p_k)
\]

Entropy is also zero for a pure node. Information gain measures the reduction in entropy after a split.

In scikit-learn, `criterion="gini"` and `criterion="entropy"` are supported options.

## 3. Implementation

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=5,
    criterion="gini",
    random_state=42,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("F1-score:", f1_score(y_test, predictions))
```

## 4. Controlling Complexity

Useful hyperparameters:

- `max_depth`: maximum tree depth.
- `min_samples_split`: minimum samples required to split an internal node.
- `min_samples_leaf`: minimum samples required in a leaf.
- `criterion`: split-quality measure.

A deeper tree can fit more complex patterns but may overfit. Choose complexity using validation or cross-validation.

## 5. Feature Importance

```python
for index, importance in enumerate(model.feature_importances_):
    print(f"Feature {index}: {importance:.3f}")
```

Impurity-based importance indicates how much features contribute to splits in the fitted tree. It can be biased toward certain features and does not establish causation.

## Advantages

- Easy to visualize for small trees.
- Captures nonlinear relationships and interactions.
- Does not generally require feature scaling.
- Can handle numerical and encoded categorical features.

## Limitations

- Deep trees can overfit.
- Small data changes can produce different trees.
- Individual tree predictions can have high variance.
- Impurity-based feature importance can be misleading.

## Common Mistakes

- Allowing unlimited depth without checking generalization.
- Evaluating only on training data.
- Treating feature importance as causal evidence.
- Forgetting to stratify a classification split when appropriate.

## Quick Revision

- Gini and entropy measure node impurity.
- Splits aim to improve class purity.
- `max_depth` and `min_samples_leaf` control complexity.
- Scaling is usually unnecessary for tree-based models.

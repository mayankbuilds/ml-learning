# Classification Evaluation Metrics

Classification metrics measure how well predicted class labels or scores agree with the actual labels.

## 1. Confusion Matrix

For binary classification:

| | Predicted positive | Predicted negative |
|---|---|---|
| Actual positive | True Positive (TP) | False Negative (FN) |
| Actual negative | False Positive (FP) | True Negative (TN) |

- **TP:** Positive case correctly identified.
- **TN:** Negative case correctly identified.
- **FP:** Negative case incorrectly predicted positive.
- **FN:** Positive case incorrectly predicted negative.

The cost of FP versus FN depends on the application.

## 2. Accuracy

$$
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$

Accuracy measures the fraction of predictions that are correct.

It can be misleading when classes are imbalanced. For example, predicting every case as negative achieves high accuracy when almost all cases are negative.

## 3. Precision

$$
\text{Precision}=\frac{TP}{TP+FP}
$$

Precision measures how many predicted positives are actually positive.

Use it when false positive predictions are particularly costly.

## 4. Recall

$$
\text{Recall}=\frac{TP}{TP+FN}
$$

Recall measures how many actual positives the model identifies.

Use it when missing positive cases is particularly costly.

## 5. F1-score

$$
F_1=2\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}
$$

F1 is the harmonic mean of precision and recall. It is useful when both matter, but it ignores true negatives and does not directly express business cost.

## 6. ROC-AUC

ROC-AUC measures how well a model's scores rank positive cases above negative cases across thresholds.

- AUC of 1: perfect ranking on the evaluated data.
- AUC around 0.5: ranking comparable to random ordering for a balanced interpretation of positive and negative labels.
- AUC below 0.5: ranking is worse than the reversed ranking.

ROC-AUC evaluates ranking, not a single chosen threshold. For highly imbalanced data, also consider precision-recall analysis.

## 7. Implementation

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

X, y = make_classification(
    n_samples=600,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    weights=[0.8, 0.2],
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000),
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, zero_division=0))
print("Recall:", recall_score(y_test, predictions, zero_division=0))
print("F1-score:", f1_score(y_test, predictions, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, probabilities))
print(classification_report(y_test, predictions, zero_division=0))
```

ROC-AUC uses predicted scores/probabilities, not just hard class labels.

## 8. Choosing Metrics

| Situation | Useful metrics to inspect |
|---|---|
| Balanced classes and similar error costs | Accuracy, F1 |
| False positives are costly | Precision |
| False negatives are costly | Recall |
| Need a balance of precision and recall | F1 |
| Need threshold-independent ranking evaluation | ROC-AUC |
| Highly imbalanced positive class | Precision, recall, F1, precision-recall curve |

No single metric is best for every problem. Select metrics based on the actual objective and the cost of errors.

## Common Mistakes

- Reporting accuracy alone on imbalanced data.
- Confusing precision with recall.
- Calculating ROC-AUC from predicted labels when probability scores are available.
- Changing the threshold without considering precision-recall trade-offs.
- Reporting test-set metrics without explaining the evaluation split.

## Quick Revision

- Precision focuses on predicted positives.
- Recall focuses on actual positives.
- F1 balances precision and recall.
- ROC-AUC evaluates score ranking across thresholds.
- The confusion matrix shows the types of classification errors.

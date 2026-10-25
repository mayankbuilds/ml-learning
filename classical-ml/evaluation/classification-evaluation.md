# Classification Evaluation

## 1. Confusion Matrix

A confusion matrix summarizes predicted classes against actual classes.

For binary classification:

- **True Positive (TP):** Positive case correctly predicted.
- **True Negative (TN):** Negative case correctly predicted.
- **False Positive (FP):** Negative case incorrectly predicted positive.
- **False Negative (FN):** Positive case incorrectly predicted negative.

## 2. Important Metrics

### Accuracy

The fraction of all predictions that are correct.

$$
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$

Accuracy can be misleading when classes are imbalanced.

### Precision

The fraction of predicted positive cases that are actually positive.

$$
\text{Precision}=\frac{TP}{TP+FP}
$$

High precision means fewer false positives among predicted positives.

### Recall

The fraction of actual positive cases correctly identified.

$$
\text{Recall}=\frac{TP}{TP+FN}
$$

High recall means fewer positive cases are missed.

### F1-Score

The harmonic mean of precision and recall.

$$
F_1=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}
$$

F1 is useful when both precision and recall matter.

### ROC-AUC

ROC-AUC measures how well a model ranks positive examples above negative examples across classification thresholds.

- `1.0`: Perfect ranking.
- `0.5`: Random-ranking baseline in the standard binary setting.
- Below `0.5`: Ranking may be reversed or otherwise poor.

ROC-AUC is not the same as accuracy at a particular threshold.

## 3. Scikit-Learn Metrics

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
```

For ROC-AUC, pass continuous scores or probabilities for the positive class, not hard class predictions.

## 4. Metric Selection

- Balanced classes with similar error costs: accuracy may be informative.
- False positives are costly: focus on precision.
- False negatives are costly: focus on recall.
- Both precision and recall matter: consider F1.
- Ranking across thresholds matters: consider ROC-AUC.

## 5. Key Takeaway

Choose metrics based on the cost of different errors and the objective of the problem. Always inspect the confusion matrix alongside summary metrics.

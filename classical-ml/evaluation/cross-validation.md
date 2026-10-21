# Cross-Validation

## 1. What Is Cross-Validation?

Cross-validation evaluates a model across multiple train/validation splits instead of relying on a single split.

It helps estimate performance variability and supports model selection, especially when the dataset is relatively small.

## 2. K-Fold Cross-Validation

In K-fold cross-validation:

1. Divide the dataset into K folds.
2. Train on K-1 folds.
3. Evaluate on the remaining fold.
4. Repeat until every fold has served as the validation fold.
5. Summarize the scores across all folds.

With 5-fold cross-validation, the model is fitted and evaluated five times.

## 3. Scikit-Learn Example

```python id="c92wzq"
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000),
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy",
)

print("Fold scores:", scores)
print("Mean accuracy:", scores.mean())
print("Standard deviation:", scores.std())
```

## 4. Stratified K-Fold

`StratifiedKFold` attempts to preserve class proportions across folds. It is commonly useful for classification, especially when classes are imbalanced.

For regression, ordinary K-fold or a suitable task-specific splitting strategy is generally used.

## 5. Important Considerations

- Keep the final test set separate from model selection.
- Put preprocessing inside the pipeline to prevent leakage between folds.
- Use the same cross-validation strategy when comparing candidate models.
- The mean score summarizes performance; the standard deviation indicates variation across folds.
- Time-dependent data often requires time-aware validation rather than random folds.

## 6. Key Takeaway

Cross-validation provides a more robust model-selection estimate than a single validation split, but it does not eliminate data leakage or guarantee future performance.

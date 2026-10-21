# Model Comparison

## 1. Purpose

Model comparison evaluates candidate algorithms under consistent conditions to help select a model that generalizes well.

## 2. Fair Comparison Rules

- Use the same training data and evaluation strategy for all candidates.
- Use the same cross-validation splits when comparing models.
- Fit preprocessing separately within each training fold using pipelines.
- Select metrics based on the problem and its error costs.
- Avoid tuning repeatedly against the final test set.
- Consider training time, prediction latency, interpretability, and model complexity alongside predictive performance.

## 3. Cross-Validation Comparison

`cross_validate` can evaluate multiple metrics for each candidate.

```python id="psu5bq"
from sklearn.model_selection import cross_validate, StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=["accuracy", "f1", "roc_auc"],
)
```

For regression, use appropriate metrics such as MAE, RMSE, and R² instead.

## 4. Selecting a Model

A model with the highest score is not always the best practical choice.

Consider:

- Whether its performance is consistent across folds.
- Which errors it makes.
- Whether its complexity is justified by the improvement.
- Whether its preprocessing and inference requirements fit deployment constraints.

## 5. Final Test Evaluation

Use cross-validation or validation data for model selection. After selecting the model and finalizing its configuration, evaluate it once on the held-out test set for a less biased final estimate.

## 6. Key Takeaway

A fair comparison changes the model, not the evaluation conditions. Select models using validation evidence, then reserve the test set for the final assessment.

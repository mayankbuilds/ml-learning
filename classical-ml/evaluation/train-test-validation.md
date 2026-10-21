# Training, Validation, and Test Sets

## 1. Purpose

Separate data subsets serve different purposes:

- **Training set:** Fits model parameters.
- **Validation set:** Helps select models, hyperparameters, and decision thresholds.
- **Test set:** Estimates final generalization performance after model selection.

## 2. Typical Workflow

1. Separate the test set.
2. Use training data to fit candidate models.
3. Use validation data or cross-validation to select configurations.
4. Finalize the selected workflow.
5. Evaluate it on the untouched test set.

## 3. Train/Test Split

Scikit-learn provides `train_test_split`.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)
```

For classification, consider `stratify=y` when preserving class proportions is appropriate.

## 4. Common Split Ratios

Examples include 80/20 or 70/15/15 for training, validation, and test sets. The appropriate split depends on dataset size and the evaluation design; there is no universally optimal ratio.

## 5. Important Considerations

- Keep the test set out of model and hyperparameter selection.
- Use stratified splits when class balance matters.
- Use time-aware splits for temporal prediction tasks.
- Fit preprocessing only on training data.
- Small datasets may benefit from cross-validation instead of a separate validation set.

## 6. Key Takeaway

Every data subset should have a clearly defined role. Repeatedly tuning decisions based on test results turns the test set into a validation set and weakens the final estimate.

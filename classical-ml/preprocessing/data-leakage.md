# Data Leakage

## 1. What Is Data Leakage?

Data leakage happens when information that would not legitimately be available during model training influences the training process or evaluation.

Leakage can produce overly optimistic validation or test results that fail to generalize to unseen data.

## 2. Common Types

### Preprocessing Leakage

Fitting a scaler, imputer, encoder, or feature selector on the complete dataset before splitting it.

**Incorrect:**

```python
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

The scaler has used statistics from the future test set.

**Correct:**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Target Leakage

A feature contains information that directly or indirectly reveals the target.

For example, using a post-diagnosis treatment field to predict whether a patient will receive that diagnosis.

### Temporal Leakage

Future information is used to predict a past event. Time-dependent problems may require chronological splitting instead of random splitting.

## 3. How Pipelines Help

A scikit-learn pipeline fitted on training data learns its preprocessing transformations from that training data. Calling `predict()` on test data applies the already-fitted transformations without refitting them.

Pipelines also make cross-validation safer when the entire pipeline is passed to the validation procedure.

## 4. Prevention Checklist

- Split data before fitting preprocessing transformations.
- Fit imputers, encoders, scalers, and feature selectors only on training data.
- Keep the test set separate until final evaluation.
- Check whether features would actually be available at prediction time.
- Use chronological splits when future observations must not influence past predictions.
- Apply cross-validation to the full preprocessing-and-model pipeline.

## 5. Key Takeaway

A high test score is trustworthy only when the evaluation procedure reflects how the model will encounter genuinely unseen data.

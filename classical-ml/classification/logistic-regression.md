# Logistic Regression

## 1. Overview

Logistic Regression is a supervised classification algorithm that models class probabilities using a logistic function.

Despite its name, it is commonly used for classification rather than predicting a continuous target.

For binary classification, it first calculates a linear score:

$$
z=b_0+b_1x_1+\cdots+b_px_p
$$

It then converts that score into a probability using the sigmoid function:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

The output lies between 0 and 1 and can be interpreted as an estimated probability for class 1 when the model is appropriately fitted and calibrated.

## 2. Decision Threshold

A default threshold of 0.5 is commonly used:

- Probability at least 0.5: predict class 1.
- Probability below 0.5: predict class 0.

The threshold is a choice, not a universal rule. Lowering it generally increases positive predictions; raising it generally decreases them.

Choose a threshold according to the costs of false positives and false negatives, using validation data rather than repeatedly tuning against the test set.

## 3. Implementation

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000),
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, predictions))
print("F1-score:", f1_score(y_test, predictions))
print("First five probabilities:", probabilities[:5])
```

Scaling is useful for Logistic Regression because optimization and regularization can be sensitive to feature scales. The pipeline fits the scaler on training data only.

## 4. Changing the Threshold

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score

threshold = 0.35
custom_predictions = (probabilities >= threshold).astype(int)

print(
    "Precision:",
    precision_score(y_test, custom_predictions, zero_division=0),
)
print(
    "Recall:",
    recall_score(y_test, custom_predictions, zero_division=0),
)
```

This illustrates threshold adjustment. Select a threshold using validation data when the application requires one.

## 5. Multiclass Classification

Logistic Regression also supports multiclass classification. In scikit-learn, the default multinomial formulation is used for supported multiclass configurations.

The `predict_proba()` method returns class probabilities for the classes represented by the fitted model.

## 6. Regularization

Scikit-learn's `LogisticRegression` applies regularization by default.

- `C` controls the inverse of regularization strength.
- Smaller `C` means stronger regularization.
- Larger `C` means weaker regularization.

Tune `C` using cross-validation when necessary.

## Advantages

- Efficient baseline for many classification problems.
- Produces probabilities.
- Coefficients can be interpreted, subject to scaling and feature relationships.
- Works well when the decision boundary is reasonably linear in the feature representation.

## Limitations

- A linear decision boundary may not capture complex relationships.
- Correlated features can make coefficient interpretation unstable.
- Probability quality depends on model fit and calibration.
- Scaling is generally helpful for numerical features.

## Common Mistakes

- Calling Logistic Regression a regression model for continuous targets.
- Assuming a 0.5 threshold is always optimal.
- Using accuracy alone on imbalanced data.
- Fitting preprocessing on the complete dataset before splitting.
- Interpreting coefficients as causal effects.

## Quick Revision

- Sigmoid maps a score to a probability.
- `predict()` returns class labels.
- `predict_proba()` returns estimated class probabilities.
- The threshold controls the label decision.
- `C` controls inverse regularization strength.

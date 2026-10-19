# Support Vector Machines (SVM)

## 1. Overview

A Support Vector Machine is a supervised learning algorithm that can classify observations by learning a decision boundary.

The maximum-margin idea seeks a boundary with a large margin between classes. The observations that constrain this boundary are called support vectors.

SVMs can represent nonlinear decision boundaries through kernel functions.

## 2. Margin and Support Vectors

The margin is the distance between the decision boundary and its closest relevant training observations.

A larger margin can help generalization, although actual performance depends on the data, regularization, and model configuration.

The `C` parameter controls the trade-off between margin width and training errors:

- Smaller `C`: stronger regularization; violations are tolerated more.
- Larger `C`: weaker regularization; training violations are penalized more strongly.

## 3. Kernels

Kernels allow SVMs to model nonlinear relationships through similarities between observations.

Common choices in scikit-learn:

- `linear`: linear decision boundary.
- `rbf`: flexible nonlinear boundary.
- `poly`: polynomial relationships.

The RBF kernel's `gamma` parameter controls how strongly individual training observations influence the decision boundary. Very large values can lead to highly localized boundaries and overfitting.

## 4. Implementation

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
    SVC(kernel="rbf", C=1.0, gamma="scale"),
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("F1-score:", f1_score(y_test, predictions))
```

Scaling is generally important for SVM because distances, margins, and kernel calculations depend on feature scales.

## 5. Probability Estimates

`SVC` does not provide probability estimates by default. Setting `probability=True` enables probability estimates but adds computational overhead.

If probabilities are required, evaluate their calibration as well as classification performance.

## 6. Advantages

- Effective in many high-dimensional settings.
- Can model linear and nonlinear decision boundaries.
- The margin-based approach offers useful regularization control.

## 7. Limitations

- Training can be expensive on large datasets.
- Performance depends on kernel and hyperparameter choices.
- Scaling is generally important.
- The fitted model can be difficult to interpret.
- Probability estimates are not automatically calibrated.

## Common Mistakes

- Using unscaled numerical features without considering their ranges.
- Tuning `C` or `gamma` against the test set.
- Assuming the RBF kernel is always better than a linear kernel.
- Treating decision scores as calibrated probabilities.

## Quick Revision

- Support vectors help determine the decision boundary.
- `C` controls regularization strength.
- Kernels allow nonlinear decision boundaries.
- RBF `gamma` controls the locality of influence.
- Scale features and tune hyperparameters using validation or cross-validation.

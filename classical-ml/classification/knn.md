# k-Nearest Neighbors (kNN)

## 1. Overview

k-Nearest Neighbors is a distance-based classification algorithm.

For a new observation, it finds the $k$ closest training observations and predicts a class using their labels, typically through majority voting.

It is a non-parametric, instance-based method: it does not learn a fixed coefficient vector like Logistic Regression. Instead, it retains training data for prediction.

## 2. Distance

Euclidean distance between two observations is:

$$
d(x,z)=\sqrt{\sum_{j=1}^{p}(x_j-z_j)^2}
$$

Other distance metrics are available, depending on the data and estimator configuration.

## 3. Choosing k

- Small $k$: more sensitive to individual observations and noise.
- Large $k$: smoother decisions but can miss local patterns.
- An even $k$ can create voting ties in binary classification.

Choose $k$ using validation or cross-validation.

## 4. Implementation

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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
    KNeighborsClassifier(n_neighbors=5),
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("F1-score:", f1_score(y_test, predictions))
```

Scaling is especially important for distance-based models. Without scaling, features with large numerical ranges can dominate distance calculations.

## 5. Important Hyperparameters

- `n_neighbors`: number of neighbors, $k$.
- `weights`: uniform or distance-based voting.
- `metric`: distance measure.
- `p`: Minkowski distance parameter; `p=2` corresponds to Euclidean distance.

## Advantages

- Simple and intuitive.
- Can model nonlinear decision boundaries.
- Has few fitting assumptions.

## Limitations

- Prediction can be expensive on large datasets.
- Requires storing training observations.
- Sensitive to feature scaling and irrelevant features.
- Performance can degrade in high-dimensional spaces.
- Imbalanced classes can influence neighbor voting.

## Common Mistakes

- Forgetting to scale numerical features.
- Choosing $k$ without validation.
- Using too many irrelevant features.
- Assuming a higher $k$ is always better.

## Quick Revision

- kNN predicts using nearby labeled observations.
- Small $k$ can have high variance.
- Large $k$ can oversmooth decision boundaries.
- Scaling and distance choice matter.

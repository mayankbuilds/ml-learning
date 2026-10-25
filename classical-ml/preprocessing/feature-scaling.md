# Feature Scaling

## 1. What Is Feature Scaling?

Feature scaling transforms numerical features to comparable scales. It can improve optimization and distance calculations when features have very different magnitudes.

Example:

- Age: 18–60
- Annual income: 20,000–200,000

Without scaling, income may dominate distance calculations because its numerical magnitude is much larger.

## 2. Standardization

`StandardScaler` subtracts the training mean and divides by the training standard deviation.

$$
z = \frac{x-\mu}{\sigma}
$$

The transformed training feature typically has mean near 0 and standard deviation near 1.

Standardization does not guarantee that values lie between 0 and 1.

## 3. Min-Max Normalization

`MinMaxScaler` usually transforms training values into the range 0 to 1.

$$
x' = \frac{x-x_{\min}}{x_{\max}-x_{\min}}
$$

This is sensitive to extreme values. Values outside the training range can transform to values below 0 or above 1.

## 4. Scikit-learn Implementation

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
```

Fit each scaler on training data, then use its `transform()` method on validation and test data.

## 5. Which Models Benefit?

Scaling is generally important for:

- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- K-Means clustering
- Gradient-based optimization, including many neural networks

Decision trees and random forests generally do not require feature scaling.

## 6. Important Considerations

- Fit the scaler on training data only.
- Scaling does not make a feature normally distributed.
- Standardization is not automatically better than Min-Max scaling.
- Consider outliers and the requirements of the chosen algorithm.

## 7. Key Takeaway

Choose scaling based on the algorithm and data distribution. Prevent leakage by learning scaling statistics from training data only.

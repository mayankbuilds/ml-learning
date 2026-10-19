"""Demonstrate correct preprocessing and data leakage."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Training data and test data have very different distributions.
X = np.array([
    [10.0], [12.0], [14.0], [16.0], [18.0],
    [20.0], [22.0], [24.0], [26.0], [28.0],
    [30.0], [32.0], [34.0], [36.0], [38.0],
    [40.0], [42.0], [44.0], [46.0], [48.0],
    [50.0], [52.0], [54.0], [56.0], [58.0],
    [60.0], [62.0], [64.0], [66.0], [68.0],
    [70.0], [72.0], [74.0], [76.0], [78.0],
    [80.0], [82.0], [84.0], [86.0], [88.0],
])

# Hold out the final observations to simulate a distribution shift.
X_train, X_test = X[:30], X[30:]

# Incorrect: fit the scaler using both training and test data.
leaky_scaler = StandardScaler()
leaky_scaler.fit(X)
X_test_leaky = leaky_scaler.transform(X_test)

# Correct: learn scaling parameters from training data only.
correct_scaler = StandardScaler()
correct_scaler.fit(X_train)
X_test_correct = correct_scaler.transform(X_test)

print("Training mean:", X_train.mean())
print("Test mean:", X_test.mean())
print("Full dataset mean:", X.mean())

print("\nIncorrect scaler mean:", leaky_scaler.mean_[0])
print("Correct scaler mean:", correct_scaler.mean_[0])

print("\nFirst test value:", X_test[0, 0])
print("Scaled with leakage:", round(X_test_leaky[0, 0], 3))
print("Scaled correctly:", round(X_test_correct[0, 0], 3))

print("\nCorrect scaler learned from training data only:",
      np.isclose(correct_scaler.mean_[0], X_train.mean()))

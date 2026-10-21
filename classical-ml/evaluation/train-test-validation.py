"""Demonstrate training, validation, and test splitting."""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)

# Reserve 20% for the final test set.
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Reserve 25% of the remaining 80% for validation.
# This gives 60% train, 20% validation, and 20% test overall.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val,
)

total = len(X)

print(f"Total samples: {total}")
print(f"Training:   {len(X_train):3} ({len(X_train) / total:.1%})")
print(f"Validation: {len(X_val):3} ({len(X_val) / total:.1%})")
print(f"Test:       {len(X_test):3} ({len(X_test) / total:.1%})")

print("\nTraining class proportions:")
print(y_train.mean().round(3))
print("Validation class proportions:")
print(y_val.mean().round(3))
print("Test class proportions:")
print(y_test.mean().round(3))

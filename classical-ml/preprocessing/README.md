# Data Preprocessing

Data preprocessing prepares raw data for machine learning models by handling missing values, converting categorical features, and transforming numerical features.

## Learning Objectives

- Handle missing numerical and categorical values.
- Encode categorical features appropriately.
- Understand standardization and normalization.
- Build preprocessing pipelines with scikit-learn.
- Prevent data leakage during training and evaluation.

## Topics

1. [Missing Values](missing-values.md)
2. [Categorical Encoding](encoding.md)
3. [Feature Scaling](feature-scaling.md)
4. [Pipelines](pipelines.md)
5. [Data Leakage](data-leakage.md)

## Recommended Order

1. Inspect and handle missing data.
2. Encode categorical columns.
3. Understand when feature scaling is necessary.
4. Combine transformations using pipelines.
5. Apply transformations correctly to training and test data.

## Key Reminder

Fit preprocessing transformations on training data only. Apply the learned transformations to validation, test, and future data without fitting them again on those datasets.

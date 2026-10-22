
# Practical Model Selection

## What is Model Selection?

Model selection is the process of comparing candidate models using an appropriate evaluation strategy and choosing one that best meets the problem's objective.

## Typical Workflow

1. Define the problem and select an appropriate metric.
2. Split data into training and test sets.
3. Define candidate models.
4. Compare models using the same cross-validation strategy.
5. Select the best candidate using the chosen metric.
6. Tune its hyperparameters using training data.
7. Evaluate the final selected model on the untouched test set.

## Important Considerations

- Use the same folds and scoring metric for a fair comparison.
- Scale features for models that are sensitive to feature magnitudes, using a pipeline to prevent data leakage.
- Decision trees and random forests generally do not require feature scaling.
- Accuracy may be misleading on imbalanced datasets; consider precision, recall, F1, or ROC-AUC where appropriate.
- The best model depends on the metric, dataset, computational cost, interpretability, and deployment requirements.

## Cross-Validation vs Test Set

Cross-validation helps compare and tune models using training data. The test set estimates final performance after the model-selection decisions are complete.

Repeatedly selecting models based on test results leaks information from the test set into the development process.

## Important Distinction

The model with the highest cross-validation score is the best candidate under the chosen evaluation setup, not necessarily the best model for every use case.

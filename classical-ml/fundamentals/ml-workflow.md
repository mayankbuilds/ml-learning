# Machine Learning Workflow

## 1. What Is a Machine Learning Problem?

A machine learning problem uses data to learn patterns that can help make predictions or discover structure in data.

In supervised learning, the training dataset contains input features and a known target.

- **Features (`X`):** Input variables used by the model.
- **Target (`y`):** The value the model is trained to predict.
- **Training:** Learning model parameters from training data.
- **Prediction:** Applying the trained model to new inputs.
- **Generalization:** Performing well on previously unseen data.

Example: Predicting house prices.

| Area (sq. ft.) | Bedrooms | Price |
|---:|---:|---:|
| 900 | 2 | 35 lakh |
| 1200 | 3 | 48 lakh |
| 1600 | 3 | 65 lakh |

Here, area and bedrooms are features, while price is the target.

```python
import pandas as pd

df = pd.DataFrame({
    "area": [900, 1200, 1600, 2000, 2400],
    "bedrooms": [2, 3, 3, 4, 4],
    "price": [35, 48, 65, 82, 98],
})

X = df[["area", "bedrooms"]]
y = df["price"]

print("Feature shape:", X.shape)
print("Target shape:", y.shape)
```

## 2. Types of Machine Learning Tasks

### Regression

Predicts a numerical quantity.

Examples: house price, delivery time, temperature.

Common metrics: MAE, MSE, RMSE, and R².

### Classification

Predicts a category or class.

Examples: spam detection, fraud detection, disease classification.

Common metrics: accuracy, precision, recall, F1-score, and ROC-AUC.

### Clustering

Groups observations according to similarities without requiring predefined target labels.

Example: grouping customers by purchasing behavior.

A clustering result does not automatically imply that the discovered groups are meaningful; interpretation and validation are still required.

## 3. Standard Supervised ML Workflow

1. Define the problem and prediction target.
2. Collect and inspect the data.
3. Perform exploratory data analysis (EDA).
4. Separate features (`X`) and target (`y`).
5. Split the data into training and testing sets.
6. Preprocess the data using training data appropriately.
7. Train a simple baseline.
8. Train candidate models.
9. Evaluate using metrics appropriate for the task.
10. Compare results and investigate errors.
11. Improve the model using justified experiments.
12. Save the final pipeline and document its limitations.

The workflow is iterative. New findings during EDA or evaluation may require revisiting earlier steps.

## 4. Train/Test Split

The training set is used to fit the model. The test set estimates how well the final model performs on unseen data.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

X = df[["area", "bedrooms"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
```

This tiny example demonstrates the API; its five observations are not sufficient for a reliable real-world performance estimate.

### Important Rules

- Do not train the model on the test set.
- Do not use test results repeatedly to select hyperparameters.
- Use validation data or cross-validation for model selection.
- Keep the test set for final evaluation.
- Choose the split strategy based on the data and task.

For classification, `stratify=y` can preserve class proportions when appropriate. For time-dependent data, a chronological split is often more suitable than a random split.

## 5. Preprocessing and Data Leakage

Preprocessing includes handling missing values, encoding categorical variables, and scaling numerical features.

**Data leakage** occurs when training or model selection uses information that would not legitimately be available at prediction time.

Incorrect approach:

1. Fit a scaler on the entire dataset.
2. Split the dataset.
3. Train the model.

The scaler has already used information from the test set.

Better approach: fit preprocessing only on training data. A scikit-learn pipeline helps enforce this rule.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000),
)

# X_train and y_train should already be separated
pipeline.fit(X_train, y_train)
```

This classification example assumes the training features are numerical and the training target contains class labels. Use suitable encoding and preprocessing for other data types.

## 6. Evaluation and Generalization

Training performance describes how well a fitted model performs on its training data. Test performance estimates performance on held-out data.

A large difference between the two may indicate overfitting, although the data split, dataset size, and metric also matter.

Always select metrics according to the task and the cost of different prediction errors.

## 7. Reproducibility

Use fixed random seeds where supported, record important preprocessing and model settings, and document the dataset and evaluation method.

```python
RANDOM_STATE = 42
```

A fixed seed makes many randomized experiments repeatable, but does not guarantee identical results in every environment or on every hardware configuration.

## Common Mistakes

- Using the target or target-derived information as a feature.
- Preprocessing before splitting when the transformation learns from data.
- Choosing metrics without considering the task.
- Reporting training performance as evidence of generalization.
- Tuning against the test set repeatedly.
- Treating a single small test split as conclusive proof of model quality.

## Quick Revision

- `X` contains features; `y` contains the target.
- `fit()` learns from training data.
- `predict()` produces predictions.
- A baseline establishes a reference point.
- Validation supports model selection.
- The test set estimates final generalization performance.
- Leakage can make evaluation results misleading.

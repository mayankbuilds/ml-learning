# Machine Learning Fundamentals

Comprehensive concept notes and revision guide for the foundational principles, workflows, and core diagnostics of machine learning.

Related deep-dive topics:
- [ml-workflow.md](file:///mnt/E_Drive/Projects/ml-learning/classical-ml/fundamentals/ml-workflow.md)
- [overfitting-underfitting.md](file:///mnt/E_Drive/Projects/ml-learning/classical-ml/fundamentals/overfitting-underfitting.md)
- [baseline-models.md](file:///mnt/E_Drive/Projects/ml-learning/classical-ml/fundamentals/baseline-models.md)
- [model-persistence.md](file:///mnt/E_Drive/Projects/ml-learning/classical-ml/fundamentals/model-persistence.md)
- [data-leakage.md](file:///mnt/E_Drive/Projects/ml-learning/classical-ml/preprocessing/data-leakage.md)
- [train-test-validation.md](file:///mnt/E_Drive/Projects/ml-learning/classical-ml/evaluation/train-test-validation.md)

---

## 1. What Is Machine Learning?

Machine Learning (ML) is an engineering and scientific paradigm focused on constructing computer programs that automatically improve their performance at a given task through experience and exposure to data, rather than through exclusively handcrafted rule-based logic.

### Formal Formulation (Tom Mitchell, 1997)
> "A computer program is said to learn from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$."

In mathematical terms, supervised machine learning seeks an approximating function $\hat{f}$ such that:

$$
y = f(X) + \epsilon
$$

Where:
- $X \in \mathbb{R}^{n \times d}$: Feature matrix representing $n$ samples and $d$ input dimensions.
- $y \in \mathbb{R}^n$: True ground-truth target vector.
- $\hat{f}(X; \theta)$: Learned model parameterized by $\theta$.
- $\epsilon$: Irreducible noise component with $\mathbb{E}[\epsilon] = 0$ and $\text{Var}(\epsilon) = \sigma^2$.

### Key Terminology

| Concept | Definition | Example |
|---|---|---|
| **Features ($X$)** | Input independent variables used to describe each instance. | House square footage, number of rooms, postal code. |
| **Target ($y$)** | Ground truth label or quantity to predict. | House sales price (continuous), spam/ham (categorical). |
| **Sample / Instance** | A single row or observation in the dataset. | One specific house listing. |
| **Model Parameters ($\theta$)** | Internal coefficients learned automatically from data during training. | Weights $w$ and bias $b$ in linear models. |
| **Hyperparameters** | External configuration knobs set before training starts. | Regularization strength $C$, tree depth `max_depth`, learning rate $\eta$. |
| **Loss Function ($\mathcal{L}$)** | Error penalty computed on a single training example. | Squared error $(y_i - \hat{y}_i)^2$. |
| **Cost Function ($J$)** | Aggregate loss averaged across the entire training dataset. | Mean Squared Error $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$. |

---

## 2. Taxonomy of Machine Learning

```
                            Machine Learning
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
 Supervised Learning      Unsupervised Learning     Reinforcement Learning
   ├── Regression           ├── Clustering            ├── Agent & Environment
   └── Classification       ├── Dimensionality        ├── States & Actions
                                Reduction             └── Policy & Reward
                            └── Density Estimation
```

### 1. Supervised Learning
The training dataset contains both inputs $X$ and verified labels $y$. The objective is to learn a mapping $X \to y$ that accurately predicts targets for new, unseen inputs.
- **Regression:** Continuous numerical target (e.g., forecasting temperatures, housing prices, delivery durations). Metrics: MAE, MSE, RMSE, $R^2$.
- **Classification:** Categorical target labels (e.g., medical diagnosis, fraudulent transaction detection, document tagging). Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.

### 2. Unsupervised Learning
The dataset contains features $X$ without target labels $y$. The objective is to discover latent structure, groupings, or compressed representations.
- **Clustering:** Partitioning observations into coherent clusters based on geometric proximity or statistical density (e.g., K-Means, DBSCAN).
- **Dimensionality Reduction:** Compressing high-dimensional feature spaces while preserving maximum variance or pairwise distances (e.g., PCA, t-SNE, UMAP).
- **Anomaly Detection:** Flagging rare observations that deviate substantially from typical data density.

### 3. Semi-Supervised & Self-Supervised Learning
- **Semi-Supervised:** Combines a small pool of labeled data with a large body of unlabeled data to improve boundary estimation without excessive labeling expense.
- **Self-Supervised:** Generates supervisory training signals directly from the raw data itself (e.g., masked token prediction, contrastive representation learning).

### 4. Reinforcement Learning (RL)
An autonomous agent interacts with a dynamic environment through trial and error, taking actions to maximize cumulative expected discounted reward over time.

---

## 3. The End-to-End Machine Learning Lifecycle

A reliable machine learning project follows an iterative, disciplined workflow:

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 1. Problem Formulation & Metric Definition                             │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 2. Data Collection, Inspection & Cleaning                              │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 3. Exploratory Data Analysis (EDA)                                     │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 4. Train / Validation / Test Splitting (PREVENTS LEAKAGE)              │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 5. Feature Engineering & Preprocessing (Fit on Train only!)            │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 6. Baseline Reference Model (DummyRegressor / DummyClassifier)         │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 7. Model Exploration & Candidate Training                              │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 8. Hyperparameter Tuning & Cross-Validation                            │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 9. Final Evaluation on Unseen Test Set & Error Analysis                │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 10. Pipeline Serialization, Deployment & Monitoring                    │
  └────────────────────────────────────────────────────────────────────────┘
```

### Critical Workflow Rules
1. **Define success before modeling:** Select evaluation metrics aligned with operational costs (e.g., False Negatives vs False Positives) before comparing algorithms.
2. **Split data early:** Isolate the test set before computing any scaling statistics, imputations, or feature selections.
3. **Establish a baseline immediately:** Measure a simple naive baseline before declaring any complex model successful.
4. **Treat preprocessing as part of the model:** Encapsulate transformers and estimators inside a scikit-learn [`Pipeline`](file:///mnt/E_Drive/Projects/ml-learning/classical-ml/preprocessing/pipelines.md).
5. **Inspect residuals and misclassifications:** Aggregate benchmark numbers alone do not reveal systematic bias or failure modes.

---

## 4. Generalization, Overfitting, and Underfitting

Generalization is the model's capacity to deliver accurate predictions on new, previously unseen data sampled from the identical target distribution.

### The Diagnostic Spectrum

```
   High Bias                                                      High Variance
 (Underfitting)                   Optimal Fit                    (Overfitting)
       X                                                               X   X
    X     X                       X   X   X                          X       X
  ───────────                   ─────────────                      ─────────────
 Model too simple             Captures true pattern             Memorizes noise
 High train error              Low train error                   Near-zero train error
 High test error               Low test error                    High test error
```

### Detailed Diagnostics Table

| Characteristic | Underfitting (High Bias) | Good Generalization | Overfitting (High Variance) |
|---|---|---|---|
| **Training Error** | High | Low / Reasonable | Extremely low (near zero) |
| **Validation / Test Error** | High (comparable to train) | Low (close to train) | Significantly higher than train |
| **Gap (Val - Train Error)** | Small | Small | Large |
| **Sensitivity to Data Changes** | Low | Moderate | High (drastic parameter shifts) |
| **Learned Function** | Overly rigid / simplistic | Smooth, representative | Excessively complex / wiggly |
| **Primary Root Cause** | Insufficient model capacity, missing features, over-regularization | Balanced model capacity matching data complexity | Excess parameters, noisy data, data leakage, lack of regularization |

### Remediation Strategies

#### How to Address Underfitting
1. Increase model capacity (e.g., transition from linear models to tree ensembles or kernel SVMs).
2. Engineer informative interaction features or domain-specific attributes.
3. Relax regularization constraints (increase $C$ in Logistic Regression, decrease $\alpha$ in Ridge/Lasso).
4. Verify data quality and ensure the target is actually predictable from provided inputs.

#### How to Address Overfitting
1. Gather more representative training samples.
2. Reduce model capacity (restrict `max_depth`, increase `min_samples_leaf`, prune trees).
3. Introduce or increase regularization penalties ($L_1$ Lasso, $L_2$ Ridge, weight decay).
4. Remove uninformative or collinear features using feature selection.
5. Use Cross-Validation to validate hyperparameter decisions rather than fitting to noise.
6. Check for data leakage between training and validation splits.

---

## 5. The Bias–Variance Tradeoff

The Bias-Variance Tradeoff is the fundamental theoretical tension governing supervised learning performance.

### Mathematical Decomposition of Expected Prediction Error

For a squared-error loss function, the expected test error at a query point $x_0$ decomposes into three distinct non-negative components:

$$
\mathbb{E} \left[ (y - \hat{f}(x_0))^2 \right] = \underbrace{\Big( \mathbb{E}[\hat{f}(x_0)] - f(x_0) \Big)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E} \left[ \Big( \hat{f}(x_0) - \mathbb{E}[\hat{f}(x_0)] \Big)^2 \right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible Noise}}
$$

### 1. Bias Squared ($\text{Bias}^2$)
Measures the error introduced by approximating a complex real-world phenomenon with a simplified model class. High bias indicates the model systematically makes wrong assumptions regardless of sample size.

### 2. Variance ($\text{Variance}$)
Measures the variability of model predictions if trained on different random subsets of the same data generation process. High variance indicates excessive sensitivity to small fluctuations in the training sample.

### 3. Irreducible Error ($\sigma^2$)
The variance of the inherent noise $\epsilon$ in the underlying system. No algorithm, regardless of capacity or data volume, can reduce error below $\sigma^2$.

### Tradeoff Dynamics

$$
\text{Total Expected Error} = \text{Bias}^2 + \text{Variance} + \sigma^2
$$

As model complexity increases:
- **Bias monotonically decreases:** The model becomes expressive enough to represent intricate patterns.
- **Variance monotonically increases:** The model fits specific sample idiosyncrasies and noise.
- **Optimal Model Complexity:** The trough where the sum of $\text{Bias}^2$ and $\text{Variance}$ is minimized.

---

## 6. Data Splitting Strategies & Cross-Validation

Evaluating a model on the data used to fit its parameters creates optimistic, unreliable estimates. A rigorous splitting protocol is required.

```
Total Dataset
 ├── Training Set (~70%)    ──► Fit model parameters (weights, coefficients)
 ├── Validation Set (~15%)  ──► Tune hyperparameters, select best architecture
 └── Test Set (~15%)        ──► Final unbiased generalization estimate (used once!)
```

### Splitting Protocols

### 1. Simple Holdout Split (`train_test_split`)
Divides data randomly into train and test portions.
- **Use when:** Dataset is very large and class distributions are well balanced.
- **Scikit-Learn usage:**
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )
  ```

### 2. Stratified Splitting (`stratify=y`)
Ensures that the relative proportions of target classes remain identical across train and test partitions.
- **Use when:** Handling classification tasks, especially with imbalanced target labels.
- Prevents rare classes from vanishing completely from either partition.

### 3. K-Fold Cross-Validation
Partitions the training data into $K$ equally sized folds. The model is trained on $K-1$ folds and validated on the remaining fold, repeating $K$ times so every observation serves as validation once.

$$
\text{CV Score} = \frac{1}{K} \sum_{k=1}^K \text{Score}_k
$$

- Standard choice: $K = 5$ or $K = 10$.
- Provides both an expected performance estimate and an estimate of performance variability (standard deviation across folds).

### 4. Stratified K-Fold Cross-Validation (`StratifiedKFold`)
Combines K-Fold with stratification, preserving target class proportions within every fold. The standard default for all classification problems.

### 5. Time-Series Splitting (`TimeSeriesSplit`)
When observations exhibit temporal dependencies, random shuffling causes future information to leak into past predictions.
- **Rule:** Training folds must strictly precede validation folds in chronological order (expanding or rolling window).

```
Fold 1: [Train] [Val]
Fold 2: [ Train   ] [Val]
Fold 3: [   Train     ] [Val]
```

### 6. Grouped Splitting (`GroupKFold`)
When multiple records belong to the same entity (e.g., repeated measurements from the same patient, device, or user).
- **Rule:** All observations belonging to a given group must reside exclusively in either the training set or the validation set, never split across both.

---

## 7. Baseline Models: Why Every Project Needs One

A baseline provides the empirical reference floor against which all complex candidate models are judged.

### Why Baselines Are Crucial
- **The Imbalance Illusion:** In a fraud detection dataset with 99% legitimate transactions, a naive model predicting "legitimate" for every transaction achieves 99% accuracy. A machine learning model achieving 98% accuracy is worse than trivial guessing.
- **Cost-Benefit Justification:** A Random Forest or Deep Neural Network requiring significant compute, latency, and maintenance must justify its overhead over a simple mean predictor or single-rule baseline.

### Standard Scikit-Learn Baselines

```python
from sklearn.dummy import DummyRegressor, DummyClassifier

# Regression: Predict the target mean or median
dummy_reg = DummyRegressor(strategy="mean")
dummy_reg.fit(X_train, y_train)
baseline_pred = dummy_reg.predict(X_test)

# Classification: Predict the most frequent class or stratified proportions
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
baseline_clf_pred = dummy_clf.predict(X_test)
```

| Task | Baseline Strategy | Recommended Metric Comparison |
|---|---|---|
| **Regression** | Mean of training $y$ | Compare MAE / RMSE; Baseline has $R^2 = 0.0$. |
| **Regression (outliers)** | Median of training $y$ | Compare MAE against median prediction. |
| **Classification (balanced)** | Uniform random or single feature rule | Accuracy, F1-score. |
| **Classification (imbalanced)** | Majority class (`most_frequent`) | Compare Balanced Accuracy, Precision, Recall, PR-AUC. |

---

## 8. Data Leakage: Prevention and Safeguards

**Data leakage** occurs when information from outside the training dataset is inadvertently introduced into the model fitting process, producing artificially optimistic evaluation scores that collapse in production.

```
       WRONG (Data Leakage)                      CORRECT (Safe Pipeline)
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│     Full Dataset (All Rows)     │       │     Full Dataset (All Rows)     │
└────────────────┬────────────────┘       └────────────────┬────────────────┘
                 ▼                                         ▼
┌─────────────────────────────────┐                  Train/Test Split
│ Fit Scaler/Imputer on FULL data │                        │
└────────────────┬────────────────┘         ┌──────────────┴──────────────┐
                 ▼                          ▼                             ▼
         Train/Test Split             [Train Data]                   [Test Data]
                 │                          │                             │
    Test data statistics leak into          ▼                             ▼
       training parameters!          Fit Pipeline on Train     Transform Test ONLY
                                     (Learns mu, sigma)        (Uses train mu, sigma)
```

### Common Leakage Vectors

1. **Preprocessing Leakage:**
   Computing `mean`, `std`, `min`, `max`, or categorical vocabularies across the entire dataset before splitting.
   - *Fix:* Always perform `train_test_split` first, then call `fit_transform()` on training data and `transform()` on test data.
2. **Target Leakage:**
   Including features that are directly populated or updated only after the target event occurs (e.g., using `discharge_date` or `treatment_code` to predict whether a patient will be admitted to ICU).
   - *Fix:* Verify feature availability at the exact historical moment the prediction must be generated in production.
3. **Temporal Leakage:**
   Shuffling time-series observations randomly, allowing future trends to predict past records.
   - *Fix:* Use chronological splits (`TimeSeriesSplit`).
4. **Duplicate / Group Leakage:**
   Having near-duplicate rows or multiple entries from the same subject present in both train and test partitions.
   - *Fix:* Group-aware splitting (`GroupKFold`).

### Prevention with Scikit-Learn Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Define numerical and categorical transformations
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, ["age", "income", "credit_score"]),
    ("cat", cat_pipeline, ["employment_type", "state"]),
])

# Complete, leak-proof pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

# Preprocessing parameters are learned STRICTLY from X_train!
pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
```

---

## 9. Model Persistence & Reproducibility

Once a model pipeline is trained and validated, it must be persisted for offline inference, web service deployment, or batch scoring without retraining.

### 1. Why Persist the Full Pipeline?
Never save the raw estimator in isolation if input data requires imputation, encoding, or scaling. Saving the complete `Pipeline` guarantees that the exact transformations, imputation statistics, and learned scalers are applied identically to future inferences.

### 2. Using `joblib` for Serialization

```python
import joblib

# Persist the complete pipeline to disk
joblib.dump(pipeline, "artifacts/model_pipeline.joblib")

# Load pipeline into an inference service
loaded_pipeline = joblib.load("artifacts/model_pipeline.joblib")

# Predict on new un-preprocessed raw input
predictions = loaded_pipeline.predict(X_new)
```

### 3. Security Considerations
Pickle- and joblib-serialized files can execute arbitrary Python code upon loading. Never load `.joblib` or `.pkl` files obtained from unauthenticated or untrusted external sources.

### 4. Reproducibility Best Practices
- **Explicit Random Seeds:** Fix `random_state=42` across data splitters, cross-validators, and stochastic algorithms.
- **Environment Pinning:** Record Python, scikit-learn, and library versions in `requirements.txt` or `pyproject.toml`.
- **Pipeline Architecture:** Eliminate manual data-manipulation steps between raw input and prediction.

---

## 10. End-to-End Implementation Template

Below is a self-contained, leak-proof, reproducible workflow implementing all fundamental concepts:

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# 1. Generate Synthetic Data
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_classes=2,
    weights=[0.85, 0.15],  # Imbalanced: 85% vs 15%
    random_state=42,
)

# 2. Train/Test Split (Preserve Class Proportions with stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Establish Naive Baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_preds = baseline.predict(X_test)
print("=== Baseline Performance ===")
print("Baseline Accuracy:", baseline.score(X_test, y_test))

# 4. Construct Leak-Proof Pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(class_weight="balanced", random_state=42)),
])

# 5. Cross-Validation on Training Split
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
print("\n=== Cross-Validation Results ===")
print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# 6. Fit Final Model on Full Training Set and Evaluate on Held-out Test Set
pipeline.fit(X_train, y_train)
test_preds = pipeline.predict(X_test)
test_probs = pipeline.predict_proba(X_test)[:, 1]

print("\n=== Final Test Set Evaluation ===")
print(f"Test ROC-AUC: {roc_auc_score(y_test, test_probs):.4f}")
print("\nClassification Report:\n", classification_report(y_test, test_preds))

# 7. Model Persistence
joblib.dump(pipeline, "final_pipeline.joblib")
print("Pipeline serialized successfully to 'final_pipeline.joblib'")
```

---

## 11. Common Mistakes & Pitfalls

1. **Splitting After Scaling:** Fitting a scaler or encoder on the complete dataset before calling `train_test_split` leaks test statistics into the training process.
2. **Ignoring Class Imbalance:** Measuring standard classification accuracy on heavily skewed datasets instead of Balanced Accuracy, F1-Score, or PR-AUC.
3. **Repeatedly Tuning Against the Test Set:** Using test set performance to choose hyperparameters transforms the test set into a validation set, invalidating its final generalization estimate.
4. **Evaluating Without Baselines:** Declaring a model "90% accurate" when a naive majority-class rule achieves 92%.
5. **Treating High Training Score as Proof of Quality:** High training performance combined with poor validation performance indicates severe overfitting, not a successful model.
6. **Ignoring Group or Time Structure:** Using standard random splits on time-dependent or grouped observation datasets.
7. **Saving Models Without Transformers:** Exporting an estimator while discarding the categorical encoders and numerical scalers required to preprocess production inputs.

---

## 12. Quick Revision Cheatsheet

- **Supervised:** Inputs $X$ + Labels $y$; predicts continuous (Regression) or discrete (Classification) outputs.
- **Unsupervised:** Inputs $X$ only; discovers grouping (Clustering) or representation (Dimensionality Reduction).
- **Features & Target:** `X.shape == (n_samples, n_features)`; `y.shape == (n_samples,)`.
- **Underfitting:** High bias, poor training score, poor test score. Fix by increasing complexity or relaxing regularization.
- **Overfitting:** High variance, excellent training score, poor test score. Fix with regularization, more data, or pruning.
- **Bias-Variance Decomposition:** $\text{Error} = \text{Bias}^2 + \text{Variance} + \sigma^2$.
- **Split Protocol:** Train (learn parameters) $\to$ Validation (tune hyperparameters) $\to$ Test (final check).
- **Stratification:** Maintains class ratios; essential for imbalanced classification tasks.
- **Data Leakage:** Information from the future or test partition contaminating training. Avoid via `Pipeline`.
- **Baseline:** Mandatory reference anchor (`DummyRegressor`, `DummyClassifier`) before building complex models.
- **Persistence:** Save complete `Pipeline` (transformers + estimator) using `joblib`.

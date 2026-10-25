# Classical Machine Learning

A comprehensive reference and revision guide covering algorithms, mathematical formulations, preprocessing pipelines, model selection, and evaluation across classical supervised and unsupervised learning.

Related deep-dive topic guides:
- **Preprocessing:** [missing-values.md](../classical-ml/preprocessing/missing-values.md) | [encoding.md](../classical-ml/preprocessing/encoding.md) | [feature-scaling.md](../classical-ml/preprocessing/feature-scaling.md) | [pipelines.md](../classical-ml/preprocessing/pipelines.md)
- **Regression:** [linear-regression.md](../classical-ml/regression/linear-regression.md) | [regularization.md](../classical-ml/regression/regularization.md) | [regression-metrics.md](../classical-ml/regression/regression-metrics.md)
- **Classification:** [logistic-regression.md](../classical-ml/classification/logistic-regression.md) | [decision-trees.md](../classical-ml/classification/decision-trees.md) | [random-forest.md](../classical-ml/classification/random-forest.md) | [knn.md](../classical-ml/classification/knn.md) | [svm.md](../classical-ml/classification/svm.md) | [classification-metrics.md](../classical-ml/classification/classification-metrics.md)
- **Clustering:** [k-means.md](../classical-ml/clustering/k-means.md) | [dbscan.md](../classical-ml/clustering/dbscan.md) | [clustering-evaluation.md](../classical-ml/clustering/clustering-evaluation.md)
- **Model Selection:** [grid-search.md](../classical-ml/model-selection/grid-search.md) | [randomized-search.md](../classical-ml/model-selection/randomized-search.md) | [model-selection.md](../classical-ml/model-selection/model-selection.md)

---

## 1. Algorithm Taxonomy & Conceptual Map

```text
                           Classical Machine Learning
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
   Supervised Learning                                 Unsupervised Learning
   ├── Regression                                      ├── Clustering
   │   ├── Linear Regression (OLS)                     │   ├── K-Means (Centroid-based)
   │   ├── Ridge Regression (L2)                       │   └── DBSCAN (Density-based)
   │   ├── Lasso Regression (L1)                       └── Dimensionality Reduction
   │   └── ElasticNet (L1 + L2)                            └── PCA (Variance maximization)
   └── Classification
       ├── Logistic Regression (Probabilistic linear)
       ├── k-Nearest Neighbors (Instance-based / Non-parametric)
       ├── Decision Trees (Recursive partitioning / Non-linear)
       ├── Random Forest (Ensemble / Bagging)
       └── Support Vector Machines (Margin maximization / Kernels)
```

### Parametric vs. Non-Parametric Models

| Category | Definition | Advantages | Disadvantages | Examples |
|---|---|---|---|---|
| **Parametric** | Assumes a fixed functional form $f(X; \theta)$ with a finite, constant number of parameters independent of sample size $n$. | Fast inference, low memory footprint, mathematically interpretable. | Can underfit if true data distribution violates assumptions (high bias). | Linear Regression, Logistic Regression, Linear SVM. |
| **Non-Parametric** | Makes no rigid structural assumption about the underlying distribution; parameter count can grow with dataset size $n$. | Highly expressive, flexible boundary representation (low bias). | Computationally demanding on large datasets, prone to overfitting (high variance). | k-Nearest Neighbors, Decision Trees, RBF-Kernel SVM. |

---

## 2. Data Preprocessing & Feature Engineering

Garbage in, garbage out. Most classical estimators (except tree-based models) require careful numerical transformation before training.

### 1. Missing Value Imputation
- **Univariate Imputation (`SimpleImputer`):**
  - Continuous data: `strategy='median'` (robust against outliers) or `strategy='mean'`.
  - Categorical data: `strategy='most_frequent'` or `strategy='constant', fill_value='missing'`.
- **Multivariate Imputation (`KNNImputer`, `IterativeImputer`):** Estimates missing values by referencing correlated feature values across similar observations.
- **Missingness Indicator (`MissingIndicator`):** Creates binary flags recording whether values were absent, preserving potential signal in missingness patterns.

### 2. Categorical Encoding
- **One-Hot Encoding (`OneHotEncoder`):**
  - Converts nominal categories with no natural ordering (e.g., Color: Red, Blue, Green) into binary columns.
  - Set `handle_unknown='ignore'` to prevent errors when encountering novel categories during inference.
  - Set `drop='first'` for unregularized linear models to eliminate dummy variable trap multicollinearity.
- **Ordinal Encoding (`OrdinalEncoder`):** Maps ordered categories (e.g., Low, Medium, High $\to$ 0, 1, 2) to integer ranks.
- **Target Encoding (`TargetEncoder`):** Encodes high-cardinality categories using the expected target value, applying smoothing regularization to prevent leakage.

### 3. Feature Scaling
When algorithms compute geometric distances or optimize convex loss surfaces via gradient descent, disparate feature scales distort optimization.

| Method | Formula | Scikit-Learn Class | Properties & Best Use |
|---|---|---|---|
| **Standardization (Z-score)** | $z = \frac{x - \mu}{\sigma}$ | `StandardScaler` | Zero mean ($\mu = 0$), unit variance ($\sigma = 1$). Ideal for Logistic Regression, SVM, PCA, Ridge, Lasso. |
| **Min-Max Normalization** | $x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ | `MinMaxScaler` | Bounds values strictly into $[0, 1]$. Ideal for neural networks and algorithms requiring bounded positive inputs. |
| **Robust Scaling** | $x_{\text{robust}} = \frac{x - \text{median}}{\text{IQR}}$ | `RobustScaler` | Uses median and Interquartile Range ($Q_3 - Q_1$). Robust against heavy outliers. |

#### When Scaling Matters vs. Does Not Matter
- **Mandatory:** kNN (distance-based), K-Means (Euclidean distance), SVM (margin calculation), PCA (variance calculation), regularized linear models (regularization penalty applies equally to all weights).
- **Not Required:** Decision Trees, Random Forests, Gradient Boosted Trees (splits are orthogonal and monotonic per individual feature).

---

## 3. Supervised Learning: Regression

Predicting a continuous target $y \in \mathbb{R}$ from features $x \in \mathbb{R}^d$.

### 1. Ordinary Least Squares (OLS) Linear Regression

Linear regression assumes a linear relationship between input features and target:

$$
\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_d x_d = w^T x + b
$$

#### Objective Function (Residual Sum of Squares)
Minimizes the sum of squared differences between actual and predicted values:

$$
J(w) = \frac{1}{2n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \frac{1}{2n} \|Xw - y\|_2^2
$$

#### Closed-Form Solution (The Normal Equation)
Setting the gradient $\nabla_w J(w) = 0$ yields:

$$
w = (X^T X)^{-1} X^T y
$$

*Computational Complexity:* Matrix inversion $(X^T X)^{-1}$ scales as $\mathcal{O}(d^3)$. For large feature counts ($d > 10,000$), Gradient Descent is preferred over direct matrix inversion.

#### Classical Gauss-Markov Assumptions
1. **Linearity:** Relationship between features and target is linear in parameters.
2. **Strict Exogeneity:** $\mathbb{E}[\epsilon \mid X] = 0$ (error terms are independent of features).
3. **No Multicollinearity:** Feature columns are linearly independent ($X^T X$ is invertible).
4. **Homoscedasticity:** Constant error variance across all feature levels ($\text{Var}(\epsilon_i) = \sigma^2$).
5. **No Autocorrelation:** Residuals are uncorrelated across observations ($\text{Cov}(\epsilon_i, \epsilon_j) = 0, \forall i \neq j$).

---

### 2. Regularized Regression (Ridge, Lasso, ElasticNet)

When features exhibit high multicollinearity or model capacity causes overfitting, regularization penalties constrain coefficient magnitudes.

```text
       L1 Penalty (Lasso)                     L2 Penalty (Ridge)
          w2                                     w2
          ▲                                      ▲
        ◆ │                                    ╭───╮
      ◆   ◆                                   │     │
    ────┼───┼────► w1                       ────┼───┼────► w1
      ◆   ◆                                   │     │
        ◆ │                                    ╰───╯
          ▼                                      ▼
Sharp diamond corners hit axes              Smooth circle shrinks weights
  --> Exact Sparsity (w_j = 0)                --> Coefficients near 0
```

#### Ridge Regression ($L_2$ Regularization)
Adds a squared magnitude penalty to the loss function:

$$
J_{\text{Ridge}}(w) = \frac{1}{2n} \|Xw - y\|_2^2 + \alpha \sum_{j=1}^d w_j^2
$$

- Closed-form solution: $w = (X^T X + \alpha I)^{-1} X^T y$.
- **Key Behavior:** Shrinks coefficients toward zero continuously, but never sets them exactly to zero. Robust to multicollinearity because $(X^T X + \alpha I)$ is always invertible for $\alpha > 0$.

#### Lasso Regression ($L_1$ Regularization)
Adds an absolute magnitude penalty to the loss function:

$$
J_{\text{Lasso}}(w) = \frac{1}{2n} \|Xw - y\|_2^2 + \alpha \sum_{j=1}^d |w_j|
$$

- **Key Behavior:** Due to diamond-shaped geometric constraints, Lasso drives irrelevant feature coefficients strictly to zero. Acts as an automatic feature selection mechanism.

#### ElasticNet
Combines both $L_1$ and $L_2$ penalties:

$$
J_{\text{ElasticNet}}(w) = \frac{1}{2n} \|Xw - y\|_2^2 + \alpha \left( \rho \|w\|_1 + \frac{1 - \rho}{2} \|w\|_2^2 \right)
$$

- Controlled by `l1_ratio` ($\rho$). Handles groups of correlated features where Lasso would arbitrarily pick just one.

---

### 3. Regression Evaluation Metrics

| Metric | Formula | Interpretation & Behavior |
|---|---|---|
| **Mean Absolute Error (MAE)** | $\frac{1}{n} \sum_{i=1}^n \lvert y_i - \hat{y}_i \rvert$ | Direct, intuitive average error in target units. Robust to extreme outliers; linear error penalty. |
| **Mean Squared Error (MSE)** | $\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$ | Quadratically penalizes large errors. Highly sensitive to outliers; differentiable objective function. |
| **Root Mean Squared Error (RMSE)** | $\sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$ | Retains quadratic penalty of MSE while returning to original target units for interpretability. |
| **R² Score (Determination)** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Fraction of target variance explained by features. Baseline mean prediction achieves $R^2 = 0.0$; can be negative on test sets. |
| **Adjusted R²** | $1 - \left[ \frac{(1 - R^2)(n - 1)}{n - p - 1} \right]$ | Penalizes the addition of non-informative features ($p$ = feature count, $n$ = sample count). |

---

## 4. Supervised Learning: Classification

Predicting a discrete class label $y \in \{0, 1\}$ or $y \in \{1, \dots, C\}$.

### 1. Logistic Regression

Models the posterior probability of the positive class using the sigmoid (logistic) link function:

$$
z = w^T x + b, \qquad P(y=1 \mid x) = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

#### Log-Odds (Logit Transformation)
The log-odds of the positive class is a linear function of inputs:

$$
\ln \left( \frac{P(y=1 \mid x)}{1 - P(y=1 \mid x)} \right) = w^T x + b
$$

#### Binary Cross-Entropy / Log Loss
Parameters $w, b$ are learned by maximizing the log-likelihood (minimizing negative log-likelihood):

$$
J(w) = -\frac{1}{n} \sum_{i=1}^n \Big[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \Big] + \frac{1}{2C} \|w\|_2^2
$$

- $C$: Inverse regularization parameter ($C = \frac{1}{\lambda}$). Smaller $C$ enforces stronger regularization.

#### Decision Thresholding
- Standard rule: Predict class 1 if $\hat{p} \geq 0.5$.
- **Threshold Tuning:** Lowering the threshold (e.g., 0.2) increases sensitivity/recall (catches more positive cases at the expense of false alarms). Raising threshold (e.g., 0.8) increases precision.

---

### 2. k-Nearest Neighbors (kNN)

An instance-based, lazy learning algorithm:
1. Stores the entire training dataset in memory.
2. For a new query point $x_0$, computes distances to all stored instances.
3. Selects the $k$ closest points and assigns the majority class (or distance-weighted vote).

#### Distance Metrics
- **Euclidean ($L_2$):** $d(u, v) = \sqrt{\sum (u_i - v_i)^2}$
- **Manhattan ($L_1$):** $d(u, v) = \sum \|u_i - v_i\|$
- **Minkowski ($L_p$):** $d(u, v) = \left( \sum \|u_i - v_i\|^p \right)^{1/p}$

#### Parameter Dynamics of $k$
- **Small $k$ (e.g., $k=1$):** Captures fine-grained local decision boundaries; highly sensitive to noise; high variance / overfitting.
- **Large $k$ (e.g., $k=50$):** Smooths boundaries; approaches global class distribution; high bias / underfitting.
- **Prerequisite:** Feature scaling is strictly required; unscaled features with large numerical magnitudes dominate distance metrics.

---

### 3. Decision Trees (CART)

Recursively partitions the feature space into axis-aligned rectangular regions using greedy binary splits.

```text
                   [Feature X1 <= 2.5]
                       /         \
                     Yes          No
                     /             \
          [Feature X2 <= 1.8]   Class: B (100%)
              /         \
            Yes          No
            /             \
      Class: A (90%)   Class: B (80%)
```

#### Splitting Criteria
- **Gini Impurity (Default):**
  $$
  I_G(p) = 1 - \sum_{i=1}^C p_i^2
  $$
  Measures the probability of misclassifying a randomly chosen element if labeled according to class distribution. Pure node has $I_G = 0$.
- **Entropy & Information Gain:**
  $$
  H(p) = -\sum_{i=1}^C p_i \log_2(p_i), \qquad \text{Gain} = H(\text{Parent}) - \sum \frac{n_{\text{child}}}{n} H(\text{Child})
  $$
- **Regularization Controls:** Restrict tree depth (`max_depth`), require minimum samples per split (`min_samples_split`), or enforce leaf size (`min_samples_leaf`).

---

### 4. Ensemble Learning: Random Forests

Combines a collection of de-correlated decision trees to dramatically reduce variance without increasing bias.

#### Core Mechanisms
1. **Bagging (Bootstrap Aggregating):**
   Each individual tree is trained on a distinct bootstrap sample of size $n$ drawn uniformly at random **with replacement** from the training set. Approximately 63.2% of samples are included per tree; the remaining 36.8% form the Out-of-Bag (OOB) validation set.
2. **Feature Subsampling (Random Subspaces):**
   At every split within every tree, only a random subset of features ($m = \sqrt{d}$ for classification, $m = d/3$ for regression) is considered. This prevents strong dominant features from causing all trees to make identical splits, ensuring tree independence.
3. **Aggregation:**
   - Classification: Majority voting across all trees ($\hat{y} = \text{mode}(\hat{y}_1, \dots, \hat{y}_B)$).
   - Regression: Arithmetic average of predictions ($\hat{y} = \frac{1}{B} \sum_{b=1}^B \hat{y}_b$).

---

### 5. Support Vector Machines (SVM)

Finds the optimal separating hyperplane that maximizes the margin (geometric distance between the decision boundary and the nearest training points, termed **support vectors**).

```text
               x  (Class +1)
             x   /
           x    /  <-- Margin boundary (w^T x + b = +1)
          ──── / ──────────────────────────────────
              /    <-- Optimal Hyperplane (w^T x + b = 0)
          ── / ────────────────────────────────────
            /  o   <-- Margin boundary (w^T x + b = -1)
           /  o
          o       (Class -1)

      Margin Width = 2 / ||w||
```

#### Mathematical Formulation (Soft Margin)
Minimizes weight norm while penalizing misclassifications using slack variables $\xi_i$:

$$
\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i \quad \text{subject to} \quad y_i (w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$

- **$C$ Parameter:** Trade-off between margin width and margin violations. Large $C$ forces narrow margins with few errors (overfitting risk); small $C$ permits a wider margin with more violations (underfitting risk).

#### The Kernel Trick
Enables non-linear classification by implicitly mapping inputs into high-dimensional feature spaces via inner products:

$$
K(x_i, x_j) = \phi(x_i)^T \phi(x_j)
$$

- **Radial Basis Function (RBF / Gaussian) Kernel:**
  $$
  K(x_i, x_j) = \exp \left( -\gamma \|x_i - x_j\|^2 \right)
  $$
  - **$\gamma$ Parameter:** Controls the radius of influence of each support vector. Large $\gamma$ leads to sharp, isolated decision boundaries (overfitting); small $\gamma$ yields smooth, generalized boundaries.

---

### 6. Classification Evaluation Metrics

```text
                    Actual Positive (1)        Actual Negative (0)
                ┌──────────────────────────┬──────────────────────────┐
  Predicted     │   True Positive (TP)     │   False Positive (FP)    │
  Positive (1)  │   (Correct hit)          │   (Type I Error / Alarm) │
                ├──────────────────────────┼──────────────────────────┤
  Predicted     │   False Negative (FN)    │   True Negative (TN)     │
  Negative (0)  │   (Type II Error / Miss) │   (Correct rejection)    │
                └──────────────────────────┴──────────────────────────┘
```

| Metric | Formula | Practical Focus |
|---|---|---|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall correctness. Unreliable for imbalanced datasets! |
| **Precision** | $\frac{TP}{TP + FP}$ | "Of all predicted positive, how many were truly positive?" Critical when False Positives are expensive (e.g., spam detection). |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | "Of all actual positives, how many did we capture?" Critical when False Negatives are catastrophic (e.g., tumor diagnosis). |
| **Specificity** | $\frac{TN}{TN + FP}$ | True Negative Rate. True rejections out of all actual negatives. |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of precision and recall. Punishes extreme imbalances between precision and recall. |
| **ROC-AUC** | Area Under Receiver Operating Characteristic Curve | True Positive Rate vs False Positive Rate across all thresholds. Evaluates ranking ability; threshold-independent. |
| **PR-AUC** | Area Under Precision-Recall Curve | Precision vs Recall across all thresholds. Preferred evaluation metric for rare positive classes (e.g., fraud). |

---

## 5. Unsupervised Learning: Clustering

Partitioning unlabelled observations into clusters where intra-cluster similarity is maximized and inter-cluster similarity is minimized.

### 1. K-Means Clustering

A centroid-based iterative partitioning algorithm.

#### Objective (Within-Cluster Sum of Squares / Inertia)
Minimizes the sum of squared Euclidean distances between points and their assigned cluster centroid:

$$
J = \sum_{j=1}^K \sum_{x_i \in C_j} \|x_i - \mu_j\|^2
$$

#### Lloyd's Algorithm Iteration
1. **Initialize:** Select $K$ initial centroids $\mu_1, \dots, \mu_K$ (using K-Means++).
2. **Assignment Step:** Assign each data point to its nearest centroid:
   $$
   C_j = \{ x_i : \|x_i - \mu_j\|^2 \leq \|x_i - \mu_l\|^2, \forall l \neq j \}
   $$
3. **Update Step:** Recompute centroids as the mean of assigned observations:
   $$
   \mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i
   $$
4. Repeat steps 2-3 until centroids stabilize (convergence).

#### Choosing $K$
- **Elbow Method:** Plot Inertia vs. $K$. Select the elbow point where the rate of inertia reduction visibly slows down.
- **Silhouette Analysis:** Measures how close each point is to points in its own cluster versus the nearest neighboring cluster:
  $$
  s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, 1]
  $$
  - $a(i)$: Mean intra-cluster distance.
  - $b(i)$: Mean nearest-cluster distance.
  - Scores close to $+1$ indicate strong, distinct clustering.

---

### 2. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

Discovers clusters of arbitrary shape based on spatial point density and isolates outlier noise points.

#### Parameters
- **$\varepsilon$ (Eps):** Radius of neighborhood around a data point.
- **`min_samples`:** Minimum number of points required within the $\varepsilon$-neighborhood to form a dense region.

#### Point Classifications
- **Core Point:** Has at least `min_samples` points within its $\varepsilon$-neighborhood.
- **Border Point:** Within $\varepsilon$ of a Core Point, but has fewer than `min_samples` neighbors.
- **Noise Point:** Neither a Core nor a Border point. Assigned label `-1`.

#### K-Means vs. DBSCAN Comparison

| Feature | K-Means | DBSCAN |
|---|---|---|
| **Cluster Geometry** | Assumes convex, spherical clusters | Discovers complex, arbitrary shapes |
| **Number of Clusters $K$** | Must be specified beforehand | Discovered automatically |
| **Noise & Outliers** | Assigns every point to a cluster | Automatically flags outliers as noise (`-1`) |
| **Varying Densities** | Handles varying densities decently | Struggles with clusters of varying densities |
| **Scale Sensitivity** | High (Euclidean distance) | High (requires feature scaling) |

---

## 6. Model Selection & Hyperparameter Optimization

- **Model Parameters:** Learned during training directly from data (weights $w$, biases $b$, split thresholds).
- **Hyperparameters:** Configured prior to training (learning rate $\eta$, $C$, $\alpha$, `n_estimators`, `max_depth`).

### Search Strategies

```text
      GridSearchCV                           RandomizedSearchCV
  Param 2                                Param 2
     ▲                                      ▲
     │  ●   ●   ●   ●                       │      ●        ●
     │  ●   ●   ●   ●                       │  ●        ●
     │  ●   ●   ●   ●                       │        ●        ●
     │  ●   ●   ●   ●                       │    ●        ●
     └────────────────► Param 1             └────────────────► Param 1
  Evaluates every combination             Samples random combinations
  (Computationally expensive)             (Budget-friendly, broad coverage)
```

1. **`GridSearchCV`:**
   Exhaustively trains and evaluates models on every coordinate in a specified discrete grid. Guaranteed to find the best candidate in the grid, but computationally intractable for large parameter spaces.
2. **`RandomizedSearchCV`:**
   Samples a fixed number of parameter configurations (`n_iter`) from defined statistical distributions (e.g., `scipy.stats.loguniform`). Often discovers superior configurations faster by allocating evaluations across continuous parameter spans.

---

## 7. Master Classical ML Algorithm Cheatsheet

| Algorithm | Model Type | Parametric? | Requires Scaling? | Sensitive to Outliers? | Interpretability | Primary Hyperparameters | Best Suited For |
|---|---|---|---|---|---|---|---|
| **Linear Regression** | Regression | Yes | Yes (when regularized) | High | High | `fit_intercept`, `positive` | Linear relationships, fast baselines |
| **Ridge Regression** | Regression | Yes | Yes | Moderate | High | `alpha`, `solver` | Multicollinear features, preventing overfitting |
| **Lasso Regression** | Regression | Yes | Yes | Moderate | High | `alpha`, `max_iter` | Sparse feature selection, high-dimensional data |
| **Logistic Regression** | Classification | Yes | Yes | Moderate | High | `C`, `penalty`, `solver` | Probability estimation, linear boundaries, baseline |
| **k-Nearest Neighbors** | Both | No | **Strictly Yes** | High | Moderate | `n_neighbors`, `weights`, `metric` | Low-dimensional data, non-linear local patterns |
| **Decision Trees** | Both | No | **No** | Low | High | `max_depth`, `min_samples_leaf` | Interpretable rules, tabular data, non-linear splits |
| **Random Forest** | Both | No | **No** | Low | Moderate | `n_estimators`, `max_depth`, `max_features` | Tabular data, non-linear relationships, robust default |
| **Support Vector Machines** | Both | Non-param (RBF) / Param (Linear) | **Strictly Yes** | High | Low | `C`, `kernel`, `gamma` | Medium-sized complex datasets, text/image embeddings |
| **K-Means** | Clustering | Yes | **Strictly Yes** | High | High | `n_clusters`, `init`, `n_init` | Spherical clusters, customer segmentation |
| **DBSCAN** | Clustering | No | **Strictly Yes** | Low (isolates noise) | Moderate | `eps`, `min_samples`, `metric` | Non-spherical geometries, anomaly detection |

---

## 8. Common Implementation Pitfalls

1. **Forgetting to Scale for Distance & Kernel Models:** Running kNN, SVM, or K-Means on raw unscaled features causes variables with large scales to dominate all distance computations.
2. **Using One-Hot Encoding on High-Cardinality Variables with Trees:** Generating hundreds of sparse binary columns degrades tree splitting efficiency. Use target encoding or ordinal grouping instead.
3. **Optimizing Accuracy on Imbalanced Classes:** Achieving 98% accuracy on a dataset with 2% positive labels is trivial and uninformative; optimize F1-score or PR-AUC instead.
4. **Interpreting Tree Feature Importance Naively:** Default Gini importance (`feature_importances_`) inflates the importance of continuous and high-cardinality features. Use `permutation_importance` for unbiased evaluation.
5. **Evaluating Out-of-Fold Performance on the Test Set:** Running hyperparameter grid searches directly on the test set leads to severe selection bias. Always tune using cross-validation on the training set.

---

## 9. Quick Revision Summary

- **OLS Normal Equation:** $w = (X^T X)^{-1} X^T y$.
- **Ridge vs Lasso:** Ridge ($L_2$) shrinks weights smoothly; Lasso ($L_1$) induces exact sparsity.
- **Logistic Sigmoid:** $\sigma(z) = \frac{1}{1 + e^{-z}}$; maps real values to probability $[0, 1]$.
- **Decision Trees:** Split nodes using Gini Impurity or Information Gain; invariant to monotonic feature scaling.
- **Random Forest:** Reduces tree variance via Bootstrap Aggregation + Random Feature Subspacing.
- **SVM Margin:** Maximizes $\frac{2}{\|w\|}$; RBF kernel projects features to infinite-dimensional Hilbert space.
- **Evaluation:** Precision = $\frac{TP}{TP+FP}$; Recall = $\frac{TP}{TP+FN}$; F1 = harmonic mean.
- **K-Means Inertia:** Within-cluster sum of squares; choose $K$ via elbow curve or silhouette score.
- **DBSCAN:** Core, Border, and Noise points ($-1$); discovers arbitrary cluster contours without fixing $K$.
- **Leak-Proof Architecture:** Package all preprocessing and estimators within a `Pipeline` before cross-validation.

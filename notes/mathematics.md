# Mathematics for Machine Learning

A rigorous concept reference and revision guide detailing the foundational mathematics powering machine learning: Linear Algebra, Multivariable Calculus, Numerical Optimization, Probability Theory, and Statistical Inference.

Related deep-dive topic guides:
- **Linear Algebra:** [linear-algebra-notes.md](file:///mnt/E_Drive/Projects/ml-learning/mathematics/linear-algebra/linear-algebra-notes.md) | [scalars-vectors.md](file:///mnt/E_Drive/Projects/ml-learning/mathematics/linear-algebra/scalars-vectors.md) | [vector-magnitude-euclidean-distance.md](file:///mnt/E_Drive/Projects/ml-learning/mathematics/linear-algebra/vector-magnitude-euclidean-distance.md) | [matrices.md](file:///mnt/E_Drive/Projects/ml-learning/mathematics/linear-algebra/matrices.md) | [matrix-operations.md](file:///mnt/E_Drive/Projects/ml-learning/mathematics/linear-algebra/matrix-operations.md) | [matrix-transpose.md](file:///mnt/E_Drive/Projects/ml-learning/mathematics/linear-algebra/matrix-transpose.md) | [special-matrices.md](file:///mnt/E_Drive/Projects/ml-learning/mathematics/linear-algebra/special-matrices.md)
- **Calculus:** [calculus.md](file:///mnt/E_Drive/Projects/ml-learning/mathematics/calculus/calculus.md)
- **Optimization:** [optimization.md](file:///mnt/E_Drive/Projects/ml-learning/mathematics/optimization/optimization.md)
- **Probability:** [probability-notes.md](file:///mnt/E_Drive/Projects/ml-learning/mathematics/probability/probability-notes.md)
- **Statistics:** [statistics-notes.md](file:///mnt/E_Drive/Projects/ml-learning/mathematics/statistics/statistics-notes.md)

---

## 1. Mathematical Pillars of Machine Learning

```
                            Mathematics in ML
                                   │
      ┌────────────────┬───────────┴───────────┬────────────────┐
      ▼                ▼                       ▼                ▼
Linear Algebra     Calculus               Optimization     Probability & Statistics
  Data spaces,      Parameter gradients,   Cost minimization, Quantifying uncertainty,
  projections,      sensitivity analysis,  loss landscapes,   hypothesis testing,
  transformations   backpropagation        convergence        parameter estimation
```

| Mathematical Pillar | Core Role in ML | Direct Algorithmic Applications |
|---|---|---|
| **Linear Algebra** | Vectorized representation of samples, features, weights, and high-dimensional transformations. | Feature matrices \(X\), dot products in linear classifiers, PCA dimensionality reduction, SVD. |
| **Multivariable Calculus** | Measuring rates of change across multi-parameter loss surfaces. | Gradient vectors (\(\nabla \mathcal{L}\)), Chain Rule in neural networks (backpropagation), Hessian curvature. |
| **Optimization** | Iterative algorithms to identify optimal parameter configurations that minimize objective losses. | Gradient Descent (Batch, Mini-batch, SGD), convex analysis, regularized loss constraints. |
| **Probability Theory** | Quantifying uncertainty, stochastic phenomena, and modeling data distributions. | Bayes' Theorem, Naive Bayes, Logistic Regression probabilities, Gaussian generative models. |
| **Statistical Inference** | Rigorous evaluation, hypothesis testing, estimator properties, and generalization proofs. | Central Limit Theorem, confidence intervals, A/B testing, MLE and MAP estimation, p-values. |

---

## 2. Linear Algebra

Linear algebra provides the computational grammar of machine learning.

### 1. Vectors and Spaces

- **Scalar (\(s \in \mathbb{R}\)):** Single numeric quantity (rank-0 tensor).
- **Vector (\(v \in \mathbb{R}^d\)):** Ordered sequence of \(d\) real numbers representing a point or directional displacement in \(d\)-dimensional space (rank-1 tensor):
  \[
  v = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_d \end{bmatrix} \in \mathbb{R}^d
  \]
- **Feature Vector Representation:** An individual observation with \(d\) attributes is represented as a column vector \(x \in \mathbb{R}^d\).

### 2. Vector Operations & Dot Product

Let \(u, v \in \mathbb{R}^d\):

- **Vector Addition:** \(u + v = [u_1 + v_1, \dots, u_d + v_d]^T\).
- **Scalar Multiplication:** \(\alpha u = [\alpha u_1, \dots, \alpha u_d]^T\).
- **Dot Product (Inner Product):**
  - **Algebraic definition:**
    \[
    u \cdot v = u^T v = \sum_{i=1}^d u_i v_i = u_1 v_1 + u_2 v_2 + \dots + u_d v_d
    \]
  - **Geometric definition:**
    \[
    u \cdot v = \|u\| \|v\| \cos(\theta)
    \]
    Where \(\theta\) is the angle between the two vectors.
  - **Orthogonality:** Two non-zero vectors are perpendicular (orthogonal) if and only if \(u \cdot v = 0\).
  - **ML Application:** The fundamental linear prediction is a dot product: \(\hat{y} = w^T x + b\).

### 3. Vector Norms & Distance Metrics

A norm \(\|\cdot\|\) maps a vector to a non-negative scalar measuring its length/magnitude.

| Norm | Name | Mathematical Formula | ML Application |
|---|---|---|---|
| **\(L_1\) Norm** | Manhattan / Taxicab | \(\|v\|_1 = \sum_{i=1}^d \|v_i\|\) | Lasso Regularization (induces weight sparsity). |
| **\(L_2\) Norm** | Euclidean Norm | \(\|v\|_2 = \sqrt{\sum_{i=1}^d v_i^2} = \sqrt{v^T v}\) | Ridge Regularization, Euclidean distance, SVM margins. |
| **\(L_\infty\) Norm** | Max / Chebyshev | \(\|v\|_\infty = \max_{i} \|v_i\|\) | Adversarial perturbation bounds, worst-case error. |
| **Cosine Similarity** | Angular Metric | \(\cos(\theta) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}\) | Text embeddings, recommendation systems (scale-invariant). |

```python
import numpy as np

u = np.array([3.0, 4.0])
v = np.array([1.0, 2.0])

# Norms
print("L1 norm:", np.linalg.norm(u, ord=1))       # 7.0
print("L2 norm:", np.linalg.norm(u, ord=2))       # 5.0
print("L-inf norm:", np.linalg.norm(u, ord=np.inf)) # 4.0

# Cosine similarity
cos_sim = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
print("Cosine similarity:", cos_sim)
```

### 4. Matrices and Matrix Transformations

A matrix \(A \in \mathbb{R}^{m \times n}\) is a 2D grid with \(m\) rows and \(n\) columns.

#### Feature Matrix Convention
A dataset of \(n\) observations each with \(d\) features is represented as an \(n \times d\) matrix:

\[
X = \begin{bmatrix}
x_{11} & x_{12} & \dots & x_{1d} \\
x_{21} & x_{22} & \dots & x_{2d} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \dots & x_{nd}
\end{bmatrix} \in \mathbb{R}^{n \times d}
\]

#### Matrix Multiplication
For \(A \in \mathbb{R}^{m \times k}\) and \(B \in \mathbb{R}^{k \times n}\), the product \(C = AB \in \mathbb{R}^{m \times n}\) is defined by:

\[
C_{ij} = \sum_{l=1}^k A_{il} B_{lj}
\]

- **Properties:**
  - Non-commutative: \(AB \neq BA\) in general.
  - Associative: \(A(BC) = (AB)C\).
  - Distributive: \(A(B + C) = AB + AC\).

#### Transpose and Symmetry
- **Transpose (\(A^T\)):** Swaps rows and columns: \((A^T)_{ij} = A_{ji}\).
- **Properties:**
  - \((A^T)^T = A\)
  - \((AB)^T = B^T A^T\) (order reverses!)
  - \((A + B)^T = A^T + B^T\)
- **Symmetric Matrix:** A square matrix where \(A = A^T\). Covariance matrices \(X^T X\) are always symmetric and positive semi-definite.

### 5. Matrix Inverses & Solving Linear Systems

For a square matrix \(A \in \mathbb{R}^{n \times n}\), its inverse \(A^{-1}\) satisfies:

\[
A A^{-1} = A^{-1} A = I_n
\]

- An inverse exists if and only if \(\det(A) \neq 0\) (matrix is non-singular / full rank).
- Solving linear system \(Ax = b \implies x = A^{-1}b\).
- **Moore-Penrose Pseudo-Inverse (\(A^+\)):** For rectangular or singular matrices:
  \[
  A^+ = (A^T A)^{-1} A^T
  \]
  Used directly in the Ordinary Least Squares (OLS) closed-form solution: \(w = (X^T X)^{-1} X^T y\).

### 6. Eigenvalues and Eigenvectors

For a square matrix \(A \in \mathbb{R}^{n \times n}\), a non-zero vector \(v \in \mathbb{R}^n\) is an **eigenvector** with corresponding **eigenvalue** \(\lambda \in \mathbb{R}\) if:

\[
A v = \lambda v
\]

- **Geometric Meaning:** Matrix \(A\) transforms vector \(v\) without changing its directional axis; it only stretches or shrinks \(v\) by scalar factor \(\lambda\).
- **Characteristic Equation:** \(\det(A - \lambda I) = 0\).
- **Spectral Theorem for Real Symmetric Matrices:** If \(A\) is real and symmetric, all its eigenvalues are real, and its eigenvectors are mutually orthogonal.
- **Connection to Principal Component Analysis (PCA):**
  The sample covariance matrix \(\Sigma = \frac{1}{n} X^T X\) is symmetric. The eigenvectors of \(\Sigma\) represent the principal directions of maximal variance, and the eigenvalues quantify the variance explained along each axis.

---

## 3. Multivariable Calculus

Calculus provides the mathematical tools to evaluate and improve model parameters iteratively.

### 1. Derivatives and Rates of Change

The derivative of single-variable function \(f(x)\) measures the instantaneous rate of change:

\[
f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
\]

#### Essential Derivative Rules

| Rule | Function \(f(x)\) | Derivative \(f'(x)\) |
|---|---|---|
| **Power Rule** | \(x^n\) | \(n x^{n-1}\) |
| **Constant Multiple** | \(c \cdot g(x)\) | \(c \cdot g'(x)\) |
| **Sum Rule** | \(g(x) + h(x)\) | \(g'(x) + h'(x)\) |
| **Product Rule** | \(g(x) h(x)\) | \(g'(x)h(x) + g(x)h'(x)\) |
| **Quotient Rule** | \(\frac{g(x)}{h(x)}\) | \(\frac{g'(x)h(x) - g(x)h'(x)}{[h(x)]^2}\) |
| **Exponential** | \(e^{ax}\) | \(a e^{ax}\) |
| **Natural Log** | \(\ln(x)\) | \(\frac{1}{x} \quad (x > 0)\) |
| **Sigmoid** | \(\sigma(x) = \frac{1}{1 + e^{-x}}\) | \(\sigma(x)(1 - \sigma(x))\) |

---

### 2. The Chain Rule

For composite functions \(f(x) = g(h(x))\):

\[
\frac{df}{dx} = \frac{dg}{dh} \cdot \frac{dh}{dx}
\]

In multivariable settings where \(z = f(x, y)\) with \(x = g(t)\) and \(y = h(t)\):

\[
\frac{dz}{dt} = \frac{\partial z}{\partial x} \frac{dx}{dt} + \frac{\partial z}{\partial y} \frac{dy}{dt}
\]

**ML Significance:** The Chain Rule enables **Backpropagation**, allowing gradient computation across composed layers in deep neural networks and complex pipelines.

---

### 3. Partial Derivatives and the Gradient Vector

When a scalar function depends on multiple variables \(f(x_1, x_2, \dots, x_d)\), the **partial derivative** \(\frac{\partial f}{\partial x_i}\) measures change with respect to \(x_i\) while treating all other variables as constants.

#### The Gradient Vector (\(\nabla f\))
The vector of all first-order partial derivatives:

\[
\nabla f(x) = \begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_d}
\end{bmatrix} \in \mathbb{R}^d
\]

#### Fundamental Properties of the Gradient
1. **Direction of Steepest Ascent:** \(\nabla f(x)\) points in the direction where \(f\) increases most rapidly locally.
2. **Direction of Steepest Descent:** \(-\nabla f(x)\) points in the direction where \(f\) decreases most rapidly locally.
3. **Magnitude:** \(\|\nabla f(x)\|_2\) is the maximum directional derivative (rate of steepest ascent).
4. **Stationary Points:** If \(\nabla f(x) = \mathbf{0}\), the point is a local minimum, local maximum, or saddle point.

---

### 4. Second Derivatives and the Hessian Matrix

The **Hessian matrix** \(H \in \mathbb{R}^{d \times d}\) collects all second-order partial derivatives of a scalar function:

\[
H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
\]

\[
H(f) = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \dots & \frac{\partial^2 f}{\partial x_1 \partial x_d} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \dots & \frac{\partial^2 f}{\partial x_2 \partial x_d} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_d \partial x_1} & \frac{\partial^2 f}{\partial x_d \partial x_2} & \dots & \frac{\partial^2 f}{\partial x_d^2}
\end{bmatrix}
\]

- By Schwarz's theorem, if second derivatives are continuous, \(H\) is symmetric (\(\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}\)).
- **Curvature Classification at a Stationary Point (\(\nabla f = 0\)):**
  - **Positive Definite (\(H \succ 0\), all \(\lambda_i > 0\)):** Strictly convex local minimum.
  - **Negative Definite (\(H \prec 0\), all \(\lambda_i < 0\)):** Strictly concave local maximum.
  - **Indefinite (some \(\lambda_i > 0\), some \(\lambda_j < 0\)):** Saddle point.

---

## 4. Optimization

Machine learning training translates directly into solving an optimization problem:

\[
\theta^* = \arg\min_{\theta} \mathcal{L}(\theta)
\]

### 1. Convexity: The Ideal Landscape

```
       Convex Function                         Non-Convex Function
            Loss                                      Loss
             ▲                                         ▲
             │ \             /                         │  /\       /\
             │  \     _     /                          │ /  \  _  /  \
             │   \___/ \___/                           │/    \/ \/    \
             └───────────────► w                       └───────────────► w
          Single Global Minimum                     Multiple Local Minima &
        (Gradient points to goal)                        Saddle Points
```

#### Mathematical Definition of Convexity
A function \(f: \mathbb{R}^d \to \mathbb{R}\) is convex if for all \(x, y \in \text{dom}(f)\) and \(\alpha \in [0, 1]\):

\[
f(\alpha x + (1 - \alpha) y) \leq \alpha f(x) + (1 - \alpha) f(y)
\]

- **Crucial ML Theorem:** For any convex optimization problem, **every local minimum is guaranteed to be a global minimum**.
- **Convex Models in Classical ML:** Ordinary Least Squares Linear Regression, Ridge Regression, Lasso Regression, Logistic Regression, Support Vector Machines (dual formulation).
- **Non-Convex Models:** Deep Neural Networks, K-Means clustering.

---

### 2. Gradient Descent Variants

Gradient descent updates parameter vector \(\theta\) in the negative gradient direction:

\[
\theta^{(t+1)} = \theta^{(t)} - \eta \nabla \mathcal{L}(\theta^{(t)})
\]

Where \(\eta > 0\) is the **learning rate**.

#### Impact of Learning Rate (\(\eta\))
- **\(\eta\) too small:** Extremely slow convergence; high computational cost; easily trapped in saddle regions.
- **\(\eta\) too large:** Parameter updates overshoot the minimum, oscillate wildly, and may diverge to infinity.
- **Learning Rate Schedules:** Decaying \(\eta\) over training time (\(\eta_t = \frac{\eta_0}{1 + k t}\) or cosine annealing).

#### Comparative Analysis of Gradient Descent Variants

| Variant | Gradient Batch Size | Pros | Cons |
|---|---|---|---|
| **Batch Gradient Descent (BGD)** | Full dataset (\(n\) samples) | Deterministic, smooth convergence trajectory; exact gradient calculation. | Computationally slow and memory-intensive for large datasets; can get stuck in saddle points. |
| **Stochastic Gradient Descent (SGD)** | 1 sample per step | Extremely fast step updates; low memory requirement; noise helps escape shallow local minima. | High gradient variance; erratic, oscillating convergence trajectory; does not settle without learning rate decay. |
| **Mini-Batch Gradient Descent** | Small subset \(B\) (e.g., 32, 64, 128) | Balances vectorization throughput with gradient stability; standard default in modern ML. | Introduces batch-size hyperparameter to tune. |

---

## 5. Probability Theory

Probability provides the mathematical framework for handling uncertainty, modeling noisy observations, and learning from finite samples.

### 1. Axioms of Probability (Kolmogorov)

For sample space \(\Omega\) and events \(A \subseteq \Omega\):
1. **Non-negativity:** \(P(A) \geq 0\).
2. **Total Probability:** \(P(\Omega) = 1\).
3. **Countable Additivity:** If \(A_1, A_2, \dots\) are mutually disjoint events, then:
   \[
   P\left( \bigcup_{i=1}^\infty A_i \right) = \sum_{i=1}^\infty P(A_i)
   \]

### 2. Conditional Probability & Independence

- **Conditional Probability:** Probability of event \(A\) given that event \(B\) has occurred:
  \[
  P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \qquad P(B) > 0
  \]
- **Statistical Independence:** Knowing \(B\) occurred provides zero information about \(A\):
  \[
  P(A \mid B) = P(A) \iff P(A \cap B) = P(A) P(B)
  \]
- **Conditional Independence:** Given \(C\), knowledge of \(B\) provides no additional information about \(A\):
  \[
  P(A \cap B \mid C) = P(A \mid C) P(B \mid C)
  \]
  *ML Foundation:* The core assumption of the **Naive Bayes Classifier** (features are conditionally independent given the class label).

---

### 3. Bayes' Theorem

Reverses conditional probabilities, converting prior beliefs into posterior probabilities after observing evidence:

\[
P(\theta \mid D) = \frac{P(D \mid \theta) P(\theta)}{P(D)} = \frac{P(D \mid \theta) P(\theta)}{\int P(D \mid \theta') P(\theta') d\theta'}
\]

```
  Posterior           Likelihood        Prior
  P(θ | D)     =      P(D | θ)    *     P(θ)     /     P(D)
 (Updated belief     (Data probability (Belief before    (Evidence /
  given data D)       under parameter θ) observing data) Marginal likelihood)
```

---

### 4. Random Variables, Expectations, and Moments

- **Discrete Random Variable:** Takes values from a countable set; described by a **Probability Mass Function (PMF)** \(p(x) = P(X = x)\).
- **Continuous Random Variable:** Takes values across continuous intervals; described by a **Probability Density Function (PDF)** \(f(x)\) where \(P(a \leq X \leq b) = \int_a^b f(x) dx\).

#### Expectation (\(\mathbb{E}[X]\))
The probability-weighted average of all possible values:

\[
\mathbb{E}[X] = \begin{cases} \sum_{i} x_i P(X = x_i) & \text{(discrete)} \\ \int_{-\infty}^\infty x f(x) dx & \text{(continuous)} \end{cases}
\]

- **Linearity of Expectation:** \(\mathbb{E}[\alpha X + \beta Y] = \alpha \mathbb{E}[X] + \beta \mathbb{E}[Y]\) (holds even if \(X\) and \(Y\) are dependent!).

#### Variance and Standard Deviation
Measures dispersion around the expected value:

\[
\text{Var}(X) = \sigma^2 = \mathbb{E}\big[ (X - \mathbb{E}[X])^2 \big] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
\]

\[
\sigma = \sqrt{\text{Var}(X)}
\]

#### Covariance and Correlation
Measures linear association between two random variables \(X\) and \(Y\):

\[
\text{Cov}(X, Y) = \mathbb{E}\big[ (X - \mathbb{E}[X])(Y - \mathbb{E}[Y]) \big] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]
\]

\[
\rho_{X, Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]
\]

---

### 5. Essential Distributions in Machine Learning

| Distribution | Type | PDF / PMF | Parameters | ML Context |
|---|---|---|---|---|
| **Bernoulli** | Discrete | \(P(X=k) = p^k (1-p)^{1-k}, \quad k \in \{0, 1\}\) | \(p \in [0, 1]\) | Binary classification targets, coin tosses. |
| **Binomial** | Discrete | \(P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}\) | \(n \in \mathbb{N}, p \in [0, 1]\) | Count of positive labels in \(n\) trials. |
| **Uniform** | Continuous | \(f(x) = \frac{1}{b - a}, \quad x \in [a, b]\) | \(a, b \in \mathbb{R}\) | Randomized search hyperparameter bounds. |
| **Gaussian (Normal)** | Continuous | \(f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)\) | \(\mu \in \mathbb{R}, \sigma > 0\) | Measurement noise, regression residuals, weight initialization. |
| **Multivariate Normal** | Continuous | \(f(x) = \frac{1}{(2\pi)^{d/2} \|\Sigma\|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)\right)\) | \(\mu \in \mathbb{R}^d, \Sigma \in \mathbb{R}^{d \times d}\) | Gaussian Mixture Models, multivariate features. |

#### Empirical Rule for Normal Distributions (\(\mathcal{N}(\mu, \sigma^2)\))
- \(\mu \pm 1\sigma\): Covers \(\approx 68.27\%\) of the probability mass.
- \(\mu \pm 2\sigma\): Covers \(\approx 95.45\%\) of the probability mass.
- \(\mu \pm 3\sigma\): Covers \(\approx 99.73\%\) of the probability mass.

---

## 6. Statistics & Inference

Statistics provides the principles for drawing sound conclusions from finite, noisy data samples.

### 1. Population vs. Sample & Bessel's Correction

- **Population:** Entire universe of entities under study (\(\mu, \sigma^2\)).
- **Sample:** Finite subset observed in practice (\(\bar{x}, s^2\)).

#### Sample Mean
\[
\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i
\]

#### Unbiased Sample Variance (Bessel's Correction)
Dividing by \(n\) systematically underestimates population variance because deviations are calculated around the sample mean \(\bar{x}\) rather than the true population mean \(\mu\). Dividing by \(n - 1\) yields an unbiased estimator:

\[
s^2 = \frac{1}{n - 1} \sum_{i=1}^n (x_i - \bar{x})^2
\]

---

### 2. Central Limit Theorem (CLT)

Let \(X_1, X_2, \dots, X_n\) be independent and identically distributed (i.i.d.) random variables with mean \(\mu\) and finite variance \(\sigma^2\). As sample size \(n \to \infty\), the normalized sample mean converges in distribution to a standard normal distribution:

\[
Z = \frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)
\]

- **Standard Error of the Mean:** \(\text{SE} = \frac{\sigma}{\sqrt{n}}\).
- **Practical Significance in ML:**
  - Guarantees that cross-validation score averages across folds approximate a normal distribution regardless of individual sample score distributions.
  - Justifies constructing confidence intervals around model benchmark metrics.

---

### 3. Maximum Likelihood Estimation (MLE) vs. MAP

How do we estimate model parameters \(\theta\) given observations \(D = \{x_1, \dots, x_n\}\)?

#### Maximum Likelihood Estimation (MLE)
Finds the parameter configuration that maximizes the likelihood of observing the training data:

\[
\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} L(\theta) = \arg\max_{\theta} \prod_{i=1}^n p(x_i \mid \theta)
\]

Taking the natural logarithm simplifies products into sums:

\[
\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \sum_{i=1}^n \ln p(x_i \mid \theta) = \arg\min_{\theta} \left[ -\sum_{i=1}^n \ln p(x_i \mid \theta) \right]
\]

#### Deriving Linear Regression from Gaussian MLE
Assume true targets are generated by linear function plus Gaussian noise: \(y_i = w^T x_i + \epsilon_i\), where \(\epsilon_i \sim \mathcal{N}(0, \sigma^2)\). The log-likelihood is:

\[
\ln L(w) = -\frac{n}{2} \ln(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^n (y_i - w^T x_i)^2
\]

Maximizing log-likelihood with respect to \(w\) is **mathematically identical to minimizing Mean Squared Error (MSE)**!

#### Maximum A Posteriori (MAP)
Incorporates prior knowledge about parameters via Bayes' Theorem:

\[
\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} P(\theta \mid D) = \arg\max_{\theta} \left[ \sum_{i=1}^n \ln p(x_i \mid \theta) + \ln P(\theta) \right]
\]

- **Mathematical Bridge to Regularization:**
  - Assuming a **Zero-mean Gaussian Prior** \(P(w) \sim \mathcal{N}(0, \tau^2)\) on weights directly yields **\(L_2\) Ridge Regularization**!
  - Assuming a **Laplace Prior** \(P(w) \propto \exp(-\lambda \|w\|_1)\) on weights directly yields **\(L_1\) Lasso Regularization**!

---

### 4. Hypothesis Testing & Significance

Determines whether observed experimental differences between models are statistically significant or attributable to random sampling variability.

```
                         Truth: H0 True             Truth: H0 False
                     ┌──────────────────────────┬──────────────────────────┐
  Decision:          │   Correct Decision       │   Type II Error (β)      │
  Fail to Reject H0  │   (Confidence: 1 - α)    │   (False Negative)       │
                     ├──────────────────────────┼──────────────────────────┤
  Decision:          │   Type I Error (α)       │   Correct Decision       │
  Reject H0          │   (False Positive)       │   (Power: 1 - β)         │
                     └──────────────────────────┴──────────────────────────┘
```

- **Null Hypothesis (\(H_0\)):** Model A and Model B exhibit identical predictive performance.
- **Alternative Hypothesis (\(H_1\)):** Model A performs significantly better than Model B.
- **Significance Level (\(\alpha\)):** Threshold for rejecting \(H_0\) (commonly \(\alpha = 0.05\)).
- **p-value:** The probability of observing an evaluation metric difference as large as or larger than the observed score, assuming \(H_0\) is true.
  - If \(p < \alpha\): Reject \(H_0\); conclude the performance improvement is statistically significant.
  - If \(p \geq \alpha\): Fail to reject \(H_0\); insufficient evidence that models differ.

---

## 7. Master Formula Reference Cheatsheet

| Concept | Mathematical Formula | Usage in ML |
|---|---|---|
| **Dot Product** | \(u^T v = \sum_{i=1}^d u_i v_i\) | Linear score calculation (\(w^T x + b\)). |
| **Euclidean Norm (\(L_2\))** | \(\|x\|_2 = \sqrt{\sum x_i^2}\) | Ridge penalty, Euclidean distance in kNN/K-Means. |
| **Manhattan Norm (\(L_1\))** | \(\|x\|_1 = \sum \|x_i\|\) | Lasso penalty, sparse weight inducement. |
| **Normal Equation** | \(w = (X^T X)^{-1} X^T y\) | Exact analytical solution for OLS linear regression. |
| **Eigenvalues / Eigenvectors** | \(Av = \lambda v\) | Principal Component Analysis (PCA covariance decomposition). |
| **Gradient Vector** | \(\nabla f(x) = [\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_d}]^T\) | Direction of steepest ascent in optimization. |
| **Gradient Descent Step** | \(\theta \leftarrow \theta - \eta \nabla \mathcal{L}(\theta)\) | Parameter update in iterative optimization. |
| **Sigmoid Function** | \(\sigma(z) = \frac{1}{1 + e^{-z}}\) | Probability transformation in Logistic Regression. |
| **Binary Cross-Entropy** | \(-\frac{1}{n} \sum [y \ln \hat{p} + (1 - y) \ln(1 - \hat{p})]\) | Loss function for binary classification. |
| **Bayes' Theorem** | \(P(\theta \mid D) = \frac{P(D \mid \theta) P(\theta)}{P(D)}\) | Posterior estimation, Bayesian inference, MAP. |
| **Gaussian PDF** | \(\frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)\) | Normal distribution density, likelihood modeling. |
| **Sample Variance** | \(s^2 = \frac{1}{n - 1} \sum (x_i - \bar{x})^2\) | Unbiased variance estimation (Bessel's correction). |
| **Pearson Correlation** | \(r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}\) | Linear feature correlation assessment in EDA. |

---

## 8. Common Mathematical Pitfalls in ML

1. **Confusing Non-Singular with Ill-Conditioned Matrices:** Even if \(\det(X^T X) \neq 0\), severe multicollinearity results in a massive condition number, making direct matrix inversion numerically unstable. Use Ridge regression or SVD instead.
2. **Assuming Zero Correlation Implies Independence:** Pearson \(r = 0\) only proves the absence of a **linear** relationship. Variables can share a deterministic non-linear relationship (e.g., \(y = x^2\)) while having zero Pearson correlation.
3. **Misinterpreting p-values:** A p-value is **not** the probability that the null hypothesis is true, nor is it the probability that results occurred by pure chance. It is \(P(\text{Data as extreme} \mid H_0 \text{ is true})\).
4. **Dividing by \(n\) Instead of \(n-1\) for Sample Variance:** Omitting Bessel's correction introduces negative bias in sample variance and standard deviation estimates for small sample sizes.
5. **Ignoring Vanishing/Exploding Gradients:** In deep architectures or unscaled feature spaces, chained derivatives can exponentially shrink to zero or explode, stalling optimization.

---

## 9. Quick Revision Summary

- **Inner Product:** Measures alignment and projection; computes \(\hat{y} = w^T x + b\).
- **Norms:** \(L_1\) norm generates diamond constraints \(\implies\) sparsity; \(L_2\) norm generates circular constraints \(\implies\) weight shrinkage.
- **Normal Equation:** \(w = (X^T X)^{-1} X^T y\); optimal OLS parameters in closed form.
- **PCA:** Eigenvectors of covariance matrix \(\Sigma\) identify axes of maximal data variance.
- **Gradient:** Vector of partial derivatives pointing in the direction of steepest local increase.
- **Convexity:** Guarantees that any local minimum found by gradient descent is the global minimum.
- **Chain Rule:** Enables backpropagation: \(\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}\).
- **Bayes' Theorem:** \(\text{Posterior} \propto \text{Likelihood} \times \text{Prior}\).
- **MLE vs MAP:** MLE finds parameters maximizing data likelihood; MAP adds prior distributions (Gaussian prior = Ridge, Laplace prior = Lasso).
- **CLT:** Regardless of underlying population distribution, the sample mean converges to a normal distribution as \(n \to \infty\).

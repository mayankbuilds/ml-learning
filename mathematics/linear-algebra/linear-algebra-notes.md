# Linear Algebra for Machine Learning

Practical reference for the core linear algebra concepts used in machine learning, with NumPy examples.

## 1. Scalars, vectors, and matrices

- **Scalar:** one number, e.g. `7`.
- **Vector:** an ordered collection of numbers, e.g. `[2, 3, 5]`.
- **Matrix:** a rectangular grid of numbers, e.g. `[[1, 2], [3, 4]]`.

```python
import numpy as np

s = np.array(7)
v = np.array([2, 3, 5])
M = np.array([[1, 2], [3, 4]])

print(s.ndim, s.shape)  # 0, ()
print(v.ndim, v.shape)  # 1, (3,)
print(M.ndim, M.shape)  # 2, (2, 2)
```

In a dataset, rows commonly represent samples and columns represent features. `X.shape == (100, 4)` means 100 samples and 4 features.

## 2. Vector operations

Let `a = [2, 3]`, `b = [4, 1]`.

- Addition: `a + b = [6, 4]`
- Subtraction: `a - b = [-2, 2]`
- Scalar multiplication: `3a = [6, 9]`
- Dot product: `a · b = 2*4 + 3*1 = 11`

```python
a = np.array([2, 3])
b = np.array([4, 1])

print(a + b)        # [6 4]
print(a - b)        # [-2  2]
print(3 * a)        # [6 9]
print(np.dot(a, b)) # 11
```

The dot product returns a single number. In ML, weighted sums such as `w · x + b` appear in linear models and neural-network units.

## 3. Magnitude (norm) and Euclidean distance

The Euclidean norm measures vector length:

$$
\|v\|_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}
$$

For `[3, 4]`, the norm is `5`.

Distance between vectors is the norm of their difference:

$$
d(a,b) = \|a-b\|_2
$$

For `a = [1, 2]` and `b = [4, 6]`, the distance is `sqrt(3² + 4²) = 5`.

```python
v = np.array([3, 4])
a = np.array([1, 2])
b = np.array([4, 6])

print(np.linalg.norm(v))      # 5.0
print(np.linalg.norm(b - a))  # 5.0
```

ML connection: distances are used in nearest-neighbor methods and clustering. Feature scales matter, so scaling may be necessary.

## 4. Matrix shape, indexing, and operations

For `A.shape == (m, n)`, the matrix has `m` rows and `n` columns.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print(A.shape)  # (2, 3)
print(A[0, 1])  # 2: first row, second column
```

For matrices of the same shape:

- `A + B`: element-wise addition
- `A - B`: element-wise subtraction
- `k * A`: scalar multiplication
- `A * B`: element-wise multiplication in NumPy
- `A @ B`: matrix multiplication

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print(A + B)  # [[ 6  8], [10 12]]
print(A * B)  # [[ 5 12], [21 32]]
print(A @ B)  # [[19 22], [43 50]]
```

Matrix multiplication shape rule:

$$
(m,n) @ (n,p) \longrightarrow (m,p)
$$

The inner dimensions must match: columns of the first matrix equal rows of the second. The output has the outer dimensions.

## 5. Transpose

Transpose swaps rows and columns. In NumPy, use `.T`.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print(A.T)
# [[1 4]
#  [2 5]
#  [3 6]]

print(A.shape)    # (2, 3)
print(A.T.shape)  # (3, 2)
```

Transposing twice returns the original matrix: `(A.T).T == A`.

## 6. Special matrices

- **Zero matrix:** every entry is zero.
- **Identity matrix:** ones on the main diagonal, zeros elsewhere; acts like 1 in matrix multiplication when dimensions match.
- **Diagonal matrix:** non-zero entries can appear only on the main diagonal.

```python
Z = np.zeros((2, 3))
I = np.eye(3)
D = np.diag([2, 3, 4])

print(I @ D)  # D
```

## 7. Determinant and inverse

For a 2×2 matrix,

$$
A=\begin{bmatrix}a&b\\c&d\end{bmatrix},
\qquad \det(A)=ad-bc
$$

Example:

$$
\det\begin{bmatrix}2&3\\1&4\end{bmatrix}
=2(4)-3(1)=5
$$

A square matrix with a non-zero determinant is invertible. If its determinant is zero, its ordinary inverse does not exist.

```python
A = np.array([[2, 3],
              [1, 4]])

print(np.linalg.det(A))  # approximately 5.0
print(np.linalg.inv(A))
```

Floating-point results may be very close to zero rather than exactly zero.

## 8. Linear transformations

A matrix can transform a vector:

$$
\begin{bmatrix}2&0\\0&3\end{bmatrix}
\begin{bmatrix}1\\2\end{bmatrix}
=
\begin{bmatrix}2\\6\end{bmatrix}
$$

```python
A = np.array([[2, 0],
              [0, 3]])
x = np.array([1, 2])

print(A @ x)  # [2 6]
```

This matrix scales the first coordinate by 2 and the second by 3. In ML, matrix multiplications transform feature vectors and are central to neural-network layers.

## 9. Eigenvalues and eigenvectors

A non-zero vector `v` is an eigenvector of a square matrix `A` if applying `A` changes only its scale, not its direction:

$$
Av = \lambda v
$$

Here, $\lambda$ is the eigenvalue, or scale factor.

For `A = [[2, 0], [0, 3]]`, `[1, 0]` is an eigenvector with eigenvalue 2, and `[0, 1]` is an eigenvector with eigenvalue 3.

```python
A = np.array([[2, 0],
              [0, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)   # [2. 3.]
print(eigenvectors)  # eigenvectors are stored as columns
```

ML connection: PCA uses eigenvectors and eigenvalues to identify important directions in data and the amount of variation along those directions.

## 10. Common mistakes

1. Confusing `A * B` (element-wise in NumPy) with `A @ B` (matrix multiplication).
2. Forgetting that matrix multiplication requires matching inner dimensions.
3. Confusing a vector's norm with the sum of its entries.
4. Assuming `.T` modifies the original array in place.
5. Assuming every square matrix has an inverse.
6. Forgetting that eigenvectors are stored as columns in NumPy's `np.linalg.eig` output.

## 11. Quick reference

| Concept | NumPy |
|---|---|
| Shape | `A.shape` |
| Number of dimensions | `A.ndim` |
| Dot product | `np.dot(a, b)` or `a @ b` |
| Norm | `np.linalg.norm(v)` |
| Transpose | `A.T` |
| Element-wise multiplication | `A * B` |
| Matrix multiplication | `A @ B` |
| Identity matrix | `np.eye(n)` |
| Diagonal matrix | `np.diag(values)` |
| Determinant | `np.linalg.det(A)` |
| Inverse | `np.linalg.inv(A)` |
| Eigenvalues/eigenvectors | `np.linalg.eig(A)` |

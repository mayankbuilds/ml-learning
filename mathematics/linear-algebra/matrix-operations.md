# Matrix Operations

**Location:** `mathematics/linear-algebra/matrix-operations.md`

## 1. Matrix Addition

Two matrices with the same shape can be added by adding corresponding elements.

\[
A=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad
B=\begin{bmatrix}5&6\\7&8\end{bmatrix}
\]

\[
A+B=\begin{bmatrix}6&8\\10&12\end{bmatrix}
\]

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print(A + B)
```

Ordinary matrix addition requires matching shapes. NumPy broadcasting is a separate feature that can allow some different-shaped arrays to be added.

## 2. Matrix Subtraction

Subtract corresponding elements. For ordinary subtraction, the shapes must match.

\[
A-B=\begin{bmatrix}-4&-4\\-4&-4\end{bmatrix}
\]

```python
print(A - B)
```

Subtraction is generally not commutative: `A - B` is usually different from `B - A`.

## 3. Scalar Multiplication

Multiply every element of a matrix by the same scalar.

\[
3A=\begin{bmatrix}3&6\\9&12\end{bmatrix}
\]

```python
print(3 * A)
```

## 4. Element-wise Multiplication vs Matrix Multiplication

These are different operations.

### Element-wise multiplication

Multiply values at matching positions. NumPy uses `*` for this.

\[
A*B=\begin{bmatrix}1\cdot5&2\cdot6\\3\cdot7&4\cdot8\end{bmatrix}
=\begin{bmatrix}5&12\\21&32\end{bmatrix}
\]

```python
print(A * B)
```

### Standard matrix multiplication

For `A @ B`, the number of columns in `A` must equal the number of rows in `B`.

If `A` has shape `(m, n)` and `B` has shape `(n, p)`, then `A @ B` has shape `(m, p)`.

For the example above:

\[
AB=\begin{bmatrix}
1\cdot5+2\cdot7 & 1\cdot6+2\cdot8\\
3\cdot5+4\cdot7 & 3\cdot6+4\cdot8
\end{bmatrix}
=\begin{bmatrix}19&22\\43&50\end{bmatrix}
\]

```python
print(A @ B)
# Equivalent for these 2-D arrays:
print(np.matmul(A, B))
```

The result is different from `A * B`.

### Shape rule

- `(2, 3) @ (3, 4)` is valid and produces shape `(2, 4)`.
- `(2, 3) @ (2, 4)` is invalid because the inner dimensions `3` and `2` do not match.

For matrices, `A @ B` is generally not equal to `B @ A`; multiplication order matters.

## Practice

Run this in Python:

```python
import numpy as np

A = np.array([[2, 1],
              [0, 3]])
B = np.array([[4, 2],
              [5, 1]])

print("Addition:\\n", A + B)
print("Subtraction:\\n", A - B)
print("Scalar multiplication:\\n", 2 * A)
print("Element-wise multiplication:\\n", A * B)
print("Matrix multiplication:\\n", A @ B)
print("Result shape:", (A @ B).shape)
```

Expected results for checking your code:

- `A + B` → `[[6, 3], [5, 4]]`
- `A - B` → `[[-2, -1], [-5, 2]]`
- `2 * A` → `[[4, 2], [0, 6]]`
- `A * B` → `[[8, 2], [0, 3]]`
- `A @ B` → `[[13, 5], [15, 3]]`
- `(A @ B).shape` → `(2, 2)`

## ML connection

Feature matrices and weight matrices often interact through matrix multiplication. Understanding the shape rule helps you check whether the dimensions of inputs and weights are compatible.

## Quick recap

- `A + B`: corresponding elements add; ordinary addition needs equal shapes.
- `A - B`: corresponding elements subtract.
- `k * A`: every element is scaled by `k`.
- `A * B`: NumPy element-wise multiplication.
- `A @ B`: standard matrix multiplication; inner dimensions must match.

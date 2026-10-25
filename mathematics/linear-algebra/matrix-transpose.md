# Transpose of a Matrix

**Save to:** `mathematics/linear-algebra/matrix-transpose.md`

## 1. What is a transpose?

The transpose swaps a matrix's rows and columns. It is written as `A.T` in NumPy and as $A^T$ in mathematics.

Example:

    A = [[1, 2, 3],
         [4, 5, 6]]

    A.T = [[1, 4],
           [2, 5],
           [3, 6]]

The original shape is `(2, 3)` and the transpose shape is `(3, 2)`.

## 2. NumPy

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])

print("Original:\n", A)
print("Original shape:", A.shape)

AT = A.T

print("Transpose:\n", AT)
print("Transpose shape:", AT.shape)
```

Expected transpose:

```text
[[1 4]
 [2 5]
 [3 6]]
```

Expected shape: `(3, 2)`.

## 3. Key rules

- A matrix with shape `(m, n)` has transpose shape `(n, m)`.
- The element at row `i`, column `j` moves to row `j`, column `i`.
- Transposing twice returns the original matrix: `(A.T).T` equals `A`.
- A square matrix stays the same shape, though its values may move.

## 4. ML connection

Rows often represent samples and columns represent features. Transposing changes that orientation, which can be useful when an operation expects features and samples in the opposite arrangement. Always check shapes before matrix multiplication.

## 5. Practice

```python
import numpy as np

A = np.array([[2, 4, 6],
              [1, 3, 5]])

print(A.T)
print("Shape:", A.T.shape)
print("Transpose twice:\n", A.T.T)
```

Expected output for `A.T`:

```text
[[2 1]
 [4 3]
 [6 5]]
```

Shape: `(3, 2)`. Transpose twice should return the original matrix.

## Quick recap

- Transpose swaps rows and columns.
- NumPy syntax: `A.T`
- Shape `(m, n)` becomes `(n, m)`.
- `(A.T).T` returns `A`.

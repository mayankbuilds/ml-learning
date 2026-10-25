# Special Matrices: Zero, Identity, and Diagonal

**Save to:** `mathematics/linear-algebra/special-matrices.md`

## 1. Zero matrix

Every element is zero.

```python
import numpy as np

Z = np.zeros((2, 2))
print(Z)
```

For integer zeros: `np.zeros((2, 2), dtype=int)`.

## 2. Identity matrix

A square matrix with ones on the main diagonal and zeros elsewhere.

$$
I=\begin{bmatrix}1&0&0\\0&1&0\\0&0&1\end{bmatrix}
$$

```python
I = np.eye(3)
print(I)
```

For a compatible matrix `A`, multiplying by the appropriately sized identity matrix preserves it: `A @ I = A` and `I @ A = A`.

## 3. Diagonal matrix

All entries outside the main diagonal are zero. Diagonal entries can be any values, including zero.

```python
D = np.diag([2, 5, 3])
print(D)
```

This creates:

$$
D=\begin{bmatrix}2&0&0\\0&5&0\\0&0&3\end{bmatrix}
$$

Every identity matrix is diagonal, but a diagonal matrix is not necessarily an identity matrix.

## 4. Practice

```python
import numpy as np

Z = np.zeros((3, 3), dtype=int)
I = np.eye(3, dtype=int)
D = np.diag([4, 0, 7])

print("Zero matrix:\\n", Z)
print("Identity matrix:\\n", I)
print("Diagonal matrix:\\n", D)
print("I @ D:\\n", I @ D)
```

Check that the zero matrix contains only zeros, the identity matrix has ones on its main diagonal, and the diagonal matrix has zeros outside its main diagonal. `I @ D` should equal `D`.

## ML connection

- Zero matrices are useful in initialization and mathematical operations.
- Identity matrices represent the identity transformation and simplify matrix multiplication.
- Diagonal matrices represent transformations that scale coordinates along diagonal directions.

## Quick recap

- `np.zeros((m, n))` creates a zero matrix.
- `np.eye(n)` creates an identity matrix.
- `np.diag([a, b, c])` creates a diagonal matrix from the supplied diagonal values.

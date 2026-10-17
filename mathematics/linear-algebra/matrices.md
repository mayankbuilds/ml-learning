# Matrices

## 1. What is a matrix?

A **matrix** is a rectangular arrangement of values organized into rows and columns.

Example:

```text
A = [[85, 90],
     [78, 88],
     [92, 81]]
```

This could represent marks for three students across two subjects.

- Each row represents one student/sample.
- Each column represents one subject/feature.

## 2. Matrix shape and dimensions

The matrix above has 3 rows and 2 columns, so its shape is `(3, 2)` and it is a 3 × 2 matrix.

In general, an `m × n` matrix has `m` rows and `n` columns. For a matrix containing real numbers, we may write `A ∈ R^(m×n)`.

Remember: shape is written as **(rows, columns)**.

## 3. Creating a matrix with NumPy

```python
import numpy as np

A = np.array([
    [85, 90],
    [78, 88],
    [92, 81]
])

print(A)
print("Shape:", A.shape)       # (3, 2)
print("Dimensions:", A.ndim)   # 2
print("Rows:", A.shape[0])     # 3
print("Columns:", A.shape[1])  # 2
print("Size:", A.size)         # 6
```

- `shape`: length of each axis, e.g. `(3, 2)`.
- `ndim`: number of axes, e.g. `2`.
- `size`: total number of elements, e.g. `6`.

## 4. Accessing matrix elements

NumPy indexing starts at zero.

```python
print(A[0, 0])  # 85: first row, first column
print(A[1, 0])  # 78: second row, first column
print(A[2, 1])  # 81: third row, second column

print(A[0])     # [85 90]: first row
print(A[:, 0])  # [85 78 92]: first column
```

The expression `A[row, column]` selects one element. A colon `:` selects all entries along that axis.

## 5. Matrices in machine learning

A feature matrix often stores multiple samples in rows and their features in columns.

For example, three houses described by area, number of bedrooms, and age:

```python
X = np.array([
    [1200, 2, 10],
    [1800, 3, 5],
    [900, 1, 15]
])

print(X.shape)  # (3, 3)
```

Here, each row represents a house and each column represents a feature. The target values (such as prices) are commonly stored separately, often in an array called `y`.

Feature units and scales can differ, so preprocessing may be needed for some ML algorithms.

## 6. Quick reference

| Concept | Meaning | Example |
|---|---|---|
| Matrix | Values arranged in rows and columns | `[[1, 2], [3, 4]]` |
| Shape | `(rows, columns)` | `(3, 2)` |
| `ndim` | Number of axes | `2` for a matrix |
| `size` | Total number of elements | `6` for a 3 × 2 matrix |
| `A[i, j]` | Element at row `i`, column `j` | `A[0, 0]` |
| `A[:, 0]` | All elements in the first column | First column |

## 7. Practice tasks

1. Create a NumPy matrix with two rows and three columns.
2. Print its `shape`, `ndim`, and `size`.
3. Print the element in the second row and third column.
4. Select the first row and the second column.
5. Create a feature matrix for four students with three subject scores each. Check its shape.

## Key takeaways

- A matrix is organized into rows and columns.
- Shape is written as `(rows, columns)`.
- A regular matrix has two axes, so its NumPy `ndim` is `2`.
- NumPy uses zero-based indexing.
- In many ML datasets, rows represent samples and columns represent features.

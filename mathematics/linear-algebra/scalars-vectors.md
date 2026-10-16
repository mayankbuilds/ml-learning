# Scalars and Vectors

## 1. Scalars

A **scalar** is a single numerical value.

Examples:
- Temperature: `28`
- Exam score: `85`
- Learning rate: `0.01`

Scalars are commonly represented by lowercase symbols such as `x`, `a`, or `c`.

In NumPy:

```python
import numpy as np

learning_rate = np.array(0.01)

print(learning_rate)       # 0.01
print(learning_rate.ndim)  # 0
print(learning_rate.shape) # ()
```

A zero-dimensional NumPy array has no axes. Its shape is `()`.

## 2. Vectors

A **vector** is an ordered collection of numerical values.

For example, marks in three subjects can be represented as:

`x = [85, 90, 78]`

The vector has three elements, and their order matters. A vector can also be written as a column vector.

In NumPy:

```python
import numpy as np

marks = np.array([85, 90, 78])

print(marks)       # [85 90 78]
print(marks.ndim)  # 1
print(marks.shape) # (3,)
print(marks.size)  # 3
```

- `ndim`: number of dimensions/axes.
- `shape`: size along each axis.
- `size`: total number of elements.

For a one-dimensional array with three elements, `shape` is `(3,)`.

## 3. Vector magnitude

For a two-dimensional vector `v = [a, b]`, its magnitude is:

`||v|| = sqrt(a² + b²)`

Example: `v = [3, 4]`, so `||v|| = sqrt(3² + 4²) = 5`.

Using NumPy:

```python
import numpy as np

v = np.array([3, 4])
print(np.linalg.norm(v))  # 5.0
```

For a vector with more components, the Euclidean magnitude is the square root of the sum of the squares of all components.

## 4. Basic vector operations

Given `a = [1, 2, 3]` and `b = [10, 20, 30]`:

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

print(a + b)        # [11 22 33]
print(b - a)        # [ 9 18 27]
print(2 * a)        # [2 4 6]
print(np.dot(a, b)) # 140
```

### Element-wise addition

`a + b = [1+10, 2+20, 3+30] = [11, 22, 33]`

### Scalar multiplication

`2a = [2, 4, 6]`

Multiplying a vector by a scalar multiplies every component by that scalar.

### Dot product

For vectors of equal length, the dot product is the sum of pairwise products:

`a · b = sum(a[i] * b[i])`

In this example: `(1×10) + (2×20) + (3×30) = 140`.

The dot product returns a scalar.

## 5. Why this matters in Machine Learning

A vector can represent the numerical features of one data sample. For example, a house could be represented by a vector containing its area, number of bedrooms, and age. A student's marks across subjects can also be represented as a vector.

Vector operations let us perform calculations on several related values efficiently. Real ML datasets often contain many vectors, which are later organized into matrices or higher-dimensional arrays.

## 6. Quick reference

| Concept | Meaning | Example |
|---|---|---|
| Scalar | One numerical value | `5` |
| Vector | Ordered collection of values | `[1, 2, 3]` |
| `ndim` | Number of array dimensions | `1` for `[1, 2, 3]` |
| `shape` | Size of each axis | `(3,)` |
| `size` | Total number of elements | `3` |
| Magnitude | Length/norm of a vector | `norm([3, 4]) = 5` |
| Dot product | Sum of pairwise products | `[1,2,3] · [10,20,30] = 140` |

## 7. Practice tasks

1. Create a NumPy scalar for a temperature of `28`. Print its `ndim` and `shape`.
2. Create a vector containing three subject marks. Print its `ndim`, `shape`, and `size`.
3. Add `5` to every element of the marks vector.
4. Create vectors `a = [2, 4]` and `b = [3, 5]`. Compute their sum and dot product.
5. Calculate the magnitude of `[3, 4]` using `np.linalg.norm`.

## Key takeaways

- A scalar is a single value; a vector is an ordered collection of values.
- A NumPy scalar-like 0-D array has `ndim == 0` and `shape == ()`.
- A one-dimensional vector with three elements has `ndim == 1` and `shape == (3,)`.
- Vector addition and subtraction are element-wise when shapes are compatible.
- Multiplication by a scalar scales each component.
- The dot product of two vectors of equal length is a scalar.

# Vector Magnitude and Euclidean Distance

## 1. Vector magnitude

Vector magnitude (or Euclidean norm) is the length of a vector.

For a 2D vector `v = [x, y]`:

`||v|| = sqrt(x² + y²)`

Example: `v = [3, 4]`

`||v|| = sqrt(3² + 4²) = sqrt(25) = 5`

For an n-dimensional vector, square every component, sum the squares, and take the square root.

## 2. Magnitude with NumPy

```python
import numpy as np

v = np.array([3, 4])
print(np.linalg.norm(v))  # 5.0
```

For a 3D vector:

```python
v = np.array([2, 3, 6])
print(np.linalg.norm(v))  # 7.0
```

## 3. Euclidean distance

Euclidean distance is the straight-line distance between two points or vectors.

For 2D points `A = (x1, y1)` and `B = (x2, y2)`:

`d(A, B) = sqrt((x2 - x1)² + (y2 - y1)²)`

Example: `A = [1, 2]`, `B = [4, 6]`

`B - A = [3, 4]`

`d(A, B) = sqrt(3² + 4²) = 5`

## 4. Key connection

Euclidean distance is the magnitude of the difference vector:

`d(a, b) = ||b - a||`

This means we can subtract the vectors and calculate the magnitude of the result.

## 5. Euclidean distance with NumPy

```python
import numpy as np

A = np.array([1, 2])
B = np.array([4, 6])

difference = B - A
distance = np.linalg.norm(B - A)

print("Difference:", difference)  # [3 4]
print("Distance:", distance)      # 5.0
```

Both vectors must have compatible shapes for this subtraction. Normally, they have the same number of components.

## 6. Machine learning connection

Distance measures can help algorithms compare data points. For example, clustering methods may use Euclidean distance to assess how close points are in feature space.

A small distance means points are close under the selected representation and metric; it does not guarantee semantic similarity in every problem. Feature scaling can also affect Euclidean distance.

## 7. Quick reference

| Concept | Meaning | Example |
|---|---|---|
| Magnitude | Length of one vector | `norm([3, 4]) = 5` |
| Euclidean distance | Straight-line distance between two points/vectors | Distance from `[1, 2]` to `[4, 6]` is `5` |
| NumPy magnitude/distance | `np.linalg.norm(v)` | Returns a numerical scalar |
| Difference vector | Component-wise subtraction | `[4, 6] - [1, 2] = [3, 4]` |

## 8. Practice tasks

1. Calculate the magnitude of `[6, 8]` by hand and verify it with NumPy.
2. Calculate the magnitude of `[1, 2, 2]`.
3. Find the Euclidean distance between `[2, 1]` and `[5, 5]`.
4. Use `np.linalg.norm(B - A)` to calculate distance between two same-length vectors.
5. Explain why Euclidean distance equals the magnitude of the difference vector.

## Key takeaways

- Magnitude measures the length of a vector.
- Euclidean distance measures the straight-line distance between two points/vectors.
- `np.linalg.norm(v)` calculates the Euclidean norm for a standard numeric vector.
- Euclidean distance can be computed as `np.linalg.norm(B - A)`.
- In ML, distance is meaningful relative to the features, scaling, and metric being used.

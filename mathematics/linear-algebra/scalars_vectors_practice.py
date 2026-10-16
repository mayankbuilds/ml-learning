
import numpy as np

# Task 1: Create a scalar representing a temperature of 28.
temperature = np.array(28)

# Task 2: Create a vector with three subject marks.
marks = np.array([75, 88, 92])

# Task 3: Print the scalar's dimensions and shape.
print("Temperature:", temperature)
print("Temperature ndim:", temperature.ndim)
print("Temperature shape:", temperature.shape)

# Task 4: Print the vector's dimensions, shape, and size.
print("Marks:", marks)
print("Marks ndim:", marks.ndim)
print("Marks shape:", marks.shape)
print("Marks size:", marks.size)

# Task 5: Add 5 bonus marks to each subject.
bonus_marks = marks + 5
print("After bonus:", bonus_marks)

# Task 6: Calculate the magnitude of [3, 4].
v = np.array([3, 4])
magnitude = np.linalg.norm(v)
print("Magnitude:", magnitude)

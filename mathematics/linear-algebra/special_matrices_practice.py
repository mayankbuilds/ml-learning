import numpy as np

Z = np.zeros((3,3,), dtype=int)
I = np.eye(3,dtype=int)
D = np.diag([4,0,7])

print("zero matrix:",Z)
print("identity matrix:",I)
print("diagonal matrix:",D)
print("I@D:",I@D)
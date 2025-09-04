import numpy as np

FY1_pos = np.array([17800, 0, 1800])
v1 = np.array([17800, 0, 0])
vector = -v1 / np.linalg.norm(v1) * 120
print(vector)

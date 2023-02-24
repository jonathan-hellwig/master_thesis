import numpy as np

A = np.random.randint(0, 10, (10,10))
b = np.random.randint(0, 10, (10,1))
x = np.linalg.solve(A, b)
r = b - A @ x
print('Norm des Residuums =', np.linalg.norm(r))

import matplotlib.pyplot as plt
import numpy as np

# Define the problem parameters
L = 10.0  # Length of the beam
n = 100  # Number of discretization points
E = 200e9  # Young's modulus (Pa)
x = np.linspace(0, L, n)
dx = L / (n - 1)

# Define the second moment of area and distributed load functions
I = lambda x: 1e-6 * (1 + 0.5 * x)  # Example: linear variation
w = lambda x: 1000 * np.sin(np.pi * x / L)  # Example: sinusoidal load

# Initialize the matrices and vectors for the finite difference method
A = np.zeros((n, n))
b = np.zeros(n)

# Construct the system of equations
for i in range(1, n - 1):
    xi = x[i]
    I1 = I(xi - dx)
    I2 = I(xi)
    I3 = I(xi + dx)
    A[i, i - 1] = E * I1 / dx**2
    A[i, i] = -2 * E * I2 / dx**2
    A[i, i + 1] = E * I3 / dx**2
    b[i] = w(xi)

# Apply boundary conditions
A[0, 0] = 1
A[1, 0] = -1 / dx
A[1, 1] = 1 / dx
b[0] = 0
b[1] = 0

# Solve the system of equations
v = np.linalg.solve(A, b)

# Compute stress
d2v_dx2 = np.gradient(np.gradient(v, dx), dx)
sigma_x = -E * d2v_dx2

# Plot the results
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(x, v, label="Displacement")
plt.xlabel("x (m)")
plt.ylabel("Displacement (m)")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x, sigma_x, label="Stress (x-direction)", color="r")
plt.xlabel("x (m)")
plt.ylabel("Stress (Pa)")
plt.legend()

plt.tight_layout()
plt.show()

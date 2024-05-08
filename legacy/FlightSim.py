# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 04:00:18 2023

@author: Michel Gordillo
"""

import numpy as np


def aircraft_eom(X, t, input_vector, args):
    # unpack the state vector
    x, y, z, u, v, w, phi, theta, psi = X
    # unpack the input vector
    delta_e, delta_a, delta_r, delta_t = input_vector
    # unpack any additional arguments
    g, m, S, c, b, I_x, I_y, I_z = args
    # calculate the derivative of the state vector
    dx = u * np.cos(theta) * np.cos(psi)
    dy = u * np.cos(theta) * np.sin(psi)
    dz = -g + (u * np.sin(theta))
    du = (delta_t * c * np.cos(theta)) / m - (g * np.sin(theta))
    dv = (
        (delta_t * c * np.sin(theta) * np.cos(phi)) / m
        + (delta_e * S * c) / (m * b)
        - (u * w)
    )
    dw = (
        (delta_t * c * np.sin(theta) * np.sin(phi)) / m
        + (delta_a * S * c) / (m * b)
        - (u * v)
    )
    dphi = (delta_r * S * c) / (I_x * b) + (v * w)
    dtheta = (delta_a * S * c) / (I_y * b) - (u * w)
    dpsi = (delta_r * S * c) / (I_z * b) - (u * v)
    return np.array([dx, dy, dz, du, dv, dw, dphi, dtheta, dpsi])


def runge_kutta(eom, X0, t, u, h, args):
    n = len(t)
    X = X0
    Y = np.zeros([n, len(X0)])
    Y[0] = X0
    for i in range(1, n):
        k1 = h * eom(X, t[i - 1], u, args)
        k2 = h * eom(X + 0.5 * k1, t[i - 1] + 0.5 * h, u, args)
        k3 = h * eom(X + 0.5 * k2, t[i - 1] + 0.5 * h, u, args)
        k4 = h * eom(X + k3, t[i - 1] + h, u, args)
        X = X + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        Y[i] = X
    return Y


# Define the initial conditions and input
X0 = [0, 0, 1000, 100, 0, 0, np.radians(5), np.radians(5), np.radians(5)]
t = np.linspace(0, 10, 100)
u = np.zeros(4)
args = (9.81, 1000, 10, 1, 20, 2, 2, 2)

# Solve the equations of motion
X = runge_kutta(aircraft_eom, X0, t, u, 0.1, args)
x, y, z, u, v, w, phi, theta, psi = X.T

# Plot the results
import matplotlib.pyplot as plt

plt.figure()
plt.subplot(3, 3, 1)
plt.plot(t, x)
plt.title("X Position")
plt.subplot(3, 3, 2)
plt.plot(t, y)
plt.title("Y Position")
plt.subplot(3, 3, 3)
plt.plot(t, z)
plt.title("Z Position")
plt.subplot(3, 3, 4)
plt.plot(t, u)
plt.title("X Velocity")
plt.subplot(3, 3, 5)
plt.plot(t, v)
plt.title("Y Velocity")
plt.subplot(3, 3, 6)
plt.plot(t, w)
plt.title("Z Velocity")
plt.subplot(3, 3, 7)
plt.plot(t, phi)
plt.title("Roll Angle")
plt.subplot(3, 3, 8)
plt.plot(t, theta)
plt.title("Pitch Angle")
plt.subplot(3, 3, 9)
plt.plot(t, psi)
plt.title("Yaw Angle")
plt.tight_layout()
plt.show()

fig = plt.figure()

ax = fig.add_subplot(projection="3d")
# ax.view_init(vertical_axis='y')
ax.set_proj_type(proj_type="ortho")

ax.plot3D(x, y, z)

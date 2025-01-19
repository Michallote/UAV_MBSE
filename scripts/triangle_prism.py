import random

import numpy as np
import plotly.graph_objects as go

# -------------------------
# Step 1: Define a triangular prism in a simple position
# -------------------------
# We'll label the 6 vertices as:
#   0 -> (0, 0, 0)
#   1 -> (1, 0, 0)
#   2 -> (0, 1, 0)
#   3 -> (0, 0, 1)
#   4 -> (1, 0, 1)
#   5 -> (0, 1, 1)

x_orig = np.array([0, 1, 0, 0, 1, 0], dtype=float)
y_orig = np.array([0, 0, 1, 0, 0, 1], dtype=float)
z_orig = np.array([0, 0, 0, 0.5, 0.5, 0.5], dtype=float)

# Triangular faces specified by indices (i, j, k)
i = [0, 3, 0, 0, 1, 1, 2, 2]
j = [1, 4, 1, 4, 2, 5, 0, 3]
k = [2, 5, 4, 3, 5, 4, 3, 5]

# -------------------------
# Step 2: Create and apply a random transformation
#    2a) Generate a random rotation matrix
#    2b) Generate a random translation vector
#    2c) Apply transform to each vertex
# -------------------------


def euler_to_rotation_matrix(alpha, beta, gamma):
    """
    Return the rotation matrix from Euler angles (alpha, beta, gamma),
    using the Z-Y-X convention or any consistent convention you prefer.
    """
    # For simplicity, let's do rotations in X, then Y, then Z (intrinsic rotations).
    # But you can use any other convention. Just be consistent.

    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)],
        ]
    )

    Ry = np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
    )

    Rz = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )

    # Combine the rotations.
    # The final rotation matrix depends on the order in which you apply them.
    R = Rz @ Ry @ Rx
    return R


# 2a) Generate random Euler angles in [0, 2π)
alpha = random.uniform(0, 2 * np.pi)
beta = random.uniform(0, 2 * np.pi)
gamma = random.uniform(0, 2 * np.pi)

# Obtain a 3x3 rotation matrix
R = euler_to_rotation_matrix(alpha, beta, gamma)

# 2b) Generate a random translation vector (e.g., in the range [-1, 1])
t = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])

# 2c) Apply the transformation R * [x, y, z]^T + t to each vertex
xyz_orig = np.vstack([x_orig, y_orig, z_orig])  # shape (3, N)
xyz_transformed = R @ xyz_orig + t.reshape((3, 1))

# Separate back into x, y, z
x_trans = xyz_transformed[0, :]
y_trans = xyz_transformed[1, :]
z_trans = xyz_transformed[2, :]

# -------------------------
# Step 3: Create the plotly figure
# -------------------------
mesh = go.Mesh3d(
    x=x_trans,
    y=y_trans,
    z=z_trans,
    i=i,
    j=j,
    k=k,
    color="lightblue",
    opacity=0.6,
    flatshading=True,
)

fig = go.Figure(data=[mesh])

fig.update_layout(
    title="Randomly Rotated & Translated Triangular Prism",
    scene=dict(
        xaxis=dict(nticks=4, range=[-2, 2]),
        yaxis=dict(nticks=4, range=[-2, 2]),
        zaxis=dict(nticks=4, range=[-2, 2]),
        aspectmode="cube",
    ),
)

fig.show()


coords = np.c_[x_trans, y_trans, z_trans]

oc = np.c_[x_orig, y_orig, z_orig]

f1 = [0, 1, 4, 3, 0]
f2 = [1, 2, 5, 4, 1]
f3 = [2, 0, 3, 5, 2]


def foo(coords):
    coords_lambda = lambda coords: " ".join(
        [str(tuple(map(float, row))) for row in coords]
    )
    cmd = r"\addplot3 coordinates {" + coords_lambda(coords) + "};"
    print(cmd)
    return cmd


foo(coords[f1])
foo(coords[f2])
foo(coords[f3])


shit = """
    \coordinate (A) at {a};
    \coordinate (B) at {b};
    \coordinate (C) at {c};
    \coordinate (D) at {d};
    \coordinate (E) at {e};
    \coordinate (F) at {f};
"""


# 2a) Generate random Euler angles in [0, 2π)
alpha = random.uniform(0, 2 * np.pi)
beta = random.uniform(0, 2 * np.pi)
gamma = random.uniform(0, 2 * np.pi)

# Obtain a 3x3 rotation matrix
R = euler_to_rotation_matrix(alpha, beta, gamma)

# 2b) Generate a random translation vector (e.g., in the range [-1, 1])
t = np.array([random.uniform(0, 1), random.uniform(0, 1), 0.1])

# 2c) Apply the transformation R * [x, y, z]^T + t to each vertex
xyz_orig = np.vstack([x_orig, y_orig, z_orig])  # shape (3, N)
xyz_transformed = R @ xyz_orig + t.reshape((3, 1))


a = dict(
    zip(
        ["a", "b", "c", "d", "e", "f"],
        list(map(lambda x: (tuple(map(float, x))), xyz_transformed.T)),
    )
)
b = shit.format(**a).replace("[", "(").replace("]", ")")
print(b)

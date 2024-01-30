import numpy as np
import plotly.graph_objects as go

from src.aerodynamics.airfoil import Airfoil

airfoil = Airfoil.from_file("data/airfoils/FX 73-CL2-152.dat")
airfoil.set_z(0)
airfoil = airfoil.resample(49)
data = airfoil.data


def all_different(lst):
    """Check if all elements in the list are different."""
    return len(set(lst)) == len(lst)


n = len(data) - 1

indices = np.array(
    [result for i in range(n) if all_different(result := [i, i + 1, n - i])]
)

i, j, k = indices.T


triangles = data[indices]


data
# extract the lists of x, y, z coordinates of the triangle vertices and connect them by a line
Xe = []
Ye = []
Ze = []
for T in tri_points:
    Xe.extend([T[k % 3][0] for k in range(4)] + [None])
    Ye.extend([T[k % 3][1] for k in range(4)] + [None])
    Ze.extend([T[k % 3][2] for k in range(4)] + [None])


fig = go.Figure(
    data=[
        go.Mesh3d(
            x=airfoil.x,
            y=airfoil.y,
            z=np.zeros_like(airfoil.x),
            i=i,
            j=j,
            k=k,
            color="blue",
            opacity=0.5,
        )
    ]
)

# define the trace for triangle sides
lines = go.Scatter3d(
    x=Xe, y=Ye, z=Ze, mode="lines", name="", line=dict(color="rgb(70,70,70)", width=1)
)

fig.add_trace(lines)

fig.update_layout(
    scene=dict(
        xaxis=dict(showbackground=False),
        yaxis=dict(showbackground=False),
        zaxis=dict(showbackground=False),
    ),
    title="3D Mesh Plot of Triangles",
)

fig.show()


def plot_triangles_mesh(triangles):
    """
    Plots a list of triangles as a mesh in 3D space.

    Parameters:
    triangles (list of lists): A list where each element is a list of three vertices,
                               and each vertex is represented by a tuple (x, y, z).
    """
    # Extract all vertices and flatten the list
    vertices = [vertex for triangle in triangles for vertex in triangle]
    x, y, z = zip(*vertices)

    # Create a list of indices for the triangles
    indices = list(range(len(vertices)))

    # Define the triangles through their vertex indices
    # Each triangle is defined by three consecutive integers
    i, j, k = indices[0::3], indices[1::3], indices[2::3]

    fig = go.Figure(
        data=[go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color="blue", opacity=0.5)]
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
        title="3D Mesh Plot of Triangles",
    )

    fig.show()


# Example usage
triangles = [
    [(0, 0, 0), (1, 0, 1), (0, 1, 0)],
    [(1, 0, 0), (1, 1, 0), (0, 1, 0)],
    [(1, 0, 1), (2, 0, 1), (1, 1, 1)],
]

plot_triangles_mesh(triangles)


# Define two 1D arrays
x = np.array([0.0, 0.2, 0.5, 0.6, 1.0])
y = np.array([0.0, 0.1, 0.5, 0.8, 1.0])

# Create 2D grids for X and Y coordinates
xx, yy = np.meshgrid(x, y)
zz = np.zeros_like(xx)
zz[-1][-1] = 1


vec_v = np.array(
    [np.diff(xx, axis=0), np.diff(yy, axis=0), np.diff(zz, axis=0)]
).transpose(1, 2, 0)
vec_u = np.array(
    [np.diff(xx, axis=1), np.diff(yy, axis=1), np.diff(zz, axis=1)]
).transpose(1, 2, 0)

# Calculate the cross product, which gives a vector perpendicular to the triangle's surface
# The magnitude of this vector is twice the area of the triangle
cross_product = np.cross(vec_u[:-1, :], vec_v[:, :-1], axis=2)
area = np.linalg.norm(cross_product, axis=2)

vec_vr = np.array(
    [
        np.diff(np.flip(xx), axis=0),
        np.diff(np.flip(yy), axis=0),
        np.diff(np.flip(zz), axis=0),
    ]
).transpose(1, 2, 0)
vec_ur = np.array(
    [
        np.diff(np.flip(xx), axis=1),
        np.diff(np.flip(yy), axis=1),
        np.diff(np.flip(zz), axis=1),
    ]
).transpose(1, 2, 0)


cross_product_r = np.cross(vec_ur[:-1, :], vec_vr[:, :-1], axis=2)
area_r = np.linalg.norm(cross_product_r, axis=2)

area + np.flip(area_r)

# Sum the areas of all triangles
total_area = (
    np.sum(area) / 2 + np.sum(area_r) / 2
)  # Divided by 2 because each area is counted twice


coordinates = np.array([xx, yy, zz])

triangles_coordinates = np.array(
    [
        coordinates[:, :-1, :-1], #bottom left vertices -> 0
        coordinates[:, :-1, 1:], #bottom right vertices -> 1
        coordinates[:, 1:, :-1], #upper left vertices -> 2
    ]
)

triangles_coordinates[:, :, 0, 0]

triangles_coordinates[]

centroids = np.mean(triangles_coordinates, axis = 0)

centroid = np.sum(np.sum(area*centroids, axis = 1), axis = 1)/ np.sum(area)
#centroid of the first triangle = centroids[:,0,0]

area * centroids / np.sum(area)

np.sum(area*centroids, axis = 1)
np.sum((area*centroids)[0]       , axis = 0)
# Reshape to (num_triangles, 3 points, 3 coordinates)
num_triangles = triangle_x.shape[1] * triangle_x.shape[2]
reshaped_triangles = np.zeros((num_triangles, 3, 3))

# Populate the reshaped array with the triangle coordinates
for i in range(3):  # For each vertex in the triangle
    reshaped_triangles[:, i, 0] = triangle_x[i].flatten()  # x-coordinates
    reshaped_triangles[:, i, 1] = triangle_y[i].flatten()  # y-coordinates
    reshaped_triangles[:, i, 2] = triangle_z[i].flatten()  # z-coordinates

zz_r = np.flip(zz)

zz_r[0][-1] = 2
zz_r[-1][0] = 3

triangle_z = np.array([zz_r[:-1, :-1], zz_r[1:, :-1], zz_r[:-1, 1:]])

import plotly.graph_objects as go

fig = go.Figure(data=[go.Surface(z=zz, x=xx, y=yy)])

# Add vertices (as points)
fig.add_trace(
    go.Scatter3d(
        x=xx.flatten(),
        y=yy.flatten(),
        z=zz.flatten(),
        mode="markers",
        marker=dict(size=6, color="black"),
    )
)


fig.update_layout(
    title="Mt Bruno Elevation",
    # autosize=False,
    # width=500,
    # height=500,
    margin=dict(l=65, r=50, b=65, t=90),
)
fig.show()

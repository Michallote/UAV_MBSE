import numpy as np
import plotly
import plotly.graph_objects as go

from src.geometry.spatial_array import SpatialArray


def _get_surface_intersections(xx, yy, zz, x_value: float) -> np.ndarray:
    """Calculate the intersection points of a surface with a given vertical plane at x_value."""

    # We are assuming a linear interpolation between grid points.
    # First, identify rows where intersections occur
    mask = (xx[:, :-1] <= x_value) & (xx[:, 1:] >= x_value)

    mask = xx < x_value

    # Calculate the y-coordinates of the intersection points
    # Using linear interpolation: y = y1 + (y2 - y1) * ((x_value - x1) / (x2 - x1))
    y_intersections = yy[:, :-1][mask] + (yy[:, 1:] - yy[:, :-1])[mask] * (
        (x_value - xx[:, :-1][mask]) / (xx[:, 1:] - xx[:, :-1][mask])
    )

    # Create intersection points array
    intersection_points = np.column_stack(
        [
            np.full(y_intersections.shape, x_value),
            y_intersections,
            np.zeros_like(y_intersections),
        ]
    )

    return intersection_points


# Define two 1D arrays
x = np.array([0.0, 0.2, 0.5, 0.6, 1.0])
y = np.array([0.0, 0.1, 0.5, 0.8, 1.0])

# Create 2D grids for X and Y coordinates
xx, yy = np.meshgrid(x, y)
zz = np.zeros_like(xx)

# Intersection plane
x_value = 0.43

# Calculate intersection points
intersection_points = _get_surface_intersections(xx, yy, zz, x_value)
print(intersection_points)


# Function to normalize data
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


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

    # Calculate the metric for each triangle, e.g., average z value
    triangle_metric = [(z[ii] + z[jj] + z[kk]) / 3 for ii, jj, kk in zip(i, j, k)]

    # Normalize the metric
    normalized_metric = normalize(np.array(triangle_metric))

    # Choose a colorscale
    colorscale = "Viridis"

    # Map normalized metric to colors
    colors = [
        plotly.colors.convert_colors_to_same_type(
            plotly.colors.sample_colorscale(colorscale, val)
        )[0][0]
        for val in normalized_metric
    ]

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k, facecolor=colors, colorscale=colorscale
            )
        ]
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
triangles = np.array(
    [
        [(0.5, -0.2, 0.5), (1.5, -0.20, 1.5), (0.5, 0.8, 0.5)],
        [(0, 0, 1), (2, 0, 1), (1, 1, 1)],
        [(0, 0, 0), (2, 0, 0), (1, 1, 0)],
    ]
)

plot_triangles_mesh(triangles)


def check_bounding_box_overlap(t1, t2) -> bool:
    """Calculates bounding box ovrlap

    Parameters
    ----------
    t1 : np.ndarray
        Triangle 1
    t2 : np.ndarray
        Triangle 2

    Returns
    -------
    bool
        True when BB overlap, False when there is no overlap
    """
    t1 = SpatialArray(t1)
    t2 = SpatialArray(t2)

    return not (
        np.min(t2.x) > np.max(t1.x)
        or np.min(t2.y) > np.max(t1.y)
        or np.min(t2.z) > np.max(t1.z)
        or np.min(t1.x) > np.max(t2.x)
        or np.min(t1.y) > np.max(t2.y)
        or np.min(t1.z) > np.max(t2.z)
    )


t1, t2, t3 = triangles
plot_triangles_mesh(np.array([t1, t2]))
check_bounding_box_overlap(t1, t2)

check_bounding_box_overlap(t2, t3 + 5)

plot_triangles_mesh(np.array([t1, t2]))

import numpy as np


def dot2(v):
    """Compute dot product of a vector with itself."""
    return np.dot(v, v)


def udTriangle(v1, v2, v3, p):
    """
    Compute the unsigned distance from a point to a triangle.

    :param v1: First vertex of the triangle
    :param v2: Second vertex of the triangle
    :param v3: Third vertex of the triangle
    :param p: The point for distance computation
    :return: Unsigned distance from the point to the triangle
    """
    v21 = v2 - v1
    p1 = p - v1
    v32 = v3 - v2
    p2 = p - v2
    v13 = v1 - v3
    p3 = p - v3
    nor = np.cross(v21, v13)

    def sign(x):
        """Return the sign of x."""
        return np.copysign(1.0, x)

    term1 = sign(np.dot(np.cross(v21, nor), p1))
    term2 = sign(np.dot(np.cross(v32, nor), p2))
    term3 = sign(np.dot(np.cross(v13, nor), p3))

    if term1 + term2 + term3 < 2.0:
        d1 = dot2(v21 * np.clip(np.dot(v21, p1) / dot2(v21), 0.0, 1.0) - p1)
        d2 = dot2(v32 * np.clip(np.dot(v32, p2) / dot2(v32), 0.0, 1.0) - p2)
        d3 = dot2(v13 * np.clip(np.dot(v13, p3) / dot2(v13), 0.0, 1.0) - p3)
        return np.sqrt(min(min(d1, d2), d3))
    else:
        return np.sqrt(np.dot(nor, p1) * np.dot(nor, p1) / dot2(nor))


# Example usage
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])
p = np.array([0.5, 0.5, 0.5])


# Example usage
triangles = np.array([[v1, v2, v3], [v1 + 1, v2 + 1, v3 + 1]])

plot_triangles_mesh(triangles)

distance = udTriangle(v1, v2, v3, p)

udTriangle(v1, v2, v3, np.array([1, 1, 1]))
udTriangle(v1, v2, v3, np.array([0, 0, 0]))
print(distance)


x = np.linspace(-0.5, 2, 50)
y = np.linspace(-0.5, 2, 50)

xx, yy = np.meshgrid(x, y)
zz = np.zeros_like(xx)

for i in range(len(xx)):
    for j in range(len(yy)):
        p = np.array([xx[i, j], yy[i, j], 0.5])
        zz[i, j] = udTriangle(v1, v2, v3, p)

t1 = SpatialArray([v1, v2, v3])


# Create the figure
fig = go.Figure(
    data=[
        go.Surface(z=np.zeros_like(xx) + 0.5, x=xx, y=yy, surfacecolor=zz),
        go.Mesh3d(x=t1.x, y=t1.y, z=t1.z),
    ]
)

# Update layout for a better view
fig.update_layout(
    title="3D Surface Plot",
    # autosize=False,
    # width=500,
    # height=500,
    margin=dict(l=65, r=50, b=65, t=90),
    scene=dict(
        xaxis=dict(showbackground=False),
        yaxis=dict(showbackground=False),
        zaxis=dict(showbackground=False),
    ),
)

# Show the figure
fig.show()

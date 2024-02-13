import numpy as np
import plotly.graph_objects as go

from src.geometry.spatial_array import SpatialArray


def evaluate_surface_intersection(
    xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, p: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """Evaluates surfaces intersections between a meshgrid like surface
    and a plane defined by a point and a plane.

    Parameters
    ----------
    xx : np.ndarray
        X matrix array.
    yy : np.ndarray
        Y matrix array.
    zz : np.ndarray
        Z matrix array
    p : np.ndarray
        plane point position
    n : np.ndarray
        normal plane

    Returns
    -------
    np.ndarray
        intersection curves
    """
    n = n / np.linalg.norm(n)

    # First we calculate the Q vectors.
    # Stack the xx, yy, zz grids to form a 3D array where each element is a 3D coordinate
    coordinates = np.stack((xx, yy, zz), axis=-1)

    # Subtract p from each 3D coordinate
    q = coordinates - p

    dot_products = np.einsum("ijk,k->ij", q, n)

    sdf_sign = dot_products > 0

    intersections = []

    for i in range(sdf_sign.shape[0]):
        for j in range(sdf_sign.shape[1]):
            try:
                u_idx = [[i, j], [i, j + 1]]

                if sdf_sign[*u_idx[0]] != sdf_sign[*u_idx[1]]:  # Horizontal change
                    intersections.append(u_idx)
            except:
                pass
            try:
                v_idx = [[i, j], [i + 1, j]]

                if sdf_sign[*v_idx[0]] != sdf_sign[*v_idx[1]]:  # Vertical change
                    intersections.append(v_idx)
            except:
                pass

    intersection_points = []

    for indices in intersections:
        id_a, id_b = indices
        p1, p2 = coordinates[*id_a], coordinates[*id_b]

        intersection_point = line_plane_intersection(p1, p2, p, n)
        intersection_points.append(intersection_point)

    return np.array(intersection_points)


def line_plane_intersection(
    p1: np.ndarray, p2: np.ndarray, p: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """
    Calculate the intersection point of a line segment with an infinite plane.

    Parameters:
    p1 (np.ndarray): The starting point of the line segment.
    p2 (np.ndarray): The ending point of the line segment.
    p (np.ndarray): A point on the plane.
    n (np.ndarray): The normal vector of the plane.

    Returns:
    np.ndarray: The intersection point if it exists, otherwise None.
    """
    line_dir = p2 - p1  # Direction vector of the line
    n_dot_line_dir = np.dot(n, line_dir)  # Perpendicular distance to p2' through n

    # Check if the line and plane are parallel
    if np.isclose(n_dot_line_dir, 0):
        return None

    # Calculate the parameter t for the line equation
    t = np.dot(n, p - p1) / n_dot_line_dir

    # Check if the intersection point lies within the line segment
    if 0 <= t <= 1:
        intersection_point = p1 + t * line_dir
        return intersection_point
    else:
        return None


def project_points_to_plane(points, plane_point, plane_normal) -> np.ndarray:
    """Return a curve in 3D space proyected to a plane using a set of local coordinates"""
    # Normalize the normal vector
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Create local coordinate system on the plane
    # Find a vector not parallel to plane_normal
    if not np.isclose(plane_normal[0], 0):
        vec = np.array([-plane_normal[1], plane_normal[0], 0])
    else:
        vec = np.array([0, plane_normal[2], -plane_normal[1]])
    # First local axis (U)
    U = np.cross(plane_normal, vec)
    U = U / np.linalg.norm(U)
    # Second local axis (V)
    V = np.cross(U, plane_normal)
    V = V / np.linalg.norm(V)

    projected_points = []
    for point in points:
        # Projection of the point onto the plane
        diff = point - plane_point
        projected_point = point - np.dot(diff, plane_normal) * plane_normal

        # Convert to local coordinates
        x_local = np.dot(projected_point - plane_point, U)
        y_local = np.dot(projected_point - plane_point, V)

        projected_points.append((x_local, y_local))

    return np.array(projected_points)


xx, yy = np.meshgrid(x, y)
zz = np.sin(xx**2) + np.cos(yy**2)

p = np.array([0.43, 0, 0])
n = np.array([1, 0, 0])

xx, yy, zz = surface.xx, surface.yy, surface.zz

curve_1 = SpatialArray(
    evaluate_surface_intersection(xx[:, :75], yy[:, :75], zz[:, :75], p, n)
)

curve_2 = SpatialArray(
    evaluate_surface_intersection(xx[:, 75:], yy[:, 75:], zz[:, 75:], p, n)
)

curve = SpatialArray(np.vstack([curve_1, np.flip(curve_2, axis=0)]))

fig = go.Figure(
    data=[
        go.Surface(z=zz[:, :75], x=xx[:, :75], y=yy[:, :75]),
    ]
)


fig.add_trace(
    go.Scatter3d(
        x=curve.x,
        y=curve.y,
        z=curve.z,
        mode="lines+markers",  # Combine lines and markers
        line=dict(color="black", width=10),  # Thick black line
        marker=dict(size=5, color="red"),  # Red markers
    )
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
        aspectmode="data",
    ),
)

# Show the figure
fig.show()


def create_checkerboard_colored_surface(
    xx: np.ndarray, yy: np.ndarray, zz: np.ndarray
) -> go.Figure:
    """
    Create a 3D surface plot with flat shading and a checkerboard pattern of blue and red colors.

    Parameters:
    xx (np.ndarray): A 2D array for x-coordinates.
    yy (np.ndarray): A 2D array for y-coordinates.
    zz (np.ndarray): A 2D array for z-coordinates (surface heights).

    Returns:
    go.Figure: Plotly figure with the specified 3D surface plot.
    """
    # Create an alternating matrix of 0s and 1s
    checkerboard_pattern = np.indices(zz.shape).sum(axis=0) % 2

    # Define a color scale with 0 as blue and 1 as red
    colorscale = [[0, "blue"], [1, "red"]]

    # Assign colors based on the checkerboard pattern
    surface_color = checkerboard_pattern

    # Create the surface plot with flat shading
    surface = go.Surface(
        z=zz,
        x=xx,
        y=yy,
        surfacecolor=surface_color,
        colorscale=colorscale,
        flatshading=True,
        # lighting=dict(
        #     diffuse=0.8, specular=0.3, ambient=0.1, roughness=0.1, fresnel=0.1
        # ),
    )

    # Create and return the figure
    fig = go.Figure(data=[surface])

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
            aspectmode="data",
        ),
    )

    return fig


create_checkerboard_colored_surface(xx, yy, zz).show()


import matplotlib.cm as cm
import numpy as np
import plotly.graph_objs as go
from scipy.spatial import Delaunay

u = np.linspace(0, 2 * np.pi, 24)
v = np.linspace(-1, 1, 8)
u, v = np.meshgrid(u, v)
u = u.flatten()
v = v.flatten()

# evaluate the parameterization at the flattened u and v
tp = 1 + 0.5 * v * np.cos(u / 2.0)
x = tp * np.cos(u)
y = tp * np.sin(u)
z = 0.5 * v * np.sin(u / 2.0)

# define 2D points, as input data for the Delaunay triangulation of U
points2D = np.vstack([u, v]).T
tri = Delaunay(points2D)  # triangulate the rectangle U

import numpy as np
import plotly.figure_factory as ff
from scipy.spatial import Delaunay

u = np.linspace(0, 2 * np.pi, 24)
v = np.linspace(-1, 1, 8)
u, v = np.meshgrid(u, v)
u = u.flatten()
v = v.flatten()

tp = 1 + 0.5 * v * np.cos(u / 2.0)
x = tp * np.cos(u)
y = tp * np.sin(u)
z = 0.5 * v * np.sin(u / 2.0)


curve = SpatialArray(evaluate_surface_intersection(xx, yy, zz, p, n))

points2D = project_points_to_plane(curve, p, n)
tri = Delaunay(points2D)
simplices = tri.simplices

fig = ff.create_trisurf(
    x=curve.x,
    y=curve.y,
    z=curve.z,
    colormap="Portland",
    simplices=simplices,
    title="Mobius Band",
)
fig.show()

from scipy.spatial import ConvexHull

hull = ConvexHull(points2D)
points2D[hull.vertices]

from typing import Any

import numpy as np
import triangle
from matplotlib.path import Path
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient


def create_boundary_dict(
    polygon_points: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Create a boundary dictionary for a polygon compatible with the `triangle` library.

    This function converts a set of polygon points into a dictionary format containing
    vertices and segments that define the polygon's boundary. The polygon is ensured
    to have counterclockwise orientation.

    Parameters:
    ----------
    polygon_points : np.ndarray
        An array of shape (n, 2) representing the vertices of the polygon.

    Returns:
    -------
    dict[str, np.ndarray]
        A dictionary containing:
        - 'vertices': An array of the polygon's vertices.
        - 'segments': An array of line segments connecting the vertices, forming the boundary.
    """
    polygon = Polygon(polygon_points)

    # Ensure the polygon vertices are oriented counterclockwise
    polygon = orient(polygon, sign=1.0)

    # Convert to a format compatible with `triangle`
    polygon_dict = {
        "vertices": np.array(polygon.exterior.coords[:-1]),
        "segments": np.array(
            [[i, (i + 1) % len(polygon_points)] for i in range(len(polygon_points))]
        ),
    }

    return polygon_dict


def create_mesh_from_boundary(
    boundary_coordinates: np.ndarray, max_area: float
) -> tuple[dict, dict]:
    """
    Generate a triangulated mesh from polygon boundary coordinates.

    This function creates a constrained Delaunay triangulated mesh based on the
    input boundary coordinates and a specified maximum triangle area constraint.

    Parameters:
    ----------
    boundary_coordinates : np.ndarray
        An array of shape (n, 2) representing the vertices of the polygon boundary.

    max_area : float
        The maximum allowable area for triangles in the mesh. Smaller values create
        finer meshes, while larger values create coarser meshes.

    Returns:
    -------
    tuple[dict, dict]
        - The first element is the triangulated mesh as a dictionary returned by `triangle.triangulate`.
        - The second element is the boundary dictionary used for the triangulation.
    """
    boundary_dict = create_boundary_dict(boundary_coordinates)

    # Add a maximum triangle area constraint for refinement
    mesh_dict = triangle.triangulate(boundary_dict, f"pqa{max_area}")
    return mesh_dict, boundary_dict


def random_points_inside_curve(curve: np.ndarray, num_points: int) -> np.ndarray:
    """
    Generate points inside a closed curve.

    Parameters
    ----------
    curve : np.ndarray
        The curve to generate points inside.
    num_points : int
        The number of points to generate.

    Returns
    -------
    np.ndarray
        The points inside the curve.
    """
    # Compute the bounding box of the curve
    min_x, min_y = np.min(curve, axis=0)
    max_x, max_y = np.max(curve, axis=0)

    boundary = Path(curve)

    inner_points = np.mean(curve, axis=0)

    while len(inner_points) < num_points:
        new_points = np.random.uniform(
            low=[min_x, min_y], high=[max_x, max_y], size=(num_points, 2)
        )
        inside = boundary.contains_points(new_points)
        inner_points = np.vstack((inner_points, new_points[inside]))

    return inner_points[:num_points]

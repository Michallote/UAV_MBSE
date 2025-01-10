from typing import Optional

import numpy as np
import triangle
from matplotlib.path import Path
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from geometry.projections import construct_orthonormal_basis, project_points_to_plane
from src.geometry.transformations import compute_curve_normal


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
    boundary_coordinates: np.ndarray, max_area: Optional[float] = None
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

    if max_area:
        constrain = f"pqa{max_area}"
    else:
        constrain = "pq"

    boundary_dict = create_boundary_dict(boundary_coordinates)
    # Add a maximum triangle area constraint for refinement
    mesh_dict = triangle.triangulate(boundary_dict, constrain)

    assert (
        "triangles" in mesh_dict
    ), f"Discretization faled on points: \n{boundary_coordinates}"
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


def compute_3d_planar_mesh(
    boundary: np.ndarray,
    plane_point: Optional[np.ndarray] = None,
    plane_normal: Optional[np.ndarray] = None,
    max_area: Optional[float] = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Compute a 3D mesh on a given plane defined by a point and normal.

    Parameters
    ----------
    boundary : np.ndarray
        The boundary points defining the region to mesh (N x 3).
    plane_point : np.ndarray
        A point on the plane (3D vector).
    plane_normal : np.ndarray
        The normal vector of the plane (3D vector).

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
        mesh_dict : dict
            Dictionary containing the mesh data, including transformed vertices.
        boundary_dict : dict
            Dictionary containing boundary information.
    """
    if plane_normal is None:
        plane_normal = compute_curve_normal(boundary)

    if plane_point is None:
        plane_point = np.array(boundary[0])

    # Project boundary points onto the plane
    projected_points = project_points_to_plane(
        boundary, plane_point=plane_point, plane_normal=plane_normal
    )

    # Construct an orthonormal basis for the plane
    u, v, _w = construct_orthonormal_basis(plane_normal)

    # Create the mesh using the projected boundary points
    mesh_dict, boundary_dict = create_mesh_from_boundary(
        boundary_coordinates=projected_points, max_area=max_area
    )

    # Transform vertices to 3D space of the plane
    vertices = mesh_dict["vertices"]
    local_basis = np.array([u, v])
    transformed_vertices = np.einsum("ij,jk->ik", vertices, local_basis) + plane_point
    mesh_dict["vertices"] = transformed_vertices
    boundary_dict["vertices"] = (
        np.einsum("ij,jk->ik", boundary_dict["vertices"], local_basis) + plane_point
    )

    return mesh_dict, boundary_dict

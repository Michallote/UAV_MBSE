from typing import Optional

import numpy as np

from geometry.planar import curve_area, curve_centroid
from geometry.transformations import compute_curve_normal


def construct_orthonormal_basis(plane_normal: np.ndarray) -> np.ndarray:
    """Creates a local coordinate system from a normal vector
    plane_normal -> [0,0,1] produces u -> [1,0,0] v -> [0,1,0]

    plane_normal -> [1,0,0] produces u -> [0,0,-1] v -> [0,1,0]



    Parameters
    ----------
    plane_normal : np.ndarray
        Vector normal to a plane where the u, v unit vectors reside

    Returns
    -------
    np.ndarray
        A 3x3 matrix where each row represents one of the
        unit vectors (U, V, W) of the local coordinate system.
    """

    # Normalize the normal vector
    w = plane_normal / np.linalg.norm(plane_normal)

    # Create local coordinate system on the plane
    # Find a vector not parallel to plane_normal
    if not np.isclose(w[0], 0):
        initial_vector = np.array([-w[1], w[0], 0])
    else:
        initial_vector = np.array([0, w[2], -w[1]])
    # Second local axis (V)
    v = initial_vector
    v = v / np.linalg.norm(v)
    # First local axis (U)
    u = np.cross(v, w)
    u = u / np.linalg.norm(u)

    return np.array([u, v, w])


def project_points_to_plane(
    points: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray
) -> np.ndarray:
    """Return a curve in 3D space proyected to a plane using a set of local coordinates"""
    # Normalize the normal vector
    w = plane_normal / np.linalg.norm(plane_normal)

    u, v, w = construct_orthonormal_basis(w)

    projected_points = []
    for point in points:
        # Projection of the point onto the plane
        diff = point - plane_point
        projected_point = point - np.dot(diff, w) * w

        # Convert to local coordinates
        x_local = np.dot(projected_point - plane_point, u)
        y_local = np.dot(projected_point - plane_point, v)

        projected_points.append((x_local, y_local))

    return np.array(projected_points)


def create_projection_and_basis(
    boundary: np.ndarray,
    plane_point: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a projection of a 3D boundary onto a plane and constructs an orthonormal basis.

    Parameters
    ----------
    boundary : np.ndarray
        A 3D closed curve represented as an array of points (N x 3).
    plane_point : np.ndarray, optional
        A point on the plane. If None, the first point of the boundary is used.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        projected_points : np.ndarray
            The boundary points projected onto the plane (N x 3).
        local_basis : np.ndarray
            Orthonormal basis of the plane as a 3x3 matrix.
        plane_point : np.ndarray
            The point on the plane used for projection.

    Raises
    ------
    ValueError
        If the boundary has fewer than 3 points or is not 3D.
    """
    if boundary.shape[0] < 3 or boundary.shape[1] != 3:
        raise ValueError("Boundary must have at least 3 points and be a 3D curve.")

    # Use the first boundary point as the plane point if not specified
    if plane_point is None:
        plane_point = np.array(boundary[0])

    # Compute the plane normal
    plane_normal = compute_curve_normal(boundary)

    # Construct an orthonormal basis for the plane
    local_basis = construct_orthonormal_basis(plane_normal)

    # Project boundary points onto the plane
    projected_points = project_points_to_plane(
        boundary, plane_point=plane_point, plane_normal=plane_normal
    )

    return projected_points, local_basis, plane_point


def transform_vertices_to_plane(
    vertices: np.ndarray, local_basis: np.ndarray, plane_point: np.ndarray
) -> np.ndarray:
    """
    Transforms a set of vertices to the coordinate system defined by a local basis
    and a plane point.

    Parameters
    ----------
    vertices : np.ndarray
        The vertices to transform, represented as an array of shape (N, 2).
    local_basis : np.ndarray
        A 2x3 array representing the local basis vectors [u, v].
    plane_point : np.ndarray
        A 3D point defining the origin of the plane in global coordinates.

    Returns
    -------
    np.ndarray
        Transformed vertices in the 3D coordinate system, shape (N, 3).
    """
    # Handle single point case
    if vertices.ndim == 1:
        vertices = np.vstack([vertices])
    # Handle full (u, v, w) 3D basis
    if local_basis.shape != (2, 3):
        local_basis = local_basis[:2]

    assert vertices.ndim == 2 and vertices.shape[-1] == 2, ValueError(
        "Vertices must have shape (N, 2)."
    )
    assert local_basis.shape == (2, 3), ValueError("Local Basis in incorrect format.")
    assert plane_point.shape == (3,), ValueError("Plane point must be a 3D vector.")

    # Perform the transformation
    transformed_vertices = np.einsum("ij,jk->ik", vertices, local_basis) + plane_point
    return transformed_vertices


def compute_space_curve_centroid(curve: np.ndarray) -> np.ndarray:
    """
    Computes the centroid of a 3D curve by projecting it onto a plane,
    calculating the centroid in the 2D plane, and transforming it back to 3D.

    Parameters
    ----------
    curve : np.ndarray
        A 3D curve represented as an array of points with shape (N, 3).
        The curve should have at least 3 points.

    Returns
    -------
    np.ndarray
        The 3D coordinates of the centroid of the curve.

    Raises
    ------
    ValueError
        If the curve has fewer than 3 points, is not a 3D curve,
          or if it is not a valid 3D vector.
    """
    # Input validation
    if curve.ndim != 2 or curve.shape[1] != 3:
        raise ValueError("Curve must be a 2D array with shape (N, 3), where N >= 3.")
    if curve.shape[0] < 3:
        raise ValueError("Curve must have at least 3 points to compute a centroid.")

    # Create the projection, basis, and plane origin
    projected_curve, basis, projection_plane_point = create_projection_and_basis(curve)

    # Compute the centroid of the projected curve in 2D
    centroid_projection = curve_centroid(projected_curve)

    # Transform the centroid back into 3D space
    centroid_3d = transform_vertices_to_plane(
        centroid_projection, basis, projection_plane_point
    )

    return np.squeeze(centroid_3d)


def compute_space_curve_area(curve: np.ndarray) -> float:
    """
    Computes the area enclosed by a 3D curve by projecting it onto a plane.

    Parameters
    ----------
    curve : np.ndarray
        A 3D curve represented as an array of points with shape (N, 3).
        The curve should have at least 3 points.

    Returns
    -------
    float
        The area enclosed by the projected curve in the plane.

    Raises
    ------
    ValueError
        If the curve has fewer than 3 points or is not a 3D curve.
    """
    # Input validation
    if curve.ndim != 2 or curve.shape[1] != 3:
        raise ValueError("Curve must be a 2D array with shape (N, 3), where N >= 3.")
    if curve.shape[0] < 3:
        raise ValueError("Curve must have at least 3 points to compute the area.")

    plane_point = np.array(curve[0])
    plane_normal = compute_curve_normal(curve)
    projected_coordinates = project_points_to_plane(curve, plane_point, plane_normal)
    return curve_area(projected_coordinates)

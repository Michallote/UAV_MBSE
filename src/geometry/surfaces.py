"""Surface intersection algorithms"""

from typing import Generator

import numpy as np

from src.geometry.spatial_array import SpatialArray

# Define a type alias for better readability
IndexPair = tuple[int, int]
LineSegmentIndices = list[tuple[IndexPair, IndexPair]]


def triangle_area(v1, v2, v3) -> float:
    """Calculate the area of a triangle in 3D.
    v1,v2,v3 are vertices of the triangle.

    Returns
    -------
    float
        Area of triangle
    """
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))  # type: ignore


def surface_centroid_area(xx, yy, zz) -> tuple[np.ndarray, float]:
    """Calculates the centroid of a surface using triangulation methods

    Parameters
    ----------
    xx : np.ndarray
        x-matrix of surface coordinates
    yy : np.ndarray
        y-matrix of surface coordinates
    zz : np.ndarray
        z-matrix of surface coordinates
    """
    # Calculate the area of bottom left half surface
    centroid_l, area_l = _calculate_centroid_of_half_surface(xx, yy, zz)
    # Calculate the area of top right half surface
    centroid_r, area_r = _calculate_centroid_of_half_surface(
        np.flip(xx), np.flip(yy), np.flip(zz)
    )
    area = area_l + area_r

    centroid = (centroid_l * area_l + centroid_r * area_r) / area

    return centroid, area


def _calculate_centroid_of_half_surface(xx, yy, zz) -> tuple[np.ndarray, float]:
    """Calculates the centroid of a surface using triangulation methods

    Parameters
    ----------
    xx : np.ndarray
        x-matrix of surface coordinates
    yy : np.ndarray
        y-matrix of surface coordinates
    zz : np.ndarray
        z-matrix of surface coordinates
    """

    coordinates = np.array([xx, yy, zz])

    triangles_coordinates = np.array(
        [
            coordinates[:, :-1, :-1],  # bottom left vertices -> 0
            coordinates[:, :-1, 1:],  # bottom right vertices -> 1
            coordinates[:, 1:, :-1],  # upper left vertices -> 2
        ]
    )
    # to acces the triangle associated with the first vertex
    # use: triangle_coordinates[:, 0, 0]

    area_matrix = _calculate_area_of_half_surface(xx, yy, zz)

    centroids = np.mean(triangles_coordinates, axis=0)

    centroid = np.sum(np.sum(area_matrix * centroids, axis=1), axis=1) / np.sum(
        area_matrix
    )

    return centroid, np.sum(area_matrix)


def _calculate_area_of_half_surface(
    xx: np.ndarray, yy: np.ndarray, zz: np.ndarray
) -> np.ndarray:
    """Calculates the area of surface triangles (bottom half) and reurns the 
    array of each triangle formed by
    triangle[0,0] -> (vertex[0,0], vertex[1,0], vertex[0,1])
    area[0,0] -> area(triangle[0,0])



    Parameters
    ----------
    xx : np.ndarray
        _description_
    yy : np.ndarray
        _description_
    zz : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        Area matrix of the input vertices dimensions are (n-1), (m-1) of input shape.
    """

    # Calculate the vectors for two sides of each small triangle
    # This calculates only half of the surface
    vec_v = np.array(
        [np.diff(xx, axis=0), np.diff(yy, axis=0), np.diff(zz, axis=0)]
    ).transpose(1, 2, 0)
    vec_u = np.array(
        [np.diff(xx, axis=1), np.diff(yy, axis=1), np.diff(zz, axis=1)]
    ).transpose(1, 2, 0)

    # Calculate the cross product, which gives a vector perpendicular to the triangle's surface
    # The magnitude of this vector is twice the area of the triangle
    cross_product = np.cross(vec_u[:-1, :], vec_v[:, :-1], axis=2)
    areas = np.linalg.norm(cross_product, axis=2)

    return areas / 2


def evaluate_surface_intersection(
    xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, p: np.ndarray, n: np.ndarray
) -> SpatialArray:
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

    sdf = signed_distance_from_plane(xx, yy, zz, p, n)

    sdf_sign = sdf > 0

    indices = find_transitions(sdf_sign)

    intersection_points = np.array(
        [
            line_plane_intersection(p1, p2, p, n)
            for p1, p2 in generate_line_segments(xx, yy, zz, indices)
        ]
    )

    # Calculate the variation for each column (max value - min value)
    variations = intersection_points.max(axis=0) - intersection_points.min(axis=0)

    # Find the index of the column with the largest variation
    max_delta_col = np.argmax(variations)

    sorted_idx = intersection_points[:, max_delta_col].argsort()

    # Sort the array based on the values in the identified column
    return SpatialArray(intersection_points[sorted_idx])


def signed_distance_from_plane(
    xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, p: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """Returns the signed distance function between points of a surface and a plane defined
    by a point p and a normal vector n.

    Parameters
    ----------
    xx : np.ndarray
        _description_
    yy : np.ndarray
        _description_
    zz : np.ndarray
        _description_
    p : np.ndarray
        _description_
    n : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    # First we calculate the Q vectors.
    # Stack the xx, yy, zz grids to form a 3D array where each element is a 3D coordinate
    coordinates = np.stack((xx, yy, zz), axis=-1)
    # Subtract p from each 3D coordinate
    q = coordinates - p
    # Perform the dot product between elements in the k axis.
    sdf_plane = np.einsum("ijk,k->ij", q, n)
    return sdf_plane


def find_transitions(
    sdf_arr: np.ndarray,
) -> list[LineSegmentIndices]:
    """
    Find indices in a array where there is a transition from True to False
    or from positive to negative.

    Parameters:
    arr (list): A list of boolean or numeric values.

    Returns:
    list: Indices of elements where the specified transitions occur.
    """

    if sdf_arr.dtype is not np.dtype("bool"):
        sdf_arr = sdf_arr > 0

    transitions = []

    for i in range(sdf_arr.shape[0]):
        for j in range(sdf_arr.shape[1]):
            try:
                u_idx = ((i, j), (i, j + 1))

                if sdf_arr[i, j] != sdf_arr[i, j + 1]:  # Horizontal change
                    transitions.append(u_idx)
            except (
                IndexError
            ):  # Allows the last row to be checked fo changes within row
                pass
            try:
                v_idx = ((i, j), (i + 1, j))

                if sdf_arr[i, j] != sdf_arr[i + 1, j]:  # Vertical change
                    transitions.append(v_idx)
            except (
                IndexError
            ):  # Allows the last column to be checked for changes within column
                pass

    return transitions

def find_transitions_np(sdf_arr: np.ndarray) -> list[LineSegmentIndices]:
    """
    Find indices in an array where there is a transition from True to False
    or from positive to negative.

    Parameters:
    sdf_arr (np.ndarray): A NumPy array of boolean or numeric values.

    Returns:
    List[LineSegmentIndices]: Indices of elements where the specified transitions occur.
    """

    # Convert to boolean array if not already (True for positive, False for non-positive)
    if sdf_arr.dtype != np.bool_:
        sdf_arr = sdf_arr > 0

    transitions = []

    # Find horizontal transitions
    horizontal_changes = sdf_arr[:, :-1] != sdf_arr[:, 1:]
    h_indices = np.argwhere(horizontal_changes)
    for i, j in h_indices:
        transitions.append(((i, j), (i, j + 1)))

    # Find vertical transitions
    vertical_changes = sdf_arr[:-1, :] != sdf_arr[1:, :]
    v_indices = np.argwhere(vertical_changes)
    for i, j in v_indices:
        transitions.append(((i, j), (i + 1, j)))

    return transitions

def generate_line_segments(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    indices: list[LineSegmentIndices],
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate 3D line segments from coordinate arrays and index pairs.

    Takes coordinate arrays (xx, yy, zz) of a surface, and a list of
    index pairs. Each index pair specifies the start and end points of a line
    segment of the surface. The function yields these line segments as tuples
    of numpy arrays.

    Parameters:
    xx (np.ndarray): Array of x-coordinates.
    yy (np.ndarray): Array of y-coordinates.
    zz (np.ndarray): Array of z-coordinates.
    indices (List[LineSegmentIndices]): List of index pairs defining line segments.

    Yields:
    Generator[Tuple[np.ndarray, np.ndarray], None, None]: A generator that yields
    tuples of numpy arrays representing the start and end points of each line segment.
    """

    coordinates = np.stack((xx, yy, zz), axis=-1)

    for id_a, id_b in indices:
        p1 = coordinates[id_a[0], id_a[1]]
        p2 = coordinates[id_b[0], id_b[1]]
        yield p1, p2


def line_plane_intersection(
    p1: np.ndarray, p2: np.ndarray, p: np.ndarray, n: np.ndarray
) -> np.ndarray | None:
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

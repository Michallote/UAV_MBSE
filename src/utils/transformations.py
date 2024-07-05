"""Geometric Transformations module."""

from typing import Literal

import numpy as np


def transform_coordinates(
    coordinates: np.ndarray,
    center: np.ndarray,
    twist: float,
    chord: float,
    offset: np.ndarray,
    wingspan: float,
) -> np.ndarray:
    """
    Applies translations and rotations to airfoil data points

    Parameters
    ----------
    section : Section object
        Contains all relevant information about the transformation.

    Returns
    -------
    cords3d : np.ndarray
        Curve Coordinates.

    """
    # twist = -np.radians(section.Twist)
    # chord = section.chord
    # offset = np.array([section.xOffset,section.yOffset])
    # wingspan = section.wingspan

    # coordinates = section.airfoil.get_data( dim = '2D', output_format = 'np')
    # center = section.airfoil.center

    if twist != 0:
        rotmat = rotation_matrix2d(twist)
        # coordinates = [rotmat@r for r in (coordinates-center)] + center
        coordinates = np.dot(coordinates - center, rotmat.T) + center

    coordinates = coordinates * chord + offset
    # Dimension adder  (3 x 2) @ (2 x 1) = (3 x 1)
    matrix_to_r3 = np.array([[1, 0], [0, 1], [0, 0]])
    # Broadcast the result over the rows of B
    cords3d = np.dot(coordinates, matrix_to_r3.T) + np.array([0, 0, wingspan])

    # cords3d = np.c_[coordinates, wingspan*np.ones(len(coordinates))]

    return cords3d


def transform_to_global_coordinate_system(
    array: np.ndarray, globalpos: np.ndarray, is_fin: bool
) -> np.ndarray:
    """
    Transforms the curve from its local reference frame to the global coordinate system GCS

    Parameters
    ----------
    array : np.ndarray
        Curves.
    globalpos : np.ndarray
        DESCRIPTION.
    surface_type : SurfaceType
        DESCRIPTION.

    Returns
    -------
    array : np.ndarray
        The array with the applied transformations.

    """

    if is_fin:
        rotmat3d = rotation_matrix3d(-90, axis="x", units="degrees")
        # cords3d = [rotmat3d@r for r in cords3d]
        # Broadcast the result over the rows of B
        array = np.dot(array, rotmat3d.T)

    array = array + globalpos
    return array


def rotation_matrix2d(theta: float) -> np.ndarray:
    """2D Rotation Matrix"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def rotation_matrix3d(theta: float, axis="x", units="radians") -> np.ndarray:
    """3D Rotation Matrix"""
    if units == "degrees":
        theta = np.radians(theta)

    c, s = np.cos(theta), np.sin(theta)

    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    elif axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    elif axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    else:
        raise NameError("Invalid axis")


def get_ref_coordinate_system(
    reference_system="SW", units="mm"
) -> tuple[float, list[str], dict[str, int]]:
    """_summary_

    Parameters
    ----------
    reference_system : str, available options: XFLR5 SW Python MATLAB
        data to be transformed into the reference system, by default "SW"
    units : str, 'm' -> meters, 'mm' -> milimeters, 'in' -> inches
        _description_, by default "mm"

    Returns
    -------
    tuple[float, list[str], dict[str, int]]
        units, reference_system, reflect_axis
    """
    units_dict = {"m": 1.0, "mm": 1000.0, "in": 1 / 39.3701}

    units_factor = units_dict[units]

    coordinate_system = {
        "XFLR5": ["x", "z", "y"],
        "SW": ["z", "y", "x"],
        "Python": ["x", "y", "z"],
        "MATLAB": ["x", "y", "z"],
    }
    reflections = {
        "XFLR5": {"x": 1, "z": -1, "y": 1},
        "SW": {"z": 1, "y": 1, "x": -1},
        "Python": {"x": 1, "y": 1, "z": 1},
        "MATLAB": {"x": 1, "y": 1, "z": 1},
    }

    reference_system_order = coordinate_system[reference_system]
    reflect_axis = reflections[reference_system]

    return units_factor, reference_system_order, reflect_axis


def reflect_curve_by_plane(data: np.ndarray, normal_vector: np.ndarray) -> np.ndarray:
    """
    Reflects a set of 3D points (a curve) across a plane defined by a normal vector passing through the origin.

    Args:
        data (np.ndarray): A 3D numpy array of shape (n, 3) representing the coordinates of n points.
        normal_vector (np.ndarray): A 1D numpy array of shape (3,) representing the normal vector of the plane.

    Returns:
        np.ndarray: A 3D numpy array of the reflected points with the same shape as `data`.
    """
    # Normalize the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # Calculate the dot product of each point with the normal vector
    dot_product = np.dot(data, normal_vector)

    # Calculate the reflected points
    reflected_points = data - 2 * np.outer(dot_product, normal_vector)

    return reflected_points


def get_plane_normal_vector(plane: Literal["xy", "xz", "yz"]) -> np.ndarray:
    """Returns the normal vector associated with a basis plane in 3D cartesian coordinates

    Parameters
    ----------
     - mirror_plane : Literal[&quot;xy&quot;, &quot;xz&quot;, &quot;yz&quot;]
            The mirror plane to retrieve the normal vector

    Returns
    -------
    np.ndarray
        Normal vector of the input plane
    """

    normal_vectors = {
        "xy": np.array([0, 0, 1]),
        "xz": np.array([0, 1, 0]),
        "yz": np.array([0, 1, 0]),
    }
    return normal_vectors[plane]

from functools import partial
from typing import Any

import numpy as np

from src.geometry.surfaces import triangle_normal


def create_triangle_prism(
    triangle_coordinates: np.ndarray, thickness: float, midsurface: bool = True
) -> np.ndarray:
    """
    Create the coordinates of a triangular prism from a base triangle and a specified thickness.

    This function generates a triangular prism by extending a base triangle along its normal vector by a given thickness.
    The base triangle is defined by its three vertices.

    Parameters:
        triangle_coordinates (np.ndarray): An array containing the 3D coordinates of the three vertices of the triangle.
        thickness (float): The distance by which the prism is extended along the triangle's normal vector.

    Returns:
        np.ndarray: An array containing the coordinates of the four vertices of the triangular prism.
    """
    v1, v2, v3 = triangle_coordinates
    normal = triangle_normal(v1, v2, v3)
    v4 = v1 + thickness * normal
    prism = np.array([v1, v2, v3, v4])

    if midsurface:
        prism = prism - 0.5 * thickness * normal
    return prism


def triangulate_mesh(x, y, z, i, j, k) -> np.ndarray[Any, np.dtype[Any]]:
    triangle_indices = np.vstack((i, j, k)).T
    vertices = np.vstack((x, y, z)).T
    triangles = vertices[triangle_indices]
    return triangles


def compute_inertia_tensor_of_shell(
    triangles: np.ndarray, density: float, thickness: float, midsurface: bool = True
) -> np.ndarray:
    """$x^2$"""

    partial_create_prism = partial(
        create_triangle_prism, thickness=thickness, midsurface=midsurface
    )
    prisms = list(map(partial_create_prism, triangles))

    prisms_coordinates = np.array(prisms)

    inertia_tensor = np.stack(
        list(map(triangle_prism_inertia_tensor, prisms_coordinates)), axis=-1
    )
    inertia_tensor = np.sum(inertia_tensor, axis=-1) * density

    return inertia_tensor


def triangle_prism_inertia_tensor(prism_coordinates):

    x, y, z = prism_coordinates.T

    Ixx = squared_moment_term(y) + squared_moment_term(z)  # $\int_m{y^2 + z^2}dm$
    Iyy = squared_moment_term(x) + squared_moment_term(z)  # $\int_m{x^2 + z^2}dm$
    Izz = squared_moment_term(x) + squared_moment_term(y)  # $\int_m{x^2 + y^2}dm$
    Ixy = product_moment_term(x, y)  # $\int_m{x*y}dm$
    Ixz = product_moment_term(x, z)  # $\int_m{x*z}dm$
    Iyz = product_moment_term(y, z)  # $\int_m{y*z}dm$

    jacobian = np.linalg.det(transformation_jacobian(prism_coordinates))

    return (
        np.array([[Ixx, -Ixy, -Ixz], [-Ixy, Iyy, -Iyz], [-Ixz, -Iyz, Izz]]) * jacobian
    )


def squared_moment_term(x: np.ndarray) -> float:
    """
    Compute the second moment of mass (integral of x^2 over the volume) for a tetrahedron defined by its vertices.

    The function calculates the integral of x^2 over the volume of a triangular prism with vertices at coordinates (x1, y1, z1),
    (x2, y2, z2), (x3, y3, z3), and (x4, y4, z4), under a linear transformation mapping a standard coordinate system
    to the a tetrahedron's coordinate system. The result is obtained using the formula derived from integrating the squared
    x-coordinate of the transformation, accounting for the volume of the transformed region.

    The specific transformation and integration are:
        x = x1 + (x2 - x1) * ε + (x3 - x1) * η + (x4 - x1) * ζ
    where ε, η, and ζ vary from 0 to 1 with the constraints that η < 1-ε and ζ < 1.
    This calculates the area of a triangular prism.


    Parameters:
        x1, x2, x3, x4 (float): x-coordinates of the tetrahedron vertices.

    Returns:
        float: The second moment of mass, which is a scalar value representing the integral of x^2 over the tetrahedron's volume.
    """
    x1, x2, x3, x4 = x

    return (1 / 12) * (
        x1**2
        + x2**2
        + x3**2
        + 2 * x4**2
        + 2 * x3 * x4
        + x2 * (x3 + 2 * x4)
        - x1 * (x2 + x3 + 2 * x4)
    )


def product_moment_term(x: np.ndarray, y: np.ndarray):

    x1, x2, x3, x4 = x
    y1, y2, y3, y4 = y

    return (1 / 24) * (
        -x3 * y1
        - 2 * x4 * y1
        + x3 * y2
        + 2 * x4 * y2
        + 2 * x3 * y3
        + 2 * x4 * y3
        + x1 * (2 * y1 - y2 - y3 - 2 * y4)
        + 2 * x3 * y4
        + 4 * x4 * y4
        + x2 * (-y1 + 2 * y2 + y3 + 2 * y4)
    )


def transformation_jacobian(tetrahedron_coordinates: np.ndarray) -> np.ndarray:
    """
    Compute the Jacobian matrix of the transformation from the standard coordinate system
    to the coordinate system defined by a tetrahedron with specified vertex coordinates.

    This transformation maps a point (ε, η, ζ) in the standard coordinate system to a point
    (x, y, z) in the tetrahedron's coordinate system using the linear combination:
        x = x1 + (x2 - x1) * ε + (x3 - x1) * η + (x4 - x1) * ζ
        y = y1 + (y2 - y1) * ε + (y3 - y1) * η + (y4 - y1) * ζ
        z = z1 + (z2 - z1) * ε + (z3 - z1) * η + (z4 - z1) * ζ
    where (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4) are the coordinates of the tetrahedron's vertices.

    Parameters:
        tetrahedron_coordinates (np.ndarray): A numpy array containing the coordinates of the tetrahedron vertices.
        The array should have the shape (4, 3), where each row represents a vertex and the columns correspond to
        x, y, and z coordinates respectively.

    Returns:
        np.ndarray: A 3x3 Jacobian matrix representing the partial derivatives of the transformation functions
        with respect to ε, η, and ζ.

    Example:
        >>> tetrahedron_coordinates = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> jacobian(tetrahedron_coordinates)
        array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])
    """

    ((x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)) = tetrahedron_coordinates

    return np.array(
        [
            [-x1 + x2, -x1 + x3, -x1 + x4],
            [-y1 + y2, -y1 + y3, -y1 + y4],
            [-z1 + z2, -z1 + z3, -z1 + z4],
        ]
    )


if __name__ == "__main__":

    triangle = np.array(
        [
            [0.25, 0, 0],
            [0.70315389352, 0, -0.21130913087],
            [0.29357787137, 0, -0.49809734905],
        ]
    )
    density = 1000
    thickness = 0.15

    prism = create_triangle_prism(triangle, thickness=thickness)

    density * triangle_prism_inertia_tensor(prism)

    triangle = (
        np.array(
            [
                [250.0, -76.24407719, -50.562258709],
                [703.153893518, 8.015242, -244.345425983],
                [293.577871374, 122.371776976, -507.347450676],
            ]
        )
        / 1000
    )

    prism = create_triangle_prism(triangle, thickness=thickness, midsurface=False)

    density * triangle_prism_inertia_tensor(prism)

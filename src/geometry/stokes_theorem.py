import numpy as np

from geometry.intersection import enforce_closed_curve
from geometry.transformations import compute_curve_normal


def compute_stokes_curve_area(curve: np.ndarray) -> float:
    """
    Computes the area enclosed by a closed 3D curve using Stokes' theorem.

    Parameters
    ----------
    curve : np.ndarray
        An array of shape (n, 3) representing the vertices of the curve
        in 3D space. The curve does not need to be explicitly closed;
        the function will ensure closure if required.

    Returns
    -------
    float
        The area enclosed by the curve.
    """
    curve = enforce_closed_curve(curve)
    nx, ny, nz = compute_curve_normal(curve)
    x, y, z = curve.T

    xi, xf = x[:-1], x[1:]
    yi, yf = y[:-1], y[1:]
    zi, zf = z[:-1], z[1:]

    areas = (
        nx * (yi * zf - zi * yf) + ny * (zi * xf - xi * zf) + nz * (xi * yf - yi * xf)
    )
    area = (1 / 2) * np.sum(areas)

    return area


def compute_stokes_centroid(curve) -> np.ndarray:
    """
    Computes the centroid of a closed 3D curve using Stokes' theorem. The curve is assumed
    to lie in a plane and is represented as a series of connected vertices in 3D space.

    Parameters
    ----------
    curve : np.ndarray
        An array of shape (n, 3) representing the vertices of the curve
        in 3D space. The curve does not need to be explicitly closed;
        the function will ensure closure if required.

    Returns
    -------
    np.ndarray
        A 1D array [xc, yc, zc] representing the centroid of the curve.
    """

    xc = compute_x_centroid(curve)
    yc = compute_y_centroid(curve)
    zc = compute_z_centroid(curve)

    return np.array([xc, yc, zc])


def compute_x_centroid(curve: np.ndarray) -> float:
    """
    Compute the x-coordinate of the centroid of a 3D planar curve using Stokes' Theorem.

    This function calculates the x-centroid (x_c) of a closed 3D planar curve based on
    the vector field derived through the application of Stokes' Theorem. The curve is assumed
    to lie in a plane and is represented as a series of connected vertices in 3D space.

    Parameters:
    ----------
    curve : np.ndarray
        A (N x 3) NumPy array representing the vertices of the 3D curve, where each
        row is a point (x, y, z). The curve must be closed (i.e., the last point should
        connect to the first).

    Returns:
    -------
    float
        The x-coordinate of the centroid of the curve (x_c).

    Example:
    --------
    >>> curve = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0]])
    >>> xc = compute_x_centroid(curve)
    >>> print(xc)
    0.0  # Expected for a symmetric curve in the xy-plane centered at the origin.
    """

    curve = enforce_closed_curve(curve)
    nx, ny, nz = compute_curve_normal(curve)

    x, y, z = curve.T

    xi, xf = x[:-1], x[1:]
    yi, yf = y[:-1], y[1:]
    zi, zf = z[:-1], z[1:]

    area = compute_stokes_curve_area(curve)

    xc = (
        nx * (zf - zi) * (xf * (2 * yf + yi) + xi * (yf + 2 * yi))
        + ny * (xf + xi) * (xf * zi - xi * zf)
        + ny * (yf + yi) * (yi * zf - yf * zi)
        + nz * (xf + xi) * (xi * yf - xf * yi)
    )

    xc = (1 / 6) * np.sum(xc) / area

    return xc


def compute_y_centroid(curve) -> float:
    """
    Computes the y-coordinate of the centroid of a closed 3D curve.

    Parameters
    ----------
    curve : np.ndarray
        An array of shape (n, 3) representing the vertices of the curve
        in 3D space. The curve does not need to be explicitly closed;
        the function will ensure closure if required. The curve is assumed
    to lie in a plane and is represented as a series of connected vertices in 3D space.

    Returns
    -------
    float
        The y-coordinate of the centroid.
    """

    curve = enforce_closed_curve(curve)
    nx, ny, nz = compute_curve_normal(curve)

    x, y, z = curve.T

    xi, xf = x[:-1], x[1:]
    yi, yf = y[:-1], y[1:]
    zi, zf = z[:-1], z[1:]

    area = compute_stokes_curve_area(curve)

    yc = (
        nz * (yf + yi) * (xi * yf - xf * yi)
        - nz * (zf + zi) * (xi * zf - xf * zi)
        - nx * (yf + yi) * (-yi * zf + yf * zi)
        + ny * (xf - xi) * (yf * (2 * zf + zi) + yi * (zf + 2 * zi))
    )

    yc = (1 / 6) * np.sum(yc) / area

    return yc


def compute_z_centroid(curve) -> float:
    """
    Computes the z-coordinate of the centroid of a closed 3D curve.

    Parameters
    ----------
    curve : np.ndarray
        An array of shape (n, 3) representing the vertices of the curve
        in 3D space. The curve does not need to be explicitly closed;
        the function will ensure closure if required. The curve is assumed
    to lie in a plane and is represented as a series of connected vertices in 3D space.

    Returns
    -------
    float
        The z-coordinate of the centroid.
    """

    curve = enforce_closed_curve(curve)
    nx, ny, nz = compute_curve_normal(curve)

    x, y, z = curve.T

    xi, xf = x[:-1], x[1:]
    yi, yf = y[:-1], y[1:]
    zi, zf = z[:-1], z[1:]

    area = compute_stokes_curve_area(curve)

    zc = (
        nx * (zf + zi) * (yi * zf - yf * zi)
        - nx * (xf + xi) * (xf * yi - xi * yf)
        - ny * (zf + zi) * (xi * zf - xf * zi)
        + nz * (yf - yi) * (xf * (2 * zf + zi) + xi * (zf + 2 * zi))
    )

    z_centroid = (1 / 6) * np.sum(zc) / area

    return z_centroid

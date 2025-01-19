import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from geometry.intersection import enforce_closed_curve


def curve_area(coordinates: np.ndarray) -> float:
    """Calculates the area of a closed contour using greens theorem
    https://leancrew.com/all-this/2018/01/greens-theorem-and-section-properties/

    Parameters
    ----------
    coordinates : np.ndarray
        Coordinates of the 2D curve

    Returns
    -------
    float
        area of the region enclosed by the curve
    """
    coordinates = orient_counter_clockwise(coordinates)

    x, y = coordinates.T
    xi = x
    xf = np.roll(x, -1)
    yi = y
    yf = np.roll(y, -1)
    return np.sum((xf + xi) * (-yi + yf) / 2)


def inertia_of_shell(coordinates: np.ndarray, x0=None, y0=None):
    """
    Compute the components of the area moment of inertia tensor using
    Green's Theorem. The contour must be closed. for a closed shell

    Parameters
    ----------
    coordinates : np.ndarray
        Coordinates of the closed contour (N x 2 array).
    x0 : float, optional
        X-coordinate of the reference point (default: centroid).
    y0 : float, optional
        Y-coordinate of the reference point (default: centroid).

    Returns
    -------
    tuple
        Ixx, Ixy, Iyy, Jz (area polar moment of inertia).

    Raises
    ------
    ValueError
        If the input is invalid or has fewer than 3 points.
    """
    if coordinates.shape[0] < 3:
        raise ValueError("Contour must have at least 3 points.")

    x, y = coordinates.T
    # x = x - x0
    # y = y - y0
    xi = x
    xf = np.roll(x, -1)
    yi = y
    yf = np.roll(y, -1)

    # Curve length
    L = np.sqrt((yf - yi) ** 2 + (xf - xi) ** 2)

    # Centroid
    xc = (1 / (2 * np.sum(L))) * np.sum((xi + xf) * L)
    yc = (1 / (2 * np.sum(L))) * np.sum((yi + yf) * L)

    if (x0 is None) or (y0 is None):
        print("Computing Ixx, Iyy, Ixy from the centroid")
        xf = xf - xc
        xi = xi - xc
        yf = yf - yc
        yi = yi - yc
    elif isinstance(x0, (float, int)) and isinstance(y0, (float, int)):
        xf = xf - x0
        xi = xi - x0
        yf = yf - y0
        yi = yi - y0

    Ixx = (1 / 3) * (yi**2 + yi * yf + yf**2) * L
    Ixy = 1 / 6 * (2 * xf * yf + xi * yf + xf * yi + 2 * xi * yi) * L
    Iyy = (1 / 3) * (xi**2 + xi * xf + xf**2) * L

    Jz = Ixx + Iyy

    return np.sum(Ixx), np.sum(Ixy), np.sum(Iyy), np.sum(Jz)


def curve_centroid(curve: np.ndarray) -> np.ndarray:
    """
    Compute the centroid of a closed curve using Green's theorem.

    Parameters
    ----------
    curve : np.ndarray
        Coordinates of the 2D curve (N x 2 array).

    Returns
    -------
    np.ndarray
        The centroid as a 2D array [xc, yc].
    """

    curve = orient_counter_clockwise(enforce_closed_curve(curve))
    area = curve_area(curve)

    x, y = curve.T

    xi, xf = x[:-1], x[1:]
    yi, yf = y[:-1], y[1:]

    # Centroid
    sc = 1 / (6 * area)
    xc = sc * np.sum((xi + xf) * (xi * yf - xf * yi))
    yc = sc * np.sum((yi + yf) * (xi * yf - xf * yi))

    return np.array([xc, yc])


def centroid_drang(pts):
    "Location of centroid."

    if pts[0] != pts[-1]:
        pts = pts + pts[:1]
    x = [c[0] for c in pts]
    y = [c[1] for c in pts]
    sx = sy = 0
    a = curve_area(np.array(pts))
    for i in range(len(pts) - 1):
        sx += (x[i] + x[i + 1]) * (x[i] * y[i + 1] - x[i + 1] * y[i])
        sy += (y[i] + y[i + 1]) * (x[i] * y[i + 1] - x[i + 1] * y[i])
    return sx / (6 * a), sy / (6 * a)


def orient_counter_clockwise(curve: np.ndarray) -> np.ndarray:
    """Enforces that the closed curves are oriented in a counter-clockwise manner

    Parameters
    ----------
    curve : np.ndarray
        Counter clockwise oriented curve

    Returns
    -------
    np.ndarray
        counter clockwise ordered coordinates
    """

    curve = enforce_closed_curve(curve)
    polygon = Polygon(curve)
    # Ensure the polygon vertices are oriented counterclockwise
    polygon = orient(polygon, sign=1.0)
    return np.array(polygon.exterior.coords)

"""Interpolation functions module"""
from typing import Callable, Tuple

import numpy as np


def _linear_interpolate(curve: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Perform linear interpolation on a curve at given indices.

    Parameters
    ----------
    curve : np.ndarray
        The curve to be interpolated, can be (n, m) dimensional.
    indices : np.ndarray
        The indices at which to interpolate.

    Returns
    -------
    np.ndarray
        The interpolated values.
    """
    left_indices = np.floor(indices).astype(int)
    right_indices = np.ceil(indices).astype(int)
    fraction = indices - left_indices

    delta = curve[right_indices] - curve[left_indices]
    return curve[left_indices] + delta * fraction[:, None]


def resample_curve(curve: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Resample a curve (array) based on linear interpolation.

    Parameters
    ----------
    curve : np.ndarray
        The curve to be resampled, can be (n, m) dimensional.
    num_samples : int
        Number of samples in the resampled curve.

    Returns
    -------
    np.ndarray
        The resampled curve.
    """
    original_length = curve.shape[0]
    interpolated_indices = np.linspace(0, original_length - 1, num_samples)
    return _linear_interpolate(curve, interpolated_indices)


def vector_interpolation(
    x: np.ndarray, xp: np.ndarray, curve: np.ndarray
) -> np.ndarray:
    """
    Perform multi-dimensional interpolation of curves.

    Parameters
    ----------
    x : float
        The value to interpolate.
    xp : np.ndarray
        The x-coordinates of the data points.
    curve : np.ndarray
        The curve to interpolate.

    Returns
    -------
    np.ndarray
        The interpolated values.
    """
    interpolation_indices = np.interp(x, xp, np.arange(len(xp)))
    return _linear_interpolate(curve, interpolation_indices)


def find_max(
    f: Callable[[np.ndarray], np.ndarray], n_iter: int = 4
) -> Tuple[float, float]:
    """
    Finds the maximum value of a function using iterative interpolation.

    Args:
    f (Callable[[np.ndarray], np.ndarray]): The function to be maximized.
    n_iter (int, optional): Number of iterations for refining the search. Default is 4.

    Returns:
    List[float]: The list of function values at the interpolation points in the last iteration.

    """
    num_points = 10  # Number of interpolation points

    # Define the x limits for interpolation
    x_interp_min = 0.05
    x_interp_max = 0.95

    # Iterate to refine the search for the maximum
    for _ in range(n_iter):
        # Define the x points for interpolation
        x_interp = np.linspace(x_interp_min, x_interp_max, num_points)

        # Obtain function values at interpolation points
        values = f(x_interp)

        # Sort the function values and get the indices of the highest values
        sorted_indices = np.argsort(values)

        # Indices of the top two highest values
        i_max = sorted_indices[-1]
        i_second_max = sorted_indices[-2]

        # Redefine the x limits for the next iteration
        x_interp_max = x_interp[i_max]
        x_interp_min = x_interp[i_second_max]

    return x_interp_max, np.max(values)  # type: ignore

"""Interpolation functions module"""

from typing import Callable, Tuple

import numpy as np


def ndarray_linear_interpolate(curve: np.ndarray, indices: np.ndarray) -> np.ndarray:
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

    # Reshape fraction to be broadcast-compatible with curve
    fraction_shape = (-1,) + (1,) * (curve.ndim - 1)
    fraction = fraction.reshape(fraction_shape)

    delta = curve[right_indices] - curve[left_indices]
    return curve[left_indices] + delta * fraction


def resample_curve(curve: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Resamples the curve (array) to a target number of points while preserving
    the distribution of the original points. Based on linear interpolation.

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
    return ndarray_linear_interpolate(curve, interpolated_indices)


def resample_curve_equidistant(
    curve: np.ndarray, target_segment_length: float
) -> np.ndarray:
    """
    Resamples the curve (array) to ensure equidistant points, irrespective of the
    original point distribution, based on linear interpolation with a given element length.

    Parameters
    ----------
    curve : np.ndarray
        The curve to be resampled, can be (n, m) dimensional.
    target_segment_length : float
        The desired length of each element in the resampled curve.

    Returns
    -------
    np.ndarray
        The resampled curve.
    """
    # Compute distances between consecutive points in the curve
    segment_lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    # Start cumulative distances at 0
    segment_lengths = np.insert(segment_lengths, 0, 0)

    # Cumulative distances along the curve
    cumulative_distances = np.cumsum(segment_lengths)
    original_length = cumulative_distances[-1]

    # Determine the number of segments based on the target segment length
    num_samples = max(int(np.round(original_length / target_segment_length)), 1)
    # Generate evenly spaced distances along the curve
    target_distances = np.linspace(0, original_length, num_samples + 1)

    # Map the target distances to indices for interpolation
    interpolated_indices = np.interp(
        target_distances,
        cumulative_distances,
        np.arange(len(cumulative_distances)),
    )

    return ndarray_linear_interpolate(curve, interpolated_indices)


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
    return ndarray_linear_interpolate(curve, interpolation_indices)


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


# Pad matrix to normalize
def pad_arrays(arr1: np.ndarray, arr2: np.ndarray, constant_values=-999) -> np.ndarray:
    """Pads array to normalize the shape

    Parameters
    ----------
     - arr1 : _type_
            array 1
     - arr2 : _type_
            array 2

    Returns
    -------
    np.ndarray
        Single hstacked padded array
    """
    max_cols = max(arr1.shape[0], arr2.shape[0])
    mat1_padded = np.pad(
        arr1,
        ((0, max_cols - arr1.shape[0]), (0, 0)),
        mode="constant",
        constant_values=constant_values,
    )
    mat2_padded = np.pad(
        arr2,
        ((0, max_cols - arr2.shape[0]), (0, 0)),
        mode="constant",
        constant_values=constant_values,
    )

    return np.hstack([mat1_padded, mat2_padded])


def compute_curve_length(coordinates: np.ndarray) -> float:
    """Computes the curve length from straight segments between coordinates.

    Parameters
    ----------
    coordinates : np.ndarray
        Array of curve coordinates, can be either 2D or 3D

    Returns
    -------
    float
        Curve length computed
    """
    return np.sum(np.linalg.norm(np.diff(coordinates, axis=0), axis=1))

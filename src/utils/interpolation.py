"""Interpolation functions module"""
from typing import Callable, Tuple

import numpy as np


def resample_curve(array3d, nsamples: int):
    """
    Resample an array based on linear interpolation between indexes.

    Parameters
    ----------
    array3d : np.array()
              Can be (n,m) dimentional
    nsamples : int


    Returns
    -------
    resample : TYPE
        DESCRIPTION.

    """
    n_orig = len(array3d)  # Read original array size
    t = np.linspace(
        0, n_orig - 1, nsamples
    )  # Resample as if index was the independent variable
    np_int = np.vectorize(int)  # Create function applicable element-wise
    right = np_int(np.ceil(t))  # Array of upper bounds of each new element
    left = np_int(np.floor(t))  # Array of lower bounds of each new element

    # Linear interpolation p = a + (b-a)*t

    delta = array3d[right] - array3d[left]  # (b-a)
    t_p = t - left  # t Array of fraction between a -> b for each element
    resample = (
        array3d[left] + delta * t_p[:, None]
    )  # p Element - wise Linear interpolation

    return resample


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

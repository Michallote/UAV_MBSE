# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 01:05:31 2022

@author: Michel Gordillo
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

n = 150

theta = np.linspace(0, 4 * np.pi, n)

array3d = np.array([np.cos(theta), np.sin(theta), theta / (4 * np.pi)]).T

nsamples = 30


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
    t_p = t - left  # Array of fraction between a -> b for each element
    resample = (
        array3d[left] + delta * t_p[:, None]
    )  # Element - wise Linear interpolation

    return resample


arr_resample = resample_curve(array3d, nsamples)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(*array3d.T, marker=".", markerfacecolor="black")
ax.plot(*arr_resample.T, marker=".", ls="")

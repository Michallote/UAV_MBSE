# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:46:54 2022

@author: Michel Gordillo
"""
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import LightSource

from legacy._legacy_main_MBSE import (AeroSurface, Aircraft, Section,
                                      SurfaceType)


@dataclass
class GeometricCurve:
    """Represents a parametric curve with data as lists of the x, y, and z coordinates"""

    x: np.ndarray = np.array([])
    y: np.ndarray = np.array([])
    z: np.ndarray = np.array([])

    def set_data(self, coordinates):

        if coordinates.shape[1] == 2:
            self.data = pd.DataFrame(coordinates, columns=["x", "y"])

        elif coordinates.shape[1] == 3:
            self.data3d = pd.DataFrame(coordinates, columns=["x", "y", "z"])
            self.x = self.data3d.x.to_numpy()
            self.y = self.data3d.y.to_numpy()
            self.z = self.data3d.z.to_numpy()


@dataclass
class GeometricSurface:
    """Represents a geometric surface with data as lists of the x, y, and z coordinates
    for each location of a patch. A surface from the points is specified by the matrices
    and will then connect those points by linking the values next to each other in the matrix
    """

    xx: np.ndarray = np.array([])
    yy: np.ndarray = np.array([])
    zz: np.ndarray = np.array([])

    def __post_init__(self) -> None:
        self.curves: List[GeometricCurve] = []

    def add_curve(self, curve: GeometricCurve) -> None:
        """Add a parametric curve to the list of surfaces."""
        self.curves.append(curve)

    def surf_from_curves(self):

        self.xx = [curve.x for curve in self.curves]
        self.yy = [curve.y for curve in self.curves]
        self.zz = [curve.z for curve in self.curves]

    def set_color(self, color) -> None:
        self.color: "str" = color


class GeometryProcessor:
    """Behaviour Oriented Class
    Processes Aircraft to create 3D models."""

    def __init__(self, UAV: Aircraft) -> None:
        self.aircraft = UAV
        self.surfaces: List[GeometricSurface] = []

    def create_geometries(self) -> None:
        for surface in self.aircraft.surfaces:
            geosurface = GeometricSurface()
            globalpos = surface.position
            surf_type = surface.surf_type  # SurfaceType.FIN

            for section in surface.sections:
                curve = GeometricCurve()
                airfoil_cordinates = transform_coordinates(section)
                curve.set_data(airfoil_cordinates)
                curve3d = transform_to_GCS(airfoil_cordinates, globalpos, surf_type)
                curve.set_data(curve3d)

                geosurface.add_curve(curve)

            geosurface.surf_from_curves()

            self.add_surface(geosurface)

    def add_surface(self, surface: GeometricSurface) -> None:
        """Add a geometric surface (x,y,z) data matrices to the list of surfaces."""
        self.surfaces.append(surface)

    def find_surfaces(self, surf_type: SurfaceType) -> List[AeroSurface]:
        """Find all lifting surfaces with a particular type in the surfaces list"""
        return [surface for surface in self.surfaces if surface.surf_type is surf_type]


def transform_coordinates(section: Section) -> np.ndarray:
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
    twist = -np.radians(section.Twist)
    chord = section.chord
    offset = np.array([section.xOffset, section.yOffset])
    wingspan = section.wingspan

    coordinates = section.airfoil.get_data(dim="2D", output_format="np")
    center = section.airfoil.center

    if twist != 0:
        rotmat = rotation_matrix2d(twist)
        coordinates = [rotmat @ r for r in (coordinates - center)] + center

    coordinates = coordinates * chord + offset

    cords3d = np.c_[coordinates, wingspan * np.ones(len(coordinates))]

    return cords3d


def transform_to_GCS(
    cords3d: np.ndarray, globalpos: np.ndarray, surf_type: SurfaceType
) -> np.ndarray:
    """
    Transforms the curve from its local reference frame to the global coordinate system GCS

    Parameters
    ----------
    cords3d : np.ndarray
        Curves.
    globalpos : np.ndarray
        DESCRIPTION.
    surf_type : SurfaceType
        DESCRIPTION.

    Returns
    -------
    cords3d : TYPE
        DESCRIPTION.

    """

    if surf_type == SurfaceType.FIN:
        rotmat3d = rotation_matrix3d(-90, axis="x", units="degrees")
        cords3d = [rotmat3d @ r for r in cords3d]

    cords3d = cords3d + globalpos
    return cords3d


def rotation_matrix2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def rotation_matrix3d(theta: float, axis="x", units="radians") -> np.ndarray:

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


def plot_surface(UAV):

    surface = UAV.surfaces[0]

    fig = plt.figure(num=4, clear=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.view_init(vertical_axis="y")

    # (theta, phi) = np.meshgrid(np.linspace(0, 2 * np.pi, 280),
    #                            np.linspace(0, 2 * np.pi, 280))

    # x = (3 + np.cos(phi)) * np.cos(theta)
    # z = (3 + np.cos(phi)) * np.sin(theta)
    # y = np.sin(phi)
    xx = [section.airfoil.data.x.to_numpy() for section in surface.sections]
    yy = [section.airfoil.data.y.to_numpy() for section in surface.sections]
    zz = [section.airfoil.data.z.to_numpy() for section in surface.sections]

    k = 4

    xlim = [-k, k]  # 2k
    ylim = [-k / 2, k / 2]  # k
    zlim = [-k, k]  # 2k

    # box_aspect = (4,2,4) #(x,y,z)
    box_aspect = (4, 4, 2)  # (z,x,y)

    def fun(t, f):
        return (np.cos(f + 2 * t) + 1) / 2

    dplot = ax.plot_surface(xx, yy, zz, facecolors=cm.jet(fun(xx, yy)))
    ax.set(
        xlabel="x",
        ylabel="y",
        zlabel="z",
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        # xticks = [-4, -2, 2, 4],
        # yticks = [-4, -2, 2, 4],
        # zticks = [-1, 0, 1],
        title="Donut!",
    )

    ax.set_box_aspect(aspect=box_aspect)
    plt.get_backend()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    fig.tight_layout()

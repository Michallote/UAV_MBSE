"""Aerodynamics Module"""

from __future__ import annotations

import copy
import os
from functools import cached_property
from typing import Self

import matplotlib.pyplot as plt
import numpy as np

from src.geometry.spatial_array import SpatialArray
from src.utils.interpolation import find_max, resample_curve


class Airfoil:
    """Represents airfoil data and properties."""

    def __init__(self, name: str, data: np.ndarray) -> None:
        self.name = name
        self.data = data
        self.segment_airfoil()

    @staticmethod
    def from_file(file_path: str) -> Airfoil:
        """Creates Airfoil object from a file

        Parameters
        ----------
        file_path : str
            File path of the .dat file

        Returns
        -------
        Airfoil
            Instance of Airfoil created from file
        """
        # Read the first line for the airfoil name
        with open(file_path, "r", encoding="utf-8") as file:
            name = file.readline().strip()

        # Now read the rest of the file for the data
        data = np.loadtxt(file_path, comments="#", skiprows=1)

        return Airfoil(name=name, data=data)

    @property
    def x(self) -> np.ndarray:
        """Returns the numpy array of x-values of the data coordinates"""
        return self.data[:, 0]

    @property
    def y(self) -> np.ndarray:
        """Returns the numpy array of y-values of the data coordinates"""
        return self.data[:, 1]

    @property
    def z(self) -> np.ndarray:
        """Returns the numpy array of z-values of the data coordinates"""
        return self.data[:, 2]

    @cached_property
    def area(self) -> float:
        """Calculates the area of a closed contour using greens theorem
        https://leancrew.com/all-this/2018/01/greens-theorem-and-section-properties/
        """
        xi = self.x
        xf = np.roll(xi, -1)
        yi = self.y
        yf = np.roll(yi, -1)
        return np.sum((xf + xi) * (-yi + yf) / 2)  # type: ignore
        # return -np.trapz(y = self.y, x = self.x) #Numerical integration

    @cached_property
    def center(self) -> SpatialArray:
        """Aerodynamic Center of the airfoil"""
        return SpatialArray(
            [
                0.25,
                0.5
                * (
                    np.interp(0.25, self.extrados.x, self.extrados.y)
                    + np.interp(0.25, self.intrados.x, self.intrados.y)
                ),
            ]
        )

    def segment_airfoil(self) -> None:
        """
        Segments the airfoil data into leading edge, trailing edge, intrados, and extrados.
        """
        data = self.data
        # Sorting 'data' by 'x' ascending and then by 'y' descending
        # Leading Edge is the furthest west point of the airfoil,
        # if two points are on the same x coordinate, then the upper point is selected
        indices = np.lexsort((-self.y, self.x))
        index_le = indices[0]
        # Select relevant points for later calculations
        self.leading_edge = SpatialArray(data[index_le])
        self.trailing_edge = SpatialArray(data[0])
        self.trailing_edge2 = SpatialArray(data[-1])

        self.intrados = SpatialArray(data[index_le + 1 :])
        # Reverse the extrados so x is in ascending order
        self.extrados = SpatialArray(data[index_le::-1])
        self.index_le = index_le
        self.index_te = 0
        self.index_te2 = -1

    @cached_property
    def centroid(self) -> SpatialArray:
        """Calculates the centroid of the airfoil"""
        x = self.x
        y = self.y

        def xc(x, y):
            """Calculates the xc centroid of a closed contour using greens theorem
            https://leancrew.com/all-this/2018/01/greens-theorem-and-section-properties/
            """
            xi = x
            xf = np.roll(x, -1)
            yi = y
            yf = np.roll(y, -1)
            return (1 / 6) * (xf + xi) * (xi * yf - xf * yi)

        def yc(x, y):
            """Calculates the yc centroid of a closed contour using greens theorem
            https://leancrew.com/all-this/2018/01/greens-theorem-and-section-properties/
            """
            xi = x
            xf = np.roll(x, -1)
            yi = y
            yf = np.roll(y, -1)
            return (1 / 6) * (yf + yi) * (xi * yf - xf * yi)

        area = self.area
        x_centroid = np.sum(xc(x, y)) / area
        y_centroid = np.sum(yc(x, y)) / area

        return SpatialArray([x_centroid, y_centroid])

    def thickness(self, t):
        """
        Returns the camber (camber/chord) value with %c (0->1) as input
        -------

        """
        return np.interp(t, self.extrados.x, self.extrados.y) - np.interp(
            t, self.intrados.x, self.intrados.y
        )

    def camber(self, t):
        """
        Returns the thickness (t/c) value with %c (0->1) as input
        """
        return 0.5 * (
            np.interp(t, self.extrados.x, self.extrados.y)
            + np.interp(t, self.intrados.x, self.intrados.y)
        )

    def max_thickness(self, n_iter=4) -> float:
        """Returns max thickness length"""
        _, max_thickness = find_max(self.thickness, n_iter=n_iter)
        return max_thickness

    def max_camber(self, n_iter=4) -> float:
        """Returns max camber length"""
        _, max_camber = find_max(self.camber, n_iter=n_iter)
        return max_camber

    def max_thickness_position(self, n_iter=4) -> float:
        """Returns x coordinate of max thickness"""
        x_thickness, _ = find_max(self.thickness, n_iter=n_iter)
        return x_thickness

    def max_camber_position(self, n_iter=4) -> float:
        """Returns x coordinate of max camber"""
        x_camber, _ = find_max(self.camber, n_iter=n_iter)
        return x_camber

    def resample(self, n_samples: int) -> Airfoil:
        """Resamples the airfoil and returns another Airfoil object with
        the new resampled coordinates."""
        coordinates = resample_curve(self.data, n_samples)
        return Airfoil(name=self.name, data=coordinates)

    def set_z(self, z: float):
        """Sets the z position on the array

        Parameters
        ----------
        z : float
            z-coordinate
        """
        import numpy as np

        data = self.data
        # Create a column of zeros
        zeros_column = np.zeros((data.shape[0], 1))

        # Horizontally stack the original array with the zeros column
        self.data = np.hstack((data, zeros_column))

    def plot_airfoil(self):
        """Plot the airfoil"""
        # plt.scatter(self.data.x, self.data.y,marker='o',edgecolors='black',s=3)
        plt.plot(
            self.x,
            self.y,
            "r",
            marker=".",
            markeredgecolor="black",
            markersize=3,
        )
        plt.axis("equal")
        plt.xlim((-0.05, 1.05))
        plt.legend([self.name])


def slice_shift(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the x[i] and x[i+1] arrays for numerical calculations xi, xf
    """
    return x[:-1], x[1:]


class AirfoilFactory:
    """A singleton class to hold configuration data.
    Factory class to create Airfoil instances."""

    _instance = None
    folder_path = ""
    _cache = {}

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super(AirfoilFactory, cls).__new__(cls)
            cls._instance.folder_path = None
        return cls._instance

    def set_folder_path(self, path: str) -> None:
        """Set the folder path."""
        self.folder_path = path

    def get_folder_path(self) -> str:
        """Get the folder path."""
        return self.folder_path

    def create_airfoil(self, foilname: str) -> Airfoil:
        """Create an Airfoil instance from a file in the configured folder."""

        if foilname in self._cache:
            return copy.deepcopy(self._cache[foilname])
        else:
            filename = f"{foilname}.dat"

        folder_path = self.get_folder_path()
        full_path = os.path.join(folder_path, filename)
        return Airfoil.from_file(full_path)

    def cache_airfoils(self) -> None:
        """Precompute Airfoils and store them to a dictionary."""
        # Iterate over all files in the 'data' directory
        for root, _, files in os.walk(self.folder_path):
            for filename in files:
                if filename.endswith(".dat"):
                    # Construct the full path to the file
                    full_path = os.path.join(root, filename)

                    # Execute foo on the .dat file
                    airfoil = Airfoil.from_file(full_path)
                    self._cache[airfoil.name] = airfoil


def main():
    """Main function"""
    airfoil = Airfoil.from_file("data/airfoils/FX 73-CL2-152.dat")
    airfoil_factory = AirfoilFactory()
    airfoil_factory.set_folder_path("data/airfoils")
    airfoil_factory.cache_airfoils()
    print(airfoil)


if __name__ == "__main__":
    main()

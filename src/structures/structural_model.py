"""Structural Module for Aircraft Analysis"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from src.geometry.geometry_processing import (
    AircraftGeometry,
    GeometricCurve,
    GeometricSurface,
)
from src.geometry.spatial_array import SpatialArray
from src.utils.interpolation import vector_interpolation


@dataclass
class Material:
    """Represents a class with material properties"""

    name: str
    density: float  # kg/m**3


class StructuralRib:
    """
    Represents a structural rib of the aircraft, defined by a curve.
    """

    curve: GeometricCurve
    thickness: float
    material: Material

    def __init__(
        self, curve: GeometricCurve, thickness: float, material: Material
    ) -> None:
        self.curve = curve
        self.thickness = thickness
        self.material = material

    @property
    def area(self) -> float:
        """Calculate the area of the rib."""
        return self.curve.area

    @property
    def centroid(self) -> SpatialArray:
        """Calculate the centroid of the rib."""
        data = self.curve.data
        n = len(data) - 1
        indices = np.array(
            [result for i in range(n) if all_different(result := [i, i + 1, n - i])]
        )
        # i, j, k = indices.T
        triangles = data[indices]

        centroids = np.sum(triangles, axis=1) / 3
        areas = np.vstack([triangle_area(*triangle) for triangle in triangles])

        centroid = np.sum(centroids * areas, axis=0) / np.sum(areas)

        return centroid


class StructuralSpar:
    """
    Represents a wing spar, defined by its cross-sectional area and centroid.
    """

    area: float
    centroid: SpatialArray

    def __init__(self, area: float, centroid: SpatialArray) -> None:
        self.area = area
        self.centroid = centroid


class SurfaceCoating:
    """Represents the surface coating (Monokote).
    Uses SI units only."""

    surface: GeometricSurface
    thickness: float
    density: float

    def __init__(self, surface: GeometricSurface) -> None:
        self.surface = surface

    @cached_property
    def _centroid_area(self) -> tuple[SpatialArray, float]:
        """Computes the centroid and area of the surface
          using a vectorized triangulation approach.

        Returns
        -------
        tuple[SpatialArray, float], units: m, m**2
            centroid, area
        """

        xx = self.surface.xx
        yy = self.surface.yy
        zz = self.surface.zz

        centroid, area = calculate_centroid_of_surface(xx, yy, zz)
        return SpatialArray(centroid), area

    @property
    def centroid(self) -> SpatialArray:
        """Returns the centroid of the surface

        Returns
        -------
        SpatialArray
            centroid
        """
        centroid, _ = self._centroid_area
        return centroid

    @property
    def area(self) -> float:
        """Returns the total area of the surface
        Returns: float, area
        """
        _, area = self._centroid_area
        return area

    @property
    def mass(self) -> float:
        """Returns the computed mass of the coating.
        Returns: float, units: kg
        """
        return self.area * self.thickness * self.density


class StructuralModel:
    """
    Structural model of the aircraft.
    """

    aircraft: AircraftGeometry
    max_rib_spacing: float  # in metres
    ribs: list[StructuralRib]
    spars: list[StructuralSpar]

    def __init__(self, aircraft: AircraftGeometry, max_rib_spacing: float) -> None:
        self.aircraft = aircraft
        self.max_rib_spacing = max_rib_spacing
        self.ribs = []
        self.spars = []

    def calculate_ribs(self):
        """Calculate ribs based on the max rib spacing and the aircraft geometry."""
        # Loop through each AeroSurface in the aircraft
        for surface in self.aircraft.surfaces:
            ribs = self._calculate_ribs_for_surface(surface)
            self.ribs.extend(ribs)

    def _calculate_ribs_for_surface(
        self, surface: GeometricSurface
    ) -> list[StructuralRib]:
        """Calculate ribs for a single geometric surface."""
        ribs = []
        # Implementation of rib calculation and interpolation goes here
        wingspans = surface.wingspans
        # Number of ribs between sections
        n_ribs = np.ceil((wingspans[1:] - wingspans[:-1]) / self.max_rib_spacing)

        section_ribs = [
            np.linspace(wingspans[i], wingspans[i + 1], int(n + 1))
            for i, n in enumerate(n_ribs)
        ]

        rib_positions = np.unique(np.concatenate(section_ribs))

        section_curves = np.array([curve.data for curve in surface.curves])

        ribs_geometry = vector_interpolation(rib_positions, wingspans, section_curves)
        balsa = Material(name="Balsa", density=139)

        for i, curve in enumerate(ribs_geometry):
            name = f"{surface.surface_type.name[0]}_rib_{i}"
            rib = StructuralRib(
                curve=GeometricCurve(name=name, data=curve),
                thickness=3 / 16 * 2.54 / 100,
                material=balsa,
            )
            ribs.append(rib)

        return ribs

    def calculate_spars(self):
        """Calculate the position and characteristics of wing spars."""
        # Loop through each AeroSurface in the aircraft
        for surface in self.aircraft.surfaces:
            spar = self._calculate_spar_for_surface(surface)
            self.spars.append(spar)

    def _calculate_spar_for_surface(self, surface: GeometricSurface) -> StructuralSpar:
        """Calculate spar for a single geometric surface."""
        # Implementation of spar calculation goes here
        # ...
        return StructuralSpar(area=0, centroid=SpatialArray([0, 0, 0]))  # Placeholder


def all_different(lst):
    """Check if all elements in the list are different."""
    return len(set(lst)) == len(lst)


def triangle_area(v1, v2, v3):
    """Calculate the area of a triangle in 3D."""
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))  # type: ignore


def calculate_centroid_of_surface(xx, yy, zz) -> tuple[np.ndarray, float]:
    """Calculates the centroid of a surface using triangulation methods

    Parameters
    ----------
    xx : np.ndarray
        x-matrix of surface coordinates
    yy : np.ndarray
        y-matrix of surface coordinates
    zz : np.ndarray
        z-matrix of surface coordinates
    """
    # Calculate the area of bottom left half surface
    centroid_l, area_l = _calculate_centroid_of_half_surface(xx, yy, zz)
    # Calculate the area of top right half surface
    centroid_r, area_r = _calculate_centroid_of_half_surface(
        np.flip(xx), np.flip(yy), np.flip(zz)
    )
    area = area_l + area_r

    centroid = (centroid_l * area_l + centroid_r * area_r) / area

    return centroid, area


def _calculate_centroid_of_half_surface(xx, yy, zz) -> tuple[np.ndarray, float]:
    """Calculates the centroid of a surface using triangulation methods

    Parameters
    ----------
    xx : np.ndarray
        x-matrix of surface coordinates
    yy : np.ndarray
        y-matrix of surface coordinates
    zz : np.ndarray
        z-matrix of surface coordinates
    """

    coordinates = np.array([xx, yy, zz])

    triangles_coordinates = np.array(
        [
            coordinates[:, :-1, :-1],  # bottom left vertices -> 0
            coordinates[:, :-1, 1:],  # bottom right vertices -> 1
            coordinates[:, 1:, :-1],  # upper left vertices -> 2
        ]
    )
    # to acces the triangle associated with the first vertex
    # use: triangle_coordinates[:, 0, 0]

    area_matrix = _calculate_area_of_half_surface(xx, yy, zz)

    centroids = np.mean(triangles_coordinates, axis=0)

    centroid = np.sum(np.sum(area_matrix * centroids, axis=1), axis=1) / np.sum(
        area_matrix
    )

    return centroid, np.sum(area_matrix)


def _calculate_area_of_half_surface(
    xx: np.ndarray, yy: np.ndarray, zz: np.ndarray
) -> np.ndarray:
    """Calculates the area of surface triangles (bottom half) and reurns the array of each triangle formed by
    triangle[0,0] -> (vertex[0,0], vertex[1,0], vertex[0,1])
    area[0,0] -> area(triangle[0,0])



    Parameters
    ----------
    xx : np.ndarray
        _description_
    yy : np.ndarray
        _description_
    zz : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        Area matrix of the input vertices dimensions are (n-1), (m-1) of input shape.
    """

    # Calculate the vectors for two sides of each small triangle
    # This calculates only half of the surface
    vec_v = np.array(
        [np.diff(xx, axis=0), np.diff(yy, axis=0), np.diff(zz, axis=0)]
    ).transpose(1, 2, 0)
    vec_u = np.array(
        [np.diff(xx, axis=1), np.diff(yy, axis=1), np.diff(zz, axis=1)]
    ).transpose(1, 2, 0)

    # Calculate the cross product, which gives a vector perpendicular to the triangle's surface
    # The magnitude of this vector is twice the area of the triangle
    cross_product = np.cross(vec_u[:-1, :], vec_v[:, :-1], axis=2)
    areas = np.linalg.norm(cross_product, axis=2)

    return areas / 2


if __name__ == "__main__":
    pass

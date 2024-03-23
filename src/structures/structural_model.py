"""Structural Module for Aircraft Analysis"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from src.geometry.aircraft_geometry import (
    AircraftGeometry,
    GeometricCurve,
    GeometricSurface,
)
from src.geometry.spatial_array import SpatialArray
from src.geometry.surfaces import surface_centroid_area
from src.structures.spar import StructuralSpar
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
        return self.curve.centroid


class SurfaceCoating:
    """Represents the surface coating (Monokote).
    Uses SI units only."""

    surface: GeometricSurface
    thickness: float
    material: Material

    def __init__(
        self, surface: GeometricSurface, thickness: float, material: Material
    ) -> None:
        self.surface = surface
        self.thickness = thickness
        self.material = material

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

        centroid, area = surface_centroid_area(xx, yy, zz)
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
        return self.area * self.thickness * self.material.density


class StructuralModel:
    """
    Structural model of the aircraft.
    """

    aircraft: AircraftGeometry
    max_rib_spacing: float  # in metres
    ribs: list[StructuralRib]
    spars: list[StructuralSpar]

    def __init__(
        self,
        aircraft: AircraftGeometry,
        max_rib_spacing: float,
        rib_thickness: float,
    ) -> None:
        self.aircraft = aircraft
        self.max_rib_spacing = max_rib_spacing
        self.ribs = []
        self.spars = []

        self.calculate_ribs()
        self.calculate_spars()

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

        # Ensures the original positions of the sections are maintained.
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
            spar = StructuralSpar.from_surface_and_plane(
                surface, p=np.array([0.43, 0, 0]), n=np.array([1, 0, 0])
            )
            self.spars.append(spar)

    def calculate_main_spar(self, surface):

        p = np.array([0.43, 0, 0])
        n = np.array([1, 0, 0])

        StructuralSpar.from_surface_and_plane(surface, p=p, n=n)

    def surface_coating(self):

        for surface in self.aircraft.surfaces:
            coating = SurfaceCoating(surface)


if __name__ == "__main__":
    pass

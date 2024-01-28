"""Structural Module for Aircraft Analysis"""
from dataclasses import dataclass

import numpy as np

from src.aerodynamics.data_structures import AeroSurface, Aircraft
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
        return self.curve.centroid


class StructuralSpar:
    """
    Represents a wing spar, defined by its cross-sectional area and centroid.
    """

    area: float
    centroid: SpatialArray

    def __init__(self, area: float, centroid: SpatialArray) -> None:
        self.area = area
        self.centroid = centroid


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


if __name__ == "__main__":
    # Example usage
    aircraft = Aircraft(...)  # Initialize Aircraft object
    structural_model = StructuralModel(aircraft, max_rib_spacing=0.15)
    structural_model.calculate_ribs()
    # structural_model.calculate_spars()

    # Access ribs and spars
    for rib in structural_model.ribs:
        print(f"Rib Area: {rib.area}, Centroid: {rib.centroid}")

    for spar in structural_model.spars:
        print(f"Spar Area: {spar.area}, Centroid: {spar.centroid}")

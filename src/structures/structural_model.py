"""Structural Module for Aircraft Analysis"""

from __future__ import annotations

from functools import cached_property
from itertools import chain
from typing import Any, Generator, Iterable, Iterator, Literal, Union, overload

import numpy as np

from src.aerodynamics.data_structures import PointMass, SurfaceType
from src.geometry.aircraft_geometry import (
    AircraftGeometry,
    GeometricCurve,
    GeometricSurface,
)
from src.geometry.spatial_array import SpatialArray
from src.geometry.surfaces import create_surface_mesh, surface_centroid_area
from src.materials import Material
from src.structures.spar import StructuralSpar
from src.utils.interpolation import vector_interpolation


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

    @property
    def mass(self) -> PointMass:
        """Returns the computed mass of the rib.
        Returns: float, units: kg
        """
        return PointMass(
            self.area * self.thickness * self.material.density,
            coordinates=self.centroid,
            tag="Rib",
        )

    @property
    def mesh(self) -> tuple[np.ndarray, ...]:
        """Returns the coordinates neccessary to plot the object as a mesh object

        - x, y, z are the coordinates of points in the mesh
        - i, j, k are the indices of points that comprise each indiviual triangle
        """

        curve = self.curve
        x, y, z = curve.x, curve.y, curve.z
        indices = curve.triangulation_indices()
        i, j, k = indices.T

        return x, y, z, i, j, k


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
    def mass(self) -> PointMass:
        """Returns the computed mass of the coating.
        Returns: float, units: kg
        """
        return PointMass(
            self.area * self.thickness * self.material.density,
            coordinates=self.centroid,
            tag="SurfaceCoating",
        )

    @property
    def mesh(self) -> tuple[np.ndarray, ...]:
        """Returns the 2D mesh of the spar."""
        xx = self.surface.xx
        yy = self.surface.yy
        zz = self.surface.zz
        x, y, z, i, j, k = create_surface_mesh(xx, yy, zz)
        return x, y, z, i, j, k


class SurfaceStructure:
    """Mirrors a surface on the aircraft, holding structural elements."""

    def __init__(self, surface: GeometricSurface, config: dict):
        self.surface = surface
        self.spars = []
        self.ribs = []
        self.coatings = []
        self.config = config

    def initialize_structure(self):
        """Initializes the structure based on the configuration dictionary."""

        config = self.config
        # Initialize the main spar
        main_spar_config = config.get("main_spar")
        if main_spar_config:
            self.add_spar(
                StructuralSpar.create_main_spar(self.surface, **main_spar_config)
            )

        # Initialize the secondary spar
        secondary_spar_config = config.get("secondary_spar")
        if secondary_spar_config:
            self.add_spar(
                StructuralSpar.create_spar(self.surface, **secondary_spar_config)
            )

        # Calculate and add ribs
        rib_config = config.get("ribs")
        if rib_config:
            calculated_ribs = self.calculate_ribs(self.surface, **rib_config)
            for rib in calculated_ribs:
                self.add_rib(rib)

        # Initialize the surface coating
        coating_config = config.get("surface_coating")
        if coating_config:
            self.add_coating(SurfaceCoating(self.surface, **coating_config))

    def add_spar(self, spar: StructuralSpar):
        self.spars.append(spar)

    def add_rib(self, rib: StructuralRib):
        self.ribs.append(rib)

    def add_coating(self, coating: SurfaceCoating):
        self.coatings.append(coating)

    @staticmethod
    def calculate_ribs(
        surface: GeometricSurface,
        material: Material,
        max_spacing: float,
        thickness: float = 3 / 16 * 2.54 / 100,
    ) -> list[StructuralRib]:
        """Calculate ribs for a single geometric surface."""
        ribs = []
        # Implementation of rib calculation and interpolation goes here
        wingspans = surface.wingspans
        # Number of ribs between sections
        n_ribs = np.ceil((wingspans[1:] - wingspans[:-1]) / max_spacing)

        # Ensures the original positions of the sections are maintained.
        section_ribs = [
            np.linspace(wingspans[i], wingspans[i + 1], int(n + 1))
            for i, n in enumerate(n_ribs)
        ]

        rib_positions = np.unique(np.concatenate(section_ribs))

        section_curves = np.array([curve.data for curve in surface.curves])

        ribs_geometry = vector_interpolation(rib_positions, wingspans, section_curves)

        for i, curve in enumerate(ribs_geometry):
            name = f"{surface.surface_type.name[0]}_rib_{i}"
            rib = StructuralRib(
                curve=GeometricCurve(name=name, data=curve),
                thickness=thickness,
                material=material,
            )
            ribs.append(rib)

        return ribs

    def collect_masses(self) -> Generator[PointMass, None, None]:
        """Returns a generator of component point masses

        Returns
        -------
        Generator[PointMass, None, None]
            Point Masses generator.
        """
        return (item.mass for item in chain(self.spars, self.ribs, self.coatings))

    def weight_summary(self) -> PointMass:
        """Prints a Summary of the weight of components

        Returns
        -------
        PointMass
            _description_
        """

        def collect(xs):
            return (x.mass for x in xs)

        print(compute_mass_center(collect(self.spars), tag="Spars"))
        print(compute_mass_center(collect(self.ribs), tag="Ribs"))
        print(compute_mass_center(collect(self.coatings), tag="Coatings"))
        print(
            compute_mass_center(self.collect_masses(), tag="Total " + self.surface.name)
        )
        return compute_mass_center(self.collect_masses(), tag=self.surface.name)

    def summary_data(self) -> list[dict]:
        """Generates summary weight data of properties:
            'mass' 'x' 'y' 'z' 'tag' 'material' 'structure'
        Returns
        -------
        list[dict]
            A list of component weight data dictionaries.
        """
        surface_type = self.surface.surface_type.name
        properties = collect_properties(self.spars, self.ribs, self.coatings)

        # Applying the function using dictionary comprehension
        for values in properties:
            values["surface"] = surface_type

        return properties

    @property
    def mass(self) -> PointMass:
        """_summary_

        Returns
        -------
        PointMass
            _description_
        """
        return compute_mass_center(self.collect_masses(), tag=self.surface.name)

    def components(
        self,
    ) -> Generator[StructuralRib | StructuralSpar | SurfaceCoating, None, None]:
        """Returns a generator of component point masses

        Returns
        -------
        Generator[PointMass, None, None]
            Point Masses generator.
        """
        return (item for item in chain(self.spars, self.ribs, self.coatings))

    @property
    def surface_type(self) -> SurfaceType:
        return self.surface.surface_type


class StructuralModel:
    """
    Structural model of the aircraft.
    """

    aircraft: AircraftGeometry
    structures: list[SurfaceStructure]

    def __init__(
        self,
        aircraft: AircraftGeometry,
        configuration: dict,
    ) -> None:
        self.aircraft = aircraft
        self.configuration = configuration
        self.structures = []
        self.ext_spars = []

        for surface in aircraft.surfaces:
            surface_type = surface.surface_type
            struct = SurfaceStructure(surface, configuration[surface_type])
            struct.initialize_structure()
            self.structures.append(struct)

    def summary_data(self) -> list[Any]:
        """Gather the weight summary

        Returns
        -------
        list[Any]
            _description_
        """

        properties = []

        for structure in self.structures:
            properties.extend(structure.summary_data())

        if self.ext_spars:
            ext_properties = collect_properties(self.ext_spars, [], [])

            for values in ext_properties:
                values["surface"] = "EXTERNAL"

            properties.extend(ext_properties)

        return properties

    @overload
    def components(
        self, yield_structure: Literal[False]
    ) -> Iterator[Union[StructuralRib, StructuralSpar, SurfaceCoating]]: ...

    @overload
    def components(self, yield_structure: Literal[True]) -> Iterator[
        tuple[
            SurfaceStructure | None,
            Union[StructuralRib, StructuralSpar, SurfaceCoating],
        ]
    ]: ...

    def components(self, yield_structure=True) -> Iterator:
        """Yields each structure along with its component.

        Yields
        -------
        Iterator[Tuple[SurfaceStructure, Union[StructuralRib, StructuralSpar, SurfaceCoating]]]
            Tuples of (structure, component_structure).
        """
        # Assuming 'self.structures' is iterable and contains structures that have a 'components' method.
        # Also assuming 'self.ext_spars' contains additional spars or similar components to yield.

        if not yield_structure:
            components = chain(
                *(structure.components() for structure in self.structures),
                self.ext_spars,
            )

            for component in components:
                yield component
        else:

            for structure in self.structures:
                for component in structure.components():
                    yield (structure, component)
            for ext_spar in self.ext_spars:
                yield (None, ext_spar)

    @property
    def mass(self) -> PointMass:

        masses = [wing.mass for wing in self.components(yield_structure=False)]
        return compute_mass_center(masses, tag=self.aircraft.name)


def compute_mass_center(point_masses: Iterable[PointMass], tag="") -> PointMass:
    """Performs center of mass operations

    Parameters
    ----------
    point_masses : list[PointMass]
        List of point masses
    tag : str, optional
        resulting tag, by default ""

    Returns
    -------
    PointMass
        Center of Mass of the point masses.
    """

    point_masses = tuple(point_masses)

    weighted_coordinates = np.array(
        [point_mass.mass * point_mass.coordinates for point_mass in point_masses]
    )

    weighted_coordinates = np.sum(weighted_coordinates, axis=0)

    total_mass = sum(point_mass.mass for point_mass in point_masses)

    coordinates = weighted_coordinates / total_mass

    return PointMass(total_mass, SpatialArray(coordinates), tag=tag)


def collect_properties(
    spars: list[StructuralSpar],
    ribs: list[StructuralRib],
    coatings: list[SurfaceCoating],
) -> list[dict]:
    """Creates a list of dictionaries with the weight properties of components

    Parameters
    ----------
    spars : list[StructuralSpar]

    ribs : list[StructuralRib]

    coatings : list[SurfaceCoating]


    Returns
    -------
    list[dict]
        list of component weight data dictionaries
    """
    component_properties = []

    # Collect properties for spars
    for index, spar in enumerate(spars, start=0):
        properties = get_mass_properties(spar.mass)
        properties["comp_id"] = index
        properties["material"] = spar.material.name
        properties["structure"] = spar.__class__.__name__
        component_properties.append(properties)

    # Collect properties for ribs
    for index, rib in enumerate(ribs, start=0):
        properties = get_mass_properties(rib.mass)
        properties["comp_id"] = index
        properties["material"] = rib.material.name
        properties["structure"] = rib.__class__.__name__
        component_properties.append(properties)

    # Collect properties for coatings
    for index, coating in enumerate(coatings, start=0):
        properties = get_mass_properties(coating.mass)
        properties["comp_id"] = index
        properties["material"] = coating.material.name
        properties["structure"] = coating.__class__.__name__
        component_properties.append(properties)

    return component_properties


# Helper function to create property dictionary for each item
def get_mass_properties(point_mass: PointMass) -> dict[str, Any]:
    """Helper function to create property dictionary for each item

    Parameters
    ----------
    point_mass : PointMass
        point mass object

    Returns
    -------
    dict[str, Any]
        properties dictionary
    """
    return {
        "mass": point_mass.mass,
        "x": point_mass.coordinates.x,
        "y": point_mass.coordinates.y,
        "z": point_mass.coordinates.z,
        "tag": point_mass.tag,
    }


if __name__ == "__main__":
    pass

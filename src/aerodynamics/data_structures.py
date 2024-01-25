"""Module providing classes with object relational mapping 
for XFLR5 planes"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum, auto
from typing import Iterator

import numpy as np
import pandas as pd

from src.aerodynamics.airfoil import Airfoil, AirfoilFactory
from src.geometry.spatial_array import SpatialArray
from src.utils.xml_parser import parse_xml_file


class SurfaceType(Enum):
    """Aerodynamic Surfaces types"""

    MAINWING = auto()
    SECONDWING = auto()
    ELEVATOR = auto()
    FIN = auto()

    # @classmethod
    def __repr__(self) -> str:
        return f"{self.name} ({self.value})"


class Aircraft:
    """Represents an Aircraft with Aerodynamic Surfaces and Properties."""

    name: str
    surfaces: list[AeroSurface]
    has_body: bool
    inertia: list[PointMass] | None
    description: str
    units: dict

    def __init__(
        self,
        name: str,
        surfaces: list[AeroSurface],
        has_body: bool,
        inertia: list[PointMass] | None,
        description: str,
        units: dict,
    ) -> None:
        self.name = name
        self.surfaces = surfaces
        self.has_body = has_body
        self.inertia = inertia
        self.description = description
        self.units = units

    def add_surface(self, surface: AeroSurface) -> None:
        """Add a lifting surface to the list of surfaces."""
        self.surfaces.append(surface)

    def find_surfaces(self, surf_type: SurfaceType) -> list[AeroSurface]:
        """Find all lifting surfaces with a particular type in the surfaces list"""
        return [surface for surface in self.surfaces if surface.surf_type is surf_type]

    def print_parameters(self):
        """Print to console all data frames"""
        for surface in self.surfaces:
            description = surface.df[
                [
                    "Wingspan",
                    "Chord",
                    "Twist",
                    "xOffset",
                    "yOffset",
                    "Dihedral",
                    "FoilName",
                ]
            ].to_string()
            print(f"\n Name: {surface.name}\n{description}")

    def ribs_for_analysis(self):
        """Generates a list with ribs locations for each surface"""
        for surface in self.surfaces:
            list(surface.df["Wingspan"])

    @property
    def df(self) -> pd.DataFrame:
        """Return a dataframe of the aircraft sections"""
        df_list = []
        for surface in self.surfaces:
            df = surface.df.copy()
            df["surface_type"] = surface.surf_type
            df_list.append(df)

        return pd.concat(df_list, ignore_index=True)

    @staticmethod
    def from_dict(data: dict) -> Aircraft:
        """Creates an Aircraft instance from a dictionary"""
        return parse_plane(data)

    @staticmethod
    def from_xml(path: str):
        """Creates an Aircraft instance from an XML file"""
        plane_data = parse_xml_file(path)
        return Aircraft.from_dict(plane_data)

    def __iter__(self) -> Iterator[AeroSurface]:
        for surface in self.surfaces:
            yield surface


class AeroSurface:
    """Basic representation of an aerodynamic surface on the aircraft."""

    name: str
    position: SpatialArray
    color: tuple | None
    surf_type: SurfaceType
    tilt: float
    symmetric: bool
    is_fin: bool
    is_double_fin: bool
    is_sym_fin: bool
    inertia: dict
    sections: list[Section]

    def __init__(
        self,
        name: str,
        position: SpatialArray,
        color: tuple | None,
        surf_type: SurfaceType,
        tilt: float,
        symmetric: bool,
        is_fin: bool,
        is_double_fin: bool,
        is_sym_fin: bool,
        inertia: dict,
        sections: list[Section],
    ) -> None:
        self.name = name
        self.position = position
        self.color = color
        self.surf_type = surf_type
        self.tilt = tilt
        self.symmetric = symmetric
        self.is_fin = is_fin
        self.is_double_fin = is_double_fin
        self.is_sym_fin = is_sym_fin
        self.inertia = inertia
        self.sections = sections
        # Initialize the y_offsets of the sections
        self.calc_y_offset()

    def add_section(self, section: Section) -> None:
        """Add a section to the list of sections."""
        self.sections.append(section)

    def set_color(self, color: tuple) -> None:  # Specify the type of color if possible
        """Set surface color

        Parameters
        ----------
        color : tuple
            (R,G,B) tuple 0 -> 255
        """
        self.color = color

    @property
    def df(self) -> pd.DataFrame:
        """Transform the sections stored into a dataframe for visualization"""
        sections = self.sections
        df_dict = [
            {key: value for key, value in asdict(section).items() if key != "airfoil"}
            for section in sections
        ]
        return pd.DataFrame(df_dict)

    def calc_y_offset(self) -> None:
        """Calculates the y_offset component for each section given."""
        var_sections = self.sections
        delta_span = [
            var_sections[i].distance_to(var_sections[i + 1])
            for i in range(len(var_sections) - 1)
        ]
        dihedral = np.sin(np.radians([section.dihedral for section in self.sections]))[
            :-1
        ]
        y_offsets = np.insert(np.cumsum(delta_span * dihedral), 0, 0.0)

        for section, y_offset in zip(self.sections, y_offsets):
            section.y_offset = y_offset

    def get_ribs_position(self, rib_spacing: float = 0.15) -> np.ndarray:
        """Get rib spacing"""
        wingspans = np.array([section.wingspan for section in self.sections])
        n_ribs = np.ceil((wingspans[1:] - wingspans[:-1]) / rib_spacing)
        section_ribs = [
            np.linspace(wingspans[i], wingspans[i + 1], int(n + 1))
            for i, n in enumerate(n_ribs)
        ]
        ribs_position = np.unique(np.concatenate(section_ribs))
        return ribs_position

    def __repr__(self) -> str:
        return (
            f"({self.name}, {repr(self.surf_type)}, No. Sections: {len(self.sections)})"
        )


@dataclass
class Section:
    """Represents a wing section with assigned airfoil properties."""

    wingspan: float
    chord: float
    x_offset: float
    y_offset: float
    dihedral: float
    twist: float
    foil_name: str
    airfoil: Airfoil  # Replace 'Any' with the specific type if known
    x_panels: int
    x_panel_dist: str
    y_panels: int
    y_panel_dist: str

    def distance_to(self, other: "Section") -> float:
        """
        Calculate the distance to another Section based on wingspan.

        Args:
        other (Section): Another section to compare with.

        Returns:
        float: The difference in wingspan between this section and the other.
        """
        return abs(self.wingspan - other.wingspan)


@dataclass
class PointMass:
    """Represents a punctual mass for Inertia calculations"""

    mass: float
    coordinates: SpatialArray
    tag: str | None


def create_point_mass(point_mass_data: dict) -> PointMass:
    """Returns a Point Mass for a parsed PM dictionary

    Parameters
    ----------
    point_mass_data : dict
        Dictionary from the parsed plane structure

    Returns
    -------
    PointMass
        Puntual mass in space.
    """
    return PointMass(
        mass=point_mass_data["Mass"],
        coordinates=SpatialArray(point_mass_data["coordinates"]),
        tag=point_mass_data.get("Tag"),
    )


def create_plane_inertia(inertia_data: dict) -> list[PointMass]:
    """Parses the pint masses in the aircraft into a PointMass list

    Parameters
    ----------
    inertia_data : dict
        inertia data from dictionary

    Returns
    -------
    List[PointMass]
        _description_
    """
    point_masses = [create_point_mass(pm) for pm in inertia_data["Point_Mass"]]
    return point_masses


def create_color(color_data: dict) -> tuple:
    """Creates a color tuple from a dict."""
    red = color_data["red"]
    green = color_data["green"]
    blue = color_data["blue"]
    alpha = color_data["alpha"]

    return (red, green, blue, alpha)


def create_section(section_data: dict) -> Section:
    """Parses a section dictionary data into an Section instance"""
    foil_name = section_data["Right_Side_FoilName"]
    airfoil = AirfoilFactory().create_airfoil(foil_name)

    return Section(
        wingspan=section_data["y_position"],
        chord=section_data["Chord"],
        x_offset=section_data["xOffset"],
        y_offset=0,
        dihedral=section_data["Dihedral"],
        twist=section_data["Twist"],
        x_panels=section_data["x_number_of_panels"],
        x_panel_dist=section_data["x_panel_distribution"],
        y_panels=section_data["y_number_of_panels"],
        y_panel_dist=section_data["y_panel_distribution"],
        foil_name=foil_name,
        airfoil=airfoil,
    )


def create_wing(wing_data: dict) -> AeroSurface:
    """Parses an input wing definition dictionary to an AeroSurface instance"""
    sections = [create_section(sec) for sec in wing_data["Sections"]["Section"]]
    color = create_color(wing_data["Color"]) if "Color" in wing_data else None
    inertia = wing_data["Inertia"]
    surface = AeroSurface(
        name=wing_data["Name"],
        position=SpatialArray(wing_data["Position"]),
        color=color,
        surf_type=SurfaceType[wing_data["Type"]],
        tilt=wing_data["Tilt_angle"],
        symmetric=wing_data["Symetric"],
        is_fin=wing_data["isFin"],
        is_double_fin=wing_data["isDoubleFin"],
        is_sym_fin=wing_data["isSymFin"],
        inertia=inertia,
        sections=sections,
    )

    return surface


def parse_plane(data: dict) -> Aircraft:
    """Creates an Aircraft instance from an input dicionary.
    The input dictionary is provided by the XFLR5 xml after parsing"""
    wings = [create_wing(wing) for wing in data["Plane"]["wing"]]

    inertia = (
        create_plane_inertia(data["Plane"]["Inertia"])
        if "Inertia" in data["Plane"]
        else None
    )
    plane = Aircraft(
        name=data["Plane"]["Name"],
        surfaces=wings,
        has_body=data["Plane"]["has_body"],
        inertia=inertia,
        description=data["Plane"].get("Description"),
        units=data["Units"],
    )
    return plane

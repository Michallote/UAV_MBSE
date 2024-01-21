from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List

import numpy as np
import pandas as pd

from src.aerodynamics.airfoil import Airfoil
from src.geometry.spatial_array import SpatialArray
from src.utils.interpolation import resample_curve
from src.utils.xml_parser import parse_xml_file


class SurfaceType(Enum):
    """Aerodynamic Surfaces types"""

    MAINWING = auto()
    SECONDWING = auto()
    ELEVATOR = auto()
    FIN = auto()

    # @classmethod
    def __repr__(self):
        return "{} ({})".format(self.name, self.value)

    def to_str(self):
        return "{}".format(self.name)


class Aircraft:
    """Represents an Aircraft with Aerodynamic Surfaces and Properties."""

    def __init__(self, name, surfaces, has_body, inertia, description, units) -> None:
        self.name: str = name
        self.surfaces: List[AeroSurface] = surfaces
        self.has_body: bool = has_body
        self.inertia: List[PointMass] = inertia
        self.description: str = description
        self.units: dict = units

    def add_surface(self, surface: AeroSurface) -> None:
        """Add a lifting surface to the list of surfaces."""
        self.surfaces.append(surface)

    def find_surfaces(self, surf_type: SurfaceType) -> List[AeroSurface]:
        """Find all lifting surfaces with a particular type in the surfaces list"""
        return [surface for surface in self.surfaces if surface.surf_type is surf_type]

    def print_parameters(self):
        """Print to console all data frames"""
        for surface in self.surfaces:
            print(
                "\n Name: {}\n{}".format(
                    surface.name,
                    surface.df[
                        [
                            "Wingspan",
                            "Chord",
                            "Twist",
                            "xOffset",
                            "yOffset",
                            "Dihedral",
                            "FoilName",
                        ]
                    ].to_string(),
                )
            )

    def ribs_for_analysis(self):
        """Generates a list with ribs locations for each surface"""
        for surface in self.surfaces:
            list(surface.df["Wingspan"])

    def get_aircraft_df(self):
        self.df = pd.concat(
            [surface.df for surface in self.surfaces], ignore_index=True
        )
        return self.df.copy()


class AeroSurface:
    """Basic representation of an aerodynamic surface on the aircraft."""

    def __init__(
        self,
        name: str,
        position: np.ndarray,
        surf_type: SurfaceType,
        tilt: float,
        symmetric: bool,
        is_fin: bool,
        is_doublefin: bool,
        is_symfin: bool,
        sections: List[Section] = None,
    ):
        self.name = name
        self.position = position
        self.surf_type = surf_type
        self.tilt = tilt
        self.symmetric = symmetric
        self.is_fin = is_fin
        self.is_doublefin = is_doublefin
        self.is_symfin = is_symfin
        self.sections = sections if sections is not None else []

    def add_section(self, section: Section) -> None:
        """Add a section to the list of sections."""
        self.sections.append(section)

    def set_color(self, color: tuple) -> None:  # Specify the type of color if possible
        self.color = color

    def add_dataframe(self, df: Any) -> None:  # Specify the type of df if possible
        self.df = df

    def calc_y_offset(self) -> None:
        var_sections = self.sections
        delta_span = [
            var_sections[i].distance_to(var_sections[i + 1])
            for i in range(len(var_sections) - 1)
        ]
        dihedral = np.sin(np.radians([section.Dihedral for section in self.sections]))[
            :-1
        ]
        yOffsets = np.insert(np.cumsum(delta_span * dihedral), 0, 0.0)

        for section, yOffset in zip(self.sections, yOffsets):
            section.yOffset = yOffset

    def get_ribs_position(self, rib_spacing: float = 0.15) -> np.ndarray:
        wingspans = np.array([section.wingspan for section in self.sections])
        n_ribs = np.ceil((wingspans[1:] - wingspans[:-1]) / rib_spacing)
        section_ribs = [
            np.linspace(wingspans[i], wingspans[i + 1], int(n + 1))
            for i, n in enumerate(n_ribs)
        ]
        ribs_position = np.unique(np.concatenate(section_ribs))
        return ribs_position

    def get_ribs_df(self) -> None:
        df = self.df
        ribs_position = self.get_ribs_position()
        wingspans = df["Wingspan"].to_numpy()
        areas = df["Area"].to_numpy()
        ribs_area = np.interp(ribs_position, wingspans, areas)
        centroids = np.array(list(df["Centroid"]))
        ribs_centroid = resample_curve(ribs_position, wingspans, centroids)

    def __repr__(self) -> str:
        return f"({self.name}, {self.surf_type}, No. Sections: {len(self.sections)})"


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
    mass: float
    coordinates: SpatialArray
    tag: str


def create_point_mass(point_mass_data: dict) -> PointMass:
    return PointMass(
        mass=point_mass_data["Mass"],
        coordinates=SpatialArray(point_mass_data["coordinates"]),
        tag=point_mass_data.get("Tag"),
    )


def create_inertia(inertia_data: dict) -> List[PointMass]:
    point_masses = [create_point_mass(pm) for pm in inertia_data["Point_Mass"]]
    return point_masses


def create_color(color_data: dict) -> tuple:
    return tuple(
        red=color_data["red"],
        green=color_data["green"],
        blue=color_data["blue"],
        alpha=color_data["alpha"],
    )


def create_section(section_data: dict) -> Section:
    return Section(
        y_position=section_data["y_position"],
        chord=section_data["Chord"],
        xOffset=section_data["xOffset"],
        dihedral=section_data["Dihedral"],
        twist=section_data["Twist"],
        x_number_of_panels=section_data["x_number_of_panels"],
        x_panel_distribution=section_data["x_panel_distribution"],
        y_number_of_panels=section_data["y_number_of_panels"],
        y_panel_distribution=section_data["y_panel_distribution"],
        left_side_foil_name=section_data["Left_Side_FoilName"],
        right_side_foil_name=section_data["Right_Side_FoilName"],
    )


def create_wing(wing_data: dict) -> AeroSurface:
    sections = [create_section(sec) for sec in wing_data["Sections"]["Section"]]
    color = create_color(wing_data["Color"]) if "Color" in wing_data else None
    inertia = create_inertia(wing_data["Inertia"]) if "Inertia" in wing_data else None
    return AeroSurface(
        name=wing_data["Name"],
        wing_type=wing_data["Type"],
        color=color,
        position=SpatialArray(wing_data["Position"]),
        tilt_angle=wing_data["Tilt_angle"],
        symetric=wing_data["Symetric"],
        is_fin=wing_data["isFin"],
        is_double_fin=wing_data["isDoubleFin"],
        is_sym_fin=wing_data["isSymFin"],
        inertia=inertia,
        sections=sections,
    )


def parse_plane(data: dict) -> Aircraft:
    wings = [create_wing(wing) for wing in data["Plane"]["wing"]]
    inertia = (
        create_inertia(data["Plane"]["Inertia"]) if "Inertia" in data["Plane"] else None
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


def print_keys_and_types(d, indent_level=0):
    """
    Recursively prints the keys and types of items in a nested dictionary or list of dictionaries.

    :param d: The dictionary or list of dictionaries to explore.
    :param indent_level: Current indentation level for pretty printing.
    """
    indent = "    " * indent_level
    if isinstance(d, dict):
        for key, value in d.items():
            print(f"{indent}{key}: {type(value).__name__}")
            if isinstance(value, (dict, list)):
                print_keys_and_types(value, indent_level + 1)
    elif isinstance(d, list):
        for index, item in enumerate(d):
            print(f"{indent}[{index}]: {type(item).__name__} - List Item")
            if isinstance(item, (dict, list)):
                print_keys_and_types(item, indent_level + 1)


print_keys_and_types(parsed_xml)

# Example usage
plane_data = {}  # Replace with your actual data dictionary
plane_object = parse_plane(plane_data)

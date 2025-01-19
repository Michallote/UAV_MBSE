from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Self

from src.utils.xml_parser import parse_xml_file

# Data Model Classes


@dataclass
class Unit:
    """Physical Units representative"""

    name: str
    power: float = 1.0

    def __str__(self) -> str:
        return f"{self.name}^{self.power}" if self.power != 1.0 else self.name


@dataclass
class PhysicalProperty:
    """Represents physical properties of materials"""

    name: str
    units: List[Unit]

    def __str__(self) -> str:
        units_str = "*".join(map(str, self.units))
        return f"{self.name} [{units_str}]"


@dataclass
class MaterialProperty:
    physical_property: PhysicalProperty
    value: Optional[float] = None
    description: Optional[str] = None


class Material:
    """Represents a class with material properties"""

    def __init__(self, name: str, properties: dict[str, list[MaterialProperty]]):
        self.name = name
        self.properties = properties

        properties_map = {}

        for prop in properties.values():

            if not isinstance(prop, list):
                prop = [prop]

            for param in prop:
                name = param.physical_property.name.casefold()
                properties_map[name] = param

        self._properties_map = properties_map

    def __getitem__(self, key) -> MaterialProperty:
        return self._properties_map[key.casefold()]

    def __repr__(self) -> str:
        available_properties = ", ".join(self._properties_map.keys())
        return f"{self.name} ({available_properties})"

    @property
    def density(self) -> float:
        """Returns the material density value."""
        return self["density"].value  # type: ignore


# Utility Functions


def parse_unit(unit_data: dict) -> list[Unit]:
    """
    Parses unit data from the given dictionary format into a human-readable string representation.

    Args:
        unit_data: The unit data in dictionary format, which may be a single dictionary or a list of dictionaries.

    Returns:
        A string representation of the unit, including handling of powers.
    """
    units = unit_data["Unit"]

    if isinstance(units, dict):
        units = [units]  # Convert to list for uniform processing

    return [
        Unit(unit.get("Name"), float(unit.get("@attributes", {}).get("power", 1.0)))
        for unit in units
    ]


def extract_material_properties(materials_xml: dict) -> dict[str, PhysicalProperty]:
    """Extracts material properties from parsed XML data.

    Args:
        materials_xml: The materials data parsed from an XML file.

    Returns:
        A dictionary mapping material names to their physical properties.
    """
    parameter_details = materials_xml["Materials"]["MatML_Doc"]["Metadata"][
        "ParameterDetails"
    ]
    properties = {
        detail["Name"]: PhysicalProperty(
            name=detail["Name"],
            units=parse_unit(detail["Units"]) if "Units" in detail else [Unit("1", 1)],
        )
        for detail in parameter_details
    }
    return properties


def find_material(material_list, material_name: str) -> Dict:
    return next(
        (
            mat["BulkDetails"]
            for mat in material_list
            if mat["BulkDetails"].get("Name") == material_name
        ),
        None,
    )


def parse_properties(
    properties, property_id: Dict[str, str], parameter_id: Dict[str, PhysicalProperty]
):

    if isinstance(properties, list):
        return [
            parsed_property
            for property_data in properties
            if (
                parsed_property := parse_properties(
                    property_data, property_id, parameter_id
                )
            )
            is not None
        ]

    match properties:

        case {
            "@attributes": {"property": pr_id},
            "ParameterValue": parameter_value,
            "Data": parameter_data,
            "Qualifier": qualifier,
        }:
            # logging.debug("case_1")
            # logging.debug(f"{qualifier=}")
            logging.debug(1, property_id[pr_id])
            physical_property = property_id[pr_id]
            params = parse_parameter(parameter_value, parameter_id)
            logging.debug(params, parameter_data, qualifier)
            return (physical_property, params)

        case {
            "@attributes": {"property": pr_id},
            "ParameterValue": parameter_value,
            "Data": parameter_data,
        }:
            # logging.debug("case_2")
            # logging.debug(property_id[pr_id])
            logging.debug(2, property_id[pr_id])
            physical_property = property_id[pr_id]
            params = parse_parameter(parameter_value, parameter_id)
            logging.debug(params, parameter_data)
            return (physical_property, params)

        case _:
            logging.debug(f"No property cases: {properties}")


def parse_parameter(parameter_value, parameter_id: Dict[str, PhysicalProperty]):

    if isinstance(parameter_value, list):
        return [
            value
            for parameter in parameter_value
            if (value := parse_parameter(parameter, parameter_id)) is not None
        ]

    match parameter_value:

        case {
            "@attributes": {"parameter": pa_id, "format": "float"},
            "Data": value,
            "Qualifier": {
                "@attributes": {"name": "Variable Type"},
                "#text": "Dependent",
            },
        }:
            return MaterialProperty(parameter_id[pa_id], value)

        case _:
            logging.debug(f"Unmatched params: {parameter_value}")


# Main Execution Flow
class MaterialLibrary:
    """Class parser for MatML docs of materials"""

    _instance = None
    _materials: Dict[str, Material] = {}

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super(MaterialLibrary, cls).__new__(cls)
        return cls._instance

    @classmethod
    def load_materials(cls, xml_path: str):
        """Loads materials from the XML file and populates the library."""
        materials_xml = parse_xml_file(xml_path)
        properties_map = extract_material_properties(materials_xml)
        property_id = cls._extract_property_id(materials_xml)
        parameter_id = cls._extract_parameter_id(materials_xml, properties_map)

        cls._populate_materials(
            materials_xml["Materials"]["MatML_Doc"]["Material"],
            property_id,
            parameter_id,
        )

        return cls

    @staticmethod
    def _extract_property_id(materials_xml: dict) -> Dict[str, str]:
        """Extracts property ID to name mapping."""
        return {
            pr["@attributes"]["id"]: pr["Name"]
            for pr in materials_xml["Materials"]["MatML_Doc"]["Metadata"][
                "PropertyDetails"
            ]
        }

    @staticmethod
    def _extract_parameter_id(
        materials_xml: dict, properties_map: dict
    ) -> Dict[str, PhysicalProperty]:
        """Extracts parameter ID to PhysicalProperty mapping."""
        return {
            element["@attributes"]["id"]: properties_map.get(
                element["Name"],
                PhysicalProperty(name=element["Name"], units=[Unit("1", 1)]),
            )
            for element in materials_xml["Materials"]["MatML_Doc"]["Metadata"][
                "ParameterDetails"
            ]
        }

    @classmethod
    def _populate_materials(cls, materials_data: List[dict], property_id, parameter_id):
        """Populates the library with Material objects from XML data."""
        for material_data in materials_data:
            name = material_data["BulkDetails"]["Name"]
            properties = material_data["BulkDetails"]["PropertyData"]
            attributes = dict(parse_properties(properties, property_id, parameter_id))  # type: ignore
            cls._materials[name.casefold()] = Material(name, attributes)

    def add_material(self, name, attributes):
        self._materials[name.casefold()] = Material(name, attributes)

    @classmethod
    def get_material(cls, material_name: str):
        """Retrieve a material by name, acting similarly to __call__ for the class."""
        if not cls._materials:
            raise ValueError(
                "Material library is empty. Ensure materials are loaded before accessing."
            )
        try:
            instance = cls()
            return instance[material_name]
        except KeyError:
            raise KeyError(f"Material '{material_name}' not found in the library.")

    def __getitem__(self, material_name: str) -> Material:
        """Allows access to materials by name."""
        return self._materials[material_name.casefold()]

    def __repr__(self) -> str:
        materials_str = map(lambda x: f'"{x}"', self._materials.keys())
        return f"MaterialLibrary({', '.join(materials_str)})"

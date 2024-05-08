from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

from src.utils.xml_parser import explore_dictionary, parse_xml_file


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
                name = param.physical_property.name
                properties_map[name] = param

        self._properties_map = properties_map

    def __getitem__(self, key) -> MaterialProperty:
        return self._properties_map[key]

    def __repr__(self) -> str:
        available_properties = ", ".join(self._properties_map.keys())
        return f"{self.name} ({available_properties})"


@dataclass
class PhysicalProperty:
    """Represents physical properties of materials"""

    name: str
    units: list[Unit]

    def __repr__(self):

        units = [str(unit) for unit in self.units]
        return f"PhysicalProperty({self.name} [{'*'.join(units)}])"


class Unit:
    """Physical Units representative"""

    def __init__(self, name: str, power: float = 1.0):
        self.name = name
        self.power = power

    def __repr__(self):
        if self.power == 1.0:
            return f"{self.name}"
        else:
            return f"{self.name}^{self.power}"

    def __str__(self):
        if self.power == 1.0:
            return f"{self.name}"
        else:
            return f"{self.name}^{self.power}"


@dataclass
class MaterialProperty:

    physical_property: PhysicalProperty
    value: Optional[float] = None
    property_table: Optional[namedtuple] = None  # type: ignore
    description: Optional[str] = None


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


materials = parse_xml_file(
    "data/materials/xml_libraries/Materiales_Engineering_Data.xml"
)

# Maps property name to it's PhysicaProperty obj
properties_map = extract_material_properties(materials)
matml_metadata = materials["Materials"]["MatML_Doc"]["Metadata"]
parameter_id = {
    element["@attributes"]["id"]: element["Name"]
    for element in matml_metadata["ParameterDetails"]
}

parameter_id = {
    element["@attributes"]["id"]: properties_map.get(
        element["Name"], PhysicalProperty(name=element["Name"], units=[Unit("1", 1)])
    )
    for element in matml_metadata["ParameterDetails"]
}

property_id = {
    pr["@attributes"]["id"]: pr["Name"] for pr in matml_metadata["PropertyDetails"]
}

matml_materials = materials["Materials"]["MatML_Doc"]["Material"]


def find_material(material_list, material_name):
    return next(
        mat["BulkDetails"]
        for mat in material_list
        if mat["BulkDetails"].get("Name") == material_name
    )


steel = find_material(matml_materials, "Structural Steel")
balsa = find_material(matml_materials, "Balsa")


def parse_property(properties):

    if isinstance(properties, list):
        return [
            parsed_property
            for property_data in properties
            if (parsed_property := parse_property(property_data)) is not None
        ]

    match properties:

        case {
            "@attributes": {"property": pr_id},
            "ParameterValue": parameter_value,
            "Data": parameter_data,
            "Qualifier": qualifier,
        }:
            # print("case_1")
            # print(f"{qualifier=}")
            print(1, property_id[pr_id])
            physical_property = property_id[pr_id]
            params = parse_parameter(parameter_value)
            print(params)
            return (physical_property, params)

        case {
            "@attributes": {"property": pr_id},
            "ParameterValue": parameter_value,
            "Data": parameter_data,
        }:
            # print("case_2")
            # print(property_id[pr_id])
            print(2, property_id[pr_id])
            physical_property = property_id[pr_id]
            params = parse_parameter(parameter_value)
            print(params)
            return (physical_property, params)

        case _:
            print(f"No property cases: {properties}")


def parse_parameter(parameter_value):

    if isinstance(parameter_value, list):
        return [
            value
            for parameter in parameter_value
            if (value := parse_parameter(parameter)) is not None
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
            print(f"Unmatched params: {parameter_value}")


material_library = {}

for material in matml_materials:

    # material = balsa
    material = material["BulkDetails"]

    name = material["Name"]
    material.get("Description")
    material.get("Class")
    properties = material["PropertyData"]
    attributes = dict(parse_property(properties))

    material_library[material["Name"]] = Material(name, attributes)


def dprint(obj):

    n = 0
    if hasattr(obj, "__len__"):
        n = len(obj)

    print(obj, type(obj), n)

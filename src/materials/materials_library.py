from __future__ import annotations
from collections import namedtuple

from dataclasses import dataclass

from src.utils.xml_parser import parse_xml_file, explore_dictionary

@dataclass
class Material:
    """Represents a class with material properties"""

    name: str
    density: float  # kg/m**3

@dataclass
class PhysicalProperty:
    """Represents physical properties of materials"""

    name: str
    units: list[Unit]

    def __repr__(self):

        units = [str(unit) for unit in self.units]
        return f"{self.name} [{'*'.join(units)}]"
    
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

class MaterialProperty:

    physical_property: PhysicalProperty
    scalar_value: float | None
    property_table: namedtuple

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

    return [Unit(unit.get("Name"), float(unit.get("@attributes", {}).get("power", 1.0))) for unit in units]

def extract_material_properties(materials_xml: dict) -> dict[str, PhysicalProperty]:
    """Extracts material properties from parsed XML data.

    Args:
        materials_xml: The materials data parsed from an XML file.

    Returns:
        A dictionary mapping material names to their physical properties.
    """
    parameter_details = materials_xml["Materials"]["MatML_Doc"]["Metadata"]["ParameterDetails"]
    properties = {
        detail["Name"]: PhysicalProperty(
            name=detail["Name"],
            units=parse_unit(detail["Units"]) if "Units" in detail else [Unit('1', 1)]
        )
        for detail in parameter_details
    }
    return properties


materials = parse_xml_file(
    "data/materials/xml_libraries/Materiales_Engineering_Data.xml"
)

#Maps property name to it's PhysicaProperty obj
properties_map = extract_material_properties(materials)
matml_metadata = materials["Materials"]["MatML_Doc"]["Metadata"]
parameter_id = {element["@attributes"]["id"]: element["Name"] for element in matml_metadata["ParameterDetails"]}

property_id = {
    pr["@attributes"]["id"]: properties_map.get(pr["Name"], pr["Name"])
    for pr in matml_metadata["PropertyDetails"]
}

matml_materials = materials["Materials"]["MatML_Doc"]["Material"]

steel = next(mat['BulkDetails'] for mat in matml_materials if mat['BulkDetails'].get('Name') == 'Structural Steel')

for material in matml_materials:

    material = matml_materials[1]["BulkDetails"]

    material["Name"]
    material.get("Description")
    material.get("Class")

    material.keys()

    properties = material["PropertyData"]

    for property_data in properties:
        #property_data = properties[2]

        pr_id = property_data["@attributes"]["property"]

        physical_property = property_id[pr_id]
        print(1)
        print(physical_property)

        property_data.keys()
        len(property_data["ParameterValue"])
        #property_data["Data"] # Nothing useful
        #property_data["Qualifier"] # Nothing useful

        parameter_value = property_data["ParameterValue"]
        print(parameter_value)

        d = {}

        for parameter in parameter_value:
            pa_id = parameter['@attributes'].get('parameter')
            varname = parameter_id[pa_id]
            print(f'{varname=}, {pa_id=}')
            d[varname] = parameter.get('Data')


        [parameter['@attributes'].get('parameter') for parameter in parameter_value]

        parameter_id[]

def dprint(obj):

    n = 0
    if hasattr(obj, '__len__'):
        n = len(obj)

    print(obj, type(obj), n)


def extract_parameters(properties):
    # Initialize a list to hold all extracted parameters and their data
    extracted_data = []

    for property_data in properties:
        # Extract property ID
        pr_id = property_data["@attributes"]["property"]

        # Initialize a dictionary to store data for the current property
        property_dict = {"property_id": pr_id, 'property_name':property_id[pr_id], "parameters": []}

        # Handle the 'ParameterValue' field, ensuring compatibility with both list and single dictionary cases
        parameter_values = property_data.get("ParameterValue", [])
        if isinstance(parameter_values, dict):
            # If 'ParameterValue' is a single dictionary, convert it into a list of one dictionary
            parameter_values = [parameter_values]

        for parameter in parameter_values:
            pa_id = parameter['@attributes'].get('parameter')
            data = parameter.get('Data')
            parameter_name = parameter_id[pa_id]
            # Append the parameter information to the property_dict
            if parameter_name != 'Options Variable':
                property_dict["parameters"].append({"parameter_id": pa_id, 'parameter_name': parameter_name, "data": data})

            # Handle possible 'Qualifier' field, checking for both list and single dictionary cases
            qualifiers = parameter.get('Qualifier', [])
            if isinstance(qualifiers, dict):
                # If 'Qualifier' is a single dictionary, convert it into a list of one dictionary
                qualifiers = [qualifiers]
            # Extract Qualifier data if necessary, similar to how 'parameter' data is handled
            for qualifier in qualifiers:
                q_name = qualifier["@attributes"]["name"]
                q_data = qualifier.get("#text")
                #property_dict["parameters"].append({"qualifier_name": q_name, "qualifier_data": q_data})

        # Append the property_dict to the extracted_data list
        extracted_data.append(property_dict)

    return extracted_data

# Example usage
properties_extracted = extract_parameters(properties)
for property_data in properties_extracted:
    print(property_data)

    [parameter['parameter_name'].name for parameter in property_data['parameters']]



# aircraft_dict = parse_xml_file('data/xml/Mobula.xml')

# class Qualifier:
#     def __init__(self, name, value=None):
#         self.name = name
#         self.value = value
#         # No additional_info needed for this simplified correction

#     def __repr__(self):
#         return f"Qualifier(name={self.name}, value={self.value})"

# class ParameterValue:
#     def __init__(self, parameter, format, data, qualifiers):
#         self.parameter = parameter
#         self.format = format
#         self.data = data
#         self.qualifiers = []

#         for q in qualifiers:
#             # Adjusting for the correct creation of Qualifier objects
#             if isinstance(q, dict):
#                 attrs = q.get('@attributes', {})
#                 text = q.get('#text', None)
#                 # Ensuring 'name' is not duplicated
#                 self.qualifiers.append(Qualifier(name=attrs.pop('name', None), value=text))


#     def __repr__(self):
#         return f"ParameterValue(parameter={self.parameter}, format={self.format}, data={self.data}, qualifiers={self.qualifiers})"

# # Assuming the rest of your class structures are correct, let's focus on fixing the Qualifier creation.

# class PropertyData:
#     def __init__(self, property, data_format, data_text, parameter_values):
#         self.property = property
#         self.data_format = data_format
#         self.data_text = data_text
#         self.parameter_values = [ParameterValue(**param_value['@attributes'], data=param_value['Data'], qualifiers=param_value.get('Qualifier', [])) for param_value in parameter_values]

#     def __repr__(self):
#         return f"PropertyData(property={self.property}, data_format={self.data_format}, data_text={self.data_text}, parameter_values={self.parameter_values})"

# class BulkDetails:
#     def __init__(self, name, property_data):
#         self.name = name
#         self.property_data = [PropertyData(**{
#             'property': pd['@attributes']['property'],
#             'data_format': pd['Data']['@attributes']['format'],
#             'data_text': pd['Data'].get('#text', '-'),  # Defaulting to '-' if '#text' is not present
#             'parameter_values': pd['ParameterValue']
#         }) for pd in property_data]

#     def __repr__(self):
#         return f"BulkDetails(name={self.name}, property_data={self.property_data})"


# class Material:
#     def __init__(self, bulk_details):
#         # Adjusting the constructor to correctly map dictionary keys to class parameters
#         adjusted_bulk_details = {
#             'name': bulk_details['Name'],  # Correcting the case to match the class parameter
#             'property_data': bulk_details['PropertyData']  # Assuming PropertyData is structured correctly for the BulkDetails constructor
#         }
#         self.bulk_details = BulkDetails(**adjusted_bulk_details)

#     def __repr__(self):
#         return f"Material(bulk_details={self.bulk_details})"

# # Example usage
# material_data = {
#     'Name': 'Balsa', 
#     'PropertyData': [
#         # Assuming the property data from your dictionary goes here
#     ]
# }

# material_data = matml_materials[1]["BulkDetails"]

# material_obj = Material(bulk_details=material_data) 
# print(material_obj)



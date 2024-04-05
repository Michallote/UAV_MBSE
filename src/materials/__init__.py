from consts import XML_MATERIAL_LIBRARY as __XML_MAT_LIB

from .materials_library import (
    Material,
    MaterialLibrary,
    MaterialProperty,
    PhysicalProperty,
    Unit,
)

__all__ = [
    "MaterialLibrary",
    "Material",
    "PhysicalProperty",
    "Unit",
    "MaterialProperty",
]

__version__ = "1.0.0"


MaterialLibrary().load_materials(__XML_MAT_LIB)

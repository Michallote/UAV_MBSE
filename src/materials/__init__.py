from consts import XML_MATERIAL_LIBRARY as __XML_MAT_LIB

import os
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

if os.path.exists(__XML_MAT_LIB):
    MaterialLibrary().load_materials(__XML_MAT_LIB)

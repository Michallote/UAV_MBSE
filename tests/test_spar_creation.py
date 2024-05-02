import numpy as np
import plotly.graph_objects as go

from src.aerodynamics.airfoil import AirfoilFactory
from src.aerodynamics.data_structures import Aircraft
from src.geometry.aircraft_geometry import AircraftGeometry
from src.geometry.spatial_array import SpatialArray
from src.geometry.surfaces import evaluate_surface_intersection
from src.structures.structural_model import StructuralSpar

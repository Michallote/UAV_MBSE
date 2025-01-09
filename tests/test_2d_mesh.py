from typing import Any

import numpy as np

from aerodynamics.data_structures import AeroSurface
from consts import XML_MATERIAL_LIBRARY
from geometry.aircraft_geometry import GeometricCurve
from geometry.interpolation import resample_curve_with_element_length
from geometry.meshing import compute_3d_planar_mesh
from materials.materials_library import MaterialLibrary
from src.aerodynamics.airfoil import Airfoil
from src.geometry.meshing import create_mesh_from_boundary
from structures.spar import FlatSpar, StructuralSpar
from visualization.plotly_plotter import plot_2d_mesh


def test_discretization_mesh_gen():
    """Test that the mesh created by the annealing algorithm is valid."""

    airfoil = Airfoil.from_file("data/airfoils/FX 73-CL2-152 T.E..dat")
    airfoil = airfoil.with_trailing_edge_gap(te_gap=0.03, blend_distance=1.0)

    curve = airfoil.data
    length = np.round(airfoil.trailing_edge_gap, 2)
    curve = resample_curve_with_element_length(curve, length)

    # n_points = np.round(airfoil.area / (0.5 * length * length * np.sin(np.pi / 3)))

    max_area = length**2 / 2

    mesh_dict, boundary_dict = create_mesh_from_boundary(
        boundary_coordinates=curve, max_area=max_area
    )

    plot_2d_mesh(
        boundary_dict, mesh_dict, title="Airfoil Meshed", save="airfoil_mesh.html"
    )


def test_flat_spar_mesh_creation():

    curve = np.array(
        [
            [1.57927483, 0.11249055, 0.0],
            [1.57927483, 0.11249055, 0.2],
            [1.57927483, 0.11249055, 0.4],
            [1.57927483, 0.11249055, 0.6],
            [1.57927483, 0.10460901, 0.6],
            [1.57927483, 0.09672747, 0.6],
            [1.57927483, 0.08884593, 0.6],
            [1.57927483, 0.08884593, 0.4],
            [1.57927483, 0.08884593, 0.2],
            [1.57927483, 0.08884593, 0.0],
        ]
    )

    gc = GeometricCurve(name="spar", data=curve)

    plane_point = gc.data[0]
    plane_normal = gc.normal

    mesh_dict, boundary_dict = compute_3d_planar_mesh(curve, plane_point, plane_normal)

    vertices = mesh_dict["vertices"]
    mesh_dict["vertices"] = vertices[:, 1:]
    boundary_dict["vertices"] = boundary_dict["vertices"][:, 1:]

    plot_2d_mesh(boundary_dict, mesh_dict, title="Spar Mesh")

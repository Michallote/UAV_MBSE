import numpy as np

from src.aerodynamics.airfoil import Airfoil
from src.geometry.meshing import create_mesh_from_boundary
from src.utils.interpolation import resample_curve_with_element_length
from visualization.plotly_plotter import plot_2d_mesh


def test_annealing_mesh_gen():
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

import numpy as np
import pandas as pd
import plotly.express as px

from src.aerodynamics.airfoil import Airfoil
from src.geometry.transformations import rotation_matrix2d
from src.utils.interpolation import pad_arrays
from tests.test_intersection_algorithms import plot_curves


def test_basic_functions():

    airfoil = Airfoil.from_file("data/airfoils/FX 73-CL2-152 T.E..dat")

    airfoil_te_gap = airfoil.with_trailing_edge_gap(te_gap=0.0, blend_distance=1.0)
    airfoil_te_gap_1 = airfoil_te_gap.with_trailing_edge_gap(
        te_gap=0.03, blend_distance=0.15
    )
    airfoil_te_gap_2 = airfoil_te_gap.with_trailing_edge_gap(
        te_gap=0.03, blend_distance=1.0
    )

    # Plot using Plotly Express
    plot_curves(airfoil_te_gap.data, airfoil_te_gap_1.data, airfoil_te_gap_2.data)

    airfoil = Airfoil.from_file("data/databases/airfoil_coordinates_db/s1223.dat")

    airfoil_te_gap = airfoil.with_trailing_edge_gap(te_gap=0.01, blend_distance=0.5)

    airfoil_te_gap_2 = airfoil.with_trailing_edge_gap(te_gap=0.02, blend_distance=1.0)

    plot_curves(airfoil.data, airfoil_te_gap.data, airfoil_te_gap_2.data)

    np.savetxt(
        "S1223_te_gap.csv",
        pad_arrays(
            pad_arrays(airfoil.data, airfoil_te_gap.data), airfoil_te_gap_2.data
        ),
        delimiter=",",
        fmt="%.8f",
        header="s1223_x,s1223_y,s1223_TE_x,s1223_TE_y,s1223_TE2_x,s1223_TE2_y",
    )

    tuple(map(tuple, airfoil_te_gap_2.data[[0, -1]]))

    vec = airfoil_te_gap_2.data[[0, -1]]
    vec = np.diff(vec, axis=0).T

    airfoil_te_gap_2.data[[0, -1]] + np.dot(rotation_matrix2d(np.pi / 2), vec).T

    t = 0.02
    dc = np.array((t, airfoil.camber(1.0) - airfoil.camber(1 - t)))
    airfoil.trailing_edge + dc

    dt = np.dot(rotation_matrix2d(np.pi / 2), dc)
    airfoil.trailing_edge + dt


def test_transition_function():

    xoc = np.linspace(0, 1, 50)
    blend_distance = 0.5

    df = []

    for blend_distance in [1.0, 0.5, 0.25, 0.125, 0.125 / 2]:

        arg = np.minimum((1.0 - xoc) * (1.0 / blend_distance - 1.0), 15.0)
        thickness_factor = xoc * np.exp(-arg)

        df.append(
            pd.DataFrame(
                {
                    "x": xoc,
                    "y": thickness_factor,
                    "blend_distance": np.zeros_like(xoc) + blend_distance,
                }
            )
        )

    df = pd.concat(df)

    fig = px.line(df, x="x", y="y", color="blend_distance", markers=True)
    fig.show()

    # import tikzplotly

    # tikzplotly.save("figure.tex", fig)

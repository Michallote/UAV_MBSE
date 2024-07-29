import numpy as np
import pandas as pd
import plotly.express as px

from src.aerodynamics.airfoil import Airfoil
from src.utils.interpolation import pad_arrays
from src.utils.transformations import rotation_matrix2d
from tests.test_intersection_algorithms import plot_curves

ENABLE_RETURN = False


def test_basic_functions():

    airfoil = Airfoil.from_file("data/airfoils/FX 73-CL2-152 T.E..dat")

    airfoil_te_gap = airfoil.with_trailing_edge_gap(te_gap=0.0, blend_distance=1.0)

    target_te = 0.03

    airfoil_te_gap_1 = airfoil_te_gap.with_trailing_edge_gap(
        te_gap=target_te, blend_distance=0.15
    )
    airfoil_te_gap_2 = airfoil_te_gap.with_trailing_edge_gap(
        te_gap=target_te, blend_distance=1.0
    )

    te_gap = np.linalg.norm(
        airfoil_te_gap_1.trailing_edge - airfoil_te_gap_1.trailing_edge2
    )
    assert np.isclose(te_gap, target_te, rtol=0.0001)

    te_gap = np.linalg.norm(
        airfoil_te_gap_2.trailing_edge - airfoil_te_gap_2.trailing_edge2
    )

    assert np.isclose(te_gap, target_te, rtol=0.0001)

    if ENABLE_RETURN:
        return airfoil, airfoil_te_gap, airfoil_te_gap_1, airfoil_te_gap_2


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

    if ENABLE_RETURN:
        return df

    # import tikzplotly

    # tikzplotly.save("figure.tex", fig)


if __name__ == "__main__":

    ENABLE_RETURN = True

    airfoils = test_basic_functions()
    plot_curves(*(airfoil.data for airfoil in airfoils))

    df = test_transition_function()
    fig = px.line(df, x="x", y="y", color="blend_distance", markers=True)
    fig.show()

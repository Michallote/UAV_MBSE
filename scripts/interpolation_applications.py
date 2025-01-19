import numpy as np

from geometry.interpolation import ndarray_linear_interpolate, resample_curve
from src.aerodynamics.airfoil import Airfoil
from src.geometry.spatial_array import SpatialArray
from tests.test_intersection_algorithms import plot_curves

a1 = Airfoil.from_file("data/databases/airfoil_coordinates_db/s1210.dat").resample(100)
a2 = Airfoil.from_file("data/airfoils/GOE 383 AIRFOIL.dat").resample(100)
a3 = ndarray_linear_interpolate(
    curve=np.array([a1.data, a2.data]), indices=np.array([0, 0.5, 1])
)

fig = plot_curves(*a3)
# tikzplotly.save("airfoil_blending.tex", fig)

n = 100
a4 = Airfoil.from_file("data/databases/airfoil_coordinates_db/s1223rtl.dat").resample(n)
a5 = a4.resample(2 * n - 1)
a6 = a4.resample(n // 2)


fig2 = plot_curves(a4.data, a5.data, a6.data)
# tikzplotly.save("airfoil_resampling.tex", fig2)


import plotly.graph_objects as go

from src.aerodynamics.airfoil import AirfoilFactory
from src.aerodynamics.data_structures import Aircraft
from src.geometry.aircraft_geometry import AircraftGeometry

airfoil_factory = AirfoilFactory()
airfoil_factory.set_folder_path("data/airfoils")
airfoil_factory.cache_airfoils()
aircraft = Aircraft.from_xml("data/xml/Mobula2.xml")

aircraft_geom = AircraftGeometry(aircraft)

curves = aircraft_geom.surfaces[0].curves

curves[1]
curves[-3]

curves_arr = np.array([curves[1].data, curves[-3].data])

curves_interp = ndarray_linear_interpolate(
    curve=curves_arr, indices=np.array([0, 0.25, 0.5, 0.75, 1])
)

fig = go.Figure()

for curve in curves_interp:
    curve = SpatialArray(curve)
    fig.add_trace(
        go.Scatter3d(
            x=curve.x,
            y=curve.y,
            z=curve.z,
            mode="lines",
            # name=name,
            showlegend=True,
        ),
    )

zoom = 1.2
up = {"x": 0, "y": 1, "z": 0}
eye = {"x": -1.5, "y": 1.5, "z": 2}

eye = {key: value / zoom for key, value in eye.items()}
camera = dict(up=up, eye=eye)

fig.update_layout(
    # title=aircraft.name,
    scene=dict(camera=camera, aspectmode="data"),
    # template="plotly_dark",
)

fig.show()
# tikzplotly.save("spatial_interpolation.tex", fig)


curves[0].data


import io

import numpy as np

# Example arrays
array1 = np.random.rand(5, 3)
array2 = np.random.rand(4, 3)
array3 = np.random.rand(6, 3)


def write_tikz_surface_file(arrays: list[np.ndarray], filename="output.dat"):
    # Create a StringIO object
    output = io.StringIO()

    # Write each array to the StringIO object
    for array in arrays:
        np.savetxt(output, array, delimiter=" ")
        output.write("\n")

    # Get the content of the StringIO object
    content = output.getvalue()

    # Write the content to a .dat file

    with open(filename, "w") as file:
        file.write(content)

    print(f"Arrays saved to {filename}.")


n = 60
arrays = [curve.resample(n_samples=n).data for curve in curves]

n = len(arrays)
ims = 0

n = n + (n - 1) * ims

arrays = resample_curve(np.array(arrays), num_samples=n)


write_tikz_surface_file(
    arrays, filename="/home/neptorqc2/Documents/TesisMich/Thesis/Graphs/CSV/output.dat"
)

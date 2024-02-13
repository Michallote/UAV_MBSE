import numpy as np
import plotly.graph_objects as go

from src.aerodynamics.airfoil import AirfoilFactory
from src.aerodynamics.data_structures import Aircraft
from src.geometry.aircraft_geometry import AircraftGeometry
from src.geometry.spatial_array import SpatialArray
from src.geometry.surfaces import evaluate_surface_intersection
from src.structures.structural_model import StructuralSpar

airfoil_factory = AirfoilFactory()
airfoil_factory.set_folder_path("data/airfoils")
airfoil_factory.cache_airfoils()
aircraft = Aircraft.from_xml("data/xml/Mobula2.xml")

aircraft_geom = AircraftGeometry(aircraft)
# x = np.linspace(0, 10, 150)
# y = np.linspace(0, 10, 150)

# xx, yy = np.meshgrid(x, y)
# zz = np.sin(xx**2) + np.cos(yy**2)

surface = aircraft_geom.surfaces[0]

xx = surface.xx
yy = surface.yy
zz = surface.zz

p = np.array([0.307126562252638, 0, 0])
n = np.array([1, 0, 0])


curve_1 = SpatialArray(
    evaluate_surface_intersection(xx, yy, zz, np.array([0.43, 0, 0]), n)
)


spar = StructuralSpar.from_surface_and_plane(surface, p, n)

curve = spar.curve

fig = go.Figure(
    data=[
        go.Surface(z=zz, x=xx, y=yy),
    ]
)

fig.add_trace(
    go.Scatter3d(
        x=curve.x,
        y=curve.y,
        z=curve.z,
        mode="lines+markers",  # Combine lines and markers
        line=dict(color="black", width=10),  # Thick black line
        marker=dict(size=5, color="red"),  # Red markers
    )
)

fig.add_trace(
    go.Scatter3d(
        x=curve_1.x,
        y=curve_1.y,
        z=curve_1.z,
        mode="lines+markers",  # Combine lines and markers
        line=dict(color="red", width=10),  # Thick black line
        marker=dict(size=5, color="black"),  # Red markers
    )
)

# Update layout for a better view
fig.update_layout(
    title="3D Surface Plot",
    # autosize=False,
    # width=500,
    # height=500,
    margin=dict(l=65, r=50, b=65, t=90),
    scene=dict(
        xaxis=dict(showbackground=False),
        yaxis=dict(showbackground=False),
        zaxis=dict(showbackground=False),
        aspectmode="data",
    ),
)

# Show the figure
fig.show()

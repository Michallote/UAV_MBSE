import os
import webbrowser
from itertools import chain
from os.path import join, normpath

import numpy as np
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from plotly.offline import plot
from plotly.subplots import make_subplots

from src.geometry.aircraft_geometry import (
    AircraftGeometry,
    GeometricCurve,
    GeometricSurface,
)
from src.structures.structural_model import StructuralModel, SurfaceStructure
from src.visualization.base_class import BaseAircraftPlotter


class PlotlyAircraftPlotter(BaseAircraftPlotter):
    """
    Creates a visualization for Aircraft Surfaces and Curves using the Plotly backend
    """

    @staticmethod
    def plot_surface(surface, fig: go.Figure, row, col, color=None):
        """
        Add a surface plot to the figure.

        Parameters:
        surface (GeometricSurface): The geometric surface to plot.
        fig (go.Figure): The Plotly figure to add the plot to.
        color (str, optional): The color of the surface. Defaults to None.
        """
        if color is None:
            color = surface.color
            # color = "rgb" + (tuple(value for value in color))

        surfacecolor, colorscale = PlotlyAircraftPlotter._surface_color(
            surface.zz, color
        )

        fig.add_trace(
            go.Surface(
                x=surface.xx,
                y=surface.yy,
                z=surface.zz,
                surfacecolor=surfacecolor,
                colorscale=colorscale,
                showscale=False,
            ),
            row=row,
            col=col,
        )

    @staticmethod
    def plot_curve(
        curve: GeometricCurve,
        fig: go.Figure,
        row,
        col,
        name="Curve",
        showlegend=True,
        color=None,
    ):
        """
        Add a curve plot to the figure.

        Parameters:
        curve (GeometricCurve): The geometric curve to plot.
        fig (go.Figure): The Plotly figure to add the plot to.
        """
        line_properties = {}

        if color is not None:
            line_properties["color"] = color

        fig.add_trace(
            go.Scatter3d(
                x=curve.x,
                y=curve.y,
                z=curve.z,
                mode="lines",
                name=name,
                showlegend=showlegend,
                line=line_properties,
            ),
            row,
            col,
        )

    def plot_aircraft(self, aircraft: AircraftGeometry):
        """
        Plots the aircraft using the geometric data.
        """
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"is_3d": True}, {"is_3d": True}]],
            subplot_titles=[
                "Surface Plot",
                "Curve Plot",
            ],
        )

        # Plot surfaces
        for surface in aircraft.surfaces:
            self.plot_surface(surface, fig, row=1, col=1)

        # Plot curves and borders
        for surface in aircraft.surfaces:
            for i, curve in enumerate(chain(surface.curves, surface.borders)):
                name = _create_name(surface, curve, i)

                self.plot_curve(
                    curve, fig, row=1, col=1, showlegend=False, color="black"
                )
                self.plot_curve(
                    curve, fig, row=1, col=2, name=name, color=f"rgba{surface.color}"
                )

        zoom = 1.2
        up = {"x": 0, "y": 1, "z": 0}
        eye = {"x": -1.5, "y": 1.5, "z": 2}

        eye = {key: value / zoom for key, value in eye.items()}
        camera = dict(up=up, eye=eye)

        fig.update_layout(
            title=aircraft.name,
            scene=dict(camera=camera),
            scene2=dict(camera=camera),
            template="plotly_dark",
        )

        # fig.show()

        add_subplot_synchronization(
            fig, filename=aircraft.name, output_path="data/output"
        )

    @staticmethod
    def _surface_color(z, color, transparency=True):
        def normalize_color_value(value):
            if isinstance(value, (int, float)):
                return value / 255
            elif 0 <= value <= 1:
                return value
            else:
                raise ValueError(
                    "Color value should be an integer or a float between 0 and 1", value
                )

        if isinstance(color, tuple):
            if not all(isinstance(v, (int, float)) for v in color):
                raise ValueError(
                    "All elements of color tuple must be integers or floats"
                )

            if len(color) not in [3, 4]:
                raise ValueError(
                    "Color tuple must have either 3 (RGB) or 4 (RGBA) elements"
                )

            # Normalize RGB values and ensure alpha value is present
            color = tuple(normalize_color_value(v) for v in color)

            if transparency:
                # Reduce the alpha value by half for transparency
                color = color[:3] + (color[3] / 2,)

            color_str = f"{'rgba' if len(color) == 4 else 'rgb'}{color}"
        else:
            color_str = str(color)

        surfacecolor = np.zeros_like(z) + 1
        colorscale = [[0, color_str], [1, color_str]]
        return surfacecolor, colorscale

    def plot_structure(self, structure: StructuralModel):

        fig = go.Figure()

        # # Add the 3D curve
        # fig.add_trace(
        #     go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="blue", width=10))
        # )

        for component_structure in structure.structures:

            for rib in component_structure.ribs:
                curve = rib.curve
                x, y, z = curve.x, curve.y, curve.z

                n = len(curve) - 1

                indices = np.array(
                    [result for i in range(n) if all_different(result := [i, i + 1, n - i])]
                )
                i, j, k = indices.T

                fig.add_trace(
                        go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=0.5)
                    )


        # Add a surface under the curve to simulate filling
        
        fig.show()

    @staticmethod
    def plot_component(component: SurfaceStructure):



def add_subplot_synchronization(
    fig: go.Figure, filename: str, output_path: str = "data/output"
):
    """Adds camera synchronization into a figure with 2 subplots."""
    # get the a div
    div = plot(fig, include_plotlyjs=False, output_type="div")

    # Use BeautifulSoup to parse the HTML and extract the div id
    soup = BeautifulSoup(div, "html.parser")
    div_id = soup.find("div", class_="plotly-graph-div")["id"]  # type: ignore

    # your custom JS code
    js = """
        <script>
        var gd = document.getElementById('_div_id_');
        var isUnderRelayout = false;
        
        function updateCamera(sceneToUpdate, newCamera) {
            if (!isUnderRelayout) {
                isUnderRelayout = true;
                Plotly.relayout(gd, sceneToUpdate, newCamera)
                    .then(() => { isUnderRelayout = false });
            }
        }
        
        gd.on('plotly_relayout', (eventData) => {
            if (eventData['scene.camera']) {
                updateCamera('scene2.camera', gd.layout.scene.camera);
            }
            else if (eventData['scene2.camera']) {
                updateCamera('scene.camera', gd.layout.scene2.camera);
            }
        });
        </script>"""

    js = js.replace("_div_id_", div_id)  # type: ignore

    # Merge everything
    html_content = (
        '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>' + div + js
    )

    # Write to an HTML file
    file_path = f"{filename}.html"

    file_path = join(normpath(output_path), file_path)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(html_content)

    # Open in default browser
    webbrowser.open("file://" + os.path.realpath(file_path))


def _create_name(surface: GeometricSurface, curve: GeometricCurve, i: int) -> str:
    """Helper function to create curve name to fit in frame"""
    name = surface.surface_type.name[0] + str(i) + "_" + curve.name
    name = name.replace(" ", "").replace("-", "")

    if len(name) > 18:
        name = name[:7] + ".." + name[-5:]
    elif len(name) > 13:
        name = name[:9] + "..."

    return name

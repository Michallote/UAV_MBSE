import os
import webbrowser
from itertools import chain
from os.path import join, normpath
from typing import Any

import numpy as np
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from plotly.offline import plot
from plotly.subplots import make_subplots

from src.geometry.aircraft_geometry import (
    AircraftGeometry,
    GeometricCurve,
    GeometricSurface,
    all_different,
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
        def normalize_color_value(value: int | float) -> float:
            if isinstance(value, (int, float)):
                if 0 <= value <= 1:
                    return value

                return value / 255

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

        surfacecolor = np.zeros_like(z, dtype=float) + 1
        colorscale = [[0, color_str], [1, color_str]]
        return surfacecolor, colorscale

    def plot_structure(self, structure: StructuralModel):

        # Assuming `structure` is an object that has a `components` method returning iterable components
        fig = go.Figure()

        labels = []
        legend_added = set()

        for surface_structure, component_structure in structure.components(
            yield_structure=True
        ):

            x, y, z, i, j, k = component_structure.mesh

            if surface_structure is not None:
                surface_type = surface_structure.surface_type
                surface_name = surface_type.name
            else:
                surface_name = "External"

            component_name = type(component_structure).__name__
            name = f"{surface_name}_{component_name}"
            labels.append(
                dict(
                    name=name, surface_name=surface_name, component_name=component_name
                )
            )
            # Decide whether to show this trace in the legend
            show_legend = False
            if name not in legend_added:
                show_legend = True
                legend_added.add(
                    name
                )  # Add this name to the set so it won't be added again

            # vc = np.full_like(i, 'black', dtype=str)
            # vc = ['black' for _ in range(len(i))]

            # Add trace with a unique name and same legendgroup for toggling
            fig.add_trace(
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=i,
                    j=j,
                    k=k,
                    opacity=0.5,
                    name=f"{name}",  # Unique name for each trace
                    legendgroup=f"{name}",  # Unique group for each trace
                    showlegend=show_legend,  # Ensure the legend is shown
                    # vertexcolor=vc,
                )
            )

            triangles = np.vstack((i, j, k)).T
            vertices = np.vstack((x, y, z)).T
            tri_points = vertices[triangles]

            def pad_array(x: np.ndarray) -> np.ndarray:

                tri_order = np.array([0, 1, 2, 0])
                return np.pad(
                    x[tri_order], pad_width=((0, 1), (0, 0)), constant_values=np.nan
                )

            wireframe = np.vstack([*map(pad_array, tri_points)])

            xe, ye, ze = wireframe.T

            show_legend = False
            if "Wireframe" not in legend_added:
                show_legend = True
                legend_added.add(
                    "Wireframe"
                )  # Add this name to the set so it won't be added again

            labels.append(
                dict(
                    name="Wireframe",
                    surface_name=surface_name,
                    component_name=component_name,
                )
            )

            # define the trace for triangle sides
            lines = go.Scatter3d(
                x=xe,
                y=ye,
                z=ze,
                mode="lines",
                line=dict(color="rgb(70,70,70)", width=1),
                name="Wireframe",  # Unique name for each trace
                legendgroup="Wireframe",  # Unique group for each trace
                showlegend=show_legend,  # Ensure the legend is shown
            )

            fig.add_trace(lines)

        surfaces_n = np.array([item["surface_name"] for item in labels])
        components_n = np.array([item["component_name"] for item in labels])

        unique_surfaces = np.unique(surfaces_n)
        unique_components = np.unique(components_n)

        surface_map = {surf: (surfaces_n == surf) for surf in unique_surfaces}
        component_map = {comp: (components_n == comp) for comp in unique_components}

        def create_button(label, mask) -> dict[str, Any]:
            return {
                "label": f"Show {label}",
                "method": "restyle",
                "args": [{"visible": [(True if i else "legendonly") for i in mask]}],
            }

        buttons = [
            create_button(label, mask)
            for label, mask in chain(surface_map.items(), component_map.items())
        ]

        buttons.insert(
            0,
            {
                "label": "Show All",
                "method": "restyle",
                "args": [{"visible": [True for i in range(len(labels))]}],
            },
        )

        updatemenus = [
            {
                "buttons": buttons,
                "direction": "down",  # Layout direction of buttons
                "pad": {"r": 5, "t": 5},  # Padding from the right and top edges
                "showactive": True,
                "x": 0.05,  # Button group's left edge is aligned with the left edge of the plot
                "xanchor": "left",  # Anchor the button group to the left
                "y": 0.95,  # Position the button group above the plot
                "yanchor": "top",  # Anchor the button group to the top
                "font": {"size": 16},  # Increase the font size here
            }
        ]

        # Set up the legend to toggle visibility per group
        fig.update_layout(
            legend=dict(
                itemsizing="constant",
                tracegroupgap=0,
                title="Components",  # Optional: Add a title to the legend
            ),
            updatemenus=updatemenus,
            scene=dict(
                aspectmode="data"  # This ensures that the scaling is equal along all axes
            ),
        )

        zoom = 1.2
        up = {"x": 0, "y": 1, "z": 0}
        eye = {"x": -1.5, "y": 1.5, "z": 2}

        eye = {key: value / zoom for key, value in eye.items()}
        camera = dict(up=up, eye=eye)

        fig.update_layout(
            title=dict(
                text=structure.aircraft.name, font=dict(size=24), automargin=True
            ),
            scene=dict(camera=camera),
            template="plotly_dark",
            margin=(dict(l=0, r=0, t=0, b=0)),
            legend=dict(
                xanchor="right",
                x=1.0,
                yanchor="top",
                y=1.0,
                # orientation="h",
                # entrywidth=entrywidth,  # change it to 0.3
                # entrywidthmode="fraction",
            ),
            paper_bgcolor="Black",
            plot_bgcolor="black",
        )

        fig.show()


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

import os
import webbrowser
from itertools import chain
from os.path import join, normpath
from typing import Any, Optional

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from bs4 import BeautifulSoup
from plotly.offline import plot
from plotly.subplots import make_subplots

from geometry.intersection import enforce_closed_curve
from src.geometry.aircraft_geometry import (
    AircraftGeometry,
    GeometricCurve,
    GeometricSurface,
)
from src.structures.structural_model import StructuralModel
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


def plot_2d_mesh(
    boundary_dict: dict,
    mesh_dict: dict,
    title: str,
    save: Optional[str] = None,
    show: bool = True,
) -> go.Figure:
    """
    Plot a 2D mesh generated by the triangulation process using Plotly.

    This function visualizes the triangulated mesh, highlighting the triangles and
    the boundary of the original polygon.

    Parameters
    ----------
    boundary_dict : dict
        A dictionary containing the vertices of the polygon boundary, typically obtained
        from the `create_boundary_dict` function. The key "vertices" refers to the array
        of coordinates defining the boundary.
    mesh_dict : dict
        A dictionary containing the triangulated mesh returned by `triangle.triangulate`.
        It includes:
        - 'vertices': The coordinates of all points in the triangulation.
        - 'triangles': The indices of vertices forming each triangle.
    title : str
        The title of the plot.
    save : Optional[str], default None
        File path to save the plot as an HTML file. If None, the plot will not be saved.
    show : bool, default True
        Whether to display the plot interactively.

    Returns
    -------
    go.Figure
        The Plotly figure object.
    """
    fig = go.Figure()
    showlegend = True
    # Add triangles to the plot
    for i, triangle_indices in enumerate(mesh_dict["triangles"]):
        simplex = mesh_dict["vertices"][triangle_indices]
        simplex = enforce_closed_curve(simplex)
        fig.add_trace(
            go.Scatter(
                x=simplex[:, 0],
                y=simplex[:, 1],
                fill="toself",
                # fillcolor="darkviolet",
                hoveron="fills",  # select where hover is active
                # line_color="darkviolet",
                text=f"Element: {i}, Vertices: {triangle_indices}",
                hoverinfo="text",
                mode="lines",
                line=dict(width=1),
                showlegend=showlegend,
                name="Elements",
                legendgroup="Elements",
            )
        )
        showlegend = False

    # Add element numbering
    # Calculate centroids for each triangle
    centroids = np.mean(mesh_dict["vertices"][mesh_dict["triangles"]], axis=1)
    element_numbers = list(
        map(lambda x: f"{x}", np.arange(len(mesh_dict["triangles"])))
    )

    # Scatter trace for element numbers at centroids
    element_labels_trace = go.Scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        mode="text",  # Only display text
        text=element_numbers,
        hoverinfo="none",  # No hover info for the text
        textposition="middle center",  # Centered on the centroid
        showlegend=False,  # Do not display in legend
        name="Element Numbers",
        legendgroup="Elements",  # Group with the element traces
        visible=False,
    )
    fig.add_trace(element_labels_trace)

    # Add mesh vertices with dynamic modes and hover info
    vertex_numbers = list(map(lambda x: f"{x}", np.arange(len(mesh_dict["vertices"]))))
    vertex_labels = [f"Vertex {i+1}" for i in range(len(mesh_dict["vertices"]))]
    fig.add_trace(
        go.Scatter(
            x=mesh_dict["vertices"][:, 0],
            y=mesh_dict["vertices"][:, 1],
            mode="markers",
            text=vertex_numbers,
            hoverinfo="text+x+y",
            marker=dict(color="red"),
            showlegend=True,
            name="Vertices",
            legendgroup="Vertices",
            textposition="top center",
        )
    )

    # Add boundary lines
    boundary_vertices = boundary_dict["vertices"]
    boundary_vertices = enforce_closed_curve(boundary_vertices)
    fig.add_trace(
        go.Scatter(
            x=boundary_vertices[:, 0],
            y=boundary_vertices[:, 1],
            mode="lines+markers",
            line=dict(color="blue", width=2),
            marker=dict(size=6, color="blue"),
            name="Boundary",
        )
    )

    text_options: dict[str, dict] = {
        "mode": {"Vertices": "text+markers", "Element Numbers": "text"},
        "text": {"Vertices": vertex_numbers, "Element Numbers": element_numbers},
        "visible": {"Vertices": True, "Element Numbers": True},
    }

    markers_options: dict[str, dict] = {
        "mode": {"Vertices": "markers", "Element Numbers": "text"},
        "text": {"Vertices": vertex_labels, "Element Numbers": element_numbers},
        "visible": {"Vertices": True, "Element Numbers": False},
    }

    # Configure updatemenus
    buttons = [
        {
            "label": "Markers Only",
            "method": "update",
            "args": [
                {
                    "mode": [
                        markers_options["mode"].get(trace.name, trace.mode)
                        for trace in fig.data
                    ],
                    "text": [
                        markers_options["text"].get(trace.name, trace.text)
                        for trace in fig.data
                    ],
                    "visible": [
                        markers_options["visible"].get(trace.name, trace.visible)
                        for trace in fig.data
                    ],
                },
                {"title": f"{title} (Markers Only)"},
            ],
        },
        {
            "label": "Text + Markers",
            "method": "update",
            "args": [
                {
                    "mode": [
                        text_options["mode"].get(trace.name, trace.mode)
                        for trace in fig.data
                    ],
                    "text": [
                        text_options["text"].get(trace.name, trace.text)
                        for trace in fig.data
                    ],
                    "visible": [
                        text_options["visible"].get(trace.name, trace.visible)
                        for trace in fig.data
                    ],
                },
                {"title": f"{title} (Text + Markers)"},
            ],
        },
    ]

    updatemenus = [
        {
            "buttons": buttons,
            "direction": "down",
            "pad": {"r": 5, "t": 5},
            "showactive": True,
            "x": 0.05,
            "xanchor": "left",
            "y": 0.95,
            "yanchor": "top",
        }
    ]

    # Configure the layout
    fig.update_layout(
        title=title,
        xaxis=dict(title="X", showgrid=True, zeroline=False),
        yaxis=dict(title="Y", showgrid=True, zeroline=False, scaleanchor="x"),
        showlegend=True,
        updatemenus=updatemenus,
    )

    # Save the plot to a file if specified
    if save:
        fig.write_html(save)

    # Show the plot interactively if specified
    if show:
        fig.show()

    return fig


def add_mesh_triangles(
    fig: go.Figure,
    vertices: np.ndarray,
    triangles: np.ndarray,
    add_numbering: bool = False,
    element_coloring: str = "default_sequence",
) -> go.Figure:
    """
    Add 3D triangular mesh elements to the Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure to which the triangles are added.
    vertices : np.ndarray
        Array of vertex coordinates.
    triangles : np.ndarray
        Array of triangle vertex indices.
    showlegend : bool, default True
        Whether to show the legend for this trace.

    Returns
    -------
    go.Figure
        The updated Plotly figure.
    """

    x, y, z = vertices.T
    i, j, k = triangles.T

    if element_coloring == "default_sequence":
        # Get the current active template
        current_template = pio.templates.default
        # Retrieve the color sequence from the active template
        color_sequence = pio.templates[current_template].layout.colorway

        element_color = [
            color_sequence[i % len(color_sequence)] for i in range(len(triangles))
        ]
    else:
        element_color = None

    element_numbers = list(range(len(triangles)))

    text_list = [
        f"Element {element} <br>Vertices: {idx}"
        for element, idx in zip(element_numbers, triangles)
    ]

    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            facecolor=element_color,
            name="Elements",
            legendgroup="Elements",
            showlegend=True,
            opacity=0.8,
            text=text_list,
            hoverinfo="text+x+y+z",
        )
    )

    # Draw triangle borders
    for i, triangle_indices in enumerate(triangles):
        simplex = vertices[triangle_indices]
        simplex = enforce_closed_curve(simplex)
        fig.add_trace(
            go.Scatter3d(
                x=simplex[:, 0],
                y=simplex[:, 1],
                z=simplex[:, 2],
                mode="lines",
                line=dict(width=2, color="black"),
                hoverinfo="none",
                showlegend=False,
                name=None,
                legendgroup="Elements",
            )
        )

    # Draw Vertices
    vertex_numbers = list(range(len(vertices)))
    vertex_labels = list(map(lambda s: f"Vertex: {s}", vertex_numbers))
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            text=vertex_labels,
            marker=dict(size=4, color="red"),
            name="Vertices",
            textposition="bottom center",
            textfont=dict(size=14, color="red"),
        )
    )

    # Draw Element Numbers on Centroid
    centroids = np.sum(vertices[triangles], axis=1) / 3

    fig.add_trace(
        go.Scatter3d(
            x=centroids[:, 0],
            y=centroids[:, 1],
            z=centroids[:, 2],
            mode="text",
            text=element_numbers,
            textfont=dict(size=14, color="black"),
            name="Element Numbers",
            hoverinfo="none",  # Optional: Avoid hover tooltips for text
            legendgroup="Elements",
            textposition="middle center",
            showlegend=False,
            visible=False,
        )
    )

    if add_numbering:
        # Define the states you wish each button takes the figure to:

        # No numbering enabled
        markers_options = {
            "button_label": "Show Geometry Only",
            "mode": {"Vertices": "markers"},
            "text": {"Vertices": vertex_labels},
            "visible": {"Vertices": True, "Element Numbers": False},
            "opacity": {"Elements": 0.85},
        }
        # Show numbers on top of elements and vertices
        text_options = {
            "button_label": "Show Numbers",
            "mode": {"Vertices": "text+markers"},
            "text": {"Vertices": vertex_numbers},
            "visible": {"Vertices": True, "Element Numbers": True},
            "opacity": {"Elements": 0.6},
        }

        traces = fig.data

        # Configure updatemenus
        buttons = [
            create_button_options(traces, markers_options),
            create_button_options(traces, text_options),
        ]

        updatemenus = [
            {
                "buttons": buttons,
                "direction": "down",
                "pad": {"r": 5, "t": 5},
                "showactive": True,
                "x": 0.05,
                "xanchor": "left",
                "y": 0.95,
                "yanchor": "top",
            }
        ]

        # Configure the layout
        fig.update_layout(
            updatemenus=updatemenus,
        )

    return fig


def create_button_options(traces, options: dict) -> dict[str, Any]:
    """
    Creates a button configuration for Plotly's `updatemenus` to update specific trace attributes
    based on the provided options. This function abstracts the process of defining the button's
    behavior by dynamically mapping trace names to their desired states.

    Parameters
    ----------
    traces : list[plotly.graph_objects.Figure.data]
        A list of traces from the figure (e.g., `fig.data`) to which the updates will apply.
    options : dict
        A dictionary specifying the desired configuration for the button. Keys in the dictionary
        include:
        - "button_label": str
            The label to display on the button.
        - Other keys (e.g., "mode", "text", "visible"): dict[str, Any]
            Dictionaries where keys are trace names and values are the desired attribute values
            for each trace.

    Returns
    -------
    dict[str, Any]
        A dictionary defining the button's configuration. This includes:
        - "label": The button label.
        - "method": Always set to "update" to update the figure dynamically.
        - "args": A list containing a dictionary mapping trace attributes (e.g., "mode", "text",
          "visible") to their updated values for each trace.

    Example
    -------
    Example usage with a Plotly figure:

    text_options = {
        "button_label": "Text + Markers",
        "mode": {"Vertices": "text+markers"},
        "text": {"Vertices": vertex_numbers},
        "visible": {"Vertices": True, "Element Numbers": True},
    }

    traces = fig.data

    # Generate a button configuration
    button = create_button_options(traces, text_options)

    updatemenus = [
        {
            "buttons": [button],
            "direction": "down",
            "showactive": True,
        }
    ]

    fig.update_layout(updatemenus=updatemenus)
    """

    button_label = options.pop("button_label")

    args = {
        key: [value.get(trace.name, getattr(trace, key, None)) for trace in traces]
        for key, value in options.items()
    }

    return {
        "label": button_label,
        "method": "update",
        "args": [args],
    }


def add_boundary_trace(fig, boundary_vertices):
    """
    Add boundary trace for 3D mesh.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure to which the boundary trace is added.
    boundary_vertices : np.ndarray
        Array of boundary vertex coordinates.

    Returns
    -------
    go.Figure
        The updated Plotly figure.
    """
    boundary_vertices = enforce_closed_curve(boundary_vertices)
    fig.add_trace(
        go.Scatter3d(
            x=boundary_vertices[:, 0],
            y=boundary_vertices[:, 1],
            z=boundary_vertices[:, 2],
            mode="lines+markers",
            line=dict(color="blue", width=3),
            marker=dict(size=4, color="blue"),
            name="Boundary",
        )
    )
    return fig


def plot_3d_mesh(
    boundary_dict: dict,
    mesh_dict: dict,
    title: str,
    add_numbering: bool = False,
    element_coloring: str = "default_sequence",
    save: Optional[str] = None,
    show: bool = True,
) -> go.Figure:
    """
    Plot a 3D mesh generated by a triangulation process using Plotly.

    This function visualizes the triangulated 3D mesh, highlighting the triangles and
    the boundary of the original polygon.

    Parameters
    ----------
    boundary_dict : dict
        A dictionary containing the vertices of the polygon boundary. The key "vertices"
        refers to the array of coordinates defining the boundary.
    mesh_dict : dict
        A dictionary containing the triangulated mesh. It includes:
        - 'vertices': The coordinates of all points in the triangulation.
        - 'triangles': The indices of vertices forming each triangle.
    title : str
        The title of the plot.
    save : Optional[str], default None
        File path to save the plot as an HTML file. If None, the plot will not be saved.
    show : bool, default True
        Whether to display the plot interactively.

    Returns
    -------
    go.Figure
        The Plotly figure object.
    """
    fig = go.Figure()

    # Add triangles to the plot
    fig = add_mesh_triangles(
        fig,
        mesh_dict["vertices"],
        mesh_dict["triangles"],
        add_numbering=add_numbering,
        element_coloring=element_coloring,
    )

    # Add boundary lines
    fig = add_boundary_trace(fig, boundary_dict["vertices"])

    # Configure the layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X", showgrid=True, zeroline=False),
            yaxis=dict(title="Y", showgrid=True, zeroline=False),
            zaxis=dict(title="Z", showgrid=True, zeroline=False),
            aspectmode="data",  # This ensures that the scaling is equal along all axes
        ),
        showlegend=True,
    )

    # Save the plot to a file if specified
    if save:
        fig.write_html(save)

    # Show the plot interactively if specified
    if show:
        fig.show()

    return fig

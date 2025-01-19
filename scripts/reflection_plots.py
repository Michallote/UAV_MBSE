import numpy as np
import plotly.graph_objs as go

from src.geometry.transformations import reflect_curve_by_plane

# Example usage
data = np.arange(10 * 3).reshape(-1, 3)
normal_vector = np.array([1, 1, 1])

reflected_data = reflect_curve_by_plane(data, normal_vector)
data
reflected_data


# Create traces for the original and reflected data
trace1 = go.Scatter3d(
    x=data[:, 0],
    y=data[:, 1],
    z=data[:, 2],
    mode="markers",
    marker=dict(size=5, color="blue"),
    name="Original Data",
)

trace2 = go.Scatter3d(
    x=reflected_data[:, 0],
    y=reflected_data[:, 1],
    z=reflected_data[:, 2],
    mode="markers",
    marker=dict(size=5, color="red"),
    name="Reflected Data",
)

# Create the layout
layout = go.Layout(
    title="Original and Reflected Data",
    scene=dict(
        xaxis=dict(title="X-axis"),
        yaxis=dict(title="Y-axis"),
        zaxis=dict(title="Z-axis"),
    ),
)

# Create the figure and plot it
fig = go.Figure(data=[trace1, trace2], layout=layout)
fig.show()

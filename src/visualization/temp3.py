import numpy as np
import plotly.graph_objects as go


# Function to create a 3D ellipse
def create_3d_ellipse(center, radii, rotation, num_points=100):
    t = np.linspace(0, 2 * np.pi, num_points)
    x = radii[0] * np.cos(t)
    y = radii[1] * np.sin(t)
    z = radii[2] * np.sin(t)  # Modify this to create different shapes
    # Apply rotation
    x, y, z = np.dot(rotation, np.array([x, y, z]))
    x += center[0]
    y += center[1]
    z += center[2]
    return x, y, z


# Define parameters for the ellipse
center = [0, 0, 0]
radii = [5, 2, 10]
rotation_matrix = np.array(
    [
        [1, 0, 0],
        [0, np.cos(np.pi / 4), -np.sin(np.pi / 4)],
        [0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
    ]
)

# Generate ellipse points
x, y, z = create_3d_ellipse(center, radii, rotation_matrix)

# Create a 3D plot using Plotly
fig = go.Figure()

# Add the 3D curve
fig.add_trace(
    go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="blue", width=10))
)

# Add a surface under the curve to simulate filling
fig.add_trace(go.Mesh3d(x=x, y=y, z=z, color="lightblue", opacity=0.5))

# Update the layout to enhance visualization
fig.update_layout(
    title="3D Ellipse with Simulated Filling",
    scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
)

# Show the plot
fig.show()

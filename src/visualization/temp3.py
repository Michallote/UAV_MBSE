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


x = [0, 1, 0, 1]
z = [0, 0, 1, 1]
y = [0, 0, 0, -1]

# Create a triangle by connecting these points
i = [0, 1]
j = [1, 2]
k = [2, 3]


indices = np.array(
    [result for i in range(n) if all_different(result := [i, i + 1, n - i])]
)
i, j, k = indices.T


i[0], j[0], k[0]

fig = go.Figure()

fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=0.5))
fig.add_trace(go.Scatter3d(x=x, y=y, z=z))
fig.show()


fig = go.Figure(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=0.5))
fig.add_trace(go.Scatter3d(x=x, y=y, z=z))
fig.show()


curve = rib.curve.resample(15)
x, y, z = curve.x, curve.y, curve.z

n = len(curve) - 1
indices = np.array(
    [result for i in range(n) if all_different(result := [i, i + 1, n - i])]
)
i, j, k = indices.T

fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=0.5))
fig.add_trace(go.Scatter3d(x=x, y=y, z=z))


x = [0, 1, 0]
z = [0, 0, 1]
y = [0, 0, 0]

# Create a triangle by connecting these points
i = [0]
j = [1]
k = [14]


fig = go.Figure()

fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=0.5))
fig.add_trace(go.Scatter3d(x=x, y=y, z=z))
fig.show()
x[15]

import numpy as np
import plotly.graph_objects as go

n = 10
t = 0.2
# Generate a simple surface
x = -(np.linspace(0, 1, 4)) / n
y = -np.linspace(0, 1, 4) / n
xx, yy = np.meshgrid(x, y)
print(xx)
print(yy)
zz = -0.05 / (xx - t) - 0.05 / (yy - t)
zz = -zz + 5 * np.abs(yy) ** 0.5 + 40 * xx**2
np.cos(xx**2 + yy**2) ** 2
# Flatten indices for node numbering
node_numbers = np.arange(xx.size).reshape(xx.shape)

# Create the surface plot with node numbers
fig_surface = go.Figure()

# Add the surface
fig_surface.add_trace(
    go.Surface(
        z=zz,
        x=xx,
        y=yy,
        showscale=False,
        opacity=0.8,
        colorscale=[
            [0, "lightcoral"],
            [1, "lightblue"],  # Light blue at minimum
        ],  # Light red at maximum
    )
)

# Add the grid lines
for i in range(xx.shape[0]):
    fig_surface.add_trace(
        go.Scatter3d(x=xx[i], y=yy[i], z=zz[i], mode="lines", line=dict(color="black"))
    )
for j in range(xx.shape[1]):
    fig_surface.add_trace(
        go.Scatter3d(
            x=xx[:, j], y=yy[:, j], z=zz[:, j], mode="lines", line=dict(color="black")
        )
    )

# Add node numbers
# for i in range(xx.shape[0]):
#     for j in range(xx.shape[1]):
#         fig_surface.add_trace(
#             go.Scatter3d(
#                 x=[xx[i, j]],
#                 y=[yy[i, j]],
#                 z=[zz[i, j]],
#                 mode="text",
#                 text=[str(node_numbers[i, j])],
#                 textposition="top center",
#             )
#         )

# Add node indices
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        fig_surface.add_trace(
            go.Scatter3d(
                x=[xx[i, j]],
                y=[yy[i, j]],
                z=[zz[i, j]],
                mode="text",
                text=[f"({i}, {j})"],
                textposition="middle right",
            )
        )


fig_surface.add_trace(
    go.Scatter3d(
        x=xx.ravel(),
        y=yy.ravel(),
        z=zz.ravel(),
        mode="markers",
        marker=dict(size=3, color="black"),
    )
)

# Save surface with grid and numbers

fig_surface.update_layout(
    showlegend=False,  # Remove the legend
    font=dict(size=30),  # Increase font size
    paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
    plot_bgcolor="rgba(0,0,0,0)",
    scene=dict(  # Turn off 3D axes
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
    ),
)


# fig_surface.write_image("data/surface_with_grid_and_indices.svg")
fig_surface.show()

# # Function to create surface mesh
# def create_surface_mesh(xx, yy, zz):
#     indices = np.arange(len(xx.ravel()), dtype=int).reshape(xx.shape)
#     tl_idx = indices[:-1, :-1]  # top left vertices -> 0
#     bl_idx = indices[1:, :-1]  # bottom left vertices -> 1
#     tr_idx = indices[:-1, 1:]  # top right vertices -> 2
#     br_idx = indices[1:, 1:]  # bottom right vertices

#     x, y, z = xx.ravel(), yy.ravel(), zz.ravel()

#     # First triangle
#     i1 = tl_idx.ravel()
#     j1 = bl_idx.ravel()
#     k1 = tr_idx.ravel()

#     # Second triangle
#     i2 = bl_idx.ravel()
#     j2 = br_idx.ravel()
#     k2 = tr_idx.ravel()

#     # Combine triangles
#     i = np.concatenate([i1, i2])
#     j = np.concatenate([j1, j2])
#     k = np.concatenate([k1, k2])

#     return x, y, z, i, j, k


# # Generate mesh for the surface
# x_flat, y_flat, z_flat, i_mesh, j_mesh, k_mesh = create_surface_mesh(xx, yy, zz)

# # Create a plot for the mesh
# fig_mesh = go.Figure()

# # Add mesh triangles
# fig_mesh.add_trace(
#     go.Mesh3d(
#         x=x_flat,
#         y=y_flat,
#         z=z_flat,
#         i=i_mesh,
#         j=j_mesh,
#         k=k_mesh,
#         color="lightblue",
#         opacity=0.50,
#     )
# )

# # Add triangle numbers
# for idx, (i, j, k) in enumerate(zip(i_mesh, j_mesh, k_mesh)):
#     # Compute centroid of the triangle
#     cx = (x_flat[i] + x_flat[j] + x_flat[k]) / 3
#     cy = (y_flat[i] + y_flat[j] + y_flat[k]) / 3
#     cz = (z_flat[i] + z_flat[j] + z_flat[k]) / 3
#     fig_mesh.add_trace(
#         go.Scatter3d(
#             x=[cx],
#             y=[cy],
#             z=[cz],
#             mode="text",
#             text=[str(idx)],
#             textposition="middle center",
#         )
#     )

# # Save mesh with triangle numbers
# fig_mesh.show()
# fig_mesh.write_image("data/mesh_with_triangle_numbers.pdf")

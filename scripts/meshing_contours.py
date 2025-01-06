import matplotlib.pyplot as plt
import numpy as np
import triangle as tr


def generate_mesh(boundary_vertices):
    """
    Generates a triangular mesh for a given boundary.

    Parameters:
    - boundary_vertices: Array of vertices forming a closed boundary (Nx2 array).

    Returns:
    - Mesh dictionary containing nodes and triangles.
    """
    # Define the input for the Triangle library
    boundary = {
        "vertices": boundary_vertices,
        "segments": np.arange(len(boundary_vertices)),
    }

    # Ensure segments are closed (connect last point to the first)
    boundary["segments"] = np.column_stack(
        (
            np.arange(len(boundary_vertices)),
            np.roll(np.arange(len(boundary_vertices)), -1),
        )
    )

    # Generate the mesh using Triangle
    mesh = tr.triangulate(
        boundary, "p"
    )  # 'p' ensures triangulation respects the boundary
    return mesh


def plot_mesh(mesh):
    """
    Plots a triangular mesh using Matplotlib.

    Parameters:
    - mesh: Mesh dictionary containing nodes and triangles.
    """
    plt.figure(figsize=(8, 8))
    plt.triplot(
        mesh["vertices"][:, 0],
        mesh["vertices"][:, 1],
        mesh["triangles"],
        "-k",
        linewidth=0.5,
    )
    plt.plot(mesh["vertices"][:, 0], mesh["vertices"][:, 1], "ro", markersize=2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Generated Mesh")
    plt.gca().set_aspect("equal")
    plt.show()


# Example: Input boundary vertices
boundary_vertices = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 1.5]]
)

# Generate the mesh
mesh = generate_mesh(boundary_vertices)

# Plot the mesh
plot_mesh(mesh)

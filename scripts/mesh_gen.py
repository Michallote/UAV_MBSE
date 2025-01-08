import matplotlib.pyplot as plt
import numpy as np
import triangle
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

# Define the polygon vertices
polygon_points = np.array([[0, 0], [2, -1], [1, 0], [2, 1]])
polygon = Polygon(polygon_points)

# Ensure the polygon vertices are oriented counterclockwise (required for some algorithms)
polygon = orient(polygon, sign=1.0)

# Convert to a format compatible with `triangle`
polygon_dict = {
    "vertices": np.array(polygon.exterior.coords[:-1]),
    "segments": np.array(
        [[i, (i + 1) % len(polygon_points)] for i in range(len(polygon_points))]
    ),
}

# Perform constrained Delaunay triangulation using `triangle`
triangulated = triangle.triangulate(polygon_dict, "p")

# Plot the triangulated mesh
plt.figure(figsize=(6, 6))
for triangle_indices in triangulated["triangles"]:
    simplex = triangulated["vertices"][triangle_indices]
    plt.fill(simplex[:, 0], simplex[:, 1], edgecolor="k", alpha=0.3)

# Plot the boundary of the original polygon
plt.plot(polygon_points[:, 0], polygon_points[:, 1], "o-", color="blue")
plt.title("Constrained Delaunay Triangulated Mesh")
plt.axis("equal")
plt.savefig("delaunay.png")

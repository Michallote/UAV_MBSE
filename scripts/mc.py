import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

from aerodynamics.airfoil import Airfoil


def generate_mesh(boundary_vertices):
    """
    Generates a triangular mesh for a given boundary using Delaunay triangulation.

    Parameters:
    - boundary_vertices: Array of vertices forming a closed boundary (Nx2 array).

    Returns:
    - Triangulation object (points and simplices).
    """
    # Create a bounding box and fill interior points if necessary
    min_x, min_y = np.min(boundary_vertices, axis=0)
    max_x, max_y = np.max(boundary_vertices, axis=0)

    # Generate interior points for better mesh quality (optional)
    grid_x, grid_y = np.meshgrid(
        np.linspace(min_x, max_x, 30), np.linspace(min_y, max_y, 30)
    )
    interior_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Filter points inside the boundary using the ray-crossing algorithm
    from matplotlib.path import Path

    path = Path(boundary_vertices)
    mask = path.contains_points(interior_points)
    interior_points = interior_points[mask]

    # Combine boundary and interior points
    all_points = np.vstack([boundary_vertices, interior_points])

    # Perform Delaunay triangulation
    delaunay = Delaunay(all_points)
    return all_points, delaunay


def plot_mesh(points, delaunay, boundary_vertices):
    """
    Plots the Delaunay triangulation mesh.

    Parameters:
    - points: Array of all mesh points.
    - delaunay: Delaunay triangulation object.
    - boundary_vertices: Array of boundary vertices for highlighting.
    """
    plt.figure(figsize=(8, 8))

    # Plot the triangles
    plt.triplot(points[:, 0], points[:, 1], delaunay.simplices, "-k", linewidth=0.5)

    # Highlight boundary points
    plt.plot(
        boundary_vertices[:, 0],
        boundary_vertices[:, 1],
        "r-",
        linewidth=1.5,
        label="Boundary",
    )
    plt.scatter(points[:, 0], points[:, 1], s=5, color="blue", label="Mesh Points")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Generated Mesh using Delaunay")
    plt.legend()
    plt.gca().set_aspect("equal")
    plt.savefig("mesh.png")


airfoil = Airfoil.from_file("data/airfoils/FX 73-CL2-152 T.E..dat")

airfoil_te_gap = airfoil.with_trailing_edge_gap(te_gap=0.03, blend_distance=1.0)

# Example: Input boundary vertices
boundary_vertices = airfoil_te_gap.data

# Generate the mesh
points, delaunay = generate_mesh(boundary_vertices)

# Plot the mesh
plot_mesh(points, delaunay, boundary_vertices)

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay, Voronoi


###############################################################################
# Sutherland–Hodgman polygon clipping
###############################################################################
def sutherland_hodgman_clip(subject_polygon, clip_polygon):
    """
    Clip a polygon (subject_polygon) against another polygon (clip_polygon)
    using the Sutherland-Hodgman algorithm.

    Each polygon is a list (or array) of points [x, y], assumed to be in order
    (counter-clockwise) and closed (first point = last point). If not closed,
    this function will handle it gracefully by closing them internally.

    Returns a new polygon (list of (x, y)) that is the clipped region.
    If the subject_polygon is fully outside, it may return an empty list.
    """
    # Ensure arrays, remove last point if it duplicates the first
    subj = np.array(subject_polygon, dtype=float)
    clip = np.array(clip_polygon, dtype=float)
    if not np.allclose(subj[0], subj[-1]):
        subj = np.vstack([subj, subj[0]])
    if not np.allclose(clip[0], clip[-1]):
        clip = np.vstack([clip, clip[0]])

    def inside(p, p1, p2):
        # Check if point p is to the left of the directed edge from p1->p2
        return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0]) >= 0

    def compute_intersection(p1, p2, c1, c2):
        """
        Compute intersection between the edge p1->p2 and c1->c2.
        Using line-line intersection in 2D.
        """
        A1 = p2[1] - p1[1]
        B1 = p1[0] - p2[0]
        C1 = A1 * p1[0] + B1 * p1[1]

        A2 = c2[1] - c1[1]
        B2 = c1[0] - c2[0]
        C2 = A2 * c1[0] + B2 * c1[1]

        determinant = A1 * B2 - A2 * B1
        if abs(determinant) < 1e-14:
            # Lines are parallel or coincident; no single intersection
            return None

        x = (B2 * C1 - B1 * C2) / determinant
        y = (A1 * C2 - A2 * C1) / determinant
        return np.array([x, y])

    output_list = subj[:-1]  # drop the duplicated last point for iteration

    # For each edge of clip_polygon
    for i in range(len(clip) - 1):
        c1 = clip[i]
        c2 = clip[i + 1]
        input_list = output_list
        output_list = []
        if len(input_list) == 0:
            break

        s = input_list[-1]
        for p in input_list:
            if inside(p, c1, c2):
                if not inside(s, c1, c2):
                    # s->p crosses into clipping region
                    inter = compute_intersection(s, p, c1, c2)
                    if inter is not None:
                        output_list.append(inter)
                output_list.append(p)
            elif inside(s, c1, c2):
                # s->p crosses out of clipping region
                inter = compute_intersection(s, p, c1, c2)
                if inter is not None:
                    output_list.append(inter)
            s = p

    if len(output_list) == 0:
        return np.array([]).reshape((0, 2))

    # Close the polygon if needed
    if not np.allclose(output_list[0], output_list[-1]):
        output_list.append(output_list[0])

    return np.array(output_list)


###############################################################################
# Polygon area and centroid
###############################################################################
def polygon_area_centroid(polygon):
    """
    Compute the signed area and centroid of a polygon given as (N x 2).
    Polygon is assumed closed or we will treat the last point = first point.

    Returns (area, cx, cy).
    Area is positive if polygon is CCW, negative if CW.
    """
    if len(polygon) < 3:
        return 0.0, 0.0, 0.0

    poly = np.array(polygon)
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])

    x = poly[:, 0]
    y = poly[:, 1]
    # Shoelace formula
    a = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    if abs(a) < 1e-14:
        return 0.0, 0.0, 0.0

    cx = (1.0 / (6.0 * a)) * np.sum(
        (x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])
    )
    cy = (1.0 / (6.0 * a)) * np.sum(
        (y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])
    )
    return a, cx, cy


###############################################################################
# Convert Voronoi object to finite polygons (2D)
###############################################################################
import numpy as np
from scipy.spatial import Voronoi


def voronoi_finite_polygons_2d(vor, radius=1e6):
    """
    Reconstruct infinite Voronoi regions in a 2D diagram to finite polygons
    by 'stretching' them until they intersect a large circle of given radius.

    Parameters
    ----------
    vor : scipy.spatial.Voronoi
        Voronoi diagram object.
    radius : float
        Distance to 'stretch' infinite ridges until they hit a bounding circle.

    Returns
    -------
    new_regions : list of lists
        Each entry is a list of vertex indices (into new_vertices) that define
        a polygonal region.
    new_vertices : (M, 2) ndarray
        Coordinates of the adapted Voronoi vertices.
    """
    # Center of the Voronoi diagram
    center = vor.points.mean(axis=0)
    # Copy the original vertices to a mutable list
    new_vertices = vor.vertices.tolist()
    # This will hold the new list of regions (each region is list of vertex indices)
    new_regions = []

    # For each point i, we have a region index: vor.point_region[i].
    # That region index can be used to look up the list of vertex indices in vor.regions.
    for region_idx in vor.point_region:
        region = vor.regions[region_idx]
        # If it's empty or contains -1, it is an open (infinite) region or invalid
        if not region or any(v == -1 for v in region):
            # We'll reconstruct it from ridges
            # Collect the sets of vertices for all ridges that
            # belong to this region (i.e., the region_idx matches
            # the region of either p1 or p2).
            region_vertices = []

            # Go through each ridge and see if it belongs to this region
            for i_ridge, (p1, p2) in enumerate(vor.ridge_points):
                # region of p1, region of p2
                r1 = vor.point_region[p1]
                r2 = vor.point_region[p2]
                if region_idx in (r1, r2):
                    # This ridge belongs to the region
                    v_ridge = vor.ridge_vertices[i_ridge]
                    region_vertices.append(v_ridge)

            # Now we gather all unique vertices from these ridges
            # Some of these may be -1 (meaning infinite).
            # We'll replace those with "far" points along the direction from center.
            pts = []
            for rverts in region_vertices:
                for v in rverts:
                    if v != -1:
                        pts.append(v)

            # We'll handle infinite edges separately
            # For each ridge that contains -1, we find the finite vertex
            # and extend it outwards from the center to 'radius'.
            for rverts in region_vertices:
                if -1 in rverts:
                    # find the finite vertex
                    finite_idx = rverts[0] if rverts[1] == -1 else rverts[1]
                    # direction from center to finite vertex
                    finite_coord = np.array(new_vertices[finite_idx])
                    direction = finite_coord - center
                    direction /= np.linalg.norm(direction)
                    # new "far" point
                    far_point = finite_coord + direction * radius
                    # add to the vertex list
                    new_vertices.append(far_point.tolist())
                    far_idx = len(new_vertices) - 1
                    # store this index
                    pts.append(far_idx)

            # Remove duplicates, then sort them in CCW order to form a polygon
            pts = list(set(pts))
            # If everything is infinite or no valid vertices, skip
            if len(pts) == 0:
                new_regions.append([])
                continue

            # We must order these points in a CCW polygon
            pts_coords = np.array([new_vertices[v] for v in pts])
            c = pts_coords.mean(axis=0)
            angles = np.arctan2(pts_coords[:, 1] - c[1], pts_coords[:, 0] - c[0])
            order = np.argsort(angles)
            pts_sorted = [pts[i] for i in order]
            new_regions.append(pts_sorted)

        else:
            # Region is already finite
            new_regions.append(region)

    return new_regions, np.array(new_vertices)


###############################################################################
# Main Centroidal Voronoi Tessellation function
###############################################################################
def centroidal_voronoi_tessellation(
    boundary, element_size, n_interior=200, max_iters=50, tol=1e-3, random_seed=42
):
    """
    Perform Centroidal Voronoi Tessellation (CVT) inside a closed polygon.

    Parameters:
    -----------
    boundary : (N x 2) array
        Closed loop of boundary points (counter-clockwise).
    element_size : float
        Desired spacing for boundary resampling (assumed handled by a black-box).
        We will call `resample_contour(boundary, element_size)` to get boundary points.
    n_interior : int
        Number of random interior generators to start with.
    max_iters : int
        Maximum number of CVT iterations.
    tol : float
        Convergence tolerance based on generator movement.
    random_seed : int
        Random seed for reproducibility.

    Returns:
    --------
    final_points : (M x 2) array
        The final generator points (including boundary + interior).
    """
    np.random.seed(random_seed)

    # -------------------------------------------------------------------------
    # 1) Resample boundary (assume we have this function available)
    # -------------------------------------------------------------------------
    # For demonstration, we'll define a trivial stub here.
    # In practice, you would replace this with your actual resampling function!
    def resample_contour(bound, h):
        """
        Stub for boundary resampling.
        Replace with your actual implementation that returns
        boundary points at ~ 'h' spacing along 'bound'.
        """
        # This trivial approach just returns the input boundary
        # (assuming it's already spaced).
        return bound

    boundary_resampled = resample_contour(boundary, element_size)
    # Ensure boundary is closed
    if not np.allclose(boundary_resampled[0], boundary_resampled[-1]):
        boundary_resampled = np.vstack([boundary_resampled, boundary_resampled[0]])

    # -------------------------------------------------------------------------
    # 2) Create random interior points
    # -------------------------------------------------------------------------
    min_xy = np.min(boundary_resampled, axis=0)
    max_xy = np.max(boundary_resampled, axis=0)

    # Use matplotlib.path to test if points are inside
    from matplotlib.path import Path

    path = Path(boundary_resampled)

    interior_points = []
    i = 0
    # Generate random points in bounding box, keep only those inside
    while len(interior_points) < n_interior:
        candidate = min_xy + (max_xy - min_xy) * np.random.rand(2)
        if path.contains_points([candidate])[0]:
            interior_points.append(candidate)
        i += 1
        if i > 100000:
            # Avoid infinite loop in pathological cases
            break
    interior_points = np.array(interior_points)

    # Combine boundary (fixed) and interior (movable) points
    # We'll keep track of which are boundary vs interior
    boundary_count = len(boundary_resampled) - 1  # minus 1 because it is closed
    generators = np.vstack([boundary_resampled[:-1], interior_points])

    # Indices of boundary points that should remain fixed
    boundary_indices = np.arange(boundary_count)

    # -------------------------------------------------------------------------
    # 3) CVT Iteration
    # -------------------------------------------------------------------------
    for iteration in range(max_iters):
        vor = Voronoi(generators)
        regions, vertices = voronoi_finite_polygons_2d(vor, radius=1e5)

        # We will store the updated generator locations here
        new_generators = generators.copy()

        # For each generator, find its Voronoi region polygon
        for i_pt, region_idx in enumerate(vor.point_region):
            if i_pt in boundary_indices:
                # Boundary generator stays in place
                continue

            # The Voronoi region is given by 'regions[region_idx]'
            region_vertices_idx = regions[region_idx]
            if len(region_vertices_idx) == 0:
                continue
            poly_coords = vertices[region_vertices_idx]

            # 3.1) Clip polygon_coords with the boundary polygon
            clipped_poly = sutherland_hodgman_clip(poly_coords, boundary_resampled)
            if len(clipped_poly) < 3:
                continue

            # 3.2) Compute centroid of clipped polygon
            area, cx, cy = polygon_area_centroid(clipped_poly)
            if abs(area) < 1e-14:
                continue

            # Update generator
            new_generators[i_pt] = [cx, cy]

        # Check convergence: max distance moved by interior points
        dist_moved = np.linalg.norm(new_generators - generators, axis=1)
        max_move = np.max(dist_moved[boundary_count:])  # only interior
        generators = new_generators

        if max_move < tol:
            print(
                f"Converged at iteration {iteration+1}, max interior move = {max_move:.5f}"
            )
            break

    # Final generator set
    return generators


###############################################################################
# Example usage
###############################################################################
if __name__ == "__main__":
    # Example: define a simple "airfoil-like" closed shape
    # (Here just a crude shape for demonstration)
    airfoil_boundary = np.array(
        [
            [0.0, 0.0],
            [0.3, 0.05],
            [0.6, 0.08],
            [1.0, 0.06],
            [1.2, 0.03],
            [1.3, 0.0],
            [1.2, -0.03],
            [1.0, -0.06],
            [0.6, -0.08],
            [0.3, -0.05],
            [0.0, 0.0],  # close
        ]
    )

    # Perform Centroidal Voronoi Tessellation
    element_size = 0.05
    final_gens = centroidal_voronoi_tessellation(
        airfoil_boundary,
        element_size,
        n_interior=100,
        max_iters=30,
        tol=1e-3,
        random_seed=1,
    )

    # Visualize final result
    plt.figure(figsize=(8, 6))
    plt.plot(airfoil_boundary[:, 0], airfoil_boundary[:, 1], "r-", label="Boundary")
    plt.scatter(
        final_gens[:, 0], final_gens[:, 1], s=10, c="blue", label="CVT Generators"
    )
    plt.gca().set_aspect("equal")
    plt.title("Final CVT Generators inside an Airfoil Boundary")
    plt.legend()
    plt.savefig("cvt_airfoil.png")
    plt.show()

    # Optionally, build a triangular mesh from these points
    tri = Delaunay(final_gens)
    plt.figure(figsize=(8, 6))
    plt.triplot(final_gens[:, 0], final_gens[:, 1], tri.simplices, color="k", lw=0.5)
    plt.plot(airfoil_boundary[:, 0], airfoil_boundary[:, 1], "r-", label="Boundary")
    plt.gca().set_aspect("equal")
    plt.title("Triangulated Mesh of CVT Generators")
    plt.legend()
    plt.savefig("cvt_airfoil_mesh.png")
    plt.show()

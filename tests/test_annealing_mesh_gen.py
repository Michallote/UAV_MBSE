import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point, Polygon
from tqdm import tqdm

from geometry.interpolation import resample_curve_equidistant
from geometry.meshing import random_points_inside_curve
from src.aerodynamics.airfoil import Airfoil


def test_annealing_mesh_gen():
    """Test that the mesh created by the annealing algorithm is valid."""

    airfoil = Airfoil.from_file("data/airfoils/FX 73-CL2-152 T.E..dat")
    airfoil = airfoil.with_trailing_edge_gap(te_gap=0.03, blend_distance=1.0)

    curve = airfoil.data
    length = np.round(airfoil.trailing_edge_gap, 2)
    curve = resample_curve_equidistant(curve, length)

    n_points = np.round(airfoil.area / (0.5 * length * length * np.sin(np.pi / 3)))
    inner_points = random_points_inside_curve(curve, int(n_points))

    # Create a function that spaces out the inner points using a spring model where all nodes repel each other including the boundary nodes however the boundary nodes are fixed in place

    optimized_points = optimize_points_spring_system(
        vertices=inner_points,
        boundary=curve,
        k_repulsion=0.001,
        n_iterations=100,
        dt=0.001,
    )

    plt.figure(figsize=(8, 8))
    plt.plot(curve[:, 0], curve[:, 1], "r-")
    plt.plot(inner_points[:, 0], inner_points[:, 1], "g.", markersize=2)
    plt.plot(optimized_points[:, 0], optimized_points[:, 1], "bx", markersize=2)
    plt.savefig("optimized_points.svg")


def test_voronoi():

    airfoil = Airfoil.from_file("data/airfoils/FX 73-CL2-152 T.E..dat")
    airfoil = airfoil.with_trailing_edge_gap(te_gap=0.03, blend_distance=1.0)

    curve = airfoil.data
    length = np.round(airfoil.trailing_edge_gap, 2)
    curve = resample_curve_equidistant(curve, length)

    n_points = np.round(airfoil.area / (0.5 * length * length * np.sin(np.pi / 3)))
    inner_points = random_points_inside_curve(curve, int(n_points))

    points = np.vstack([curve, inner_points])

    vor = Voronoi(points)

    fig = voronoi_plot_2d(vor)
    plt.savefig("voronoi.png")

    mask = vertices_in_boundary(vor.vertices, boundary=curve)
    new_inner = vor.vertices[mask]
    points = np.vstack([curve, new_inner])
    vor = Voronoi(points)

    fig = voronoi_plot_2d(vor)
    plt.savefig("voronoi2.png")


def repulsion_force(p1, p2, k=1.0, min_dist=1e-6):
    """
    Compute a repulsive force exerted on p1 by p2 using a simple 1/d^2 law.

    Parameters
    ----------
    p1, p2 : array-like, shape (2,)
        Coordinates of the two points in 2D.
    k : float
        Strength constant for the repulsion.
    min_dist : float
        Minimum distance to avoid numerical explosions if p1 ~ p2.

    Returns
    -------
    force : np.ndarray, shape (2,)
        The repulsive force vector acting on p1.
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    delta = p1 - p2
    dist = np.linalg.norm(delta, axis=0)
    if dist < min_dist:
        dist = min_dist
    # Force magnitude ~ k / dist^2
    # Direction is from p2 -> p1
    F = k / dist**2
    dir_unit = delta / dist  # unit direction
    return F * dir_unit


def optimize_points_spring_system(
    vertices: np.ndarray,
    boundary: np.ndarray,
    n_iterations: int = 200,
    k_repulsion: float = 0.1,
    dt: float = 0.01,
    damping: float = 0.95,
    enforce_inside: bool = True,
    min_dist: float = 1e-6,
):
    """
    Relax interior points using a simple repulsion-based 'spring' system.
    Boundary points remain fixed but repel the interior points.

    Parameters
    ----------
    vertices : (N, 2) np.ndarray
        The interior points to be optimized (moved).
    boundary : (M, 2) np.ndarray
        The boundary (closed curve) where points are fixed in place.
        Must be a closed loop or at least define a valid polygon with Shapely.
    n_iterations : int
        Number of iterations to run the relaxation.
    k_repulsion : float
        Repulsion constant (stronger means points push each other away more).
    dt : float
        Time-step for the relaxation updates.
    damping : float
        Factor by which velocities are scaled after each iteration.
        (a value < 1.0 helps stabilize the system).
    enforce_inside : bool
        If True, a point that moves outside the polygon is reverted to its old position.

    Returns
    -------
    vertices : (N, 2) np.ndarray
        The relaxed (optimized) interior points.
    """
    # Convert the boundary into a Shapely polygon
    poly = Polygon(boundary)
    if not poly.is_valid:
        raise ValueError("Boundary is not a valid polygon.")

    # Make sure boundary is closed (in case it's not)
    if not np.allclose(boundary[0], boundary[-1]):
        boundary = np.vstack([boundary, boundary[0]])

    # Initialize velocities for interior points
    velocities = np.zeros_like(vertices)

    # Identify boundary nodes as fixed
    # (We treat them only as repellers, never as points to update.)
    boundary_fixed = boundary

    for _iteration in tqdm(range(n_iterations)):
        # Compute repulsion forces for each interior vertex
        forces = np.zeros_like(vertices)

        # 1) Repulsion among interior vertices (all-vs-all)
        #    O(N^2) loop. For large N, more advanced methods might be preferred.
        repulsion_nodes = np.vstack([vertices, boundary_fixed])

        # 2) Repulsion from boundary nodes (which are fixed)
        b_delta = np.stack([(repulsion_nodes - vertex) for vertex in vertices], axis=0)
        # print("{b_delta.shape=}")

        b_dist = np.linalg.norm(b_delta, axis=-1)
        # print("{b_dist.shape=}")
        # b_dist = np.where(b_dist < min_dist, min_dist, b_dist)
        unit_vec = b_delta / b_dist[:, :, np.newaxis]

        av_length = 0.03

        force = k_repulsion * (b_dist - av_length)

        f_boundary = force[:, :, np.newaxis] * unit_vec
        # print("{f_boundary.shape=}")
        boundary_forces = np.sum(f_boundary, axis=0)

        # Sum all forces
        forces += boundary_forces

        # Backup old positions to restore if needed
        old_positions = vertices.copy()

        # 3) Update velocities and positions
        new_damping = damping
        new_velocities = (velocities + dt * forces) * new_damping
        new_vertices = vertices + dt * new_velocities

        i = 0
        while np.mean(vertex_inside_poly(new_vertices, poly)) < 0.7 and i <= 15:
            new_damping *= 0.95
            new_velocities = (velocities + dt * forces) * new_damping
            new_vertices = vertices + dt * new_velocities
            i += 1
            # print(i)

        velocities = new_velocities
        vertices = new_vertices

        # 5) Optional: Keep points inside polygon by reverting moves if outside
        if enforce_inside:
            mask = ~vertex_inside_poly(vertices, poly)

            # Revert to old position and zero out velocity
            vertices[mask] = old_positions[mask]
            velocities[mask] = 0.0

    return vertices


def vertices_in_boundary(vertices: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Checks if vertices are inside of the curve defined in the boundary

    Parameters
    ----------
    vertices : np.ndarray
        vertices to check if inside boundary
    boundary : np.ndarray
        boundary coordinates of a closed curve

    Returns
    -------
    np.ndarray
        Boolean array of points contained in the boundary

    Raises
    ------
    ValueError
        Raised for invalid Boundaries
    """
    poly = Polygon(boundary)

    if not poly.is_valid:
        raise ValueError("Boundary is not a valid polygon.")

    return np.array([poly.contains(Point(vertex)) for vertex in ((vertices))])


def vertex_inside_poly(vertices: np.ndarray, poly: Polygon) -> np.ndarray:
    """Check if vertices are inside a polygon.

    Parameters
    ----------
    vertices : np.ndarray
        Array of vertices to check.
    poly : Polygon
        Polygon of the boundary.

    Returns
    -------
    np.ndarray
        Boolean array indicating if each vertex is inside the polygon.
    """
    return np.array([poly.contains(Point(vertex)) for vertex in ((vertices))])

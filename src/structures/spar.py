"""Module for creating and handling Spar & Main Beam strats"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Optional

import numpy as np

# Create a sliding window view of size 2
from scipy.optimize import NonlinearConstraint, differential_evolution, minimize

from src.aerodynamics.data_structures import PointMass
from src.geometry.aircraft_geometry import GeometricCurve, GeometricSurface
from src.geometry.spatial_array import SpatialArray
from src.geometry.surfaces import (
    construct_orthonormal_basis,
    create_surface_mesh,
    evaluate_surface_intersection,
    project_points_to_plane,
)
from src.geometry.transformations import get_plane_normal_vector, reflect_curve_by_plane
from src.materials import Material
from src.structures.inertia_tensor import (
    compute_inertia_tensor_of_shell,
    triangulate_mesh,
)
from src.utils.interpolation import resample_curve
from src.utils.intersection import (
    calculate_intersection_curve,
    enforce_closed_curve,
    generate_intersection_registry,
    offset_curve,
)


class UnconvergedException(Exception):
    """Represents a convergence problem in an optimization method"""


class StructuralSpar:
    """
    Represents a wing spar, defined by its cross-sectional area and centroid.
    """

    spar: SparStrategy

    def __init__(
        self,
        surface: GeometricSurface,
        strategy: SparStrategy,
        material: Material,
        thickness: float,
        **kwargs,
    ) -> None:
        self.surface = surface
        self._strategy = strategy
        self.material = material

    def _create_main_spar(self, surface, thickness, **kwargs):
        strategy = self._strategy
        self.spar = strategy.create_main_spar(surface, thickness, **kwargs)

    def _create_spar(self, surface, thickness, **kwargs):
        strategy = self._strategy
        self.spar = strategy.create_spar(surface, thickness, **kwargs)

    @property
    def mass(self) -> PointMass:
        """Mass of the spar"""
        volume = self.spar.volume
        density = self.material.density
        return PointMass(volume * density, coordinates=self.centroid, tag="Spar")

    @property
    def centroid(self) -> SpatialArray:
        """Centroid of the spar"""
        return self.spar.centroid

    @staticmethod
    def create_main_spar(
        surface: GeometricSurface,
        strategy: SparStrategy,
        material: Material,
        thickness: float,
        **kwargs,
    ) -> StructuralSpar:
        instance = StructuralSpar(surface, strategy, material, thickness)
        instance._create_main_spar(surface, thickness, **kwargs)
        return instance

    @staticmethod
    def create_spar(
        surface: GeometricSurface,
        strategy: SparStrategy,
        material: Material,
        thickness: float,
        **kwargs,
    ) -> StructuralSpar:
        instance = StructuralSpar(surface, strategy, material, thickness)
        instance._create_spar(surface, thickness, **kwargs)
        return instance

    @property
    def mesh(self) -> tuple[np.ndarray, ...]:
        """Returns the coordinates neccessary to plot the object as a mesh object

        - x, y, z are the coordinates of points in the mesh
        - i, j, k are the indices of points that comprise each indiviual triangle
        """

        x, y, z, i, j, k = self.spar.mesh

        return x, y, z, i, j, k

    def inertia(self, origin: np.ndarray) -> np.ndarray:
        x, y, z, i, j, k = self.mesh

        triangles = triangulate_mesh(x, y, z, i, j, k)

        triangles = triangles - origin

        return compute_inertia_tensor_of_shell(
            triangles, thickness=self.thickness, density=self.material.density
        )

    @property
    def thickness(self) -> float:
        return self.spar.thickness

    def mirror(self, mirror_plane: Literal["xy", "xz", "yz"] = "xy") -> StructuralSpar:

        self.spar
        mirrored_spar_struct = StructuralSpar(
            surface=self.surface,
            strategy=self._strategy,
            material=self.material,
            thickness=self.thickness,
        )
        mirrored_spar_struct.spar = self.spar.mirror(mirror_plane)
        return mirrored_spar_struct


class SparStrategy(ABC):
    """Represents different spar construction methods"""

    thickness: float

    @abstractmethod
    def __init__(self):
        """Initializes the geometry implementation"""

    @staticmethod
    @abstractmethod
    def create_main_spar(surface: GeometricSurface, thickness: float) -> SparStrategy:
        """Creates a main spar choosing the optimal position for that type of spar.
        Usually by maximizing and objective function."""

    @staticmethod
    @abstractmethod
    def create_spar(*args, **kwargs) -> SparStrategy:
        """Creates a main spar choosing the optimal position for that type of spar.
        Usually by maximizing and objective function."""

    @property
    @abstractmethod
    def volume(self) -> float:
        """Returns the volume of the spar"""

    @property
    @abstractmethod
    def centroid(self) -> SpatialArray:
        """Returns the centroid of the spar volume"""

    @property
    @abstractmethod
    def mesh(self) -> tuple[np.ndarray, ...]:
        """Returns the 3D mesh of the spar."""

    @abstractmethod
    def mirror(self, mirror_plane: Literal["xy", "xz", "yz"]) -> SparStrategy:
        """Returns the mirrored geometry of the Spar"""


class FlatSpar(SparStrategy):
    """Creates a Flat Spar intersecting the lifting surface geometry."""

    def __init__(self, curve: GeometricCurve, thickness: float) -> None:
        self.curve = curve
        self.thickness = thickness

    @staticmethod
    def create_spar(
        surface: GeometricSurface,
        thickness: float,
        p: Optional[np.ndarray] = None,
        n: np.ndarray = np.array([1, 0, 0]),
        chord_position: Optional[float] = 0.75,
    ) -> FlatSpar:
        """Creates the main spar by optimizing th position of the spar as to maximize surface area

        Parameters
        ----------
        surface : GeometricSurface
            _description_

        Returns
        -------
        GeometricCurve
            _description_
        """
        if chord_position is not None:
            root = surface.curves[0]
            chord = root.trailing_edge - root.leading_edge
            p = root.leading_edge + chord_position * chord

        curve = FlatSpar.curve_from_surface_and_plane(surface, p, n)
        return FlatSpar(curve=curve, thickness=thickness)

    @staticmethod
    def create_main_spar(surface: GeometricSurface, thickness: float) -> FlatSpar:
        """Creates the main spar by optimizing th position of the spar as to maximize surface area

        Parameters
        ----------
        surface : GeometricSurface
            _description_

        Returns
        -------
        GeometricCurve
            _description_
        """
        x_optimum = FlatSpar.find_maximum_area(surface)
        p = np.array([x_optimum, 0, 0])
        n = np.array([1, 0, 0])
        curve = FlatSpar.curve_from_surface_and_plane(surface, p, n)

        return FlatSpar(curve=curve, thickness=thickness)

    @staticmethod
    def find_maximum_area(surface) -> tuple:
        """
        Finds the x value that maximizes the area of the spar by using optimization techniques.

        Args:
            surface: The surface object with attributes xx, yy, and zz representing the surface coordinates.

        Returns:
            The optimal x value and the corresponding maximum area.
        """
        min_x, max_x = np.min(surface.xx), np.max(surface.xx)
        length = max_x - min_x

        min_x = min_x + 0.025 * length
        max_x = max_x - 0.025 * length
        # Bounds for x as per the problem statement
        bounds = [(min_x, max_x)]
        # Initial guess for x
        initial_guess = np.array([min_x + max_x]) / 2

        def calculate_area(x: np.ndarray[float, Any], surface) -> float:
            """
            Helper function to calculate the negative area for a given x to facilitate maximization.

            Args:
                x: The x position to evaluate the area at.
                surface: The surface object with xx, yy, zz attributes.

            Returns:
                The negative of the calculated area to allow for maximization using a minimization function.
            """
            p = np.array([x[0], 0, 0])
            n = np.array([1, 0, 0])

            spar = FlatSpar.curve_from_surface_and_plane(surface, p=p, n=n)

            return -spar.area

        result = minimize(
            calculate_area,
            initial_guess,
            args=(surface,),
            bounds=bounds,
            method="Powell",
        )

        if not result.success:
            raise ValueError("Optimization did not converge")

        optimal_x = result.x[0]
        # max_area = -result.fun  # Converting back to positive as we minimized the negative
        return optimal_x

    @staticmethod
    def curve_from_surface_and_plane(
        surface: GeometricSurface, p: np.ndarray, n: np.ndarray
    ) -> GeometricCurve:
        """Creates a Geometric Curve by intersecting a surface with a plane
        defined by a point in space and a normal vector

        Parameters
        ----------
        surface : GeometricSurface
            Surface delimiting the spar
        p : np.ndarray
            Position of a point on the plane
        n : np.ndarray
            Normal vector of the plane

        Returns
        -------
        GeometricCurve
            spar from the intersection
        """
        xx, yy, zz = surface.xx, surface.yy, surface.zz

        # Splitting the surface by average leading edge index
        # so it's easy to order into a closed curve

        le_index = round(np.mean([curve.airfoil.index_le for curve in surface.curves]))  # type: ignore

        curve_1 = SpatialArray(
            evaluate_surface_intersection(
                xx[:, :le_index], yy[:, :le_index], zz[:, :le_index], p, n
            )
        )

        curve_2 = SpatialArray(
            evaluate_surface_intersection(
                xx[:, le_index:], yy[:, le_index:], zz[:, le_index:], p, n
            )
        )

        data = np.vstack([curve_1, np.flip(curve_2, axis=0)])

        # Eliminate repeated points to avoid triangulation errors (negative volume triangles)
        # Finding unique rows and preserving order
        _, idx = np.unique(data, axis=0, return_index=True)
        unique_ordered_data = data[np.sort(idx)]

        curve = GeometricCurve(name="Spar", data=unique_ordered_data)

        # Improve curve resolution
        n_original = len(curve)
        n_seg = n_original - 1
        resolution_level = 2

        n_samples = n_original + n_seg * resolution_level
        curve = curve.resample(n_samples)

        return curve

    @property
    def volume(self) -> float:
        return self.curve.area * self.thickness

    @property
    def centroid(self) -> SpatialArray:
        return self.curve.centroid

    @property
    def mesh(self) -> tuple[np.ndarray, ...]:
        """Returns the 2D mesh of the spar."""

        curve = self.curve
        x, y, z = curve.x, curve.y, curve.z
        indices = curve.triangulation_indices()
        i, j, k = indices.T

        return x, y, z, i, j, k

    def mirror(
        self, mirror_plane: Literal["xy"] | Literal["xz"] | Literal["yz"] = "xy"
    ) -> FlatSpar:
        """Creates a mirrored version of the Spar's Geometry

        Parameters
        ----------
         - mirror_plane : Literal[&#39;xy&#39;] | Literal[&#39;xz&#39;] | Literal[&#39;yz&#39;]
                plane about to make the reflection

        Returns
        -------
        SparStrategy
            Mirrored Spar Strategy
        """
        mirrored_curve = self.curve.mirror(mirror_plane)

        return FlatSpar(curve=mirrored_curve, thickness=self.thickness)


class TorsionBoxSpar(SparStrategy):
    """Creates a Torsion Box intersecting the lifting surface geometry."""

    def __init__(
        self,
        contour: GeometricCurve,
        thickness: float,
        origin: SpatialArray,
        basis: np.ndarray,
        length: float,
        mesh_resolution: int = 7,
    ) -> None:
        self.contour = contour
        self.thickness = thickness
        self.origin = origin
        self.basis = basis
        self.length = length
        self.mesh_resolution = mesh_resolution

        inner_contour = offset_curve(contour.data, -thickness)
        self.inner_contour = GeometricCurve(name="Inner Contour", data=inner_contour)

    @staticmethod
    def create_spar(
        surface: GeometricSurface,
        thickness: float,
        p: np.ndarray,
        n: np.ndarray,
        width: float,
        height: float,
        length: float,
    ) -> TorsionBoxSpar:
        """Creates the main spar by optimizing the position of the spar as to maximize surface area

        Parameters
        ----------
        surface : GeometricSurface
            _description_

        Returns
        -------
        GeometricCurve
            _description_
        """

        x, y = project_points_to_plane(np.c_[*p], p, n).flatten()

        orthonormal_basis = construct_orthonormal_basis(n)

        box_contour = np.array(
            [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
        )

        contour = GeometricCurve(
            name="Spar Cross Section", data=enforce_closed_curve(box_contour)
        )

        return TorsionBoxSpar(
            contour,
            thickness,
            origin=SpatialArray(p),
            basis=orthonormal_basis,
            length=length,
        )

    @staticmethod
    def create_main_spar(surface: GeometricSurface, thickness: float) -> TorsionBoxSpar:
        """Creates the main spar by finding the optimal torsion box that passes through all wing sections.

        Parameters
        ----------
        surface : GeometricSurface
            The surface that houses the spar
        thickness : float
            Material thickness

        Returns
        -------
        TorsionBoxSpar
            TorsionBox Spar
        """

        root_curve = surface.curves[0]
        plane_point = root_curve.leading_edge
        plane_normal = root_curve.normal

        curves_data = [curve.data for curve in surface.curves]

        curve = find_intersection_region(curves_data, plane_point, plane_normal)

        x1_optimum, x2_optimum = TorsionBoxSpar.find_maximum_moment_of_inertia(
            curve, thickness
        )

        x, y, width, height = get_embedded_rectangle(x1_optimum, x2_optimum, curve)
        # Create the counter clockwise box contour
        box_contour = np.array(
            [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
        )

        local_origin = SpatialArray(np.mean(box_contour, axis=0))
        box_contour = box_contour - local_origin

        orthonormal_basis = construct_orthonormal_basis(plane_normal)
        u, v, w = orthonormal_basis

        global_origin = u * local_origin.x + v * local_origin.y + plane_point

        length = np.dot(
            surface.curves[-1].leading_edge - surface.curves[0].leading_edge, w
        )

        contour = GeometricCurve(
            name="Spar Cross Section", data=enforce_closed_curve(box_contour)
        )

        return TorsionBoxSpar(
            contour,
            thickness,
            origin=SpatialArray(global_origin),
            basis=orthonormal_basis,
            length=length,
        )

    @staticmethod
    def find_maximum_moment_of_inertia(
        curve: np.ndarray, thickness: float, method="brute-force"
    ) -> np.ndarray:
        """
        Optimizes the moment of inertia for a given surface and thickness.

        Parameters:
        - surface: GeometricSurface - The surface to optimize on.
        - thickness: float - Thickness of the material.
        - method: str - Optimization method, either 'brute-force' or 'scipy-optimize'.

        Returns:
        NumPy array containing optimized parameters.
        """

        x_max, y_max, x_min, y_min, delta_x, delta_y = find_domain_limits(curve)
        print(
            f"Finding torsion box in region: { x_max, y_max, x_min, y_min, delta_x, delta_y = }"
        )
        # for minimize use: initial_guess = [x_mean - 0.125 * delta_x, x_mean + 0.125 * delta_x]
        margin = 0.05
        x_bounds = (x_min + margin * delta_x, x_max - margin * delta_x)
        bounds = [x_bounds, x_bounds]

        if method == "brute-force":
            optimal_params = brute_force_search(curve, thickness, x_bounds)
        elif method == "scipy-optimize":
            optimal_params = scipy_optimize(curve, thickness, bounds)
        else:
            raise ValueError("Invalid optimization method specified.")

        return optimal_params

    @property
    def volume(self) -> float:
        return (self.contour.area - self.inner_contour.area) * self.length

    @property
    def centroid(self) -> SpatialArray:

        u, v, w = self.basis
        return SpatialArray(self.origin + w * self.length)

    @property
    def mesh(self) -> tuple[np.ndarray, ...]:
        """Returns the 2D mesh of the spar."""
        xx, yy, zz = self.create_surface()
        x, y, z, i, j, k = create_surface_mesh(xx, yy, zz)
        return x, y, z, i, j, k

    def create_surface(self):
        """Creates the surface from the curves defined in the contour.

        Returns
        -------
        _type_
            _description_
        """

        n = self.mesh_resolution

        origin = self.origin
        length = self.length

        u, v, w = self.basis
        contour = self.contour

        contour_3d = np.c_[contour.x] * u + np.c_[contour.y] * v + origin
        tip_contour = contour_3d + length * w
        curves = np.array([contour_3d, tip_contour])
        curves = resample_curve(curves, n)

        geo_curves = [
            GeometricCurve(name=f"TorsionBox_Section_{i}", data=data)
            for i, data in enumerate(curves)
        ]

        xx = np.array([curve.x for curve in geo_curves])
        yy = np.array([curve.y for curve in geo_curves])
        zz = np.array([curve.z for curve in geo_curves])

        return xx, yy, zz

    def mirror(
        self, mirror_plane: Literal["xy"] | Literal["xz"] | Literal["yz"] = "xy"
    ) -> TorsionBoxSpar:
        """Creates a mirrored version of the Spar's Geometry

        Parameters
        ----------
         - mirror_plane : Literal[&#39;xy&#39;] | Literal[&#39;xz&#39;] | Literal[&#39;yz&#39;]
                plane about to make the reflection

        Returns
        -------
        SparStrategy
            Mirrored Spar Strategy
        """

        contour = self.contour
        thickness = self.thickness
        origin = self.origin
        basis = self.basis
        length = self.length
        mesh_resolution = self.mesh_resolution

        mirror_normal = get_plane_normal_vector(mirror_plane)

        data = np.vstack([origin, basis])
        data = reflect_curve_by_plane(data, normal_vector=mirror_normal)
        origin, basis = data[0], data[1:]

        return TorsionBoxSpar(
            contour=contour,
            thickness=thickness,
            origin=origin,
            basis=basis,
            length=length,
            mesh_resolution=mesh_resolution,
        )


def find_domain_limits(
    curve: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Finds domain limits, returns max and min values for
    x and y as well as the deltas.

    Parameters:
    curve: np.ndarray coordinates of points. Must be a 2D array.
    """
    x_max, y_max = np.max(curve, axis=0)
    x_min, y_min = np.min(curve, axis=0)

    delta_x = x_max - x_min
    delta_y = y_max - y_min

    return x_max, y_max, x_min, y_min, delta_x, delta_y


def find_intersection_region(
    curves: list[np.ndarray], plane_point: np.ndarray, plane_normal: np.ndarray
) -> np.ndarray:
    """Calculates the intersecting region to a set of 3D parallel curves
    by projecting them to the plane

    Parameters
    ----------
    curves : list[np.ndarray]
        List of coordinates of curves in space to be intersected into a plane
    plane_point : np.ndarray
        Point coordinate on the plane
    plane_normal : np.ndarray
        Plane normal vector.


    Returns
    -------
    np.ndarray
        Intersection region
    """

    root_curve = curves.pop(0)

    intersection = project_points_to_plane(root_curve, plane_point, plane_normal)

    for i, curve in enumerate(curves):
        # print(i + 1, curve.name)

        projected_data = project_points_to_plane(curve, plane_point, plane_normal)
        intersection = enforce_closed_curve(intersection)
        projected_data = enforce_closed_curve(projected_data)
        intersection = calculate_intersection_curve(
            intersection, projected_data, radius=0.000001
        )

    return intersection


def brute_force_search(
    curve: np.ndarray, thickness: float, x_bounds: tuple[float, float]
) -> np.ndarray:
    """Optimizes the objective function by performing a brute force search on the
    1D domains of the variables

    Parameters
    ----------
    curve : np.ndarray
        curve where the rectangle is fitted
    thickness : float
        thickness of the inner wall for area moment of inertia calculations
    x_bounds : tuple[float, float]
        range for independent variables

    Returns
    -------
    np.ndarray
        x_1 and x_2 optimal values.
    """
    # Implementation of brute force search
    x1_opt, x2_opt = x_bounds
    for i in range(3):
        x_space = np.linspace(x1_opt, x2_opt, 15 + i * 5)
        xx1, xx2 = np.meshgrid(x_space, x_space)
        mask = xx1 < xx2
        metric = np.zeros_like(xx1)
        param_pairs = np.vstack([xx1[mask], xx2[mask]]).T
        evals = np.array(
            [objective(params, curve, thickness) for params in param_pairs]
        )
        metric[mask] = evals
        # Find the index of the minimum value in the flattened array
        sorted_index = np.argsort(metric.flatten())

        mask_best = np.unravel_index(sorted_index, metric.shape)

        best_3_regions = np.hstack([xx1[mask_best][:3], xx2[mask_best][:3]])

        x1_opt, x2_opt = np.min(best_3_regions), np.max(best_3_regions)

    min_index_flat = np.argmin(metric)
    # Convert this index to two-dimensional indices
    x_opt = np.unravel_index(min_index_flat, metric.shape)
    x1_opt = xx1[x_opt]
    x2_opt = xx2[x_opt]
    optimized_params = np.array([x1_opt, x2_opt])

    return optimized_params


def scipy_optimize(
    curve: np.ndarray,
    thickness: float,
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    curve : np.ndarray
        curve where the rectangle is fitted
    thickness : float
        thickness of the inner wall for area moment of inertia calculations
    bounds : list[tuple[float, float]]
        ranges for optimization of independent variables

    Returns
    -------
    np.ndarray
        optimized parameters

    Raises
    ------
    UnconvergedException
        The result did not converge to an answer
    """

    nlc = NonlinearConstraint(lambda x: x[0] - x[1], -np.inf, 6 * thickness)

    # Implementation of scipy optimization
    solution = differential_evolution(
        safe_objective,
        bounds=bounds,
        args=(curve, thickness),
        strategy="best1bin",
        maxiter=1000,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=None,
        callback=None,
        disp=False,
        polish=True,
        init="latinhypercube",
        atol=0,
        constraints=nlc,
    )
    if solution.success:
        optimized_params = solution.x
        return optimized_params
    else:
        raise UnconvergedException(
            "Optimization failed. Verify that the wing geometry is suitable for this method"
        )


def get_embedded_rectangle(x1, x2, curve) -> tuple[float, float, float, float]:
    """Find the maximum embedded rectangle in curve"""

    x_max, y_max, x_min, y_min, delta_x, delta_y = find_domain_limits(curve)

    if x1 > x2:
        x1, x2 = x2, x1

    box_envelope = np.array(
        [
            [x1, y_min - delta_y],
            [x1, y_max + delta_y],
            [x2, y_max + delta_y],
            [x2, y_min - delta_y],
        ]
    )

    registry = generate_intersection_registry(
        box_envelope,
        curve,
    )

    intersections = registry.get_intersections("a0") + registry.get_intersections("a2")
    # intersections is a list of dictionaries
    sorted_intersections = sorted(
        intersections,
        key=lambda x: (x["intersection_point"][0], x["intersection_point"][1]),
    )

    # Determine the corners of the torsion box
    box_corners = np.array(
        [
            intersect_data["intersection_point"]
            for intersect_data in sorted_intersections
        ]
    )

    bc_x, bc_y = np.max(box_corners[[0, 2]], axis=0)  # bottom corner
    tc_x, tc_y = np.min(box_corners[[1, 3]], axis=0)  # top corner

    bottom_corner = np.array([x1, bc_y])
    top_corner = np.array([x2, tc_y])
    # Now, sorted_intersections is sorted first by the x-coordinate and then by t

    # Plotting for visualization
    x, y = bottom_corner
    width, height = top_corner - bottom_corner

    return x, y, width, height


def second_moment_of_inertia(width, height):
    """Calculate the second moment of inertia of the rectangle."""
    return np.array([(1 / 12) * width * height**3, (1 / 12) * height * width**3])


def objective(params, curve, thickness):
    """Objective function to maximize the second moment of inertia."""
    weights = np.array([1.0, 0.0, 0.0])
    x1, x2 = params
    x, y, width, height = get_embedded_rectangle(x1, x2, curve)

    b = width - 2 * thickness
    h = height - 2 * thickness

    Ixx, Iyy = second_moment_of_inertia(width, height) - second_moment_of_inertia(b, h)
    area = height * width - b * h

    if np.isclose(area, 0):
        return 0

    J0 = Ixx + Iyy

    metric = weights[0] * Ixx + weights[1] * Iyy + weights[2] * J0
    metric = metric / area

    return -metric * 10e6  # Negative because we want to maximize


def safe_objective(params, *args):
    params = np.clip(params, [x_bounds[0]], [x_bounds[1]])
    return objective(params, *args)

"""Module for creating and handling Spar & Main Beam strats"""

from __future__ import annotations

from abc import ABC
from collections import deque

import matplotlib.path as mpath
import numpy as np

# Create a sliding window view of size 2
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import minimize

from src.aerodynamics.airfoil import slice_shift
from src.aerodynamics.data_structures import PointMass
from src.geometry.aircraft_geometry import GeometricCurve, GeometricSurface
from src.geometry.spatial_array import SpatialArray
from src.geometry.surfaces import evaluate_surface_intersection, project_points_to_plane
from src.structures.structural_model import Material
from src.utils.intersection import calculate_intersection_curve, enforce_closed_curve
from src.utils.transformations import rotation_matrix2d


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
        self.spar = strategy.create_main_spar(surface, thickness, **kwargs)

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


class SparStrategy(ABC):
    @staticmethod
    def create_main_spar(surface: GeometricSurface, thickness: float) -> SparStrategy:
        pass

    @property
    def volume(self) -> float:
        pass

    @property
    def centroid(self) -> SpatialArray:
        pass


class FlatSpar(SparStrategy):
    """Creates a Flat Spar intersecting the lifting surface geometry."""

    def __init__(self, curve: GeometricCurve, thickness: float) -> None:
        self.curve = curve
        self.thickness = thickness

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

        def calculate_area(x: float, surface) -> float:
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
        curve = GeometricCurve(name="Spar", data=data)

        return curve

    @property
    def volume(self) -> float:
        return self.curve.area * self.thickness

    @property
    def centroid(self) -> SpatialArray:
        return self.curve.centroid


class TorsionBoxSpar(SparStrategy):

    @staticmethod
    def create_main_spar(surface: GeometricSurface, thickness: float) -> TorsionBoxSpar:

        p_optimum = TorsionBoxSpar.find_maximum_moment_of_inertia(surface)

    @staticmethod
    def find_maximum_moment_of_inertia(surface: GeometricSurface, thickess: float):

        curve = surface.curves[0]
        curve_2 = surface.curves[3]

        tangent_vectors = np.diff(curve.data, axis=0)

        plane_point = curve.leading_edge
        plane_normal = curve.normal

        import plotly.express as px

        pdata = project_points_to_plane(curve.data, plane_point, plane_normal)
        pdata_2 = project_points_to_plane(curve_2.data, plane_point, plane_normal)

        x, y = pdata[:, 0], pdata[:, 1]

        import plotly.figure_factory as ff
        import plotly.graph_objects as go

        tangent_vectors = np.diff(pdata, axis=0)

        u, v = tangent_vectors[:, 0], tangent_vectors[:, 1]

        rotmat = rotation_matrix2d(theta=np.radians(-90))

        normals = np.dot(tangent_vectors, rotmat.T)
        scale = np.median(np.linalg.norm(normals, axis=1))
        normals = normals / np.linalg.norm(normals, axis=1).reshape(-1, 1)
        un, vn = normals[:, 0], normals[:, 1]

        # Computing the SDF a line adapted from Inigo Quilez blog
        point = np.array([0.3, 0.05])
        xi, xf = slice_shift(pdata)

        p_a = point - xf

        h = np.einsum("ij,ij->i", tangent_vectors, p_a)
        h = h / np.linalg.norm(tangent_vectors, axis=1) ** 2

        np.max(h)

        # Create a path from the curve points
        path = mpath.Path(pdata)
        # Check if the point lies inside the curve
        inside = path.contains_points(pdata_2)  # type: ignore

        path_2 = mpath.Path(pdata_2)
        inside_2 = path_2.contains_points(pdata)  # type: ignore

        pdata_3 = np.vstack([pdata_2[inside], pdata[inside_2]])
        x, y = pdata_3[:, 0], pdata_3[:, 1]

        tangent_vectors = np.diff(pdata_3, axis=0)

        u, v = tangent_vectors[:, 0], tangent_vectors[:, 1]

        # Perform the dot product between elements in the k axis.
        # sdf_plane = np.einsum("ij,k->ij", q, n)
        # Create quiver figure
        fig_1 = ff.create_quiver(
            x[:-1],
            y[:-1],
            u,
            v,
            scale=1,
            arrow_scale=0.3,
            name="Tangents",
            line=dict(width=1, color="red"),
        )

        # Add points to figure
        fig_1.add_trace(
            go.Scatter(x=x, y=y, mode="markers", marker_size=4, name="Intersection")
        )
        fig_1.add_trace(
            go.Scatter(
                x=pdata_2[:, 0],
                y=pdata_2[:, 1],
                mode="markers",
                marker_size=4,
                name="Tip",
            )
        )
        fig_1.add_trace(
            go.Scatter(
                x=pdata[:, 0],
                y=pdata[:, 1],
                mode="markers",
                marker_size=4,
                name="Root",
            )
        )

        quiver_fig = ff.create_quiver(
            x=pdata[:-1, 0],  # x-coordinates of the arrow locations
            y=pdata[:-1, 1],  # y-coordinates of the arrow locations
            u=un,  # x components of the normal vectors
            v=vn,  # y components of the normal vectors
            scale=scale,
            arrow_scale=0.3,
            name="Normals",
            line=dict(width=1, color="green"),
        )

        # Extract the Scatter trace containing the quiver arrows from the quiver_fig
        quiver_data = quiver_fig.data

        # Add each trace from the quiver plot to the existing figure
        for trace in quiver_data:
            fig_1.add_trace(trace)

        fig_1.update_layout(scene=dict(aspectmode="data"))

        fig_1.update_xaxes(constrain="domain")
        fig_1.update_yaxes(scaleanchor="x")

        # Show the updated figure with the added normal vectors quiver plot
        fig_1.show()


def find_instersection_region(surface):

    i_list = []

    root_curve = surface.curves[0]
    plane_point = root_curve.leading_edge
    plane_normal = root_curve.normal

    intersection = project_points_to_plane(root_curve.data, plane_point, plane_normal)

    for i, curve in enumerate(surface.curves[1:]):
        print(i + 1, curve.name)

        projected_data = project_points_to_plane(curve.data, plane_point, plane_normal)
        intersection = enforce_closed_curve(intersection)
        projected_data = enforce_closed_curve(projected_data)
        curve1, curve2 = intersection, projected_data
        intersection = calculate_intersection_curve(
            intersection, projected_data, radius=0.000001
        )
        i_list.append(intersection)

    intersection = GeometricCurve(name="Intersection Region", data=intersection)

    import plotly.graph_objects as go

    fig = go.Figure()

    for curve in surface.curves:

        projected_data = project_points_to_plane(curve.data, plane_point, plane_normal)
        projected_data = GeometricCurve(name="", data=projected_data)

        fig.add_trace(
            go.Scatter(
                x=projected_data.x,
                y=projected_data.y,
                mode="lines+markers",
                marker_size=4,
                name=curve.name,
            )
        )

    for i, intersection in enumerate(i_list):
        intersection = GeometricCurve(name="", data=intersection)
        fig.add_trace(
            go.Scatter(
                x=intersection.x,
                y=intersection.y,
                mode="lines+markers",
                marker_size=4,
                name=f"intersection_{i}",
            )
        )

    fig.show()

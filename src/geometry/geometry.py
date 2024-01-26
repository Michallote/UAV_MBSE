"""Geometry Module"""
from __future__ import annotations

import os
from itertools import chain
from os.path import join, normpath

import numpy as np

from src.aerodynamics.airfoil import Airfoil
from src.aerodynamics.data_structures import AeroSurface, Aircraft, Section, SurfaceType
from src.geometry.spatial_array import SpatialArray
from src.utils.interpolation import resample_curve
from src.utils.transformations import (
    get_ref_coordinate_system,
    transform_coordinates,
    transform_to_global_coordinate_system,
)


class GeometricCurve:
    """Represents a parametric curve with data as lists of the x, y, and z coordinates"""

    name: str
    data: np.ndarray
    airfoil: Airfoil | None

    def __init__(
        self, name: str, data: np.ndarray, airfoil: Airfoil | None = None
    ) -> None:
        self.name = name
        self.data = data
        self.airfoil = airfoil

    @property
    def x(self) -> np.ndarray:
        """Returns the numpy array of x-values of the data coordinates"""
        return self.data[:, 0]

    @property
    def y(self) -> np.ndarray:
        """Returns the numpy array of y-values of the data coordinates"""
        return self.data[:, 1]

    @property
    def z(self) -> np.ndarray:
        """Returns the numpy array of z-values of the data coordinates"""
        return self.data[:, 2]

    @property
    def normal(self) -> SpatialArray:
        """
        Calculate the average normal vector for a 3D curve

        Parameters:
        curve (np.ndarray): Numpy array of shape (n, 3) representing the curve with x, y, z coordinates.

        Returns:
        np.ndarray: Array of normal vectors for each point on the curve.
        """

        # Calculate tangent vectors as differences between successive points
        tangents = np.diff(self.data, axis=0)
        # Shift arrays to compute cross product pairs
        xi, xf = tangents[:-1], tangents[1:]
        normals = np.cross(xi, xf)
        # Normalize normal vectors and compute mean
        normal = np.mean(normals / np.linalg.norm(normals, axis=1), axis=0)

        return SpatialArray(normal)

    @property
    def gamma(self) -> float:
        """Angle of rotation along the y-axis."""
        return np.arccos(self.normal.z)  # type: ignore

    def to_gcs(self, reference_system: str = "SW", units: str = "mm"):
        """
        GCS = Global Coordinate System
        Returns the coordinates as a Numpy array.
        """
        units_factor, output_reference_system, reflect_axis = get_ref_coordinate_system(
            reference_system, units
        )
        data = units_factor * self.data

        factors = np.array([reflect_axis[col] for col in ["x", "y", "z"]])
        # Multiply the data array with the factors
        reflected_data = data * factors

        # Create a mapping from axis names to indices
        axis_to_index = {axis: index for index, axis in enumerate(["x", "y", "z"])}

        # Translate new axis order to indices
        new_order_indices = [axis_to_index[axis] for axis in output_reference_system]

        return reflected_data[:, new_order_indices]

    def resample(self, n_samples: int) -> GeometricCurve:
        """Resample the curve to have n_samples

        Parameters
        ----------
        n_samples : int
            The number of points the new geometric curve will have.

        Returns
        -------
        GeometricCurve
            A resampled GeometricCurve
        """
        coordinates = resample_curve(self.data, n_samples)
        if self.airfoil is not None:
            airfoil = self.airfoil.resample(n_samples)
            return GeometricCurve(self.name, coordinates, airfoil=airfoil)
        else:
            return GeometricCurve(self.name, coordinates)

    @property
    def area(self):
        """
        Stokes theorem used for computing the area through a parameterized
        line integral of a 3D curve.

        """
        x = self.x
        y = self.y
        z = self.z
        gamma = self.gamma

        roll = lambda x: (x, np.roll(x, -1, axis=0))

        xi, xf = roll(x)
        yi, yf = roll(y)
        zi, zf = roll(z)

        return (0.5) * (
            (xf + xi) * ((yf - yi) * (np.cos(gamma)) + (zf - zi) * (np.sin(gamma)))
        )

    @property
    def leading_edge(self) -> SpatialArray:
        """Retrieves the leading edge of the airfoil as a SpatialArray.

        Returns
        -------
        SpatialArray
            leading edge coordinates

        Raises
        ------
        AttributeError
            airfoil is not set for the curve i.e. is a border guiding curve
        """

        try:
            leading_edge_index = self.airfoil.index_le  # type: ignore
            return SpatialArray(self.data[leading_edge_index])
        except AttributeError as exc:
            raise AttributeError(
                "The airfoil attribute must be set before accessing the leading edge."
            ) from exc

    @property
    def trailing_edge(self) -> SpatialArray:
        """Retrieves the upper surface trailing edge of the airfoil as a SpatialArray.

        Returns
        -------
        SpatialArray
            trailing edge coordinates

        Raises
        ------
        AttributeError
            airfoil is not set for the curve i.e. is a border guiding curve
        """
        try:
            trailing_edge_index = self.airfoil.index_te  # type: ignore
            return SpatialArray(self.data[trailing_edge_index])
        except AttributeError as exc:
            raise AttributeError(
                "The airfoil attribute must be set before accessing the trailing edge."
            ) from exc

    @property
    def trailing_edge2(self) -> SpatialArray:
        """Retrieves the lower surface trailing edge of the airfoil as a SpatialArray.

        Returns
        -------
        SpatialArray
            trailing edge coordinates

        Raises
        ------
        AttributeError
            airfoil is not set for the curve i.e. is a border guiding curve
        """
        try:
            trailing_edge_index = self.airfoil.index_te2  # type: ignore
            return SpatialArray(self.data[trailing_edge_index])
        except AttributeError as exc:
            raise AttributeError(
                "The airfoil attribute must be set before accessing the trailing edge."
            ) from exc

    def __len__(self) -> int:
        """Return the len of the data attribute"""
        return len(self.data)

    @staticmethod
    def from_section(
        section: Section, globalpos: SpatialArray, surf_type: SurfaceType
    ) -> GeometricCurve:
        """Create a GeometricCurve from a section, global_postion,
        and surface_type

        Parameters
        ----------
        section : Section
            section object
        globalpos : SpatialArray
            global_position of the surface the section belongs to
        surf_type : SurfaceType, FIN triggers a 90 degree rotation.
            MAINWING, SECONDARYWING and ELEVATOR do not trigger a rotation.

        Returns
        -------
        GeometricCurve
            geometric curve array with the coordinates in 3D space of the curve.
        """
        airfoil = section.airfoil
        twist = -np.radians(section.twist)
        chord = section.chord
        offset = np.array([section.x_offset, section.y_offset])
        wingspan = section.wingspan
        coordinates = airfoil.data
        center = airfoil.center

        # Section Curve 3D
        airfoil_cordinates = transform_coordinates(
            coordinates, center, twist, chord, offset, wingspan
        )

        is_fin = surf_type is SurfaceType.FIN

        curve3d = transform_to_global_coordinate_system(
            airfoil_cordinates, globalpos, is_fin
        )

        return GeometricCurve(name=airfoil.name, data=curve3d, airfoil=airfoil)


class GeometricSurface:
    """Represents a geometric surface with data as arrays of the x, y, and z coordinates
    for each location of a patch. A surface from the points is specified by the matrices
    and will then connect those points by linking the values next to each other in the matrix
    """

    curves: list[GeometricCurve]
    borders: list[GeometricCurve]
    name: str | None
    color: tuple[float] | None
    surface_type: SurfaceType
    xx: np.ndarray
    yy: np.ndarray
    zz: np.ndarray

    def __init__(
        self,
        curves: list[GeometricCurve],
        surface_type: SurfaceType,
        name=None,
        color=None,
    ) -> None:
        self.curves: list[GeometricCurve] = curves
        self.surface_type = surface_type
        self.name = name
        self.color = color
        self.standarize_curves()
        self.borders: list[GeometricCurve] = self.borders_from_curves()
        self.surf_from_curves()

    def add_curve(self, curve: GeometricCurve) -> None:
        """Add a parametric curve to the list of airfoil curves."""
        self.curves.append(curve)
        self.surf_from_curves()

    def add_border(self, curve: GeometricCurve) -> None:
        """Add a parametric border (Border of Attack or Trailing Edge) to the list of wing leading or trailing edges."""
        self.borders.append(curve)

    def surf_from_curves(self):
        """Compute the surface matrices from the curves"""
        self.standarize_curves()

        self.xx = np.array([curve.x for curve in self.curves])
        self.yy = np.array([curve.y for curve in self.curves])
        self.zz = np.array([curve.z for curve in self.curves])

    def borders_from_curves(self) -> list[GeometricCurve]:
        """Compute the leading and trailing edges of the provided curves"""
        le = np.array([curve.leading_edge for curve in self.curves])
        te = np.array([curve.trailing_edge for curve in self.curves])
        te2 = np.array([curve.trailing_edge2 for curve in self.curves])

        return [
            GeometricCurve(name="leading_edge", data=le),
            GeometricCurve(name="trailing_edge", data=te),
            GeometricCurve(name="trailing_edge2", data=te2),
        ]

    def set_color(self, color) -> None:
        """Set the color of the surface"""
        self.color = color

    def standarize_curves(self, n_samples=150) -> None:
        """Verifies that all curves have the same number of points"""
        curve_lengths = [len(curve) for curve in self.curves]

        if len(set(curve_lengths)) != 1:
            print("Not all curves have same length...resampling")
            self.resample_curves(n_samples)

    def resample_curves(self, n_samples: int):
        """Resample all curves to enforce an uniform number of coordinates per curve"""
        for curve in self.curves:
            curve = curve.resample(n_samples)

    def get_curve_list(self) -> list[np.ndarray]:
        return [curve.to_gcs() for curve in self.curves]

    @staticmethod
    def from_aero_surface(surface: AeroSurface) -> GeometricSurface:
        """Creates a Geometric Surface from an AeroSurface instance

        Parameters
        ----------
        surface : AeroSurface
            Aerodynamic definition of the lifting surface

        Returns
        -------
        GeometricSurface
            Geometry
        """
        globalpos = surface.position
        surf_type = surface.surf_type  # SurfaceType.FIN
        curves = [
            GeometricCurve.from_section(section, globalpos, surf_type)
            for section in surface.sections
        ]
        return GeometricSurface(
            curves=curves,
            surface_type=surf_type,
            name=surface.name,
            color=surface.color,
        )


class GeometryProcessor:
    """Behaviour Oriented Class
    Processes Aircraft to create 3D models."""

    def __init__(self, aircraft: Aircraft) -> None:
        self.aircraft = aircraft
        self.surfaces: list[GeometricSurface] = self._create_geometry()

    def _create_geometry(self) -> list[GeometricSurface]:
        """Creates the surface

        Returns
        -------
        list[GeometricSurface]
            _description_
        """
        return [
            GeometricSurface.from_aero_surface(surface) for surface in self.aircraft
        ]

    def add_surface(self, surface: GeometricSurface) -> None:
        """Add a geometric surface (x,y,z) data matrices to the list of surfaces."""
        self.surfaces.append(surface)

    def find_surfaces(self, surf_type: SurfaceType) -> list[GeometricSurface]:
        """Find all lifting surfaces with a particular type in the surfaces list"""
        return [
            surface for surface in self.surfaces if surface.surface_type is surf_type
        ]

    def export_curves(
        self,
        output_path: str,
        ext: str = "sldcrv",
        reference_system: str = "SW",
        units: str = "mm",
    ):
        """
        This method generates .txt files of the Geometric Curves in the specified folder

        Parameters
        ----------
        output_path : string . Location of the output folder where files will be saved
        ext : string. File extension ('txt' or 'sldcrv') default is 'sldcrv'
            DESCRIPTION The name of the extension of the export files
        reference_system : string , optional
            DESCRIPTION. The output system of reference default is 'SW' (SolidWorks).

            Output Reference Systems:
                            'SW'     'XFLR5'    'Python'   'MATLAB'
                Chordwise    -Z         X          X          X
                 Vertical     Y         Z          Y          Y
        (Wing)   Spanwise     X        -Y          Z          Z


        units : string , 'mm' milimeters, 'm' meters, 'in' inches
            DESCRIPTION. The units for data output, default is 'mm'.

        Returns
        -------
        None.

        """

        main_folder = join(normpath(output_path), f"Curves_{self.aircraft.name}")

        os.makedirs(main_folder, exist_ok=True)

        # Export the Section Airfoil Files
        for i, (surface, curve) in enumerate(self.get_curve_iterator()):
            local_folder = join(main_folder, surface.surface_type.name)
            os.makedirs(local_folder, exist_ok=True)

            file_name = f"{i}_{curve.name}_{surface.name}.{ext}".replace(" ", "_")
            file_name = join(local_folder, file_name)

            output_data = curve.to_gcs(reference_system, units)

            np.savetxt(file_name, output_data, fmt="%f")

    def get_curve_iterator(self):
        """
        Returns an iterator that iterates over each surface and its curves and borders.

        Yields
        ------
        tuple
            A tuple containing a surface and a curve or border.

        """
        for surface in self.surfaces:
            for curve in chain(surface.borders, surface.curves):
                yield surface, curve

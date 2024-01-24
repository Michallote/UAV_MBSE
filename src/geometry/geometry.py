"""Geometry Module"""
from __future__ import annotations

import numpy as np
import pandas as pd

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

    def __init__(self, name: str, data: np.ndarray) -> None:
        self.name = name
        self.data = data

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

    def to_gcs(self):
        """
        GCS = Global Coordinate System
        Returns the coordinates as a Numpy array.
        """
        units, reference_system, reflect_axis = get_ref_coordinate_system(
            output_reference_system="SW", output_units="mm"
        )
        data = units * self.data
        output_data = data.mul(reflect_axis, axis="columns")[reference_system]

        return output_data

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
        return GeometricCurve(self.name, coordinates)

    def section_area(self):
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

        return (1.0 / 2.0) * (
            (xf + xi) * ((yf - yi) * (np.cos(gamma)) + (zf - zi) * (np.sin(gamma)))
        )

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

        return GeometricCurve(name=airfoil.name, data=curve3d)


@dataclass
class GeometricSurface:
    """Represents a geometric surface with data as lists of the x, y, and z coordinates
    for each location of a patch. A surface from the points is specified by the matrices
    and will then connect those points by linking the values next to each other in the matrix
    """

    xx: np.ndarray = np.array([])
    yy: np.ndarray = np.array([])
    zz: np.ndarray = np.array([])

    def __post_init__(self) -> None:
        self.curves: List[GeometricCurve] = []
        self.borders: List[GeometricCurve] = []

    def add_curve(self, curve: GeometricCurve) -> None:
        """Add a parametric curve to the list of airfoil curves."""
        self.curves.append(curve)

    def add_border(self, curve: GeometricCurve) -> None:
        """Add a parametric border (Border of Attack or Trailing Edge) to the list of wing leading or trailing edges."""
        self.borders.append(curve)

    def surf_from_curves(self):
        self.standarize_curves()

        self.xx = np.array([curve.x for curve in self.curves])
        self.yy = np.array([curve.y for curve in self.curves])
        self.zz = np.array([curve.z for curve in self.curves])

    def edge_from_curves():
        pass

    def set_color(self, color) -> None:
        self.color = color

    def add_surf_plot(self, ax, color="default", ls=LightSource(azdeg=-35, altdeg=45)):
        if color == "default":
            color = self.color
        # rgb = ls.shade(self.yy, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')

        ax.plot_surface(
            self.xx, self.yy, self.zz, lightsource=ls, color=color
        )  # , facecolors=color

    def standarize_curves(self, nsamples=150):
        """Verifies that all curves have the same number of points"""
        curve_lengths = [len(curve) for curve in self.curves]

        if len(set(curve_lengths)) != 1:
            print("Not all curves have same length...resampling")

            for curve in self.curves:
                curve.resample(nsamples)

    def get_curve_list(self):
        return [curve.get_npdata(CGS=True) for curve in self.curves]


class GeometryProcessor:
    """Behaviour Oriented Class
    Processes Aircraft to create 3D models."""

    def __init__(self, aircraft: Aircraft) -> None:
        self.aircraft = aircraft
        self.surfaces: list[GeometricSurface] = []

    def create_geometries(self) -> None:
        for surface in self.aircraft:
            geosurface = GeometricSurface()
            geosurface.set_color(surface.color)
            globalpos = surface.position
            surf_type = surface.surf_type  # SurfaceType.FIN

            BoA = []
            BoS = []
            BoS2 = []
            centroid_list = []
            area_list = []

            for section in surface.sections:
                curve = GeometricCurve()
                airfoil = section.airfoil
                curve.set_name(airfoil.name)
                twist = -np.radians(section.Twist)
                chord = section.chord
                offset = np.array([section.xOffset, section.yOffset])
                wingspan = section.wingspan
                coordinates = section.airfoil.get_data(dim="2D", output_format="np")
                center = section.airfoil.center
                centroid = section.airfoil.centroid

                # Section Curve 3D
                airfoil_cordinates = transform_coordinates(
                    coordinates, center, twist, chord, offset, wingspan
                )
                curve.set_data(airfoil_cordinates)
                curve3d = transform_to_GCS(airfoil_cordinates, globalpos, surf_type)

                curve.set_data(curve3d)

                centroid3d = transform_coordinates(
                    centroid, center, twist, chord, offset, wingspan
                )
                globalcentroid3d = transform_to_GCS(centroid3d, globalpos, surf_type)
                curve.set_properties(globalcentroid3d, (chord**2) * airfoil.area)

                ledges_pointers = [
                    airfoil.BoA.name,
                    airfoil.BoS.name,
                    airfoil.BoS2.name,
                ]  # Recall the pointers in the array
                BoA_cords, BoS_cords, BoS2_cords = curve3d[ledges_pointers]

                BoA.append(BoA_cords)
                BoS.append(BoS_cords)
                BoS2.append(BoS2_cords)
                centroid_list.append(curve.centroid.as_numpy())
                area_list.append(curve.area)

                geosurface.add_curve(curve)

            for i, element in enumerate([BoA, BoS, BoS2]):
                border = GeometricCurve()
                border.set_name(["BoA", "BoS", "BoS2"][i])
                coordinate = np.array(element)
                border.set_data(coordinate)

                geosurface.add_border(border)

            surface.df["Area"] = area_list

            surface.df["Centroid"] = centroid_list

            surface.df["BoA"] = BoA

            surface.df["BoS"] = BoS

            surface.df["BoS2"] = BoS2

            surface.df["Surface"] = surf_type

            geosurface.surf_from_curves()

            self.add_surface(geosurface)

    def add_surface(self, surface: GeometricSurface) -> None:
        """Add a geometric surface (x,y,z) data matrices to the list of surfaces."""
        self.surfaces.append(surface)

    def find_surfaces(self, surf_type: SurfaceType) -> List[AeroSurface]:
        """Find all lifting surfaces with a particular type in the surfaces list"""
        return [surface for surface in self.surfaces if surface.surf_type is surf_type]

    def find_aspect_ratios(self):
        """
        This method returns all minimum and maximum coordinates for x, y and z
        of the stored surface data. Then determines optimum aspect ratio for the plot.
        (So unit length scales are the same)

        Notes: Probably unneeded as we can derive the same values directly from the Axes3d object

        minx, maxx, miny, maxy, minz, maxz = ax.get_w_lims()

        Returns
        -------
        xlim : TYPE
            DESCRIPTION.
        ylim : TYPE
            DESCRIPTION.
        zlim : TYPE
            DESCRIPTION.
        box_aspect : TYPE
            DESCRIPTION.

        """

        max_x = np.max([np.max(surface.xx) for surface in self.surfaces])
        max_y = np.max([np.max(surface.yy) for surface in self.surfaces])
        max_z = np.max([np.max(surface.zz) for surface in self.surfaces])

        min_x = np.min([np.min(surface.xx) for surface in self.surfaces])
        min_y = np.min([np.min(surface.yy) for surface in self.surfaces])
        min_z = np.min([np.min(surface.zz) for surface in self.surfaces])

        def minmax(array):
            return np.array([np.min(array), np.max(array)])

        xlim = [min_x, max_x]
        ylim = [min_y, max_y]
        zlim = [min_z, max_z]

        def interval(x):
            return x[1] - x[0]

        lims = [xlim, ylim, zlim]
        lims_len = [interval(lim) for lim in lims]

        k = np.min(lims_len)
        # k = np.argsort(lims_len)
        # k = lims_len[ki[0]]
        box_aspect = tuple(lims_len / k)

        return xlim, ylim, zlim, box_aspect

    def plot_aircraft(self, plot_num=1):
        fig = plt.figure(num=plot_num, clear=True, figsize=plt.figaspect(0.5))
        # fig, (axs,axs1) = plt.subplots(1, 2, num=plot_num, clear=True)
        ax = fig.add_subplot(1, 2, 2, projection="3d")

        ax.view_init(vertical_axis="y")
        ax.set_proj_type(proj_type="ortho")
        ls = LightSource(azdeg=-35, altdeg=45)

        for surface in self.surfaces:
            surface.add_surf_plot(ax)

        # Set plot parameter to enforce correct scales

        xlim, ylim, zlim, box_aspect = find_aspect_ratios(ax)

        ax.set(
            xlabel="x",
            ylabel="y",
            zlabel="z",
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            # xticks = [-4, -2, 2, 4],
            # yticks = [-4, -2, 2, 4],
            # zticks = [-1, 0, 1],
            title=self.aircraft.name,
        )

        # if Bug is not fixed yet!!!! -> box_aspect needs to be shifted right
        box_aspect = tuple(np.roll(box_aspect, shift=1))
        ax.set_box_aspect(aspect=box_aspect)

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.view_init(vertical_axis="y")
        ax1.set_proj_type(proj_type="ortho")

        for surface in self.surfaces:
            for curve in surface.curves:
                data = curve.get_npdata()
                ax1.plot3D(*data.T)

            for BoX in surface.borders:
                data = BoX.get_npdata()
                ax1.plot3D(*data.T)

        ax1.set(
            xlabel="x",
            ylabel="y",
            zlabel="z",
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            # xticks = [-4, -2, 2, 4],
            # yticks = [-4, -2, 2, 4],
            # zticks = [-1, 0, 1],
            title=self.aircraft.name,
        )
        ax1.set_box_aspect(aspect=box_aspect)

        # ax.set_axis_off()
        # ax1.set_axis_off()

        def on_move(event):
            if event.inaxes != ax1 and event.inaxes != ax:
                return
            azimuth, elevation = event.inaxes.azim, event.inaxes.elev
            ax.azim = azimuth
            ax.elev = elevation
            ax1.azim = azimuth
            ax1.elev = elevation

            fig.canvas.draw()

        def on_move2(event):
            """
            Synchronize the camera view angles, zoom and pan of both subplots,
            using the 'motion_notify_event' event of the
            figure class every time the mouse moves, it updates
            the view angles of both subplots at the same time.
            """
            if event.inaxes != ax1 and event.inaxes != ax:
                return
            azimuth, elevation = event.inaxes.azim, event.inaxes.elev
            xlim, ylim, zlim = (
                event.inaxes.get_xlim(),
                event.inaxes.get_ylim(),
                event.inaxes.get_zlim(),
            )

            axs = [ax, ax1]

            for a_x in axs:
                a_x.azim = azimuth
                a_x.elev = elevation
                a_x.set_xlim(xlim)
                a_x.set_ylim(ylim)
                a_x.set_zlim(zlim)

            fig.canvas.draw()

        fig.canvas.mpl_connect("motion_notify_event", on_move2)
        # fig.canvas.mpl_connect('button_press_event', on_press)
        # fig.canvas.mpl_connect('button_release_event', on_release)

        # print(ax.viewLim)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        # fig.tight_layout()

        return None

    def plot_aircraft_test(self, plot_num=1):
        fig, ax = plt.subplots(
            num=plot_num,
            clear=True,
            gridspec_kw={"width_ratios": [1, 1]},
            constrained_layout=True,
        )

        spec2 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
        # fig, (axs,axs1) = plt.subplots(1, 2, num=plot_num, clear=True)
        ax = fig.add_subplot(spec2[0, 1], projection="3d")

        ax.view_init(vertical_axis="y")
        ax.set_proj_type(proj_type="ortho")

        for surface in self.surfaces:
            surface.add_surf_plot(ax)

        # Set plot parameter to enforce correct scales

        xlim, ylim, zlim, box_aspect = find_aspect_ratios(ax)

        ax.set(
            xlabel="x",
            ylabel="y",
            zlabel="z",
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            # xticks = [-4, -2, 2, 4],
            # yticks = [-4, -2, 2, 4],
            # zticks = [-1, 0, 1],
            title=self.aircraft.name,
        )

        # if Bug is not fixed yet!!!! -> box_aspect needs to be shifted right
        box_aspect = tuple(np.roll(box_aspect, shift=1))
        ax.set_box_aspect(aspect=box_aspect)

        ax1 = fig.add_subplot(spec2[0, 0], projection="3d")
        ax1.view_init(vertical_axis="y")
        ax1.set_proj_type(proj_type="ortho")

        for surface in self.surfaces:
            for curve in surface.curves:
                data = curve.get_npdata()
                ax1.plot3D(*data.T)

            for BoX in surface.borders:
                data = BoX.get_npdata()
                ax1.plot3D(*data.T)

        ax1.set(
            xlabel="x",
            ylabel="y",
            zlabel="z",
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            # xticks = [-4, -2, 2, 4],
            # yticks = [-4, -2, 2, 4],
            # zticks = [-1, 0, 1],
            title=self.aircraft.name,
        )
        ax1.set_box_aspect(aspect=box_aspect)

        # print(ax.viewLim)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        # fig.tight_layout()

        return None

    def export_curves(self, output_reference_system="SW", output_units="mm"):
        """
        This method generates .txt files of the Geometric Curves in the specified folder

        Parameters
        ----------
        output_reference_system : string , optional
            DESCRIPTION. The output system of reference default is 'SW' (SolidWorks).

            Output Reference Systems:
                            'SW'     'XFLR5'    'Python'   'MATLAB'
                Chordwise    -Z         X          X          X
                 Vertical     Y         Z          Y          Y
        (Wing)   Spanwise     X        -Y          Z          Z


        output_units : string , 'mm' milimeters, 'M' meters, 'in' inches
            DESCRIPTION. The units for data output, default is 'mm'.

        Returns
        -------
        None.

        """
        units_dict = {"M": 1.0, "mm": 1000.0, "in": 1 / 39.3701}

        units = units_dict[output_units]

        coordinate_system = {
            "XFLR5": ["x", "z", "y"],
            "SW": ["z", "y", "x"],
            "Python": ["x", "y", "z"],
            "MATLAB": ["x", "y", "z"],
        }
        reflections = {
            "XFLR5": {"x": 1, "z": -1, "y": 1},
            "SW": {"z": 1, "y": 1, "x": -1},
            "Python": {"x": 1, "y": 1, "z": 1},
            "MATLAB": {"x": 1, "y": 1, "z": 1},
        }

        reference_system = coordinate_system[output_reference_system]
        reflect_axis = reflections[output_reference_system]

        mainfoldername = os.path.join(
            os.path.normpath(WORKING_DIRECTORY), "Curves_" + self.aircraft.name
        )

        if not os.path.exists(mainfoldername):
            os.makedirs(mainfoldername)

        for aero_surf, geo_surf in zip(self.aircraft.surfaces, self.surfaces):
            localpath = os.path.join(mainfoldername, aero_surf.name)
            identifier = aero_surf.surf_type.to_str()

            if not os.path.exists(localpath):
                os.makedirs(localpath)

            # Export the Section Airfoil Files
            for i, curve in enumerate(geo_surf.curves):
                data = units * curve.data3d
                output_data = data.mul(reflect_axis, axis="columns")[reference_system]
                name = curve.name
                fname = os.path.join(
                    localpath,
                    "Section_" + str(i) + "_" + name + " " + identifier + ".txt",
                )

                np.savetxt(fname, output_data.values, fmt="%f")

            # Export the Surface Trailing and Leading Edges Files
            for i, border in enumerate(geo_surf.borders):
                data = units * border.data3d
                output_data = data.mul(reflect_axis, axis="columns")[reference_system]

                name = border.name

                fname = os.path.join(localpath, name + " " + identifier + ".txt")

                np.savetxt(fname, output_data.values, fmt="%f")

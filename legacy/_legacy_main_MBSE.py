# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:44:43 2022

@author: Michel Gordillo

Model-Based Systems Engineering
------------------------------------------------------------------------------
UNAM Aero Design
------------------------------------------------------------------------------
"""
# %% Imports

# System Commands
import os
import tkinter as tk
from tkinter import filedialog  # Open File Explorer to select files

# Modules
import SolidWorksVBA as SWVBA

# from geometry_tools import GeometryProcessor

# Scientific & Engineering
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import LightSource

# plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
# plt.rcParams['svg.fonttype'] = 'none'

# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('svg')

# Data Management

import xml.etree.ElementTree as ET
import pandas as pd

# Software Design Tools

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List

# %% Global Variables

AIRFOIL_FOLDER_PATH = "E:/Documentos/Aero Design/Temporada 2023/Avanzada/airfoils"  # All airfoil .dat files must be on this folder
WORKING_DIRECTORY = "E:/Documentos/Aero Design/Temporada 2023/Avanzada"  # XML File with the aircraft definition must be saved at this directory
XMLfile = "E:/Documentos/Aero Design/Temporada 2023/Avanzada/Propuesta final.xml"

# %% Classes
# %%% Data Definition Classes


class Airfoil:
    """Represents airfoil data and properties."""

    def __init__(self, name, path) -> None:
        self.name = name
        self.path = path
        self.data = pd.DataFrame([], columns=["x", "y"])
        self.data3d = pd.DataFrame([], columns=["x", "y", "z"])

    def read_data(self, file: str = None):
        """
        Method that imports data from a .dat file

        Parameters
        ----------
        file : str, optional
            Full file path. The default is None -> Automatically parses the
            file path from the objects attributes.

        Returns
        -------
        None.

        """
        if not file:
            file = self.parse_filepath()

        self.data = pd.read_table(
            file, delim_whitespace=True, skiprows=[0], names=["x", "y"], index_col=False
        )

    def parse_filepath(self) -> str:
        """Parse the airfoil .dat file from the object attributes."""

        return self.path + "/" + self.name + ".dat"

    def set_data(self, coordinates: np.ndarray):
        """
        Saves the coordinates data to the airfoil. Handles automatically
        2D and 3D input coordinates.

        Parameters
        ----------
        coordinates : np.ndarray
            numpy array of Airfoil coordinates.
            shape must be (n,m) Where:
                n is the number of points
                m is the dimension of data
                    2D : m = 2 ; 3D : m = 3

        Returns
        -------
        None.

        """

        if coordinates.shape[1] == 2:
            self.data = pd.DataFrame(coordinates, columns=["x", "y"])

        elif coordinates.shape[1] == 3:
            self.data3d = pd.DataFrame(coordinates, columns=["x", "y", "z"])

    def get_data(self, dim="2D", output_format="np"):

        datadict = {"2D": self.data, "3D": self.data3d}

        if dim not in ["2D", "3D"]:
            print('Wrong dimension format ( "2D" , "3D" ) ')
            raise TypeError

        if output_format == "np":
            return datadict[dim].to_numpy()
        elif output_format == "df":
            return datadict[dim]
        else:
            print('Wrong output format ( "np" , "df" ) ')
            raise TypeError

    def compute_properties(self):
        self.area = -np.trapz(self.data.y, x=self.data.x)  # Numerical integration
        datasort = self.data.sort_values(["x", "y"], ascending=[True, False])
        self.BoA = datasort.iloc[0]
        # self.BoAidx = datasort.iloc[0].name
        self.BoS = self.data.iloc[0]
        self.BoS2 = self.data.iloc[-1]

        self.extrados = self.data.loc[self.BoS.name : self.BoA.name]
        self.extrados = self.extrados.iloc[
            ::-1, :
        ]  # Reverse the order of the array (interpolation requirement)

        n = 0
        if datasort.x.iloc[0] == datasort.x.iloc[1]:
            n = 1

        self.intrados = self.data.loc[self.BoA.name + n : self.BoS2.name]

        self.center = np.array(
            [
                0.25,
                0.5
                * (
                    np.interp(0.25, self.extrados.x, self.extrados.y)
                    + np.interp(0.25, self.intrados.x, self.intrados.y)
                ),
            ]
        )

    def calc_inertia(self):
        # self.centroid = self.data.mean(axis=0)

        x = self.data.x.to_numpy()
        y = self.data.y.to_numpy()

        N = range(len(self.data) - 1)
        M = np.array(
            [(x[i] - x[i + 1]) * (y[i] + y[i + 1]) / 2 for i in N]
        )  # Area of each trapz
        My = np.array([(x[i] + x[i + 1]) / 2 for i in N]) * M
        Mx = np.array([(y[i] + y[i + 1]) / 4 for i in N]) * M
        X = sum(My) / sum(M)
        Y = sum(Mx) / sum(M)

        self.centroid = Point([X, Y])

        def area_moment_of_inertia(x, y):
            xi = x
            xf = np.roll(x, -1)
            yi = y
            yf = np.roll(y, -1)

            Iyy = (1 / 12) * (xi * yf - xf * yi) * ((xf**2) + (xf * xi) + (xi**2))
            Ixx = (1 / 12) * (xi * yf - xf * yi) * ((yf**2) + (yf * yi) + (yi**2))
            Ixy = (
                (1 / 24)
                * (xi * yf - xf * yi)
                * (2 * xf * yf + xi * yf + xf * yi + 2 * xi * yi)
            )

            Ixx = np.sum(Ixx)
            Iyy = np.sum(Iyy)
            Ixy = np.sum(Ixy)

            return np.array([[Ixx, Ixy], [Ixy, Iyy]])

    @property
    def centroid(self):
        x = self.data.x.to_numpy()
        y = self.data.y.to_numpy()

        def xc(x, y):
            """Calculates the xc centroid of a closed contour using greens theorem
            https://leancrew.com/all-this/2018/01/greens-theorem-and-section-properties/
            """
            xi = x
            xf = np.roll(x, -1)
            yi = y
            yf = np.roll(y, -1)
            return (1 / 6) * (xf + xi) * (xi * yf - xf * yi)

        def yc(x, y):
            """Calculates the yc centroid of a closed contour using greens theorem
            https://leancrew.com/all-this/2018/01/greens-theorem-and-section-properties/
            """
            xi = x
            xf = np.roll(x, -1)
            yi = y
            yf = np.roll(y, -1)
            return (1 / 6) * (yf + yi) * (xi * yf - xf * yi)

        def Area_GreensTheorem(x, y):
            """Calculates the area of a closed contour using greens theorem
            https://leancrew.com/all-this/2018/01/greens-theorem-and-section-properties/
            """
            xi = x
            xf = np.roll(x, -1)
            yi = y
            yf = np.roll(y, -1)
            return (xf + xi) * (-yi + yf) / 2

        A = np.sum(Area_GreensTheorem(x, y))
        X = np.sum(xc(x, y)) / A
        Y = np.sum(yc(x, y)) / A

        return Point([X, Y])

    def thickness(self, t):
        """
        Returns the camber (camber/chord) value with %c (0->1) as input
        -------

        """
        return np.interp(t, self.extrados.x, self.extrados.y) - np.interp(
            t, self.intrados.x, self.intrados.y
        )

    def camber(self, t):
        """
        Returns the thickness (t/c) value with %c (0->1) as input
        """
        return 0.5 * (
            np.interp(t, self.extrados.x, self.extrados.y)
            + np.interp(t, self.intrados.x, self.intrados.y)
        )

    def max_thickness(self, n_iter=4):

        thicks = find_max(self.thickness)
        max_thick = max(thicks)

        return max_thick

    def max_camber(self, n_iter=4):

        n_camber = find_max(self.camber)
        max_camb = max(n_camber)

        return max_camb

    def plot_airfoil(self):
        # plt.scatter(self.data.x, self.data.y,marker='o',edgecolors='black',s=3)
        plt.plot(
            self.data.x,
            self.data.y,
            "r",
            marker=".",
            markeredgecolor="black",
            markersize=3,
        )
        plt.axis("equal")
        plt.xlim((-0.05, 1.05))
        plt.legend([self.name])


def find_max(f, n_iter=4):

    n_puntos = 10

    # Se definen los límites en "x" para aplicar la interpolación
    x_interp_min = 0.05
    x_interp_max = 0.95

    # Se comienzan las iteraciones
    for i in range(n_iter):
        # Se definen los n puntos en x donde se va a realizar la interpolación
        x_interp = np.linspace(x_interp_min, x_interp_max, n_puntos)

        # Se obtienen las interpolaciones
        thicks = f(x_interp)

        # Se define una lista con los grosores ordenados de menor a mayor
        thicks_ordenados = np.argsort(thicks)

        i_sup = thicks_ordenados[-1]
        i_inf = thicks_ordenados[-2]

        # Se definen las nuevas fronteras en x para la interpolación
        x_interp_max = x_interp[i_sup]
        x_interp_min = x_interp[i_inf]

    return thicks


def slice_shift(x):
    """
    Returns the x[i] and x[i+1] arrays for numerical calculations xi, xf
    """
    return x[:-1], x[1:]


@dataclass
class Section:
    """Represents an wing section with assigned airfoil properties."""

    wingspan: float
    chord: float
    xOffset: float
    yOffset: float
    Dihedral: float
    Twist: float
    FoilName: str
    airfoil: Airfoil
    x_number_of_panels: int
    x_panel_distribution: str
    y_number_of_panels: int
    y_panel_distribution: str

    def distance_to(self, other):
        return self.wingspan - other.wingspan


class SurfaceType(Enum):
    """Aerodynamic Surfaces types"""

    MAINWING = auto()
    SECONDWING = auto()
    ELEVATOR = auto()
    FIN = auto()

    # @classmethod
    def __repr__(self):
        return "{} ({})".format(self.name, self.value)

    def to_str(self):
        return "{}".format(self.name)


@dataclass
class AeroSurface:
    """Basic representation of an aerodynamic surface at the aircraft."""

    name: str
    position: np.ndarray
    surf_type: SurfaceType
    tilt: float
    symmetric: bool
    is_fin: bool
    is_doublefin: bool
    is_symfin: bool
    # sections: List[Section] = []

    def __post_init__(self) -> None:
        self.sections = []

    def add_section(self, section: Section) -> None:
        """Add a section to the list of sections."""
        self.sections.append(section)

    def set_color(self, color):
        self.color = color

    def add_dataframe(self, df):
        self.df = df

    def calc_yOffset(self):
        var_sections = self.sections
        delta_span = [
            var_sections[i].distance_to(var_sections[i + 1])
            for i in range(len(var_sections) - 1)
        ]  # Distance between sections
        dihedral = np.sin(np.radians([section.Dihedral for section in self.sections]))[
            :-1
        ]
        yOffsets = np.insert(np.cumsum(delta_span * dihedral), 0, 0.0)

        for section, yOffset in zip(self.sections, yOffsets):
            section.yOffset = yOffset

    def get_ribs_position(self, rib_spacing=0.15):
        wingspans = np.array([section.wingspan for section in self.sections])

        # Number of ribs between sections
        n_ribs = np.ceil((wingspans[1:] - wingspans[:-1]) / rib_spacing)

        section_ribs = [
            np.linspace(wingspans[i], wingspans[i + 1], int(n + 1))
            for i, n in enumerate(n_ribs)
        ]

        ribs_position = np.unique(np.concatenate(section_ribs))

        return ribs_position

    def get_ribs_df(self):
        df = self.df

        # Interpolate Section Areas
        ribs_position = self.get_ribs_position()
        wingspans = df["Wingspan"].to_numpy()
        areas = df["Area"].to_numpy()
        ribs_area = np.interp(ribs_position, wingspans, areas)

        # Interpolate Centroids
        centroids = np.array(list(self.df["Centroid"]))
        ribs_centroid = multi_dim_interp(ribs_position, wingspans, centroids)

    def __repr__(self):
        return "({0}, {1}, No. Sections: {2})".format(
            self.name, self.surf_type, len(self.sections)
        )


def multi_dim_interp(x, xp, array3d):
    """
    Parameters
    ----------
    x : values to interpolate float.
    xp : TYPE
        DESCRIPTION.
    array3d : TYPE
        DESCRIPTION.

    Returns
    -------
    interpolation : TYPE
        DESCRIPTION.

    """

    fp = np.arange(len(xp))
    t = np.interp(
        x, xp, fp
    )  # Array with floats 4.333 is an element in a 33% away of element 4 in xp and 67% away from element 5

    np_int = np.vectorize(int)  # Create function applicable element-wise
    right = np_int(np.ceil(t))  # Array of upper bounds of each new element
    left = np_int(np.floor(t))  # Array of lower bounds of each new element

    # Linear interpolation p = a + (b-a)*t

    delta = array3d[right] - array3d[left]  # (b-a)
    t_p = t - left  # t Array of interpolation fractions between a -> b for each element
    interpolation = (
        array3d[left] + delta * t_p[:, None]
    )  # p Element - wise Linear interpolation
    return interpolation


class Aircraft:
    """Represents an Aircraft with Aerodynamic Surfaces and Properties."""

    def __init__(self, name) -> None:
        self.surfaces: List[AeroSurface] = []
        self.name: str = name

    def add_surface(self, surface: AeroSurface) -> None:
        """Add a lifting surface to the list of surfaces."""
        self.surfaces.append(surface)

    def find_surfaces(self, surf_type: SurfaceType) -> List[AeroSurface]:
        """Find all lifting surfaces with a particular type in the surfaces list"""
        return [surface for surface in self.surfaces if surface.surf_type is surf_type]

    def print_parameters(self):
        """Print to console all data frames"""
        for surface in self.surfaces:
            print(
                "\n Name: {}\n{}".format(
                    surface.name,
                    surface.df[
                        [
                            "Wingspan",
                            "Chord",
                            "Twist",
                            "xOffset",
                            "yOffset",
                            "Dihedral",
                            "FoilName",
                        ]
                    ].to_string(),
                )
            )

    def ribs_for_analysis(self):
        """Generates a list with ribs locations for each surface"""
        for surface in self.surfaces:
            list(surface.df["Wingspan"])

    def get_aircraft_df(self):
        self.df = pd.concat(
            [surface.df for surface in self.surfaces], ignore_index=True
        )
        return self.df.copy()


def parse_xml(XMLfile) -> Aircraft:
    """Inputs: XML File directory
    Outputs: Aircraft Object

     Parses an XML exported from XFLR5 into Aircraft object."""

    # try:
    #     if os.path.exists(XMLfile):
    #         xml_file_path = XMLfile
    # except NameError: XMLfile = 'None'

    if os.path.exists(XMLfile):
        xml_file_path = XMLfile
    else:

        root = tk.Tk()  # Initialise library for pop windows
        root.withdraw()  # Close GUI window

        print(
            "The defined XML file {} does not exist.\n Select one on the File Explorer Window".format(
                XMLfile
            )
        )

        while True:
            xml_file_path = filedialog.askopenfilename(
                filetypes=[("XML Documents", "*.xml")]
            )

            if xml_file_path == "":
                raise SystemExit("No XML file selected. Ending execution")

            ext = os.path.splitext(xml_file_path)[-1].casefold()

            if ext == ".xml":
                print(f"Source file found: {xml_file_path}")
                break
            else:
                print(f"Unknown extension option: {ext}.")

    XML_tree = ET.parse(xml_file_path)
    XML_apex = XML_tree.getroot()  # Get the top hierarchical elements
    units, XML_UAV = list(XML_apex)

    UAV = Aircraft(name=XML_UAV.find("Name").text)

    for XMLsurface in XML_UAV.findall("wing"):

        UAV.add_surface(parse_surfxml(XMLsurface))

    print("Aircraft created: {}".format(UAV.name))

    return UAV


def parse_surfxml(XMLsurface: ET.Element) -> AeroSurface:
    """Parses a wing child ET.Element into a AeroSurface object."""

    xflr5_to_py = [0, 2, 1]  # flip y and z axis [x,y,z]_xfl -> [x,z,y]_py

    def get_text(element: ET.Element, tag: str) -> str:
        return element.find(tag).text

    def to_bool(text):
        return eval(text.capitalize())

    properties = {item.tag: item.text for item in XMLsurface}

    name = properties["Name"]

    position = np.array(eval(properties["Position"]))
    position = position[xflr5_to_py]

    surf_type = SurfaceType[properties["Type"]]
    tilt = float(properties["Tilt_angle"])
    symmetric = to_bool(properties["Symetric"])
    is_fin = to_bool(properties["isFin"])
    is_doublefin = to_bool(properties["isDoubleFin"])
    is_symfin = to_bool(properties["isSymFin"])

    surface = AeroSurface(
        name, position, surf_type, tilt, symmetric, is_fin, is_doublefin, is_symfin
    )
    colortuple = tuple(
        [int(element.text) / 255 for element in XMLsurface.find("Color")]
    )
    surface.set_color(colortuple)
    XMLsections = XMLsurface.find("Sections").findall("Section")
    sectionsDF = sections_dataframe(XMLsections)
    surface.add_dataframe(sectionsDF)

    for XMLsection in XMLsections:

        surface.add_section(parse_sectionxml(XMLsection))

    surface.calc_yOffset()

    print("Surface created: {}\nType: {} \n".format(surface.name, surface.surf_type))

    return surface


def parse_sectionxml(XMLsection: ET.Element) -> Section:
    """Parses a section child ET.Element into a Section object."""

    parameters = {item.tag: item.text for item in XMLsection}

    wingspan = float(parameters["y_position"])
    chord = float(parameters["Chord"])
    xOffset = float(parameters["xOffset"])
    # yOffset =  float
    Dihedral = float(parameters["Dihedral"])
    Twist = float(parameters["Twist"])
    FoilName = parameters["Right_Side_FoilName"]
    x_number_of_panels = int(parameters["x_number_of_panels"])
    x_panel_distribution = parameters["x_panel_distribution"]
    y_number_of_panels = int(parameters["y_number_of_panels"])
    y_panel_distribution = parameters["y_panel_distribution"]

    airfoil = Airfoil(name=FoilName, path=AIRFOIL_FOLDER_PATH)
    airfoil.read_data()
    airfoil.compute_properties()

    section = Section(
        wingspan=wingspan,
        chord=chord,
        xOffset=xOffset,
        yOffset=0,
        Dihedral=Dihedral,
        Twist=Twist,
        FoilName=FoilName,
        airfoil=airfoil,
        x_number_of_panels=x_number_of_panels,
        x_panel_distribution=x_panel_distribution,
        y_number_of_panels=y_number_of_panels,
        y_panel_distribution=y_panel_distribution,
    )

    return section


def sections_dataframe(XMLsections):
    """
    Creates a DataFrame of wing sections

    Parameters
    ----------
    XMLsections : TYPE: ET:Element
        XML Sections of wing elements.

    Returns
    -------
    df : pd.DataFrame
        DESCRIPTION.

    """
    data = [[field.text for field in section] for section in XMLsections]
    headers = [field.tag for field in XMLsections[0]]
    df = pd.DataFrame(data, columns=headers)
    df.rename(
        columns={"Right_Side_FoilName": "FoilName", "y_position": "Wingspan"},
        inplace=True,
    )
    df = df.drop(["Left_Side_FoilName"], axis=1)
    df = df.astype(
        {
            "Wingspan": "float64",
            "Chord": "float64",
            "xOffset": "float64",
            "Dihedral": "float64",
            "Twist": "float64",
            "x_number_of_panels": "int64",
            "y_number_of_panels": "int64",
            "FoilName": "str",
        }
    )

    # Calculate vertical distance consecuence of dihedral angles
    delta_span = df["Wingspan"].diff().dropna().to_numpy()  # Distance between sections
    dihedral = (np.sin(np.radians(df["Dihedral"]))[:-1]).to_numpy()
    yOffset = np.insert(np.cumsum(delta_span * dihedral), 0, 0.0)
    df["yOffset"] = yOffset

    return df


# %%% Geometry Processing
@dataclass
class GeometricCurve:
    """Represents a parametric curve with data as lists of the x, y, and z coordinates"""

    x: np.ndarray = np.array([])
    y: np.ndarray = np.array([])
    z: np.ndarray = np.array([])

    def set_data(self, coordinates):

        if coordinates.shape[1] == 2:
            self.data = pd.DataFrame(coordinates, columns=["x", "y"])

        elif coordinates.shape[1] == 3:
            self.data3d = pd.DataFrame(coordinates, columns=["x", "y", "z"])
            self.x = self.data3d.x.to_numpy()
            self.y = self.data3d.y.to_numpy()
            self.z = self.data3d.z.to_numpy()

    def set_name(self, name):
        self.name = name

    def get_npdata(self, GCS=False):
        """
        GCS = Global Coordinate System
        Returns the coordinates as a Numpy array.
        """
        if GCS:

            units, reference_system, reflect_axis = get_ref_coordinate_system(
                output_reference_system="SW", output_units="mm"
            )
            data = units * self.data3d
            output_data = data.mul(reflect_axis, axis="columns")[reference_system]

            return output_data.to_numpy()

        return self.data3d.to_numpy()

    def resample(self, nsamples, get=False):
        coordinates = resample_curve(self.get_npdata(), nsamples)
        self.set_data(coordinates)

        if get:
            return coordinates

    def section_area(self):
        """
        Stokes theorem used for computing the area through a parameterized
        line integral of a 3D curve.

        """
        x = self.x
        y = self.y
        z = self.z
        gamma = self.gamma

        def roll(x):
            return x, np.roll(x, -1, axis=0)

        xi, xf = roll(x)
        yi, yf = roll(y)
        zi, zf = roll(z)

        return (1.0 / 2.0) * (
            (xf + xi) * ((yf - yi) * (np.cos(gamma)) + (zf - zi) * (np.sin(gamma)))
        )

    def set_properties(self, centroid, area):
        self.centroid = centroid
        self.area = area

    def __len__(self):
        return len(self.data3d)


@dataclass
class GeometricSurface:
    """Represents a geometric surface with data as lists of the x, y, and z coordinates
    for each location of a patch. A surface from the points is specified by the matrices
    and will then connect those points by linking the values next to each other in the matrix"""

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

    def __init__(self, UAV: Aircraft) -> None:
        self.aircraft = UAV
        self.surfaces: List[GeometricSurface] = []

    def create_geometries(self) -> None:
        for surface in self.aircraft.surfaces:
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

    # def export_to_SolidWorksVBA(self):
    #     SWVBA.CURVE_COUNTER = 0
    #     macro_contents = SWVBA.set_preamble('MacroSW')

    #     for surface in self.surfaces:

    #         coordinates_lists = surface.get_curve_list(CGS = True)

    #         SWVBA.insert_curves_features(coordinates_lists,name_list,folder_name)

    #     macro_contents += SWVBA.set_end_code()


def find_aspect_ratios(ax):
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

    minx, maxx, miny, maxy, minz, maxz = ax.get_w_lims()

    xlim = [minx, maxx]
    ylim = [miny, maxy]
    zlim = [minz, maxz]

    def interval(x):
        return abs(x[1] - x[0])

    lims = [xlim, ylim, zlim]
    lims_len = [interval(lim) for lim in lims]
    k = np.min(lims_len)
    box_aspect = tuple(lims_len / k)

    return xlim, ylim, zlim, box_aspect


def transform_coordinates(
    coordinates, center, twist, chord, offset, wingspan
) -> np.ndarray:
    """
    Applies translations and rotations to airfoil data points

    Parameters
    ----------
    section : Section object
        Contains all relevant information about the transformation.

    Returns
    -------
    cords3d : np.ndarray
        Curve Coordinates.

    """
    # twist = -np.radians(section.Twist)
    # chord = section.chord
    # offset = np.array([section.xOffset,section.yOffset])
    # wingspan = section.wingspan

    # coordinates = section.airfoil.get_data( dim = '2D', output_format = 'np')
    # center = section.airfoil.center

    if twist != 0:
        rotmat = rotation_matrix2d(twist)
        # coordinates = [rotmat@r for r in (coordinates-center)] + center
        coordinates = np.dot(coordinates - center, rotmat.T) + center

    coordinates = coordinates * chord + offset
    # Dimension adder  (3 x 2) @ (2 x 1) = (3 x 1)
    matrix_to_R3 = np.array([[1, 0], [0, 1], [0, 0]])
    # Broadcast the result over the rows of B
    cords3d = np.dot(coordinates, matrix_to_R3.T) + np.array([0, 0, wingspan])

    # cords3d = np.c_[coordinates, wingspan*np.ones(len(coordinates))]

    return cords3d


def transform_to_GCS(
    cords3d: np.ndarray, globalpos: np.ndarray, surf_type: SurfaceType
) -> np.ndarray:
    """
    Transforms the curve from its local reference frame to the global coordinate system GCS

    Parameters
    ----------
    cords3d : np.ndarray
        Curves.
    globalpos : np.ndarray
        DESCRIPTION.
    surf_type : SurfaceType
        DESCRIPTION.

    Returns
    -------
    cords3d : TYPE
        DESCRIPTION.

    """

    if surf_type == SurfaceType.FIN:
        rotmat3d = rotation_matrix3d(-90, axis="x", units="degrees")
        # cords3d = [rotmat3d@r for r in cords3d]
        # Broadcast the result over the rows of B
        cords3d = np.dot(cords3d, rotmat3d.T)

    cords3d = cords3d + globalpos
    return cords3d


def rotation_matrix2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def rotation_matrix3d(theta: float, axis="x", units="radians") -> np.ndarray:

    if units == "degrees":
        theta = np.radians(theta)

    c, s = np.cos(theta), np.sin(theta)

    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    elif axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    elif axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    else:
        raise NameError("Invalid axis")


def get_ref_coordinate_system(output_reference_system="SW", output_units="mm"):
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

    return units, reference_system, reflect_axis


def resample_curve(array3d, nsamples: int):
    """
    Resample an array based on linear interpolation between indexes.

    Parameters
    ----------
    array3d : np.array()
              Can be (n,m) dimentional
    nsamples : int


    Returns
    -------
    resample : TYPE
        DESCRIPTION.

    """
    n_orig = len(array3d)  # Read original array size
    t = np.linspace(
        0, n_orig - 1, nsamples
    )  # Resample as if index was the independent variable
    np_int = np.vectorize(int)  # Create function applicable element-wise
    right = np_int(np.ceil(t))  # Array of upper bounds of each new element
    left = np_int(np.floor(t))  # Array of lower bounds of each new element

    # Linear interpolation p = a + (b-a)*t

    delta = array3d[right] - array3d[left]  # (b-a)
    t_p = t - left  # t Array of interpolation fractions between a -> b for each element
    resample = (
        array3d[left] + delta * t_p[:, None]
    )  # p Element - wise Linear interpolation

    return resample


# %% Main


def main(UAV: Aircraft):
    """Main function."""

    UAV.print_parameters()

    UAV_Geometry = GeometryProcessor(UAV)
    UAV_Geometry.create_geometries()
    UAV_Geometry.plot_aircraft()
    # UAV_Geometry.plot_aircraft_test()
    # UAV_Geometry.export_curves()

    UAV.surfaces[0].get_ribs_df()

    return UAV, UAV_Geometry


if __name__ == "__main__":

    # create the factory
    UAV = parse_xml(XMLfile)

    rUAV, rUAV_Geometry = main(UAV)

# UAV = parse_xml()
# a = 2

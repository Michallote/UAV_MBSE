# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 04:25:05 2022

@author: Michel Gordillo
"""

# %% Imports

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["svg.fonttype"] = "none"

# from mpl_toolkits import mplot3d

# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('svg')

# %% Classes


class Wing:
    def __init__(self, element):

        self.name = element.find("Name").text
        xflr5pos = np.fromstring(element.find("Position").text, dtype=float, sep=",")
        xflr5_to_py = [0, 2, 1]  # flip y and z axis [x,y,z]_xfl -> [x,z,y]_py
        self.pos = xflr5pos[xflr5_to_py]
        self.type = element.find("Type").text
        self.tilt = float(element.find("Tilt_angle").text)
        self.is_sym = element.find("Symetric").text.casefold() == "true"
        self.is_fin = element.find("isFin").text.casefold() == "true"
        self.is_doublefin = element.find("isDoubleFin").text.casefold() == "true"
        self.is_symfin = element.find("isSymFin").text.casefold() == "true"
        self.sections = self.construct_sections(
            element.find("Sections").findall("Section")
        )
        self.xml = element

    def __repr__(self):
        return "Wing obj:({0}, {1}, No. Sections: {2}])".format(
            self.name, self.type, self.sections.shape[0]
        )

    def construct_sections(self, sections):
        data = [[field.text for field in section] for section in sections]
        headers = [field.tag for field in sections[0]]
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

        # Calculate vertical distance because of  dihedral field
        delta_span = (
            df["Wingspan"].diff().dropna().to_numpy()
        )  # Distance between sections
        dihedral = (np.sin(np.radians(df["Dihedral"]))[:-1]).to_numpy()
        yOffset = np.insert(np.cumsum(delta_span * dihedral), 0, 0.0)
        df["yOffset"] = yOffset

        df["Airfoil"] = [Airfoil(foil) for foil in df.FoilName]
        return df


# @dataclass
class Airfoil:
    def __init__(
        self,
        airfoil,
        folder="E:/Documentos/Thesis - Master/Master/XFLR5 exports/airfoils",
    ):
        self.name = airfoil
        self.path = folder + os.sep + airfoil + ".dat"
        self.data = pd.read_table(
            self.path,
            delim_whitespace=True,
            skiprows=[0],
            names=["x", "y"],
            index_col=False,
        )
        # self.name = str(*self.airfoil.columns.values)

        self.parameters()

    def __repr__(self):
        return self.name

    def parameters(self):
        self.area = -np.trapz(self.data.y, x=self.data.x)
        datasort = self.data.sort_values(["x", "y"], ascending=[True, False])
        self.BoA = datasort.head(1)
        self.BoS = self.data.head(1)
        self.BoS2 = self.data.tail(1)

        self.extrados = self.data.loc[self.BoS.index[0] : self.BoA.index[0]]
        self.extrados = self.extrados.iloc[::-1, :]
        self.intrados = self.data.loc[self.BoA.index[0] : self.BoS2.index[0]]

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

    def set_data(self, coordinates):

        if coordinates.shape[1] == 2:
            self.data = pd.DataFrame(coordinates, columns=["x", "y"])

        elif coordinates.shape[1] == 3:
            self.data3d = pd.DataFrame(coordinates, columns=["x", "y", "z"])

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


@dataclass
class Section:

    wingspan: float
    chord: float
    xOffset: float
    yOffset: float
    Dihedral: float
    Twist: float
    FoilName: str
    airfoil: Any  # Airfoil class
    x_number_of_panels: int
    x_panel_distribution: int
    y_number_of_panels: int
    y_panel_distribution: int

    def distance_to(self, other):
        return self.wingspan - other.wingspan

    def construct_sections(self, sections):
        data = [[field.text for field in section] for section in sections]
        headers = [field.tag for field in sections[0]]
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

        # Calculate vertical distance because of  dihedral field
        delta_span = (
            df["Wingspan"].diff().dropna().to_numpy()
        )  # Distance between sections
        dihedral = (np.sin(np.radians(df["Dihedral"]))[:-1]).to_numpy()
        yOffset = np.insert(np.cumsum(delta_span * dihedral), 0, 0.0)
        df["yOffset"] = yOffset

        df["Airfoil"] = [Airfoil(foil) for foil in df.FoilName]
        return df


def transform_coordinates(wing):
    sectiondict = wing.sections.to_dict()

    for idx in wing.sections.index:

        twist = -np.radians(sectiondict["Twist"][idx])
        chord = sectiondict["Chord"][idx]
        offset = np.array([sectiondict["xOffset"][idx], sectiondict["yOffset"][idx]])
        wingspan = sectiondict["Wingspan"][idx]
        globalpos = wing.pos

        coordinates = sectiondict["Airfoil"][idx].data.to_numpy()
        center = sectiondict["Airfoil"][idx].center

        if twist != 0:
            rotmat = rotation_matrix2d(twist)
            coordinates = [rotmat @ r for r in (coordinates - center)] + center

        coordinates = coordinates * chord + offset
        # wing.sections.Airfoil[idx].set_data(coordinates)
        # airfoil.data = coordinates

        cords3d = np.c_[coordinates, wingspan * np.ones(len(coordinates))]

        if wing.is_fin:
            rotmat3d = rotation_matrix3d(-90, axis="x", units="degrees")
            cords3d = [rotmat3d @ r for r in cords3d]

        cords3d = cords3d + globalpos

        wing.sections.Airfoil[idx].set_data(cords3d)


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


# %% Main
def main() -> None:
    """Main function."""

    airfoilpaths = "E:/Documentos/Thesis - Master/Master/XFLR5 exports/airfoils"

    path = "E:/Documentos/Thesis - Master/Master"
    os.chdir(path)

    # Reading the data inside the xml
    # file to a variable under the name
    # data
    mytree = ET.parse(
        "XFLR5 exports/N0009_cg0.245_H1.5_V0.10_T2N_VU0.10_Vv0.03_L0.6.xml"
    )
    myroot = mytree.getroot()
    units, plane = list(myroot)

    aero_surfaces = plane.findall("wing")

    # aero_surfaces[0].getchildren()

    aero_surfaces[0].find("Position")

    wing = Wing(aero_surfaces[0])
    wing2 = Wing(aero_surfaces[1])
    fin = Wing(aero_surfaces[2])

    perfil = Airfoil(wing.sections.loc[6, "FoilName"], folder=airfoilpaths)

    perfil.plot_airfoil()

    perfil.parameters()

    # dicttest = wing.sections.to_dict()
    # perfil.data.to_numpy()

    # twist = -np.radians(dicttest['Twist'][0])
    # chord = dicttest['Chord'][0]
    # offset = np.array([dicttest['xOffset'][0],dicttest['yOffset'][0]])
    # wingspan = dicttest['Wingspan'][0]
    # globalpos = wing.pos

    # coordinates = perfil.data.to_numpy()
    # center = dicttest['Airfoil'][0].center

    # if twist != 0:
    #     rotmat=rotation_matrix2d(twist)
    #     coordinates = [rotmat@r for r in (coordinates-center)] + center

    # coordinates = coordinates*chord + offset

    # #airfoil.data = coordinates

    # cords3d = np.c_[coordinates, wingspan*np.ones(len(coordinates))] + globalpos

    transform_coordinates(
        wing
    )  # Must add method to prevent duplicate transformations -> do not overwrite original 2d airfoil data
    transform_coordinates(wing2)
    transform_coordinates(fin)

    wing.sections.Airfoil[0].data3d

    # fig = plt.figure()
    # ax = plt.axes(projection='3d',aspect = 'equal')
    # ax.view_init(azim=-45, elev=26, vertical_axis='y')
    # #ax.axis('equal')
    # ax.plot3D(*cords3d.T)
    # ax.plot3D(*(cords3d*0.5).T )
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    # ax.grid(False)

    transform_coordinates(
        wing
    )  # Must add method to prevent duplicate transformations -> do not overwrite original 2d airfoil data
    transform_coordinates(wing2)
    transform_coordinates(fin)

    wing.sections.Airfoil[0].data3d

    fig = plt.figure(tight_layout=True)
    ax = plt.axes(projection="3d", aspect="auto")
    ax.set_zlim(0, 2)
    ax.set_xlim(0, 2)
    ax.set_ylim(-1, 1)
    ax.view_init(azim=-45, elev=26, vertical_axis="y")

    # ax.axis('equal')
    for surface in list([wing, wing2, fin]):
        for airfoil in list(surface.sections.Airfoil):
            curve = airfoil.data3d.to_numpy()
            ax.plot3D(*curve.T)

    # ax.plot3D(*cords3d.T)
    # ax.plot3D(*(cords3d*0.5).T )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(False)

    # plt.plot(*coordinates.T, 'r',marker='.',markeredgecolor='black', markersize=3)
    # plt.axis('equal')
    # plt.xlim((-0.05, 1.05))


if __name__ == "__main__":
    main()

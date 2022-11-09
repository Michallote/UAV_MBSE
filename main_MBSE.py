# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:44:43 2022

@author: Michel Gordillo

Model Based Systems Engineering
------------------------------------------------------------------------------
UNAM Aero Design
------------------------------------------------------------------------------
"""
# %% Imports

#System Commands
import os
import tkinter as tk 
from tkinter import filedialog #Open File Explorer to select files

#Modules
# import geometry_tools
# from geometry_tools import GeometryProcessor

# Scientific & Engineering
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
#plt.rcParams['svg.fonttype'] = 'none'

#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('svg')

#Data Management

import xml.etree.ElementTree as ET
import pandas as pd

# Software Design Tools

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List

# %% Global Variables

AIRFOIL_FOLDER_PATH = 'E:/Documentos/Thesis - Master/Master/XFLR5 exports/airfoils'   #All airfoil .dat files must be on this folder     
    
WORKING_DIRECTORY = 'E:/Documentos/Thesis - Master/Master' # XML File with the aircraft definition must be saved at this directory

XMLfile = 'E:/Documentos/Thesis - Master/Master/XFLR5 exports/N0009_cg0.245_H1.5_V0.10_T2N_VU0.10_Vv0.03_L0.6.xml'

# %% Classes
# %%% Data Definition Classes
class VacationDaysShortageError(Exception):
    """Custom error that is raised when not enough vacation days are available."""

    def __init__(self, requested_days: int, remaining_days: int, message: str) -> None:
        self.requested_days = requested_days
        self.remaining_days = remaining_days
        self.message = message
        super().__init__(message)

@dataclass
class Airfoil():
    """Represents airfoil data and properties."""
    
    name: str
    path: 'str' = AIRFOIL_FOLDER_PATH
    
    def __post_init__(self) -> None:
        self.data = pd.DataFrame([],columns=['x','y'])
        self.data3d = pd.DataFrame([],columns=['x','y','z'])
    
    def read_data(self, file : str = None ):
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
        
        self.data = pd.read_table(file,delim_whitespace=True,skiprows=[0],names=['x','y'],index_col=False)
        
    def parse_filepath(self) -> str:
        """  Parse the airfoil .dat file from the object attributes.  """

        return self.path + '/' + self.name + '.dat'
    
    def set_data(self,coordinates : np.ndarray):
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
            self.data = pd.DataFrame(coordinates,columns=['x','y'])
            
        elif coordinates.shape[1] == 3:
            self.data3d = pd.DataFrame(coordinates,columns=['x','y','z'])
    
    def get_data(self, dim = '2D', output_format = 'np'):
        
        datadict = {
        "2D": self.data,
        "3D": self.data3d
        }
        
        if output_format == 'np':
            return datadict[dim].to_numpy()
        elif output_format == 'df':
            return datadict[dim]
        else:
            print('Wrong output format ( "np" , "df" ) ')
            raise TypeError
        
    def compute_properties(self):
        self.area = -np.trapz(self.data.y,x = self.data.x) #Numerical integration
        datasort = self.data.sort_values(['x','y'], ascending = [True,False])
        self.BoA = datasort.iloc[0]
        self.BoS = self.data.iloc[0]
        self.BoS2 = self.data.iloc[-1]
        
        self.extrados = self.data.loc[self.BoS.name : self.BoA.name]
        self.extrados = self.extrados.iloc[::-1, :] #Reverse the order of the array (interpolation requirement)
        self.intrados = self.data.loc[self.BoA.name : self.BoS2.name]
        
        self.center = np.array([0.25,0.5*(np.interp(0.25, self.extrados.x, self.extrados.y)
                         +np.interp(0.25, self.intrados.x, self.intrados.y))])
        
    def inertia(self):
        self.centroid = self.data.mean(axis=0)
        
        x = self.data.x.to_numpy() 
        y = self.data.y.to_numpy()
        
        N = range(len(self.data)-1)
        M = np.array([(x[i]-x[i+1])*(y[i]+y[i+1])/2 for i in N]) #Area of each trapz
        My = np.array([(x[i]+x[i+1])/2 for i in N])*M
        Mx = np.array([(y[i]+y[i+1])/4 for i in N])*M
        X = sum(My)/sum(M)
        Y =sum(Mx)/sum(M)
        
        self.centroid = pd.DataFrame([X , Y],columns=['x','y'])
        
        
        # for i in range(len(self.data)-1):
        #     M[i]=(x[i]-x[i+1])*(y[i]+y[i+1])/2
        #     My[i]=([x[i]+x[i+1]]/2)*M[i]
        #     Mx[i]=([y[i]+y[i+1]]/4)*M[i]
    
    def plot_airfoil(self):
        #plt.scatter(self.data.x, self.data.y,marker='o',edgecolors='black',s=3)
        plt.plot(self.data.x, self.data.y, 'r',marker='.',markeredgecolor='black', markersize=3)
        plt.axis('equal')
        plt.xlim((-0.05, 1.05))
        plt.legend([self.name])


@dataclass
class Section():
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
        return (self.wingspan - other.wingspan)

class SurfaceType(Enum):
    """Aerodynamic Surfaces types"""

    MAINWING = auto()
    SECONDWING = auto()
    ELEVATOR = auto()
    FIN = auto()
    
    #@classmethod
    def __repr__(self):
        return '{} ({})'.format(self.name,self.value)


# def read_factory() -> SurfaceType:
#     """Constructs an SurfaceType factory based on the input text."""

#     factories = {
#         "low": FastExporter(),
#         "high": HighQualityExporter(),
#         "master": MasterQualityExporter(),
#     }
#     while True:
#         export_quality = input("Enter desired output quality (low, high, master): ")
#         if export_quality in factories:
#             return factories[export_quality]
#         print(f"Unknown output quality option: {export_quality}.")


@dataclass
class AeroSurface():
    """Basic representation of an aerodynamic surface at the aircraft."""

    name: str
    position: np.ndarray
    surf_type: SurfaceType
    tilt: float
    symmetric: bool
    is_fin: bool
    is_doublefin: bool
    is_symfin: bool
    #sections: List[Section] = []
    
    def __post_init__(self) -> None:
        self.sections = []
 
    def add_section(self, section: Section) -> None:
        """Add a section to the list of sections."""
        self.sections.append(section)
        
    def add_dataframe(self,df):
        self.df = df
        
    def calc_yOffset(self):
        var_sections = self.sections
        delta_span= [var_sections[i].distance_to(var_sections[i+1]) for i in range(len(var_sections)-1)] #Distance between sections
        dihedral = (np.sin(np.radians([section.Dihedral for section in self.sections]))[:-1])
        yOffsets =  np.insert(np.cumsum(delta_span*dihedral),0,0.0)
        
        for section, yOffset in zip(self.sections,yOffsets):
            section.yOffset = yOffset
    
    def __repr__(self):
        return "({0}, {1}, No. Sections: {2})".format(self.name, self.surf_type, len(self.sections))
    

class Aircraft:
    """Represents an Aircraft with Aerodynamic Surfaces and Properties."""

    def __init__(self,name) -> None:
        self.surfaces: List[AeroSurface] = []
        self.name: str = name

    def add_surface(self, surface: AeroSurface) -> None:
        """Add a lifting surface to the list of surfaces."""
        self.surfaces.append(surface)

    def find_surfaces(self, surf_type: SurfaceType) -> List[AeroSurface]:
        """Find all lifting surfaces with a particular type in the surfaces list"""
        return [surface for surface in self.surfaces if surface.surf_type is surf_type]
    
    def print_parameters(self):
        """ Print to console all data frames  """
        for surface in self.surfaces:
            print('Name: {}\n{}'.format(surface.name,surface.df[['Wingspan', 'Chord', 'Twist','xOffset','yOffset', 'Dihedral', 'FoilName']].to_string()))


def parse_xml() -> Aircraft:
    """Parses an XML exported from XFLR5 into Aircraft object."""
    
    # try: 
    #     if os.path.exists(XMLfile):
    #         xml_file_path = XMLfile
    # except NameError: XMLfile = 'None'
    
    if os.path.exists(XMLfile):
        xml_file_path = XMLfile
    else:
        
        root = tk.Tk() #Initialise library for pop windows
        root.withdraw() #Close GUI window
        
        print('The defined XML file {} does not exist.\n Select one on the File Explorer Window'.format(XMLfile))
        
        while True:
            xml_file_path = filedialog.askopenfilename(filetypes=[("XML Documents", "*.xml")])
            
            if xml_file_path == '':
                raise SystemExit("No XML file selected. Ending execution")
            
            ext = os.path.splitext(xml_file_path)[-1].casefold()
            
            if ext == '.xml':
                print(f"Source file found: {xml_file_path}")
                break
            else:
                print(f"Unknown extension option: {ext}.")
            
    XML_tree = ET.parse(xml_file_path)
    XML_apex = XML_tree.getroot() #Get the top hierarchical elements
    units, XML_UAV = list(XML_apex)
    
    UAV = Aircraft(name = XML_UAV.find('Name').text)
    
    for XMLsurface in XML_UAV.findall('wing'):
        
        UAV.add_surface(parse_surfxml(XMLsurface))
    
    print('Aircraft created: {}'.format(UAV.name))
    
    return UAV
    

def parse_surfxml(XMLsurface : ET.Element) -> AeroSurface:
     """Parses a wing child ET.Element into a AeroSurface object."""   
     
     xflr5_to_py = [0,2,1] #flip y and z axis [x,y,z]_xfl -> [x,z,y]_py
     
     def get_text(element: ET.Element, tag: str) -> str: return element.find(tag).text
     
     def to_bool(text): return eval(text.capitalize())
     
     properties = {item.tag:item.text for item in XMLsurface}
     
     name = properties['Name']
     
     position = np.array(eval(properties['Position']))
     position = position[xflr5_to_py]
     
     surf_type = SurfaceType[properties['Type']]
     tilt = float(properties['Tilt_angle'])
     symmetric = to_bool(properties['Symetric'])
     is_fin = to_bool(properties['isFin'])
     is_doublefin = to_bool(properties['isDoubleFin'])
     is_symfin = to_bool(properties['isSymFin'])
     
     surface = AeroSurface(name, position, surf_type, tilt, symmetric, is_fin, is_doublefin, is_symfin)
     
     XMLsections = XMLsurface.find('Sections').findall('Section')
     sectionsDF = sections_dataframe(XMLsections)
     surface.add_dataframe(sectionsDF)
     
     for XMLsection in XMLsections:
        
        surface.add_section(parse_sectionxml(XMLsection))
     
     surface.calc_yOffset()   
     
     print('Surface created: {}\nType {}: '.format(surface.name,surface.surf_type))   
     
     return surface

def parse_sectionxml(XMLsection : ET.Element) -> Section:
    """Parses a section child ET.Element into a Section object."""
    
    parameters = {item.tag:item.text for item in XMLsection}
    
    wingspan = float( parameters['y_position'] )
    chord =  float( parameters['Chord'] )
    xOffset =  float( parameters['xOffset'] )
    #yOffset =  float
    Dihedral =  float( parameters['Dihedral']  )
    Twist =  float( parameters['Twist'] )
    FoilName =  parameters['Right_Side_FoilName'] 
    x_number_of_panels =  int(  parameters['x_number_of_panels'] )
    x_panel_distribution =    parameters['x_panel_distribution'] 
    y_number_of_panels =  int(parameters['y_number_of_panels'] )
    y_panel_distribution =   parameters['y_panel_distribution']
    
    airfoil = Airfoil(name = FoilName, path = AIRFOIL_FOLDER_PATH)
    airfoil.read_data()
    airfoil.compute_properties()
    
    section = Section(wingspan = wingspan, chord = chord, xOffset = xOffset, yOffset = 0, Dihedral = Dihedral, Twist = Twist,
            FoilName = FoilName,airfoil=airfoil,x_number_of_panels = x_number_of_panels,
            x_panel_distribution = x_panel_distribution, y_number_of_panels = y_number_of_panels, y_panel_distribution = y_panel_distribution)
    
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
    df = pd.DataFrame(data, columns = headers)
    df.rename(columns={'Right_Side_FoilName': 'FoilName','y_position':'Wingspan'}, inplace=True)
    df = df.drop(['Left_Side_FoilName'],axis=1)
    df = df.astype({'Wingspan':'float64', 'Chord':'float64', 'xOffset':'float64', 'Dihedral':'float64', 'Twist':'float64',
   'x_number_of_panels':'int64', 'y_number_of_panels':'int64','FoilName':'str'})
 
 
    # Calculate vertical distance consecuence of dihedral angles
    delta_span= df['Wingspan'].diff().dropna().to_numpy() #Distance between sections
    dihedral = (np.sin(np.radians(df['Dihedral']))[:-1]).to_numpy()
    yOffset =  np.insert(np.cumsum(delta_span*dihedral),0,0.0)
    df['yOffset'] = yOffset
 
    return df

# %%% Geometry Processing

@dataclass
class GeometricCurve():
    """Represents a parametric curve with data as lists of the x, y, and z coordinates"""

    x: np.ndarray = np.array([])
    y: np.ndarray = np.array([])
    z: np.ndarray = np.array([])
    
    def set_data(self,coordinates):
    
        if coordinates.shape[1] == 2:
            self.data = pd.DataFrame(coordinates,columns=['x','y'])
            
        elif coordinates.shape[1] == 3:
            self.data3d = pd.DataFrame(coordinates,columns=['x','y','z'])
            self.x = self.data3d.x.to_numpy()
            self.y = self.data3d.y.to_numpy()
            self.z = self.data3d.z.to_numpy()
            
    def get_npdata(self):
        return self.data3d.to_numpy()
    
    def resample(self, nsamples , get = False):
        coordinates = resample_curve(self.get_npdata(), nsamples)
        self.set_data(coordinates)
        
        if get:
            return resample_curve(self.get_npdata(), nsamples)
    
    def __len__(self):
        return len(self.data3d)    
        
        
@dataclass
class GeometricSurface():
    """Represents a geometric surface with data as lists of the x, y, and z coordinates
    for each location of a patch. A surface from the points is specified by the matrices 
    and will then connect those points by linking the values next to each other in the matrix"""
    
    xx: np.ndarray = np.array([])
    yy: np.ndarray = np.array([])
    zz: np.ndarray = np.array([])
    
    
    def __post_init__(self) -> None:
        self.curves : List[GeometricCurve] = []
        self.borders : List[GeometricCurve] = []

    def add_curve(self, curve: GeometricCurve) -> None:
        """Add a parametric curve to the list of surfaces."""
        self.curves.append(curve)
        
    def add_border(self, curve: GeometricCurve) -> None:
        """Add a parametric border to the list of surfaces."""
        self.borders.append(curve)

    def surf_from_curves(self):
        
        self.standarize_curves()
        
        self.xx = np.array([curve.x for curve in self.curves])
        self.yy = np.array([curve.y for curve in self.curves])
        self.zz = np.array([curve.z for curve in self.curves])
        
    def set_color(self,color) -> None:
        self.color: 'str' = color    
        
    def add_surf_plot(self,ax,color = 'C0'):
        ax.plot_surface(self.xx, self.yy, self.zz) #, facecolors=color
        
    def standarize_curves(self, nsamples = 150):
        """ Verifies that all curves have the same number of points """
        it = iter(self.curves)
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            
             print('Not all curves have same length...resampling')
             
             for curve in self.curves:
                 curve.resample(nsamples)
             
             

class GeometryProcessor():
    """ Behaviour Oriented Class
    Processes Aircraft to create 3D models."""

    def __init__(self, UAV: Aircraft) -> None:
        self.aircraft =  UAV
        self.surfaces: List[GeometricSurface] = []
        
    def create_geometries(self) -> None:
        for surface in self.aircraft.surfaces:
            geosurface = GeometricSurface()
            globalpos = surface.position
            surf_type = surface.surf_type #SurfaceType.FIN
            
            for section in surface.sections:
                curve = GeometricCurve()
                airfoil_cordinates = transform_coordinates(section)
                curve.set_data(airfoil_cordinates)
                curve3d = transform_to_GCS(airfoil_cordinates, globalpos, surf_type)
                curve.set_data(curve3d)
                
                geosurface.add_curve(curve)
                
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
        
        def minmax(array): return np.array([np.min(array),np.max(array)])
        
        xlim = [min_x, max_x]
        ylim = [min_y, max_y]
        zlim = [min_z, max_z]
        
        def interval(x): return x[1] - x[0]
        
        lims = [xlim , ylim , zlim]
        lims_len = [interval(lim) for lim in lims]
        
        k = np.min(lims_len)
        #k = np.argsort(lims_len)
        #k = lims_len[ki[0]]
        box_aspect = tuple(lims_len/k)
        
        return xlim, ylim, zlim, box_aspect
        
        
    
    def plot_aircraft(self,plot_num=1):
        
        fig = plt.figure(num=plot_num, clear=True,figsize=plt.figaspect(0.5))
        #fig, (axs,axs1) = plt.subplots(1, 2, num=plot_num, clear=True)
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        
        ax.view_init(vertical_axis='y')
        ax.set_proj_type(proj_type='ortho')
  
        
        for surface in self.surfaces:
            surface.add_surf_plot(ax)

        #Set plot parameter to enforce correct scales        

        xlim, ylim, zlim, box_aspect  = find_aspect_ratios(ax)
        
        
        ax.set(xlabel='x', 
               ylabel='y', 
               zlabel='z',
               xlim = xlim,
               ylim = ylim,
               zlim = zlim,
               # xticks = [-4, -2, 2, 4],
               # yticks = [-4, -2, 2, 4],
               # zticks = [-1, 0, 1],
               title=self.aircraft.name)
        
        # if Bug is not fixed yet!!!! -> box_aspect needs to be shifted right
        box_aspect = tuple(np.roll(box_aspect,shift = 1))
        ax.set_box_aspect(aspect = box_aspect)
        
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.view_init(vertical_axis='y')
        ax1.set_proj_type(proj_type='ortho')
        
        for surface in self.surfaces:
            for curve in surface.curves:
                data = curve.get_npdata()
                ax1.plot3D(*data.T)
        
        ax1.set(xlabel='x', 
               ylabel='y', 
               zlabel='z',
               xlim = xlim,
               ylim = ylim,
               zlim = zlim,
               # xticks = [-4, -2, 2, 4],
               # yticks = [-4, -2, 2, 4],
               # zticks = [-1, 0, 1],
               title=self.aircraft.name)
        ax1.set_box_aspect(aspect = box_aspect)
        
        #print(ax.viewLim)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        #fig.tight_layout()
        
        return None

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
    
    xlim = [minx,maxx]
    ylim = [miny,maxy]
    zlim = [minz,maxz]
    
    def interval(x): return abs(x[1] - x[0])
    
    lims = [xlim , ylim , zlim]
    lims_len = [interval(lim) for lim in lims]
    k = np.min(lims_len)
    box_aspect = tuple(lims_len/k)
        
    return xlim, ylim, zlim, box_aspect        
    

def transform_coordinates(section: Section)->np.ndarray:
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
    twist = -np.radians(section.Twist)
    chord = section.chord
    offset = np.array([section.xOffset,section.yOffset])
    wingspan = section.wingspan
    
    
    coordinates = section.airfoil.get_data( dim = '2D', output_format = 'np')
    center = section.airfoil.center
    
    if twist != 0:
        rotmat=rotation_matrix2d(twist)
        coordinates = [rotmat@r for r in (coordinates-center)] + center
    
    coordinates = coordinates*chord + offset
    
    cords3d = np.c_[coordinates, wingspan*np.ones(len(coordinates))] 
    
    return cords3d
    

def transform_to_GCS(cords3d: np.ndarray ,globalpos : np.ndarray, surf_type : SurfaceType)->np.ndarray:
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
        rotmat3d = rotation_matrix3d(-90,axis = 'x', units='degrees')
        cords3d = [rotmat3d@r for r in cords3d]
    
    cords3d = cords3d + globalpos
    return cords3d
    
def rotation_matrix2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def rotation_matrix3d(theta: float, axis = 'x', units = 'radians') -> np.ndarray:
    
    if units == 'degrees':
        theta = np.radians(theta)
    
    c, s = np.cos(theta), np.sin(theta)
    
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    elif axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
    elif axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    else:
        raise NameError("Invalid axis")
 

def resample_curve(array3d, nsamples:int):
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
    n_orig = len(array3d) # Read original array size
    t = np.linspace(0,n_orig-1,nsamples) #Resample as if index was the independent variable
    np_int = np.vectorize(int) #Create function applicable element-wise
    right = np_int(np.ceil(t)) #Array of upper bounds of each new element
    left = np_int(np.floor(t)) #Array of lower bounds of each new element
    
    # Linear interpolation p = a + (b-a)*t 
    
    delta = array3d[right]-array3d[left] # (b-a)
    t_p = t - left # t Array of fraction between a -> b for each element
    resample = array3d[left] + delta*t_p[:,None] # p Element - wise Linear interpolation 
    
    return resample


# %% Main

def main(UAV: Aircraft):
    """Main function."""
    
    UAV.print_parameters()
    
    UAV_Geometry = GeometryProcessor(UAV)
    UAV_Geometry.create_geometries()
    UAV_Geometry.plot_aircraft()
    
    return UAV, UAV_Geometry
    

if __name__ == "__main__":
    
    # create the factory
    UAV = parse_xml()
    
    rUAV, rUAV_Geometry = main(UAV)

# UAV = parse_xml()   
# a = 2


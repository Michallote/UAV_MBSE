# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:06:03 2023

@author: Michel Gordillo
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import PolyCollection
import matplotlib as mpl

from dataclasses import dataclass
from main_MBSE import Airfoil

def path_join(path, filename):
    os.path.normpath(path)
    return os.path.join(path,filename)
    
def listdir(directory, ext = ''):
    return [file for file in os.listdir(directory) if file.endswith(ext)]

# class Point():
#     def __init__(self, x, y, z = None):
#         self.x = x
#         self.y = y
#         self.z = z
        
#     def __array__(self):
#         if self.z is None:
#             return np.array([self.x, self.y])
#         else:
#             return np.array([self.x, self.y, self.z])
        
#     def from_array(self, arr):
#         if len(arr) == 2:
#             self.x = arr[0]
#             self.y = arr[1]
#         elif len(arr) == 3:
#             self.x = arr[0]
#             self.y = arr[1]
#             self.z = arr[2]
#         else:
#             raise ValueError("Array must have length 2 or 3")
        
class Point():
    def __init__(self, x, y=None, z=None):
        if y is None and z is None:
            self._arr = np.asarray(x)
        else:
            self._arr = np.array([x, y, z]) if z is not None else np.array([x, y])
        
    def __array__(self):
        return self._arr
    
    @property
    def x(self):
        return self._arr[0]
    
    @x.setter
    def x(self, value):
        self._arr[0] = value
        
    @property
    def y(self):
        return self._arr[1]
    
    @y.setter
    def y(self, value):
        self._arr[1] = value
    
    @property
    def z(self):
        return self._arr[2] if len(self._arr) > 2 else None
    
    @z.setter
    def z(self, value):
        if len(self._arr) == 2 and value is not None:
            self._arr = np.append(self._arr, value)
        elif len(self._arr) == 3:
            if value is not None:
                self._arr[2] = value
            else:
                self._arr = self._arr[:-1]
    
    def __add__(self, other):
        print('Add method happened')
        return Point(*self._arr + other)
    
    def __repr__(self):
        return str(self._arr)
    

class AirfoilCalc(Airfoil):
    
    def calc_inertia(self):    
        self.area = -np.trapz(self.data.y,x = self.data.x) #Numerical integration
        
        x = self.data.x.to_numpy()
        y = self.data.y.to_numpy()
        
        xi, xf = slice_shift(x)
        yi, yf = slice_shift(y)
        
        #Area of each element
        A = (-xf + xi)*(yi + yf)/2
        
        My = A*(xf + xi)/2
        Mx = A*(yf + yi)/4
        
        def Area_GreensTheorem(x,y):
            """Calculates the area of a closed contour using greens theorem
            https://leancrew.com/all-this/2018/01/greens-theorem-and-section-properties/
            """
            xi = x
            xf = np.roll(x,-1)
            yi = y
            yf = np.roll(y,-1)
            return (xf + xi)*(-yi + yf)/2
        
        A2 = Area_GreensTheorem(x,y)
        
        print('Area M1 {} , Area M2 {},  Area M3 {}'.format(self.area,np.sum(A),np.sum(A2)))
        
        #Centroid
        X = np.sum(My)/np.sum(A)
        Y = np.sum(Mx)/np.sum(A)
        
        centroid = Point(X,Y)
        #self.centroid = centroid
        
        #Moment of Inertia
        
        yn = (yf+yi)/4
        xn = (xf+xi)/2
        
        #About Zero
        def area_moment_inertia(xn,yn,A, x0 = 0,y0 = 0):
            xn = xn - x0
            yn = yn - y0
            Ixx = np.sum(yn*yn*A)
            Iyy = np.sum(xn*xn*A)
            Ixy = np.sum(xn*yn*A)
            Jz = np.sum((xn*xn+yn*yn)*A)
            return Ixx, Iyy, Ixy, Jz
        
        
        def area_moment_inertia_2(x,y,xn,yn,A, x0 = 0,y0 = 0):
            
            xi, xf = slice_shift(x)
            yi, yf = slice_shift(y)
            
            xn = xn - x0
            yn = yn - y0
            
            
            b = -(xf - xi)
            h = (yf + yi)/2
            
            Ixx = (b*h**3)/12
            Iyy = (h*b**3)/12 
            
            Ixx = Ixx + A*yn*yn
            Iyy = Iyy + A*xn*xn
            Jz = Ixx + Iyy
            
            return np.sum(Ixx), np.sum(Iyy), 0, np.sum(Jz)
        
        def inertia_of_shell(x,y,x0=None,y0=None):
            """
            Compute the components of the area moment of inertia tensor using
            Greens Theorem. Contour must be closed.
            
            Closed lamina of uniform density with boundary specified by (x,y)
            Centroid in (x0,y0)
            https://mathworld.wolfram.com/AreaMomentofInertia.html
            https://leancrew.com/all-this/2018/01/greens-theorem-and-section-properties/
            https://math.blogoverflow.com/2014/06/04/greens-theorem-and-area-of-polygons/
            Returns
            -------
            Moment of inertia tensor.
            Ixx, Ixy, Iyy
            """
            # x = x - x0
            # y = y - y0
            xi = x
            xf = np.roll(x,-1)
            yi = y
            yf = np.roll(y,-1)
            
            #Curve length
            L = np.sqrt((yf-yi)**2 + (xf-xi)**2)
            
            # Centroid
            xc = (1/(2*np.sum(L)))*np.sum((xi+xf)*L)
            yc = (1/(2*np.sum(L)))*np.sum((yi+yf)*L)
            
            if (x0 is None) or (y0 is None):
                print('Computing Ixx, Iyy, Ixy from the centroid')
                xf = xf - xc
                xi = xi - xc
                yf = yf - yc
                yi = yi - yc
            elif isinstance(x0, (float, int)) and isinstance(y0, (float, int)) :
                xf = xf - x0
                xi = xi - x0
                yf = yf - y0
                yi = yi - y0
            
            
            Ixx = (1/3)*(yi**2 + yi*yf + yf**2)*L
            Ixy = 1/6*(2*xf*yf + xi*yf + xf*yi + 2*xi*yi)*L
            Iyy = (1/3)*(xi**2 + xi*xf + xf**2)*L
            
            Jz = Ixx + Iyy
            
            return  np.sum(Ixx),  np.sum(Ixy),  np.sum(Iyy), np.sum(Jz)
        
        Ixx0, Iyy0, Ixy0, Jz0 = area_moment_inertia(xn,yn,A)
        Ixx1, Iyy1, Ixy1, Jz1 = area_moment_inertia_2(x,y,xn,yn,A)
        Ixx2, Iyy2, Ixy2, Jz2 = area_moment_inertia(xn,yn,A, x0 =centroid.x, y0 = centroid.y)
        Ixx3, Iyy3, Ixy3, Jz3 = area_moment_inertia_2(x,y,xn,yn,A,  x0 = centroid.x, y0 = centroid.y)
        Ixx4, Iyy4, Ixy4, Jz4 = inertia_of_shell(x,y)
        
        
        return pd.DataFrame(
        {'Ixx': [Ixx0, Ixx1, Ixx2, Ixx3, Ixx4], 'Iyy': [Iyy0, Iyy1, Iyy2, Iyy3, Iyy4], 'Ixy': [Ixy0, Ixy1, Ixy2, Ixy3, Ixy4], 'Jz': [Jz0, Jz1, Jz2, Jz3, Jz4]})
            
    
    def thickness(self, t):
        """
        Returns the camber (camber/chord) value with %c (0->1) as input
        -------
        
        """
        return np.interp(t, self.extrados.x, self.extrados.y) - np.interp(t, self.intrados.x, self.intrados.y)
    
    def camber(self, t):
        """
        Returns the thickness (t/c) value with %c (0->1) as input
        """
        return 0.5*(np.interp(t, self.extrados.x, self.extrados.y) + np.interp(t, self.intrados.x, self.intrados.y))
    
    def max_thickness(self, n_iter = 4):
        
        thicks = find_max(self.thickness)
        max_thick = max(thicks)
        
        return max_thick
    
    def max_camber(self, n_iter = 4):
        
        n_camber = find_max(self.camber)
        max_camb = max(n_camber)
        
        return max_camb
    
    def plot_airfoil(self, n_points = 35):
        t = np.linspace(0.01, 0.98, n_points)
        
        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)
        ax0.plot(self.data.x, self.data.y, 'r',marker='.',markeredgecolor='black', markersize=3)
        ax0.axis('equal')
        ax1.axis('equal')
        ax0.set(xlim=(-0.05, 1.05))
        ax1.set(xlim=(-0.05, 1.05))
        
        ax1.plot(t,airfoil.thickness(t),'-.')
        ax1.legend(['Thickness'])
        ax0.plot(t,airfoil.camber(t),':', color = 'orange')
        ax0.legend([self.name,'Camber'])
        
        data = self.data.to_numpy()
        
        tri = np.array([data[:-1], data[1:] , np.flip(data,axis=0)[:-1]])
        
        
        _,n,_ = tri.shape

        mask = np.array([(i != (n-1)/2) and (i != n/2) for i in range(n)])
        tri = tri[:,mask]
        
        t1 = plt.Polygon(tri[:,50])
        ax0.add_patch(t1)
        
        fig.show()
    
def find_max( f , n_iter = 4):
    
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

AIRFOIL_FOLDER_PATH = 'E:/Documentos/Thesis - Master/Master/XFLR5 exports/airfoils'  #All airfoil .dat files must be on this folder     
    
foil_list = listdir(AIRFOIL_FOLDER_PATH, ext = '.dat')

airfoils = {}
inertia_initialized = None

for foil in foil_list:
    name = os.path.splitext(foil)[0]
    file = path_join(AIRFOIL_FOLDER_PATH,foil)
    data = pd.read_table(file,delim_whitespace=True,skiprows=[0],names=['x','y'],index_col=False)
    airfoil = AirfoilCalc(name = name, path = file)
    airfoil.data = data
    airfoil.compute_properties()
    if inertia_initialized is None:
        inertia = airfoil.calc_inertia()
        inertia_initialized = True
    else:
        inertia = pd.concat([inertia,airfoil.calc_inertia()],ignore_index=True)
    
    print(airfoil.calc_inertia())
    
    airfoils[name] = airfoil
    airfoil.plot_airfoil()
    
    
    
    
    
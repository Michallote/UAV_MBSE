from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.axes3d import Axes3D

from src.geometry.geometry_processing import (
    AircraftGeometry,
    GeometricCurve,
    GeometricSurface,
)
from src.visualization.base_class import BaseAircraftPlotter


class MatplotlibAircraftPlotter(BaseAircraftPlotter):
    """
    Creates a visualization for Aircraft Surfaces and Curves using the Matplotlib backend
    """

    def _find_aspect_ratios(self, vertical_axis="y"):
        """
        Calculates the aspect ratios for plotting based on the geometric surfaces.

        Returns:
        tuple: (xlim, ylim, zlim, box_aspect)
        """
        max_x = np.max([np.max(surface.xx) for surface in self.surfaces])
        max_y = np.max([np.max(surface.yy) for surface in self.surfaces])
        max_z = np.max([np.max(surface.zz) for surface in self.surfaces])

        min_x = np.min([np.min(surface.xx) for surface in self.surfaces])
        min_y = np.min([np.min(surface.yy) for surface in self.surfaces])
        min_z = np.min([np.min(surface.zz) for surface in self.surfaces])

        xlim = (min_x, max_x)
        ylim = (min_y, max_y)
        zlim = (min_z, max_z)

        lims_len = [max_x - min_x, max_y - min_y, max_z - min_z]
        k = np.min(lims_len)
        box_aspect = tuple(lims_len / k)

        (xlim, ylim, zlim) = self.roll_to_vertical_axis(
            (xlim, ylim, zlim), vertical_axis=vertical_axis
        )

        box_aspect = self.roll_to_vertical_axis(box_aspect, vertical_axis=vertical_axis)

        return xlim, ylim, zlim, box_aspect

    @staticmethod
    def roll_to_vertical_axis(args, vertical_axis: str):
        axis = {"x": 0, "y": 1, "z": 2}

        roll = 2 - axis[vertical_axis]

        args = list(args)

        return [args[i - roll] for i, _ in enumerate(args)]

    @staticmethod
    def plot_surface(
        surface: GeometricSurface,
        ax,
        color="default",
        ls=LightSource(azdeg=-35, altdeg=45),
        vertical_axis="y",
    ):
        """add surface plot"""
        if color == "default":
            color = surface.color

            color = tuple(value / 255 for value in color)
        # rgb = ls.shade(self.yy, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')

        xx = surface.xx
        yy = surface.yy
        zz = surface.zz

        xx, yy, zz = MatplotlibAircraftPlotter.roll_to_vertical_axis(
            (xx, yy, zz), vertical_axis=vertical_axis
        )

        ax.plot_surface(xx, yy, zz, lightsource=ls, color=color)

    @staticmethod
    def plot_curve(curve: GeometricCurve, ax: Axes3D, vertical_axis):
        """Plot a xurve in the selected referenci system"""
        x, y, z = MatplotlibAircraftPlotter.roll_to_vertical_axis(
            (curve.x, curve.y, curve.z), vertical_axis=vertical_axis
        )
        ax.plot3D(x, y, z)

    def plot_aircraft(self, aircraft: AircraftGeometry, plot_num=1, vertical_axis="y"):
        """
        Plots the aircraft using the geometric data.

        Parameters:
        plot_num (int): Plot number identifier.
        """
        fig = plt.figure(num=plot_num, clear=True, figsize=plt.figaspect(0.5))
        ax: Axes3D = fig.add_subplot(1, 2, 1, projection="3d")
        ax1: Axes3D = fig.add_subplot(1, 2, 2, projection="3d")

        xlim, ylim, zlim, box_aspect = self._find_aspect_ratios(
            vertical_axis=vertical_axis
        )
        # box_aspect = tuple(np.roll(box_aspect, shift=1))
        # ax.view_init(vertical_axis="y")
        ax.set_box_aspect(aspect=box_aspect)
        ax.set_proj_type(proj_type="ortho")
        # ax1.view_init(vertical_axis="y")
        ax1.set_box_aspect(aspect=box_aspect)
        ax1.set_proj_type(proj_type="ortho")
        ax1.shareview(ax)

        labels = ["x", "y", "z"]

        xlabel, ylabel, zlabel = self.roll_to_vertical_axis(
            labels, vertical_axis=vertical_axis
        )

        # Common settings for both axes
        for ax_i in [ax, ax1]:
            # ax_i.view_init(vertical_axis="y")

            ax_i.set(
                xlabel=xlabel,
                ylabel=ylabel,
                zlabel=zlabel,
                xlim=xlim,
                ylim=ylim,
                zlim=zlim,
                title=aircraft.name,
            )

            # Setting the view angle to make y-axis appear vertical

            # ax_i.view_init(azim=-30, elev=-210, roll=90)  #
        # Plot surfaces
        ls = LightSource(azdeg=-35, altdeg=45)
        for surface in aircraft.surfaces:
            self.plot_surface(surface, ax, ls=ls, vertical_axis=vertical_axis)

        # Plot curves and borders
        for surface in aircraft.surfaces:
            for curve in chain(surface.curves, surface.borders):
                self.plot_curve(curve, ax1, vertical_axis=vertical_axis)

        # fig.canvas.mpl_connect("motion_notify_event", on_move)
        # plt.get_current_fig_manager().window.showMaximized()
        fig.show()

        return fig


# Usage example (assuming you have an aircraft and surfaces ready):
# plotter = AircraftPlotter(aircraft, surfaces)
# plotter.plot_aircraft(plot_num=1)

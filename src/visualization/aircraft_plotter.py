from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

from src.geometry.geometry import GeometricSurface


class AircraftPlotter:
    def __init__(self, aircraft, surfaces: list[GeometricSurface]):
        """
        Initializes the plotter with an aircraft and its geometric surfaces.

        Parameters:
        aircraft (Aircraft): The aircraft to be plotted.
        surfaces (list[GeometricSurface]): The geometric surfaces of the aircraft.
        """
        self.aircraft = aircraft
        self.surfaces = surfaces

    def _find_aspect_ratios(self):
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

        return xlim, ylim, zlim, box_aspect

    def plot_aircraft(self, plot_num=1):
        """
        Plots the aircraft using the geometric data.

        Parameters:
        plot_num (int): Plot number identifier.
        """
        fig = plt.figure(num=plot_num, clear=True, figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 2, projection="3d")
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")

        # Common settings for both axes
        for ax_i in [ax, ax1]:
            ax_i.view_init(vertical_axis="y")
            ax_i.set_proj_type(proj_type="ortho")
            ax_i.set_xlabel("x")
            ax_i.set_ylabel("y")
            ax_i.set_zlabel("z")
            ax_i.set_title(self.aircraft.name)

        # Plot surfaces
        ls = LightSource(azdeg=-35, altdeg=45)
        for surface in self.surfaces:
            surface.add_surf_plot(ax, ls=ls)

        # Plot curves and borders
        for surface in self.surfaces:
            for curve in chain(surface.curves, surface.borders):
                data = curve.data
                ax1.plot3D(*data.T)

        # Set aspect ratios
        xlim, ylim, zlim, box_aspect = self._find_aspect_ratios()
        for ax_i in [ax, ax1]:
            ax_i.set_xlim(xlim)
            ax_i.set_ylim(ylim)
            ax_i.set_zlim(zlim)
            ax_i.set_box_aspect(aspect=box_aspect)

        # Event handling for synchronization of axes
        def on_move(event):
            if event.inaxes in [ax, ax1]:
                azimuth, elevation = event.inaxes.azim, event.inaxes.elev
                for ax_i in [ax, ax1]:
                    ax_i.azim = azimuth  # type: ignore
                    ax_i.elev = elevation
                fig.canvas.draw()

        fig.canvas.mpl_connect("motion_notify_event", on_move)
        # plt.get_current_fig_manager().window.showMaximized()

        return fig


# Usage example (assuming you have an aircraft and surfaces ready):
# plotter = AircraftPlotter(aircraft, surfaces)
# plotter.plot_aircraft(plot_num=1)

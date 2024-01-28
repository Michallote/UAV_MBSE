from abc import ABC, abstractmethod


class BaseAircraftPlotter(ABC):
    """Represents an Abstract Base Plotter for different backends"""

    @abstractmethod
    def plot_aircraft(self, aircraft):
        """
        Plot an aircraft.

        Args:
            aircraft: The aircraft to plot.
        """

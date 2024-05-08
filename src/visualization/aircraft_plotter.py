from typing import Literal, overload

from src.visualization.base_class import BaseAircraftPlotter
from src.visualization.matplotlib_plotter import MatplotlibAircraftPlotter
from src.visualization.plotly_plotter import PlotlyAircraftPlotter


class AircraftPlotter:
    """
    A factory class for creating aircraft plotter instances
    based on different backend technologies.
    """

    @overload
    @staticmethod
    def get_plotter(
        backend: Literal["Plotly"] | Literal["plotly"],
    ) -> PlotlyAircraftPlotter:
        ...

    @overload
    @staticmethod
    def get_plotter(
        backend: Literal["Matplotlib"] | Literal["matplotlib"],
    ) -> MatplotlibAircraftPlotter:
        ...

    @staticmethod
    def get_plotter(backend: str = "Plotly") -> BaseAircraftPlotter:
        """
        Returns a plotter class based on the specified backend.

        Args:
            backend (str): The name of the backend to use for plotting.
                           Options are 'Matplotlib' or 'Plotly'.

        Returns:
            object: A class reference to either MatplotlibAircraftPlotter or PlotlyAircraftPlotter.

        Raises:
            ValueError: If an unsupported backend is specified.
        """
        if backend.lower() == "matplotlib":
            return MatplotlibAircraftPlotter()
        elif backend.lower() == "plotly":
            return PlotlyAircraftPlotter()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

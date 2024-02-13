from src.aerodynamics.airfoil import AirfoilFactory
from src.aerodynamics.data_structures import Aircraft
from src.geometry.aircraft_geometry import AircraftGeometry
from src.structures.spar import find_instersection_region  # type: ignore
from src.structures.structural_model import StructuralModel
from src.visualization import AircraftPlotter


def main():
    airfoil_factory = AirfoilFactory()
    airfoil_factory.set_folder_path("data/airfoils")
    airfoil_factory.cache_airfoils()
    aircraft = Aircraft.from_xml("data/xml/Mobula2.xml")

    aircraft_geom = AircraftGeometry(aircraft)
    aircraft_geom.export_curves(
        output_path="data/output", reference_system="SW", units="mm"
    )
    visualizer = AircraftPlotter.get_plotter(backend="Plotly")

    visualizer.plot_aircraft(aircraft_geom)

    surface = aircraft_geom.surfaces[0]

    find_instersection_region(surface)


if __name__ == "__main__":
    main()

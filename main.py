from src.aerodynamics.airfoil import AirfoilFactory
from src.aerodynamics.data_structures import Aircraft
from src.geometry.geometry import GeometryProcessor
from src.visualization.aircraft_plotter import AircraftPlotter


def main():
    airfoil_factory = AirfoilFactory()
    airfoil_factory.set_folder_path("data/airfoils")
    airfoil_factory.cache_airfoils()

    aircraft = Aircraft.from_xml("data/xml/Mobula2.xml")

    geometry = GeometryProcessor(aircraft)
    geometry.export_curves(output_path="data/output", reference_system="SW", units="mm")
    visualizer = AircraftPlotter(aircraft, geometry.surfaces)

    visualizer.plot_aircraft()


if __name__ == "__main__":
    main()

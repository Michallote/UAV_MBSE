from src.aerodynamics.airfoil import AirfoilFactory
from src.aerodynamics.data_structures import Aircraft
from src.geometry.geometry_processing import AircraftGeometry
from src.structures.structural_model import StructuralModel
from src.visualization import AircraftPlotter  # type: ignore


def main():
    airfoil_factory = AirfoilFactory()
    airfoil_factory.set_folder_path("data/airfoils")
    airfoil_factory.cache_airfoils()
    aircraft = Aircraft.from_xml("data/xml/Mobula2.xml")

    aircraft_geom = AircraftGeometry(aircraft)
    aircraft_geom.export_curves(
        output_path="data/output", reference_system="SW", units="mm"
    )
    visualizer = AircraftPlotter.get_plotter(backend="Matplotlib")

    visualizer.plot_aircraft(aircraft_geom)

    structural_model = StructuralModel(aircraft_geom, max_rib_spacing=0.15)
    structural_model.calculate_ribs()
    input()


if __name__ == "__main__":
    main()

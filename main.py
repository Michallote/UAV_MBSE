from src.aerodynamics.airfoil import AirfoilFactory
from src.aerodynamics.data_structures import Aircraft, SurfaceType
from src.geometry.aircraft_geometry import AircraftGeometry
from src.structures.spar import FlatSpar, TorsionBoxSpar
from src.structures.structural_model import Material, StructuralModel
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

    balsa = Material()

    structure_config = {
        SurfaceType.MAINWING: dict(
            main_spar=TorsionBoxSpar, secondary_spar=FlatSpar, material=balsa
        )
    }

    structure = StructuralModel(aircraft_geom)

    surface = aircraft_geom.surfaces[0]

    optimum = TorsionBoxSpar.find_maximum_moment_of_inertia(surface, thickness=0.0003)
    print(optimum)


if __name__ == "__main__":
    main()

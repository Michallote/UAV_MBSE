from consts import MONOKOTE_THICKNESS, XML_MATERIAL_LIBRARY
from src.aerodynamics.airfoil import AirfoilFactory
from src.aerodynamics.data_structures import Aircraft, SurfaceType
from src.geometry.aircraft_geometry import AircraftGeometry
from src.materials import MaterialLibrary
from src.structures.spar import FlatSpar, TorsionBoxSpar
from src.structures.structural_model import StructuralModel
from src.utils.units import Units as units
from src.visualization import AircraftPlotter


def main():

    airfoil_factory = AirfoilFactory()
    airfoil_factory.set_folder_path("data/airfoils")
    airfoil_factory.cache_airfoils()

    aircraft = Aircraft.from_xml("data/xml/Mobula2.xml")

    te_gap_config = {
        SurfaceType.MAINWING: {"te_gap_width": 3.0 * units.mm, "blend_distance": 0.75},
        SurfaceType.ELEVATOR: {"te_gap_width": 2.5 * units.mm},
        SurfaceType.FIN: {"te_gap_width": 2.5 * units.mm},
    }

    aircraft.set_trailing_edge_gaps(te_gap_config)

    aircraft_geom = AircraftGeometry(aircraft)
    aircraft_geom.export_curves(
        output_path="data/output", ext="sldcrv", reference_system="SW", units="mm"
    )

    visualizer = AircraftPlotter.get_plotter(backend="Plotly")

    visualizer.plot_aircraft(aircraft_geom)

    materials = MaterialLibrary()
    materials.load_materials(XML_MATERIAL_LIBRARY)
    balsa = materials["balsa"]
    triplay = materials["triplay"]
    monokote = materials["monokote"]

    main_flat_spar_balsa = {
        "strategy": FlatSpar,
        "material": balsa,
        "thickness": 0.003175,
    }

    secondary_flat_spar_balsa = {
        "strategy": FlatSpar,
        "material": balsa,
        "thickness": 0.003175,
        "chord_position": 0.64,
    }

    main_spar_triplay = {
        "strategy": TorsionBoxSpar,
        "material": triplay,
        "thickness": 0.003,
    }
    rib_config = {"max_spacing": 0.15, "material": balsa, "thickness": 0.003175}
    coating_config = {"material": monokote, "thickness": MONOKOTE_THICKNESS}

    structure_config = {
        SurfaceType.MAINWING: dict(
            main_spar=main_spar_triplay,
            secondary_spar=secondary_flat_spar_balsa,
            ribs=rib_config,
            surface_coating=coating_config,
        ),
        SurfaceType.ELEVATOR: dict(
            main_spar=main_flat_spar_balsa,
            secondary_spar=secondary_flat_spar_balsa,
            ribs=rib_config,
            surface_coating=coating_config,
        ),
        SurfaceType.FIN: dict(
            main_spar=main_flat_spar_balsa,
            secondary_spar=secondary_flat_spar_balsa,
            ribs=rib_config,
            surface_coating=coating_config,
        ),
    }

    structure = StructuralModel(aircraft_geom, structure_config)

    print("Structure Mass Center of Gravity: \n", structure.mass)
    print("Structure Tensor of Inertia: \n", structure.inertia)

    visualizer.plot_structure(structure)


if __name__ == "__main__":
    main()

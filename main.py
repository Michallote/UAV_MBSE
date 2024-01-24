from src.aerodynamics.airfoil import AirfoilFactory
from src.aerodynamics.data_structures import Aircraft
from src.utils.xml_parser import parse_xml_file


def main():
    airfoil_factory = AirfoilFactory()
    airfoil_factory.set_folder_path("data/airfoils")
    airfoil_factory.cache_airfoils()

    airfoil = airfoil_factory.create_airfoil("GOE 383 AIRFOIL")

    plane_data = parse_xml_file("data/xml/Mobula2.xml")
    aircraft = Aircraft.from_dict(plane_data)


if __name__ == "__main__":
    main()

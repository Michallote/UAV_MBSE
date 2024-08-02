# UAV_MBSE

## Overview
UAV_MBSE (Unmanned Aerial Vehicle Model-Based Systems Engineering) is a project aimed at applying MBSE methodologies to the design, analysis, and validation of UAV systems. This repository contains all relevant models, documentation, and code necessary for the development and simulation of UAVs using MBSE techniques.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
This project leverages Model-Based Systems Engineering (MBSE) to streamline the development process of UAVs. By utilizing MBSE, we aim to enhance the consistency, traceability, and integration of UAV system models throughout the development lifecycle.


### Project Organization
------------

    .
    ├── LICENSE
    ├── README.md
    ├── pyproject.toml
    ├── consts.py
    ├── main.py
    ├── requirements_base.txt
    ├── requirements_full.txt
    ├── data
    │   ├── airfoils
    │   │   ├── FX 73-CL2-152.dat
    │   │   ├── GOE 383 AIRFOIL.dat
    │   │   └── NACA 0009.dat
    │   ├── analysis
    │   │   └── Mobula_T1-16_0 m_s-VLM2.xml
    │   ├── databases
    │   │   ├── airfoil_coordinates_db.zip
    │   │   └── airfoil_polars_db.zip
    │   ├── materials
    │   │   └── xml_libraries
    │   │       └── Materiales_Engineering_Data.xml
    │   ├── output
    │   │   └── Mobula.html
    │   └── xml
    │       ├── Mobula2.xml
    │       ├── Mobula_TE.xml
    │       ├── Mobula.xml
    │       └── test_sample.xml
    ├── mathematica
    │   ├── AreaIntegrals-GreenStokes.nb
    │   ├── Export to Python.nb
    │   └── ThesisTestRotatingFrames.nb
    ├── scripts
    │   ├── cantileaver_beam.py
    │   ├── file_leading_edge.py
    │   ├── file_renamer.py
    │   ├── interpolation_applications.py
    │   ├── poc_add_plot_synchronization.py
    │   └── reflection_plots.py
    ├── src
    │   ├── __init__.py
    │   ├── aerodynamics
    │   │   ├── __init__.py
    │   │   ├── airfoil_polar_features.py
    │   │   ├── airfoil.py
    │   │   ├── analisis_importer.py
    │   │   └── data_structures.py
    │   ├── geometry
    │   │   ├── __init__.py
    │   │   ├── aircraft_geometry.py
    │   │   ├── spatial_array.py
    │   │   └── surfaces.py
    │   ├── materials
    │   │   ├── __init__.py
    │   │   └── materials_library.py
    │   ├── propulsion
    │   │   ├── __init__.py
    │   │   └── propeller_importer.py
    │   ├── structures
    │   │   ├── __init__.py
    │   │   ├── fem_solver.py
    │   │   ├── inertia_tensor.py
    │   │   ├── spar.py
    │   │   ├── structural_model.py
    │   │   └── temp_abc.py
    │   ├── utils
    │   │   ├── __init__.py
    │   │   ├── interpolation.py
    │   │   ├── intersection.py
    │   │   ├── transformations.py
    │   │   ├── units.py
    │   │   └── xml_parser.py
    │   └── visualization
    │       ├── __init__.py
    │       ├── aircraft_plotter.py
    │       ├── base_class.py
    │       ├── matplotlib_plotter.py
    │       └── plotly_plotter.py
    └── tests
        ├── test_intersection_algorithms.py
        ├── test_linear_interpolation.py
        ├── test_spar_creation.py
        ├── test_te_gap_algorithm.py
        └── test_xml_parser.py

    19 directories, 72 files

Made with tree command:
```bash 
tree -I '__pycache__|legacy'
```
## Prerequisites
Before downloading the repository make sure to have installed python, git and git-lfs (Large File System).
Optionally, for development purposes it is recommended to install Visual Studio Code as an IDE.

git-lfs is only neccessary if you want to download the databases .zip files:
- `data/databases/airfoil_coordinates_db.zip`
- `data/databases/airfoil_polars_db.zip`


### On Linux
On Debian-based distributions.

Install git:

    sudo apt install git-all

Install git-lfs: https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md

### On Windows

Install git and git-lfs:

Git LFS is included in the distribution of Git for Windows: https://gitforwindows.org/


## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Michallote/UAV_MBSE.git
    cd UAV_MBSE
    ```

2. Download the databases:
    ```bash
    git lfs fetch --all
    git lfs pull
    ```

3. It is highly advisable to create a virtual environment for this project:
    ```bash
    python -m venv .venv
    ```
    Here, .venv is the name of the virtual environment directory. You can name it anything you like.

4. Activate the virtual environment:
    Activating the virtual environment will isolate your Python/Django setup and ensure that all the dependencies are maintained within this environment. The activation command differs based on your operating system.

    - On Windows:

        ```bash
        .venv\Scripts\activate
        ```
    - On macOS and Linux:

        ```bash
        source .venv/bin/activate
        ```

5. Install the necessary dependencies:

    For running scripts only:
    ```bash
    pip install -r requirements_base.txt
    ```

    For development:
    ```bash
    pip install -r requirements_dev.txt
    ```


## Usage
Instructions on how to use the project:

1. Navigate to the project directory:
    ```bash
    cd UAV_MBSE
    ```

2. Run the main script:
    ```bash
    python main.py
    ```

3. For additional functionality and running simulations, refer to the documentation provided in the `docs` directory.

## Features
- **Model-Based Design:** Utilize MBSE methodologies for UAV design.
- **Simulation:** Perform comprehensive simulations of UAV models.
- **Analysis Tools:** Tools for analyzing UAV performance and design.
- **Integration:** Seamless integration with other MBSE tools and platforms.

## Documentation
Detailed documentation is available in the `docs` directory. This includes user guides, API documentation, and model descriptions.

## Contributing
We welcome contributions from the community. To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

Please ensure that your code adheres to the coding standards and includes appropriate tests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements
We would like to thank all contributors and supporters of this project.

---

For more information, please visit the [project repository](https://github.com/Michallote/UAV_MBSE).


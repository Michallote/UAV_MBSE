# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 02:18:45 24

@author: Michel Gordillo
"""

import io
import os
import re

import numpy as np
import pandas as pd


def read_propeller_apc_file(file_path: str) -> pd.DataFrame:
    """
    Reads a propeller data file and returns its content as a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the propeller data file.

    Returns:
    - pd.DataFrame: DataFrame containing the propeller data.

    Raises:
    - FileNotFoundError: If the file is not found at the specified path.
    - ValueError: If the file content is not as expected.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Store every index of new tables delimited by the RPM header
    sep = [i for i, line in enumerate(lines) if "RPM" in line]

    # RPM values
    rpm_list = [float(lines[element].rstrip().split()[-1]) for element in sep]

    sep.append(len(lines) - 1)

    # Check for tair files:
    for line in lines[: sep[0]]:
        if "TAIR" in line:
            line = line.strip()
            raise NotImplementedError(f"TAIR polar input file detected: {line=}")

    _table_headers = lines[sep[0] + 2].split()

    # Store the beggining line and end of each table in a list to slice
    intervals = np.array([np.array(sep[:-1]) + 1, sep[1:]]).transpose()

    # or, alternatively, there's the `ignore_index` option in the `pd.concat()` function:

    df_list = []

    for interval, rpm in zip(intervals, rpm_list):
        # Slice file
        table_lines = lines[interval[0] : interval[1]]

        # headers
        headers = table_lines[1].split()
        data_lines = table_lines[3:]
        data = pd.read_csv(
            io.StringIO("".join(data_lines)), names=headers, delim_whitespace=True
        )
        data.dropna(axis=0, inplace=True)
        data["RPM"] = rpm
        data["J"] = data["J"].astype(float)

        data.dropna(axis=0, inplace=True)
        df_list.append(data)

    df_propeller = pd.concat(df_list, ignore_index=True)

    return df_propeller


def format_apc_propeller(df_propeller: pd.DataFrame) -> pd.DataFrame:
    """
    Formats and converts units of the propeller DataFrame.

    Parameters:
    - df_propeller (pd.DataFrame): DataFrame containing the propeller data.

    Returns:
    - pd.DataFrame: DataFrame with formatted and converted propeller data.
    """
    conversion_factors = {
        "V": 0.44704,
        "PWR": 745.7,
        "Torque": 0.112985,
        "Thrust": 4.44822,
        "omega": 2 * np.pi / 60,
    }

    # % Units
    df_propeller["V"] = conversion_factors["V"] * df_propeller["V"]  # type: ignore
    df_propeller["PWR"] = conversion_factors["PWR"] * df_propeller["PWR"]
    df_propeller["Torque"] = conversion_factors["Torque"] * df_propeller["Torque"]
    df_propeller["Thrust"] = conversion_factors["Thrust"] * df_propeller["Thrust"]
    df_propeller["omega"] = conversion_factors["omega"] * df_propeller["RPM"]
    # Clean up
    df_propeller[df_propeller < 0] = None
    df_propeller.dropna(axis=0, inplace=True)

    df_propeller["EnginePower"] = df_propeller["Torque"] * df_propeller["omega"]
    df_propeller["PropellerPower"] = df_propeller["Thrust"] * df_propeller["V"]
    df_propeller["Efficiency"] = (
        df_propeller["PropellerPower"] / df_propeller["EnginePower"]
    )

    return df_propeller


def parse_database(folder_path: str) -> pd.DataFrame:
    """
    Parses a directory of APC propeller data files and compiles them into a single DataFrame.

    Parameters:
    - folder_path (str): The path to the folder containing APC propeller data files.

    Returns:
    - pd.DataFrame: A DataFrame containing the compiled data from all the APC propeller files.

    Raises:
    - FileNotFoundError: If the specified folder does not exist.
    - Exception: For any other issues encountered while parsing the files.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    db_tables = []
    for file in os.listdir(folder_path):
        try:
            filename = os.path.join(folder_path, file)
            diameter, pitch = parse_prop_parameters(filename)
            df_propeller = read_propeller_apc_file(filename)
            df_propeller = format_apc_propeller(df_propeller)

            df_propeller["diameter"] = diameter * 2.54  # in -> cm
            df_propeller["pitch"] = pitch * 2.54  # in -> cm
            df_propeller["name"] = file.replace(".dat", "")

            db_tables.append(df_propeller)
        except NotImplementedError as e:
            print(f"Error processing file {filename}: {e}")

    if not db_tables:
        raise ValueError("No valid data found in the provided folder.")

    df_db = pd.concat(db_tables)
    df_db["name"] = df_db["name"].astype("category")

    return df_db


def parse_prop_parameters(file_path: str) -> tuple[float, float]:
    """Retrieve diameter and pitch from each apc file

    Parameters
    ----------
    file_path : str
        File path of the .dat table of propeller data

    Returns
    -------
    tuple[float, float]
        prop diameter, prop pitch
    """
    with open(file_path, "r", encoding="utf-8") as f:
        line = f.readline()

    # Regular expression to find numeric patterns (including decimals)
    numeric_pattern = r"\d*\.?\d+"
    matches = re.findall(numeric_pattern, line)

    if len(matches) < 2:
        raise ValueError(f"Could not find enough numeric values in the line: {line}")

    diameter, pitch = matches[:2]
    return float(diameter), float(pitch)


if __name__ == "__main__":
    df_propeller = read_propeller_apc_file(
        "data/propellers/PERFILES_WEB/PERFILES2/PER3_14x14.dat"
    )
    df_propeller = format_apc_propeller(df_propeller)

    df_db = parse_database(folder_path="data/propellers/PERFILES_WEB/PERFILES2")
    import plotly.express as px

    fig = px.scatter_matrix(
        df_propeller,
        dimensions=["V", "J", "Pe", "Ct", "Cp", "Efficiency"],
        color="RPM",
    )
    fig.show()

"""
    import plotly.express as px

    fig = px.scatter(
        df_propeller,
        x="EnginePower",
        y="Thrust",
        color="RPM",
        color_continuous_scale="Viridis",
        #      range_x=[0, 1000],
        #       range_y=[0, df_propeller["Thrust"].max()],
    )
    fig.show()

    fig = px.scatter(
        df_propeller,
        x="EnginePower",
        y="Thrust",
        color="V",
        color_continuous_scale="Magma",
        #        range_x=[0, 1000],
        #        range_y=[0, df_propeller["Thrust"].max()],
    )
    fig.show()

    fig = px.scatter(
        df_propeller,
        x="V",
        y="Thrust",
        color="EnginePower",
        color_continuous_scale="Magma",
        #        range_y=[0, df_propeller["Thrust"].max()],
    )
    fig.show()

    fig = px.scatter(
        df_propeller,
        x="Thrust",
        y="EnginePower",
        color="RPM",
        color_continuous_scale="Viridis",
        #        range_x=[0, df_propeller["Thrust"].max()],
        #        range_y=[0, 1000],
    )
    fig.show()

    fig = px.scatter(
        df_propeller,
        x="EnginePower",
        y="Thrust",
        color="V",
        color_continuous_scale="Viridis",
        # range_x=[0, 1000],
        # range_y=[0, df_propeller["Thrust"].max()],
    )
    fig.show()

import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=2, shared_yaxes=True)

fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 3, 4]), row=1, col=1)

fig.add_trace(go.Scatter(x=[20, 30, 40], y=[5, 5, 5]), row=1, col=2)

fig.add_trace(go.Scatter(x=[2, 3, 4], y=[600, 700, 800]), row=2, col=1)

fig.add_trace(go.Scatter(x=[4000, 5000, 6000], y=[7000, 8000, 9000]), row=2, col=2)

fig.update_layout(title_text="Multiple Subplots with Shared Y-Axes")
fig.show()


import plotly.express as px

df = px.data.iris()
fig = px.scatter(
    df,
    x="sepal_width",
    y="sepal_length",
    color="species",
    marginal_y="violin",
    marginal_x="box",
    trendline="ols",
    template="simple_white",
)
fig.show()


import plotly.express as px

fig = px.scatter_matrix(
    df_propeller,
    dimensions=["V", "Thrust", "PWR", "RPM"],
    color="Efficiency",
)
fig.show()
"""

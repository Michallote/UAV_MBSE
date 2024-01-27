# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 02:18:45 24

@author: Michel Gordillo
"""

import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_propeller_apc_file(file_path: str) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Store every index of new tables delimited by the RPM header
    sep = [i for i, line in enumerate(lines) if "RPM" in line]

    # RPM values
    RPM_list = [float(lines[element].rstrip().split()[-1]) for element in sep]

    sep.append(len(lines) - 1)

    table_headers = lines[sep[0] + 2].split()
    print(table_headers)

    # Store the beggining line and end of each table in a list to slice
    intervals = np.array([np.array(sep[:-1]) + 1, sep[1:]]).transpose()

    # or, alternatively, there's the `ignore_index` option in the `pd.concat()` function:

    df_list = []

    for interval, RPM in zip(intervals, RPM_list):
        # Slice file
        table_lines = lines[interval[0] : interval[1]]

        # headers
        headers = table_lines[1].split()
        data_lines = table_lines[3:]
        data = pd.read_csv(
            io.StringIO("".join(data_lines)), names=headers, delim_whitespace=True
        )
        data.dropna(axis=0, inplace=True)
        data["RPM"] = RPM
        data["J"] = data["J"].astype(float)

        data.dropna(axis=0, inplace=True)
        df_list.append(data)

    df_propeller = pd.concat(df_list, ignore_index=True)

    return df_propeller


def format_apc_propeller(df_propeller):
    # % Units
    df_propeller["V"] = 0.44704 * df_propeller["V"]
    df_propeller["J"] = df_propeller["J"]
    df_propeller["Pe"] = df_propeller["Pe"]
    df_propeller["Ct"] = df_propeller["Ct"]
    df_propeller["Cp"] = df_propeller["Cp"]
    df_propeller["PWR"] = 745.7 * df_propeller["PWR"]
    df_propeller["Torque"] = 0.112985 * df_propeller["Torque"]
    df_propeller["Thrust"] = 4.44822 * df_propeller["Thrust"]
    df_propeller["omega"] = (2 * np.pi / 60) * df_propeller["RPM"]

    # % Clean-up
    df_propeller[df_propeller < 0] = None
    df_propeller.dropna(axis=0, inplace=True)

    df_propeller["EnginePower"] = df_propeller["Torque"] * df_propeller["omega"]
    df_propeller["PropellerPower"] = df_propeller["Thrust"] * df_propeller["V"]
    df_propeller["Efficiency"] = (
        df_propeller["PropellerPower"] / df_propeller["EnginePower"]
    )

    return df_propeller


def parse_database(folder_path: str):
    for file in os.listdir(folder_path):
        filename = os.path.join(folder_path, file)
        df_propeller = read_propeller_apc_file(filename)
        df_propeller = format_apc_propeller(df_propeller)


if __name__ == "__main__":
    df_propeller = read_propeller_apc_file(
        "data/propellers/PERFILES_WEB/PERFILES2/PER3_14x14.dat"
    )
    df_propeller = format_apc_propeller(df_propeller)

    file_path = "data/propellers/PERFILES_WEB/PERFILES2/PER3_14x14.dat"

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

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

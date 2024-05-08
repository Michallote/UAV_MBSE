"Reynolds" "Cl_Cd_max" "alpha_0" "alpha_i" "Cd_min" "Cl_i" "alpha_s" "Cl_max" "Cl_alpha" "grado_perdida" "Cm_0"


import os
import re

import numpy as np
import pandas as pd

folder_path = "data/databases/airfoil_polars_db"
polar_files = os.listdir(folder_path)
filename = polar_files[50]
file_path = folder_path + "/" + filename


def extract_parameters(file_path) -> tuple[dict[str, float], pd.DataFrame]:
    parameters = {}

    with open(file_path, "r") as file:
        lines = file.readlines()

    for i, line in enumerate(lines):

        if "Calculated polar for:" in line:
            name = ""

        if ("Re" in line) and ("Mach" in line) and ("Ncrit" in line):
            parameters_names = re.findall(r"\b(\w+)\s*=", line)
            matches = re.findall(r"[\d\.]+(?:\s[eE]\s\d+)?", line)
            matches = list(map(lambda x: float(x.replace(" ", "")), matches))
            parameters.update(dict(zip(parameters_names, matches)))

        if "alpha" in line:
            # Column names HAVE whitespaces in them. Disgraceful
            parsed_line = line.replace(" Xtr", "_Xtr")
            column_names = parsed_line.split()
            start_row = i + 2
            break

    values = np.loadtxt(file_path, skiprows=start_row)
    df = pd.DataFrame(values, columns=column_names)

    return parameters, df


parameters, df = extract_parameters(file_path)

df["CL_CD"] = df["CL"] / df["CD"]


"Reynolds" "Cl_Cd_max" "alpha_0" "alpha_i" "Cd_min" "Cl_i" "alpha_s" "Cl_max" "Cl_alpha" "grado_perdida" "Cm_0"

import plotly.express as px

# Assuming df is your DataFrame
# You can customize the title, x-axis, and y-axis labels as needed
labels = {key: key.title() for key in df.columns}
fig = px.scatter(
    df, x="alpha", y="CL", title="Airfoil Polars", hover_data=list(df.columns)
)

x_buttons = [
    dict(
        label=label,
        method="update",
        args=[
            {"x": [df[label]]},
            {"xaxis": {"title": labels[label]}},
        ],
    )
    for label in df.columns
]

y_buttons = [
    dict(
        label=label,
        method="update",
        args=[
            {"y": [df[label]]},
            {"yaxis": {"title": labels[label]}},
        ],
    )
    for label in df.columns
]

# Update the layout to include dropdown menus for x and y axes
fig.update_layout(
    updatemenus=[
        dict(
            buttons=x_buttons,
            direction="down",
            showactive=True,
            x=0.05,
            xanchor="left",
            y=0.95,
            yanchor="top",
        ),
        dict(
            buttons=y_buttons,
            direction="down",
            showactive=True,
            x=0.2,
            xanchor="left",
            y=0.95,
            yanchor="top",
        ),
    ]
)

# Add dropdown title using annotations
fig.update_layout(
    annotations=[
        dict(
            text="X axis:",
            showarrow=False,
            x=0.05,
            y=0.98,
            xref="paper",  # X reference set to 'paper'
            yref="paper",  # Y reference set to 'paper'
            align="left",  # Text alignment
        ),
        dict(
            text="Y axis:",
            showarrow=False,
            x=0.2,
            y=0.98,
            xref="paper",  # X reference set to 'paper'
            yref="paper",  # Y reference set to 'paper'
            align="left",  # Text alignment
        ),
    ]
)

fig.show()

import os
import webbrowser

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from bs4 import BeautifulSoup
from plotly.offline import plot
from plotly.subplots import make_subplots

# Read data from a csv
z_data = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv"
)

trace1 = go.Surface(scene="scene1", z=z_data.values, colorscale="Blues")
trace2 = go.Surface(scene="scene2", z=z_data.values, colorscale="Greens")

f = make_subplots(rows=1, cols=2, specs=[[{"is_3d": True}, {"is_3d": True}]])

f.append_trace(trace1, 1, 1)
f.append_trace(trace2, 1, 2)

fig = go.Figure(f)

# get the a div
div = plot(fig, include_plotlyjs=False, output_type="div")
# retrieve the div id (you probably want to do something smarter here with beautifulsoup)

# Use BeautifulSoup to parse the HTML and extract the div id
soup = BeautifulSoup(div, "html.parser")
div_id = soup.find("div", class_="plotly-graph-div")["id"]  # type: ignore

# your custom JS code
js = """
    <script>
    var gd = document.getElementById('_div_id_');
    var isUnderRelayout = false;
    
    function updateCamera(sceneToUpdate, newCamera) {
        if (!isUnderRelayout) {
            isUnderRelayout = true;
            Plotly.relayout(gd, sceneToUpdate, newCamera)
                .then(() => { isUnderRelayout = false });
        }
    }
    
    gd.on('plotly_relayout', (eventData) => {
        if (eventData['scene.camera']) {
            updateCamera('scene2.camera', gd.layout.scene.camera);
        }
        else if (eventData['scene2.camera']) {
            updateCamera('scene.camera', gd.layout.scene2.camera);
        }
    });
    </script>"""

js = js.replace("_div_id_", div_id)

# Merge everything
html_content = (
    '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>' + div + js
)

# Write to an HTML file

filename = "Mobula"

file_path = f"{filename}.html"
with open(file_path, "w") as file:
    file.write(html_content)

# Open in default browser
webbrowser.open("file://" + os.path.realpath(file_path))

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from IPython.core.display import HTML
from IPython.display import display
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
div_id = div.split("=")[1].split()[0].replace("'", "").replace('"', "")
# your custom JS code
js = """
    <script>
    var gd = document.getElementById('{div_id}');
    var isUnderRelayout = false

    gd.on('plotly_relayout', () => {{
      console.log('relayout', isUnderRelayout)
      if (!isUnderRelayout) {{
        Plotly.relayout(gd, 'scene2.camera', gd.layout.scene.camera)
          .then(() => {{ isUnderRelayout = false }}  )
      }}

      isUnderRelayout = true;
    }})
    </script>""".format(
    div_id=div_id
)
# merge everything
div = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>' + div + js
# show the plot
display(HTML(div))

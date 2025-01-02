import json

from src.utils.xml_parser import parse_xml_file
import graphviz


def main():

    plane_data = parse_xml_file("data/xml/test_sample.xml")

    with open("data/xml/test_sample.json", mode="w", encoding="utf-8") as f:
        json.dump(plane_data, f)



# Load JSON graph data
with open("data/xml/test_sample.json") as f:
    graph_data = json.load(f)

# Initialize DOT graph
G = graphviz.Graph()

# Iterate through nodes and edges
for node_id, node_data in graph_data.items():
    G.node(node_id, node_data["label"])
    for neighbor_id, edge_data in node_data.get("neighbors", {}).items():
        G.edge(node_id, neighbor_id, label=edge_data["label"])

# Write DOT file
G.write("data/xml/test_sample.dot")


# main()

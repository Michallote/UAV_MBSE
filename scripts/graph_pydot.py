import json

import pydot


def parse_json_to_graph(data, graph, parent_name=None):
    """
    Recursively parse JSON data and add nodes and edges to the graph.
    """
    for key, value in data.items():
        if isinstance(value, dict):
            node_name = key
            graph.add_node(pydot.Node(node_name, label=key))
            if parent_name:
                graph.add_edge(pydot.Edge(parent_name, node_name))
            parse_json_to_graph(value, graph, node_name)
        elif isinstance(value, list):
            node_name = key
            graph.add_node(pydot.Node(node_name, label=key))
            if parent_name:
                graph.add_edge(pydot.Edge(parent_name, node_name))
            for index, item in enumerate(value):
                child_node_name = f"{key}_{index}"
                graph.add_node(pydot.Node(child_node_name, label=f"{key} [{index}]"))
                graph.add_edge(pydot.Edge(node_name, child_node_name))
                if isinstance(item, dict):
                    parse_json_to_graph(item, graph, child_node_name)
        else:
            leaf_node_name = f"{parent_name}_{key}"
            label = f"{key}: {value}"
            graph.add_node(pydot.Node(leaf_node_name, label=label))
            if parent_name:
                graph.add_edge(pydot.Edge(parent_name, leaf_node_name))


def json_to_dot(json_path, dot_output_path):
    """
    Converts JSON data into a DOT file that can be used with Graphviz to create a diagram.
    """
    with open(json_path) as json_file:
        data = json.load(json_file)

    graph = pydot.Dot(graph_type="digraph")
    parse_json_to_graph(data, graph)
    graph.write(dot_output_path, format="dot")


# Example usage
json_to_dot("data/xml/test_sample.json", "output_graph.dot")

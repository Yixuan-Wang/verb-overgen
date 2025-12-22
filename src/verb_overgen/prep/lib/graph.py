"""
Utilities for working with dependency parse as graphs.
"""

from typing import Any
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from ..common import Parse


def create_parse_graph(parse: Parse):
    """
    Convert a parse in `dict` of adjacency lists to a `networkx.DiGraph` object.
    <ROOT> is at node 0.
    Words and relations are stored as node and edge attribute `"label"`, respectively.

    Args:
        parse (Parse): The parse to convert.

    Returns:
        nx.DiGraph: The parse as a graph.
    """
    g = nx.DiGraph([(fro, to) for fro, to in enumerate(parse["arc"], 1)])
    nx.set_node_attributes(g, {i: {"label": w} for i, w in enumerate(parse["word"], 1)})
    nx.set_node_attributes(g, {0: {"label": "<ROOT>"}})
    nx.set_edge_attributes(
        g,
        {
            (fro, to): {"label": r}
            for (fro, to), r in zip(enumerate(parse["arc"], 1), parse["rel"])
        },
    )
    return g

def add_attr_to_graph(
    graph_: nx.DiGraph,
    **attrs: list[Any],
):
    """
    Add attributes to the graph.

    Args:
        graph_ (nx.DiGraph): The graph to add attributes to.
        **attrs (dict[str, list[Any]]): The attributes to add.

    Returns:
        nx.DiGraph: The graph with the attributes added.
    """
    for k, v in attrs.items():
        nx.set_node_attributes(graph_, {i + 1: {k: v[i]} for i in range(len(v))})

    return graph_

def inject_pos_to_graph(
    graph_: nx.DiGraph,
    pos: list[str],
):
    """
    Inject the POS tags into the graph as node labels and node attributes.

    Args:
        graph_ (nx.DiGraph): The graph to inject the POS tags into.
        pos (list[str]): The list of POS tags.

    Returns:
        nx.DiGraph: The graph with the POS tags injected.
    """
    for i, data in graph_.nodes.data():
        if i > 0:
            data["label"] += f"\n{pos[i - 1]}"
            data["pos"] = pos[i - 1]

    return graph_


def draw_parse_graph(
    graph: nx.DiGraph,
    *,
    highlight_node = lambda g, n, d: (None, None),
    highlight_edge = lambda g, u, v, d: (None, None),
):
    """
    Draw a `nx.DiGraph` object representing a parse on a matplotlib figure.

    Args:
        graph (nx.DiGraph): The parse to draw.

    Returns:
        matplotlib.figure.Figure: The figure containing the parse.
    """
    # Position all nodes in a line (y=0) at integer x-coordinates
    pos = {i: (i, 0) for i in range(len(graph.nodes()))}
# Draw the nodes, specifying the node label from the 'label' attribute
    fig, ax = plt.subplots(figsize=(max(len(graph.nodes()), 12), 6))
    nx.draw_networkx_nodes(graph, pos, node_size=800, node_color="none")  # Border color
    _labels = nx.draw_networkx_labels(
        graph,
        pos,
        ax=ax,
        labels={i: graph.nodes[i]["label"] for i in graph.nodes()},
        font_size=10,
        font_family=plt.rcParams["font.family"][0] if plt.rcParams["font.family"] else "sans-serif"
    )

    for i, t in _labels.items():
        weight, color = highlight_node(graph, i, graph.nodes[i])
        if color is not None:
            t.set_color(color)
        if weight is not None:
            t.set_fontweight(weight)



    # # Draw edges with an arc style
    arc_rad = 0.3  # Controls curvature of the edges
    edge_arcs = nx.draw_networkx_edges(
        graph, pos, arrowstyle="->", arrowsize=15, connectionstyle=f"arc3,rad={arc_rad}"
    )

    edge_labels = {(u, v): d["label"] for u, v, d in graph.edges(data=True)}

    # Draw edge labels (dependency types) at the center of the curved edge
    for ((u, v), dep_label), arc in zip(edge_labels.items(), edge_arcs): # type: ignore
        path = arc.get_path()
        mid_x = (path.vertices[0, 0] + 2 * path.vertices[1, 0] + path.vertices[2, 0]) / 4
        mid_y = (path.vertices[0, 1] + 2 * path.vertices[1, 1] + path.vertices[2, 1]) / 4

        d = graph.edges[u, v]
        weight, color = highlight_edge(graph, u, v, d)

        ax.text(
            mid_x,
            mid_y,
            dep_label,
            ha="center",
            va="center",
            color=color if color is not None else "black",
            fontsize=8,
            fontweight=weight if weight is not None else "normal",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="white", alpha=0.95),
        )

    ax.axis("off")
    return fig

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


def draw_parse_graph(graph: nx.DiGraph):
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
    fig, ax = plt.subplots(figsize=(12, 6))
    nx.draw_networkx_nodes(graph, pos, node_size=800, node_color="none")  # Border color
    nx.draw_networkx_labels(
        graph,
        pos,
        ax=ax,
        labels={i: graph.nodes[i]["label"] for i in graph.nodes()},
        font_size=10,
    )

    # Draw edges with an arc style
    arc_rad = 0.3  # Controls curvature of the edges
    nx.draw_networkx_edges(
        graph, pos, arrowstyle="->", arrowsize=15, connectionstyle=f"arc3,rad={arc_rad}"
    )

    # Prepare the edge labels
    edge_labels = {(u, v): d["label"] for u, v, d in graph.edges(data=True)}

    # Function to get the midpoint of the 'arc3' style at t = 0.5
    def arc3_midpoint(x1, y1, x2, y2, rad):
        """
        Return the coordinates of the midpoint (t=0.5) of the
        quadratic Bézier curve drawn by 'arc3, rad=...'.
        """
        dx = x2 - x1
        dy = y2 - y1

        length = np.hypot(dx, dy)
        if length == 0:
            # Avoid division by zero if both nodes are at the same position
            return x1, y1

        # Unit vector along the line from (x1, y1) to (x2, y2)
        ux, uy = dx / length, dy / length

        # A perpendicular unit vector
        # (For rad > 0, arc is "above" the line; for rad < 0, "below".)
        px, py = -uy, ux

        # Midpoint of the straight line
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        # Distance to shift the control point from the midpoint
        # arc_rad is effectively the fraction of the line length used as the "height"
        shift = rad * length / 2.0

        # Control point
        cx, cy = mx + shift * px, my + shift * py

        # Quadratic Bezier at t=0.5: B(0.5) = 0.25*P0 + 0.5*P1 + 0.25*P2
        mid_x = 0.25 * x1 + 0.5 * cx + 0.25 * x2
        mid_y = - (0.25 * y1 + 0.5 * cy + 0.25 * y2)

        return mid_x, mid_y

    # Draw edge labels (dependency types) at the center of the curved edge
    for (u, v), dep_label in edge_labels.items():
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        # Compute the true midpoint of the Arc3 at t = 0.5
        mid_x, mid_y = arc3_midpoint(x1, y1, x2, y2, arc_rad)

        ax.text(
            mid_x,
            mid_y,
            dep_label,
            ha="center",
            va="center",
            color="red",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.7),
        )

    ax.axis("off")
    return fig

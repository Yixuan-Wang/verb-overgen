"""Shared plotting helpers for finale analyses."""

from __future__ import annotations

import matplotlib.pyplot as plt

ACL_COL_WIDTH = 3.18  # acl.sty \columnwidth, inches


def apply_acl_style() -> None:
    """Apply ACL paper-style Matplotlib defaults."""
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{mathptmx}",
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def despine(ax: plt.Axes) -> None:
    """Show only the left and bottom axis lines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.tick_params(top=False, right=False)

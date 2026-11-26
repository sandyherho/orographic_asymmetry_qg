"""Figure 1: the model.

(a) A meridional-mean section through the two-layer atmosphere: the
rigid lid, the interface between the layers, the ground with the
mountain, and the zonal winds of the layers in the case that is new in
this study (westerlies aloft over surface easterlies). (b) Plan view of
the reference mountain in the doubly periodic domain. (c) The layer
winds of the cases studied: the classical barotropic cases lie on the
diagonal, and the shaded band is baroclinically unstable.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from orographic_asymmetry_qg.experiments import (SHEARS,  # noqa: E402
                                                 reference)
from orographic_asymmetry_qg.io import save_figure  # noqa: E402
from orographic_asymmetry_qg.params import (H_LAYER,  # noqa: E402
                                            height_of_eta, km)
from orographic_asymmetry_qg.plotting import (C, EAST, WEST,  # noqa: E402
                                              legend_below, line,
                                              panel_label, patch, style)
from orographic_asymmetry_qg.spectral import Grid  # noqa: E402


def arrow(ax, x, y, dx, color, lw=1.6):
    """A horizontal wind arrow."""
    ax.add_patch(FancyArrowPatch((x, y), (x + dx, y), arrowstyle="-|>",
                                 mutation_scale=9, lw=lw, color=color))


def section(ax):
    """Schematic section of the two-layer atmosphere."""
    ph = reference()
    x = np.linspace(0, 2 * np.pi, 600)
    oro = ph.orography
    hx = height_of_eta(oro.eta0) * np.exp(-((x - oro.x0) / oro.width) ** 2)
    H = 2 * H_LAYER
    xk = km(x) / 1e3
    ax.fill_between(xk, 0, hx / 1e3, color="#8c7a63", lw=0)
    iface = H_LAYER + 600 * np.exp(-((x - oro.x0 - 0.25) / 0.5) ** 2) \
        * np.sin(6 * (x - oro.x0)) + 0.25 * hx
    ax.plot(xk, iface / 1e3, color=C["grey"], lw=1.0, ls="--")
    ax.axhline(H / 1e3, color="k", lw=1.2)
    ax.fill_between(xk, iface / 1e3, H / 1e3, color=WEST, alpha=0.07,
                    lw=0)
    ax.fill_between(xk, hx / 1e3, iface / 1e3, color=EAST, alpha=0.07,
                    lw=0)
    for xa in (3.0, 12.0, 21.0):
        arrow(ax, xa, 7.8, 3.2, WEST)
        arrow(ax, xa + 3.2, 2.6, -3.2, EAST)
    ax.text(26.5, 7.8, "layer 1", va="center", ha="right", fontsize=7)
    ax.text(26.5, 2.6, "layer 2", va="center", ha="right", fontsize=7)
    ax.set_xlim(0, km(2 * np.pi) / 1e3)
    ax.set_ylim(0, 10.6)
    ax.set_xlabel("zonal distance (1000 km)")
    ax.set_ylabel("height (km)")


def plan(ax, fig):
    """Plan view of the reference mountain."""
    g = Grid(256, 128)
    h = height_of_eta(reference().orography.field(g))
    X, Y = km(g.X) / 1e3, km(g.Y) / 1e3
    pm = ax.pcolormesh(X, Y, h / 1e3, cmap="cividis", shading="auto",
                       rasterized=True)
    ax.contour(X, Y, h, levels=[200, 1000, 1800], colors="w",
               linewidths=0.5)
    cb = fig.colorbar(pm, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label("mountain height (km)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_linewidth(0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("zonal distance (1000 km)")
    ax.set_ylabel("meridional (1000 km)")


def wind_plane(ax):
    """Layer winds: regions and the cases studied."""
    ph = reference()
    lc = ph.critical_shear
    w = np.linspace(-2, 2, 2)
    ax.fill_between(w, w + lc, 2.6, color=C["grey"], alpha=0.18, lw=0)
    ax.fill_between(w, -2.6, w - lc, color=C["grey"], alpha=0.18, lw=0)
    ax.add_patch(Polygon([[-2, 0], [0, 0], [0, 2.6], [-2, 2.6]],
                         closed=True, fc=C["green"], alpha=0.10, lw=0))
    ax.plot(w, w, color="k", lw=0.8)
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    for lam in SHEARS:
        u2 = np.linspace(-1.5, 1.5, 2)
        ax.plot(u2, u2 + lam, color=C["grey"], lw=0.6, ls=":")
    ax.plot([-0.5, 0.5], [-0.5, 0.5], "o", ms=4, mfc="w", mec="k")
    ax.plot([-0.5], [1.0], "s", ms=4, color=C["green"])
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2.6)
    ax.set_aspect("equal")
    ax.set_xlabel("lower-layer wind $U_2$ (10 m/s)")
    ax.set_ylabel("upper-layer wind $U_1$ (10 m/s)")


def main():
    style()
    fig = plt.figure(figsize=(7.0, 4.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15],
                          width_ratios=[1.55, 1.0], hspace=0.45,
                          wspace=0.3)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    section(ax_a)
    plan(ax_b, fig)
    wind_plane(ax_c)
    panel_label(ax_a, "a", x=-0.07)
    panel_label(ax_b, "b", x=-0.16, y=1.08)
    panel_label(ax_c, "c", x=-0.28)
    legend_below(fig, [
        ("(a)", [patch("#8c7a63"), line(C["grey"], "--"), line(WEST),
                 line(EAST)],
         ["orography", "layer interface", "upper wind", "lower wind"]),
        ("(c)", [line("k"), line(C["grey"], ":"), patch(C["grey"]),
                 patch(C["green"]), line("k", "none", "o", "w"),
                 line(C["green"], "none", "s")],
         ["barotropic", "constant shear", "unstable",
          "westerly aloft, easterly below", "classical cases",
          "this study"])])
    save_figure(fig, "fig01_model")


if __name__ == "__main__":
    main()

"""Figure 9: the asymmetry with baroclinic eddies.

(a, b) Upper-layer potential vorticity at the end of the eddying runs
with a supercritical shear of 26 m/s and surface winds of -5 and +5
m/s. (c, d) Time-mean lower-layer streamfunction minus its zonal mean
(the stationary wave) of the same runs. (e) Time-mean form stress with
two standard errors against the linear prediction.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from orographic_asymmetry_qg.experiments import (RUN_GRID,  # noqa: E402
                                                 reference)
from orographic_asymmetry_qg.io import load_cache, save_figure  # noqa: E402
from orographic_asymmetry_qg.params import height_of_eta, km  # noqa: E402
from orographic_asymmetry_qg.plotting import (EAST, WEST,  # noqa: E402
                                              legend_below, line,
                                              panel_label, patch, style)
from orographic_asymmetry_qg.spectral import Grid  # noqa: E402


def main():
    style()
    e = load_cache("eddies")
    g = Grid(*RUN_GRID)
    X, Y = km(g.X) / 1e3, km(g.Y) / 1e3
    hgt = height_of_eta(reference().orography.field(g))
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.5,
                          wspace=0.15)
    qmax = np.percentile(np.abs(e["east_snapshot"][0]), 99)

    def stationary(tag):
        f = e[tag + "_psi_mean"][1]
        return f - f.mean(axis=1, keepdims=True)
    smax = max(np.abs(stationary("east")).max(),
               np.abs(stationary("west")).max())
    for i, tag in enumerate(("east", "west")):
        ax = fig.add_subplot(gs[0, i])
        ax.pcolormesh(X, Y, e[tag + "_snapshot"][0], cmap="RdBu_r",
                      vmin=-qmax, vmax=qmax, shading="auto",
                      rasterized=True)
        bx = fig.add_subplot(gs[1, i])
        bx.contourf(X, Y, stationary(tag),
                    levels=np.linspace(-smax, smax, 21), cmap="RdBu_r")
        for a in (ax, bx):
            a.contour(X, Y, hgt, levels=[300, 1500], colors="k",
                      linewidths=0.5)
            a.set_aspect("equal")
            if i == 0:
                a.set_ylabel("meridional (1000 km)")
            else:
                a.set_yticklabels([])
        bx.set_xlabel("zonal distance (1000 km)")
        panel_label(ax, "ab"[i], x=-0.12)
        panel_label(bx, "cd"[i], x=-0.12)
    cx = fig.add_subplot(gs[2, :])
    for k, (tag, col) in enumerate((("east", EAST), ("west", WEST))):
        cx.errorbar([k - 0.1], [e[tag + "_D_mean"]],
                    yerr=[2 * e[tag + "_D_sem"]], fmt="o", color=col,
                    capsize=3)
        cx.plot([k + 0.1], [e[tag + "_D_linear"]], "s", mfc="w", mec=col)
    cx.axhline(0, color="k", lw=0.3)
    cx.set_xticks([0, 1])
    cx.set_xticklabels(["$U_2$ = -5 m/s", "$U_2$ = +5 m/s"])
    cx.set_xlim(-0.6, 1.6)
    cx.set_ylabel("form stress")
    panel_label(cx, "e", x=-0.07)
    legend_below(fig, [
        ("(a-d)", [patch("#b2182b"), patch("#2166ac"), line("k")],
         ["positive", "negative", "mountain 0.3 and 1.5 km"]),
        ("(e)", [line("k", "none", "o"), line("k", "none", "s", "w")],
         ["time mean, two standard errors", "linear theory"])])
    save_figure(fig, "fig09_eddies")


if __name__ == "__main__":
    main()

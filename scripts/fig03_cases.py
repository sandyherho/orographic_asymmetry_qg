"""Figure 3: The classical westerly and easterly cases, and the sheared case.

(a, b) Meridional displacement of air crossing an infinite ridge
(250 m in a 10 km column) at +5 and -5 m/s: the exact steady QG
streamline with an 8 day drag, and the parcel argument integrated
as a nonlinear trajectory equation with the same drag. (c) Largest
departure of the parcel equation from exact QG against ridge height.
(d-f) Lower-layer
streamfunction and parcel paths of the steady nonlinear two-layer flow
over the 2.2 km reference mountain: (d) surface westerlies, (e) surface
easterlies, both without shear, and (f) surface easterlies beneath
westerlies aloft (shear 15 m/s).
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from orographic_asymmetry_qg.io import load_cache, save_figure  # noqa: E402
from orographic_asymmetry_qg.params import (height_of_eta,  # noqa: E402
                                            km)
from orographic_asymmetry_qg.plotting import (C, legend_below,  # noqa: E402
                                              line, panel_label, patch,
                                              style)

MAP_X = (0.0, 4.2)          # nondimensional window of the maps
MAP_Y = (0.55, 2.6)


def ridge_panel(ax, h, name, depth=10000.0):
    """Displacement across the ridge for one wind direction."""
    x = km(h["ridge_x"]) / 1e3
    sc = km(1.0)
    y1 = h["ridge_%s_y_drag0.65" % name] * sc
    lim = 1.25 * max(np.abs(h["ridge_west_y_drag0.65"]).max(),
                     np.abs(h["ridge_east_y_drag0.65"]).max()) * sc
    hh = height_of_eta(h["ridge_eta"], depth=depth)
    ax.fill_between(x, -lim, -lim + 0.35 * lim * hh / hh.max(),
                    color="#cdbfa9", lw=0)
    ax.plot(x, y1, color="k", lw=1.1)
    ax.plot(km(np.mod(h["ridge_%s_parcel_x" % name], 2 * np.pi)) / 1e3,
            h["ridge_%s_parcel_y" % name] * sc, color=C["orange"],
            ls=(0, (3, 2)), lw=1.4)
    ax.axhline(0, color="k", lw=0.3)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("zonal distance (1000 km)")
    ax.set_ylabel("meridional displacement (km)")


def scan_panel(ax, h):
    """Departure of the parcel equation from exact QG."""
    hs = h["scan_heights"]
    ax.loglog(hs, h["scan_error"][:, 0], "o-", color=C["blue"], ms=3.5)
    ax.loglog(hs, h["scan_error"][:, 1], "s-", color=C["red"], ms=3.5)
    ax.loglog(hs, h["scan_error"][2, 0] * (hs / hs[2]) ** 2,
              color=C["grey"], ls="--", lw=0.8)
    ax.set_xlabel("ridge height (m)")
    ax.set_ylabel("parcel - QG (relative)")


def map_panel(ax, h, name):
    """Lower-layer streamfunction and parcels for one case."""
    X, Y = np.meshgrid(h["x"], h["y"])
    sel = ((X >= MAP_X[0]) & (X <= MAP_X[1]))[0]
    sy = ((h["y"] >= MAP_Y[0]) & (h["y"] <= MAP_Y[1]))
    psi = h["%s_psi" % name][-1]
    xs, ys = km(h["x"][sel]) / 1e3, km(h["y"][sy]) / 1e3
    lev = np.linspace(-0.06, 0.06, 25)
    ax.contourf(xs, ys, psi[np.ix_(sy, sel)], levels=lev, cmap="RdBu_r",
                extend="both")
    hgt = height_of_eta(h["eta"])[np.ix_(sy, sel)]
    ax.contour(xs, ys, hgt, levels=[300, 1000, 1700], colors="#5a4a36",
               linewidths=0.6)
    p = h["%s_parcels" % name]
    for k in range(p.shape[2]):
        px, py = p[:, 0, k], p[:, 1, k]
        ok = (px >= MAP_X[0]) & (px <= MAP_X[1])
        px = np.where(ok, px, np.nan)
        ax.plot(km(px) / 1e3, km(py) / 1e3, color="k", lw=0.7)
    ax.set_aspect("equal")
    ax.set_xlim(km(MAP_X[0]) / 1e3, km(MAP_X[1]) / 1e3)
    ax.set_ylim(km(MAP_Y[0]) / 1e3, km(MAP_Y[1]) / 1e3)
    ax.set_xlabel("zonal distance (1000 km)")


def main():
    style()
    h = load_cache("cases")
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 0.8], hspace=0.35,
                          wspace=1.1)
    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_s = fig.add_subplot(gs[0, 4:])
    ridge_panel(ax_a, h, "west")
    ridge_panel(ax_b, h, "east")
    scan_panel(ax_s, h)
    maps = [fig.add_subplot(gs[1, 2 * i:2 * i + 2]) for i in range(3)]
    for ax, name in zip(maps, ("west", "east", "sheared")):
        map_panel(ax, h, name)
    maps[0].set_ylabel("meridional (1000 km)")
    for ax, s in zip([ax_a, ax_b, ax_s] + maps, "abcdef"):
        panel_label(ax, s, x=-0.38 if s in "abc" else 0.0, y=1.06)
    legend_below(fig, [
        ("(a-c)", [line("k"), line(C["orange"], ":"), patch("#cdbfa9"),
                   line(C["blue"], marker="o"), line(C["red"], marker="s"),
                   line(C["grey"], "--")],
         ["exact QG", "parcel equation", "ridge (250 m)",
          "westerly", "easterly", "height squared"]),
        ("(d-f)", [patch("#b2182b"), patch("#2166ac"), line("k"),
                   line("#5a4a36")],
         ["anticyclonic", "cyclonic", "parcel paths",
          "mountain 0.3, 1, 1.7 km"])])
    save_figure(fig, "fig03_cases")


if __name__ == "__main__":
    main()

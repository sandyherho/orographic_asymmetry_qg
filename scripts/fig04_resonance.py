"""Figure 4: stationary orographic resonance in the two-layer model.

(a) Number of free stationary Rossby waves over the plane of the layer
winds, with the barotropic diagonal on which the two classical cases lie and
the baroclinically unstable bands; contours give the zonal wavelength of
the longest stationary wave. (b) Stationary total wavenumbers along the
surface wind for the reference shears, with the barotropic law
``sqrt(beta / U)``.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from orographic_asymmetry_qg.io import load_cache, save_figure  # noqa: E402
from orographic_asymmetry_qg.params import km  # noqa: E402
from orographic_asymmetry_qg.plotting import (SHEAR_COLORS,  # noqa: E402
                                              legend_below, line,
                                              panel_label, patch, style)

CMAP = ListedColormap(["#f2f2f2", "#cfe3f3", "#7fb1d9"])


def main():
    style()
    r = load_cache("resonance")
    W, count = r["W"], r["count"]
    lc = float(r["critical_shear"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.3),
                                 gridspec_kw=dict(width_ratios=[1, 1.15]))
    fig.subplots_adjust(wspace=0.32)
    ax.pcolormesh(W, W, count, cmap=CMAP, vmin=-0.5, vmax=2.5,
                  shading="auto", rasterized=True)
    lam_km = km(2 * np.pi / r["kmin"]) / 1e3
    cs = ax.contour(W, W, lam_km, levels=[1, 2, 3, 4, 6], colors="k",
                    linewidths=0.45)
    ax.clabel(cs, fmt="%g", fontsize=5.5)
    w = np.array([-2, 2])
    for s in (1, -1):
        ax.fill_between(w, w + s * lc, s * 3, color="k", alpha=0.12,
                        lw=0)
    ax.plot(w, w, "k", lw=0.9)
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.plot([-0.5, 0.5], [-0.5, 0.5], "o", ms=4, mfc="w", mec="k")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_xlabel("lower-layer wind $U_2$ (10 m/s)")
    ax.set_ylabel("upper-layer wind $U_1$ (10 m/s)")
    u2 = r["U2_line"]
    beta = float(r["beta"])
    for i, lam in enumerate(r["shears"]):
        for j in range(2):
            k = r["ks_line"][i, :, j].copy()
            # break the line where the root ordering changes
            jump = np.abs(np.diff(k)) > 0.5
            k[1:][jump] = np.nan
            bx.plot(u2, k, color=SHEAR_COLORS[i], lw=1.1)
    with np.errstate(invalid="ignore", divide="ignore"):
        bx.plot(u2, np.where(u2 > 0, np.sqrt(beta / u2), np.nan),
                color="w", ls=(0, (2, 2)), lw=0.8)
    bx.axvline(0, color="k", lw=0.4)
    bx.set_ylim(0, 16)
    bx.set_xlim(-2, 2)
    bx.set_xlabel("lower-layer wind $U_2$ (10 m/s)")
    bx.set_ylabel("stationary wavenumber $K_s$")
    sec = bx.secondary_yaxis(
        "right", functions=(lambda k: km(2 * np.pi / np.maximum(k, 1e-3))
                            / 1e3,
                            lambda L: 2 * np.pi
                            / np.maximum(L * 1e3 / km(1.0), 1e-9)))
    sec.set_ylabel("wavelength (1000 km)")
    with np.errstate(divide="ignore"):
        sec.set_ticks([1.5, 2, 3, 5])
    panel_label(ax, "a", x=-0.22)
    panel_label(bx, "b", x=-0.14)
    legend_below(fig, [
        ("(a)", [patch(CMAP(0)), patch(CMAP(1)), patch(CMAP(2)),
                 patch("#c8c8c8"), line("k"), line("k", "none", "o", "w")],
         ["no stationary wave", "one", "two", "unstable", "barotropic",
          "classical cases"]),
        ("(b)", [line(SHEAR_COLORS[i]) for i in range(len(r["shears"]))]
         + [line("k", ":")],
         ["shear %g m/s" % (10 * s) for s in r["shears"]]
         + [r"$\sqrt{\beta/U_2}$"])])
    save_figure(fig, "fig04_resonance")


if __name__ == "__main__":
    main()

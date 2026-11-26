"""Figure 5: linear form stress and the reversal of the asymmetry.

(a) Linear orographic form stress of the reference mountain (2.2 km)
against the lower-layer wind for the reference shears. (b) Ratio of the
mean form stress on surface easterlies to that on surface westerlies,
averaged over surface speeds up to 5, 10 and 20 m/s, against the
shear. (c) Smallest mountain height for which
the linear equilibrium curve folds (multiple equilibria), for surface
easterlies and westerlies and two relaxation rates of the mean wind.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from orographic_asymmetry_qg.io import load_cache, save_figure  # noqa: E402
from orographic_asymmetry_qg.params import height_of_eta  # noqa: E402
from orographic_asymmetry_qg.plotting import (C, EAST,  # noqa: E402
                                              SHEAR_COLORS, WEST,
                                              legend_below, line,
                                              panel_label, style)

SPEED_STYLES = ("-", "--", ":")
WINDOWS = (0.5, 1.0, 2.0)


def mean_ratio(u2, D, w):
    """Mean |D| over -w <= U2 < 0 divided by the mean over 0 < U2 <= w."""
    e = (u2 < 0) & (u2 >= -w)
    we = (u2 > 0) & (u2 <= w)
    return np.abs(D[..., e]).mean(-1) / np.abs(D[..., we]).mean(-1)


def main():
    style()
    r = load_cache("resonance")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
    fig.subplots_adjust(wspace=0.42)
    ax = axs[0]
    for i, lam in enumerate(r["shears"]):
        ax.plot(r["U2_line"], r["drag"][i], color=SHEAR_COLORS[i], lw=1.0)
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel("lower-layer wind $U_2$ (10 m/s)")
    ax.set_ylabel(r"linear form stress $\langle\eta\,\partial_x\psi_2"
                  r"\rangle$")
    ax.set_xlim(-2, 2)
    bx = axs[1]
    for j, w in enumerate(WINDOWS):
        ratio = mean_ratio(r["coarse_U2"], r["coarse_drag"], w)
        bx.semilogy(r["asym_shear"], ratio, color="k",
                    ls=SPEED_STYLES[j], lw=1.0)
    bx.axhline(1, color=C["grey"], lw=0.6)
    bx.axvline(float(r["critical_shear"]), color=C["grey"], lw=0.6,
               ls="--")
    bx.set_xlabel(r"shear $U_1 - U_2$ (10 m/s)")
    bx.set_ylabel("easterly / westerly form stress")
    cx = axs[2]
    thr = height_of_eta(r["eta_threshold"]) / 1e3
    for k, kap in enumerate(r["kappas"]):
        ls = "-" if k == 0 else "--"
        cx.semilogy(r["asym_shear"], thr[:, k, 0], color=EAST, ls=ls)
        cx.semilogy(r["asym_shear"], thr[:, k, 1], color=WEST, ls=ls)
    cx.axhline(height_of_eta(float(r["eta_ref"])) / 1e3, color=C["grey"],
               lw=0.6)
    cx.axvline(float(r["critical_shear"]), color=C["grey"], lw=0.6,
               ls="--")
    cx.set_ylim(0.2, 40)
    cx.set_xlabel(r"shear $U_1 - U_2$ (10 m/s)")
    cx.set_ylabel("mountain height for folds (km)")
    for a, s in zip(axs, "abc"):
        panel_label(a, s, x=-0.3)
    kap_days = [5.21 / k for k in r["kappas"]]
    legend_below(fig, [
        ("(a)", [line(SHEAR_COLORS[i]) for i in range(len(r["shears"]))],
         ["shear %g m/s" % (10 * s) for s in r["shears"]]),
        ("(b)", [line("k", s) for s in SPEED_STYLES]
         + [line(C["grey"], "--")],
         ["|U$_2$| up to %g m/s" % (10 * w) for w in WINDOWS]
         + ["critical shear"]),
        ("(c)", [line(EAST), line(WEST), line("k"), line("k", "--"),
                 line(C["grey"])],
         ["surface easterlies", "surface westerlies",
          "relaxation %.0f d" % kap_days[0],
          "relaxation %.0f d" % kap_days[1], "reference mountain"])])
    save_figure(fig, "fig05_linear_drag")


if __name__ == "__main__":
    main()

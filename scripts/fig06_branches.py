"""Figure 6: nonlinear steady branches and their stability.

(a-d) Departure of the steady lower-layer wind from the forcing wind,
``U_2 - U_2*``, which equals the form stress divided by ``2 kappa``,
along the forcing wind for each shear: full nonlinear steady states,
stable (black) and unstable (red), with the linear prediction (clipped
to the range of the nonlinear branch) and the folds. (e)
Largest growth rate of perturbations to the steady states.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from orographic_asymmetry_qg.io import load_cache, save_figure  # noqa: E402
from orographic_asymmetry_qg.plotting import (C, SHEAR_COLORS,  # noqa: E402
                                              legend_below, line,
                                              panel_label, style)


def segments(x, y, mask):
    """``y`` with ``nan`` outside ``mask``, widened by one point so that
    the stable and unstable pieces join."""
    m2 = mask | np.r_[mask[1:], False] | np.r_[False, mask[:-1]]
    return np.where(m2, y, np.nan)


def main():
    style()
    b = load_cache("branches")
    shears = b["shears"]
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.9], hspace=0.5,
                          wspace=0.45)
    top = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ex = fig.add_subplot(gs[1, :])
    for i, lam in enumerate(shears):
        tag = "s%g" % lam
        ax = top[i]
        us, u2 = b[tag + "_lam"], b[tag + "_U2"]
        gr = b[tag + "_eig"][:, 0].real
        dev = u2 - us
        ax.plot(b[tag + "_lin_Ustar"], b[tag + "_lin_U2"]
                - b[tag + "_lin_Ustar"], color=C["grey"], lw=0.7)
        ax.plot(us, segments(us, dev, gr <= 0), color="k", lw=1.2)
        ax.plot(us, segments(us, dev, gr > 0), color=C["red"], lw=1.6)
        for f in b[tag + "_folds"]:
            ax.plot(f[0], f[1] - f[0], "o", ms=3.5, mfc="w", mec="k")
        ax.axhline(0, color="k", lw=0.3)
        ax.axvline(0, color="k", lw=0.3)
        ax.set_xlim(-1.5, 1.8)
        lo, hi = np.nanmin(dev), np.nanmax(dev)
        pad = 0.25 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title("shear %g m/s" % (10 * lam), fontsize=7.5)
        ax.set_xlabel("forcing wind $U_2^*$ (10 m/s)")
        if i == 0:
            ax.set_ylabel("$U_2 - U_2^*$ (10 m/s)")
        panel_label(ax, "abcd"[i], x=-0.35)
        ex.plot(us, gr, color=SHEAR_COLORS[i], lw=1.0)
    ex.axhline(0, color="k", lw=0.4)
    ex.axvline(0, color="k", lw=0.4)
    ex.set_xlim(-1.5, 1.8)
    ex.set_xlabel("forcing wind $U_2^*$ (10 m/s)")
    ex.set_ylabel("largest growth rate (1/5.2 d)")
    panel_label(ex, "e", x=-0.07)
    legend_below(fig, [
        ("(a-d)", [line("k"), line(C["red"]), line(C["grey"]),
                   line("k", "none", "o", "w")],
         ["stable steady state", "unstable steady state",
          "linear theory", "fold"]),
        ("(e)", [line(SHEAR_COLORS[i]) for i in range(len(shears))],
         ["shear %g m/s" % (10 * s) for s in shears])])
    save_figure(fig, "fig06_branches")


if __name__ == "__main__":
    main()

"""Figure 7: the time-dependent flow along the forcing wind.

(a, b) Slow ramps of the forcing wind up and down (0.58 m/s per 100
days), for no shear and for the reference shear, over the steady
branches; the start-up adjustment is not shown. The offset between
the ramps is the lag of the mean wind behind the moving forcing. (c)
Time-mean departure of the mean wind from the forcing, with its
standard deviation. (d) Time-mean eddy energy. (e) Form stress against
time at the three forcing winds with the largest variability under
shear.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from orographic_asymmetry_qg.io import load_cache, save_figure  # noqa: E402
from orographic_asymmetry_qg.params import T_STAR, DAY  # noqa: E402
from orographic_asymmetry_qg.plotting import (C, EAST,  # noqa: E402
                                              SHEAR_COLORS, legend_below,
                                              line, panel_label, patch,
                                              style)

COLORS = {0.0: SHEAR_COLORS[0], 1.5: SHEAR_COLORS[2]}


def ramp_panel(ax, d, b, lam):
    """Ramps and steady branch at shear ``lam``."""
    tag = "s%g" % lam
    if b is not None and tag + "_lam" in b:
        us, u2 = b[tag + "_lam"], b[tag + "_U2"]
        gr = b[tag + "_eig"][:, 0].real
        ax.plot(us, np.where(gr <= 0, u2 - us, np.nan), color=C["grey"],
                lw=3, alpha=0.5)
        ax.plot(us, np.where(gr > 0, u2 - us, np.nan), color=EAST, lw=3,
                alpha=0.35)
    for sense, col in (("up", C["blue"]), ("down", C["orange"])):
        x = d["%s_ramp_%s_U2_star" % (tag, sense)]
        y = d["%s_ramp_%s_U2" % (tag, sense)] - x
        k0 = x.size // 30            # drop the start-up adjustment
        ax.plot(x[k0:], y[k0:], color=col, lw=0.6)
    ax.axhline(0, color="k", lw=0.3)
    ax.axvline(0, color="k", lw=0.3)
    ax.set_xlim(-1.2, 0.6)
    ax.set_title("shear %g m/s" % (10 * lam), fontsize=7.5)
    ax.set_xlabel("forcing wind $U_2^*$ (10 m/s)")
    ax.set_ylabel("$U_2 - U_2^*$ (10 m/s)")


def main():
    style()
    d = load_cache("dynamics")
    try:
        b = load_cache("branches")
    except FileNotFoundError:
        b = None
    f = d["forcings"]
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9], hspace=0.62,
                          wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ramp_panel(ax_a, d, b, 0.0)
    ramp_panel(ax_b, d, b, 1.5)
    ys = []
    for lam in d["shears"]:
        for sense in ("up", "down"):
            x = d["s%g_ramp_%s_U2_star" % (lam, sense)]
            y = d["s%g_ramp_%s_U2" % (lam, sense)] - x
            ys.append(y[x.size // 30:])
    ys = np.concatenate(ys)
    pad = 0.15 * (ys.max() - ys.min())
    ax_a.set_ylim(ys.min() - pad, ys.max() + pad)
    ax_b.set_ylim(ys.min() - pad, ys.max() + pad)
    cx = fig.add_subplot(gs[1, 0])
    dx = fig.add_subplot(gs[1, 1])
    for lam in d["shears"]:
        tag = "s%g" % lam
        dev = d[tag + "_U2"] - f[:, None]
        cx.errorbar(f, dev.mean(1), yerr=dev.std(1), color=COLORS[lam],
                    marker="o", ms=3, lw=1.0, capsize=2)
        dx.semilogy(f, d[tag + "_E"].mean(1), color=COLORS[lam],
                    marker="o", ms=3, lw=1.0)
    for a in (cx, dx):
        a.axvline(0, color="k", lw=0.3)
        a.set_xlabel("forcing wind $U_2^*$ (10 m/s)")
    cx.axhline(0, color="k", lw=0.3)
    cx.set_ylabel("time-mean $U_2 - U_2^*$ (10 m/s)")
    dx.set_ylabel("time-mean eddy energy")
    ex = fig.add_subplot(gs[2, :])
    tag = "s1.5"
    var = d[tag + "_D"].std(1)
    picks = np.argsort(var)[::-1][:3]
    t = (d[tag + "_t"] - d[tag + "_t"][0]) * T_STAR / DAY / 365.25
    for k, j in enumerate(sorted(picks)):
        ex.plot(t, d[tag + "_D"][j], lw=0.6, color=SHEAR_COLORS[k + 3],
                label="%g" % f[j])
    ex.set_xlim(0, 1.5)
    ex.set_xlabel("time (years)")
    ex.set_ylabel("form stress, shear 15 m/s")
    for a, s in zip((ax_a, ax_b, cx, dx, ex), "abcde"):
        panel_label(a, s, x=-0.05 if s != "e" else -0.08, y=1.06)
    legend_below(fig, [
        ("(a, b)", [line(C["blue"]), line(C["orange"]),
                    patch("#c8c8c8"), patch("#f4b9a8")],
         ["ramp up", "ramp down", "stable steady", "unstable steady"]),
        ("(c, d)", [line(COLORS[lam], marker="o") for lam in d["shears"]],
         ["shear %g m/s" % (10 * lam) for lam in d["shears"]]),
        ("(e)", [line(SHEAR_COLORS[k + 3]) for k in range(3)],
         ["$U_2^*$ = %g m/s" % (10 * f[j]) for j in sorted(picks)])])
    save_figure(fig, "fig07_dynamics")


if __name__ == "__main__":
    main()

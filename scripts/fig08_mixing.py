"""Figure 8: information and stirring in the time-dependent regimes.

(a) Permutation entropy and Jensen-Shannon statistical complexity of
the form stress for every unsteady run (standard deviation above 1
percent of the mean), colored by the forcing wind. (b) Spectral entropy
of the time-mean lower-layer streamfunction (the stationary wave)
along the forcing wind. (c, d) Lower-layer finite-time Lyapunov
exponents over 20 days at the forcing wind with the largest form-stress
variability, with shear and without. (e) Their distributions.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from orographic_asymmetry_qg import diagnostics as dg  # noqa: E402
from orographic_asymmetry_qg.experiments import (DYN_GRID,  # noqa: E402
                                                 RUN_GRID)
from orographic_asymmetry_qg.io import load_cache, save_figure  # noqa: E402
from orographic_asymmetry_qg.params import (DAY, T_STAR,  # noqa: E402
                                            height_of_eta, km)
from orographic_asymmetry_qg.experiments import reference  # noqa: E402
from orographic_asymmetry_qg.plotting import (SHEAR_COLORS,  # noqa: E402
                                              legend_below, line,
                                              panel_label, style)
from orographic_asymmetry_qg.spectral import Grid  # noqa: E402

MARKERS = {0.0: "s", 1.5: "o"}
COLORS = {0.0: SHEAR_COLORS[0], 1.5: SHEAR_COLORS[2]}


def ch_bounds(n=120, order=5):
    """Lower and upper bounds of the complexity-entropy plane."""
    import math
    N = math.factorial(order)

    def S(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    def point(p):
        pe = np.full(N, 1.0 / N)
        js = S(0.5 * (p + pe)) - 0.5 * S(p) - 0.5 * S(pe)
        jmax = -0.5 * ((N + 1) / N * np.log(N + 1) + np.log(N)
                       - 2 * np.log(2 * N))
        H = S(p) / np.log(N)
        return H, js / jmax * H
    lo, hi = [], []
    for pm in np.linspace(1.0 / N, 1.0, n):
        p = np.full(N, (1 - pm) / (N - 1))
        p[0] = pm
        lo.append(point(p))
    for m in range(N - 1):
        for pm in np.linspace(0, 1.0 / (N - m), 12):
            p = np.zeros(N)
            p[0] = pm
            p[1:N - m] = (1 - pm) / (N - m - 1)
            hi.append(point(p))
    hi = np.array(hi)
    o = np.argsort(hi[:, 0])
    return np.array(lo), hi[o]


def stationary_entropy(psi2):
    """Spectral entropy of the time-mean lower-layer streamfunction.

    Kinetic energy is binned in unit shells of total wavenumber; low
    entropy means the stationary wave is concentrated in few scales.
    """
    g = Grid(*DYN_GRID)
    out = []
    for f in psi2:
        ph = g.fft(f)
        e = 0.5 * g.weight * g.k2 * np.abs(ph) ** 2
        idx = np.rint(np.sqrt(g.k2)).astype(int)
        spec = np.bincount(idx[g.keep], weights=e[g.keep])
        out.append(dg.spectral_entropy(spec))
    return np.array(out)


def main():
    style()
    d = load_cache("dynamics")
    L = load_cache("lagrangian")
    f = d["forcings"]
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 0.85], hspace=0.38,
                          wspace=1.3)
    ax = fig.add_subplot(gs[0, :2])
    bx = fig.add_subplot(gs[0, 2:4])
    ex = fig.add_subplot(gs[0, 4:])
    lo, hi = ch_bounds()
    ax.plot(lo[:, 0], lo[:, 1], color="#bbbbbb", lw=0.6)
    ax.plot(hi[:, 0], hi[:, 1], color="#bbbbbb", lw=0.6)
    for lam in d["shears"]:
        tag = "s%g" % lam
        D = d[tag + "_D"]
        # unsteady runs only: std of the form stress above 1% of its mean
        ok = np.isfinite(d[tag + "_entropy"]) & (
            D.std(1) > 0.01 * np.maximum(np.abs(D.mean(1)), 1e-3))
        ax.scatter(d[tag + "_entropy"][ok], d[tag + "_complexity"][ok],
                   c=f[ok], cmap="coolwarm", vmin=-1, vmax=1, s=18,
                   marker=MARKERS[lam], edgecolors="k", linewidths=0.4)
        H = stationary_entropy(d[tag + "_psi_mean"][:, 1])
        ok = np.abs(f) > 1e-9          # no response without surface wind
        bx.plot(f[ok], H[ok], color=COLORS[lam], marker=MARKERS[lam],
                ms=3, lw=1.0)
    ax.set_xlabel("permutation entropy")
    ax.set_ylabel("statistical complexity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    bx.axvline(0, color="k", lw=0.3)
    bx.set_xlabel("forcing wind $U_2^*$ (10 m/s)")
    bx.set_ylabel("stationary-wave spectral entropy")
    edges = np.linspace(0, np.percentile(L["sheared"], 99.8), 60)
    for key, col in (("sheared", COLORS[1.5]), ("unsheared", COLORS[0.0])):
        h, _ = np.histogram(L[key], edges, density=True)
        ex.semilogy(0.5 * (edges[1:] + edges[:-1]), h, color=col, lw=1.0)
    ex.set_xlabel("FTLE (1/5.2 d)")
    ex.set_ylabel("probability density")
    g = Grid(*RUN_GRID)
    hgt = height_of_eta(reference().orography.field(g))
    X, Y = km(L["X"]) / 1e3, km(L["Y"]) / 1e3
    vmax = np.percentile(L["sheared"][0], 99.5)
    for i, key in enumerate(("sheared", "unsheared")):
        cx = fig.add_subplot(gs[1, 3 * i:3 * i + 3])
        pm = cx.pcolormesh(X, Y, L[key][0], cmap="magma", vmin=0,
                           vmax=vmax, shading="auto", rasterized=True)
        cx.contour(km(g.X) / 1e3, km(g.Y) / 1e3, hgt, levels=[300, 1500],
                   colors="w", linewidths=0.5)
        cx.set_aspect("equal")
        cx.set_xlabel("zonal distance (1000 km)")
        if i == 0:
            cx.set_ylabel("meridional distance (1000 km)")
        panel_label(cx, "cd"[i], x=-0.12)
    cb = fig.colorbar(pm, ax=fig.axes[-2:], pad=0.02, fraction=0.03,
                      shrink=0.8)
    cb.set_label("FTLE over %.0f days (1/5.2 d)"
                 % (float(L["T"]) * T_STAR / DAY), fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    for a, s in zip((ax, bx, ex), "abe"):
        panel_label(a, s, x=-0.3)
    legend_below(fig, [
        ("(a, b)", [line("k", "none", "s", "w"), line("k", "none", "o", "w")],
         ["shear 0 m/s", "shear 15 m/s"]),
        ("(a)", [line("#2166ac", "none", "o"),
                 line("#b2182b", "none", "o"), line("#bbbbbb")],
         ["easterly forcing", "westerly forcing", "bounds of the plane"]),
        ("(c-e)", [line(COLORS[1.5]), line(COLORS[0.0])],
         ["shear 15 m/s, $U_2^*$ = %g m/s" % (10 * float(L["forcing"])),
          "no shear, same forcing"])])
    save_figure(fig, "fig08_mixing")


if __name__ == "__main__":
    main()

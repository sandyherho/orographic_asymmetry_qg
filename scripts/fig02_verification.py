"""Figure 2: verification of the numerical model.

(a) Temporal convergence of ETDRK4. (b) Growth rates of two-layer
normal modes in the model against the dispersion relation with drag
and cooling. (c) Nonlinear over linear form stress as the mountain is
lowered. (d) Eddy energy against the time integral of its budget. (e)
Newton residuals. (f) Grid convergence of the steady form stress.
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
from orographic_asymmetry_qg.plotting import (C, EAST, WEST,  # noqa: E402
                                              legend_below, line,
                                              panel_label, style)


def main():
    style()
    v = load_cache("verification")
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.8))
    fig.subplots_adjust(hspace=0.55, wspace=0.5)
    a = axs[0, 0]
    dts, err = v["dts"], v["err"]
    a.loglog(dts, err, "o-", color="k", ms=3.5)
    a.loglog(dts, err[0] * (dts / dts[0]) ** 4, color=C["grey"], ls="--")
    a.set_xticks(dts)
    a.set_xticklabels(["%g" % x for x in dts], rotation=30)
    a.minorticks_off()
    a.set_xlabel("time step")
    a.set_ylabel("max relative error")
    b = axs[0, 1]
    b.plot(v["growth_kk"], v["growth_theory"], color=C["grey"])
    b.plot(v["growth_k"], v["growth_model"], "o", ms=3.5, mfc="w",
           mec="k")
    b.axhline(0, color="k", lw=0.3)
    b.set_xlabel("zonal wavenumber")
    b.set_ylabel("growth rate (1/5.2 d)")
    c = axs[0, 2]
    h = height_of_eta(v["etas"]) / 1e3
    c.semilogx(h, v["amp_west"][:, 0] / v["amp_west"][:, 1], "o-",
               color=WEST, ms=3.5)
    c.semilogx(h, v["amp_east"][:, 0] / v["amp_east"][:, 1], "s-",
               color=EAST, ms=3.5)
    c.axhline(1, color="k", lw=0.3)
    c.set_xlabel("mountain height (km)")
    c.set_ylabel("nonlinear / linear stress")
    d = axs[1, 0]
    d.plot(v["budget_t"], v["budget_dE"], color="k", lw=1.2)
    d.plot(v["budget_t"], v["budget_int"], color=C["orange"], ls="--")
    d.set_xlabel("time (5.2 d)")
    d.set_ylabel("eddy energy change")
    e = axs[1, 1]
    e.semilogy(np.arange(v["newton"].size), v["newton"], "o-", color="k",
               ms=3.5)
    e.set_xlabel("Newton iteration")
    e.set_ylabel("max residual")
    f = axs[1, 2]
    nx = v["grids"][:, 0]
    res = v["resolution"]
    f.plot(nx, res[:, 0] / res[-1, 0], "o-", color="k", ms=3.5)
    f.plot(nx, res[:, 2] / res[-1, 2], "s--", color=C["grey"], ms=3.5)
    f.axhline(1, color="k", lw=0.3)
    f.set_xticks(nx)
    f.set_xticklabels(["%dx%d" % tuple(g) for g in v["grids"]])
    f.set_xlabel("grid")
    f.set_ylabel("relative to finest grid")
    for ax, s in zip(axs.flat, "abcdef"):
        panel_label(ax, s, x=-0.32)
    legend_below(fig, [
        ("(a, b)", [line("k", marker="o"), line(C["grey"], "--"),
                    line(C["grey"]), line("k", "none", "o", "w")],
         ["ETDRK4", "fourth order", "dispersion relation", "model"]),
        ("(c-f)", [line(WEST, marker="o"), line(EAST, marker="s"),
                   line("k"), line(C["orange"], "--"),
                   line(C["grey"], "--", "s")],
         ["$U_2$ = 5 m/s, no shear", "$U_2$ = -5 m/s, shear 15 m/s",
          "energy", "integrated budget", "growth rate"])])
    save_figure(fig, "fig02_verification")


if __name__ == "__main__":
    main()

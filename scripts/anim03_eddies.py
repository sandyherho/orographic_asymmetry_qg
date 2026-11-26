"""Animation 3: baroclinic eddies over the mountain.

The two eddying runs with a supercritical shear of 26 m/s, started from
their final states: surface easterlies of 5 m/s (left) and surface
westerlies of 5 m/s (right). Each panel is an oblique, lit view of the
domain (distances and heights in km, heights exaggerated) with the
2.2 km mountain and the upper-layer potential vorticity anomaly
(10^-5 1/s, the weather map) draped over the terrain. The lower panel
traces the running time mean of the acceleration of the mean wind by
the mountain's form stress (m/s per day) in each run against its
linear prediction. Playback covers about 120 days.
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from orographic_asymmetry_qg.experiments import (RUN_GRID,  # noqa: E402
                                                 reference)
from orographic_asymmetry_qg.io import load_cache  # noqa: E402
from orographic_asymmetry_qg.model import QGModel  # noqa: E402
from orographic_asymmetry_qg.params import (DAY, T_STAR,  # noqa: E402
                                            drag_tendency, height_of_eta,
                                            km, pv_si)
from orographic_asymmetry_qg.scene import (BG, DIM, EAST, PV,  # noqa: E402
                                           TEXT, WEST, TerrainView,
                                           dark_axes, render,
                                           unit_colorbar)

FPS = 30
DURATION = 24.0
DT_FRAME = 0.032
PV_UNIT = 1e-5


def main():
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    e = load_cache("eddies")
    lam = float(e["shear"])
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    fig.text(0.035, 0.955, "Baroclinic eddies over a mountain", color=TEXT,
             fontsize=19, fontweight="bold")
    fig.text(0.035, 0.918, "Upper-layer potential vorticity over the "
             "terrain; two-layer QG atmosphere at 45°N, "
             r"$\Lambda$ = 26 m s$^{-1}$, "
             "2.2 km mountain", color=DIM, fontsize=11)
    runs = []
    cols = {"east": EAST, "west": WEST}
    qlim = None
    for i, (tag, u2) in enumerate((("east", -0.5), ("west", 0.5))):
        ph = reference(shear=lam, U=u2 + 0.5 * lam, interactive=False)
        m = QGModel(ph, RUN_GRID, dt=0.01)
        m.set_state(e[tag + "_p"], u2 + 0.5 * lam, 0.0)
        m.adapt_dt(0.3)
        g = m.grid
        q = pv_si(m.grid_fields()["q"][0]) / PV_UNIT
        if qlim is None:
            qlim = float(np.percentile(np.abs(q), 99))
        ax = fig.add_axes([-0.035 + 0.455 * i, 0.31, 0.5, 0.58],
                          projection="3d")
        tv = TerrainView(ax, km(g.X), km(g.Y), height_of_eta(m.eta), PV,
                         -qlim, qlim, z_aspect=0.35, elev=34, azim=-60,
                         zoom=1.3, tint=0.65)
        tv.draw(q)
        tv.contours(summit_m=float(height_of_eta(ph.orography.eta0)))
        fig.text(0.06 + 0.455 * i, 0.86, r"$U_2$ = %+g m s$^{-1}$"
                 % (10 * u2), color=cols[tag], fontsize=12,
                 fontweight="bold")
        runs.append(dict(m=m, tv=tv, tag=tag, D=[], t=[],
                         lin=float(drag_tendency(e[tag + "_D_linear"]))))
    unit_colorbar(fig, ScalarMappable(Normalize(-qlim, qlim), PV),
                  [0.7, 0.875, 0.2, 0.014],
                  r"$q_1^{\prime}$ ($10^{-5}$ s$^{-1}$)",
                  orientation="horizontal")
    tx = fig.add_axes([0.08, 0.1, 0.84, 0.17])
    dark_axes(tx)
    tx.set_title(r"running time mean of $\dot{U}_{\mathrm{oro}} = "
                 r"\frac{1}{2}\langle\eta\,\partial_x\psi_2\rangle$ "
                 r"(dashed: linear theory)", color=DIM, fontsize=9.5)
    tx.axhline(0, color="#3a4a60", lw=0.8)
    lines = {}
    for R in runs:
        tx.axhline(R["lin"], color=cols[R["tag"]], ls="--", lw=0.9,
                   alpha=0.7)
        lines[R["tag"]], = tx.plot([], [], color=cols[R["tag"]], lw=1.6)
    n = int(DURATION * FPS)
    t_end = n * DT_FRAME * T_STAR / DAY
    tx.set_xlim(0, t_end)
    lim = 1.6 * max(abs(R["lin"]) for R in runs)
    tx.set_ylim(-lim, lim)
    tx.set_xlabel(r"$t$ (days)", fontsize=9)
    tx.set_ylabel(r"$\dot{U}_{\mathrm{oro}}$ (m s$^{-1}$ day$^{-1}$)",
                  fontsize=9)
    fig.text(0.965, 0.008, "heights exaggerated about 2000 times; colors "
             "draped on the terrain; blue cyclonic, amber anticyclonic",
             color=DIM, fontsize=8.5, ha="right")
    clock = fig.text(0.965, 0.955, "", color=TEXT, fontsize=12,
                     family="DejaVu Sans Mono", ha="right")

    def frame(k):
        for R in runs:
            m = R["m"]
            if k > 0:
                m.step(max(1, int(round(DT_FRAME / m.dt))))
            q = pv_si(m.grid_fields()["q"][0]) / PV_UNIT
            R["tv"].draw(q)
            R["D"].append(float(drag_tendency(m.form_stress(m.psi))))
            R["t"].append(k * DT_FRAME * T_STAR / DAY)
            run_mean = np.cumsum(R["D"]) / np.arange(1, len(R["D"]) + 1)
            lines[R["tag"]].set_data(R["t"], run_mean)
        clock.set_text("day %5.0f" % runs[0]["t"][-1])
        return []

    render(fig, frame, n, "anim03_eddies", fps=FPS)


if __name__ == "__main__":
    main()

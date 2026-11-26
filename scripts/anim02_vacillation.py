"""Animation 2: orographic vacillation under surface easterlies.

The interactive two-layer model at the reference shear (15 m/s) and the
easterly forcing wind with the largest form-stress variability, started
from the end of its fixed-forcing run. An oblique, lit view of the whole
domain (distances and heights in km, heights exaggerated) shows the
2.2 km mountain with the lower-layer streamfunction anomaly
(10^6 m^2/s) draped over the terrain and 500 lower-layer parcels
carried by the model. The instruments trace the acceleration of the
mean wind by the orographic form stress (m/s per day) and the surface
wind (m/s). Playback covers about 260 days.
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from orographic_asymmetry_qg.experiments import (DYN_GRID,  # noqa: E402
                                                 reference)
from orographic_asymmetry_qg.io import load_cache  # noqa: E402
from orographic_asymmetry_qg.model import QGModel  # noqa: E402
from orographic_asymmetry_qg.params import (DAY, T_STAR,  # noqa: E402
                                            drag_tendency, height_of_eta,
                                            km, ms, psi_si)
from orographic_asymmetry_qg.scene import (ACCENT, BG, DIM, PV,  # noqa: E402
                                           TEXT, TRAIL, TerrainView,
                                           dark_axes, render,
                                           unit_colorbar)

FPS = 30
DURATION = 30.0
DT_FRAME = 0.055             # model time per frame (6.9 h)
N_PARTICLES = 500
HIST = 12
LAM = 1.5
PSI_LIM = 3.2                # color limit, 10^6 m^2/s


def main():
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    d = load_cache("dynamics")
    tag = "s%g" % LAM
    j = int(np.argmax(d[tag + "_D"].std(axis=1)))
    u2s = float(d["forcings"][j])
    ph = reference(shear=LAM, U_star=u2s + 0.5 * LAM)
    m = QGModel(ph, DYN_GRID, dt=0.01)
    m.set_state(d[tag + "_p"][j], float(d[tag + "_U"][j]), 0.0)
    m.adapt_dt(0.3)
    nsub = max(1, int(round(DT_FRAME / m.dt)))
    g = m.grid
    rng = np.random.default_rng(2)
    m.add_particles(rng.uniform(0, g.lx, N_PARTICLES),
                    rng.uniform(0, g.ly, N_PARTICLES), layer=-1)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.0, 0.26, 0.93, 0.68], projection="3d")
    X, Y = km(g.X), km(g.Y)
    hgt = height_of_eta(m.eta)
    tv = TerrainView(ax, X, Y, hgt, PV, -PSI_LIM, PSI_LIM, z_aspect=0.35,
                     elev=34, azim=-60, zoom=1.35)
    tv.draw(psi_si(m.grid_fields()["psi"][-1]) / 1e6)
    tv.contours(summit_m=float(height_of_eta(ph.orography.eta0)))
    lc = Line3DCollection([[(0, 0, 0), (0, 0, 0)]], linewidths=0.8,
                          zorder=5)
    ax.add_collection3d(lc, autolim=False)
    unit_colorbar(fig, ScalarMappable(Normalize(-PSI_LIM, PSI_LIM), PV),
                  [0.94, 0.36, 0.009, 0.45],
                  r"$\psi_2^{\prime}$ ($10^6$ m$^2$ s$^{-1}$)")
    fig.text(0.035, 0.955, "Orographic vacillation under surface "
             "easterlies", color=TEXT, fontsize=19, fontweight="bold")
    fig.text(0.035, 0.918, "Two-layer QG atmosphere at 45°N: "
             r"westerlies aloft over easterlies below ($\Lambda$ = 15 "
             r"m s$^{-1}$, $U_2^*$ = %+g m s$^{-1}$), 2.2 km mountain"
             % (10 * u2s),
             color=DIM, fontsize=11)
    sx = fig.add_axes([0.07, 0.1, 0.38, 0.14])
    ux = fig.add_axes([0.56, 0.1, 0.38, 0.14])
    for a in (sx, ux):
        dark_axes(a)
        a.set_xlabel(r"$t$ (days)", fontsize=9)
    sx.set_ylabel(r"$\dot{U}_{\mathrm{oro}}$ (m s$^{-1}$ day$^{-1}$)",
                  fontsize=9)
    ux.set_ylabel(r"$U_2$ (m s$^{-1}$)", fontsize=9)
    sx.set_title(r"form-stress acceleration of the mean wind, "
                 r"$\dot{U}_{\mathrm{oro}} = \frac{1}{2}\langle\eta\,"
                 r"\partial_x\psi_2\rangle$", color=DIM, fontsize=9.5)
    ux.set_title(r"lower-layer wind $U_2$", color=DIM, fontsize=9.5)
    n = int(DURATION * FPS)
    t_days = n * nsub * m.dt * T_STAR / DAY
    Dd = drag_tendency(d[tag + "_D"][j])
    U2 = ms(d[tag + "_U2"][j])
    for a, v in ((sx, Dd), (ux, U2)):
        pad = 0.2 * (v.max() - v.min() + 1e-9)
        a.set_xlim(0, t_days)
        a.set_ylim(v.min() - pad, v.max() + pad)
    ls, = sx.plot([], [], color=ACCENT, lw=1.3)
    lu, = ux.plot([], [], color="#58a6ff", lw=1.3)
    fig.text(0.965, 0.008, "heights exaggerated about 2000 times; parcels "
             "carried by the model; blue cyclonic, amber anticyclonic",
             color=DIM, fontsize=8.5, ha="right")
    clock = fig.text(0.965, 0.955, "", color=TEXT, fontsize=12,
                     family="DejaVu Sans Mono", ha="right")
    hist = []
    hist_t, hist_d, hist_u = [], [], []
    a = np.repeat(np.linspace(0.05, 1, HIST - 1), N_PARTICLES)
    cols = TRAIL(a)
    cols[:, 3] = 0.8 * a ** 1.6
    lx, ly = km(g.lx), km(g.ly)

    def frame(k):
        if k > 0:
            m.step(nsub)
        tv.draw(psi_si(m.grid_fields()["psi"][-1]) / 1e6)
        xy = m.particles["xy"]
        p = np.array([np.mod(km(xy[0]), lx), np.mod(km(xy[1]), ly)])
        hist.append(p)
        if len(hist) > HIST:
            hist.pop(0)
        H = np.array(hist)                           # (m, 2, N)
        if H.shape[0] >= 2:
            z = tv.height_at(H[:, 0], H[:, 1]).reshape(H.shape[0], -1)
            p3 = np.stack([H[:, 0], H[:, 1], z + 0.04], axis=1)
            seg = np.stack([p3[:-1], p3[1:]], axis=1)  # (m-1, 2, 3, N)
            jump = ((np.abs(seg[:, 1, 0] - seg[:, 0, 0]) > 0.5 * lx)
                    | (np.abs(seg[:, 1, 1] - seg[:, 0, 1]) > 0.5 * ly))
            seg = np.moveaxis(seg, 3, 1)               # (m-1, N, 2, 3)
            c = cols[-seg.shape[0] * N_PARTICLES:].reshape(
                seg.shape[0], N_PARTICLES, 4).copy()
            c[jump, 3] = 0.0
            lc.set_segments(seg.reshape(-1, 2, 3))
            lc.set_color(c.reshape(-1, 4))
        td = k * nsub * m.dt * T_STAR / DAY
        hist_t.append(td)
        hist_d.append(float(drag_tendency(m.form_stress(m.psi))))
        hist_u.append(float(ms(m.U - 0.5 * LAM)))
        ls.set_data(hist_t, hist_d)
        lu.set_data(hist_t, hist_u)
        clock.set_text("day %5.0f" % td)
        return []

    render(fig, frame, n, "anim02_vacillation", fps=FPS)


if __name__ == "__main__":
    main()

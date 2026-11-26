"""Animation 1: westerly and easterly flow over a mountain, with and
without vertical shear, side by side.

Lower-layer air streams past the 2.2 km reference mountain in the
steady nonlinear two-layer flow: surface westerlies of 5 m/s (a lee
wave train), surface easterlies of 5 m/s (a confined response), and the
same easterlies beneath westerlies aloft (shear 15 m/s). Parcels are
released continuously upstream and advected by the model velocity.
Each case is an oblique view of the terrain (distances in km, height
in km, heights exaggerated about 1700 times), lit by the relief, with
the lower-layer streamfunction anomaly in 10^6 m^2/s draped over it,
height contours every 500 m and the summit marked; the parcels travel
over the surface. A fourth panel places the
three cases among the regions of zero, one and two stationary Rossby
waves in the plane of the layer winds (m/s).
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
from orographic_asymmetry_qg.params import (DAY, T_STAR,  # noqa: E402
                                            height_of_eta, km, psi_si)
from orographic_asymmetry_qg.scene import (ACCENT, BG, DIM, PV,  # noqa: E402
                                           TEXT, TRAIL, TerrainView,
                                           dark_axes, render,
                                           unit_colorbar)
from orographic_asymmetry_qg.spectral import Grid  # noqa: E402

FPS = 30
DURATION = 24.0
DT = 0.02                  # model time per frame (2.5 h)
WINDOW = (0.0, 4.2, 0.55, 2.6)
CASES = (("west", "Surface westerlies, no shear", 0.5),
         ("east", "Surface easterlies, no shear", -0.5),
         ("sheared", "Surface easterlies under westerlies aloft", -0.5))
COLORS = ("#58a6ff", "#ff7b72", ACCENT)
N_POOL = 300
HIST = 40
PSI_UNIT = 1e6             # color scale in 10^6 m^2/s
PSI_LIM = 2.4              # +- color limit in 10^6 m^2/s


def wind_map(fig, rect, r):
    """Where the three cases sit among the stationary-wave regions."""
    from matplotlib.colors import ListedColormap
    ax = fig.add_axes(rect)
    dark_axes(ax)
    cmap = ListedColormap(["#16233a", "#2a5a9e", "#5b97e6"])
    W = 10 * r["W"]
    ax.pcolormesh(W, W, r["count"], cmap=cmap, vmin=-0.5, vmax=2.5,
                  shading="auto", rasterized=True)
    w = np.array([-20, 20])
    ax.plot(w, w, color=DIM, lw=0.8)
    ax.axhline(0, color="#3a4a60", lw=0.6)
    ax.axvline(0, color="#3a4a60", lw=0.6)
    for (name, _, u2), c in zip(CASES, COLORS):
        u1 = u2 + (1.5 if name == "sheared" else 0.0)
        ax.plot(10 * u2, 10 * u1, "o", ms=9, mfc=c, mec="w", mew=1.2)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$U_2$ (m s$^{-1}$)", fontsize=9)
    ax.set_ylabel(r"$U_1$ (m s$^{-1}$)", fontsize=9)
    x0 = rect[0] + rect[2] + 0.015
    fig.text(x0, rect[1] + rect[3] - 0.01, "stationary\nRossby waves",
             color=TEXT, fontsize=10, va="top")
    for k, lab in enumerate(("none", "one", "two")):
        fig.text(x0, rect[1] + rect[3] - 0.1 - 0.035 * k,
                 "■ " + lab, color=cmap(k) if k else DIM,
                 fontsize=10, va="top")
    fig.text(x0, rect[1] + 0.02, "diagonal: no shear\n(classical cases)",
             color=DIM, fontsize=8, va="bottom")


def main():
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    h = load_cache("cases")
    g = Grid(*RUN_GRID)
    hgt = height_of_eta(reference().orography.field(g))
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    fig.text(0.035, 0.955, "Flow over a mountain, with and without vertical "
             "shear", color=TEXT, fontsize=19, fontweight="bold")
    fig.text(0.035, 0.918, "Lower-layer air crossing a 2.2 km Gaussian "
             "mountain (half-width 1130 km) in a two-layer QG atmosphere "
             "at 45\u00b0N; steady nonlinear flow", color=DIM, fontsize=11)
    x0, x1, y0, y1 = WINDOW
    sx = (g.x >= x0) & (g.x <= x1)
    sy = (g.y >= y0) & (g.y <= y1)
    Xk, Yk = km(g.X)[np.ix_(sy, sx)], km(g.Y)[np.ix_(sy, sx)]
    hw = hgt[np.ix_(sy, sx)]
    slots = ([-0.03, 0.44, 0.56, 0.47], [0.42, 0.44, 0.56, 0.47],
             [-0.03, -0.01, 0.56, 0.47])
    panels = []
    for i, (name, title, u2) in enumerate(CASES):
        ax = fig.add_axes(slots[i], projection="3d")
        tv = TerrainView(ax, Xk, Yk, hw, PV, -PSI_LIM, PSI_LIM)
        psi = h["%s_psi" % name][-1]
        tv.draw(psi_si(psi[np.ix_(sy, sx)]) / PSI_UNIT)
        tv.contours()
        fig.text(slots[i][0] + 0.065, slots[i][1] + slots[i][3] - 0.035,
                 title, color=COLORS[i], fontsize=12, fontweight="bold")
        fig.text(slots[i][0] + 0.065, slots[i][1] + slots[i][3] - 0.065,
                 r"$\Lambda$ = %s m s$^{-1}$,  $U_2$ = %+g m s$^{-1}$"
                 % ("15" if name == "sheared" else "0", 10 * u2),
                 color=TEXT, fontsize=9)
        psh = g.fft(psi)
        u = g.ifft(-g.iky * psh) + u2
        v = g.ifft(g.ikx * psh)
        lc = Line3DCollection([[(0, 0, 0), (0, 0, 0)]], linewidths=0.9,
                              zorder=5)
        ax.add_collection3d(lc, autolim=False)
        panels.append(dict(u=u, v=v, U2=u2, lc=lc, tv=tv))
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(Normalize(-PSI_LIM, PSI_LIM), PV)
    unit_colorbar(fig, sm, [0.955, 0.5, 0.009, 0.36],
                  r"$\psi_2^{\prime}$ ($10^6$ m$^2$ s$^{-1}$)")
    wind_map(fig, [0.585, 0.09, 0.24, 0.27], load_cache("resonance"))
    fig.text(0.965, 0.012, "heights exaggerated about 1700 times; parcels "
             "advected by the model velocity; blue cyclonic, amber "
             "anticyclonic", color=DIM, fontsize=8.5, ha="right")
    clock = fig.text(0.965, 0.955, "", color=TEXT, fontsize=12,
                     family="DejaVu Sans Mono", ha="right")
    rng = np.random.default_rng(0)
    n = int(DURATION * FPS)
    span = x1 - x0

    def spawn(P, m):
        """Positions of ``m`` parcels entering on the upstream edge."""
        ys = rng.uniform(y0 + 0.06, y1 - 0.06, m)
        xs = np.full(m, x0 if P["U2"] > 0 else x1)
        return np.array([xs, ys])

    for P in panels:
        # a pool spread along the window so the stream starts full
        xy = spawn(P, N_POOL)
        xy[0] = x0 + span * rng.random(N_POOL)
        P["xy"] = xy
        P["trail"] = np.repeat(xy[None], HIST, axis=0)

    def vel(P, xy):
        return np.array([g.interp(P["u"], xy[0], xy[1]),
                         g.interp(P["v"], xy[0], xy[1])])

    a = np.repeat(np.linspace(0.05, 1, HIST - 1), N_POOL)
    cols = TRAIL(a)
    cols[:, 3] = 0.85 * a ** 1.6

    def draw(P):
        t = km(P["trail"])                       # (HIST, 2, N), km
        z = P["tv"].height_at(t[:, 0], t[:, 1]).reshape(HIST, -1) + 0.04
        p3 = np.stack([t[:, 0], t[:, 1], z], axis=1)   # (HIST, 3, N)
        seg = np.stack([p3[:-1], p3[1:]], axis=1)      # (HIST-1, 2, 3, N)
        seg = np.moveaxis(seg, 3, 1).reshape(-1, 2, 3)
        P["lc"].set_segments(seg)
        P["lc"].set_color(cols)

    def frame(k):
        for P in panels:
            xy = P["xy"]
            k1 = vel(P, xy)
            k2 = vel(P, xy + 0.5 * DT * k1)
            k3 = vel(P, xy + 0.5 * DT * k2)
            k4 = vel(P, xy + DT * k3)
            xy = xy + DT / 6 * (k1 + 2 * (k2 + k3) + k4)
            out = (xy[0] < x0) | (xy[0] > x1) | (xy[1] < y0) | (xy[1] > y1)
            tr = np.roll(P["trail"], -1, axis=0)
            tr[-1] = xy
            if out.any():
                new = spawn(P, out.sum())
                xy[:, out] = new
                tr[:, :, out] = new[None]
            P["xy"], P["trail"] = xy, tr
            draw(P)
        clock.set_text("day %5.1f" % (k * DT * T_STAR / DAY))
        return []

    render(fig, frame, n, "anim01_mountain", fps=FPS)


if __name__ == "__main__":
    main()

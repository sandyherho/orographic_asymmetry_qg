"""The classical westerly and easterly cases, and their sheared counterpart
(Fig. 3).

Ridge: the exact steady barotropic flow over an infinite,
meridionally uniform ridge (250 m high in a 10 km deep column), with and
without bottom drag, and the parcel argument integrated as a
nonlinear trajectory equation with the same drag, for a westerly and an
easterly wind of 5 m/s; and the departure of the parcel equation from
the exact QG streamline as the ridge is raised from 50 m to 1 km.

Mountain: the steady nonlinear two-layer flow over the reference
mountain with a fixed lower-layer wind of +5 or -5 m/s and no shear
(the two classical cases), and with -5 m/s under a shear of 15 m/s (surface
easterlies beneath westerlies aloft). Lower-layer parcels are released
upstream and followed through the steady flow.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from orographic_asymmetry_qg import theory as th  # noqa: E402
from orographic_asymmetry_qg.experiments import (CONT_GRID,  # noqa: E402
                                                 RUN_GRID, pmap, reference)
from orographic_asymmetry_qg.io import (save_cache, write_csv,  # noqa: E402
                                        write_netcdf)
from orographic_asymmetry_qg.lagrangian import advect_frozen  # noqa: E402
from orographic_asymmetry_qg.model import QGModel  # noqa: E402
from orographic_asymmetry_qg.params import (Orography,  # noqa: E402
                                            Physics, eta_of_height)
from orographic_asymmetry_qg.spectral import Grid  # noqa: E402
from orographic_asymmetry_qg.steady import solve_steady  # noqa: E402

U_CASE = 0.5
RIDGE_DRAG = 0.65             # 8 day spin-down of the full column
RIDGE_HEIGHT = 250.0
RIDGE_ETA = float(eta_of_height(RIDGE_HEIGHT, depth=10000.0))
SCAN_HEIGHTS = (50.0, 100.0, 250.0, 500.0, 750.0, 1000.0)
CASES = (("west", 0.0, 0.5), ("east", 0.0, -0.5), ("sheared", 1.5, -0.5))
N_PARCELS = 9


def ridge(eta0=None):
    """Exact ridge solutions and parcel trajectories.

    The parcel equation with drag starts from the exact streamline of
    the damped solution (position, slope and vorticity). Westerly
    parcels cross the whole domain; easterly parcels only the ridge,
    because the evanescent easterly balance amplifies any error as an
    initial-value problem. Without drag the westerly solution is close
    to resonance with a domain mode and is shown only for reference.
    """
    eta0 = RIDGE_ETA if eta0 is None else eta0
    g = Grid(2048, 4)
    oro = Orography(kind="ridge", eta0=eta0, width=0.25, x0=np.pi)
    eta_x = oro.field(g)[0]
    k = 2 * np.pi * np.fft.rfftfreq(g.nx, g.dx)
    beta = Physics().beta

    def eta_fun(x):
        s = [(x - np.pi + m * g.lx) / 0.25 for m in range(-3, 4)]
        return eta0 * sum(np.exp(-a * a) for a in s)

    def deta_fun(x):
        s = [(x - np.pi + m * g.lx) / 0.25 for m in range(-3, 4)]
        return eta0 * sum(-2 * a / 0.25 * np.exp(-a * a) for a in s)
    out = dict(x=g.x, eta=eta_x)
    for name, U, span in (("west", U_CASE, (0.0, g.lx)),
                          ("east", -U_CASE, (np.pi - 0.45, np.pi + 0.45))):
        for drag in (0.0, RIDGE_DRAG):
            psi = th.ridge_steady_streamfunction(g.x, U, beta, eta_x,
                                                 drag=drag, nu=0.0)
            out["%s_y_drag%g" % (name, drag)] = psi / U
        y = out["%s_y_drag%g" % (name, RIDGE_DRAG)]
        dy = np.fft.irfft(1j * k * np.fft.rfft(y), g.nx)
        d2 = np.fft.irfft(-k * k * np.fft.rfft(y), g.nx)
        xs0 = span[0] if U > 0 else span[1]
        xs, ys = th.parcel_trajectory(eta_fun, U, beta, span,
                                      y0=np.interp(xs0, g.x, y, period=g.lx),
                                      slope0=np.interp(xs0, g.x, dy,
                                                       period=g.lx),
                                      drag=RIDGE_DRAG, deta_fun=deta_fun,
                                      zeta0=U * np.interp(xs0, g.x, d2,
                                                          period=g.lx),
                                      n_out=1500)
        out[name + "_parcel_x"], out[name + "_parcel_y"] = xs, ys
        yq = np.interp(xs, g.x, y, period=g.lx)
        out[name + "_parcel_error"] = np.max(np.abs(ys - yq)) / np.max(
            np.abs(y))
    return out


def mountain_case(case):
    """Steady flow over the mountain and parcel paths for one case."""
    name, lam, u2 = case
    ph = reference(shear=lam, U=u2 + 0.5 * lam, interactive=False)
    prob, x, info = solve_steady(ph, CONT_GRID)
    p, _ = prob.unpack(x)
    m = prob.m
    fine = QGModel(ph, RUN_GRID)
    pf = m.grid.resample(p, fine.grid)
    fine.set_state(p=pf)
    f = fine.grid_fields()
    g = fine.grid
    u = f["u"][-1] + fine.layer_winds()[-1]
    v = f["v"][-1]
    y0 = np.linspace(0.9, 2.2, N_PARCELS)
    x0 = np.full(N_PARCELS, 0.25 if u2 > 0 else 3.2)
    t, xy = advect_frozen(g, u, v, x0, y0, 4.6 / abs(u2), 0.01, n_out=900)
    growth = prob.spectrum(x)[0].real
    return dict(psi=f["psi"], q=f["q"], parcels=xy, t=t,
                form_stress=float(m.form_stress(m.invert(p))),
                residual=info["residual"], growth=growth)


def main():
    r = ridge()
    scan = np.array([[ridge(float(eta_of_height(hh, depth=10000.0)))[
        n + "_parcel_error"] for n in ("west", "east")]
        for hh in SCAN_HEIGHTS])
    res = pmap(mountain_case, CASES)
    g = Grid(*RUN_GRID)
    out = {"ridge_" + k: v for k, v in r.items()}
    out["ridge_eta0"] = np.array(RIDGE_ETA)
    out["ridge_height"] = np.array(RIDGE_HEIGHT)
    out["scan_heights"] = np.array(SCAN_HEIGHTS)
    out["scan_error"] = scan
    for (name, lam, u2), d in zip(CASES, res):
        for k, v in d.items():
            out["%s_%s" % (name, k)] = np.asarray(v)
    out["x"], out["y"] = g.x, g.y
    out["eta"] = reference().orography.field(g)
    save_cache("cases", **out)
    write_csv("fig03_ridge",
              ["x", "eta", "y_west_inviscid", "y_west_drag",
               "y_east_inviscid", "y_east_drag"],
              [(r["x"][i], r["eta"][i], r["west_y_drag0"][i],
                r["west_y_drag%g" % RIDGE_DRAG][i], r["east_y_drag0"][i],
                r["east_y_drag%g" % RIDGE_DRAG][i])
               for i in range(0, r["x"].size, 8)])
    write_netcdf(
        "mountain_cases",
        {"y": (g.y, "1", "meridional coordinate (units of 4505 km)"),
         "x": (g.x, "1", "zonal coordinate (units of 4505 km)")},
        {"%s_psi2" % name: (("y", "x"), out["%s_psi" % name][-1], "1",
                            "lower-layer eddy streamfunction, %s" % name)
         for name, _, _ in CASES},
        "Steady flow over the reference mountain: surface westerlies "
        "and easterlies without shear, and surface easterlies under shear",
        attrs={"%s_form_stress" % n: float(out["%s_form_stress" % n])
               for n, _, _ in CASES})


if __name__ == "__main__":
    main()

"""Verification of the numerical model (Fig. 2).

(a) Temporal convergence of ETDRK4 on a nonlinear two-layer flow over
the mountain. (b) Growth rates of two-layer normal modes in the model
against the dispersion relation with drag and cooling. (c) Nonlinear
against linear form stress as the mountain is lowered. (d) The eddy
energy integrated in time against the time integral of its budget
terms. (e) Newton convergence. (f) Grid convergence of the steady form
stress and of the leading growth rate. Also the drift of the two
inviscid invariants of the one-layer system with interactive mean wind.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from orographic_asymmetry_qg import diagnostics as dg  # noqa: E402
from orographic_asymmetry_qg import theory as th  # noqa: E402
from orographic_asymmetry_qg.experiments import (CONT_GRID,  # noqa: E402
                                                 pmap, reference)
from orographic_asymmetry_qg.io import save_cache, write_csv  # noqa: E402
from orographic_asymmetry_qg.model import QGModel  # noqa: E402
from orographic_asymmetry_qg.params import Orography, Physics  # noqa: E402
from orographic_asymmetry_qg.steady import solve_steady  # noqa: E402

DTS = (0.02, 0.01, 0.005, 0.0025)
DT_REF = 0.000625
ETAS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
GRIDS = ((48, 24), (64, 32), (96, 48))
GROWTH_K = np.arange(1, 9)
GROWTH_CASE = dict(shear=3.0, U2=-1.0)


def temporal():
    """Max error of the PV after 0.5 time units against a fine run."""
    ph = reference(shear=1.5, U=-0.2, U_star=-0.2)

    def run(dt):
        m = QGModel(ph, (64, 32), dt=dt)
        m.perturb(0.3, seed=11, kmin=1.0, kmax=8.0)
        m.step(int(round(0.5 / dt)))
        return m.p, m.U
    pr, ur = run(DT_REF)
    err = []
    for dt in DTS:
        p, u = run(dt)
        err.append(max(np.max(np.abs(p - pr)) / np.max(np.abs(pr)),
                       abs(u - ur) / abs(ur)))
    return np.array(err)


def growth():
    """Model and theoretical growth rates along the zonal wavenumber."""
    lam, u2 = GROWTH_CASE["shear"], GROWTH_CASE["U2"]
    ph = Physics(shear=lam, U=u2 + 0.5 * lam, orography=Orography("none"))
    out = []
    for k in GROWTH_K:
        m = QGModel(ph, (64, 32), dt=0.002)
        psi = np.zeros((2,) + m.grid.k2.shape, complex)
        psi[0, 0, k], psi[1, 0, k] = 1e-10, 3e-11
        m.set_state(p=m.pv_of_psi(psi))
        m.step(2000)
        a1 = np.linalg.norm(m.psi[:, 0, k])
        m.step(1000)
        a2 = np.linalg.norm(m.psi[:, 0, k])
        out.append(np.log(a2 / a1) / 2.0)
    kk = np.linspace(0.3, 8.5, 300)
    theory = [th.phillips_growth_rate(k, 0.0, u2 + lam, u2, ph.beta, ph.F,
                                      ph.drag, ph.cooling) for k in kk]
    return np.array(out), kk, np.array(theory)


def small_amplitude(args):
    """Nonlinear and linear form stress at prescribed wind."""
    lam, u2, eta0 = args
    ph = reference(shear=lam, U=u2 + 0.5 * lam, interactive=False,
                   orography__eta0=eta0)
    prob, x, info = solve_steady(ph, CONT_GRID)
    m = prob.m
    p, _ = prob.unpack(x)
    D = m.form_stress(m.invert(p))
    return D, th.linear_form_stress(ph, m.grid, nu=m.nu)


def budget():
    """Eddy energy against the time integral of its budget."""
    ph = reference(shear=1.5, U=-0.3, U_star=-0.3)
    m = QGModel(ph, (64, 32), dt=0.0025)
    m.perturb(0.2, seed=5, kmin=1.0, kmax=8.0)
    t, E, S = [], [], []
    for _ in range(2001):
        b = dg.energy_budget(m)
        t.append(m.t)
        E.append(dg.energy(m)["total"])
        S.append(b["baroclinic"] + b["orographic"] - b["drag"]
                 - b["cooling"] - b["hyperviscous"])
        m.step(1)
    t, E, S = map(np.array, (t, E, S))
    # Simpson-like cumulative integral of the budget
    cum = np.concatenate([[0], np.cumsum(0.5 * (S[1:] + S[:-1])
                                         * np.diff(t))])
    return t, E - E[0], cum


def newton_history():
    """Residual after each Newton iteration from the linear guess."""
    ph = reference(shear=1.5, U=0.25, U_star=0.25, orography__eta0=5.0)
    from orographic_asymmetry_qg.steady import (SteadyProblem,
                                                steady_from_linear)
    m = QGModel(ph, CONT_GRID)
    prob = SteadyProblem(m)
    x = steady_from_linear(m, prob)
    from scipy.linalg import lu_factor, lu_solve
    hist = []
    for _ in range(7):
        r = prob.residual(x)
        hist.append(np.max(np.abs(r)))
        if hist[-1] < 1e-13:
            break
        x = x - lu_solve(lu_factor(prob.jacobian(x)), r)
    return np.array(hist)


def resolution(grid):
    """Steady form stress and growth rate of the reference unstable
    easterly case on one grid."""
    ph = reference(shear=1.5, U=-0.5 + 0.75, U_star=-0.5 + 0.75)
    prob, x, info = solve_steady(ph, grid)
    m = prob.m
    p, U = prob.unpack(x)
    return (float(m.form_stress(m.invert(p))), float(U - 0.75),
            float(prob.spectrum(x)[0].real))


def invariants():
    """Relative drift of energy and potential enstrophy (one layer,
    inviscid, unforced, interactive mean wind) over 50 time units."""
    ph = Physics(nlayers=1, U=0.4, interactive=True, kappa=0.0, drag=0.0,
                 nu=0.0, orography=Orography(eta0=10.0))
    m = QGModel(ph, (64, 32), dt=0.002)
    m.perturb(0.05, seed=2, kmin=1.0, kmax=6.0)

    def inv():
        return (dg.energy(m)["total"] + 0.5 * m.U ** 2,
                dg.enstrophy(m) + ph.beta * m.U)
    e0, z0 = inv()
    m.step(25000)
    e1, z1 = inv()
    return abs(e1 / e0 - 1), abs(z1 / z0 - 1)


def main():
    err = temporal()
    gm, kk, gt = growth()
    jobs = [(0.0, 0.5, e) for e in ETAS] + [(1.5, -0.5, e) for e in ETAS]
    amp = np.array(pmap(small_amplitude, jobs))
    t, dE, cum = budget()
    hist = newton_history()
    res = np.array(pmap(resolution, GRIDS))
    de, dz = invariants()
    save_cache("verification", dts=np.array(DTS), err=err,
               growth_k=GROWTH_K, growth_model=gm, growth_kk=kk,
               growth_theory=gt, etas=np.array(ETAS),
               amp_west=amp[:len(ETAS)], amp_east=amp[len(ETAS):],
               budget_t=t, budget_dE=dE, budget_int=cum,
               newton=hist, grids=np.array(GRIDS), resolution=res,
               invariant_drift=np.array([de, dz]))
    write_csv("fig02a_convergence", ["dt", "max_rel_error"],
              list(zip(DTS, err)))
    write_csv("fig02b_growth", ["k", "growth_model", "growth_theory"],
              [(k, gm[i], np.interp(k, kk, gt))
               for i, k in enumerate(GROWTH_K)])
    write_csv("fig02c_small_amplitude",
              ["eta0", "D_west", "D_lin_west", "D_east", "D_lin_east"],
              [(e, amp[i, 0], amp[i, 1], amp[len(ETAS) + i, 0],
                amp[len(ETAS) + i, 1]) for i, e in enumerate(ETAS)])
    write_csv("fig02f_resolution", ["nx", "ny", "form_stress", "U2",
                                    "growth_rate"],
              [(g[0], g[1], *res[i]) for i, g in enumerate(GRIDS)])


if __name__ == "__main__":
    main()

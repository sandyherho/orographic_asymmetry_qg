"""Reference configuration and the numerical experiments.

The reference atmosphere is a two-layer beta-plane channel of one
latitude circle at 45 degrees (``2 pi`` long, ``pi`` wide, doubly
periodic) with a 4 day bottom drag, 20 day cooling of the eddy
interface, a deformation radius of 820 km and an isolated Gaussian
mountain with an e-folding half-width of 1100 km. Integrations use the
ETDRK4 model with a CFL-limited step from a ladder: long runs along the
forcing on ``DYN_GRID``, eddying runs and figures on ``RUN_GRID``.
Steady states use ``CONT_GRID``; with the grid-independent
hyperviscosity the three grids agree to about 2 percent in form stress.
"""

import os

import numpy as np

from . import diagnostics as dg
from .model import QGModel
from .params import Orography, Physics

RUN_GRID = (128, 64)
DYN_GRID = (96, 48)
FINE_GRID = (256, 128)
CONT_GRID = (64, 32)
SHEARS = (0.0, 1.0, 1.5, 2.0)
ETA_REF = 20.0
MOUNTAIN = Orography(kind="mountain", eta0=ETA_REF, width=0.25)
RIDGE = Orography(kind="ridge", eta0=ETA_REF, width=0.25)


def reference(**kw):
    """The reference two-layer physics, with fields replaced by ``kw``."""
    base = Physics(nlayers=2, shear=1.5, U=0.5, interactive=True,
                   U_star=0.5, orography=MOUNTAIN)
    return base.with_(**kw)


def surface(phys):
    """Bottom-layer wind of the column-mean wind of ``phys``."""
    return phys.layer_winds[-1]


def pmap(func, items, workers=None):
    """Map ``func`` over ``items`` in worker processes, keeping order.

    The number of workers defaults to the ``QG_WORKERS`` environment
    variable, or to the number of CPUs.
    """
    from concurrent.futures import ProcessPoolExecutor
    items = list(items)
    if workers is None:
        workers = int(os.environ.get("QG_WORKERS", os.cpu_count() or 1))
    if workers <= 1 or len(items) <= 1:
        return [func(x) for x in items]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(func, items))


def integrate(phys, t_end, grid=RUN_GRID, dt_max=0.01, cfl=0.35,
              every=0.05, t_mean=None, seed=0, amplitude=1e-3,
              state=None, U_star_of_t=None, keep_fields=False,
              check=10):
    """Integrate the model and record diagnostics.

    Parameters
    ----------
    phys : Physics
    t_end : float
        Duration.
    grid, dt_max, cfl : see ``QGModel`` and ``QGModel.adapt_dt``.
    every : float
        Sampling interval of the time series.
    t_mean : float, optional
        Start of the averaging window for time-mean fields (default:
        half of ``t_end``).
    seed, amplitude : random initial perturbation.
    state : tuple, optional
        ``(p, U, t)`` to restart from.
    U_star_of_t : callable, optional
        Time-dependent forcing wind (a slow ramp for hysteresis).
    keep_fields : bool
        Also store snapshots of the bottom-layer streamfunction.
    check : int
        Steps between CFL checks.

    Returns
    -------
    dict
        Time series ``t, U, U_star, form_stress, energy, ke1, ke2`` and
        the time means ``psi_mean`` (grid, per layer), ``form_stress_mean``,
        the final state and the model.
    """
    m = QGModel(phys, grid, dt=dt_max)
    if state is not None:
        m.set_state(*state)
    else:
        m.perturb(amplitude, seed=seed)
    t_mean = 0.5 * t_end if t_mean is None else t_mean
    rec = {k: [] for k in ("t", "U", "U_star", "form_stress", "energy",
                           "ke", "dt")}
    snaps = []
    psi_sum = 0.0
    n_mean = 0
    t0 = m.t
    next_out = t0
    while m.t < t0 + t_end - 1e-12:
        if U_star_of_t is not None:
            m.phys = m.phys.with_(U_star=float(U_star_of_t(m.t - t0)))
        m.adapt_dt(cfl, dt_max)
        m.step(check)
        if not np.isfinite(m.U):
            raise FloatingPointError("integration blew up at t=%g" % m.t)
        if m.t >= next_out - 1e-12:
            psi = m.psi
            e = dg.energy(m)
            rec["t"].append(m.t)
            rec["U"].append(m.U)
            rec["U_star"].append(m.phys.U_star)
            rec["form_stress"].append(float(m.form_stress(psi)))
            rec["energy"].append(e["total"])
            rec["ke"].append(np.asarray(e["ke"]))
            rec["dt"].append(m.dt)
            if m.t - t0 >= t_mean:
                psi_sum = psi_sum + m.grid.ifft(psi)
                n_mean += 1
            if keep_fields:
                snaps.append(m.grid.ifft(psi[-1]))
            next_out += every
    out = {k: np.array(v) for k, v in rec.items()}
    out["psi_mean"] = psi_sum / max(n_mean, 1)
    sel = out["t"] - t0 >= t_mean
    out["form_stress_mean"] = float(out["form_stress"][sel].mean())
    out["U_mean"] = float(out["U"][sel].mean())
    out["state"] = (m.p.copy(), m.U, m.t)
    out["model"] = m
    if keep_fields:
        out["snapshots"] = np.array(snaps)
    return out

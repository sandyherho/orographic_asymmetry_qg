"""Stationary resonance and linear form stress (Figs. 4 and 5).

Counts the free stationary waves of the two-layer model over the plane
of the layer winds and checks the count against the closed-form
criterion; computes the stationary wavenumbers along the surface wind
for the reference shears; and evaluates the linear form stress of the
reference mountain on the model grid, with the asymmetry ratio between
surface easterlies and westerlies of the same speed.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from orographic_asymmetry_qg import theory as th  # noqa: E402
from orographic_asymmetry_qg.experiments import (RUN_GRID,  # noqa: E402
                                                 SHEARS, pmap, reference)
from orographic_asymmetry_qg.io import (save_cache, write_csv,  # noqa: E402
                                        write_netcdf)
from orographic_asymmetry_qg.spectral import Grid  # noqa: E402

W = np.linspace(-2.0, 2.0, 401)          # layer winds for the count map
U2_LINE = np.linspace(-2.0, 2.0, 801)    # surface winds along curves
ASYM_SHEAR = np.linspace(0.0, 2.1, 43)
ASYM_SPEED = (0.25, 0.5, 1.0)
U2_COARSE = np.linspace(-2.0, 2.0, 401)
KAPPAS = (0.1, 0.5)


def drag_curve(lam):
    """Linear form stress along ``U2_LINE`` at shear ``lam``."""
    g = Grid(*RUN_GRID)
    ph = reference(shear=lam)
    return np.array([th.linear_form_stress(ph, g, u + 0.5 * lam)
                     for u in U2_LINE])


def coarse_curve(lam):
    """Linear form stress along ``U2_COARSE`` at shear ``lam``."""
    g = Grid(*RUN_GRID)
    ph = reference(shear=lam)
    return np.array([th.linear_form_stress(ph, g, u + 0.5 * lam)
                     for u in U2_COARSE])


def thresholds(D, eta0, n_layers=2):
    """Smallest orographic PV with folds, for surface easterlies and
    westerlies, for each relaxation rate in ``KAPPAS``.

    Folds of the linear equilibrium curve ``U_star = U - D / (N
    kappa)`` need ``dD/dU > N kappa``; ``D`` scales as ``eta0^2``.
    """
    dD = np.gradient(D, U2_COARSE)
    out = []
    for kap in KAPPAS:
        row = []
        for side in (U2_COARSE < 0, U2_COARSE > 0):
            m = dD[side].max()
            row.append(eta0 * np.sqrt(n_layers * kap / m) if m > 0
                       else np.inf)
        out.append(row)
    return np.array(out), dD


def main():
    ph = reference()
    beta, F = ph.beta, ph.F
    U1, U2 = np.meshgrid(W, W, indexing="ij")
    count = th.resonance_count(U1, U2, beta, F)
    pred = np.vectorize(lambda a, b: th.resonance_theorem(a, b, beta, F))(
        U1, U2)
    sub = pred >= 0
    mismatch = int(np.sum(pred[sub] != count[sub]))
    kmin = np.full(U1.shape, np.nan)
    kmax = np.full(U1.shape, np.nan)
    for i in np.ndindex(U1.shape):
        ks = th.stationary_wavenumbers(U1[i], U2[i], beta, F)
        if ks.size:
            kmin[i], kmax[i] = ks[0], ks[-1]
    ks_line = np.full((len(SHEARS), U2_LINE.size, 2), np.nan)
    for a, lam in enumerate(SHEARS):
        for b, u in enumerate(U2_LINE):
            ks = th.stationary_wavenumbers(u + lam, u, beta, F)
            ks_line[a, b, :ks.size] = ks
    drag = np.array(pmap(drag_curve, SHEARS))
    coarse = np.array(pmap(coarse_curve, ASYM_SHEAR))
    asym = np.zeros((ASYM_SHEAR.size, len(ASYM_SPEED), 2))
    thr = np.zeros((ASYM_SHEAR.size, len(KAPPAS), 2))
    dmax = np.zeros((ASYM_SHEAR.size, 2))
    for i in range(ASYM_SHEAR.size):
        for j, u in enumerate(ASYM_SPEED):
            asym[i, j] = (np.interp(-u, U2_COARSE, coarse[i]),
                          np.interp(u, U2_COARSE, coarse[i]))
        thr[i], dD = thresholds(coarse[i], ph.orography.eta0)
        dmax[i] = (dD[U2_COARSE < 0].max(), dD[U2_COARSE > 0].max())
    save_cache("resonance", W=W, count=count, predicted=pred,
               mismatch=np.array(mismatch), kmin=kmin, kmax=kmax,
               shears=np.array(SHEARS), U2_line=U2_LINE, ks_line=ks_line,
               drag=drag, asym_shear=ASYM_SHEAR,
               asym_speed=np.array(ASYM_SPEED), asym=asym,
               coarse_U2=U2_COARSE, coarse_drag=coarse,
               kappas=np.array(KAPPAS), eta_threshold=thr,
               dD_max=dmax, eta_ref=np.array(ph.orography.eta0),
               beta=np.array(beta), F=np.array(F),
               critical_shear=np.array(ph.critical_shear))
    write_csv("fig05_linear_form_stress",
              ["U2"] + ["D_shear_%g" % s for s in SHEARS],
              [[u] + list(drag[:, i]) for i, u in enumerate(U2_LINE)])
    write_csv("fig05_asymmetry",
              ["shear"] + ["ratio_u%g" % u for u in ASYM_SPEED]
              + ["eta_c_east_k%g" % k for k in KAPPAS]
              + ["eta_c_west_k%g" % k for k in KAPPAS],
              [[s] + list(np.abs(asym[i, :, 0] / asym[i, :, 1]))
               + list(thr[i, :, 0]) + list(thr[i, :, 1])
               for i, s in enumerate(ASYM_SHEAR)])
    write_netcdf(
        "stationary_resonance",
        {"U1": (W, "1", "upper-layer zonal wind (units of 10 m/s)"),
         "U2": (W, "1", "lower-layer zonal wind (units of 10 m/s)")},
        {"count": (("U1", "U2"), count, "1",
                   "number of free stationary Rossby waves"),
         "K_min": (("U1", "U2"), kmin, "1",
                   "smallest stationary total wavenumber"),
         "K_max": (("U1", "U2"), kmax, "1",
                   "largest stationary total wavenumber")},
        "Stationary Rossby resonance of the two-layer model",
        attrs=dict(beta=beta, F=F, criterion_mismatches=mismatch))


if __name__ == "__main__":
    main()

"""Nonlinear steady branches with their stability (Fig. 6).

For each reference shear the steady state of the full nonlinear
two-layer model with an interactive mean wind is found by Newton's
method (raising the mountain from a tenth of its height) at a strongly
easterly forcing, and then continued by pseudo-arclength in the forcing
wind ``U_star`` to strong westerlies. Where the corrector fails (a
dense set of resonant turning points), the continuation is restarted a
little further along, and if that fails the branch is completed
backward from the westerly end; the pieces are separated by gaps. At
every point the leading eigenvalues of the Jacobian give the stability;
folds are the turns of the branch. The linear equilibrium curve is
stored alongside.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from orographic_asymmetry_qg import theory as th  # noqa: E402
from orographic_asymmetry_qg.experiments import (CONT_GRID,  # noqa: E402
                                                 SHEARS, pmap, reference)
from orographic_asymmetry_qg.io import (has_cache, load_cache,  # noqa
                                        save_cache, write_csv,
                                        write_netcdf)
from orographic_asymmetry_qg.spectral import Grid  # noqa: E402
from orographic_asymmetry_qg.steady import folds, solve_steady  # noqa: E402

U2_START, U2_END = -1.5, 1.8
N_EIG = 6


def branch(lam):
    """Continue the steady branch at shear ``lam`` (cached per shear)."""
    name = "branch_s%g" % lam
    if has_cache(name):
        return load_cache(name)
    ph = reference(shear=lam, U=U2_START + 0.5 * lam,
                   U_star=U2_START + 0.5 * lam)
    prob, x, info = solve_steady(ph, CONT_GRID)
    if not info["converged"]:
        raise RuntimeError("no steady state at shear %g" % lam)
    lo, hi = U2_START + 0.5 * lam - 0.05, U2_END + 0.5 * lam
    pieces = []
    lam0 = ph.U_star
    while True:
        br = prob.continue_branch(x, lam0, (lo, hi), ds=0.01,
                                  ds_min=5e-4, ds_max=0.03, direction=1.0,
                                  n_max=800, n_eig=N_EIG, verbose=True)
        pieces.append(br)
        end = br["lam"][-1]
        if end >= hi or len(pieces) > 8:
            break
        # The corrector failed: restart from a Newton solve a little
        # further along, starting from the last state (a gap in the
        # branch, marked by NaN in the output).
        restarted = False
        for jump in (0.01, 0.02, 0.04, 0.08):
            xr, info = prob.newton(br["x"][-1], end + jump)
            if info["converged"]:
                x, lam0, restarted = xr, end + jump, True
                break
        if not restarted:
            # from scratch: raise the mountain at the new forcing
            for jump in (0.02, 0.05, 0.1):
                p2, xr, info = solve_steady(ph.with_(
                    U=end + jump, U_star=end + jump), CONT_GRID)
                if info["converged"]:
                    prob.set_orography_scale(1.0)
                    x, lam0, restarted = xr, end + jump, True
                    break
        if not restarted:
            break
    end = pieces[-1]["lam"][-1]
    if end < hi:
        # Complete the branch backward from the westerly end down to
        # where the forward sweep failed.
        phw = ph.with_(U=hi, U_star=hi)
        pw, xw, info = solve_steady(phw, CONT_GRID)
        if info["converged"]:
            bw = pw.continue_branch(xw, hi, (end, hi + 0.05), ds=0.01,
                                    ds_min=5e-4, ds_max=0.03,
                                    direction=-1.0, n_max=800,
                                    n_eig=N_EIG, verbose=True)
            pieces.append({k: v[::-1] for k, v in bw.items()
                           if k in ("lam", "U", "tlam", "eig", "x")})
    br = join_pieces(pieces)
    m = prob.m
    D, amp = [], []
    for xx in br["x"]:
        if not np.all(np.isfinite(xx)):
            D.append(np.nan)
            amp.append(np.nan)
            continue
        p, _ = prob.unpack(xx)
        psi = m.invert(p)
        D.append(m.form_stress(psi))
        amp.append(np.sqrt(m.grid.mean_product(psi[-1], psi[-1])))
    good = np.nonzero(np.isfinite(br["lam"]))[0]
    keep = good[np.unique(np.linspace(0, good.size - 1, 60).astype(int))]
    states = np.array([m.grid.ifft(m.invert(prob.unpack(br["x"][i])[0]))
                       for i in keep])
    fl = [f for f in folds(br) if np.isfinite(f[1])]
    out = dict(lam=br["lam"] - 0.5 * lam, U2=br["U"] - 0.5 * lam,
               eig=br["eig"], D=np.array(D), amp=np.array(amp),
               states=states, state_index=keep,
               folds=np.array([(f[1] - 0.5 * lam, f[2] - 0.5 * lam)
                               for f in fl]).reshape(-1, 2))
    save_cache(name, **out)
    return out


def join_pieces(pieces):
    """Concatenate continuation pieces, separated by NaN rows."""
    out = {}
    for k in ("lam", "U", "tlam", "eig", "x"):
        parts = []
        for i, p in enumerate(pieces):
            if i:
                gap = np.full((1,) + p[k].shape[1:], np.nan,
                              dtype=p[k].dtype)
                parts.append(gap)
            parts.append(p[k])
        out[k] = np.concatenate(parts)
    return out


def linear_curve(lam):
    """Linear equilibrium curve on the continuation grid."""
    ph = reference(shear=lam)
    g = Grid(*CONT_GRID)
    u = np.linspace(U2_START, U2_END, 661) + 0.5 * lam
    us, D = th.linear_equilibrium_curve(ph, g, u)
    return u - 0.5 * lam, us - 0.5 * lam, D


def main():
    res = pmap(branch, SHEARS)
    lin = [linear_curve(lam) for lam in SHEARS]
    out = {}
    rows = []
    for lam, r, (u, us, D) in zip(SHEARS, res, lin):
        tag = "s%g" % lam
        for k, v in r.items():
            out[tag + "_" + k] = v
        out[tag + "_lin_U2"], out[tag + "_lin_Ustar"] = u, us
        out[tag + "_lin_D"] = D
        for i in range(r["lam"].size):
            rows.append((lam, r["lam"][i], r["U2"][i], r["D"][i],
                         r["amp"][i], r["eig"][i, 0].real,
                         abs(r["eig"][i, 0].imag)))
    save_cache("branches", shears=np.array(SHEARS), **out)
    write_csv("fig06_branches",
              ["shear", "U2_star", "U2", "form_stress", "rms_psi2",
               "growth_rate", "frequency"], rows)
    for lam in SHEARS:
        tag = "s%g" % lam
        n = out[tag + "_lam"].size
        write_netcdf(
            "steady_branch_shear_%s" % ("%g" % lam).replace(".", "p"),
            {"point": (np.arange(n), "1", "index along the branch")},
            {"U2_star": (("point",), out[tag + "_lam"], "1",
                         "surface forcing wind (units of 10 m/s)"),
             "U2": (("point",), out[tag + "_U2"], "1",
                    "lower-layer zonal wind (units of 10 m/s)"),
             "form_stress": (("point",), out[tag + "_D"], "1",
                             "orographic form stress"),
             "growth_rate": (("point",), out[tag + "_eig"][:, 0].real,
                             "1", "largest growth rate of perturbations")},
            "Steady nonlinear branch, two-layer model, shear %g" % lam,
            attrs=dict(shear=lam, eta0=reference().orography.eta0,
                       kappa=reference().kappa))


if __name__ == "__main__":
    main()

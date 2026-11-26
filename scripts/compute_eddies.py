"""Robustness to baroclinic eddies (Fig. 9).

With a supercritical shear of 26 m/s (1.2 times the Phillips
threshold), transient baroclinic eddies fill the domain. The mean
winds are held fixed with a lower-layer wind of +5 or -5 m/s, and the
model is integrated for 200 time units (2.9 years), the first 80
discarded. The time-mean form stress, the time-mean (stationary)
lower-layer streamfunction and the eddy statistics are compared with
the linear prediction, to test whether the westerly-easterly asymmetry
survives the eddies.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from orographic_asymmetry_qg import theory as th  # noqa: E402
from orographic_asymmetry_qg.experiments import (RUN_GRID,  # noqa: E402
                                                 integrate, pmap, reference)
from orographic_asymmetry_qg.io import (save_cache, write_csv,  # noqa: E402
                                        write_netcdf)
from orographic_asymmetry_qg.spectral import Grid  # noqa: E402

SHEAR = 2.6
WINDS = (-0.5, 0.5)
T_RUN, T_SPIN = 200.0, 80.0


def run(u2):
    """One eddying integration with fixed mean winds."""
    ph = reference(shear=SHEAR, U=u2 + 0.5 * SHEAR, interactive=False)
    r = integrate(ph, T_RUN, every=0.1, t_mean=T_SPIN, seed=21)
    sel = r["t"] >= T_SPIN
    lin = th.linear_form_stress(ph, Grid(*RUN_GRID))
    return dict(t=r["t"], D=r["form_stress"], ke=r["ke"],
                D_mean=r["form_stress"][sel].mean(),
                D_sem=block_sem(r["form_stress"][sel]),
                psi_mean=r["psi_mean"], D_linear=lin,
                snapshot=r["model"].grid_fields()["q"], p=r["state"][0])


def block_sem(x, n_blocks=12):
    """Standard error of the mean from block averages."""
    b = np.array_split(x, n_blocks)
    m = np.array([v.mean() for v in b])
    return m.std(ddof=1) / np.sqrt(n_blocks)


def main():
    res = pmap(run, WINDS)
    out = dict(shear=np.array(SHEAR), winds=np.array(WINDS))
    for u2, r in zip(WINDS, res):
        tag = "east" if u2 < 0 else "west"
        for k, v in r.items():
            out["%s_%s" % (tag, k)] = np.asarray(v)
    save_cache("eddies", **out)
    write_csv("fig09_eddies",
              ["U2", "D_time_mean", "D_standard_error", "D_linear"],
              [(u2, r["D_mean"], r["D_sem"], r["D_linear"])
               for u2, r in zip(WINDS, res)])
    g = Grid(*RUN_GRID)
    write_netcdf(
        "eddying_time_means",
        {"layer": (np.array([1.0, 2.0]), "1", "layer (1 upper)"),
         "y": (g.y, "1", "meridional coordinate (units of 4505 km)"),
         "x": (g.x, "1", "zonal coordinate (units of 4505 km)")},
        {"psi_mean_east": (("layer", "y", "x"), out["east_psi_mean"], "1",
                           "time-mean eddy streamfunction, U2 = -5 m/s"),
         "psi_mean_west": (("layer", "y", "x"), out["west_psi_mean"], "1",
                           "time-mean eddy streamfunction, U2 = +5 m/s")},
        "Time-mean flow over the mountain with baroclinic eddies",
        attrs=dict(shear=SHEAR))


if __name__ == "__main__":
    main()

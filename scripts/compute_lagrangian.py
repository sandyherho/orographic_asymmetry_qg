"""Lagrangian stirring by the orographic flow (Fig. 8).

Starting from the final states of the fixed-forcing runs, forward
finite-time Lyapunov exponents of the lower layer are computed over 20
days on a 256 x 128 seed lattice for the surface forcing wind with the
largest form-stress variability under shear, and for the same forcing
without shear. Repeating the computation for successive windows gives
the distribution of the exponents.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from orographic_asymmetry_qg.experiments import (DYN_GRID,  # noqa: E402
                                                 pmap, reference)
from orographic_asymmetry_qg.io import (load_cache, save_cache,  # noqa
                                        write_csv)
from orographic_asymmetry_qg.lagrangian import ftle_from_model  # noqa
from orographic_asymmetry_qg.model import QGModel  # noqa: E402
from orographic_asymmetry_qg.params import DAY, T_STAR  # noqa: E402

T_FTLE = 20 * DAY / T_STAR
N_WINDOWS = 4
SEEDS = (256, 128)


def case(args):
    """FTLE fields over successive windows for one run."""
    lam, j = args
    d = load_cache("dynamics")
    tag = "s%g" % lam
    u2s = float(d["forcings"][j])
    ph = reference(shear=lam, U_star=u2s + 0.5 * lam)
    m = QGModel(ph, DYN_GRID, dt=0.01)
    m.set_state(d[tag + "_p"][j], float(d[tag + "_U"][j]), 0.0)
    m.adapt_dt(0.3)
    fields = []
    for _ in range(N_WINDOWS):
        X, Y, F = ftle_from_model(m, SEEDS, T_FTLE, layer=-1)
        fields.append(F)
    return X, Y, np.array(fields), u2s


def main():
    d = load_cache("dynamics")
    j = int(np.argmax(d["s1.5_D"].std(axis=1)))
    res = pmap(case, [(1.5, j), (0.0, j)])
    out = dict(forcing=np.array(res[0][3]), T=np.array(T_FTLE),
               X=res[0][0], Y=res[0][1], sheared=res[0][2],
               unsheared=res[1][2])
    save_cache("lagrangian", **out)
    edges = np.linspace(0, max(out["sheared"].max(),
                               out["unsheared"].max()), 61)
    hs, _ = np.histogram(out["sheared"], edges, density=True)
    hu, _ = np.histogram(out["unsheared"], edges, density=True)
    write_csv("fig08_ftle_pdf", ["ftle", "pdf_sheared", "pdf_unsheared"],
              [(0.5 * (edges[i] + edges[i + 1]), hs[i], hu[i])
               for i in range(hs.size)])


if __name__ == "__main__":
    main()

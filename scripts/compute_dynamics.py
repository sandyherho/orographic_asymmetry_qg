"""Time-dependent flow along the forcing wind (Figs. 7 and 8).

For no shear and for the reference shear (15 m/s), the interactive
two-layer model is integrated

* with the forcing wind ramped slowly up and then down across surface
  easterlies and weak westerlies (hysteresis test), and
* for 300 time units (4.3 years) at each of a set of fixed forcing
  winds, the first 80 discarded.

From the fixed-forcing runs come the time-mean form stress and mean
wind, the variability, the dominant period, and the permutation
entropy and statistical complexity of the form stress.
The final states are kept for the Lagrangian analysis and animations.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from orographic_asymmetry_qg import diagnostics as dg  # noqa: E402
from orographic_asymmetry_qg.experiments import (DYN_GRID,  # noqa: E402
                                                 integrate, pmap, reference)
from orographic_asymmetry_qg.io import (has_cache, load_cache,  # noqa
                                        save_cache, write_csv,
                                        write_netcdf)

DYN_SHEARS = (0.0, 1.5)
FORCINGS = np.round(np.arange(-1.0, 0.61, 0.1), 2)
T_RUN, T_SPIN = 300.0, 80.0
RAMP = (-1.2, 0.6)
T_RAMP = 600.0
EVERY = 0.1


def fixed(args):
    """A long integration at fixed surface forcing wind (cached)."""
    lam, u2s = args
    name = "dyn_s%g_u%+.2f" % (lam, u2s)
    if has_cache(name):
        return load_cache(name)
    ph = reference(shear=lam, U=u2s + 0.5 * lam, U_star=u2s + 0.5 * lam)
    r = integrate(ph, T_RUN, grid=DYN_GRID, every=EVERY, t_mean=T_SPIN,
                  seed=7)
    # the samples fall within one step of the nominal times; put them
    # on the common grid so that all runs share a time axis
    tc = np.arange(T_SPIN, T_RUN, EVERY)
    D = np.interp(tc, r["t"], r["form_stress"])
    U2 = np.interp(tc, r["t"], r["U"]) - 0.5 * lam
    E = np.interp(tc, r["t"], r["energy"])
    if D.std() > 1e-6 * max(abs(D.mean()), 1e-12):
        f, P = periodogram(D, EVERY)
        period = 1.0 / f[1:][np.argmax(P[1:])]
        c, h = dg.statistical_complexity(D, order=5, delay=5)
    else:       # a steady state: no oscillation, ordinal patterns of noise
        period = c = h = np.nan
    p, U = r["state"][0], r["state"][1]
    out = dict(t=tc, D=D, U2=U2, E=E, period=np.array(period),
               complexity=np.array(c), entropy=np.array(h), p=p,
               U=np.array(U), psi_mean=r["psi_mean"])
    save_cache(name, **out)
    return out


def periodogram(x, dt):
    """One-sided periodogram of a demeaned, Hann-windowed series."""
    x = (x - x.mean()) * np.hanning(x.size)
    P = np.abs(np.fft.rfft(x)) ** 2
    return np.fft.rfftfreq(x.size, dt), P


def ramp(args):
    """Slow up and down ramps of the forcing wind (cached)."""
    lam, sense = args
    name = "ramp_s%g_%s" % (lam, "up" if sense > 0 else "down")
    if has_cache(name):
        return load_cache(name)
    a, b = RAMP if sense > 0 else RAMP[::-1]
    ph = reference(shear=lam, U=a + 0.5 * lam, U_star=a + 0.5 * lam)
    r = integrate(ph, T_RAMP, grid=DYN_GRID, every=0.5, seed=3,
                  U_star_of_t=lambda t: a + 0.5 * lam + (b - a) * t / T_RAMP)
    out = dict(U2_star=r["U_star"] - 0.5 * lam, U2=r["U"] - 0.5 * lam,
               D=r["form_stress"], E=r["energy"])
    save_cache(name, **out)
    return out


def main():
    jobs = [(lam, u) for lam in DYN_SHEARS for u in FORCINGS]
    res = pmap(fixed, jobs)
    ramps = pmap(ramp, [(lam, s) for lam in DYN_SHEARS for s in (1, -1)])
    out = dict(shears=np.array(DYN_SHEARS), forcings=FORCINGS)
    rows = []
    for i, lam in enumerate(DYN_SHEARS):
        tag = "s%g" % lam
        sub = res[i * FORCINGS.size:(i + 1) * FORCINGS.size]
        for k in ("D", "U2", "E"):
            out[tag + "_" + k] = np.array([s[k] for s in sub])
        for k in ("period", "complexity", "entropy"):
            out[tag + "_" + k] = np.array([s[k] for s in sub])
        out[tag + "_t"] = sub[0]["t"]
        out[tag + "_p"] = np.array([s["p"] for s in sub])
        out[tag + "_U"] = np.array([s["U"] for s in sub])
        out[tag + "_psi_mean"] = np.array([s["psi_mean"] for s in sub])
        for j, u in enumerate(FORCINGS):
            s = sub[j]
            rows.append((lam, u, s["U2"].mean(), s["U2"].std(),
                         s["D"].mean(), s["D"].std(), s["E"].mean(),
                         float(s["period"]), float(s["entropy"]),
                         float(s["complexity"])))
        for k, sense in enumerate(("up", "down")):
            rr = ramps[2 * i + k]
            for key, v in rr.items():
                out["%s_ramp_%s_%s" % (tag, sense, key)] = v
    save_cache("dynamics", **out)
    write_csv("fig07_dynamics",
              ["shear", "U2_star", "U2_mean", "U2_std", "D_mean", "D_std",
               "energy_mean", "period", "perm_entropy", "complexity"], rows)
    for lam in DYN_SHEARS:
        tag = "s%g" % lam
        write_netcdf(
            "dynamics_shear_%s" % ("%g" % lam).replace(".", "p"),
            {"forcing": (FORCINGS, "1", "surface forcing wind (10 m/s)"),
             "time": (out[tag + "_t"], "1",
                      "time (units of 5.21 days)")},
            {"U2": (("forcing", "time"), out[tag + "_U2"], "1",
                    "lower-layer zonal wind"),
             "form_stress": (("forcing", "time"), out[tag + "_D"], "1",
                             "orographic form stress"),
             "energy": (("forcing", "time"), out[tag + "_E"], "1",
                        "eddy energy")},
            "Interactive two-layer flow at fixed forcing, shear %g" % lam,
            attrs=dict(shear=lam))


if __name__ == "__main__":
    main()

"""Every number quoted from the simulations, as one plain-text report.

Reads the caches written by the compute scripts and writes
``outputs/reports/results.txt``.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from orographic_asymmetry_qg.experiments import reference  # noqa: E402
from orographic_asymmetry_qg.io import load_cache, write_report  # noqa: E402
from orographic_asymmetry_qg.params import (BETA, DAY, L_STAR,  # noqa: E402
                                            T_STAR, U_STAR,
                                            height_of_eta, km)


def head(lines, title):
    """Append a section heading."""
    lines += ["", title, "-" * len(title)]


def scales(lines):
    """Scales and reference parameters."""
    ph = reference()
    head(lines, "Scales and reference parameters")
    lines.append("length %.0f km, velocity %.0f m/s, time %.2f days; "
                 "beta = %.2f" % (L_STAR / 1e3, U_STAR, T_STAR / DAY, BETA))
    lines.append("deformation wavenumber %.2f (L_d = %.0f km); critical "
                 "shear %.3f (%.1f m/s)" % (ph.kd, km(1 / ph.kd),
                                            ph.critical_shear,
                                            10 * ph.critical_shear))
    lines.append("mountain: eta0 = %.0f (%.0f m in the 5 km lower layer), "
                 "e-folding half-width %.0f km" % (
                     ph.orography.eta0, height_of_eta(ph.orography.eta0),
                     km(ph.orography.width)))
    lines.append("bottom drag %.2f (%.1f d), cooling %.2f (%.1f d), "
                 "relaxation of the mean wind %.2f (%.1f d), hyper"
                 "viscosity %.3g" % (ph.drag, T_STAR / DAY / ph.drag,
                                     ph.cooling, T_STAR / DAY / ph.cooling,
                                     ph.kappa, T_STAR / DAY / ph.kappa,
                                     ph.nu))


def resonance(lines):
    """Stationary resonance and linear form stress (Figs. 4, 5)."""
    r = load_cache("resonance")
    head(lines, "Stationary resonance and linear form stress (Figs. 4, 5)")
    lines.append("closed-form criterion against root count: %d mismatches "
                 "on %d subcritical points" % (
                     int(r["mismatch"]), int(np.sum(r["predicted"] >= 0))))
    u2 = r["U2_line"]
    for i, lam in enumerate(r["shears"]):
        ks = r["ks_line"][i]
        e = (u2 < 0) & np.isfinite(ks[:, 0])
        msg = "shear %4.1f m/s: " % (10 * lam)
        if e.any():
            msg += ("stationary waves for surface easterlies from U2 = "
                    "%.2f to 0 (%.0f to 0 m/s)" % (u2[e].min(),
                                                   10 * u2[e].min()))
            j = np.argmin(np.abs(u2 + 0.5))
            if np.isfinite(ks[j, 0]):
                msg += "; at -5 m/s K_s = %.2f (wavelength %.0f km)" % (
                    ks[j, 0], km(2 * np.pi / ks[j, 0]))
        else:
            msg += "no stationary wave for surface easterlies"
        lines.append(msg)
    cu, cd = r["coarse_U2"], r["coarse_drag"]
    for w in (0.5, 1.0, 2.0):
        e = (cu < 0) & (cu >= -w)
        we = (cu > 0) & (cu <= w)
        ratio = np.abs(cd[:, e]).mean(1) / np.abs(cd[:, we]).mean(1)
        lines.append("mean |D| easterly / westerly for |U2| <= %g m/s: "
                     "%.3f (no shear) to %.3f (shear %.0f m/s)"
                     % (10 * w, ratio[0], ratio[-1],
                        10 * r["asym_shear"][-1]))
    thr = height_of_eta(r["eta_threshold"])
    sh = r["asym_shear"]
    for k, kap in enumerate(r["kappas"]):
        cross = sh[np.argmax(thr[:, k, 0] < thr[:, k, 1])]
        lines.append("fold threshold, relaxation %.0f d: easterly %.0f m, "
                     "westerly %.0f m without shear; easterly below "
                     "westerly from shear %.1f m/s; minimum easterly "
                     "threshold %.0f m at %.1f m/s" % (
                         T_STAR / DAY / kap, thr[0, k, 0], thr[0, k, 1],
                         10 * cross, thr[:, k, 0].min(),
                         10 * sh[np.argmin(thr[:, k, 0])]))
    for i in (0, np.argmin(np.abs(sh - 1.5)), len(sh) - 1):
        lines.append("  shear %.1f m/s: max dD/dU east %.3f, west %.3f"
                     % (10 * sh[i], r["dD_max"][i, 0], r["dD_max"][i, 1]))


def cases(lines):
    """Westerly, easterly and sheared cases (Fig. 3)."""
    h = load_cache("cases")
    head(lines, "Infinite ridge and the mountain cases (Fig. 3)")
    sc = km(1.0)
    for name in ("west", "east"):
        y0 = h["ridge_%s_y_drag0" % name] * sc
        y1 = h["ridge_%s_y_drag0.65" % name] * sc
        lines.append("ridge %s: max |displacement| %.0f km inviscid, "
                     "%.0f km with drag" % (name, np.abs(y0).max(),
                                            np.abs(y1).max()))
    for name in ("west", "east", "sheared"):
        psi = h["%s_psi" % name][-1]
        lines.append("mountain %-8s: form stress %+.5f, rms psi2 %.4f, "
                     "residual %.1e, largest growth rate %+.4f"
                     % (name, float(h["%s_form_stress" % name]),
                        np.sqrt(np.mean(psi ** 2)),
                        float(h["%s_residual" % name]),
                        float(h["%s_growth" % name])))


def branches(lines):
    """Steady branches (Fig. 6)."""
    b = load_cache("branches")
    head(lines, "Nonlinear steady branches (Fig. 6)")
    for lam in b["shears"]:
        tag = "s%g" % lam
        ok = np.isfinite(b[tag + "_lam"])
        us = b[tag + "_lam"][ok]
        ev = b[tag + "_eig"][ok, 0]
        gr = ev.real
        un = gr > 0
        msg = "shear %4.1f m/s: %d points in %d pieces, U2* from %.2f to " \
              "%.2f; " % (10 * lam, us.size, 1 + int(np.sum(~ok)),
                          us.min(), us.max())
        if un.any():
            k = np.argmax(gr)
            msg += ("unstable for U2* in [%.3f, %.3f] (%.0f%% of points), "
                    "max growth %.3f per 5.2 d at U2* = %.3f with period "
                    "%.1f d" % (us[un].min(), us[un].max(),
                                100 * un.mean(), gr[k], us[k],
                                2 * np.pi / max(abs(ev[k].imag), 1e-9)
                                * T_STAR / DAY))
        else:
            msg += "stable everywhere (max growth %.4f)" % gr.max()
        lines.append(msg)
        f = b[tag + "_folds"]
        if len(f):
            fl = f[:, 0]
            close = np.abs(np.diff(fl))
            close = close[close < 0.05]
            lines.append("  %d folds, all at U2* between %.3f and %.3f; "
                         "neighbouring folds %.4f to %.4f apart"
                         % (len(f), fl.min(), fl.max(),
                            close.min() if close.size else np.nan,
                            close.max() if close.size else np.nan))
        else:
            lines.append("  no folds")
        D = b[tag + "_D"][ok]
        lines.append("  form stress: max |D| %.4f (westerly forcing), %.4f "
                     "(easterly forcing)" % (np.abs(D[us > 0]).max(),
                                             np.abs(D[us < 0]).max()))


def dynamics(lines):
    """Time-dependent runs (Figs. 7, 8)."""
    d = load_cache("dynamics")
    f = d["forcings"]
    head(lines, "Time-dependent flow (Figs. 7, 8)")
    for lam in d["shears"]:
        tag = "s%g" % lam
        sd = d[tag + "_D"].std(1)
        mn = np.abs(d[tag + "_D"].mean(1))
        act = sd > 0.01 * np.maximum(mn, 1e-3)
        lines.append("shear %4.1f m/s: unsteady (std of D above 1%% of its "
                     "mean) at U2* = %s"
                     % (10 * lam, ", ".join("%g" % x for x in f[act])
                        or "none"))
        for j in np.nonzero(act)[0]:
            lines.append("  U2* %+.1f: mean D %+.4f (std %.4f), U2 %+.3f "
                         "(std %.3f), energy %.4f, period %.1f d, "
                         "entropy %.3f, complexity %.3f" % (
                             f[j], d[tag + "_D"][j].mean(), sd[j],
                             d[tag + "_U2"][j].mean(),
                             d[tag + "_U2"][j].std(),
                             d[tag + "_E"][j].mean(),
                             d[tag + "_period"][j] * T_STAR / DAY,
                             d[tag + "_entropy"][j],
                             d[tag + "_complexity"][j]))
        for sense in ("up", "down"):
            x = d["%s_ramp_%s_U2_star" % (tag, sense)]
            y = d["%s_ramp_%s_U2" % (tag, sense)]
            jump = np.max(np.abs(np.diff(y - x)))
            lines.append("  ramp %s: largest change of U2 - U2* between "
                         "samples %.4f" % (sense, jump))


def lagrangian(lines):
    """Finite-time Lyapunov exponents (Fig. 8)."""
    L = load_cache("lagrangian")
    head(lines, "Lagrangian stirring (Fig. 8)")
    for key in ("sheared", "unsheared"):
        F = L[key]
        lines.append("%s, U2* = %g: mean FTLE %.3f, 95th percentile %.3f "
                     "per 5.2 d" % (key, float(L["forcing"]), F.mean(),
                                    np.percentile(F, 95)))


def eddies(lines):
    """Eddying runs (Fig. 9)."""
    e = load_cache("eddies")
    head(lines, "Supercritical eddying runs (Fig. 9)")
    for tag in ("east", "west"):
        ke = e[tag + "_ke"]
        sel = e[tag + "_t"] >= 80
        lines.append("%s: time-mean form stress %+.4f +- %.4f (2 s.e.), "
                     "linear %+.4f; rms eddy speed upper %.2f, lower %.2f"
                     % (tag, float(e[tag + "_D_mean"]),
                        2 * float(e[tag + "_D_sem"]),
                        float(e[tag + "_D_linear"]),
                        np.sqrt(2 * ke[sel, 0].mean()),
                        np.sqrt(2 * ke[sel, 1].mean())))
    lines.append("ratio |D_east / D_west| = %.3f (linear %.3f)" % (
        abs(float(e["east_D_mean"]) / float(e["west_D_mean"])),
        abs(float(e["east_D_linear"]) / float(e["west_D_linear"]))))


def verification(lines):
    """Verification (Fig. 2)."""
    v = load_cache("verification")
    head(lines, "Verification (Fig. 2)")
    err = v["err"]
    order = np.log2(err[:-1] / err[1:])
    lines.append("ETDRK4 errors %s; observed orders %s" % (
        ", ".join("%.2e" % x for x in err),
        ", ".join("%.2f" % x for x in order)))
    gm = v["growth_model"]
    gt = np.interp(v["growth_k"], v["growth_kk"], v["growth_theory"])
    pos = gt > 0.05
    lines.append("growth rates: max relative difference %.1e over %d "
                 "unstable wavenumbers" % (
                     np.max(np.abs(gm[pos] / gt[pos] - 1)), pos.sum()))
    for tag in ("west", "east"):
        a = v["amp_" + tag]
        lines.append("nonlinear / linear form stress (%s): %s" % (
            tag, ", ".join("%.3f" % x for x in a[:, 0] / a[:, 1])))
    t, dE, cum = v["budget_t"], v["budget_dE"], v["budget_int"]
    lines.append("energy budget: max |E(t) - E(0) - integral| %.1e "
                 "against max |E(t) - E(0)| %.1e" % (
                     np.max(np.abs(dE - cum)), np.max(np.abs(dE))))
    lines.append("Newton residuals: %s" % ", ".join(
        "%.1e" % x for x in v["newton"]))
    for g, r in zip(v["grids"], v["resolution"]):
        lines.append("grid %dx%d: form stress %.6f, U2 %.5f, growth %.4f"
                     % (g[0], g[1], *r))
    lines.append("inviscid invariant drift over 50 units: energy %.1e, "
                 "enstrophy %.1e" % tuple(v["invariant_drift"]))


def main():
    lines = ["Results of the two-layer orographic model",
             "=========================================",
             "All values are outputs of an idealized model with no "
             "observational data; see the README for assumptions and "
             "limitations. Winds are in units of 10 m/s and times of "
             "5.21 days unless stated."]
    for f in (scales, verification, cases, resonance, branches,
              dynamics, lagrangian, eddies):
        try:
            f(lines)
        except FileNotFoundError as exc:
            lines.append("(missing: %s)" % exc)
    write_report("results", lines)


if __name__ == "__main__":
    main()

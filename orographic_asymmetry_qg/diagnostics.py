"""Energy, budgets, spectra and information-theoretic diagnostics.

The total eddy energy of the two-layer model is

    E = 1/2 sum_j <|grad psi_j|^2> + F/2 <(psi_1 - psi_2)^2>,

and of the one-layer model ``1/2 <|grad psi|^2>``. Its budget is

    dE/dt = C_bc + C_oro - D_drag - D_cool - D_hyp,

with the baroclinic conversion ``C_bc = Lambda F <psi_1 d psi_2/dx>``,
the conversion from the mean flow by the orography ``C_oro = -U_b <eta
d psi_b / dx>`` (minus the bottom wind times the form stress), the drag
``D_drag = r <|grad psi_b|^2>``, the cooling ``D_cool = alpha F <(psi_1
- psi_2)^2>`` and the hyperviscous loss.

The information measures are the Shannon entropy of the distribution
of energy over wavenumbers (spectral entropy), the Bandt-Pompe
permutation entropy and Jensen-Shannon statistical complexity of a
time series, and the Fisher information of a one-parameter family of
probability densities.
"""

import itertools
import math

import numpy as np


# -- energy ------------------------------------------------------------------
def energy(model, p=None):
    """Kinetic, available potential and total eddy energy."""
    g = model.grid
    p = model.p if p is None else p
    psi = model.invert(p)
    ke = 0.5 * g.mean_product(psi, g.k2 * psi)
    out = dict(ke=ke)
    if model.nl == 2:
        d = psi[0] - psi[1]
        ape = 0.5 * model.F * g.mean_product(d, d)
        out.update(ape=ape, total=float(ke.sum() + ape))
    else:
        out.update(ape=0.0, total=float(ke.sum()))
    return out


def enstrophy(model, p=None):
    """Potential enstrophy ``1/2 sum_j <q_j^2>`` including orography."""
    g = model.grid
    q = (model.p if p is None else p).copy()
    q[-1] = q[-1] + model.eta_h
    return float(0.5 * np.sum(g.mean_product(q, q)))


def energy_budget(model, p=None, U=None):
    """Terms of the eddy energy budget at the current state.

    Returns
    -------
    dict
        ``dEdt`` (from the full tendency), ``baroclinic``,
        ``orographic``, ``drag``, ``cooling``, ``hyperviscous`` and the
        ``residual`` ``dEdt - (sum of sources - sinks)``.
    """
    g = model.grid
    p = model.p if p is None else p
    U = model.U if U is None else U
    psi = model.invert(p)
    dp, _ = model.tendency(p, U)
    dEdt = -float(np.sum(g.mean_product(psi, dp)))
    Uj = model.layer_winds(U)
    oro = -float(Uj[-1] * model.form_stress(psi))
    drag = float(model.phys.drag * g.mean_product(psi[-1],
                                                  g.k2 * psi[-1]))
    hyp = float(np.sum(g.mean_product(psi, model.L * p)))
    if model.nl == 2:
        bc = float(model.shear * model.F
                   * g.mean_product(psi[0], g.ikx * psi[1]))
        d = psi[0] - psi[1]
        cool = float(model.phys.cooling * model.F * g.mean_product(d, d))
    else:
        bc = cool = 0.0
    res = dEdt - (bc + oro - drag - cool - hyp)
    return dict(dEdt=dEdt, baroclinic=bc, orographic=oro, drag=drag,
                cooling=cool, hyperviscous=hyp, residual=res)


# -- spectra -----------------------------------------------------------------
def zonal_spectrum(model, layer=-1, p=None):
    """Kinetic energy of each zonal wavenumber (integer ``m_x``)."""
    g = model.grid
    psi = model.invert(model.p if p is None else p)[layer]
    e = 0.5 * g.weight * g.k2 * np.abs(psi) ** 2
    mx = g.mx[0].astype(int)
    return np.arange(mx.max() + 1), np.array(
        [e[:, mx == k].sum() for k in range(mx.max() + 1)])


def isotropic_spectrum(model, layer=-1, p=None, field=None):
    """Kinetic energy in shells of total wavenumber of width ``2 pi/Lx``."""
    g = model.grid
    if field is None:
        psi = model.invert(model.p if p is None else p)[layer]
    else:
        psi = field
    e = 0.5 * g.weight * g.k2 * np.abs(psi) ** 2
    dk = 2 * np.pi / g.lx
    k = np.sqrt(g.k2)
    nb = int(g.kmax / dk) + 1
    idx = np.rint(k / dk).astype(int)
    spec = np.bincount(idx[g.keep], weights=e[g.keep], minlength=nb + 1)
    return dk * np.arange(spec.size), spec


# -- information measures ---------------------------------------------------
def spectral_entropy(spec):
    """Normalized Shannon entropy of a nonnegative spectrum.

    ``H = -sum p_k log p_k / log n`` with ``p_k = spec_k / sum spec``
    over the ``n`` nonempty bins; 0 for a single mode, 1 for equipartition.
    """
    s = np.asarray(spec, float)
    s = s[s > 0]
    if s.size < 2:
        return 0.0
    p = s / s.sum()
    return float(-np.sum(p * np.log(p)) / np.log(s.size))


def ordinal_distribution(x, order=4, delay=1):
    """Bandt-Pompe distribution of ordinal patterns of a time series."""
    x = np.asarray(x, float)
    n = x.size - (order - 1) * delay
    if n <= 0:
        raise ValueError("series too short")
    emb = np.stack([x[i * delay:i * delay + n] for i in range(order)], 1)
    ranks = np.argsort(np.argsort(emb, axis=1, kind="stable"), axis=1)
    codes = ranks @ (order ** np.arange(order))
    perms = [sum(r * order ** i for i, r in enumerate(p))
             for p in itertools.permutations(range(order))]
    lut = {c: i for i, c in enumerate(perms)}
    counts = np.zeros(len(perms))
    for c in codes:
        counts[lut[int(c)]] += 1
    return counts / counts.sum()


def permutation_entropy(x, order=4, delay=1):
    """Normalized permutation entropy (Bandt and Pompe 2002)."""
    p = ordinal_distribution(x, order, delay)
    nz = p[p > 0]
    return float(-np.sum(nz * np.log(nz)) / np.log(math.factorial(order)))


def statistical_complexity(x, order=4, delay=1):
    """Jensen-Shannon statistical complexity (Rosso et al. 2007).

    ``C = Q_J[P, P_e] H[P]``, the product of the normalized
    permutation entropy and the normalized Jensen-Shannon divergence
    from the uniform distribution ``P_e``.
    """
    p = ordinal_distribution(x, order, delay)
    n = p.size
    pe = np.full(n, 1.0 / n)

    def S(q):
        q = q[q > 0]
        return -np.sum(q * np.log(q))

    js = S(0.5 * (p + pe)) - 0.5 * S(p) - 0.5 * S(pe)
    js_max = -0.5 * ((n + 1) / n * np.log(n + 1) + np.log(n)
                     - 2 * np.log(2 * n))
    H = S(p) / np.log(n)
    return float(js / js_max * H), float(H)


def fisher_information(lam, samples, grid_pts=200, bandwidth=None):
    """Fisher information of the densities of ``samples`` along ``lam``.

    Each row of ``samples`` is a sample of a scalar for the parameter
    value in ``lam``. Densities are Gaussian kernel estimates on a
    common grid; the information ``I(lam) = int (d_lam p)^2 / p dx`` is
    computed with the equivalent, better conditioned form ``4 int (d_lam
    sqrt(p))^2 dx`` and centered differences in ``lam``.

    Returns
    -------
    lam_mid : ndarray
        Interior parameter values.
    info : ndarray
    """
    lam = np.asarray(lam, float)
    allv = np.concatenate([np.ravel(s) for s in samples])
    lo, hi = allv.min(), allv.max()
    pad = 0.1 * (hi - lo + 1e-12)
    xg = np.linspace(lo - pad, hi + pad, grid_pts)
    dens = []
    for s in samples:
        s = np.ravel(s)
        bw = bandwidth or 1.06 * s.std() * s.size ** -0.2 + 1e-9
        d = np.exp(-0.5 * ((xg[:, None] - s[None]) / bw) ** 2).sum(1)
        dens.append(d / np.trapezoid(d, xg))
    r = np.sqrt(np.array(dens))
    dr = (r[2:] - r[:-2]) / (lam[2:] - lam[:-2])[:, None]
    return lam[1:-1], 4 * np.trapezoid(dr ** 2, xg, axis=1)

"""Closed-form and linear results.

* Stationary Rossby resonance of the two-layer model and its count in
  the plane of the layer winds (the generalization of the classical
  westerly-easterly asymmetry).
* The linear steady response to orography and the linear form stress
  for any orography on the model grid, with the bottom drag and the
  hyperviscosity of the numerical model.
* The equilibrium curve ``U_star(U)`` of the interactive mean wind in
  the linear approximation.
* The Phillips dispersion relation for baroclinic instability and the
  free Rossby wave frequencies.
* the parcel argument for a meridionally uniform ridge, written as
  a nonlinear trajectory equation, and the exact steady solution of the
  barotropic model over such a ridge.
"""

import numpy as np
from scipy.integrate import solve_ivp


# -- stationary resonance ---------------------------------------------------
def stationary_wavenumbers(U1, U2, beta, F):
    """Total wavenumbers ``K > 0`` of free stationary two-layer waves.

    Stationary (zero frequency) inviscid waves of the two-layer model
    with layer winds ``U1, U2`` and PV gradients ``Q1 = beta + F
    (U1 - U2)``, ``Q2 = beta - F (U1 - U2)`` satisfy, with ``s = K^2 +
    F``,

        U1 U2 s^2 - (Q1 U2 + Q2 U1) s + Q1 Q2 - U1 U2 F^2 = 0.

    Returns the sorted positive roots ``K`` (an empty array if none).
    """
    lam = U1 - U2
    Q1, Q2 = beta + F * lam, beta - F * lam
    a = U1 * U2
    b = -(Q1 * U2 + Q2 * U1)
    c = Q1 * Q2 - U1 * U2 * F ** 2
    if abs(a) < 1e-14:
        s = np.array([-c / b]) if abs(b) > 1e-14 else np.array([])
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            return np.array([])
        sq = np.sqrt(disc)
        s = np.array([(-b - sq) / (2 * a), (-b + sq) / (2 * a)])
    k2 = s - F
    return np.sort(np.sqrt(k2[k2 > 0]))


def barotropic_stationary_wavenumber(U, beta):
    """``K_s = sqrt(beta / U)`` for westerlies, ``nan`` otherwise."""
    U = np.asarray(U, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(U > 0, np.sqrt(beta / U), np.nan)


def resonance_count(U1, U2, beta, F):
    """Number of stationary wavenumbers (vectorized over winds)."""
    U1, U2 = np.broadcast_arrays(np.asarray(U1, float),
                                 np.asarray(U2, float))
    out = np.zeros(U1.shape, int)
    for i in np.ndindex(U1.shape):
        out[i] = stationary_wavenumbers(U1[i], U2[i], beta, F).size
    return out


def resonance_theorem(U1, U2, beta, F):
    """Number of stationary waves predicted by the closed-form criterion.

    For subcritical shear (``Q1, Q2 > 0``):

    * easterlies in both layers: no stationary wave;
    * opposite signs (``U1 U2 < 0``): exactly one stationary wave if
      ``U1 + U2 < beta / F`` (always true when the column mean is
      easterly), none otherwise;
    * westerlies in both layers: two, one or none, depending on the
      discriminant.

    The opposite-sign case follows because the quadratic in ``s``
    has a negative leading coefficient, a positive constant term and
    takes the value ``beta (beta - F (U1 + U2))`` at ``s = F``.
    Returns ``-1`` for supercritical shear, where the criterion is not
    stated.
    """
    lam = U1 - U2
    Q1, Q2 = beta + F * lam, beta - F * lam
    if Q1 <= 0 or Q2 <= 0:
        return -1
    if U1 < 0 and U2 < 0:
        return 0
    if U1 * U2 < 0:
        return int(U1 + U2 < beta / F)
    return stationary_wavenumbers(U1, U2, beta, F).size


# -- linear forced response on the model grid ------------------------------
def _modes(grid, eta_h):
    sel = grid.keep & (grid.mx > 0)
    return (sel, grid.k2[sel], grid.kx[sel], eta_h[sel], grid.weight[sel])


def linear_response(phys, grid, U=None, nu=None):
    """Steady linear response ``psi_hat`` to the orography.

    Solves, mode by mode, the steady equations without Jacobians but
    with the bottom drag and hyperviscosity of the numerical model.

    Parameters
    ----------
    phys : Physics
    grid : Grid
    U : float, optional
        Column-mean wind (default ``phys.U``).
    nu : float, optional
        Hyperviscosity coefficient (default as in ``QGModel``).

    Returns
    -------
    ndarray
        Streamfunction coefficients, shape ``(nlayers, ny, nx//2+1)``.
    """
    U = phys.U if U is None else U
    ph = phys.with_(U=U)
    eta_h = grid.truncate(grid.fft(ph.orography.field(grid)))
    if nu is None:
        nu = ph.nu if ph.nu is not None else \
            ph.nu_rate / grid.kmax ** (2 * ph.nu_order)
    sel, k2, kx, e, _ = _modes(grid, eta_h)
    hv = nu * k2 ** ph.nu_order
    r = ph.drag
    Uj, Q = ph.layer_winds, ph.pv_gradients
    out = np.zeros((ph.nlayers,) + grid.k2.shape, complex)
    if ph.nlayers == 1:
        a = 1j * kx * (Q[0] - Uj[0] * k2) - (r + hv) * k2
        out[0][sel] = -1j * kx * Uj[0] * e / a
        return out
    F = ph.F
    m = np.array([[-k2 - F, F + 0 * k2], [F + 0 * k2, -k2 - F]])
    A = np.empty((k2.size, 2, 2), complex)
    for i in range(2):
        for j in range(2):
            A[:, i, j] = -(1j * kx * Uj[i] + hv) * m[i, j]
        A[:, i, i] -= 1j * kx * Q[i]
    A[:, 1, 1] += r * k2
    a = ph.cooling * F
    A[:, 0, 0] += a
    A[:, 0, 1] -= a
    A[:, 1, 0] -= a
    A[:, 1, 1] += a
    rhs = np.zeros((k2.size, 2, 1), complex)
    rhs[:, 1, 0] = 1j * kx * Uj[1] * e
    psi = np.linalg.solve(A, rhs)[..., 0]
    out[0][sel], out[1][sel] = psi[:, 0], psi[:, 1]
    return out


def linear_form_stress(phys, grid, U=None, nu=None):
    """Linear form stress ``<eta d psi_b/dx>`` at column-mean wind ``U``."""
    U = phys.U if U is None else U
    eta_h = grid.truncate(grid.fft(phys.orography.field(grid)))
    psi = linear_response(phys, grid, U, nu)
    return float(grid.mean_product(eta_h, grid.ikx * psi[-1]))


def linear_equilibrium_curve(phys, grid, U):
    """Forcing wind ``U_star`` that holds the mean wind at ``U``.

    In equilibrium ``kappa (U - U_star) = D(U) / N``, so ``U_star = U -
    D(U) / (N kappa)``. Several ``U`` with the same ``U_star`` are
    multiple equilibria; turning points of ``U_star(U)`` are folds.
    """
    U = np.atleast_1d(np.asarray(U, float))
    D = np.array([linear_form_stress(phys, grid, u) for u in U])
    return U - D / (phys.nlayers * phys.kappa), D


def folds_of_curve(U, U_star):
    """Turning points of ``U_star(U)`` by quadratic interpolation."""
    d = np.diff(U_star)
    out = []
    for i in np.nonzero(np.sign(d[1:]) * np.sign(d[:-1]) < 0)[0] + 1:
        x = U[i - 1:i + 2]
        c = np.polyfit(x - x[1], U_star[i - 1:i + 2], 2)
        xv = -c[1] / (2 * c[0]) + x[1]
        out.append((xv, np.polyval(c, xv - x[1])))
    return np.array(out).reshape(-1, 2)


# -- instability and free waves --------------------------------------------
def phillips_growth_rate(k, ky, U1, U2, beta, F, drag=0.0, cooling=0.0):
    """Growth rate of the most unstable two-layer normal mode at ``(k, ky)``.

    Normal modes ``exp(i(k x + ky y) + sigma t)`` of the linearized
    two-layer model with uniform winds, bottom drag and Newtonian
    cooling; returns ``max Re(sigma)``. Without damping this is the
    Phillips problem.
    """
    K2 = k * k + ky * ky
    lam = U1 - U2
    Q = (beta + F * lam, beta - F * lam)
    U = (U1, U2)
    B = np.array([[-K2 - F, F], [F, -K2 - F]], complex)
    A = np.empty((2, 2), complex)
    for i in range(2):
        for j in range(2):
            A[i, j] = -1j * k * U[i] * B[i, j]
        A[i, i] -= 1j * k * Q[i]
    A[1, 1] += drag * K2
    a = cooling * F
    A += np.array([[a, -a], [-a, a]])
    sigma = np.linalg.eigvals(np.linalg.solve(B, A))
    return float(np.max(sigma.real))


def rossby_frequency(k, ky, U, beta, kd2=0.0):
    """Frequency of a barotropic (or equivalent barotropic) Rossby wave."""
    return U * k - beta * k / (k * k + ky * ky + kd2)


# -- infinite ridge ---------------------------------------------------------
def ridge_steady_streamfunction(x, U, beta, eta_x, drag=0.0, nu=0.0,
                                nu_order=4):
    """Exact steady barotropic flow over a meridionally uniform ridge.

    For ``psi = psi(x)`` the Jacobians vanish identically, so the
    linear Fourier solution of

        U (psi''' + eta') + beta psi' = -drag psi'' - nu (-d2)^n psi''

    is an exact solution of the nonlinear model. ``x`` must be a
    uniform periodic grid on which ``eta_x`` is given.
    """
    n = x.size
    L = n * (x[1] - x[0])
    k = 2 * np.pi * np.fft.rfftfreq(n, L / n)
    e = np.fft.rfft(eta_x)
    psi = np.zeros_like(e)
    nz = k > 0
    kk = k[nz]
    a = 1j * kk * (beta - U * kk ** 2) - (drag + nu * kk ** (2 * nu_order)) \
        * kk ** 2
    psi[nz] = -1j * kk * U * e[nz] / a
    return np.fft.irfft(psi, n)


def parcel_trajectory(eta_fun, U, beta, x_span, y0=0.0, slope0=0.0, pv0=0.0,
                      drag=0.0, zeta0=None, deta_fun=None, n_out=2000):
    """The parcel argument as a nonlinear trajectory equation.

    A parcel moves at the constant speed ``|U|`` with heading ``alpha``;
    its relative vorticity is the curvature vorticity ``zeta = |U|
    d(alpha)/ds = d(alpha)/dt``. Without drag it conserves the QG
    potential vorticity ``zeta + beta y + eta``:

        dx/dt = |U| cos(alpha),  dy/dt = |U| sin(alpha),
        d(alpha)/dt = zeta = pv0 - beta y - eta(x).

    With linear drag the PV is not conserved and ``zeta`` becomes a
    fourth variable, ``d zeta/dt = -beta dy/dt - eta'(x) dx/dt - drag
    zeta``, starting from ``zeta0`` (``deta_fun`` gives ``eta'``).

    The parcel starts at ``y0`` with path slope ``dy/dx = slope0``
    (heading ``arctan(slope0)`` for westerlies, ``pi + arctan(slope0)``
    for easterlies). Linearized in the slope it reduces, for either
    sign of ``U``, to the streamline equation of the exact steady QG
    flow over a ridge (``U y'' + beta y + eta = pv0`` without drag). In
    the periodic domain ``pv0`` is the domain mean of ``eta``, because
    the streamfunction has zero mean. For easterlies the equation is
    evanescent, so as an initial-value problem it amplifies errors by
    ``exp(sqrt(beta / |U|) x)`` and should only be integrated across
    the ridge.

    Returns
    -------
    x, y : ndarray
    """
    V = abs(U)
    a0 = np.arctan(slope0) + (0.0 if U > 0 else np.pi)
    if drag > 0:
        def rhs(t, s):
            x, y, a, z = s
            u, v = V * np.cos(a), V * np.sin(a)
            return [u, v, z, -beta * v - deta_fun(x) * u - drag * z]
        z0 = pv0 - beta * y0 - eta_fun(x_span[0] if U > 0 else x_span[1]) \
            if zeta0 is None else zeta0
        s0 = [0.0, y0, a0, z0]
    else:
        def rhs(t, s):
            x, y, a = s
            return [V * np.cos(a), V * np.sin(a),
                    pv0 - beta * y - eta_fun(x)]
        s0 = [0.0, y0, a0]
    x0, x1 = x_span
    s0[0] = x0 if U > 0 else x1
    T = abs(x1 - x0) / V
    sol = solve_ivp(rhs, (0, T), s0, rtol=1e-10, atol=1e-12,
                    dense_output=True, method="DOP853")
    tt = np.linspace(0, T, n_out)
    out = sol.sol(tt)
    return out[0], out[1]

"""Steady states, their continuation and their linear stability.

The steady problem is the full nonlinear model with the time
derivative set to zero. The unknowns are the independent real and
imaginary parts of the retained PV coefficients of every layer and,
when the mean wind is interactive, the mean wind ``U``; the parameter
is the forcing wind ``U_star``. Because the tendency is a quadratic
polynomial of the unknowns, central differences give its Jacobian to
rounding error, and the Jacobian is formed densely in batches. Newton's
method with an LU solve then converges quadratically, branches are
followed by pseudo-arclength continuation (Keller 1977), folds are
located where the extended system's tangent turns, and the stability
of each state is the full eigenvalue spectrum of the Jacobian.
"""

import os

import numpy as np
from scipy.linalg import eig, lu_factor, lu_solve

from .model import QGModel


class SteadyProblem:
    """Steady states of a ``QGModel``.

    Parameters
    ----------
    model : QGModel
        Defines physics, grid and hyperviscosity. If its mean wind is
        interactive the parameter is ``U_star``; otherwise the mean
        wind is fixed at ``model.U``.
    """

    def __init__(self, model):
        self.m = model
        g = model.grid
        sel = g.keep & ((g.mx > 0) | ((g.mx == 0) & (g.my > 0)))
        self.idx = np.nonzero(sel)
        self.nmode = self.idx[0].size
        neg = g.keep & (g.mx == 0) & (g.my < 0)
        ii = np.nonzero(neg)
        self.neg_idx = ii
        # mirror of (my < 0, mx = 0) is (-my, 0)
        self.neg_src = ((-g.my[ii]).astype(int) % g.ny, ii[1])
        self.interactive = bool(model.phys.interactive)
        self.n = 2 * model.nl * self.nmode + int(self.interactive)
        # unknowns are coefficients divided by the number of grid points,
        # i.e. Fourier amplitudes of the fields
        self.scale = float(g.nx * g.ny)

    # -- packing -----------------------------------------------------------
    def pack(self, p, U=None):
        """Real vector of the independent unknowns."""
        c = p[..., self.idx[0], self.idx[1]] / self.scale
        x = np.concatenate([c.real.reshape(c.shape[:-2] + (-1,)),
                            c.imag.reshape(c.shape[:-2] + (-1,))], axis=-1)
        if self.interactive:
            U = np.asarray(U, float)
            x = np.concatenate([x, U[..., None]], axis=-1)
        return x

    def unpack(self, x):
        """PV coefficients (and mean wind) from a real vector."""
        m = self.m
        nl, nm = m.nl, self.nmode
        lead = x.shape[:-1]
        re = x[..., :nl * nm].reshape(lead + (nl, nm))
        im = x[..., nl * nm:2 * nl * nm].reshape(lead + (nl, nm))
        p = np.zeros(lead + (nl,) + m.grid.k2.shape, complex)
        p[..., self.idx[0], self.idx[1]] = (re + 1j * im) * self.scale
        p[..., self.neg_idx[0], self.neg_idx[1]] = np.conj(
            p[..., self.neg_src[0], self.neg_src[1]])
        U = x[..., -1] if self.interactive else np.full(lead, m.U)
        return p, U

    # -- residual and Jacobian ---------------------------------------------
    def residual(self, x, U_star=None):
        """Packed tendency at ``x`` (batched over leading axes)."""
        m = self.m
        p, U = self.unpack(x)
        Np, _ = m.nonlinear(p, U)
        dp = Np + m.L * p
        if self.interactive:
            us = m.phys.U_star if U_star is None else U_star
            dU = (-m.phys.kappa * (U - us)
                  + m.form_stress(m.invert(p)) / m.nl)
            return self.pack(dp, dU)
        return self.pack(dp)

    def jacobian(self, x, U_star=None, batch=192):
        """Dense Jacobian of the residual by central differences.

        The residual is quadratic in ``x``, so the central difference is
        exact apart from rounding.
        """
        n = self.n
        J = np.empty((n, n))
        h = 1e-4 * max(1.0, np.max(np.abs(x)))
        for i0 in range(0, n, batch):
            i1 = min(i0 + batch, n)
            E = np.zeros((i1 - i0, n))
            E[np.arange(i1 - i0), np.arange(i0, i1)] = h
            rp = self.residual(x + E, U_star)
            rm = self.residual(x - E, U_star)
            J[:, i0:i1] = ((rp - rm) / (2 * h)).T
        return J

    def d_residual_d_param(self):
        """Derivative of the residual with respect to ``U_star``."""
        d = np.zeros(self.n)
        if self.interactive:
            d[-1] = self.m.phys.kappa
        return d

    # -- Newton ------------------------------------------------------------
    def newton(self, x0, U_star=None, tol=1e-10, maxit=40):
        """Damped Newton iteration for a steady state at fixed parameter.

        The full step is halved until the residual norm decreases
        (Armijo backtracking), which makes the iteration robust far
        from the solution and quadratic near it.

        Returns
        -------
        x : ndarray
        info : dict
            ``converged``, ``iterations``, ``residual`` (max norm).
        """
        x = np.array(x0, float)
        r = self.residual(x, U_star)
        nr = np.linalg.norm(r)
        for it in range(maxit):
            if np.max(np.abs(r)) < tol:
                return x, dict(converged=True, iterations=it,
                               residual=float(np.max(np.abs(r))))
            dx = lu_solve(lu_factor(self.jacobian(x, U_star)), r)
            a = 1.0
            while a > 1e-4:
                xn = x - a * dx
                rn = self.residual(xn, U_star)
                nn = np.linalg.norm(rn)
                if nn < (1 - 1e-4 * a) * nr:
                    break
                a /= 2
            x, r, nr = xn, rn, nn
        res = float(np.max(np.abs(r)))
        return x, dict(converged=res < tol, iterations=maxit, residual=res)

    def set_orography_scale(self, s):
        """Scale the orography of the model to ``s`` times its nominal."""
        if not hasattr(self, "_eta_h0"):
            self._eta_h0 = self.m.eta_h.copy()
            self._eta0 = self.m.eta.copy()
        self.m.eta_h = s * self._eta_h0
        self.m.eta = s * self._eta0

    def homotopy(self, x0, U_star=None, scales=None):
        """Steady state at full orography by raising it in steps.

        Starts from ``x0`` at the first scale (default ten steps from
        0.1 to 1) and uses each solution as the guess for the next.
        """
        scales = np.linspace(0.1, 1, 10) if scales is None else scales
        x = np.array(x0, float)
        info = {}
        for s in scales:
            self.set_orography_scale(s)
            x, info = self.newton(x, U_star)
            if not info["converged"]:
                break
        return x, info

    def spectrum(self, x, U_star=None, k=None, J=None):
        """Eigenvalues of the Jacobian, sorted by decreasing real part.

        The full spectrum of the dense Jacobian is computed (QR
        algorithm); ``k`` keeps only the leading ones. Arnoldi
        iteration does not converge reliably here because many weakly
        damped Rossby modes crowd the imaginary axis.
        """
        J = self.jacobian(x, U_star) if J is None else J
        w = eig(J, right=False)
        return w[np.argsort(-w.real)][:k]

    # -- continuation ------------------------------------------------------
    def _bordered(self, J, t, w):
        A = np.zeros((self.n + 1, self.n + 1))
        A[:-1, :-1] = J
        A[:-1, -1] = self.d_residual_d_param()
        A[-1] = w * t
        return A

    def _norm_weights(self, theta):
        w = np.full(self.n + 1, theta / max(self.n - 1, 1))
        w[-2] = 1.0
        w[-1] = 1.0
        return w

    def continue_branch(self, x0, lam0, lam_range, ds=0.02, ds_min=1e-4,
                        ds_max=0.06, direction=1.0, n_max=600, tol=1e-8,
                        stability=True, n_eig=6, theta=1e-2,
                        verbose=False):
        """Pseudo-arclength continuation in ``U_star``.

        The arclength is measured in the plane of the mean wind ``U``
        and the parameter ``U_star``, with weight ``theta / n`` on the
        PV unknowns. The corrector is a chord iteration with the
        bordered Jacobian of the last converged point, refreshed with a
        full Newton iteration if the chord iteration stalls.

        Parameters
        ----------
        x0 : ndarray
            A converged steady state at ``U_star = lam0``.
        lam_range : tuple
            Stop when ``U_star`` leaves this interval.
        ds, ds_min, ds_max : float
            Initial, smallest and largest arclength steps.
        direction : float
            Initial sense of travel in ``U_star``.
        stability : bool
            Compute the ``n_eig`` leading eigenvalues at every point.

        Returns
        -------
        dict
            ``lam`` and ``U`` along the branch, the leading eigenvalues
            ``eig``, the states ``x``, the arclength steps and the
            sign of the tangent's ``lam`` component (its changes mark
            folds).
        """
        if not self.interactive:
            raise ValueError("continuation needs an interactive mean wind")
        w = self._norm_weights(theta)

        def tangent(J, t_old):
            rhs = np.zeros(self.n + 1)
            rhs[-1] = 1.0
            t = np.linalg.solve(self._bordered(J, t_old, w), rhs)
            return t / np.sqrt(np.sum(w * t * t))

        x, lam = np.array(x0, float), float(lam0)
        J = self.jacobian(x, lam)
        t = np.zeros(self.n + 1)
        t[-1] = direction
        t = tangent(J, t)
        if np.sign(t[-1]) != np.sign(direction):
            t = -t
        out = dict(lam=[lam], U=[x[-1]], x=[x.copy()], tlam=[t[-1]],
                   eig=[], newton=[0])
        if stability:
            out["eig"].append(self.spectrum(x, lam, n_eig, J))
        for _ in range(n_max):
            z0 = np.concatenate([x, [lam]])
            ok = False
            for _attempt in range(12):
                zp = z0 + ds * t
                z = zp.copy()
                lu = lu_factor(self._bordered(J, t, w))
                for it in range(16):
                    with np.errstate(all="ignore"):
                        r = self.residual(z[:-1], z[-1])
                    G = np.concatenate([r, [np.sum(w * t * (z - zp))]])
                    if not np.all(np.isfinite(G)):
                        break                    # diverged: shorter step
                    if np.max(np.abs(G)) < tol:
                        ok = True
                        break
                    if it == 3:
                        lu = lu_factor(self._bordered(
                            self.jacobian(z[:-1], z[-1]), t, w))
                    z = z - lu_solve(lu, G)
                if ok:
                    break
                ds = max(ds / 2, ds_min)
            if not ok:
                if verbose:
                    print("continuation stopped: no convergence")
                break
            x, lam = z[:-1], z[-1]
            J = self.jacobian(x, lam)
            tn = tangent(J, t)
            if np.dot(w * tn, t) < 0:
                tn = -tn
            t = tn
            out["lam"].append(lam)
            out["U"].append(x[-1])
            out["x"].append(x.copy())
            out["tlam"].append(t[-1])
            out["newton"].append(it)
            if stability:
                out["eig"].append(self.spectrum(x, lam, n_eig, J))
            if verbose:
                ev = out["eig"][-1][0].real if stability else np.nan
                print("[%d] U*=%.4f U=%.4f ds=%.4f it=%d re=%.4f"
                      % (os.getpid(), lam, x[-1], ds, it, ev), flush=True)
            if it <= 7:
                ds = min(ds * 1.3, ds_max)
            elif it > 11:
                ds = max(ds * 0.6, ds_min)
            if not lam_range[0] <= lam <= lam_range[1]:
                break
        for key in ("lam", "U", "tlam", "newton"):
            out[key] = np.array(out[key])
        out["x"] = np.array(out["x"])
        if stability:
            out["eig"] = np.array(out["eig"])
        return out


def folds(branch):
    """Indices and parameter values where the branch turns back."""
    s = np.sign(branch["tlam"])
    ok = np.isfinite(s[1:]) & np.isfinite(s[:-1])
    i = np.nonzero((s[1:] != s[:-1]) & ok)[0]
    out = []
    for k in i:
        # linear interpolation of the zero of the tangent component
        t0, t1 = branch["tlam"][k], branch["tlam"][k + 1]
        a = t0 / (t0 - t1)
        lam = branch["lam"][k] + a * (branch["lam"][k + 1] - branch["lam"][k])
        U = branch["U"][k] + a * (branch["U"][k + 1] - branch["U"][k])
        out.append((k, lam, U))
    return out


def steady_from_linear(model, problem=None):
    """Packed initial guess: the linear response at the current wind."""
    from .theory import linear_response
    prob = problem or SteadyProblem(model)
    psi = linear_response(model.phys, model.grid, model.U, nu=model.nu)
    return prob.pack(model.pv_of_psi(psi), model.U)


def solve_steady(phys, grid=(64, 32), U_star=None, guess=None,
                 homotopy=True):
    """A steady state of ``phys``.

    Starts from ``guess`` or from the linear response to a tenth of the
    orography and raises the orography to its full height.
    """
    m = QGModel(phys, grid)
    prob = SteadyProblem(m)
    if guess is not None:
        x, info = prob.newton(guess, U_star)
        return prob, x, info
    prob.set_orography_scale(0.1)
    from .theory import linear_response
    psi = linear_response(phys.with_(orography__eta0=0.1
                                     * phys.orography.eta0),
                          m.grid, m.U, nu=m.nu)
    x0 = prob.pack(m.pv_of_psi(psi), m.U)
    if homotopy:
        x, info = prob.homotopy(x0, U_star)
    else:
        prob.set_orography_scale(1.0)
        x, info = prob.newton(x0, U_star)
    return prob, x, info

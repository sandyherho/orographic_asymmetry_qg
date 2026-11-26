"""The one- and two-layer quasi-geostrophic model on a beta-plane.

Prognostic variables are the spectral PV anomalies ``p_j`` (relative
vorticity plus stretching, without the orography) and, when the mean
flow is interactive, the column-mean zonal wind ``U``. With
``q_j = p_j + delta_{j,b} eta`` and the layer winds ``U_j``,

    d p_j/dt + J(psi_j, q_j) + U_j d q_j/dx + Q_jy d psi_j/dx
        = -delta_{j,b} r Lap psi_j - nu (-Lap)^n p_j,

    dU/dt = -kappa (U - U_star) + <eta d psi_b/dx> / N,

plus, in the two-layer model, Newtonian cooling ``+-alpha F (psi_1 -
psi_2)`` of the eddy interface displacement in layers 1 and 2. Here
``b`` is the bottom layer, ``N`` the number of layers and the angle
brackets a domain mean. The last term is the orographic form
stress. Time stepping is ETDRK4 (Cox and Matthews 2002) with the
coefficients evaluated by contour integrals (Kassam and Trefethen
2005); the linear part treated exactly is the hyperviscosity, all other
terms are explicit. For the mean wind the scheme reduces to classical
RK4. Lagrangian particles can be carried along with the same stages.
"""

import numpy as np

from .spectral import Grid


def etdrk4_coefficients(L, h, m=32):
    """ETDRK4 coefficients for the diagonal operator ``L`` and step ``h``.

    Parameters
    ----------
    L : ndarray
        Diagonal of the linear operator (real, nonpositive here).
    h : float
        Time step.
    m : int
        Number of points on the contour around each ``h L``.

    Returns
    -------
    dict
        ``E, E2, Q, f1, f2, f3`` with the shape of ``L``.
    """
    L = np.asarray(L, float)
    hl = h * L
    r = np.exp(1j * np.pi * (np.arange(1, m + 1) - 0.5) / m)
    z = hl[..., None] + r
    ez = np.exp(z)
    Q = h * np.real(np.mean((np.exp(z / 2) - 1) / z, axis=-1))
    f1 = h * np.real(np.mean((-4 - z + ez * (4 - 3 * z + z ** 2)) / z ** 3,
                             axis=-1))
    f2 = h * np.real(np.mean((2 + z + ez * (z - 2)) / z ** 3, axis=-1))
    f3 = h * np.real(np.mean((-4 - 3 * z - z ** 2 + ez * (4 - z)) / z ** 3,
                             axis=-1))
    return dict(E=np.exp(hl), E2=np.exp(hl / 2), Q=Q, f1=f1, f2=f2, f3=f3)


class QGModel:
    """Pseudo-spectral QG model.

    Parameters
    ----------
    phys : Physics
    grid : Grid or tuple
        A grid, or ``(nx, ny)`` for the default domain ``2 pi x pi``.
    dt : float
        Time step.
    """

    def __init__(self, phys, grid=(128, 64), dt=0.005):
        self.phys = phys
        self.grid = grid if isinstance(grid, Grid) else Grid(*grid)
        g = self.grid
        self.nl = phys.nlayers
        self.dt = float(dt)
        self._dt_max = self.dt
        self.F = phys.F
        self.shear = phys.shear if self.nl == 2 else 0.0
        self.Qy = np.array(phys.pv_gradients)
        self.eta = phys.orography.field(g)
        self.eta_h = g.truncate(g.fft(self.eta))
        self.eta = g.ifft(self.eta_h)
        k2 = g.k2
        if self.nl == 2:
            det = k2 * (k2 + 2 * self.F)
            det[0, 0] = 1.0
            self._inv = (-(k2 + self.F) / det, -self.F / det)
        else:
            k2s = k2.copy()
            k2s[0, 0] = 1.0
            self._inv = (-1.0 / k2s,)
        nu = phys.nu
        if nu is None:
            nu = phys.nu_rate / g.kmax ** (2 * phys.nu_order)
        self.nu = nu
        self.L = -nu * k2 ** phys.nu_order
        self._coef_cache = {}
        self.set_dt(self.dt)
        self.p = np.zeros((self.nl,) + k2.shape, complex)
        self.U = float(phys.U)
        self.t = 0.0
        self.particles = None
        self._vel = None

    def set_dt(self, dt):
        """Change the time step (ETDRK4 coefficients are cached)."""
        key = float("%.12g" % dt)
        if key not in self._coef_cache:
            self._coef_cache[key] = etdrk4_coefficients(self.L, key)
        self.dt = key
        self.coef = self._coef_cache[key]

    def adapt_dt(self, cfl=0.3, dt_max=None):
        """Pick a step from the ladder ``dt_max / 2^k`` with CFL <= ``cfl``.

        Restricting the steps to a ladder lets the coefficients be
        reused. Returns the new step.
        """
        dt_max = dt_max or self._dt_max
        s = self.max_speed()
        dx = min(self.grid.dx, self.grid.dy)
        dt = dt_max
        while dt * s / dx > cfl and dt > dt_max / 1024:
            dt /= 2
        if dt != self.dt:
            self.set_dt(dt)
        return dt

    # -- state -------------------------------------------------------------
    def layer_winds(self, U=None):
        """Layer zonal winds for column-mean wind ``U``, shape ``(..., N)``."""
        U = self.U if U is None else U
        off = np.array([0.0]) if self.nl == 1 else \
            np.array([0.5 * self.shear, -0.5 * self.shear])
        return np.asarray(U, float)[..., None] + off

    def invert(self, p):
        """Streamfunction coefficients from PV anomaly coefficients.

        ``p`` has shape ``(..., nlayers, ny, nx//2+1)``.
        """
        keep = self.grid.keep
        if self.nl == 1:
            return self._inv[0] * p * keep
        a, b = self._inv
        out = np.empty_like(p)
        out[..., 0, :, :] = (a * p[..., 0, :, :] + b * p[..., 1, :, :]) * keep
        out[..., 1, :, :] = (b * p[..., 0, :, :] + a * p[..., 1, :, :]) * keep
        return out

    def pv_of_psi(self, psi):
        """PV anomaly coefficients from streamfunction coefficients."""
        k2 = self.grid.k2
        if self.nl == 1:
            return -k2 * psi
        F = self.F
        return np.stack([-k2 * psi[0] + F * (psi[1] - psi[0]),
                         -k2 * psi[1] + F * (psi[0] - psi[1])])

    def set_state(self, p=None, U=None, t=None):
        """Set PV coefficients, mean wind and time."""
        if p is not None:
            self.p = self.grid.truncate(np.array(p, complex))
        if U is not None:
            self.U = float(U)
        if t is not None:
            self.t = float(t)

    def perturb(self, amplitude=1e-3, seed=0, kmin=2.0, kmax=12.0):
        """Add a random streamfunction perturbation of rms ``amplitude``."""
        g = self.grid
        rng = np.random.default_rng(seed)
        band = (g.k2 >= kmin ** 2) & (g.k2 <= kmax ** 2) & g.keep
        ph = np.exp(2j * np.pi * rng.random((self.nl,) + g.k2.shape))
        psi = g.fft(g.ifft(band * ph))      # enforce conjugate symmetry
        psi_x = g.ifft(psi)
        psi *= amplitude / np.sqrt(np.mean(psi_x ** 2))
        self.p = g.truncate(self.p + self.pv_of_psi(psi))

    # -- dynamics ----------------------------------------------------------
    def nonlinear(self, p, U, want_vel=False):
        """Explicit tendencies ``(N_p, N_U)``.

        ``p`` may carry leading batch dimensions, ``(..., nlayers, ny,
        nx//2+1)``, with ``U`` of the batch shape. If ``want_vel`` is
        true the grid velocities ``(u_j, v_j)`` of the eddies are also
        returned.
        """
        g, nl = self.grid, self.nl
        psi = self.invert(p)
        q = p.copy()
        q[..., -1, :, :] += self.eta_h
        phys = g.ifft(np.concatenate([-g.iky * psi, g.ikx * psi, q],
                                     axis=-3))
        u = phys[..., :nl, :, :]
        v = phys[..., nl:2 * nl, :, :]
        qq = phys[..., 2 * nl:, :, :]
        fl = g.fft(np.concatenate([u * qq, v * qq], axis=-3))
        Uj = self.layer_winds(U)[..., None, None]
        Qy = self.Qy[:, None, None]
        Np = -(g.ikx * fl[..., :nl, :, :] + g.iky * fl[..., nl:, :, :])
        Np -= Uj * g.ikx * q + Qy * g.ikx * psi
        Np[..., -1, :, :] += self.phys.drag * g.k2 * psi[..., -1, :, :]
        if nl == 2 and self.phys.cooling:
            th = self.phys.cooling * self.F * (psi[..., 0, :, :]
                                               - psi[..., 1, :, :])
            Np[..., 0, :, :] += th
            Np[..., 1, :, :] -= th
        Np = g.truncate(Np)
        NU = 0.0
        if self.phys.interactive:
            NU = (-self.phys.kappa * (np.asarray(U) - self.phys.U_star)
                  + self.form_stress(psi) / nl)
        if want_vel:
            return Np, NU, (u, v)
        return Np, NU

    def form_stress(self, psi):
        """Orographic form stress ``<eta d psi_b/dx>`` on the bottom layer."""
        g = self.grid
        return g.mean_product(self.eta_h, g.ikx * psi[..., -1, :, :])

    def tendency(self, p, U):
        """Full time derivative ``(dp/dt, dU/dt)``."""
        Np, NU = self.nonlinear(p, U)
        return Np + self.L * p, NU

    def _particle_velocity(self, vel, U, xy):
        """Total velocity of particles from grid eddy velocities."""
        g = self.grid
        j = self.particles["layer"]
        u, v = vel[0][j], vel[1][j]
        x, y = xy
        up = g.interp(u, x, y) + self.layer_winds(U)[j]
        vp = g.interp(v, x, y)
        return np.array([up, vp])

    def add_particles(self, x, y, layer=-1):
        """Release passive particles in ``layer`` at positions ``x, y``."""
        self.particles = dict(xy=np.array([np.ravel(x), np.ravel(y)], float),
                              layer=layer)

    def step(self, n=1):
        """Advance ``n`` ETDRK4 steps."""
        c = self.coef
        E, E2, Q = c["E"], c["E2"], c["Q"]
        f1, f2, f3 = c["f1"], c["f2"], c["f3"]
        h = self.dt
        track = self.particles is not None
        for _ in range(n):
            p, U = self.p, self.U
            Nu, NUu, *vu = self.nonlinear(p, U, track)
            a = E2 * p + Q * Nu
            Ua = U + 0.5 * h * NUu
            Na, NUa, *va = self.nonlinear(a, Ua, track)
            b = E2 * p + Q * Na
            Ub = U + 0.5 * h * NUa
            Nb, NUb, *vb = self.nonlinear(b, Ub, track)
            cc = E2 * a + Q * (2 * Nb - Nu)
            Uc = Ua + 0.5 * h * (2 * NUb - NUu)
            Nc, NUc, *vc = self.nonlinear(cc, Uc, track)
            self.p = E * p + f1 * Nu + 2 * f2 * (Na + Nb) + f3 * Nc
            self.U = U + h / 6 * (NUu + 2 * (NUa + NUb) + NUc)
            if track:
                xy = self.particles["xy"]
                k1 = self._particle_velocity(vu[0], U, xy)
                k2 = self._particle_velocity(va[0], Ua, xy + 0.5 * h * k1)
                k3 = self._particle_velocity(vb[0], Ub, xy + 0.5 * h * k2)
                k4 = self._particle_velocity(vc[0], Uc, xy + h * k3)
                self.particles["xy"] = xy + h / 6 * (k1 + 2 * (k2 + k3)
                                                     + k4)
            self.t += h

    # -- fields ------------------------------------------------------------
    @property
    def psi(self):
        """Streamfunction coefficients of the current state."""
        return self.invert(self.p)

    def grid_fields(self):
        """Grid fields of the current state.

        Returns
        -------
        dict
            ``psi``, ``zeta`` (relative vorticity), ``q`` (PV anomaly
            including the orography in the bottom layer), ``u``, ``v``
            (eddy velocities) for each layer, and ``eta``.
        """
        g = self.grid
        psi = self.psi
        q = self.p.copy()
        q[-1] += self.eta_h
        return dict(psi=g.ifft(psi), zeta=g.ifft(-g.k2 * psi), q=g.ifft(q),
                    u=g.ifft(-g.iky * psi), v=g.ifft(g.ikx * psi),
                    eta=self.eta)

    def max_speed(self):
        """Largest total wind speed on the grid."""
        f = self.grid_fields()
        uj = f["u"] + self.layer_winds()[:, None, None]
        return float(np.sqrt(uj ** 2 + f["v"] ** 2).max())

    def cfl(self):
        """Advective CFL number ``max|u| dt / dx``."""
        return self.max_speed() * self.dt / min(self.grid.dx, self.grid.dy)

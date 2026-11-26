"""Scales, model parameters and orographies.

The model is nondimensional. Lengths are scaled by ``L_STAR``, the
radius of the 45 degree latitude circle (so the zonally periodic domain
of length ``2 pi`` is one latitude circle, about 28 300 km), velocities
by ``U_STAR = 10 m/s`` and times by ``T_STAR = L_STAR / U_STAR`` (5.2
days). The planetary vorticity gradient at 45 degrees then becomes
``beta = 32.9``, and an orographic potential vorticity ``eta = 1``
corresponds to a mountain ``H_LAYER / (f0 T_STAR)`` high, about 108 m
for the 5 km lower layer (``height_of_eta``). The default bottom drag
(1.3) is a 4 day spin-down, the default cooling (0.26) 20 days.
"""

from dataclasses import dataclass, field, replace

import numpy as np

# Earth and the scaling latitude.
OMEGA = 7.292115e-5          # rotation rate (s-1)
A_EARTH = 6.371e6            # radius (m)
LAT0 = np.deg2rad(45.0)      # reference latitude
F0 = 2 * OMEGA * np.sin(LAT0)
BETA_DIM = 2 * OMEGA * np.cos(LAT0) / A_EARTH

L_STAR = A_EARTH * np.cos(LAT0)   # m
U_STAR = 10.0                     # m s-1
T_STAR = L_STAR / U_STAR          # s
DAY = 86400.0
H_LAYER = 5000.0                  # depth of each layer (m)

BETA = BETA_DIM * L_STAR ** 2 / U_STAR


def height_of_eta(eta, depth=H_LAYER):
    """Mountain height (m) of nondimensional orographic PV ``eta``.

    In QG the orography enters as ``eta = f0 h / H`` in the layer of
    depth ``H`` that touches the ground.
    """
    return np.asarray(eta) / T_STAR * depth / F0


def eta_of_height(h, depth=H_LAYER):
    """Nondimensional orographic PV of a mountain ``h`` metres high."""
    return np.asarray(h) * F0 / depth * T_STAR


def days(t):
    """Nondimensional time in days."""
    return np.asarray(t) * T_STAR / DAY


def km(x):
    """Nondimensional length in kilometres."""
    return np.asarray(x) * L_STAR / 1e3


def ms(u):
    """Nondimensional velocity in m/s."""
    return np.asarray(u) * U_STAR


def psi_si(psi):
    """Nondimensional streamfunction in m^2/s."""
    return np.asarray(psi) * U_STAR * L_STAR


def pv_si(q):
    """Nondimensional potential vorticity (or vorticity) in 1/s."""
    return np.asarray(q) / T_STAR


def drag_tendency(D, nlayers=2):
    """Acceleration of the column-mean wind by the form stress ``D``.

    In the model ``dU/dt`` contains ``D / N``; returned in m/s per day.
    """
    return np.asarray(D) / nlayers * U_STAR / T_STAR * DAY


@dataclass(frozen=True)
class Orography:
    """An idealized orography (lower-layer PV ``eta``).

    Parameters
    ----------
    kind : str
        ``"ridge"``: a meridionally uniform Gaussian ridge, the
        classical infinite barrier; ``"mountain"``: an isotropic Gaussian
        mountain; ``"elliptic"``: a Gaussian mountain with different
        zonal and meridional widths; ``"none"``.
    eta0 : float
        Peak orographic PV.
    width : float
        Zonal e-folding half-width.
    width_y : float, optional
        Meridional half-width (``"elliptic"``); defaults to ``width``.
    x0, y0 : float
        Center.
    """

    kind: str = "mountain"
    eta0: float = 10.0
    width: float = 0.25
    width_y: float = None
    x0: float = np.pi / 2
    y0: float = np.pi / 2

    def field(self, grid):
        """Grid values of ``eta``, periodized by summing images."""
        if self.kind == "none" or self.eta0 == 0:
            return np.zeros((grid.ny, grid.nx))
        wy = self.width_y if self.width_y is not None else self.width
        gx = np.zeros(grid.nx)
        for m in range(-3, 4):
            gx += np.exp(-((grid.x - self.x0 + m * grid.lx)
                           / self.width) ** 2)
        if self.kind == "ridge":
            gy = np.ones(grid.ny)
        else:
            gy = np.zeros(grid.ny)
            for m in range(-3, 4):
                gy += np.exp(-((grid.y - self.y0 + m * grid.ly) / wy) ** 2)
        return self.eta0 * gy[:, None] * gx[None, :]


@dataclass(frozen=True)
class Physics:
    """Physical parameters of the one- or two-layer model.

    Parameters
    ----------
    nlayers : int
        1 (barotropic) or 2 (equal-depth layers, 1 on top).
    beta : float
        Planetary vorticity gradient.
    kd : float
        Deformation wavenumber ``1 / L_d`` of the two-layer model; the
        layer coupling is ``F = kd^2 / 2``.
    shear : float
        Thermal-wind shear ``Lambda = U_1 - U_2``, held fixed (the
        limit of strong radiative relaxation of the mean baroclinicity).
    U : float
        Initial (or prescribed) column-mean zonal wind.
    interactive : bool
        If true the column-mean wind obeys the momentum equation with
        relaxation and orographic form stress; otherwise it is fixed.
    kappa : float
        Relaxation rate of the column-mean wind toward ``U_star``.
    U_star : float
        Zonal wind the external forcing relaxes toward.
    drag : float
        Linear (Ekman) drag on the relative vorticity of the bottom
        layer.
    cooling : float
        Newtonian cooling rate of the interface displacement of the
        eddies (two-layer model); the mean shear is held at ``shear``.
    nu : float or None
        Hyperviscosity coefficient. The default damps total wavenumber
        20 (about 1400 km wavelength) at the rate 1 (an e-folding time
        of 5.2 days) independently of the grid, so that results converge
        with resolution; ``None`` instead ties it to the grid, damping
        the largest retained wavenumber at rate ``nu_rate``.
    nu_order : int
        Hyperviscosity is ``nu (-Laplacian)^nu_order``.
    nu_rate : float
        Grid-scale damping rate used when ``nu`` is None.
    orography : Orography
    """

    nlayers: int = 2
    beta: float = BETA
    kd: float = 5.5
    shear: float = 0.0
    U: float = 1.0
    interactive: bool = False
    kappa: float = 0.5
    U_star: float = 1.0
    drag: float = 1.3
    cooling: float = 0.26
    nu: float = 20.0 ** -8
    nu_order: int = 4
    nu_rate: float = 20.0
    orography: Orography = field(default_factory=Orography)

    @property
    def F(self):
        """Layer coupling ``kd^2 / 2`` (zero in the one-layer model)."""
        return 0.5 * self.kd ** 2 if self.nlayers == 2 else 0.0

    @property
    def layer_winds(self):
        """Zonal winds ``(U_1, U_2)`` or ``(U,)``."""
        if self.nlayers == 1:
            return (self.U,)
        return (self.U + 0.5 * self.shear, self.U - 0.5 * self.shear)

    @property
    def pv_gradients(self):
        """Mean PV gradients ``(Q_1y, Q_2y)`` or ``(beta,)``."""
        if self.nlayers == 1:
            return (self.beta,)
        return (self.beta + self.F * self.shear,
                self.beta - self.F * self.shear)

    @property
    def critical_shear(self):
        """Phillips criterion ``2 beta / kd^2`` for baroclinic instability."""
        return self.beta / self.F if self.nlayers == 2 else np.inf

    def with_(self, **kw):
        """Copy with fields replaced; ``orography__eta0=...`` style keys
        change fields of the orography."""
        oro = {k.split("__", 1)[1]: kw.pop(k) for k in list(kw)
               if k.startswith("orography__")}
        out = replace(self, **kw)
        if oro:
            out = replace(out, orography=replace(out.orography, **oro))
        return out

    @classmethod
    def with_surface_wind(cls, U2, shear, **kw):
        """Two-layer parameters with lower-layer wind ``U2``."""
        return cls(nlayers=2, shear=shear, U=U2 + 0.5 * shear, **kw)

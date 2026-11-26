"""Doubly periodic Fourier grid, transforms and dealiasing.

Fields live on a uniform ``ny`` by ``nx`` grid on ``[0, Lx) x [0, Ly)``.
Spectral coefficients are those of ``numpy.fft.rfft2``, of shape
``(ny, nx // 2 + 1)``. Products are dealiased by the two-thirds rule in
each direction, which keeps every wavenumber with ``|k_x| <= (nx - 1)
// 3`` and ``|k_y| <= (ny - 1) // 3`` (in units of the fundamental) and
makes the quadratic nonlinearity exact for the retained modes, so the
semi-discrete system is a Galerkin truncation.
"""

import numpy as np
from scipy.ndimage import map_coordinates


class Grid:
    """A doubly periodic pseudo-spectral grid.

    Parameters
    ----------
    nx, ny : int
        Number of grid points in ``x`` and ``y``.
    lx, ly : float
        Domain lengths.
    """

    def __init__(self, nx, ny, lx=2 * np.pi, ly=np.pi):
        self.nx, self.ny = int(nx), int(ny)
        self.lx, self.ly = float(lx), float(ly)
        self.dx, self.dy = self.lx / self.nx, self.ly / self.ny
        self.x = np.arange(self.nx) * self.dx
        self.y = np.arange(self.ny) * self.dy
        self.X, self.Y = np.meshgrid(self.x, self.y)
        mx = np.fft.rfftfreq(self.nx, 1.0 / self.nx)
        my = np.fft.fftfreq(self.ny, 1.0 / self.ny)
        self.mx, self.my = np.meshgrid(mx, my)
        self.kx = 2 * np.pi / self.lx * self.mx
        self.ky = 2 * np.pi / self.ly * self.my
        self.k2 = self.kx ** 2 + self.ky ** 2
        self.ikx = 1j * self.kx
        self.iky = 1j * self.ky
        self.keep = ((np.abs(self.mx) <= (self.nx - 1) // 3)
                     & (np.abs(self.my) <= (self.ny - 1) // 3))
        self.keep[0, 0] = False
        self.kmax = np.sqrt(self.k2[self.keep].max())
        # Parseval weights: coefficient pairs (k, -k) of the real field
        # are stored once in the rfft half plane, except kx = 0 and the
        # Nyquist column.
        w = np.full(self.k2.shape, 2.0)
        w[:, 0] = 1.0
        if self.nx % 2 == 0:
            w[:, -1] = 1.0
        self.weight = w / (self.nx * self.ny) ** 2

    # -- transforms --------------------------------------------------------
    def fft(self, f):
        """Spectral coefficients of a real field (last two axes)."""
        return np.fft.rfft2(f)

    def ifft(self, fh):
        """Real field from spectral coefficients (last two axes)."""
        return np.fft.irfft2(fh, s=(self.ny, self.nx))

    def truncate(self, fh):
        """Zero the modes removed by the two-thirds rule and the mean."""
        return fh * self.keep

    # -- inner products ----------------------------------------------------
    def mean_product(self, ah, bh):
        """Domain mean of the product of two real fields, spectrally."""
        return np.sum(self.weight * np.real(np.conj(ah) * bh), axis=(-2, -1))

    def resample(self, fh, other):
        """Coefficients ``fh`` of this grid expressed on grid ``other``.

        Modes are copied by integer wavenumber (zero padding or
        truncation); both grids must have the same domain.
        """
        out = np.zeros(fh.shape[:-2] + other.k2.shape, complex)
        mx = min(self.nx // 2, other.nx // 2)
        my = min(self.ny // 2, other.ny // 2)
        rows = np.r_[0:my, -my + 1:0]
        fac = (other.nx * other.ny) / (self.nx * self.ny)
        out[..., rows % other.ny, :mx] = fh[..., rows % self.ny, :mx] * fac
        return out

    # -- interpolation -----------------------------------------------------
    def interp(self, f, xp, yp, order=3):
        """Periodic spline interpolation of grid field ``f`` at points.

        Parameters
        ----------
        f : ndarray (ny, nx)
            Grid values.
        xp, yp : ndarray
            Point coordinates (any shape); they may lie outside the
            domain and are wrapped periodically.
        order : int
            Spline order (3 for cubic).
        """
        ci = np.asarray(yp) / self.dy
        cj = np.asarray(xp) / self.dx
        return map_coordinates(f, [ci, cj], order=order, mode="grid-wrap",
                               prefilter=True)

    def interp_spectral(self, fh, xp, yp):
        """Exact trigonometric interpolation of coefficients ``fh``.

        Costs one complex exponential per mode and point; meant for
        verification and small numbers of points.
        """
        xp = np.asarray(xp, float).ravel()[:, None]
        yp = np.asarray(yp, float).ravel()[:, None]
        sel = np.abs(fh) > 0
        c = (fh * self.weight * self.nx * self.ny)[sel]
        ph = np.exp(1j * (self.kx[sel] * xp + self.ky[sel] * yp))
        return np.real(ph @ c)

"""Parcel trajectories and finite-time Lyapunov exponents.

Particles are advected by the total velocity of a layer, ``(U_j - d
psi_j/dy, d psi_j/dx)``, interpolated from the grid with periodic cubic
splines. Positions are not wrapped, so displacements are continuous.
In steady flow the trajectories are streamlines; in time-dependent flow
they are integrated together with the model (``QGModel.add_particles``)
using the ETDRK4 stage values. The finite-time Lyapunov exponent over a
window ``T`` is ``log(sigma_max) / T``, with ``sigma_max`` the largest
singular value of the flow-map gradient, estimated by centered
differences on the seed lattice.
"""

import numpy as np


def steady_velocity(model, layer=-1, p=None, U=None):
    """Grid total velocity ``(u, v)`` of one layer of a (steady) state."""
    g = model.grid
    p = model.p if p is None else p
    psi = model.invert(p)[layer]
    Uj = model.layer_winds(model.U if U is None else U)[layer]
    return g.ifft(-g.iky * psi) + Uj, g.ifft(g.ikx * psi)


def advect_frozen(grid, u, v, x0, y0, t_end, dt, n_out=None):
    """RK4 trajectories in a frozen velocity field.

    Parameters
    ----------
    grid : Grid
    u, v : ndarray (ny, nx)
        Total velocity on the grid.
    x0, y0 : array_like
        Initial positions.
    t_end, dt : float
        Duration and step (``t_end`` may be negative to go backward).
    n_out : int, optional
        Number of stored positions (default every step).

    Returns
    -------
    t : ndarray
    xy : ndarray, shape (n_out, 2, n_particles)
    """
    xy = np.array([np.ravel(x0), np.ravel(y0)], float)
    n = int(round(abs(t_end) / dt))
    h = np.sign(t_end) * dt
    every = max(1, n // (n_out or n))

    def vel(z):
        return np.array([grid.interp(u, z[0], z[1]),
                         grid.interp(v, z[0], z[1])])

    out, tt = [xy.copy()], [0.0]
    for k in range(1, n + 1):
        k1 = vel(xy)
        k2 = vel(xy + 0.5 * h * k1)
        k3 = vel(xy + 0.5 * h * k2)
        k4 = vel(xy + h * k3)
        xy = xy + h / 6 * (k1 + 2 * (k2 + k3) + k4)
        if k % every == 0:
            out.append(xy.copy())
            tt.append(k * h)
    return np.array(tt), np.array(out)


def seed_lattice(grid, nx, ny, x_range=None, y_range=None):
    """A regular lattice of seed points (returns X, Y of shape (ny, nx))."""
    xr = x_range or (0.0, grid.lx)
    yr = y_range or (0.0, grid.ly)
    xs = np.linspace(xr[0], xr[1], nx, endpoint=x_range is not None)
    ys = np.linspace(yr[0], yr[1], ny, endpoint=y_range is not None)
    return np.meshgrid(xs, ys)


def ftle(X0, Y0, X1, Y1, T):
    """FTLE field from initial and final positions on a seed lattice.

    The lattice must be uniform; derivatives are centered in the
    interior and one-sided at the edges.
    """
    dx = X0[0, 1] - X0[0, 0]
    dy = Y0[1, 0] - Y0[0, 0]
    a = np.gradient(X1, dy, dx)
    b = np.gradient(Y1, dy, dx)
    # flow-map gradient [[dX/dx, dX/dy], [dY/dx, dY/dy]]
    J = np.stack([np.stack([a[1], a[0]], -1), np.stack([b[1], b[0]], -1)],
                 -2)
    C = np.einsum("...ki,...kj->...ij", J, J)
    tr = C[..., 0, 0] + C[..., 1, 1]
    det = C[..., 0, 0] * C[..., 1, 1] - C[..., 0, 1] * C[..., 1, 0]
    lmax = 0.5 * (tr + np.sqrt(np.maximum(tr * tr - 4 * det, 0.0)))
    return np.log(np.sqrt(np.maximum(lmax, 1e-300))) / abs(T)


def ftle_from_model(model, n_seed=(240, 120), T=2.0, layer=-1,
                    x_range=None, y_range=None):
    """Forward FTLE over ``[t, t + T]`` with seeds carried by the model.

    The model is advanced in place by ``T`` with a fixed step, so its
    time step should already satisfy the CFL limit over the window.
    """
    g = model.grid
    X0, Y0 = seed_lattice(g, n_seed[0], n_seed[1], x_range, y_range)
    model.add_particles(X0, Y0, layer=layer)
    nsteps = int(round(T / model.dt))
    model.step(nsteps)
    xy = model.particles["xy"]
    model.particles = None
    X1 = xy[0].reshape(X0.shape)
    Y1 = xy[1].reshape(X0.shape)
    return X0, Y0, ftle(X0, Y0, X1, Y1, nsteps * model.dt)

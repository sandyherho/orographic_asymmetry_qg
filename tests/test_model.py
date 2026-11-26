"""Unit tests of the grid, the model, the theory and the diagnostics."""

import os

import numpy as np
import pytest

from orographic_asymmetry_qg import diagnostics as dg
from orographic_asymmetry_qg import theory as th
from orographic_asymmetry_qg.lagrangian import advect_frozen, ftle
from orographic_asymmetry_qg.model import QGModel, etdrk4_coefficients
from orographic_asymmetry_qg.params import (BETA, Orography, Physics,
                                            eta_of_height, height_of_eta)
from orographic_asymmetry_qg.spectral import Grid
from orographic_asymmetry_qg.steady import SteadyProblem, solve_steady

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
FLAT = Orography(kind="none")


def random_state(m, amp=0.3, seed=1):
    """A smooth random PV state of the model."""
    m.perturb(amp, seed=seed, kmin=1.0, kmax=8.0)
    return m.p.copy()


# -- grid ------------------------------------------------------------------
def test_parseval_and_mean_product():
    g = Grid(48, 24)
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((2, g.ny, g.nx))
    ah, bh = g.fft(a), g.fft(b)
    w = g.weight.copy()
    full = np.sum(w * np.real(np.conj(ah) * bh))
    assert full == pytest.approx(np.mean(a * b), rel=1e-12)


def test_dealiasing_is_exact_for_retained_modes():
    g = Grid(48, 24)
    rng = np.random.default_rng(3)
    ah = g.truncate(g.fft(rng.standard_normal((g.ny, g.nx))))
    bh = g.truncate(g.fft(rng.standard_normal((g.ny, g.nx))))
    prod = g.truncate(g.fft(g.ifft(ah) * g.ifft(bh)))
    big = Grid(3 * 48, 3 * 24)

    def pad(fh):
        out = np.zeros(big.k2.shape, complex)
        rows = np.r_[0:g.ny // 2, g.ny // 2 + 1:g.ny]
        brow = np.where(rows < g.ny // 2, rows, rows + 2 * g.ny)
        out[np.ix_(brow, np.arange(g.nx // 2))] = \
            fh[np.ix_(rows, np.arange(g.nx // 2))] * 9
        return out
    exact = big.fft(big.ifft(pad(ah)) * big.ifft(pad(bh)))
    rows = np.r_[0:g.ny // 2, g.ny // 2 + 1:g.ny]
    brow = np.where(rows < g.ny // 2, rows, rows + 2 * g.ny)
    ex = np.zeros_like(prod)
    ex[np.ix_(rows, np.arange(g.nx // 2))] = \
        exact[np.ix_(brow, np.arange(g.nx // 2))] / 9
    ex = g.truncate(ex)
    assert np.max(np.abs(prod - ex)) < 1e-10 * np.max(np.abs(ex))


def test_spline_interpolation_matches_spectral():
    g = Grid(128, 64)
    f = np.sin(3 * g.X) * np.cos(4 * g.Y) + 0.3 * np.cos(g.X + 2 * g.Y)
    fh = g.fft(f)
    rng = np.random.default_rng(1)
    xp, yp = rng.uniform(-1, 8, 50), rng.uniform(-1, 4, 50)
    exact = np.sin(3 * xp) * np.cos(4 * yp) + 0.3 * np.cos(xp + 2 * yp)
    assert np.max(np.abs(g.interp_spectral(fh, xp, yp) - exact)) < 1e-12
    assert np.max(np.abs(g.interp(f, xp, yp) - exact)) < 2e-4


# -- model -----------------------------------------------------------------
@pytest.mark.parametrize("nl", [1, 2])
def test_inversion_roundtrip(nl):
    m = QGModel(Physics(nlayers=nl, orography=FLAT), (32, 16))
    p = random_state(m)
    assert np.allclose(m.pv_of_psi(m.invert(p)), p, atol=1e-12)


def test_etdrk4_reduces_to_rk4():
    c = etdrk4_coefficients(np.zeros(3), 0.1)
    assert np.allclose(c["Q"], 0.05) and np.allclose(c["f1"], 0.1 / 6)
    assert np.allclose(c["f2"], 0.1 / 6) and np.allclose(c["f3"], 0.1 / 6)


def test_etdrk4_coefficients_stiff_limit():
    L = np.array([-1e-8, -100.0, -1e4])
    h = 0.01
    c = etdrk4_coefficients(L, h)
    z = h * L
    with np.errstate(over="ignore"):
        f1 = h * (-4 - z + np.exp(z) * (4 - 3 * z + z * z)) / z ** 3
    assert c["f1"][1:] == pytest.approx(f1[1:], rel=1e-9)


@pytest.mark.parametrize("k, ky", [(4, 2), (3, 0), (5, 4)])
def test_two_layer_growth_rate(k, ky):
    ph = Physics(shear=3.0, U=0.5, nu=0.0, orography=FLAT)
    m = QGModel(ph, (32, 16), dt=0.002)
    g = m.grid
    my = ky // 2
    psi = np.zeros((2,) + g.k2.shape, complex)
    psi[0, my, k], psi[1, my, k] = 1e-9, 3e-10
    m.set_state(p=m.pv_of_psi(psi))
    m.step(1500)
    a1 = np.abs(m.psi[:, my, k]).sum()
    m.step(500)
    a2 = np.abs(m.psi[:, my, k]).sum()
    s = th.phillips_growth_rate(k, ky, 2.0, -1.0, ph.beta, ph.F, ph.drag,
                                ph.cooling)
    if s > 0.5:
        assert np.log(a2 / a1) / 1.0 == pytest.approx(s, rel=1e-4)


def test_rossby_wave_phase_speed():
    ph = Physics(nlayers=1, U=0.3, drag=0.0, nu=0.0, orography=FLAT)
    m = QGModel(ph, (32, 16), dt=0.005)
    psi = np.zeros((1,) + m.grid.k2.shape, complex)
    psi[0, 1, 3] = 1e-3
    m.set_state(p=m.pv_of_psi(psi))
    m.step(400)
    w = th.rossby_frequency(3.0, 2.0, 0.3, ph.beta)
    expected = 1e-3 * np.exp(-1j * w * m.t)
    assert m.psi[0, 1, 3] == pytest.approx(expected, rel=1e-6)


def test_ridge_solution_is_exact_steady_state():
    ph = Physics(nlayers=1, U=0.5, drag=0.65,
                 orography=Orography(kind="ridge", eta0=20.0))
    m = QGModel(ph, (64, 32))
    psi_x = th.ridge_steady_streamfunction(m.grid.x, 0.5, ph.beta,
                                           m.eta[0], ph.drag, m.nu,
                                           ph.nu_order)
    psi = m.grid.fft(np.tile(psi_x, (m.grid.ny, 1)))[None] * m.grid.keep
    dp, _ = m.tendency(m.pv_of_psi(psi), 0.5)
    assert np.max(np.abs(dp)) < 1e-9 * np.max(np.abs(m.eta_h))


@pytest.mark.parametrize("nl", [1, 2])
def test_linear_response_is_small_amplitude_limit(nl):
    errs = []
    for eta0 in (0.01, 0.02):
        ph = Physics(nlayers=nl, shear=1.0 if nl == 2 else 0.0, U=0.4,
                     orography=Orography(eta0=eta0))
        m = QGModel(ph, (64, 32))
        psi = th.linear_response(ph, m.grid, nu=m.nu)
        dp, _ = m.tendency(m.pv_of_psi(psi), m.U)
        errs.append(np.max(np.abs(dp)) / np.max(np.abs(m.pv_of_psi(psi))))
    # the residual is the neglected Jacobian, quadratic in eta0
    assert errs[1] / errs[0] == pytest.approx(2.0, rel=0.05)


@pytest.mark.parametrize("nl", [1, 2])
def test_energy_budget_closes(nl):
    ph = Physics(nlayers=nl, shear=1.5 if nl == 2 else 0.0, U=0.3,
                 orography=Orography(eta0=10.0))
    m = QGModel(ph, (64, 32))
    random_state(m)
    b = dg.energy_budget(m)
    scale = max(abs(b[k]) for k in ("baroclinic", "orographic", "drag"))
    assert abs(b["residual"]) < 1e-10 * scale


def test_inviscid_invariants_of_charney_devore_system():
    ph = Physics(nlayers=1, U=0.4, interactive=True, kappa=0.0, drag=0.0,
                 nu=0.0, orography=Orography(eta0=10.0))
    m = QGModel(ph, (32, 16), dt=0.002)
    random_state(m, amp=0.05)

    def inv():
        e = dg.energy(m)["total"] + 0.5 * m.U ** 2
        z = dg.enstrophy(m) + ph.beta * m.U
        return e, z
    e0, z0 = inv()
    m.step(500)
    e1, z1 = inv()
    assert e1 == pytest.approx(e0, rel=1e-8)
    assert z1 == pytest.approx(z0, rel=1e-8)


def test_form_stress_momentum_budget():
    ph = Physics(nlayers=2, shear=1.0, U=0.3, interactive=True,
                 U_star=0.6, orography=Orography(eta0=10.0))
    m = QGModel(ph, (32, 16), dt=0.002)
    random_state(m, amp=0.05)
    _, dU = m.tendency(m.p, m.U)
    expect = -ph.kappa * (m.U - 0.6) + m.form_stress(m.psi) / 2
    assert dU == pytest.approx(expect, rel=1e-12)


def test_temporal_order_is_four():
    ph = Physics(shear=1.5, U=-0.3, orography=Orography(eta0=10.0))
    errs = []
    ref = None
    for dt in (0.00125, 0.01, 0.005, 0.0025):
        m = QGModel(ph, (32, 16), dt=dt)
        random_state(m, amp=0.2)
        m.step(int(round(0.2 / dt)))
        if ref is None:
            ref = m.p.copy()
            continue
        errs.append(np.max(np.abs(m.p - ref)))
    order = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(order > 3.7)


# -- steady states --------------------------------------------------------
def test_pack_unpack_roundtrip():
    m = QGModel(Physics(interactive=True, orography=FLAT), (32, 16))
    p = random_state(m)
    prob = SteadyProblem(m)
    q, U = prob.unpack(prob.pack(p, 0.7))
    assert np.allclose(q, p) and U == pytest.approx(0.7)


def test_jacobian_matches_directional_derivative():
    m = QGModel(Physics(shear=1.0, interactive=True,
                        orography=Orography(eta0=10.0)), (24, 12))
    prob = SteadyProblem(m)
    x = prob.pack(random_state(m, 0.1), 0.4)
    v = np.random.default_rng(2).standard_normal(prob.n)
    J = prob.jacobian(x)
    h = 1e-6
    fd = (prob.residual(x + h * v) - prob.residual(x - h * v)) / (2 * h)
    assert np.max(np.abs(J @ v - fd)) < 1e-6 * np.max(np.abs(fd))


def test_newton_converges_quadratically():
    ph = Physics(shear=1.0, U=0.0, interactive=True, U_star=0.0,
                 orography=Orography(eta0=10.0))
    prob, x, info = solve_steady(ph, (24, 12))
    assert info["converged"] and info["residual"] < 1e-10


# -- theory ----------------------------------------------------------------
def test_barotropic_limit_of_stationary_waves():
    F = 15.0
    ks = th.stationary_wavenumbers(0.5, 0.5, BETA, F)
    assert np.sqrt(BETA / 0.5) == pytest.approx(ks[-1], rel=1e-12)
    assert np.sqrt(BETA / 0.5 - 2 * F) == pytest.approx(ks[0], rel=1e-12)
    assert th.stationary_wavenumbers(-0.5, -0.5, BETA, F).size == 0


def test_stationary_roots_annihilate_the_determinant():
    F = 15.0
    for U1, U2 in ((0.8, -0.4), (1.2, 0.3), (0.3, -1.5)):
        lam = U1 - U2
        Q1, Q2 = BETA + F * lam, BETA - F * lam
        for K in th.stationary_wavenumbers(U1, U2, BETA, F):
            s = K * K + F
            det = (Q1 - U1 * s) * (Q2 - U2 * s) - U1 * U2 * F * F
            assert abs(det) < 1e-9 * (Q1 * Q2)


def test_resonance_theorem_agrees_with_root_count():
    F = 15.0
    rng = np.random.default_rng(5)
    for U1, U2 in rng.uniform(-2, 2, (400, 2)):
        pred = th.resonance_theorem(U1, U2, BETA, F)
        if pred >= 0:
            assert pred == th.stationary_wavenumbers(U1, U2, BETA, F).size


def test_easterly_column_has_no_resonance_but_shear_creates_one():
    F = 15.0
    assert th.resonance_theorem(-0.2, -0.8, BETA, F) == 0
    assert th.resonance_theorem(0.2, -0.8, BETA, F) == 1


def test_parcel_trajectory_linear_limit():
    g = Grid(2048, 4)
    eta0 = 0.05
    ridge = Orography(kind="ridge", eta0=eta0, width=0.25, x0=np.pi)
    eta_x = ridge.field(g)[0]

    def eta_fun(x):
        return eta0 * sum(np.exp(-((x - np.pi + m * g.lx) / 0.25) ** 2)
                          for m in range(-3, 4))
    k = 2 * np.pi * np.fft.rfftfreq(g.nx, g.dx)
    cases = ((0.45, (0.0, g.lx)), (-0.5, (np.pi - 0.5, np.pi + 0.5)))
    for U, span in cases:
        y = th.ridge_steady_streamfunction(g.x, U, BETA, eta_x) / U
        dy = np.fft.irfft(1j * k * np.fft.rfft(y), g.nx)
        xs0 = span[0] if U > 0 else span[1]
        xs, ys = th.parcel_trajectory(eta_fun, U, BETA, span,
                                      y0=np.interp(xs0, g.x, y),
                                      slope0=np.interp(xs0, g.x, dy),
                                      pv0=eta_x.mean())
        yq = np.interp(xs, g.x, y, period=g.lx)
        assert np.max(np.abs(ys - yq)) < 0.02 * np.max(np.abs(y))


def test_height_conversion_roundtrip():
    assert eta_of_height(height_of_eta(20.0)) == pytest.approx(20.0)
    assert 2000 < height_of_eta(20.0) < 2300


# -- diagnostics ------------------------------------------------------------
def test_entropies_of_simple_cases():
    assert dg.spectral_entropy([0, 1, 0, 0]) == 0.0
    assert dg.spectral_entropy(np.ones(10)) == pytest.approx(1.0)
    assert dg.permutation_entropy(np.arange(100.0)) == pytest.approx(0.0)
    x = np.random.default_rng(0).standard_normal(200000)
    assert dg.permutation_entropy(x) == pytest.approx(1.0, abs=2e-3)
    c, h = dg.statistical_complexity(x)
    assert c < 0.01 and h > 0.99


def test_fisher_information_of_gaussian_shift():
    rng = np.random.default_rng(1)
    lam = np.linspace(0, 1, 11)
    samples = [rng.standard_normal(40000) * 0.5 + m for m in lam]
    _, info = dg.fisher_information(lam, samples, grid_pts=400)
    assert np.median(info) == pytest.approx(1 / 0.25, rel=0.1)


def test_ftle_of_linear_strain():
    g = Grid(64, 64, 2 * np.pi, 2 * np.pi)
    a = 0.4
    X0, Y0 = np.meshgrid(np.linspace(2, 3, 9), np.linspace(2, 3, 9))
    T = 1.5
    X1 = np.pi + (X0 - np.pi) * np.exp(a * T)
    Y1 = np.pi + (Y0 - np.pi) * np.exp(-a * T)
    assert np.allclose(ftle(X0, Y0, X1, Y1, T), a)
    u = np.sin(g.X) * np.cos(g.Y)
    v = -np.cos(g.X) * np.sin(g.Y)
    _, xy = advect_frozen(g, u, v, [1.0], [1.0], 2.0, 0.01)
    # the cellular flow conserves its streamfunction along trajectories
    s0 = np.sin(1.0) * np.sin(1.0)
    s1 = np.sin(xy[-1, 0, 0]) * np.sin(xy[-1, 1, 0])
    assert s1 == pytest.approx(s0, rel=1e-5)


def test_pep8():
    pycodestyle = pytest.importorskip("pycodestyle")
    style = pycodestyle.StyleGuide(max_line_length=79)
    files = []
    for d in ("orographic_asymmetry_qg", "scripts", "tests"):
        for fn in sorted(os.listdir(os.path.join(ROOT, d))):
            if fn.endswith(".py"):
                files.append(os.path.join(ROOT, d, fn))
    assert style.check_files(files).total_errors == 0

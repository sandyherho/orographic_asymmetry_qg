"""orographic_asymmetry_qg: westerly-easterly orographic asymmetry with shear.

A one- and two-layer quasi-geostrophic beta-plane model of flow over
isolated orography, with an interactive mean wind, written to revisit
the classical result that westerly flow over a mountain makes a stationary
Rossby wave train while easterly flow does not. It provides a
dealiased pseudo-spectral model with ETDRK4 time stepping, closed-form
and linear theory (stationary resonance, linear form stress, normal
modes, the parcel argument), steady states by Newton's method with
pseudo-arclength continuation and exact stability spectra, Lagrangian
diagnostics, and information-theoretic measures of the flow regimes.

Modules
-------
params
    Scales, physical parameters and orographies.
spectral
    Doubly periodic Fourier grid, dealiasing and interpolation.
model
    The QG model and its ETDRK4 integrator.
theory
    Stationary resonance, linear response, normal modes, the infinite ridge.
steady
    Newton, continuation, stability.
diagnostics
    Energy budget, spectra, entropy, complexity, Fisher information.
lagrangian
    Parcel trajectories and finite-time Lyapunov exponents.
experiments
    Reference configuration and integration driver.
scene, plotting, io
    Animation helpers, figure style, output files.
"""

__version__ = "0.0.1"

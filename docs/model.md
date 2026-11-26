# Model formulation, closed-form results and numerical method

This document derives every equation solved in the `orographic_asymmetry_qg` library, states the closed-form results used to check and interpret the numerics, and describes the discretization. Symbols follow the code. Unless stated otherwise all quantities are nondimensional.

## 1. Scales

Lengths are scaled by $L_* = a\cos 45^\circ = 4505$ km, the radius of the 45° latitude circle. Velocities are scaled by $U_* = 10$ m/s and times by $T_* = L_*/U_* = 5.21$ days. The zonally periodic domain of length $2\pi$ is then one latitude circle (28 300 km), and its meridional width $\pi$ is 14 150 km. The planetary vorticity gradient at 45° becomes

$$\beta = \frac{2\Omega\cos 45^\circ}{a}\,\frac{L_*^2}{U_*} = 32.9 .$$

An orographic PV $\eta$ (Section 2) corresponds to a mountain of height $h = \eta H / (f_0 T_*)$. That is 108 m per unit $\eta$ for the $H = 5$ km lower layer, and 215 m per unit for a 10 km barotropic column.

## 2. Governing equations

### 2.1 Two layers

Two layers of equal depth $H$ lie on a $\beta$-plane under a rigid lid; layer 1 is on top. The orography of height $h_b$ enters the lower layer as the PV $\eta = f_0 h_b / H$. Each layer has a uniform zonal wind $U_j$ and a perturbation streamfunction $\psi_j$. With the layer coupling $F = k_d^2/2$, where $k_d = 1/L_d$ is the deformation wavenumber, the perturbation PV is

$$q_1 = \nabla^2\psi_1 + F(\psi_2 - \psi_1), \qquad q_2 = \nabla^2\psi_2 + F(\psi_1 - \psi_2) + \eta .$$

The mean flow has the thermal-wind shear $\Lambda = U_1 - U_2$ and the column-mean wind $U = (U_1 + U_2)/2$. The mean PV gradients are

$$Q_{1y} = \beta + F\Lambda, \qquad Q_{2y} = \beta - F\Lambda .$$

The equations of motion are

$$\partial_t q_j + J(\psi_j, q_j) + U_j\,\partial_x q_j + Q_{jy}\,\partial_x\psi_j = \mathcal{D}_j ,$$

with damping

$$\mathcal{D}_1 = +\alpha F(\psi_1 - \psi_2) - \nu(-\nabla^2)^4 q_1, \qquad \mathcal{D}_2 = -\alpha F(\psi_1 - \psi_2) - r\nabla^2\psi_2 - \nu(-\nabla^2)^4 q_2 .$$

The damping terms are:

- **Bottom drag:** linear Ekman drag $r$ on the lower-layer vorticity. The default $r = 1.3$ is a 4 day spin-down.
- **Newtonian cooling:** rate $\alpha$ on the interface displacement of the perturbations, which is proportional to $\psi_1 - \psi_2$. The default $\alpha = 0.26$ is 20 days.
- **Hyperviscosity:** $\nu = 20^{-8}$. It damps total wavenumber 20 (wavelength 1400 km) at unit rate, independently of the grid, so solutions converge with resolution.

The mean shear $\Lambda$ is held fixed. This is the limit of strong radiative relaxation of the zonal-mean baroclinicity, which is standard in doubly periodic two-layer models. The cooling of the perturbations is the same relaxation acting at a finite rate on the eddies.

### 2.2 One layer

The barotropic model has $q = \nabla^2\psi + \eta$, $Q_y = \beta$, drag on $\nabla^2\psi$, and no cooling.

### 2.3 Interactive mean wind and form stress

When the mean wind is interactive, the column-mean wind obeys

$$\frac{dU}{dt} = -\kappa\,(U - U_*) + \frac{1}{N}\,\langle \eta\,\partial_x\psi_b\rangle ,$$

where:

- $N$ is the number of layers and $b$ is the bottom layer;
- angle brackets denote a domain mean;
- $\kappa$ is the rate at which an external forcing relaxes $U$ toward $U_*$; the default $\kappa = 0.5$ is 10 days.

The last term is the orographic form stress, $D = \langle\eta\,\partial_x\psi_b\rangle = -\langle\psi_b\,\partial_x\eta\rangle$, which is a pressure force on the mountain. It follows from the domain-mean zonal momentum budget: in QG, $\partial_t\bar u_j = \overline{v_j q_j}$, and in a doubly periodic domain $\langle v\nabla^2\psi\rangle = 0$, while the interfacial form stresses $\pm F\langle v_j\psi_{3-j}\rangle$ cancel between the layers. The shear is held, so only the column mean responds. The lower-layer wind is $U_2 = U - \Lambda/2$, and the surface forcing wind is written $U_2^* = U_* - \Lambda/2$. This is the Charney–DeVore (1979) mechanism for multiple flow equilibria.

### 2.4 Invariants and the energy budget

In the one-layer system with $\kappa = r = \nu = 0$, two quantities are conserved exactly:

$$E = \tfrac12\langle|\nabla\psi|^2\rangle + \tfrac12 U^2, \qquad Z = \tfrac12\langle(\nabla^2\psi + \eta)^2\rangle + \beta U .$$

Multiplying the PV equation by $-\psi$ gives $d\langle|\nabla\psi|^2\rangle/2dt = -U\langle\eta\,\partial_x\psi\rangle = -U\,dU/dt$. Multiplying it by $q$ leaves only $-\beta\langle\eta\,\partial_x\psi\rangle = -\beta\,dU/dt$.

The two-layer eddy energy is

$$E = \tfrac12\sum_j\langle|\nabla\psi_j|^2\rangle + \tfrac{F}{2}\langle(\psi_1 - \psi_2)^2\rangle .$$

It obeys

$$\frac{dE}{dt} = \underbrace{\Lambda F\langle\psi_1\,\partial_x\psi_2\rangle}_{\text{baroclinic conversion}} \; \underbrace{- \;U_2\,D}_{\text{orographic conversion}} \; - \; r\langle|\nabla\psi_2|^2\rangle \; - \; \alpha F\langle(\psi_1-\psi_2)^2\rangle \; - \; \text{hyperviscous loss}.$$

The orographic term is positive when the form stress opposes the surface wind, which is the normal case. Energy then passes from the mean flow to the stationary wave.

## 3. Stationary resonance

### 3.1 The barotropic case and the classical asymmetry

A free barotropic Rossby wave $e^{i(kx + ly - \omega t)}$ has $\omega = Uk - \beta k/K^2$, where $K^2 = k^2 + l^2$. It is stationary ($\omega = 0$) when $K^2 = \beta/U$, which is possible only for westerlies. Orography therefore excites a resonant, stationary lee-wave train in westerlies and only a confined, evanescent response in easterlies. This is the classical westerly-easterly asymmetry (his Figs. 4.9 and 4.10).

### 3.2 The two-layer case

For stationary inviscid waves with total wavenumber $K$, write $s = K^2 + F$. The homogeneous steady equations have a nontrivial solution when

$$\big(Q_{1y} - U_1 s\big)\big(Q_{2y} - U_2 s\big) - U_1U_2F^2 = 0,$$

that is,

$$P(s) \equiv U_1U_2\,s^2 - (Q_{1y}U_2 + Q_{2y}U_1)\,s + Q_{1y}Q_{2y} - U_1U_2F^2 = 0 .$$

A stationary wave exists for each root with $s > F$. Each of these statements reduces to the classical result on the barotropic diagonal $U_1 = U_2$, where the roots are $K^2 = \beta/U$ and $K^2 = \beta/U - 2F$.

**Proposition (subcritical shear, $Q_{1y}, Q_{2y} > 0$).**

1. *Easterlies in both layers* ($U_1, U_2 < 0$): there is no stationary wave. The coefficients of $P$ are all positive, so $P$ has no positive root.
2. *Opposite signs* ($U_1U_2 < 0$): there is exactly one stationary wave if $U_1 + U_2 < \beta/F$, and none otherwise. In particular there is always one when the column mean is easterly.
   *Proof.* $P$ has a negative leading coefficient and a positive constant term, so it has exactly one positive root. Evaluating at $s = F$,
   $$P(F) = Q_{1y}Q_{2y} - F(Q_{1y}U_2 + Q_{2y}U_1) = \beta\,[\beta - F(U_1 + U_2)].$$
   The root exceeds $F$ if and only if $P(F) > 0$. ∎
3. *Westerlies in both layers*: zero, one or two stationary waves, depending on the discriminant.

So the classical asymmetry requires easterlies through the whole column. Surface easterlies beneath westerlies aloft, as in the trade-wind belt under the subtropical jet or the polar easterlies under the polar-front jet, admit a stationary orographic wave. Its wavenumber grows without bound as $U_1 \to 0^+$, so the resonance moves from short to long waves as the shear increases. `compute_resonance.py` checks the proposition against the numerical root count on a 401 × 401 grid of layer winds.

### 3.3 Linear forced response and form stress

Dropping the Jacobians, each Fourier mode $(k, l)$, $k \neq 0$, of the steady response satisfies a 2×2 complex system $\mathsf A\,\hat\psi = (ikU_2\hat\eta)\,\mathbf e_2$. The matrix is built from $\mathsf M = \begin{pmatrix}-K^2 - F & F\\ F & -K^2 - F\end{pmatrix}$ (so that $\hat q = \mathsf M\hat\psi$):

$$\mathsf A_{ij} = -(ikU_i + \nu K^8)\,\mathsf M_{ij} - ikQ_{iy}\,\delta_{ij} + rK^2\,\delta_{i2}\delta_{j2} + \alpha F\begin{pmatrix}1 & -1\\ -1 & 1\end{pmatrix}_{ij}.$$

The linear form stress is $D = \sum w_{\mathbf k}\,\mathrm{Re}(\hat\eta^*\,ik\,\hat\psi_2)$ with the Parseval weights of the grid. It is evaluated on the model grid with the model's damping, so it is the exact small-amplitude limit of the numerical model. In the barotropic case,

$$D = -\sum_{\mathbf k} w_{\mathbf k}|\hat\eta|^2\, \frac{k^2 U\, r K^2}{k^2(UK^2 - \beta)^2 + r^2K^4},$$

a sum of Lorentzians centered on the resonances $U = \beta/K^2$ that exist only for $U > 0$.

### 3.4 Linear equilibria and the fold threshold

Steady states of the interactive mean wind satisfy $U_* = U - D(U)/(N\kappa)$. The equilibrium curve folds, giving multiple equilibria, where $dD/dU > N\kappa$. Because the linear $D$ scales as $\eta_0^2$, the smallest mountain that gives folds on either side of $U_2 = 0$ is

$$\eta_c = \eta_0\,\sqrt{\frac{N\kappa}{\max_{\text{side}} dD/dU\,|_{\eta_0}}}.$$

### 3.5 The parcel argument for a ridge

For a meridionally uniform ridge, the barotropic streamfunction $\psi(x)$ makes every Jacobian vanish. The Fourier solution of

$$U(\psi''' + \eta') + \beta\psi' = -r\psi'' - \nu\,\partial_x^8\psi''$$

is therefore an exact steady solution of the nonlinear model. Without damping, the streamline $y = \psi/U$ obeys $U y'' + \beta y + \eta = \langle\eta\rangle$.

The parcel argument can be written as a nonlinear trajectory equation. A parcel moves at constant speed $|U|$ with heading $\alpha$, its relative vorticity is the curvature vorticity $d\alpha/dt$, and it conserves PV:

$$\dot x = |U|\cos\alpha, \qquad \dot y = |U|\sin\alpha, \qquad \dot\alpha = q_0 - \beta y - \eta(x).$$

Linearized in the path slope, this is the QG streamline equation for either sign of $U$. For easterlies the equation is evanescent, $y'' = (\beta/|U|)\,y + \dots$, so as an initial-value problem it amplifies any error by $e^{\sqrt{\beta/|U|}\,x}$. The easterly balance is a boundary-value statement.

### 3.6 Normal modes

Linearizing about uniform winds, normal modes $e^{i(kx + ly) + \sigma t}$ satisfy $\sigma\,\mathsf M\hat\psi = \mathsf A_0\hat\psi$. Here $\mathsf A_0$ is $\mathsf A$ without hyperviscosity. Without drag and cooling, these give the Phillips instability for $\Lambda > \beta/F = 2\beta/k_d^2$, which is 21.7 m/s for the reference $k_d = 5.5$ ($L_d = 820$ km).

## 4. Numerical method

### 4.1 Spatial discretization

The fields are expanded in Fourier series on the doubly periodic domain $[0, 2\pi)\times[0, \pi)$, using the `rfft2` half plane.

- **Products:** computed on the grid in flux form, $J(\psi, q) = \partial_x(uq) + \partial_y(vq)$.
- **Dealiasing:** the two-thirds rule in each direction keeps the modes with $|m_x| \le (n_x - 1)/3$ and $|m_y| \le (n_y - 1)/3$. The quadratic term is then exact for the retained modes. The semi-discrete system is a Galerkin truncation and conserves the invariants of Section 2.4.
- **Orography:** truncated in the same way.
- **Inversion:** 2×2 per wavenumber, in closed form. The determinant is $K^2(K^2 + 2F)$.

### 4.2 Time stepping

ETDRK4 (Cox and Matthews 2002) treats the hyperviscosity exactly and all other terms explicitly. The coefficients $\varphi$ are evaluated by contour integrals of 32 points around each $hL$ (Kassam and Trefethen 2005), which avoids cancellation at small $hL$. For the mean wind, where $L = 0$, the scheme reduces to classical RK4. The step is chosen from the ladder $\Delta t_{\max}/2^k$ so that the advective CFL number stays below 0.35, and coefficients are cached for each rung.

### 4.3 Steady states, continuation and stability

The unknowns are the independent real and imaginary parts of the retained coefficients of every layer, scaled to Fourier amplitudes, plus $U$ when it is interactive. Half of the $k_x = 0$ column is redundant by conjugate symmetry and is excluded. The tendency is a quadratic polynomial in the unknowns, so central differences give its Jacobian to rounding error. The Jacobian is formed densely, in batches of 192 columns evaluated as one stacked FFT.

- **Newton's method:** an LU solve with Armijo backtracking. A homotopy raises the mountain from a tenth of its height in ten steps.
- **Continuation:** pseudo-arclength (Keller) in $U_*$, with arclength measured in the $(U, U_*)$ plane and a weight of $10^{-2}/n$ on the PV unknowns. The corrector is a chord iteration with the bordered Jacobian of the last point. Folds are sign changes of the tangent's $U_*$ component.
- **Restarts:** near the sheared easterly resonances the corrector can fail, because many turning points lie close together. The continuation is then restarted with Newton's method a little further along, from scratch if needed. If that also fails, the branch is completed backward from the westerly end. The pieces are separated by gaps in the output.
- **Stability:** the full eigenvalue spectrum of the dense Jacobian, from the QR algorithm. Arnoldi iteration does not converge reliably, because many weakly damped Rossby modes crowd the imaginary axis.

### 4.4 Lagrangian diagnostics

Parcels are advected by the total layer velocity $(U_j - \partial_y\psi_j,\ \partial_x\psi_j)$, interpolated with periodic cubic splines. In time-dependent flow they are advanced with RK4 using the velocities of the four ETDRK4 stages, so parcel and flow share their stage values. The finite-time Lyapunov exponent is $\log\sigma_{\max}/T$, where $\sigma_{\max}$ is the largest singular value of the flow-map gradient. The gradient is estimated by centered differences on the seed lattice.

### 4.5 Information measures

- **Permutation entropy and statistical complexity:** the normalized permutation entropy $H$ (Bandt and Pompe 2002) of order 5 and delay 5 samples, and the Jensen–Shannon statistical complexity $C = Q_J H$ (Rosso et al. 2007), applied to the form stress series.
- **Spectral entropy of the stationary wave:** the normalized Shannon entropy of the kinetic energy of the time-mean lower-layer streamfunction, binned in unit shells of total wavenumber. It is low when the stationary wave is concentrated in a few scales, as at a resonance.
- **Fisher information:** the library also estimates the Fisher information $I(\lambda) = 4\int(\partial_\lambda\sqrt{p})^2\,dx$ of a one-parameter family of densities, with Gaussian kernels and centered differences. It is not used in the results: with forcing steps of 1 m/s and nearly steady runs, neighbouring distributions do not overlap and the estimate saturates.

### 4.6 Verification

The test suite (`tests/test_model.py`) and `compute_verification.py` check:

- Parseval's identity and exact dealiasing against a three-times padded product;
- spline interpolation against exact trigonometric interpolation;
- ETDRK4 coefficients against their RK4 and stiff limits, and fourth-order temporal convergence;
- normal-mode growth rates with drag and cooling against the dispersion relation, to $10^{-4}$;
- the phase of a free Rossby wave;
- the exact ridge solution (tendency residual below $10^{-9}$);
- the quadratic scaling of the linear-response residual;
- closure of the energy budget, the two inviscid invariants, and the momentum budget;
- the Jacobian against directional derivatives, and quadratic convergence of Newton's method;
- the resonance proposition against root counts;
- The parcel equation against the exact streamline;
- the entropy, complexity, Fisher-information and FTLE estimators on cases with known answers;
- grid convergence of a steady state on 48×24, 64×32 and 96×48 grids.

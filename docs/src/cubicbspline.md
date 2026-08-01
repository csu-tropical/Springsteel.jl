```@meta
CurrentModule = Springsteel
```

# CubicBSpline

The `CubicBSpline` submodule implements the Ooyama (2002) cubic B-spline spectral
transform method, which provides compact-support basis functions with a built-in sixth-order low-pass filter. Cubic B-splines are a good general choice for an underlying basis function due to their computational efficiency and flexible boundary
conditions. They are _semi-spectral_ since the basis functions are not orthogonal, and are a class of finite element bases. Another advantage of this basis is the ability to decompose the domain into _tiles_ for shared memory parallelization and/or multiple _patches_ for domain nesting or distributed memory parallelization.

> **Reference**: Ooyama, K. V., 2002: The cubic-spline transform method: Basic
> definitions and tests in a 1D single domain. *Mon. Wea. Rev.*, **130**, 2392–2415.

---

## Mathematical Overview

### Basis Functions

A domain $[x_0, x_0']$ is divided into $m$ equal cells of width $\Delta x$.
The B-spline basis function centred on node $x_n$ is

$$\varphi_n(x) = F\!\left(\frac{x - x_n}{\Delta x}\right), \qquad
F(\xi) = \begin{cases}
  \tfrac{1}{6}(1{-}|\xi|)^2\bigl[4 - (2{-}3|\xi|)(1{+}3|\xi|)\bigr] & 0 \le |\xi| \le 1 \\
  \tfrac{1}{6}(2{-}|\xi|)^3 & 1 \le |\xi| \le 2 \\
  0 & |\xi| \ge 2
\end{cases}$$

Each basis function spans four cells and has second-order continuity
($C^2$). The expansion of a field $u$ is

$$u(x) = \sum_{n \in M} a_n \,\varphi_n(x), \qquad M = \{-1, 0, 1, \ldots, m{+}1\}$$

giving $m + 3$ spectral coefficients $a_n$ for $m$ cells.

### Physical Sampling: The Mish

Field values are sampled at $\mu = 3$ **Gauss–Legendre quadrature points per cell**
(the "mish"), using the $\sqrt{3/5}$ abscissae with weights $w = [5/18, 8/18, 5/18]$.
The total number of physical points is `iDim = num_cells × 3`.

### Sizing a Spline Direction in 2-D and 3-D

A spline axis is defined by its **cell count**, not its gridpoint count. For `n` cells
and `mubar` quadrature points per cell:

| quantity | formula |
|:-- |:-- |
| physical gridpoints | `Dim = n * mubar` |
| spectral coefficients | `bDim = n + 3` |
| cell width | `DX = (max - min) / n` |

`bDim` depends on neither `mubar` nor the boundary-condition type. It follows that a
spline `iDim`/`jDim`/`kDim` must be an exact multiple of `mubar`; a value that is not
raises an `ArgumentError` naming the nearest valid cell counts.

Each spline direction therefore takes its cell count directly, which is the recommended
spelling:

```julia
gp = SpringsteelGridParameters(
    geometry    = "RRR",
    iMin = 0.0, iMax = 10.0, num_cells_i = 10,   # DX_i = 1.0, iDim = 30
    jMin = 0.0, jMax = 20.0, num_cells_j = 20,   # DX_j = 1.0, jDim = 60
    kMin = 0.0, kMax =  2.5, num_cells_k = 5,    # DX_k = 0.5, kDim = 15
    vars = Dict("u" => 1))
```

`num_cells_i` is an alias for `num_cells`. Supplying the gridpoint count
(`iDim`/`jDim`/`kDim`) instead also works and back-derives the cell count; supplying both
is fine when they agree and an error when they do not.

#### The default when a j/k axis is unspecified

If a Cartesian spline j- or k-axis gives neither a cell count nor a gridpoint count, its
cell count defaults to **uniform nodal spacing** — the number of cells that matches its
cell width to the i-direction cell width, rounded up:

```math
\texttt{num\_cells\_j} = \left\lceil \frac{j_{max} - j_{min}}{\Delta x_i} \right\rceil,
\qquad \Delta x_i = \frac{i_{max} - i_{min}}{\texttt{num\_cells}}
```

Because this is a ratio it is invariant under i-direction tiling, which is what lets a
tile inherit its parent patch's j-resolution. When the division is not exact, the
rounding leaves $\Delta x_j \neq \Delta x_i$ and a warning is emitted; set `num_cells_j`
explicitly to choose the resolution yourself and silence it.

### The Three-Step Transform Pipeline

**Forward transform (physical → spectral):**

The forward transform is split into two steps following Ooyama (2002), sections 4a–b:

1. **SB transform** — projects mish values onto the basis via weighted summation:

   $$b_n = \sum_{\text{cells}} \Delta x \sum_{\mu=1}^{3} w_\mu \,\varphi_n(x_\mu)\, u(x_\mu)$$

2. **SA transform** — solves the variational system with boundary conditions to obtain
   spectral coefficients:

   $$\hat{a} = \bigl[\Gamma(P + Q)\Gamma^T\bigr]^{-1}\,\Gamma\, b, \qquad
   a = \Gamma^T \hat{a}$$

   where $P = \int \varphi_n \varphi_{n'}\,dx$ is the Gram matrix, $Q$ incorporates
   the optional derivative-smoothing constraint, and $\Gamma$ is the **base-folding
   operator** that encodes the current set of boundary conditions.  The system is
   solved via a pre-factored Cholesky decomposition, so the SA transform is $O(m)$.

**Inverse transform (spectral → physical):**

3. **SI transform** — evaluates the B-spline expansion at any set of physical points:

   $$u(x) = \sum_n a_n \,\varphi_n(x), \qquad
   u'(x) = \sum_n a_n \,\varphi'_n(x), \qquad
   u''(x) = \sum_n a_n \,\varphi''_n(x)$$

   Value, first derivative, and second derivative are all computed simultaneously
   and stored in the three derivative slots of the physical array.

### Boundary Condition Rank and Type (Ooyama 2002, Eq. 3.2)

A homogeneous BC at boundary $x_0$ is designated **R$r$T$t$**, where
$r \in \{0,1,2,3\}$ is the *rank* (number of constraints) and $t$ identifies which
derivatives are constrained.  The rank removes $r$ border coefficients from the
spectral solve by modifying $\Gamma$.

| Constant | Rank | Condition | Physical use |
|:---|:---:|:---|:---|
| `R0` | 0 | No constraint (free boundary) | Open/interior boundaries, domain nesting |
| `R1T0` | 1 | $u(x_0) = 0$ | Dirichlet; zero-value wall |
| `R1T1` | 1 | $u'(x_0) = 0$ | Neumann; symmetry / reflecting wall |
| `R1T2` | 1 | $u''(x_0) = 0$ | Zero curvature at boundary |
| `R2T10` | 2 | $u = u' = 0$ | Symmetric reflection (value + slope) |
| `R2T20` | 2 | $u = u'' = 0$ | Antisymmetric reflection (value + curvature) |
| `R3` | 3 | $u = u' = u'' = 0$ | Full constraint; precursor to R3X nesting |
| `PERIODIC` | — | Cyclic domain | Azimuthal / periodic domains |

The base-folding coefficients used inside `calcGammaBC` (Ooyama 2002, Table 1):

| BC | $\alpha_1$ | $\beta_1$ |
|:---|:---:|:---:|
| `R1T0` | −4 | −1 |
| `R1T1` | 0 | 1 |
| `R1T2` | 2 | −1 |

| BC | $\alpha_2$ | $\beta_2$ |
|:---|:---:|:---:|
| `R2T10` | 1 | −0.5 |
| `R2T20` | −1 | 0 |

### Inhomogeneous boundary families: `R3X` and `R1T1X`

The constants above are all *homogeneous* — they constrain a derivative to **zero**.
Two families carry a nonzero boundary value instead, by adding an affine offset
`spline.ahat` to the folded solve rather than by changing $\Gamma$:

| Constant | Rank | Condition | Offset set by | Physical use |
|:---|:---:|:---|:---|:---|
| `R3X` | 3 | border trio pinned to donated values | [`set_ahat_r3x!`](@ref) | Patch nesting: a child reads its parent's border amplitudes |
| `R1T1X` | 1 | $u'(x_0) = \mathrm{d}u$ (prescribed) | [`set_ahat_neumann!`](@ref) | Rigid wall with a state-dependent flux |

`R1T1X` deserves a note, because the obvious alternative is worse. A rigid wall in a
compressible atmosphere needs a pressure condition that no homogeneous BC supplies:
with $w$ Dirichlet at the wall, $w \equiv 0$ there for all time, so the vertical
momentum equation collapses to the exact identity $\partial p'/\partial z = -g\rho_t'$
— nonzero and state-dependent. Meanwhile the semi-implicit acoustic solve eliminates
$w$, so it cannot see a wall derivative that a refit injects, and needs the `R1T1`
subspace to stay operator-consistent. `R1T1` gets the second property and loses the
first; `R1T2` gets the first and loses the second, at a measured factor of ~2.7 in
stable timestep.

`R1T1X` resolves this by using the **same** `gammaBC` as `R1T1` — hence the same
admissible subspace and the same solver stability — and carrying the boundary
derivative in `ahat`. Setting $\mathrm{d}u = 0$ reproduces the homogeneous `R1T1` fit
*bitwise*, so nothing that does not opt in can change.

For a k-direction basis the wall data is per-column, and one spline object is shared
across every column, so it cannot live on the spline. It lives on the
[`SplineBasisArray`](@ref) as `wall_du`, indexed `[column, variable, side, dr+1]`, and
is installed inside the transform's column loop. The `dr` axis is load-bearing: a 2-D
transform differentiates in `i` *before* fitting in `k`, so the `dr = 1`/`dr = 2`
passes need $\partial(\mathrm{d}u)/\partial i$ and $\partial^2(\mathrm{d}u)/\partial i^2$.
[`set_wall_derivatives!`](@ref) fills all three levels, differentiating the wall profile
through the variable's own i-basis:

```julia
gp = SpringsteelGridParameters(
    geometry = "RiRk",
    iMin = 0.0, iMax = 10.0e3, num_cells = 20, mubar = 3,
    kMin = 0.0, kMax = 10.0e3, num_cells_k = 20, kDim = 60,
    BCB = Dict("p" => CubicBSpline.R1T1X),   # bottom wall carries dp/dz
    BCT = Dict("p" => CubicBSpline.R1T1),
    vars = Dict("p" => 1))
grid = createGrid(gp)

# dp/dz = -g·ρ_t at the wall: one value per i-direction gridpoint (length iDim)
xs = grid.ibasis.data[1, 1].mishPoints
set_wall_derivatives!(grid, :bottom, "p", [-9.81 * rho_t(x) for x in xs])
gridTransform!(grid)
```

The argument order is `(grid, side, var, du)`, `side` being `:bottom` or `:top`. Setting a
derivative on a wall whose variable did not declare `R1T1X` throws rather than being
silently ignored.

---

## Boundary Condition Constants

```@docs
CubicBSpline.R0
CubicBSpline.R1T0
CubicBSpline.R1T1
CubicBSpline.R1T2
CubicBSpline.R2T10
CubicBSpline.R2T20
CubicBSpline.R3
CubicBSpline.R3X
CubicBSpline.R1T1X
CubicBSpline.PERIODIC
```

---

## Positivity-Constrained Fits

A least-squares cubic fit to a sharply-peaked positive field **undershoots on the
flanks** — mixing ratios go negative, and a model that takes a logarithm or a square
root of them fails. `SAtransform_bounded!` removes this by imposing a per-coefficient
lower bound on the SA solve.

The reason a bound on *coefficients* is enough is the **convex-hull property**: the
B-spline basis is non-negative and forms a partition of unity, so
$u(x) = \sum_n a_n \varphi_n(x) \ge \min_n a_n$ everywhere. Constraining
$a_n \ge \ell_n$ therefore bounds the *reconstruction* at every point of the domain,
not merely at the mish points where the fit was sampled.

The clip is **conservative**: the mass a column must shed is redistributed within that
same column, and any deficit that cannot be placed is accumulated into
[`bound_shortfall`](@ref) rather than silently created. A nonzero shortfall is a
diagnostic that the column was infeasible — see RULE 2 below.

### Opting in

Set the `positivity` field on the grid parameters, keyed by variable name and then by
direction:

```julia
gp = SpringsteelGridParameters(
    geometry = "RiRk",
    iMin = 0.0, iMax = 30.0e3, num_cells = 30, mubar = 3,
    kMin = 0.0, kMax = 10.0e3, num_cells_k = 20, kDim = 60,
    BCL = Dict("default" => CubicBSpline.R0),
    BCR = Dict("default" => CubicBSpline.R0),
    BCB = Dict("default" => CubicBSpline.R0),
    BCT = Dict("default" => CubicBSpline.R0),
    vars = Dict("qr" => 1, "u" => 2),
    positivity = Dict("qr" => Dict(:i => 0.0, :k => 0.0)))   # rain is non-negative
```

Only the named variable is constrained; `u` above is fitted exactly as before. On a
bare `Spline1D` the equivalent is the `lower` keyword, or
[`set_lower_bound!`](@ref) / [`set_lower_bound_from_profile!`](@ref) /
[`clear_lower_bound!`](@ref) after construction.

### Which legs to bound

A multi-dimensional transform fits one direction at a time. Two rules govern the choice:

**RULE 1 (sufficiency).** Bounding the **last** leg puts the field above the bound
everywhere the model evaluates it, because by then every earlier direction has already
been collapsed to a physical coordinate.

**RULE 2 (feasibility).** Bounding the **earlier** legs is what keeps the last one
solvable. By partition of unity $\sum_m b_m = \int u$, and $\sum_m a_m w_m = \int u$
because the $\ell_q$ penalty has zero third derivative on $\sum_m \varphi_m \equiv 1$.
So a componentwise non-negative $b$ entering a leg guarantees that leg's column mass is
non-negative, hence always conservatively fixable. Without it, a column made entirely of
ringing (empty air beside a narrow rain shaft) is infeasible and the limiter must create
mass — which shows up as a nonzero `bound_shortfall`.

In practice: bound the last leg for correctness, and the earlier legs to drive the
shortfall to zero.

### What is rejected, and why

A box constraint on coefficients cannot express positivity in every setting, so the
unsupported cases throw at construction rather than mis-clipping:

| Case | Why |
|:---|:---|
| `R1T0`, `R3` boundaries | Dirichlet folds $a_1 = -4a_2 - a_3$; a box on the free coefficients says nothing about the slaved one |
| `R3X` borders | The border trio is pinned to the parent's donated `ahat` — there is nothing left to adjust |
| Fourier / Chebyshev directions | No convex-hull property, so no coefficient box expresses positivity |
| A **nonzero** bound on an intermediate leg | An intermediate leg fits the remaining directions' inner products, not the field; a bound of zero carries through unchanged (the basis is non-negative), but a nonzero physical bound would pick up that direction's basis integral |

A fully-spline geometry (`R`, `RR`, `RiRk`, `RRR`) has neither problem on any leg.

```@docs
CubicBSpline.SAtransform_bounded!
CubicBSpline.SAtransform_bounded
CubicBSpline.set_lower_bound!
CubicBSpline.set_lower_bound_from_profile!
CubicBSpline.clear_lower_bound!
CubicBSpline.bound_shortfall
CubicBSpline.basis_integrals
```

## Parameter and Data Structures

```@docs
CubicBSpline.SplineParameters
CubicBSpline.Spline1D
```

## Internal Basis and Setup Functions

```@docs
CubicBSpline.basis
CubicBSpline.calcGammaBC
CubicBSpline.calcPQfactor
CubicBSpline.calcP1factor
CubicBSpline.calcMishPoints
CubicBSpline.setMishValues
```

## Transform Functions

```@docs
CubicBSpline.SBtransform
CubicBSpline.SBxtransform
CubicBSpline.SBxtransform!
CubicBSpline.SAtransform
CubicBSpline.SItransform
CubicBSpline.SItransform_matrix
CubicBSpline.SIxtransform
CubicBSpline.SIxxtransform
CubicBSpline.SIIntcoefficients
CubicBSpline.SIIntcoefficients!
CubicBSpline.SIInttransform
CubicBSpline.SIInttransform!
CubicBSpline.set_ahat_r3x!
CubicBSpline.set_ahat_neumann!
```

## Matrix Representations

```@docs
CubicBSpline.spline_basis_matrix
CubicBSpline.spline_1st_derivative_matrix
CubicBSpline.spline_2nd_derivative_matrix
```

## Generic Wrappers

No-prefix wrappers that delegate to the `S`-prefixed functions above, enabling
basis-type-agnostic code.

`Ixtransform` and `Ixxtransform` each have a two-argument in-place form
(`Ixtransform(spline, dest)`) alongside the allocating one-argument form. Per-column hot
paths should prefer the in-place form: it is bit-for-bit identical to the allocating
version and allocates nothing once warm. The same two-argument spelling works on the
Chebyshev basis, so cross-basis column loops need no branch.

`IInttransform(spline, [uMish,] C0)` anchors its result so that the antiderivative equals
`C0` at `xmin`, matching the Chebyshev basis. This is the gauge a caller means by "`C0` is
the value at the bottom". The spline-native `SIInttransform` keeps the Ooyama (2002)
gauge (zero near the domain centre, `C0` added uniformly) and is *not* interchangeable
with the generic wrapper.

```@docs
CubicBSpline.Btransform
CubicBSpline.Btransform!
CubicBSpline.Bxtransform
CubicBSpline.Atransform
CubicBSpline.Atransform!
CubicBSpline.Itransform!
CubicBSpline.Itransform
CubicBSpline.Ixtransform
CubicBSpline.Ixxtransform
CubicBSpline.IInttransform
```

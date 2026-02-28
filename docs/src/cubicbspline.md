```@meta
CurrentModule = Springsteel
```

# CubicBSpline

The `CubicBSpline` submodule implements the Ooyama (2002) cubic B-spline spectral
transform method, which provides compact-support basis functions with flexible boundary
conditions and a built-in sixth-order low-pass filter.

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
CubicBSpline.PERIODIC
```

## Parameter and Data Structures

```@docs
CubicBSpline.SplineParameters
CubicBSpline.Spline1D
CubicBSpline.Spline1D(::CubicBSpline.SplineParameters)
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
CubicBSpline.SAtransform
CubicBSpline.SItransform
CubicBSpline.SItransform_matrix
CubicBSpline.SIxtransform
CubicBSpline.SIxxtransform
CubicBSpline.SIIntcoefficients
CubicBSpline.SIInttransform
```

## Generic Wrappers

No-prefix wrappers that delegate to the `S`-prefixed functions above, enabling
basis-type-agnostic code.

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

```@meta
CurrentModule = Springsteel
```

# Chebyshev

The `Chebyshev` submodule implements the 1D Chebyshev spectral
transform used for the vertical direction in cylindrical- and
spherical-coordinate grids (`RZ_Grid`, `RLZ_Grid`, `SLZ_Grid`) and for
purely Chebyshev-based grids (`Z`, `ZZ`, `ZZZ`). It uses
[FFTW.jl](https://juliamath.github.io/FFTW.jl/latest/) DCT-I
transforms (`FFTW.REDFT00`) with a pre-measured plan for maximum
throughput. Grid points are placed at Chebyshev–Gauss–Lobatto (CGL)
nodes, giving spectral accuracy for smooth functions on a bounded
interval.

## Mathematical Overview

Chebyshev polynomials of the first kind, $T_k(x)$, are defined on
$x \in [-1, 1]$ by the identity $T_k(\cos\theta) = \cos(k\theta)$.
They form an orthogonal basis under the weight
$(1-x^2)^{-1/2}$, and any sufficiently smooth function $u(x)$ can be
expanded as

```math
u(x) \;=\; \sum_{k=0}^{\infty} c_k\, T_k(x),
```

truncated to a finite sum of $N+1$ terms in practice. For smooth $u$
the coefficients $c_k$ decay exponentially in $k$ — this is where the
"spectral accuracy" phrase comes from, and why Chebyshev is the
standard choice for non-periodic bounded-interval problems.

### Chebyshev–Gauss–Lobatto collocation

Springsteel evaluates $u$ at the Chebyshev–Gauss–Lobatto (CGL) nodes
$x_j = \cos(\pi j / N)$ for $j = 0, \ldots, N$. At these nodes the
polynomial interpolation problem is diagonalised by the discrete
cosine transform: if $u_j = u(x_j)$ then the coefficients are

```math
c_k \;=\; \frac{p_k}{N}
          \sum_{j=0}^{N} p_j\, u_j\, \cos\!\left(\frac{\pi j k}{N}\right),
\qquad p_j \;=\; \begin{cases} \tfrac12, & j = 0 \text{ or } j = N \\ 1, & \text{otherwise}\end{cases}
```

This is exactly a DCT-I (`FFTW.REDFT00`), so the forward and inverse
transforms run at FFT speed. The inverse is the same transform with
a factor-of-2 rescaling baked in.

### Derivatives

Differentiating a Chebyshev expansion is exact and local in
coefficient space: the recurrence
$T'_{k+1}(x) = 2(k+1) T_k(x) + \tfrac{k+1}{k-1}T'_{k-1}(x)$
gives a triangular linear map from $\{c_k\}$ to $\{c'_k\}$. Springsteel
applies this in-place for first and second derivatives and also
exposes matrix representations via [`Chebyshev.dct_matrix`](@ref),
[`Chebyshev.dct_1st_derivative`](@ref), and
[`Chebyshev.dct_2nd_derivative`](@ref) for operator assembly in the
[Solver Framework](solver.md).

### Boundary conditions

Springsteel currently implements three Chebyshev BC types at the
transform level, all homogeneous:

| Constant | Condition                | Method |
|:---------|:-------------------------|:-------|
| [`Chebyshev.R0`](@ref)   | natural (no constraint)  | zero `gammaBC` vector — no-op |
| [`Chebyshev.R1T0`](@ref) | Dirichlet $u(z_0) = 0$   | rank-1 coefficient correction vector |
| [`Chebyshev.R1T1`](@ref) | Neumann $u'(z_0) = 0$    | full $N \times N$ correction matrix via the Wang et al. (1993) global coefficient method |

Each BC combination (`BCB`/`BCT` pair) produces a `gammaBC` correction
stored on the [`Chebyshev.Chebyshev1D`](@ref) object at construction
time. The correction is then applied during the forward transform by
`CAtransform` as

```math
a \;=\; b_{\text{fill}} + \texttt{gammaBC}^{\top} b_{\text{fill}},
```

where $b_{\text{fill}}$ is the raw DCT output padded to the full mode
count. For `R0/R0` the correction is the zero vector and no work is
done; for single-ended Dirichlet or Neumann the correction is rank-1;
for Neumann at one or both ends Springsteel falls back to the full
matrix form. This is a **coefficient-modification** scheme — the
boundary rows of the spectral-to-physical evaluation are not
rewritten, unlike the solver-framework row-replacement path described
below.

The Wang et al. (1993) reference:

> Wang, H., Lacroix, S., & Labrosse, G. (1993). *A Chebyshev
> collocation method for the Stokes problem with application to the
> driven cavity.* Journal of Computational Physics, 106(1), 7–24.
> [doi:10.1006/jcph.1993.1133](https://doi.org/10.1006/jcph.1993.1133)

!!! note "Supported BCs only"
    Only `R0`, `R1T0`, and `R1T1` currently work at the transform
    level. The Ooyama higher-rank constants
    [`Chebyshev.R1T2`](@ref), [`Chebyshev.R2T10`](@ref),
    [`Chebyshev.R2T20`](@ref), and [`Chebyshev.R3`](@ref) are exported
    for naming symmetry with `CubicBSpline` but throw `DomainError` at
    grid construction. Inhomogeneous (non-zero) Dirichlet / Neumann
    values are also not yet implemented. See the
    [Contributing](contributing.md) roadmap.

#### Solver-level BC enforcement (separate path)

When the [Solver Framework](solver.md) assembles an explicit operator
matrix $\mathbf{L}$ for a `SpringsteelProblem`, it layers a *second*
BC mechanism on top of the transform-level `gammaBC`: the boundary
rows of $\mathbf{L}$ are replaced with evaluation rows (for Dirichlet)
or first-derivative rows (for Neumann) at the two boundary CGL nodes
$x_0$ and $x_N$, and the RHS at those rows is set to the BC value.
This row-replacement path handles the same R1T0 / R1T1 conditions as
the transform path — it does not add support for R1T2 or the higher
ranks. Both paths run for any problem built through the solver
framework; for base transforms (`spectralTransform!` /
`gridTransform!`) only the transform-level path applies.

### References

- Wikipedia: [Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials)
- Wikipedia: [Discrete cosine transform](https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-I)
- Trefethen, *Spectral Methods in MATLAB*, SIAM, 2000 — Chapters 6–8
  for the collocation viewpoint and derivative matrices
- Boyd, *Chebyshev and Fourier Spectral Methods*, 2nd ed., Dover, 2001
  — comprehensive reference including boundary condition handling

## Boundary Condition Constants

```@docs
Chebyshev.R0
Chebyshev.R1T0
Chebyshev.R1T1
Chebyshev.R1T2
Chebyshev.R2T10
Chebyshev.R2T20
Chebyshev.R3
```

## Parameter and Data Structures

```@docs
Chebyshev.ChebyshevParameters
Chebyshev.Chebyshev1D
```

## Internal Setup Functions

```@docs
Chebyshev.calcMishPoints
Chebyshev.calcFilterMatrix
Chebyshev.calcGammaBC
Chebyshev.calcGammaBCalt
```

## Transform Functions

```@docs
Chebyshev.CBtransform
Chebyshev.CAtransform
Chebyshev.CItransform
Chebyshev.CIxcoefficients
Chebyshev.CIxtransform
Chebyshev.CIxxtransform
Chebyshev.CIIntcoefficients
Chebyshev.CIInttransform
```

## Generic Wrappers

No-prefix wrappers that delegate to the `C`-prefixed functions above, enabling
basis-type-agnostic code.

```@docs
Chebyshev.Btransform
Chebyshev.Btransform!
Chebyshev.Atransform
Chebyshev.Atransform!
Chebyshev.Itransform
Chebyshev.Itransform!
Chebyshev.Ixtransform
Chebyshev.Ixxtransform
Chebyshev.IInttransform
```

## DCT Matrix Utilities

```@docs
Chebyshev.dct_matrix
Chebyshev.dct_1st_derivative
Chebyshev.dct_2nd_derivative
```

## BVP Solver Utilities

```@docs
Chebyshev.bvp
Chebyshev.bvp_modified_basis
Chebyshev.bvp_basis
```


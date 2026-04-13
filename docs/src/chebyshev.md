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

### Derivatives and boundary conditions

Differentiating a Chebyshev expansion is exact and local in
coefficient space: the recurrence
$T'_{k+1}(x) = 2(k+1) T_k(x) + \tfrac{k+1}{k-1}T'_{k-1}(x)$
gives a triangular linear map from $\{c_k\}$ to $\{c'_k\}$. Springsteel
applies this in-place for first and second derivatives and also
exposes matrix representations via [`Chebyshev.dct_matrix`](@ref),
[`Chebyshev.dct_1st_derivative`](@ref), and
[`Chebyshev.dct_2nd_derivative`](@ref) for operator assembly in the
[Solver Framework](solver.md).

Boundary conditions are enforced by replacing the boundary-row
equations in the assembled operator with evaluation or derivative rows
for the two boundary CGL nodes $x_0 = 1$ and $x_N = -1$. This is
simpler than the spline side because no basis coefficient is "lost"
to a projection — only two rows of the spectral-to-physical
evaluation matrix are rewritten. See the Ooyama paper's discussion of
boundary-row replacement (§4) and the
[Boundary Conditions](boundary_conditions.md) page for details.

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


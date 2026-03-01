```@meta
CurrentModule = Springsteel
```

# Solver Framework

The Springsteel solver framework provides tools for assembling and solving linear
boundary value problems (BVPs) on spectral grids. It supports all grid types
(CubicBSpline, Fourier, Chebyshev, and mixed) in 1D, 2D, and 3D.

Two solver backends are available:

- **`LocalLinearBackend`**: Built-in direct solver using LU factorisation.
  Assembles operator matrices from Kronecker products of 1D basis matrices,
  applies boundary conditions automatically, and solves
  ``\mathbf{L} \mathbf{a} = \mathbf{f}``.

- **`OptimizationBackend`**: Extension backend using
  [Optimization.jl](https://github.com/SciML/Optimization.jl) for nonlinear
  problems. Minimises a user-defined cost functional ``J(u, p)`` over spectral
  coefficients. Requires loading `Optimization.jl` separately.

## Quick Start

```julia
using Springsteel

# 1. Create a grid
gp = SpringsteelGridParameters(
    geometry = "Z",
    iMin = 0.0, iMax = 1.0, iDim = 25, b_iDim = 25,
    BCL = Dict("u" => Chebyshev.R1T0),
    BCR = Dict("u" => Chebyshev.R1T0),
    vars = Dict("u" => 1))
grid = createGrid(gp)

# 2. Assemble the operator (Poisson: u'' = f)
L = assemble_from_equation(grid, "u"; d_ii=1.0)

# 3. Define the RHS
pts = solver_gridpoints(grid, "u")
f = -pi^2 .* sin.(pi .* pts)

# 4. Solve
prob = SpringsteelProblem(grid; operator=L, rhs=f)
sol = solve(prob)

# 5. Check the result
u_analytic = sin.(pi .* pts)
@assert maximum(abs.(sol.physical .- u_analytic)) < 1e-4
```

## Boundary Condition Handling

The solver automatically applies boundary conditions based on each dimension's
basis type. No manual BC enforcement is needed.

- **CubicBSpline** dimensions use the ``\boldsymbol{\Gamma}_{\mathrm{BC}}``
  projection matrix (Ooyama 2002) to fold homogeneous boundary constraints
  directly into the operator. This reduces the system size by the number of
  constrained boundary coefficients.

- **Chebyshev** dimensions enforce BCs by replacing boundary rows of the
  assembled operator with evaluation (Dirichlet) or derivative (Neumann)
  constraint rows and setting the corresponding RHS entries to the BC values.
  The CGL point ordering produced by `calcMishPoints` places `zmin` at row 1
  and `zmax` at row N, so `BCB` (bottom/`zmin`) maps to row 1 and `BCT`
  (top/`zmax`) maps to row N.

- **Fourier** dimensions are periodic by construction; no BC modification is
  applied.

For multi-dimensional grids with mixed basis types, the solver composes these
strategies via Kronecker products: spline ``\Gamma_{\mathrm{BC}}`` matrices
are folded first, then Chebyshev boundary rows are replaced in the resulting
(possibly rectangular) system.

## Matrix Representations

Each basis module provides matrix representations of the evaluation and
derivative operators, used internally by the solver for operator assembly.
See the individual module pages for full documentation:

- **CubicBSpline**: [`spline_basis_matrix`](@ref CubicBSpline.spline_basis_matrix), [`spline_1st_derivative_matrix`](@ref CubicBSpline.spline_1st_derivative_matrix), [`spline_2nd_derivative_matrix`](@ref CubicBSpline.spline_2nd_derivative_matrix)
- **Fourier**: [`dft_matrix`](@ref Fourier.dft_matrix), [`dft_1st_derivative`](@ref Fourier.dft_1st_derivative), [`dft_2nd_derivative`](@ref Fourier.dft_2nd_derivative)
- **Chebyshev**: [`dct_matrix`](@ref Chebyshev.dct_matrix), [`dct_1st_derivative`](@ref Chebyshev.dct_1st_derivative), [`dct_2nd_derivative`](@ref Chebyshev.dct_2nd_derivative)

## Operator Assembly

```@docs
OperatorTerm
operator_matrix
assemble_operator
assemble_from_equation
```

## Problem Definition

```@docs
AbstractSolverBackend
LocalLinearBackend
OptimizationBackend
SpringsteelProblem
SpringsteelSolution
```

## Solving

```@docs
solve
solver_gridpoints
```

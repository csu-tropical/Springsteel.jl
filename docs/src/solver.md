```@meta
CurrentModule = Springsteel
```

# Solver Framework

The Springsteel solver framework assembles and solves linear boundary
value problems on any supported grid geometry, using the grid's native
basis (B-spline / Fourier / Chebyshev / mixed). It provides two usage
styles depending on your needs:

- a **declarative operator algebra DSL** (`(∂_x^2 + ∂_y^2) * u => :f`)
  that reads like the underlying mathematics and caches a workspace for
  repeated solves;
- a **direct kwarg form** (`SpringsteelProblem(grid; operator=L, rhs=f)`)
  for callers who already have an assembled operator matrix and want a
  one-shot solve.

Both styles flow through the same workspace-based solver path —
they differ only in how the problem is constructed.

## Quick start

```julia
using Springsteel

gp = SpringsteelGridParameters(
    geometry = "Z",
    iMin = 0.0, iMax = 1.0, iDim = 25, b_iDim = 25,
    vars = Dict("u" => 1, "f" => 2),
    BCL  = Dict("u" => DirichletBC(), "f" => NaturalBC()),
    BCR  = Dict("u" => DirichletBC(), "f" => NaturalBC()),
)
grid = createGrid(gp)

# Fill the RHS into its own grid slot
pts = solver_gridpoints(grid, "u")
grid.physical[:, 2, 1] .= -π^2 .* sin.(π .* pts)

# Build the problem and solve in place; result goes into grid.physical[:, 1, 1]
u = SpringsteelField(grid, "u")
prob = SpringsteelProblem(grid, ∂ᵢ^2 * u => :f)
solve!(prob)

maximum(abs.(grid.physical[:, 1, 1] .- sin.(π .* pts)))   # ≈ machine ε
```

The RHS is kept in its own grid variable (`f`, slot 2) so that the
writeback of `u` into slot 1 doesn't destroy it. Reusing the same slot
for both the unknown and the RHS is legal for a one-shot solve but
breaks repeated `solve!` — see §5 of the [Tutorial](tutorial.md) for a
full discussion.

The Pair-based constructor is the preferred v1.0 style. It builds a
workspace at construction (gammaBC fold, BC row detection, cached
factorisation) so subsequent `solve!` calls only pay for the backsolve.

## Fields

An unknown variable in an operator expression is represented by a
[`SpringsteelField`](@ref) (aliased as `Field`). It holds the variable
name and its integer slot index in `grid.params.vars`:

```julia
u = Field(grid, "u")      # == Field("u", 1) if "u" is slot 1
```

`Field` is the short name used throughout the examples. If it collides
with a type from another package loaded in the same session, use the
qualified `SpringsteelField` directly.

```@docs
SpringsteelField
```

## Operator algebra

Differential operators are built from **derivative atoms** combined with
scalar / vector coefficients and Julia's `+`, `-`, `*`, `^` operators.
The result is an `OperatorExpr` that multiplies by a `Field` to produce
a `TypedOperator` — the object that lives on the left of the `=>` in the
Pair-based constructor.

```julia
L = ∂_x^2 + ∂_y^2                     # 2D Cartesian Laplacian
L = ∂_r^2 + (1/r) * ∂_r                # cylindrical radial Laplacian
L = α * ∂_x^2 + β * ∂_x + γ            # varying coefficients
```

### Derivative atoms

```@docs
Springsteel.∂ᵢ
DerivMono
ScaledMono
OperatorExpr
TypedOperator
```

The derivative atoms come in two flavours — **generic** (axis-indexed)
and **physical** (named). The physical names are resolved against the
grid's geometry at lowering time, so the same symbol means different
things on a Cartesian vs cylindrical vs spherical grid.

**Generic atoms** act directly on the grid's `i` / `j` / `k` axes. Use
these in geometry-agnostic code:

| Symbol | ASCII | Axis |
|:------:|:-----:|:----:|
| `∂ᵢ`   | `d_i` | `:i` |
| `∂ⱼ`   | `d_j` | `:j` |
| `∂ₖ`   | `d_k` | `:k` |

**Physical atoms** resolve per geometry:

| Symbol | ASCII     | Cartesian | Cylindrical | Spherical |
|:------:|:---------:|:---------:|:-----------:|:---------:|
| `∂_x`  | `d_x`     | `:i`      | —           | —         |
| `∂_y`  | `d_y`     | `:j`      | —           | —         |
| `∂_z`  | `d_z`     | `:k`      | `:k`        | `:k` (radial) |
| `∂_r`  | `d_r`     | —         | `:i`        | —         |
| `∂_θ`  | `d_theta` | —         | `:j`        | `:i` (colatitude) |
| `∂_λ`  | `d_lambda`| —         | `:j`        | `:j` (azimuth) |

Powers build second derivatives: `∂_x^2`, `∂_r^2`, etc. Products build
mixed partials: `∂_x * ∂_y` is `∂²/∂x∂y`.

### Coefficients

Any term can be scaled by a coefficient. Supported coefficient types:

| Type              | Interpretation                              |
|:------------------|:--------------------------------------------|
| `Number`          | scalar                                      |
| `Vector{Float64}` | pre-built spatially-varying coefficient     |
| `Function`        | evaluated against `solver_gridpoints(grid)` |
| `Symbol`          | pulled from `grid.physical[:, var, 1]`      |
| `Nothing`         | unit coefficient                            |

```julia
# Scalar
prob = SpringsteelProblem(grid, 0.5 * ∂_x^2 * u => :f)

# Vector (must match the grid's physical size)
κ = compute_kappa(grid)
prob = SpringsteelProblem(grid, κ * ∂_x^2 * u => :f)

# Function — evaluated at gridpoints at construction
prob = SpringsteelProblem(grid, (x -> 1 + x[1]^2) * ∂_x^2 * u => :f)

# Symbol — reads the named grid variable at construction
prob = SpringsteelProblem(grid, :κ * ∂_x^2 * u => :f)
```

### Right-hand sides

The RHS in `L * u => rhs` can be:

- a `Symbol` — the name of a grid variable; the workspace reads
  `grid.physical[:, rhs_idx, 1]` at each solve. Ideal for time-stepping
  loops.
- a `Vector{Float64}` — a literal RHS, snapshotted once at construction.
- a `Real` — a constant, broadcast to every physical gridpoint.

## Problem construction

```@docs
SpringsteelProblem
```

Two constructors are available. They produce the same struct and flow
through the same solver path.

### Pair-based (preferred)

```julia
SpringsteelProblem(grid, L * u => rhs;
                   backend = :auto,
                   preconditioner = :default)
```

Build a problem from an operator expression. `backend` can be a
`Symbol` (`:auto`, `:dense`, `:sparse`, `:krylov`) or a backend instance.
The default `:auto` picks `LocalLinearBackend` for pure-Chebyshev grids
(where structural sparsity is low) and `SparseLinearBackend` everywhere
else.

```julia
prob = SpringsteelProblem(grid, ∂_x^2 * u => :f)                  # auto
prob = SpringsteelProblem(grid, ∂_x^2 * u => :f; backend=:krylov)
```

### Kwarg form (direct operator/rhs)

```julia
SpringsteelProblem(grid;
                   operator = L::Matrix{Float64},
                   rhs      = f::Vector{Float64},
                   parameters = Dict{String,Any}(),
                   backend = LocalLinearBackend())
```

Use this when you've assembled the operator yourself — for example via
[`assemble_from_equation`](@ref) or through custom code that doesn't fit
the operator algebra DSL. Both `operator` and `rhs` are required for
linear backends; the workspace is built at construction time.

```julia
L = assemble_from_equation(grid, "u"; d_ii=1.0)
pts = solver_gridpoints(grid, "u")
f = -π^2 .* sin.(π .* pts)
prob = SpringsteelProblem(grid; operator=L, rhs=f)
sol  = solve(prob)
```

## Solving

Two entry points, sharing one implementation:

```@docs
solve!
solve
```

- **`solve!(prob)`** mutates `grid.physical[:, var_idx, 1]` in place and
  returns `prob`. This is the fast path — use it in time-stepping loops
  where you update the RHS grid variable between calls.
- **`solve(prob)`** runs the same computation but writes the result into
  a fresh independent [`SpringsteelGrid`](@ref) narrowed to just the
  solved field(s), then returns a [`SpringsteelSolution`](@ref) owning
  that grid. The grid originally passed to `SpringsteelProblem` is
  untouched.

Both entry points work with every linear backend
([`LocalLinearBackend`](@ref), [`SparseLinearBackend`](@ref),
[`KrylovLinearBackend`](@ref)). For [`OptimizationBackend`](@ref), only
`solve` is defined (via the `Optimization.jl` package extension).

```@docs
SpringsteelSolution
```

`sol.physical` and `sol.coefficients` are views into the owned solution
grid — no data is duplicated inside the Solution itself. `sol.grid` can
be passed directly to downstream operations like
[`gridTransform!`](@ref), [`evaluate_unstructured`](@ref),
[`regularGridTransform`](@ref), or [`write_grid`](@ref).

## Backends

### `LocalLinearBackend` — dense LU

```@docs
LocalLinearBackend
```

Best for small, dense systems — typically pure-Chebyshev grids where the
assembled operator has little exploitable structure. The default for
such grids under `backend = :auto`.

### `SparseLinearBackend` — sparse LU

```@docs
SparseLinearBackend
```

The default for any grid involving a B-spline or Fourier dimension.
B-spline/Fourier/Chebyshev tensor products produce structurally sparse
operators; sparse LU keeps the cached factor small as `num_cells` grows
and is typically an order of magnitude faster than dense LU at moderate
sizes.

### `KrylovLinearBackend` — iterative

```@docs
KrylovLinearBackend
```

Uses `Krylov.gmres` for square systems and `Krylov.lsmr` for rectangular
ones (which arise after gammaBC folding). Suitable when the operator is
ill-conditioned or when memory for a direct factor is a concern. The
backend accepts a preconditioner:

```julia
prob = SpringsteelProblem(grid, L * u => :f;
                           backend = :krylov,
                           preconditioner = :diag)

# Or pass an explicit inverse operator:
Minv = compute_preconditioner(grid)
prob = SpringsteelProblem(grid, L * u => :f;
                           backend = KrylovLinearBackend(Minv))
```

Preconditioner options: `nothing` (no preconditioner), `:diag` (diagonal
of the assembled operator), or an explicit left-inverse.

### `AbstractLinearBackend`

```@docs
AbstractSolverBackend
AbstractLinearBackend
```

Supertype for the three linear backends above. `solve` and `solve!`
dispatch on `{<:AbstractLinearBackend}` so that all three share one
implementation and `OptimizationBackend` cleanly hits its own method in
the extension.

### `OptimizationBackend`

```@docs
OptimizationBackend
```

Minimises a user-supplied cost functional `J(u, p)` using
[Optimization.jl](https://github.com/SciML/Optimization.jl). Requires
`using Optimization` to be loaded in the same session to pick up the
package extension method.

```julia
using Springsteel, Optimization

cost(phys, params) = sum((phys .- target).^2)
prob = SpringsteelProblem(grid;
                           cost    = cost,
                           backend = OptimizationBackend(:LBFGS))
sol = solve(prob)
```

## Block multi-variable systems

The Pair-based constructor accepts a `Vector` of equations for coupled
multi-variable problems. Each element is one equation `L_i => RHS_i`,
where `L_i` is a sum of `TypedTerm`s that bind to one or more unknowns.
The number of equations must equal the number of unique unknowns.

```julia
u = Field(grid, "u")
v = Field(grid, "v")

eqs = [
    ∂_x * u + ∂_y * v => :f1,
    ∂_x * v - ∂_y * u => :f2,
]
prob = SpringsteelProblem(grid, eqs; backend = :sparse)
solve!(prob)
```

Under the hood `prob.workspace` is a `BlockLinearWorkspace` that
assembles the full block operator and factorises it once. Each call to
`solve!` refreshes the per-equation RHS slices, overwrites BC rows, and
writes back per-variable into the grid.

Block systems do not currently support R3X inhomogeneous BCs; use the
single-variable path for problems that need the `ahat` correction.

## Boundary conditions

BCs are per-variable and per-direction, configured on the grid via
[`SpringsteelGridParameters`](@ref). The solver automatically applies
them at problem construction time:

- **CubicBSpline** — the `Γ_BC` projection matrix folds the boundary
  constraint into the operator, reducing the system size by the BC rank.
- **Chebyshev** — the boundary row of the assembled operator is replaced
  with an evaluation or derivative constraint row; the RHS at that row
  is overwritten with the BC value.
- **Fourier** — periodic by construction; no modification.

For the full BC specification syntax see
[Boundary Conditions](boundary_conditions.md).

## Operator assembly (low level)

If you need to build an operator matrix outside the DSL — for example to
inspect it, to combine it with an external term, or to use the kwarg
constructor — `assemble_from_equation` is the ergonomic entry point:

```@docs
assemble_from_equation
operator_matrix
assemble_operator
OperatorTerm
```

`assemble_from_equation` accepts keyword arguments for every supported
derivative term (`d0`, `d_i`, `d_ii`, `d_j`, `d_jj`, `d_k`, `d_kk`, plus
mixed partials) and returns the assembled `Matrix{Float64}` with gammaBC
folding applied. `operator_matrix` gives you a single-dimension 1D basis
matrix (evaluation, first, or second derivative) at the lowest level.

## Gridpoints

```@docs
solver_gridpoints
```

Returns the physical gridpoints at which the RHS and solution live — the
same layout as `grid.physical[:, var_idx, 1]`. Use this when you need to
evaluate an analytic expression for an inhomogeneous RHS or boundary.

## See also

- [Boundary Conditions](boundary_conditions.md) — BC type system and
  per-variable Dict spec
- [SpringsteelGrid](springsteel_grid.md) — grid construction
- [Tutorial](tutorial.md) — end-to-end examples
- [CubicBSpline](cubicbspline.md) / [Fourier](fourier.md) /
  [Chebyshev](chebyshev.md) — underlying basis modules and their
  derivative matrices

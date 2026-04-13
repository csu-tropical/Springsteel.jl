# solver.jl — SpringsteelProblem type, operator assembly, and linear solver
#
# Provides:
#   • OperatorTerm       — describes a single term in a PDE operator
#   • AbstractSolverBackend / LocalLinearBackend / OptimizationBackend
#   • SpringsteelProblem — bundles grid + operator/cost + backend
#   • SpringsteelSolution — solution container
#   • operator_matrix()  — extract 1D matrix for a given dimension + derivative order
#   • assemble_operator() — Kronecker product assembly of multi-D operators
#   • assemble_from_equation() — convenience equation-oriented interface
#   • solve()            — dispatch on backend
#
# Must be included AFTER basis modules, types.jl, factory.jl.

using LinearAlgebra

# ────────────────────────────────────────────────────────────────────────────
# Operator term
# ────────────────────────────────────────────────────────────────────────────

"""
    OperatorTerm

Describes a single term in a multi-dimensional linear differential operator.

Each term specifies derivative orders in the i, j, and k dimensions, plus an
optional scalar or spatially varying coefficient.

# Fields
- `i_order::Int`: Derivative order in i-dimension (0, 1, or 2)
- `j_order::Int`: Derivative order in j-dimension (0, 1, or 2)
- `k_order::Int`: Derivative order in k-dimension (0, 1, or 2)
- `coefficient::Union{Float64, Vector{Float64}, Nothing}`: Scalar multiplier,
  diagonal (spatially varying) coefficient vector, or `nothing` (treated as 1.0)

# Example
```julia
# Laplacian term: ∂²/∂i² with unit coefficient
OperatorTerm(2, 0, 0, 1.0)

# Variable-coefficient term: a(x) ∂²/∂k²
OperatorTerm(0, 0, 2, a_values)
```

See also: [`assemble_operator`](@ref), [`assemble_from_equation`](@ref)
"""
struct OperatorTerm
    i_order::Int
    j_order::Int
    k_order::Int
    coefficient::Union{Float64, Vector{Float64}, Nothing}
end

# ────────────────────────────────────────────────────────────────────────────
# Solver backend sentinels
# ────────────────────────────────────────────────────────────────────────────

"""
    AbstractSolverBackend

Abstract supertype for solver backend sentinel types. Concrete subtypes select the
algorithm used by [`solve`](@ref) / [`solve!`](@ref).

See also: [`AbstractLinearBackend`](@ref), [`OptimizationBackend`](@ref)
"""
abstract type AbstractSolverBackend end

"""
    AbstractLinearBackend <: AbstractSolverBackend

Supertype for backends that solve linear systems ``\\mathbf{L}\\mathbf{a}=\\mathbf{f}``
through the cached workspace path. Current concrete subtypes:
[`LocalLinearBackend`](@ref), [`SparseLinearBackend`](@ref), [`KrylovLinearBackend`](@ref).

`solve` and `solve!` share a single implementation across all
`AbstractLinearBackend`s — they differ only in the factorisation strategy
and (for Krylov) the preconditioner plumbing.
"""
abstract type AbstractLinearBackend <: AbstractSolverBackend end

"""
    LocalLinearBackend <: AbstractLinearBackend

Dense LU factorisation backend. Best for small, dense systems — typically
pure-Chebyshev grids where structural sparsity is limited.
"""
struct LocalLinearBackend <: AbstractLinearBackend end

"""
    SparseLinearBackend <: AbstractLinearBackend

Sparse LU factorisation backend. The assembled operator is converted to a
`SparseMatrixCSC` and factorised via `SparseArrays.lu`, so the cached factor
stays small for the structurally sparse operators that come out of
B-spline / Fourier / Chebyshev tensor products. The `solve!` hot path is
identical to the dense backend — only the factorisation type differs.
"""
struct SparseLinearBackend <: AbstractLinearBackend end

"""
    KrylovLinearBackend(; preconditioner=nothing) <: AbstractLinearBackend

Krylov iterative backend. Uses `Krylov.gmres` for square systems and
`Krylov.lsmr` for rectangular ones. The optional `preconditioner` is passed
through as the `M` (left) kwarg to Krylov.

v1.0 note: this backend still builds the assembled sparse operator (same
as `SparseLinearBackend`) — the truly-matrix-free path that avoids the
`kron` materialisation is deferred to a future phase. The immediate win
here is the iterative-solver option and preconditioner plumbing for
ill-conditioned problems.
"""
struct KrylovLinearBackend <: AbstractLinearBackend
    preconditioner::Any
end

KrylovLinearBackend() = KrylovLinearBackend(nothing)

"""
    OptimizationBackend <: AbstractSolverBackend

Backend sentinel for the Optimization.jl extension. Requires the `Optimization`
package to be loaded; the actual `solve` dispatch lives in
`ext/SpringsteelOptimizationExt.jl`.

# Fields
- `algorithm::Symbol`: Optimisation algorithm (e.g. `:LBFGS`, `:NelderMead`)
- `options::Dict{String, Any}`: Algorithm-specific options

See also: [`solve`](@ref), [`SpringsteelProblem`](@ref)
"""
struct OptimizationBackend <: AbstractSolverBackend
    algorithm::Symbol
    options::Dict{String, Any}
end

OptimizationBackend(alg::Symbol) = OptimizationBackend(alg, Dict{String, Any}())

# ────────────────────────────────────────────────────────────────────────────
# SpringsteelProblem
# ────────────────────────────────────────────────────────────────────────────

"""
    SpringsteelProblem{B <: AbstractSolverBackend}

Bundled grid + cached solver state for a linear or optimisation problem.

For linear-backend problems (the common path), `workspace` holds the
compiled operator, gammaBC fold, BC row indices, factorisation, and solve
scratch buffers — built once at construction time and reused by every
subsequent `solve` / `solve!`. For `OptimizationBackend`, `workspace` is
`nothing` and `cost` / `parameters` carry the problem description.

# Fields
- `grid`: the discretised domain
- `backend`: solver backend sentinel
- `workspace`: cached linear-solve state (`nothing` for Optimization)
- `cost`: cost functional ``J(u, p)`` (Optimization only)
- `parameters`: kwargs bag passed through to Optimization cost callbacks

# Constructors
```julia
# v1.0 operator-algebra form (preferred)
u = Field(grid, "u")
prob = SpringsteelProblem(grid, (∂_x^2 + ∂_y^2) * u => :rhs_var)

# Legacy kwarg form — user-assembled operator and RHS
prob = SpringsteelProblem(grid; operator=L, rhs=f)
```

See also: [`solve`](@ref), [`solve!`](@ref), [`SpringsteelSolution`](@ref),
[`assemble_operator`](@ref)
"""
mutable struct SpringsteelProblem{B <: AbstractSolverBackend}
    grid::AbstractGrid
    backend::B
    workspace::Any       # populated for linear-backend problems; `nothing` for Optimization
    cost::Union{Function, Nothing}     # Optimization-only
    parameters::Dict{String, Any}      # Optimization kwargs bag
end

"""
    SpringsteelProblem(grid; operator=nothing, rhs=nothing, cost=nothing,
                       parameters=Dict{String,Any}(),
                       backend=LocalLinearBackend()) -> SpringsteelProblem

Keyword-form constructor. Use this when you have a pre-assembled operator
matrix `L` and an RHS vector `f`, or when you're configuring an
`OptimizationBackend` with a cost functional.

For linear backends, the workspace (operator fold, BC detection, cached
factorisation) is built at construction time so repeated `solve` / `solve!`
calls on the same problem are fast. If you only need one solve, this form
is as cheap as the Pair-based form. If you're iterating in a time-stepping
loop with a changing RHS, prefer the Pair-based form — it lets you update
`grid.physical[:, rhs_idx, 1]` in place and call `solve!` without
rebuilding anything:

```julia
u = Field(grid, "u")
prob = SpringsteelProblem(grid, (∂_x^2 + ∂_y^2) * u => :rhs_var)
while stepping
    grid.physical[:, rhs_idx, 1] .= new_rhs
    solve!(prob)
end
```

Block multi-variable systems, backend selection (`:dense` / `:sparse` /
`:krylov`), preconditioners, and `Function` / `Symbol` coefficient
resolution are only supported through the Pair-based constructor.
"""
function SpringsteelProblem(grid::AbstractGrid;
    operator::Union{Matrix{Float64}, Nothing} = nothing,
    rhs::Union{Vector{Float64}, Nothing} = nothing,
    cost::Union{Function, Nothing} = nothing,
    parameters::Dict{String, Any} = Dict{String, Any}(),
    backend::AbstractSolverBackend = LocalLinearBackend())

    # For linear backends with both operator and rhs supplied, build a
    # workspace on construction so `solve` and `solve!` share the v1.0
    # workspace path. OptimizationBackend keeps `workspace = nothing` and
    # takes the minimisation path in the Optimization extension.
    workspace = nothing
    if backend isa LocalLinearBackend
        if operator === nothing || rhs === nothing
            throw(ArgumentError(
                "LocalLinearBackend kwarg constructor requires both " *
                "`operator` and `rhs` — or use the Pair-based constructor"))
        end
        var_name = get(parameters, "var", "")
        if isempty(var_name)
            var_name = first(keys(grid.params.vars))
        end
        field = SpringsteelField(grid, var_name)
        workspace = _build_local_linear_workspace(grid, field, operator, rhs, backend)
    end

    return SpringsteelProblem(grid, backend, workspace, cost, parameters)
end

# ────────────────────────────────────────────────────────────────────────────
# SpringsteelSolution
# ────────────────────────────────────────────────────────────────────────────

"""
    SpringsteelSolution

Result of a non-mutating [`solve`](@ref).

A `SpringsteelSolution` owns an independent [`SpringsteelGrid`](@ref) that is
narrowed to just the solved field(s). The solver writes its output into this
grid, leaving the grid that was passed to `solve` untouched.

# Fields
- `grid`: narrowed solution grid — an independent copy that only holds the
  solved variable(s). Can be passed directly to `gridTransform!`,
  `evaluate_unstructured`, `write_grid`, etc.
- `var_idx`: slot of the primary solved variable in `grid.physical`
  (always `1` for single-field problems; for block systems, points at the
  first field — use `grid.physical[:, idx, 1]` with the block's own var dict
  to read other unknowns).
- `converged`: whether the solver converged.

# Property forwards

For backward compatibility and ergonomic access:

- `sol.physical` → `@view grid.physical[:, var_idx, 1]`
- `sol.coefficients` → `@view grid.spectral[:, var_idx]`

These are views into the owned solution grid, so no data is duplicated.

See also: [`solve`](@ref), [`SpringsteelProblem`](@ref)
"""
struct SpringsteelSolution{G <: AbstractGrid}
    grid::G
    var_idx::Int
    converged::Bool
end

function Base.getproperty(sol::SpringsteelSolution, name::Symbol)
    if name === :physical
        g = getfield(sol, :grid)
        vi = getfield(sol, :var_idx)
        return @view g.physical[:, vi, 1]
    elseif name === :coefficients
        g = getfield(sol, :grid)
        vi = getfield(sol, :var_idx)
        return @view g.spectral[:, vi]
    else
        return getfield(sol, name)
    end
end

function Base.propertynames(::SpringsteelSolution, private::Bool=false)
    return (:grid, :var_idx, :converged, :physical, :coefficients)
end

# ────────────────────────────────────────────────────────────────────────────
# operator_matrix — extract 1D basis matrix for a given dimension
# ────────────────────────────────────────────────────────────────────────────

"""
    operator_matrix(grid::AbstractGrid, dim::Symbol, order::Int, var::String="") -> Matrix{Float64}

Extract a 1D operator matrix for dimension `dim` (`:i`, `:j`, or `:k`) at derivative
`order` (0 = evaluation, 1 = first derivative, 2 = second derivative).

Dispatches to the appropriate basis module's matrix function based on the grid's
basis type for that dimension.

# Arguments
- `grid::AbstractGrid`: The grid whose basis objects provide the matrix
- `dim::Symbol`: Which dimension (`:i`, `:j`, `:k`)
- `order::Int`: Derivative order (0, 1, or 2)
- `var::String`: Variable name (default: first variable)

# Returns
- `Matrix{Float64}`: Size `(physical_dim, spectral_dim)` for the requested dimension

See also: [`assemble_operator`](@ref), [`OperatorTerm`](@ref)
"""
function operator_matrix(grid::AbstractGrid, dim::Symbol, order::Int, var::String="")
    # Determine the variable index
    if var == ""
        var_idx = 1
    else
        var_idx = grid.params.vars[var]
    end

    # Get the appropriate basis object
    basis_obj = _get_basis_object(grid, dim, var_idx)

    # Dispatch to the correct matrix function based on basis type
    return _basis_matrix(basis_obj, order)
end

# Internal: get a representative basis object for a dimension
function _get_basis_object(grid::AbstractGrid, dim::Symbol, var_idx::Int)
    if dim == :i
        ba = grid.ibasis
    elseif dim == :j
        ba = grid.jbasis
    elseif dim == :k
        ba = grid.kbasis
    else
        throw(ArgumentError("Unknown dimension: $dim. Use :i, :j, or :k"))
    end

    if ba isa NoBasisArray
        throw(ArgumentError("Dimension $dim is inactive (NoBasisArray)"))
    end

    # For multi-dimensional arrays, get the first object for this variable
    data = ba.data
    if ndims(data) == 1
        return data[var_idx]
    elseif ndims(data) == 2
        return data[1, var_idx]
    else
        return data[1, 1, var_idx]
    end
end

# Internal: get matrix from a basis object at given derivative order
function _basis_matrix(obj::CubicBSpline.Spline1D, order::Int)
    if order == 0
        return CubicBSpline.spline_basis_matrix(obj)
    elseif order == 1
        return CubicBSpline.spline_1st_derivative_matrix(obj)
    elseif order == 2
        return CubicBSpline.spline_2nd_derivative_matrix(obj)
    else
        throw(ArgumentError("Derivative order $order not supported (use 0, 1, or 2)"))
    end
end

function _basis_matrix(obj::Fourier.Fourier1D, order::Int)
    if order == 0
        return Fourier.dft_matrix(obj)
    elseif order == 1
        return Fourier.dft_1st_derivative(obj)
    elseif order == 2
        return Fourier.dft_2nd_derivative(obj)
    else
        throw(ArgumentError("Derivative order $order not supported (use 0, 1, or 2)"))
    end
end

function _basis_matrix(obj::Chebyshev.Chebyshev1D, order::Int)
    cp = obj.params
    if order == 0
        return Chebyshev.dct_matrix(cp.zDim)
    elseif order == 1
        return Chebyshev.dct_1st_derivative(cp.zDim, cp.zmax - cp.zmin)
    elseif order == 2
        return Chebyshev.dct_2nd_derivative(cp.zDim, cp.zmax - cp.zmin)
    else
        throw(ArgumentError("Derivative order $order not supported (use 0, 1, or 2)"))
    end
end

# ────────────────────────────────────────────────────────────────────────────
# assemble_operator — Kronecker product assembly
# ────────────────────────────────────────────────────────────────────────────

"""
    assemble_operator(grid::AbstractGrid, terms::Vector{OperatorTerm}, var::String="") -> Matrix{Float64}

Build a multi-dimensional operator matrix from Kronecker products of 1D matrices.

For each `OperatorTerm`, the 1D matrices for active dimensions are combined via
`kron()` and optionally scaled by the term's coefficient. The total operator is
the sum of all terms.

# Arguments
- `grid::AbstractGrid`: The grid providing basis objects
- `terms::Vector{OperatorTerm}`: List of operator terms
- `var::String`: Variable name (default: first variable)

# Returns
- `Matrix{Float64}`: Assembled operator of size `(N_phys, N_spec)` where
  `N_phys` and `N_spec` are the total physical and spectral dimensions

See also: [`OperatorTerm`](@ref), [`operator_matrix`](@ref), [`assemble_from_equation`](@ref)
"""
function assemble_operator(grid::AbstractGrid, terms::Vector{OperatorTerm}, var::String="")
    # Determine which dimensions are active
    i_active = !(grid.ibasis isa NoBasisArray)
    j_active = !(grid.jbasis isa NoBasisArray)
    k_active = !(grid.kbasis isa NoBasisArray)

    L = nothing

    for term in terms
        # Build the Kronecker product for this term
        M = _build_term_matrix(grid, term, var, i_active, j_active, k_active)

        # Apply coefficient
        if term.coefficient isa Float64
            M = term.coefficient * M
        elseif term.coefficient isa Vector{Float64}
            M = Diagonal(term.coefficient) * M
        end
        # Nothing coefficient = 1.0, no scaling needed

        if L === nothing
            L = M
        else
            L = L + M
        end
    end

    return L
end

# Internal: build the Kronecker product matrix for a single operator term
function _build_term_matrix(grid::AbstractGrid, term::OperatorTerm, var::String,
                             i_active::Bool, j_active::Bool, k_active::Bool)
    # 1D case
    if !j_active && !k_active
        return operator_matrix(grid, :i, term.i_order, var)
    end

    # 2D case
    if j_active && !k_active
        Mi = operator_matrix(grid, :i, term.i_order, var)
        Mj = operator_matrix(grid, :j, term.j_order, var)
        return kron(Mi, Mj)
    end

    if !j_active && k_active
        Mi = operator_matrix(grid, :i, term.i_order, var)
        Mk = operator_matrix(grid, :k, term.k_order, var)
        return kron(Mi, Mk)
    end

    # 3D case
    Mi = operator_matrix(grid, :i, term.i_order, var)
    Mj = operator_matrix(grid, :j, term.j_order, var)
    Mk = operator_matrix(grid, :k, term.k_order, var)
    return kron(Mi, Mj, Mk)
end

# ────────────────────────────────────────────────────────────────────────────
# assemble_from_equation — convenience equation-oriented interface
# ────────────────────────────────────────────────────────────────────────────

"""
    assemble_from_equation(grid::AbstractGrid, var::String="";
        d0=nothing, d_i=nothing, d_j=nothing, d_k=nothing,
        d_ii=nothing, d_jj=nothing, d_kk=nothing,
        d_ij=nothing, d_ik=nothing, d_jk=nothing,
        d_iij=nothing, d_iik=nothing, d_ijj=nothing,
        d_jjk=nothing, d_ikk=nothing, d_jkk=nothing,
        d_ijk=nothing,
        d_iijj=nothing, d_iikk=nothing, d_jjkk=nothing,
        d_iijk=nothing, d_ijjk=nothing, d_ijkk=nothing,
        d_iijjk=nothing, d_iijkk=nothing, d_ijjkk=nothing,
        d_iijjkk=nothing) -> Matrix{Float64}

Build an operator matrix from an equation-oriented specification. Non-nothing
keyword arguments are converted to [`OperatorTerm`](@ref) objects and assembled
via [`assemble_operator`](@ref).

Each keyword encodes a derivative monomial where letter repetitions denote
derivative order in that dimension: `d_ijj` → ``\\partial^3 / \\partial i\\,\\partial j^2``
(i.e. `OperatorTerm(1, 2, 0, c)`).  All 27 combinations of orders 0–2 in
three dimensions are available.

# Keywords
## Pure derivatives (single dimension, order 0–2)
- `d0`: identity (0th derivative)
- `d_i`, `d_j`, `d_k`: first derivatives
- `d_ii`, `d_jj`, `d_kk`: second derivatives

## Mixed 2nd-order (two dimensions, each order 1)
- `d_ij`: ``\\partial^2/\\partial i\\,\\partial j``
- `d_ik`: ``\\partial^2/\\partial i\\,\\partial k``
- `d_jk`: ``\\partial^2/\\partial j\\,\\partial k``

## Mixed 3rd-order
- `d_iij`: ``\\partial^3/\\partial i^2\\,\\partial j``
- `d_iik`: ``\\partial^3/\\partial i^2\\,\\partial k``
- `d_ijj`: ``\\partial^3/\\partial i\\,\\partial j^2``
- `d_jjk`: ``\\partial^3/\\partial j^2\\,\\partial k``
- `d_ikk`: ``\\partial^3/\\partial i\\,\\partial k^2``
- `d_jkk`: ``\\partial^3/\\partial j\\,\\partial k^2``
- `d_ijk`: ``\\partial^3/\\partial i\\,\\partial j\\,\\partial k``

## Mixed 4th-order
- `d_iijj`: ``\\partial^4/\\partial i^2\\,\\partial j^2``
- `d_iikk`: ``\\partial^4/\\partial i^2\\,\\partial k^2``
- `d_jjkk`: ``\\partial^4/\\partial j^2\\,\\partial k^2``
- `d_iijk`: ``\\partial^4/\\partial i^2\\,\\partial j\\,\\partial k``
- `d_ijjk`: ``\\partial^4/\\partial i\\,\\partial j^2\\,\\partial k``
- `d_ijkk`: ``\\partial^4/\\partial i\\,\\partial j\\,\\partial k^2``

## Mixed 5th-order
- `d_iijjk`: ``\\partial^5/\\partial i^2\\,\\partial j^2\\,\\partial k``
- `d_iijkk`: ``\\partial^5/\\partial i^2\\,\\partial j\\,\\partial k^2``
- `d_ijjkk`: ``\\partial^5/\\partial i\\,\\partial j^2\\,\\partial k^2``

## Mixed 6th-order
- `d_iijjkk`: ``\\partial^6/\\partial i^2\\,\\partial j^2\\,\\partial k^2``

Each coefficient can be a `Float64` (scalar), `Vector{Float64}` (spatially varying),
or `nothing` (term not included).

!!! note "Prefer the Pair-based `SpringsteelProblem` constructor"
    As of the v1.0 solver refactor, new code should build operators via the
    operator algebra DSL and pass them to `SpringsteelProblem` with a `Pair`:

    ```julia
    u = Field(grid, "u")
    prob = SpringsteelProblem(grid, (∂ᵢ^2 + ∂ⱼ^2) * u => :f_rhs)
    solve!(prob)
    ```

    That path caches a stateful workspace (factorisation, BC rows, eval
    matrix, scratch buffers) so repeated `solve!` calls are allocation-free,
    and gives you backend choice (`:dense` / `:sparse` / `:krylov`) plus
    `Function`/`Symbol` coefficient resolution.

    `assemble_from_equation` is kept as a thin compatibility shim that
    internally builds the same DSL expression and delegates to `_lower`, so
    the kwarg form and the DSL form always produce identical term vectors.

# Example
```julia
# 1D Poisson: u'' = f
L = assemble_from_equation(grid; d_ii=1.0)

# 2D Laplacian: ∂²u/∂r² + ∂²u/∂z²
L = assemble_from_equation(grid; d_ii=1.0, d_kk=1.0)

# General 2nd-order ODE: au'' + bu' + cu = f
L = assemble_from_equation(grid; d_ii=a, d_i=b, d0=c)

# 2D with mixed derivative: ∂²u/∂i² + 2∂²u/∂i∂j + ∂²u/∂j²
L = assemble_from_equation(grid; d_ii=1.0, d_ij=2.0, d_jj=1.0)

# 3D mixed: ∂³u/∂i∂j²
L = assemble_from_equation(grid; d_ijj=1.0)
```

See also: [`assemble_operator`](@ref), [`OperatorTerm`](@ref)
"""
function assemble_from_equation(grid::AbstractGrid, var::String="";
    d0 = nothing,
    d_i = nothing,
    d_j = nothing,
    d_k = nothing,
    d_ii = nothing,
    d_jj = nothing,
    d_kk = nothing,
    d_ij = nothing,
    d_ik = nothing,
    d_jk = nothing,
    d_iij = nothing,
    d_iik = nothing,
    d_ijj = nothing,
    d_jjk = nothing,
    d_ikk = nothing,
    d_jkk = nothing,
    d_ijk = nothing,
    d_iijj = nothing,
    d_iikk = nothing,
    d_jjkk = nothing,
    d_iijk = nothing,
    d_ijjk = nothing,
    d_ijkk = nothing,
    d_iijjk = nothing,
    d_iijkk = nothing,
    d_ijjkk = nothing,
    d_iijjkk = nothing)

    # Build an OperatorExpr via the S1 DSL and delegate to `_lower`, so the
    # kwarg form and the `L*u => rhs` DSL form share a single lowering path.
    # The DSL types (`DerivMono`, `ScaledMono`, `OperatorExpr`) live in
    # operator_algebra.jl which is included after this file, so the name
    # lookups here happen at call time — not at module parse time.
    scaled = ScaledMono[]

    # Exhaustive enumeration of all (i_order, j_order, k_order) combinations
    # where each order ∈ {0, 1, 2}  (3³ = 27 total)
    for (kw, ijo) in (
        # Pure (single dimension)
        (d0,      (0,0,0)),
        (d_i,     (1,0,0)), (d_j,     (0,1,0)), (d_k,     (0,0,1)),
        (d_ii,    (2,0,0)), (d_jj,    (0,2,0)), (d_kk,    (0,0,2)),
        # Mixed 2nd-order
        (d_ij,    (1,1,0)), (d_ik,    (1,0,1)), (d_jk,    (0,1,1)),
        # Mixed 3rd-order
        (d_iij,   (2,1,0)), (d_iik,   (2,0,1)), (d_ijj,   (1,2,0)),
        (d_jjk,   (0,2,1)), (d_ikk,   (1,0,2)), (d_jkk,   (0,1,2)),
        (d_ijk,   (1,1,1)),
        # Mixed 4th-order
        (d_iijj,  (2,2,0)), (d_iikk,  (2,0,2)), (d_jjkk,  (0,2,2)),
        (d_iijk,  (2,1,1)), (d_ijjk,  (1,2,1)), (d_ijkk,  (1,1,2)),
        # Mixed 5th-order
        (d_iijjk, (2,2,1)), (d_iijkk, (2,1,2)), (d_ijjkk, (1,2,2)),
        # Mixed 6th-order
        (d_iijjkk,(2,2,2)),
    )
        kw === nothing && continue
        orders = Dict{Symbol, Int}()
        ijo[1] > 0 && (orders[:i] = ijo[1])
        ijo[2] > 0 && (orders[:j] = ijo[2])
        ijo[3] > 0 && (orders[:k] = ijo[3])
        push!(scaled, ScaledMono(_to_coeff(kw), DerivMono(orders)))
    end

    if isempty(scaled)
        throw(ArgumentError("At least one coefficient must be non-nothing"))
    end

    lowered = _lower(OperatorExpr(scaled), grid)
    return assemble_operator(grid, lowered, var)
end

_to_coeff(x::Float64) = x
_to_coeff(x::Int) = Float64(x)
_to_coeff(x::Vector{Float64}) = x

# ────────────────────────────────────────────────────────────────────────────
# solver_gridpoints — extract physical gridpoints from any grid type
# ────────────────────────────────────────────────────────────────────────────

"""
    solver_gridpoints(grid::AbstractGrid, var::String="") -> Vector{Float64}

Extract the physical gridpoints from a grid's i-dimension basis. Works for any
grid type including those without a `getGridpoints` method.

For 1D grids returns the i-dimension mish points. For multi-D grids returns the
Kronecker-product physical point locations for the i-dimension.
"""
function solver_gridpoints(grid::AbstractGrid, var::String="")
    var_idx = var == "" ? 1 : grid.params.vars[var]
    obj = _get_basis_object(grid, :i, var_idx)
    return Springsteel.gridpoints(obj)
end

# ────────────────────────────────────────────────────────────────────────────
# Boundary condition application for solve()
# ────────────────────────────────────────────────────────────────────────────

# Get boundary row indices and BC values for a Chebyshev dimension.
# Returns (bc_local_rows, bc_values) where bc_local_rows[i] is the local row
# index in this dimension and bc_values[i] is the target value.
#
# Row ordering follows the CGL point layout produced by calcMishPoints:
# the negative scale maps cos(0)=+1 → zmin (row 1) and cos(π)=-1 → zmax (row N).
# Therefore BCB (bottom/zmin) corresponds to row 1 and BCT (top/zmax) to row N.
function _chebyshev_bc_info(obj::Chebyshev.Chebyshev1D)
    cp = obj.params
    N = cp.zDim
    rows = Int[]
    vals = Float64[]

    # BCB applies at row 1 (zmin, bottom/left boundary)
    if haskey(cp.BCB, "α0")
        push!(rows, 1); push!(vals, cp.BCB["α0"])
    elseif haskey(cp.BCB, "α1")
        push!(rows, 1); push!(vals, cp.BCB["α1"])
    end

    # BCT applies at row N (zmax, top/right boundary)
    if haskey(cp.BCT, "α0")
        push!(rows, N); push!(vals, cp.BCT["α0"])
    elseif haskey(cp.BCT, "α1")
        push!(rows, N); push!(vals, cp.BCT["α1"])
    end

    return rows, vals
end

# Build the total gammaBC transpose matrix for Kronecker product folding.
# Returns gammaBC_total' (maps reduced → full spectral space) or nothing.
function _build_gammaBC_total(grid::AbstractGrid, var_idx::Int)
    i_active = !(grid.ibasis isa NoBasisArray)
    j_active = !(grid.jbasis isa NoBasisArray)
    k_active = !(grid.kbasis isa NoBasisArray)

    matrices = Matrix{Float64}[]
    has_spline = false

    for (active, dim) in [(i_active, :i), (j_active, :j), (k_active, :k)]
        if !active
            continue
        end
        obj = _get_basis_object(grid, dim, var_idx)
        if obj isa CubicBSpline.Spline1D
            push!(matrices, Matrix(obj.gammaBC'))  # bDim × Minterior
            has_spline = true
        else
            # Use the actual column count of the basis matrix, not spectral_dim,
            # because Chebyshev operator matrices are (zDim, zDim) even when bDim < zDim.
            n = size(_basis_matrix(obj, 0), 2)
            push!(matrices, Matrix{Float64}(I, n, n))
        end
    end

    if !has_spline
        return nothing
    end

    if length(matrices) == 1
        return matrices[1]
    elseif length(matrices) == 2
        return kron(matrices[1], matrices[2])
    else
        return kron(matrices[1], matrices[2], matrices[3])
    end
end

# Build the total ahat vector for R3X inhomogeneous boundary conditions.
# For 1D grids: reads ahat directly from the representative Spline1D.
# For multi-D grids: assembles ahat_total by reading per-mode ahat vectors from
# the ibasis array, since each secondary-dimension mode has its own i-spline
# with potentially different ahat values (boundary data varies across modes).
# Returns nothing if no R3X BCs are present on any spline dimension.
function _build_ahat_total(grid::AbstractGrid, var_idx::Int)
    i_active = !(grid.ibasis isa NoBasisArray)
    j_active = !(grid.jbasis isa NoBasisArray)
    k_active = !(grid.kbasis isa NoBasisArray)

    # Check if any spline dimension has R3X
    has_r3x = false
    if i_active
        obj = _get_basis_object(grid, :i, var_idx)
        if obj isa CubicBSpline.Spline1D && CubicBSpline._has_r3x(obj.params)
            has_r3x = true
        end
    end

    if !has_r3x
        return nothing
    end

    # 1D case: just the single spline's ahat
    if !j_active && !k_active
        return copy(_get_basis_object(grid, :i, var_idx).ahat)
    end

    # Multi-D: assemble from per-mode i-splines
    # ibasis.data layout: [secondary_modes..., var_idx]
    idata = grid.ibasis.data
    b_iDim = _get_basis_object(grid, :i, var_idx).params.bDim

    if j_active && !k_active
        # 2D: RR or RL — ibasis.data[j_mode, var]
        n_modes = size(idata, 1)
        n_j = size(_basis_matrix(_get_basis_object(grid, :j, var_idx), 0), 2)
        ahat_total = zeros(b_iDim * n_j)
        for i in 1:b_iDim
            for j in 1:n_modes
                ahat_total[(i-1)*n_j + j] = idata[j, var_idx].ahat[i]
            end
        end
        return ahat_total
    end

    if !j_active && k_active
        # 2D: RZ — ibasis.data[k_mode, var]
        n_modes = size(idata, 1)
        n_k = size(_basis_matrix(_get_basis_object(grid, :k, var_idx), 0), 2)
        ahat_total = zeros(b_iDim * n_k)
        for i in 1:b_iDim
            for k in 1:n_modes
                ahat_total[(i-1)*n_k + k] = idata[k, var_idx].ahat[i]
            end
        end
        return ahat_total
    end

    # 3D case: ibasis.data[j_mode, k_mode, var]
    n_j_modes = size(idata, 1)
    n_k_modes = size(idata, 2)
    n_j = size(_basis_matrix(_get_basis_object(grid, :j, var_idx), 0), 2)
    n_k = size(_basis_matrix(_get_basis_object(grid, :k, var_idx), 0), 2)
    ahat_total = zeros(b_iDim * n_j * n_k)
    for i in 1:b_iDim
        for j in 1:n_j_modes
            for k in 1:n_k_modes
                ahat_total[(i-1)*n_j*n_k + (j-1)*n_k + k] = idata[j, k, var_idx].ahat[i]
            end
        end
    end
    return ahat_total
end

# Apply all boundary conditions for solve().
# Returns (L_bc, f_bc, gammaBC_total_transpose) where gammaBC_total_transpose
# maps reduced → full spectral space (or nothing if no spline folding).
function _apply_all_bcs(grid::AbstractGrid, L::Matrix{Float64}, f::Vector{Float64}, var_idx::Int, var::String="")
    # Step 1: Build total gammaBC and fold spline BCs
    gammaBC_T = _build_gammaBC_total(grid, var_idx)
    if gammaBC_T !== nothing
        L = L * gammaBC_T
    end

    # Step 2: Build the full evaluation matrix (after gammaBC folding) for BC rows.
    # For Dirichlet BCs, boundary rows of L are replaced with the corresponding
    # rows of the evaluation matrix (so the constraint becomes: u(boundary) = bc_val).
    # For Neumann BCs, the derivative matrix row is used instead.
    # The evaluation matrix in the (possibly reduced) spectral space is:
    M_eval_raw = assemble_operator(grid, [OperatorTerm(0, 0, 0, nothing)], var)
    if gammaBC_T !== nothing
        M_eval = M_eval_raw * gammaBC_T
    else
        M_eval = M_eval_raw
    end

    # Step 3: Identify Chebyshev dimensions and their boundary rows
    i_active = !(grid.ibasis isa NoBasisArray)
    j_active = !(grid.jbasis isa NoBasisArray)
    k_active = !(grid.kbasis isa NoBasisArray)

    # Collect physical dimension sizes for active dims (for row indexing)
    phys_dims = Int[]
    active_dims = Symbol[]
    cheby_info = Dict{Int, Tuple{Vector{Int}, Vector{Float64}, Chebyshev.Chebyshev1D}}()

    dim_idx = 0
    for (active, dim) in [(i_active, :i), (j_active, :j), (k_active, :k)]
        if !active
            continue
        end
        dim_idx += 1
        obj = _get_basis_object(grid, dim, var_idx)
        push!(active_dims, dim)
        push!(phys_dims, Springsteel.physical_dim(obj))
        if obj isa Chebyshev.Chebyshev1D
            bc_rows, bc_vals = _chebyshev_bc_info(obj)
            if !isempty(bc_rows)
                cheby_info[dim_idx] = (bc_rows, bc_vals, obj)
            end
        end
    end

    ndim = length(active_dims)

    # Step 4: For each Chebyshev dimension with BCs, find boundary rows and
    # replace them with evaluation matrix rows + set RHS to BC value.
    # For Neumann BCs, we need the derivative evaluation matrix instead;
    # build it on demand.
    for (didx, (bc_local_rows, bc_vals, obj)) in cheby_info
        cp = obj.params
        # Build derivative evaluation matrix if needed for Neumann BCs
        need_neumann = haskey(cp.BCB, "α1") || haskey(cp.BCT, "α1")
        if need_neumann
            M_deriv_raw = _build_single_dim_deriv_matrix(grid, active_dims, didx, phys_dims, var)
            if gammaBC_T !== nothing
                M_deriv = M_deriv_raw * gammaBC_T
            else
                M_deriv = M_deriv_raw
            end
        end

        for (bi, local_row) in enumerate(bc_local_rows)
            bc_val = bc_vals[bi]
            # Determine if this is a Neumann BC (row 1 = BCB/zmin, row N = BCT/zmax)
            is_neumann = (local_row == 1 && haskey(cp.BCB, "α1")) ||
                         (local_row == cp.zDim && haskey(cp.BCT, "α1"))

            # Find all global rows corresponding to this local boundary
            _replace_boundary_rows!(L, f, M_eval, is_neumann ? M_deriv : M_eval,
                                     didx, local_row, bc_val, phys_dims, is_neumann)
        end
    end

    return L, f, gammaBC_T
end

# Build a derivative evaluation matrix for a specific dimension in the Kronecker product.
function _build_single_dim_deriv_matrix(grid::AbstractGrid, active_dims::Vector{Symbol},
                                         deriv_dim::Int, phys_dims::Vector{Int},
                                         var::String="")
    ndim = length(active_dims)
    matrices = Matrix{Float64}[]

    for d in 1:ndim
        dim = active_dims[d]
        if d == deriv_dim
            push!(matrices, operator_matrix(grid, dim, 1, var))
        else
            push!(matrices, operator_matrix(grid, dim, 0, var))
        end
    end

    if ndim == 1
        return matrices[1]
    elseif ndim == 2
        return kron(matrices[1], matrices[2])
    else
        return kron(matrices[1], matrices[2], matrices[3])
    end
end

# Replace all global rows that correspond to a local boundary in a specific dimension.
function _replace_boundary_rows!(L::Matrix{Float64}, f::Vector{Float64},
                                  M_eval::Matrix{Float64}, M_bc::Matrix{Float64},
                                  bc_dim::Int, local_row::Int, bc_val::Float64,
                                  phys_dims::Vector{Int}, is_neumann::Bool)
    ndim = length(phys_dims)

    # Iterate over all combinations of indices in non-BC dimensions
    ranges = [d == bc_dim ? [local_row] : collect(1:phys_dims[d]) for d in 1:ndim]

    for idx in Iterators.product(ranges...)
        # Compute global row index (rightmost dimension varies fastest)
        global_row = 0
        stride = 1
        for d in ndim:-1:1
            global_row += (idx[d] - 1) * stride
            stride *= phys_dims[d]
        end
        global_row += 1  # 1-based

        # Replace the row with the appropriate constraint row
        if is_neumann
            L[global_row, :] .= M_bc[global_row, :]
        else
            L[global_row, :] .= M_eval[global_row, :]
        end
        f[global_row] = bc_val
    end
end

# ────────────────────────────────────────────────────────────────────────────
# solve — non-mutating entry point for any linear backend
# ────────────────────────────────────────────────────────────────────────────

"""
    solve(prob::SpringsteelProblem) -> SpringsteelSolution

Non-mutating twin of [`solve!`](@ref). Runs the cached workspace compute
path, writes the result into a fresh [`SpringsteelGrid`](@ref) narrowed to
just the solved field(s), and returns a [`SpringsteelSolution`](@ref) that
owns that grid. The grid originally passed to `SpringsteelProblem` is left
untouched.

Works on any linear backend that `solve!` supports (`LocalLinearBackend`,
`SparseLinearBackend`, `KrylovLinearBackend`). For `OptimizationBackend` a
separate method in the `Optimization.jl` package extension applies.

Use `solve` when you want the solution as a return value bundled with its
own grid — for single-shot solves, for piping into downstream operations
like `gridTransform!` / `evaluate_unstructured` / `write_grid`, or when
preserving the original grid state matters. Use `solve!` when you're
iterating in a time-stepping loop and want the fastest path with in-place
updates to `grid.physical`.

# Returns
A [`SpringsteelSolution`](@ref) whose `physical` and `coefficients` are
views into the owned solution grid.

See also: [`solve!`](@ref), [`SpringsteelProblem`](@ref), [`SpringsteelSolution`](@ref)
"""
function solve(prob::SpringsteelProblem{<:AbstractLinearBackend})
    ws = prob.workspace
    ws === nothing && throw(ArgumentError(
        "solve requires a linear-backend problem with a workspace (use " *
        "SpringsteelProblem(grid, L*u => rhs) or the kwarg-form " *
        "SpringsteelProblem(grid; operator=L, rhs=f))"))
    return _solve_to_snapshot(prob, ws)
end

# `_solve_to_snapshot(prob, ws)` method bodies are defined in
# solver_problem.jl, next to the workspace type definitions.

# ────────────────────────────────────────────────────────────────────────────
# solver_problem.jl — S2 of the solver refactor: stateful linear problems
# ────────────────────────────────────────────────────────────────────────────
#
# Introduces `SpringsteelField` / `Field` and a Pair-based constructor for
# `SpringsteelProblem` that caches the entire linear-solve workspace
# (lowered operator, gammaBC fold, evaluation matrix, boundary row indices,
# ahat correction, and LU factorisation) so that repeated `solve!` calls do
# no allocation beyond the backsolve itself.
#
# Old kwarg-constructed `SpringsteelProblem(grid; operator=L, rhs=f)` path
# is unchanged — this is strictly additive. See plan_solver_refactor.md §4
# phase S2 for the full design and the allocation hotspots this targets.

import Krylov

# ────────────────────────────────────────────────────────────────────────────
# SpringsteelField
# ────────────────────────────────────────────────────────────────────────────

"""
    SpringsteelField(grid, var_name::AbstractString)
    SpringsteelField(var_name::AbstractString, var_idx::Int)

Handle for a grid variable that can stand in as the unknown in operator
algebra (`L * u => rhs`). `Field` is an exported alias.

Holds only the variable's name and its integer slot index — no grid
reference. The primary constructor takes a grid to resolve the slot from
`grid.params.vars`, but the resulting `Field` is independent of that grid
and can be reused across grids with the same variable schema.

Boundary conditions come from the grid's per-variable BC spec at problem
construction time, not from the `Field` itself.
"""
struct SpringsteelField
    var_name::String
    var_idx::Int
end

function SpringsteelField(grid::AbstractGrid, var_name::AbstractString)
    vname = String(var_name)
    haskey(grid.params.vars, vname) ||
        throw(ArgumentError("Variable `$(vname)` not found in grid.params.vars"))
    return SpringsteelField(vname, grid.params.vars[vname])
end

const Field = SpringsteelField

# ────────────────────────────────────────────────────────────────────────────
# Backend resolution + dispatched factorisation (S4a)
# ────────────────────────────────────────────────────────────────────────────

# Auto-pick rule: pure Chebyshev grids (Z, ZZ, ZZZ) have no structural
# sparsity worth exploiting, so dense LU wins. Everything else (anything
# involving a B-spline or Fourier dimension) defaults to sparse LU so the
# factor stays small as nc grows.
function _default_backend(grid::AbstractGrid)
    all_cheb = true
    for ba in (grid.ibasis, grid.jbasis, grid.kbasis)
        ba isa NoBasisArray && continue
        if !(ba isa ChebyshevBasisArray)
            all_cheb = false
            break
        end
    end
    return all_cheb ? LocalLinearBackend() : SparseLinearBackend()
end

_resolve_backend(b::AbstractSolverBackend, ::AbstractGrid) = b
function _resolve_backend(b::Symbol, grid::AbstractGrid)
    b === :auto   && return _default_backend(grid)
    b === :dense  && return LocalLinearBackend()
    b === :sparse && return SparseLinearBackend()
    b === :krylov && return KrylovLinearBackend()
    throw(ArgumentError(
        "Unknown backend symbol `:$b` — use :auto, :dense, :sparse, or :krylov"))
end

# Wrapper that holds the sparse operator plus any preconditioner the user
# attached to the backend, so the backsolve dispatch knows which Krylov
# method to call and what extras to pass through.
struct KrylovOp
    A::SparseMatrixCSC{Float64, Int}
    preconditioner::Any
end

# Dispatched factorisation. Operators from the spline path are typically
# rectangular (n_phys > reduced_bDim), so we fall back to QR in both the
# dense and sparse cases when the matrix isn't square.
function _factorize_operator(::LocalLinearBackend, L::Matrix{Float64})
    return size(L, 1) == size(L, 2) ? factorize(L) : qr(L)
end
function _factorize_operator(::SparseLinearBackend, L::Matrix{Float64})
    S = sparse(L)
    return size(S, 1) == size(S, 2) ? lu(S) : qr(S)
end
function _factorize_operator(b::KrylovLinearBackend, L::Matrix{Float64})
    A = sparse(L)
    return KrylovOp(A, _resolve_preconditioner(b.preconditioner, A))
end

"""
    _resolve_preconditioner(spec, A) -> preconditioner

Turn a user-supplied preconditioner spec into the concrete object that gets
handed to Krylov as the `M=` (left preconditioner) kwarg. Accepted forms:

- `nothing`           → no preconditioner
- `:default` / `:diag` → diagonal preconditioner built from `diag(A)`, with
                         a small guard so zero diagonals don't explode
- an `AbstractMatrix` → used as-is (Krylov applies it via `M * v`)
- a callable          → used as-is (Krylov calls `M(v)` or similar — the
                         caller is responsible for making it a valid
                         Krylov `M` operand)

S4c only plumbs the value through; Krylov.jl validates the object at
solve time.
"""
_resolve_preconditioner(::Nothing, ::SparseMatrixCSC) = nothing

function _resolve_preconditioner(spec::Symbol, A::SparseMatrixCSC)
    if spec === :diag || spec === :default
        return _build_diag_preconditioner(A)
    end
    throw(ArgumentError(
        "Unknown preconditioner symbol `:$spec` — use :diag, :default, " *
        "or pass a matrix/callable"))
end

_resolve_preconditioner(m::AbstractMatrix, ::SparseMatrixCSC) = m
_resolve_preconditioner(f,  ::SparseMatrixCSC) = f   # callable pass-through

"""
    _build_diag_preconditioner(A) -> Diagonal

Build a diagonal left-preconditioner approximating `A^{-1}` by inverting the
absolute diagonal of `A` with a small zero-guard. Good enough for gently
ill-conditioned problems; user can override with their own operator for
anything harder.
"""
# Fold a `preconditioner=` kwarg into the backend instance. `:default` is
# the sentinel for "user didn't specify" — it preserves whatever the backend
# was constructed with. An explicit `nothing` / matrix / callable / symbol
# overrides the backend's preconditioner (for Krylov) or errors (for direct
# backends, since they can't use a preconditioner).
#
# Note: we intentionally do NOT wire a diagonal preconditioner as the Krylov
# default. BC-augmented systems have wildly varying row magnitudes (interior
# ~N², boundary ~O(1)) so naive 1/diag can hurt gmres convergence. Users who
# want it can opt in explicitly with `preconditioner=:diag`.
function _attach_preconditioner(b::AbstractSolverBackend, p)
    p === :default && return b
    p === nothing  && return b
    throw(ArgumentError(
        "preconditioner kwarg is only meaningful for Krylov-based backends; " *
        "got $(typeof(b)). Switch to backend=:krylov or pass preconditioner=nothing."))
end
function _attach_preconditioner(b::KrylovLinearBackend, p)
    p === :default && return b                    # keep backend's own setting
    return KrylovLinearBackend(p)
end

function _build_diag_preconditioner(A::SparseMatrixCSC)
    d = diag(A)
    inv_d = similar(d)
    @inbounds for i in eachindex(d)
        inv_d[i] = abs(d[i]) > 1e-14 ? 1.0 / d[i] : 1.0
    end
    return Diagonal(inv_d)
end

# Allocation-free backsolve with type-dispatched handling of UMFPACK, whose
# 3-arg `ldiv!(x, F, b)` has dimension checks that conflict with our workspace
# layout. For sparse we copy f → x and do the 2-arg in-place solve.
_backsolve!(x::AbstractVector{Float64}, F, f::AbstractVector{Float64}) =
    ldiv!(x, F, f)

function _backsolve!(x::AbstractVector{Float64},
                     F::SparseArrays.UMFPACK.UmfpackLU,
                     f::AbstractVector{Float64})
    copyto!(x, f)
    ldiv!(F, x)
    return x
end

# SuiteSparseQR (rectangular sparse) has no in-place ldiv! → fall back to
# the allocating `\` operator. Only used for overdetermined spline systems
# that can't be factorised as square LU.
function _backsolve!(x::AbstractVector{Float64},
                     F::SparseArrays.SPQR.QRSparse,
                     f::AbstractVector{Float64})
    copyto!(x, F \ f)
    return x
end

# Krylov backsolve: dispatch on shape (gmres square, lsmr rectangular) and
# pass the cached preconditioner through as the `M` kwarg.
function _backsolve!(x::AbstractVector{Float64},
                     op::KrylovOp,
                     f::AbstractVector{Float64})
    m, n = size(op.A)
    if m == n
        xk, _stats = op.preconditioner === nothing ?
            Krylov.gmres(op.A, f) :
            Krylov.gmres(op.A, f; M = op.preconditioner)
    else
        xk, _stats = op.preconditioner === nothing ?
            Krylov.lsmr(op.A, f) :
            Krylov.lsmr(op.A, f; M = op.preconditioner)
    end
    copyto!(x, xk)
    return x
end

# ────────────────────────────────────────────────────────────────────────────
# TypedOperator — `OperatorExpr * SpringsteelField`
# ────────────────────────────────────────────────────────────────────────────

"""
    TypedTerm(expr, field)

A single `OperatorExpr` acting on a single `SpringsteelField`. The
atomic building block of a `TypedOperator`.
"""
struct TypedTerm
    expr::OperatorExpr
    field::SpringsteelField
end

"""
    TypedOperator(terms)

A sum of `TypedTerm`s — the left-hand side of one equation. Produced by
multiplying an operator-algebra expression by a `SpringsteelField` and
then combining via `+`/`-`. For single-variable problems it holds one
term; for block systems each term can bind to a different field.
"""
struct TypedOperator
    terms::Vector{TypedTerm}
end

# Single-term constructors
Base.:*(e::OperatorExpr, f::SpringsteelField) = TypedOperator([TypedTerm(e, f)])
Base.:*(m::DerivMono,    f::SpringsteelField) = TypedOperator([TypedTerm(_as_expr(m), f)])
Base.:*(s::ScaledMono,   f::SpringsteelField) = TypedOperator([TypedTerm(_as_expr(s), f)])

# Combine LHS pieces
Base.:+(a::TypedOperator, b::TypedOperator) = TypedOperator(vcat(a.terms, b.terms))
Base.:-(a::TypedOperator) =
    TypedOperator([TypedTerm(-1.0 * t.expr, t.field) for t in a.terms])
Base.:-(a::TypedOperator, b::TypedOperator) = a + (-b)

# ────────────────────────────────────────────────────────────────────────────
# LocalLinearWorkspace — cached state for repeated `solve!` calls
# ────────────────────────────────────────────────────────────────────────────

mutable struct LocalLinearWorkspace
    field::SpringsteelField
    rhs_source::Any                       # Symbol / Vector{Float64} / Number
    n_phys::Int

    L_base::Matrix{Float64}               # unfolded operator (pre-BC, pre-gamma)
    gammaBC_T::Union{Nothing, Matrix{Float64}}
    M_eval_raw::Matrix{Float64}           # full-spectral → physical
    M_eval::Matrix{Float64}               # reduced-spectral → physical (post-gamma)

    ahat_total::Union{Nothing, Vector{Float64}}
    ahat_correction::Union{Nothing, Vector{Float64}}   # L_base * ahat_total
    phys_ahat::Union{Nothing, Vector{Float64}}         # M_eval_raw * ahat_total

    bc_rows::Vector{Int}                  # global BC row indices (post-gamma)
    bc_values::Vector{Float64}            # matching BC values

    L_bc::Matrix{Float64}                 # post-gamma, post-BC operator
    factorization::Any                    # cached LU / QR

    f_work::Vector{Float64}               # scratch RHS (size = rows of L_bc)
    a_raw::Vector{Float64}                # scratch reduced-spectral
    phys_out::Vector{Float64}             # scratch physical output
end

# ────────────────────────────────────────────────────────────────────────────
# Pair-based SpringsteelProblem constructor
# ────────────────────────────────────────────────────────────────────────────

"""
    SpringsteelProblem(grid, eq::Pair{TypedOperator,<:Any}; backend=LocalLinearBackend())

Stateful linear-problem constructor. `eq` is `L*u => rhs` where:

- `L` is an `OperatorExpr` built from the operator algebra DSL
- `u` is a `SpringsteelField` (carries the unknown's grid + variable slot)
- `rhs` is a `Symbol` (pulls `grid.physical[:, rhs_idx, 1]` at each solve),
  a `Vector{Float64}` (literal), or a `Number` (constant)

BCs are pulled from the grid's per-variable BC spec. Builds and caches the
lowered operator, gammaBC fold, eval matrix, BC rows, ahat correction, and
LU factorisation so that repeated `solve!` calls only pay for the backsolve.

Use `solve!(prob)` to run the solver and write into `grid.physical[:, u_idx, 1]`.
"""
function SpringsteelProblem(grid::AbstractGrid,
                             eq::Pair{TypedOperator, <:Any};
                             backend::Union{AbstractSolverBackend, Symbol} = :auto,
                             preconditioner = :default)
    top  = eq.first
    rhs0 = eq.second
    isempty(top.terms) && throw(ArgumentError("Empty TypedOperator LHS"))

    fields = unique([t.field for t in top.terms])
    length(fields) == 1 || throw(ArgumentError(
        "Single-equation constructor requires all TypedTerms to bind the " *
        "same field — got $(length(fields)). For block systems pass a " *
        "Vector{Pair} instead."))
    field = fields[1]
    haskey(grid.params.vars, field.var_name) || throw(ArgumentError(
        "Field variable `$(field.var_name)` not present in grid.params.vars"))

    backend_inst = _resolve_backend(backend, grid)
    backend_inst = _attach_preconditioner(backend_inst, preconditioner)

    combined = top.terms[1].expr
    for k in 2:length(top.terms)
        combined = combined + top.terms[k].expr
    end
    lowered = _lower(combined, grid)
    ws = _build_local_linear_workspace(grid, field, lowered, rhs0, backend_inst)

    return SpringsteelProblem(grid, backend_inst, ws, nothing, Dict{String, Any}())
end

"""
    SpringsteelProblem(grid, eqs::Vector{<:Pair{TypedOperator,<:Any}}; backend)

Block-system constructor (S3). Each element of `eqs` is one equation
`LHS_i => RHS_i` where `LHS_i` is a sum of `TypedTerm`s whose fields
identify the unknowns participating in that row. The set of unique fields
across all equations defines the unknown vector; `length(eqs)` must equal
the number of unknowns.

BCs come from each variable's grid BC spec. The resulting operator is a
block matrix `L[i,j]` where `L[i,j]` accumulates terms from equation `i`
that bind to unknown `j`. `solve!(prob)` writes each variable back into
`grid.physical[:, var_idx, 1]`.
"""
function SpringsteelProblem(grid::AbstractGrid,
                             eqs::Vector{<:Pair{TypedOperator,<:Any}};
                             backend::Union{AbstractSolverBackend, Symbol} = :auto,
                             preconditioner = :default)
    isempty(eqs) && throw(ArgumentError("Empty equation list"))
    backend_inst = _resolve_backend(backend, grid)
    backend_inst = _attach_preconditioner(backend_inst, preconditioner)

    # Collect unique fields across all equations, in first-appearance order.
    fields = SpringsteelField[]
    for eq in eqs, t in eq.first.terms
        haskey(grid.params.vars, t.field.var_name) || throw(ArgumentError(
            "Field variable `$(t.field.var_name)` not present in grid.params.vars"))
        if !any(f -> f.var_idx == t.field.var_idx, fields)
            push!(fields, t.field)
        end
    end

    length(eqs) == length(fields) || throw(ArgumentError(
        "Block system requires #equations == #unknowns " *
        "(got $(length(eqs)) eqs, $(length(fields)) unknowns)"))

    ws = _build_block_linear_workspace(grid, fields, eqs, backend_inst)

    return SpringsteelProblem(grid, backend_inst, ws, nothing, Dict{String, Any}())
end

# ────────────────────────────────────────────────────────────────────────────
# Workspace construction
# ────────────────────────────────────────────────────────────────────────────

function _build_local_linear_workspace(grid::AbstractGrid,
                                        field::SpringsteelField,
                                        terms::Vector{OperatorTerm},
                                        rhs_source,
                                        backend::AbstractSolverBackend)
    L_base = assemble_operator(grid, terms, field.var_name)
    return _build_local_linear_workspace(grid, field, L_base, rhs_source, backend)
end

# Variant that accepts a pre-assembled operator matrix. Used by the legacy
# kwarg-form `SpringsteelProblem(grid; operator=L, rhs=f)` constructor, which
# hands in a user-built matrix rather than an operator-algebra expression.
# Post-assembly steps (ahat, gammaBC fold, BC row detection, factorisation,
# scratch allocation) are identical to the terms-based path.
function _build_local_linear_workspace(grid::AbstractGrid,
                                        field::SpringsteelField,
                                        L_base::Matrix{Float64},
                                        rhs_source,
                                        backend::AbstractSolverBackend)
    var = field.var_name
    var_idx = field.var_idx

    n_phys = size(L_base, 1)

    # R3X ahat background
    ahat_total = _build_ahat_total(grid, var_idx)
    ahat_correction = ahat_total === nothing ? nothing : L_base * ahat_total

    # gammaBC fold (spline side)
    gammaBC_T = _build_gammaBC_total(grid, var_idx)

    # Full-spectral evaluation matrix
    M_eval_raw = assemble_operator(grid, [OperatorTerm(0, 0, 0, nothing)], var)
    M_eval = gammaBC_T === nothing ? M_eval_raw : M_eval_raw * gammaBC_T
    phys_ahat = ahat_total === nothing ? nothing : M_eval_raw * ahat_total

    # Apply BCs via existing helper. Start from a fresh L/f so _apply_all_bcs
    # modifies our working copies.
    f_tmp = zeros(n_phys)
    if ahat_correction !== nothing
        f_tmp .-= ahat_correction
    end
    L_tmp = copy(L_base)
    L_bc, f_bc0, _ = _apply_all_bcs(grid, L_tmp, f_tmp, var_idx, var)

    # Recover BC row indices + values by comparing the operator row before
    # and after `_apply_all_bcs`. Operator-side detection catches homogeneous
    # Dirichlet / Neumann BCs that an f-diff baseline misses (the f-value at
    # a homogeneous BC row equals the baseline, so f-diff returns zero rows).
    # BC values themselves come from f_bc0[r], which _apply_all_bcs writes as
    # `bc_val - ahat_correction[r]` when ahat is present.
    bc_rows  = Int[]
    bc_values = Float64[]
    @inbounds for r in 1:n_phys
        if _row_is_bc(L_bc, L_base, r, gammaBC_T)
            push!(bc_rows, r)
            push!(bc_values, f_bc0[r])
        end
    end

    # Cache factorisation (dispatched on backend)
    F = _factorize_operator(backend, L_bc)

    # Scratch
    f_work   = zeros(size(L_bc, 1))
    a_raw    = zeros(size(L_bc, 2))
    phys_out = zeros(n_phys)

    return LocalLinearWorkspace(field, rhs_source, n_phys,
                                 L_base, gammaBC_T, M_eval_raw, M_eval,
                                 ahat_total, ahat_correction, phys_ahat,
                                 bc_rows, bc_values,
                                 L_bc, F,
                                 f_work, a_raw, phys_out)
end

# ────────────────────────────────────────────────────────────────────────────
# solve! — the fast path
# ────────────────────────────────────────────────────────────────────────────

"""
    solve!(prob::SpringsteelProblem) -> SpringsteelProblem

Stateful solve that reuses `prob.workspace`. Refreshes each RHS (from the
grid if given as a `Symbol`), applies cached ahat/BC corrections, backsolves
against the cached factorisation, and writes each variable's physical-space
solution into `prob.grid.physical[:, var_idx, 1]`. Returns `prob`.
"""
function solve!(prob::SpringsteelProblem)
    ws = prob.workspace
    ws === nothing && throw(ArgumentError(
        "solve! requires a stateful SpringsteelProblem built via the " *
        "Pair-based constructor (SpringsteelProblem(grid, L*u => rhs))"))
    if ws isa LocalLinearWorkspace
        _solve_local_linear!(prob, ws)
    elseif ws isa BlockLinearWorkspace
        _solve_block_linear!(prob, ws)
    else
        error("Unknown workspace type: $(typeof(ws))")
    end
    return prob
end

# The hot path: stateful solve using cached workspace.
function _solve_local_linear!(prob::SpringsteelProblem, ws::LocalLinearWorkspace)
    _compute_local_linear!(prob, ws)
    _writeback_local_linear!(prob.grid, ws)
    return prob
end

# Steps 1-5: compute into `ws.a_raw` and `ws.phys_out`. No writeback.
# Shared by both `solve!` (mutating) and `solve` (non-mutating, writes the
# result into a narrowed solution grid instead of the source grid).
function _compute_local_linear!(prob::SpringsteelProblem, ws::LocalLinearWorkspace)
    grid = prob.grid
    n_phys = ws.n_phys

    # 1) Fetch RHS into f_work (pre-BC baseline)
    _fetch_rhs!(ws.f_work, ws.rhs_source, grid, n_phys)

    # 2) R3X ahat correction
    if ws.ahat_correction !== nothing
        @inbounds @simd for r in 1:length(ws.f_work)
            ws.f_work[r] -= ws.ahat_correction[r]
        end
    end

    # 3) Overwrite BC rows with cached BC values
    @inbounds for (idx, r) in enumerate(ws.bc_rows)
        ws.f_work[r] = ws.bc_values[idx]
    end

    # 4) Backsolve (this is the only unavoidable cost)
    _backsolve!(ws.a_raw, ws.factorization, ws.f_work)

    # 5) Physical recovery: phys = M_eval * a_raw + phys_ahat
    mul!(ws.phys_out, ws.M_eval, ws.a_raw)
    if ws.phys_ahat !== nothing
        @inbounds @simd for r in 1:n_phys
            ws.phys_out[r] += ws.phys_ahat[r]
        end
    end
    return nothing
end

# Step 6: write `ws.phys_out` into `target_grid.physical[:, var_idx, 1]`.
# `target_grid` may be the source grid (mutating solve!) or a narrowed copy.
# `var_idx` defaults to the workspace's original field index, which is valid
# for the source grid; callers writing into a narrowed copy should pass the
# variable's slot in that copy explicitly.
function _writeback_local_linear!(target_grid::AbstractGrid,
                                   ws::LocalLinearWorkspace,
                                   var_idx::Int = ws.field.var_idx)
    n_phys = ws.n_phys
    phys = target_grid.physical
    @inbounds for r in 1:n_phys
        phys[r, var_idx, 1] = ws.phys_out[r]
    end
    return nothing
end

# ────────────────────────────────────────────────────────────────────────────
# RHS resolution — S5 will generalise this
# ────────────────────────────────────────────────────────────────────────────

function _fetch_rhs!(dst::AbstractVector{Float64}, src::Symbol, grid::AbstractGrid, n_phys::Int)
    name = String(src)
    haskey(grid.params.vars, name) || throw(ArgumentError(
        "RHS symbol `:$name` is not a variable on this grid"))
    vi = grid.params.vars[name]
    phys = grid.physical
    @inbounds for r in 1:n_phys
        dst[r] = phys[r, vi, 1]
    end
    return dst
end

function _fetch_rhs!(dst::AbstractVector{Float64}, src::AbstractVector{<:Real},
                     ::AbstractGrid, n_phys::Int)
    length(src) == n_phys || throw(ArgumentError(
        "RHS vector length $(length(src)) does not match physical size $n_phys"))
    @inbounds for r in 1:n_phys
        dst[r] = src[r]
    end
    return dst
end

function _fetch_rhs!(dst::AbstractVector{Float64}, src::Real, ::AbstractGrid, n_phys::Int)
    fill!(dst, Float64(src))
    return dst
end

# ────────────────────────────────────────────────────────────────────────────
# S3 — Block linear workspace for multi-variable systems
# ────────────────────────────────────────────────────────────────────────────

mutable struct BlockLinearWorkspace
    fields::Vector{SpringsteelField}
    rhs_sources::Vector{Any}
    n_phys::Int                              # per-equation row count
    n_eq::Int                                # number of equations (= #unknowns)
    col_offsets::Vector{Int}                 # 0-based col offsets into the full matrix, length n_eq+1
    gammaBC_Ts::Vector{Union{Nothing,Matrix{Float64}}}
    M_eval_raws::Vector{Matrix{Float64}}     # n_phys × n_spec_full_j (one per unknown)
    M_evals::Vector{Matrix{Float64}}         # n_phys × n_reduced_j (post-gamma)
    bc_rows_global::Vector{Int}              # row indices in L_full that are BC rows
    bc_values::Vector{Float64}               # matching BC values
    L_full::Matrix{Float64}                  # (n_eq*n_phys) × sum(n_reduced_j)
    factorization::Any
    f_work::Vector{Float64}
    a_raw::Vector{Float64}                   # full reduced-spectral stack
    phys_out::Matrix{Float64}                # n_phys × n_eq
end

function _build_block_linear_workspace(grid::AbstractGrid,
                                        fields::Vector{SpringsteelField},
                                        eqs::Vector{<:Pair{TypedOperator,<:Any}},
                                        backend::AbstractSolverBackend)
    n_eq = length(eqs)

    # Per-unknown basis data
    gammaBC_Ts    = Vector{Union{Nothing,Matrix{Float64}}}(undef, n_eq)
    M_eval_raws   = Vector{Matrix{Float64}}(undef, n_eq)
    M_evals       = Vector{Matrix{Float64}}(undef, n_eq)
    n_reduced_js  = Vector{Int}(undef, n_eq)

    # Reject R3X for now — multi-var path doesn't plumb ahat yet.
    for f in fields
        _build_ahat_total(grid, f.var_idx) === nothing || throw(ArgumentError(
            "Block-system S3 does not yet support R3X inhomogeneous BCs " *
            "(variable `$(f.var_name)` has non-zero ahat). Use the " *
            "single-variable path or wait for S4."))
    end

    # First pass: per-unknown basis stuff, sanity-check n_phys
    n_phys = -1
    for (j, fld) in enumerate(fields)
        var = fld.var_name
        M_raw = assemble_operator(grid, [OperatorTerm(0, 0, 0, nothing)], var)
        M_eval_raws[j] = M_raw
        if n_phys < 0
            n_phys = size(M_raw, 1)
        else
            size(M_raw, 1) == n_phys || throw(ArgumentError(
                "Unknowns must share the same physical row count " *
                "(got $n_phys vs $(size(M_raw,1)) for `$var`)"))
        end
        gT = _build_gammaBC_total(grid, fld.var_idx)
        gammaBC_Ts[j] = gT
        M_evals[j]   = gT === nothing ? M_raw : M_raw * gT
        n_reduced_js[j] = size(M_evals[j], 2)
    end

    # Column offsets (0-based)
    col_offsets = zeros(Int, n_eq + 1)
    for j in 1:n_eq
        col_offsets[j+1] = col_offsets[j] + n_reduced_js[j]
    end
    n_cols_total = col_offsets[end]
    n_rows_total = n_eq * n_phys

    # Allocate block matrix and fill cell-by-cell
    L_full = zeros(n_rows_total, n_cols_total)

    for (i, eq) in enumerate(eqs)
        row_offset = (i - 1) * n_phys
        for j in 1:n_eq
            terms = OperatorTerm[]
            for t in eq.first.terms
                if t.field.var_idx == fields[j].var_idx
                    append!(terms, _lower(t.expr, grid))
                end
            end
            isempty(terms) && continue   # zero block
            L_ij_raw = assemble_operator(grid, terms, fields[j].var_name)
            gT = gammaBC_Ts[j]
            L_ij = gT === nothing ? L_ij_raw : L_ij_raw * gT

            cols = (col_offsets[j] + 1):col_offsets[j+1]
            rows = (row_offset + 1):(row_offset + n_phys)
            @views L_full[rows, cols] .+= L_ij
        end
    end

    # Collect BC rows + values per unknown. We re-use _apply_all_bcs on the
    # diagonal cell to extract which rows got replaced and what values were
    # placed there (matching the S2 technique). The replacement matrix (M_eval
    # or M_deriv) is reconstructed per-row into the correct column block,
    # with the remaining columns zeroed.
    bc_rows_global = Int[]
    bc_values      = Float64[]

    for j in 1:n_eq
        fld = fields[j]
        var = fld.var_name
        # Build a fresh diagonal cell (raw) and zero RHS, then apply BCs.
        diag_terms = OperatorTerm[]
        for t in eqs[j].first.terms
            if t.field.var_idx == fld.var_idx
                append!(diag_terms, _lower(t.expr, grid))
            end
        end
        # Diagonal must be non-empty, else BC constraints can't be placed.
        isempty(diag_terms) && throw(ArgumentError(
            "Equation $j has no term in unknown `$(var)` — cannot place BCs " *
            "for this variable. Reorder equations so eq i targets unknown i."))

        L_jj_raw = assemble_operator(grid, diag_terms, var)
        f_zero = zeros(n_phys)
        L_jj_bc, f_jj_bc, _ =
            _apply_all_bcs(grid, copy(L_jj_raw), f_zero, fld.var_idx, var)

        # f_jj_bc rows that changed from 0 are BC rows.
        for r in 1:n_phys
            if f_jj_bc[r] != 0.0 || _row_is_bc(L_jj_bc, L_jj_raw, r, gammaBC_Ts[j])
                # Overwrite the corresponding row in L_full. The correct
                # replacement row is L_jj_bc[r, :] in the j'th column block,
                # zeros elsewhere.
                global_row = (j - 1) * n_phys + r
                @views L_full[global_row, :] .= 0.0
                cols = (col_offsets[j] + 1):col_offsets[j+1]
                @views L_full[global_row, cols] .= L_jj_bc[r, :]
                push!(bc_rows_global, global_row)
                push!(bc_values, f_jj_bc[r])
            end
        end
    end

    # Square-system check
    size(L_full, 1) == size(L_full, 2) || throw(ArgumentError(
        "Block operator is not square (got $(size(L_full))). Each unknown's " *
        "reduced spectral dimension must sum to n_eq*n_phys."))

    F = _factorize_operator(backend, L_full)

    # Scratch
    f_work   = zeros(n_rows_total)
    a_raw    = zeros(n_cols_total)
    phys_out = zeros(n_phys, n_eq)

    rhs_sources = Any[eq.second for eq in eqs]

    return BlockLinearWorkspace(fields, rhs_sources, n_phys, n_eq, col_offsets,
                                 gammaBC_Ts, M_eval_raws, M_evals,
                                 bc_rows_global, bc_values,
                                 L_full, F,
                                 f_work, a_raw, phys_out)
end

# Detect BC rows whose RHS happened to be zero (e.g. homogeneous Dirichlet).
# In that case the f-difference trick misses them, but the L row will have
# been replaced by an M_eval row that generally disagrees with the folded
# raw-row version. Compares the row of the BC'd operator against the row of
# `L_jj_raw * gammaBC_T` (or raw if no gamma).
function _row_is_bc(L_bc::Matrix{Float64}, L_raw::Matrix{Float64}, r::Int,
                    gammaBC_T::Union{Nothing,Matrix{Float64}})
    if gammaBC_T === nothing
        @views return L_bc[r, :] != L_raw[r, :]
    else
        # Compare L_bc[r, :] against (L_raw * gammaBC_T)[r, :] = L_raw[r, :] * gammaBC_T
        @views raw_folded = L_raw[r, :]' * gammaBC_T
        @views return vec(raw_folded) != L_bc[r, :]
    end
end

function _solve_block_linear!(prob::SpringsteelProblem, ws::BlockLinearWorkspace)
    _compute_block_linear!(prob, ws)
    _writeback_block_linear!(prob.grid, ws)
    return prob
end

# Steps 1-3 + per-unknown physical recovery into `ws.phys_out`. No writeback.
function _compute_block_linear!(prob::SpringsteelProblem, ws::BlockLinearWorkspace)
    grid   = prob.grid
    n_phys = ws.n_phys
    n_eq   = ws.n_eq

    # 1) Fill f_work with each equation's RHS
    for i in 1:n_eq
        @views eq_slice = ws.f_work[((i - 1) * n_phys + 1):(i * n_phys)]
        _fetch_rhs!(eq_slice, ws.rhs_sources[i], grid, n_phys)
    end

    # 2) Overwrite BC rows
    @inbounds for k in eachindex(ws.bc_rows_global)
        ws.f_work[ws.bc_rows_global[k]] = ws.bc_values[k]
    end

    # 3) Backsolve
    _backsolve!(ws.a_raw, ws.factorization, ws.f_work)

    # 4) Per-unknown physical recovery into phys_out (no writeback)
    @inbounds for j in 1:n_eq
        cols = (ws.col_offsets[j] + 1):ws.col_offsets[j + 1]
        @views a_j = ws.a_raw[cols]
        @views phys_j = ws.phys_out[:, j]
        mul!(phys_j, ws.M_evals[j], a_j)
    end
    return nothing
end

# Writeback each unknown's phys_out column into its variable slot on the
# target grid. The caller chooses whether to pass the source grid (mutating
# solve!) or a narrowed copy (non-mutating solve).
# `var_idx_map` may be supplied when writing into a narrowed grid whose var
# slots differ from the workspace's original field indices; its length must
# equal the number of equations.
function _writeback_block_linear!(target_grid::AbstractGrid,
                                   ws::BlockLinearWorkspace,
                                   var_idx_map::Union{Nothing, Vector{Int}} = nothing)
    n_phys = ws.n_phys
    n_eq   = ws.n_eq
    phys_mat = target_grid.physical
    @inbounds for j in 1:n_eq
        var_idx = var_idx_map === nothing ? ws.fields[j].var_idx : var_idx_map[j]
        @views phys_j = ws.phys_out[:, j]
        for r in 1:n_phys
            phys_mat[r, var_idx, 1] = phys_j[r]
        end
    end
    return nothing
end

# ────────────────────────────────────────────────────────────────────────────
# _solve_to_snapshot — non-mutating solve path used by `solve(prob)`
# ────────────────────────────────────────────────────────────────────────────
#
# Compute into the cached workspace scratch, narrow-copy the source grid to a
# fresh solution grid holding only the solved field(s), and write results
# into that copy. Returns a SpringsteelSolution whose `grid` is independent
# of `prob.grid` — non-mutating contract for users who prefer the returned
# Solution object over the in-place solve! idiom.

function _solve_to_snapshot(prob::SpringsteelProblem, ws::LocalLinearWorkspace)
    _compute_local_linear!(prob, ws)

    var_name = ws.field.var_name
    sol_grid = _subgrid_for_solution(prob.grid, [var_name])
    _writeback_local_linear!(sol_grid, ws, 1)

    # Reconstruct full-spectral coefficients (unfold gammaBC + ahat) and
    # write them into the solution grid's spectral slot so `sol.coefficients`
    # views match what legacy `solve` used to return.
    a = ws.gammaBC_T === nothing ? copy(ws.a_raw) : ws.gammaBC_T * ws.a_raw
    if ws.ahat_total !== nothing
        a .+= ws.ahat_total
    end
    @inbounds for r in 1:size(sol_grid.spectral, 1)
        sol_grid.spectral[r, 1] = a[r]
    end

    return SpringsteelSolution(sol_grid, 1, true)
end

function _solve_to_snapshot(prob::SpringsteelProblem, ws::BlockLinearWorkspace)
    _compute_block_linear!(prob, ws)

    var_names = [f.var_name for f in ws.fields]
    sol_grid = _subgrid_for_solution(prob.grid, var_names)
    _writeback_block_linear!(sol_grid, ws, collect(1:length(var_names)))

    # Per-unknown spectral reconstruction. Block systems don't currently
    # support R3X ahat, and gammaBC is per-unknown.
    for j in 1:ws.n_eq
        cols = (ws.col_offsets[j] + 1):ws.col_offsets[j+1]
        @views a_j_reduced = ws.a_raw[cols]
        gT = ws.gammaBC_Ts[j]
        a_j_full = gT === nothing ? Vector{Float64}(a_j_reduced) : gT * a_j_reduced
        @inbounds for r in 1:length(a_j_full)
            sol_grid.spectral[r, j] = a_j_full[r]
        end
    end

    return SpringsteelSolution(sol_grid, 1, true)
end

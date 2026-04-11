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

# ────────────────────────────────────────────────────────────────────────────
# SpringsteelField
# ────────────────────────────────────────────────────────────────────────────

"""
    SpringsteelField(grid, var_name::AbstractString)

Bind a grid variable to a handle that can be used as the unknown in operator
algebra (`L * u => rhs`). `Field` is an exported alias.

Holds a reference to `grid`, the variable name, and its integer slot index in
`grid.params.vars`. Boundary conditions come from the grid's per-variable BC
spec — there is no separate BC argument.
"""
struct SpringsteelField
    grid::AbstractGrid
    var_name::String
    var_idx::Int
end

function SpringsteelField(grid::AbstractGrid, var_name::AbstractString)
    vname = String(var_name)
    haskey(grid.params.vars, vname) ||
        throw(ArgumentError("Variable `$(vname)` not found in grid.params.vars"))
    return SpringsteelField(grid, vname, grid.params.vars[vname])
end

const Field = SpringsteelField

# ────────────────────────────────────────────────────────────────────────────
# TypedOperator — `OperatorExpr * SpringsteelField`
# ────────────────────────────────────────────────────────────────────────────

"""
    TypedOperator(expr, field)

The result of multiplying an `OperatorExpr` (or `DerivMono`/`ScaledMono`) by a
`SpringsteelField`. Captures the unknown so `SpringsteelProblem` knows which
variable's BCs to pull from the grid and where to write the solution.
"""
struct TypedOperator
    expr::OperatorExpr
    field::SpringsteelField
end

Base.:*(e::OperatorExpr, f::SpringsteelField) = TypedOperator(e, f)
Base.:*(m::DerivMono,   f::SpringsteelField) = TypedOperator(_as_expr(m), f)
Base.:*(s::ScaledMono,  f::SpringsteelField) = TypedOperator(_as_expr(s), f)

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
                             backend::AbstractSolverBackend = LocalLinearBackend())
    top   = eq.first
    rhs0  = eq.second
    field = top.field
    field.grid === grid ||
        throw(ArgumentError("SpringsteelField.grid does not match problem grid"))

    lowered = _lower(top.expr, grid)
    ws = _build_local_linear_workspace(grid, field, lowered, rhs0)

    return SpringsteelProblem(grid, nothing, nothing, nothing,
                               Dict{String, Any}(), backend, ws)
end

# ────────────────────────────────────────────────────────────────────────────
# Workspace construction
# ────────────────────────────────────────────────────────────────────────────

function _build_local_linear_workspace(grid::AbstractGrid,
                                        field::SpringsteelField,
                                        terms::Vector{OperatorTerm},
                                        rhs_source)
    var = field.var_name
    var_idx = field.var_idx

    L_base = assemble_operator(grid, terms, var)
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

    # Recover BC row indices + values by diffing f_bc0 against the pre-BC
    # baseline (which was just -ahat_correction, i.e. zeros or the R3X
    # correction). Any row whose value was replaced sits in bc_rows.
    baseline = ahat_correction === nothing ? zeros(n_phys) : (-ahat_correction)
    bc_rows  = Int[]
    bc_values = Float64[]
    @inbounds for r in 1:n_phys
        if f_bc0[r] != baseline[r]
            push!(bc_rows, r)
            push!(bc_values, f_bc0[r])
        end
    end

    # Cache factorisation
    F = size(L_bc, 1) == size(L_bc, 2) ? factorize(L_bc) : qr(L_bc)

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

Stateful solve that reuses `prob.workspace`. Refreshes the RHS (from the
grid if it was given as a `Symbol`), applies the cached ahat/BC corrections,
backsolves against the cached factorisation, and writes the physical-space
solution into `prob.grid.physical[:, var_idx, 1]`. Returns `prob`.
"""
function solve!(prob::SpringsteelProblem)
    ws = prob.workspace
    ws === nothing && throw(ArgumentError(
        "solve! requires a stateful SpringsteelProblem built via the " *
        "Pair-based constructor (SpringsteelProblem(grid, L*u => rhs))"))
    _solve_local_linear!(prob, ws::LocalLinearWorkspace)
    return prob
end

# The hot path: stateful solve using cached workspace.
function _solve_local_linear!(prob::SpringsteelProblem, ws::LocalLinearWorkspace)
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
    ldiv!(ws.a_raw, ws.factorization, ws.f_work)

    # 5) Physical recovery: phys = M_eval * a_raw + phys_ahat
    mul!(ws.phys_out, ws.M_eval, ws.a_raw)
    if ws.phys_ahat !== nothing
        @inbounds @simd for r in 1:n_phys
            ws.phys_out[r] += ws.phys_ahat[r]
        end
    end

    # 6) Writeback into grid.physical[:, var_idx, 1]
    var_idx = ws.field.var_idx
    phys = grid.physical
    @inbounds for r in 1:n_phys
        phys[r, var_idx, 1] = ws.phys_out[r]
    end

    return prob
end

# ────────────────────────────────────────────────────────────────────────────
# RHS resolution — S5 will generalise this
# ────────────────────────────────────────────────────────────────────────────

function _fetch_rhs!(dst::Vector{Float64}, src::Symbol, grid::AbstractGrid, n_phys::Int)
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

function _fetch_rhs!(dst::Vector{Float64}, src::AbstractVector{<:Real},
                     ::AbstractGrid, n_phys::Int)
    length(src) == n_phys || throw(ArgumentError(
        "RHS vector length $(length(src)) does not match physical size $n_phys"))
    @inbounds for r in 1:n_phys
        dst[r] = src[r]
    end
    return dst
end

function _fetch_rhs!(dst::Vector{Float64}, src::Real, ::AbstractGrid, n_phys::Int)
    fill!(dst, Float64(src))
    return dst
end

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
algorithm used by [`solve`](@ref).

See also: [`LocalLinearBackend`](@ref), [`OptimizationBackend`](@ref)
"""
abstract type AbstractSolverBackend end

"""
    LocalLinearBackend <: AbstractSolverBackend

Backend sentinel for the built-in local linear solver. Uses LU factorisation
via `\\` to solve ``\\mathbf{L} \\mathbf{a} = \\mathbf{f}``.

See also: [`solve`](@ref), [`SpringsteelProblem`](@ref)
"""
struct LocalLinearBackend <: AbstractSolverBackend end

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

Composite type bundling a grid, linear operator or cost functional, and solver
backend specification.

# Fields
- `grid::AbstractGrid`: The discretised domain
- `operator::Union{Matrix{Float64}, Nothing}`: Assembled linear operator (linear problems)
- `rhs::Union{Vector{Float64}, Nothing}`: Right-hand side vector (linear problems)
- `cost::Union{Function, Nothing}`: Cost functional ``J(u, p)`` (optimisation problems)
- `parameters::Dict{String, Any}`: Problem parameters passed to the solver
- `backend::B`: Solver backend sentinel

# Example
```julia
prob = SpringsteelProblem(grid; operator=L, rhs=f)
sol = solve(prob)
```

See also: [`solve`](@ref), [`SpringsteelSolution`](@ref), [`assemble_operator`](@ref)
"""
struct SpringsteelProblem{B <: AbstractSolverBackend}
    grid::AbstractGrid
    operator::Union{Matrix{Float64}, Nothing}
    rhs::Union{Vector{Float64}, Nothing}
    cost::Union{Function, Nothing}
    parameters::Dict{String, Any}
    backend::B
end

"""
    SpringsteelProblem(grid; operator=nothing, rhs=nothing, cost=nothing,
                       parameters=Dict{String,Any}(),
                       backend=LocalLinearBackend()) -> SpringsteelProblem

Convenience constructor for [`SpringsteelProblem`](@ref).
"""
function SpringsteelProblem(grid::AbstractGrid;
    operator::Union{Matrix{Float64}, Nothing} = nothing,
    rhs::Union{Vector{Float64}, Nothing} = nothing,
    cost::Union{Function, Nothing} = nothing,
    parameters::Dict{String, Any} = Dict{String, Any}(),
    backend::AbstractSolverBackend = LocalLinearBackend())

    return SpringsteelProblem(grid, operator, rhs, cost, parameters, backend)
end

# ────────────────────────────────────────────────────────────────────────────
# SpringsteelSolution
# ────────────────────────────────────────────────────────────────────────────

"""
    SpringsteelSolution

Container for the result of [`solve`](@ref).

# Fields
- `grid::AbstractGrid`: The grid (with updated spectral/physical arrays)
- `coefficients::Vector{Float64}`: Spectral coefficients of the solution
- `physical::Vector{Float64}`: Physical-space solution values
- `converged::Bool`: Whether the solver converged
- `info::Dict{String, Any}`: Solver diagnostics (iterations, residual, etc.)

See also: [`solve`](@ref), [`SpringsteelProblem`](@ref)
"""
struct SpringsteelSolution
    grid::AbstractGrid
    coefficients::Vector{Float64}
    physical::Vector{Float64}
    converged::Bool
    info::Dict{String, Any}
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
        d_ii=nothing, d_jj=nothing, d_kk=nothing) -> Matrix{Float64}

Build an operator matrix from an equation-oriented specification. Non-nothing
keyword arguments are converted to [`OperatorTerm`](@ref) objects and assembled
via [`assemble_operator`](@ref).

# Keywords
- `d0`: Coefficient for the identity (0th derivative) term
- `d_i`: Coefficient for ``\\partial/\\partial i``
- `d_j`: Coefficient for ``\\partial/\\partial j``
- `d_k`: Coefficient for ``\\partial/\\partial k``
- `d_ii`: Coefficient for ``\\partial^2/\\partial i^2``
- `d_jj`: Coefficient for ``\\partial^2/\\partial j^2``
- `d_kk`: Coefficient for ``\\partial^2/\\partial k^2``

Each coefficient can be a `Float64` (scalar), `Vector{Float64}` (spatially varying),
or `nothing` (term not included).

# Example
```julia
# 1D Poisson: u'' = f
L = assemble_from_equation(grid; d_ii=1.0)

# 2D Laplacian: ∂²u/∂r² + ∂²u/∂z²
L = assemble_from_equation(grid; d_ii=1.0, d_kk=1.0)

# General 2nd-order ODE: au'' + bu' + cu = f
L = assemble_from_equation(grid; d_ii=a, d_i=b, d0=c)
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
    d_kk = nothing)

    terms = OperatorTerm[]

    if d0 !== nothing
        push!(terms, OperatorTerm(0, 0, 0, _to_coeff(d0)))
    end
    if d_i !== nothing
        push!(terms, OperatorTerm(1, 0, 0, _to_coeff(d_i)))
    end
    if d_j !== nothing
        push!(terms, OperatorTerm(0, 1, 0, _to_coeff(d_j)))
    end
    if d_k !== nothing
        push!(terms, OperatorTerm(0, 0, 1, _to_coeff(d_k)))
    end
    if d_ii !== nothing
        push!(terms, OperatorTerm(2, 0, 0, _to_coeff(d_ii)))
    end
    if d_jj !== nothing
        push!(terms, OperatorTerm(0, 2, 0, _to_coeff(d_jj)))
    end
    if d_kk !== nothing
        push!(terms, OperatorTerm(0, 0, 2, _to_coeff(d_kk)))
    end

    if isempty(terms)
        throw(ArgumentError("At least one coefficient must be non-nothing"))
    end

    return assemble_operator(grid, terms, var)
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
            push!(matrices, obj.gammaBC')  # bDim × Minterior
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
# solve — local linear backend
# ────────────────────────────────────────────────────────────────────────────

"""
    solve(prob::SpringsteelProblem{LocalLinearBackend}) -> SpringsteelSolution
    solve(prob::SpringsteelProblem{OptimizationBackend}) -> SpringsteelSolution

Solve the problem defined by `prob` using the specified backend.

For `LocalLinearBackend`: factorises the operator matrix and solves
``\\mathbf{L} \\mathbf{a} = \\mathbf{f}`` via LU decomposition. Boundary
conditions are automatically applied based on the grid's basis types:

- **Chebyshev**: boundary rows are replaced with evaluation/derivative
  constraint rows (Dirichlet or Neumann).
- **CubicBSpline**: the ``\\boldsymbol{\\Gamma}_{\\mathrm{BC}}`` projection
  matrix folds boundary constraints into the operator.
- **Fourier**: periodic BCs are natural; no modification needed.

Supports 1D, 2D, and 3D grids with mixed basis types.

For `OptimizationBackend`: minimises the cost functional ``J(u, p)``
using the algorithm specified in the backend (requires `Optimization.jl`).

# Returns
- [`SpringsteelSolution`](@ref) containing the spectral coefficients,
  physical-space solution, convergence flag, and solver diagnostics.

See also: [`SpringsteelProblem`](@ref), [`assemble_operator`](@ref)
"""
function solve(prob::SpringsteelProblem{LocalLinearBackend})
    if prob.operator === nothing || prob.rhs === nothing
        throw(ArgumentError("LocalLinearBackend requires both operator and rhs"))
    end

    grid = prob.grid
    L = copy(prob.operator)
    f = copy(prob.rhs)
    var = get(prob.parameters, "var", "")
    var_idx = var == "" ? 1 : grid.params.vars[var]
    info = Dict{String, Any}()

    try
        # Build R3X ahat total (before gammaBC folding, since it lives in full space)
        ahat_total = _build_ahat_total(grid, var_idx)

        # Adjust RHS for inhomogeneous R3X BCs: f -= L * ahat_total
        if ahat_total !== nothing
            f .= f .- L * ahat_total
        end

        # Apply all boundary conditions (Chebyshev rows + Spline gammaBC folding)
        L, f, gammaBC_T = _apply_all_bcs(grid, L, f, var_idx, var)

        # Check for cached factorisation
        if haskey(prob.parameters, "_factorisation")
            F = prob.parameters["_factorisation"]
        else
            if size(L, 1) == size(L, 2)
                F = factorize(L)
            else
                # Rectangular system (e.g., after gammaBC folding) — use QR
                F = qr(L)
            end
            prob.parameters["_factorisation"] = F
        end

        # Solve the system
        a_raw = F \ f
        info["factorisation"] = F

        # Recover full spectral coefficients if gammaBC was applied
        if gammaBC_T !== nothing
            a = gammaBC_T * a_raw
        else
            a = a_raw
        end

        # Add back the ahat background for R3X
        if ahat_total !== nothing
            a = a .+ ahat_total
        end

        # Compute physical values using the raw (no-BC) evaluation matrix
        M_eval = assemble_operator(grid, [OperatorTerm(0, 0, 0, nothing)], var)
        phys = M_eval * a

        return SpringsteelSolution(grid, a, phys, true, info)

    catch e
        if e isa SingularException || e isa LinearAlgebra.SingularException
            info["error"] = string(e)
            n_spec = size(prob.operator, 2)
            n_phys = size(grid.physical, 1)
            return SpringsteelSolution(grid, zeros(n_spec), zeros(n_phys), false, info)
        else
            rethrow(e)
        end
    end
end

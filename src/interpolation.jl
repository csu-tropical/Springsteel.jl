# ────────────────────────────────────────────────────────────────────────────
# Grid-to-grid spectral interpolation
# ────────────────────────────────────────────────────────────────────────────
#
# Four layers:
#   1. Data import            — grid_from_regular_data, grid_from_netcdf
#   2. Same-geometry interp   — interpolate_to_grid (matching {G,I,J,K})
#   3. Cross-geometry interp  — coordinate mappings, unstructured evaluation,
#                               interpolate_to_grid (mixed geometry pairs)
#   4. Helpers                — bounds checking, variable matching
#
# Must be included AFTER transforms_*.jl (uses _cheb_eval_pts!) and factory.jl.

# ════════════════════════════════════════════════════════════════════════════
# Layer 1: Data Import
# ════════════════════════════════════════════════════════════════════════════

# ── Validation helpers ────────────────────────────────────────────────────

"""
    _check_uniform_spacing(x::AbstractVector{<:Real}) -> Float64

Verify that `x` is uniformly spaced and return the spacing `h`.
Throws `ArgumentError` if spacing varies by more than 100ε relative to the range.
"""
function _check_uniform_spacing(x::AbstractVector{<:Real})
    length(x) < 2 && throw(ArgumentError("Coordinate vector must have at least 2 elements"))
    dx = diff(x)
    h = dx[1]
    max_range = maximum(abs.(x))
    tol = max(100 * eps(max_range), 100 * eps(abs(h)))
    if maximum(abs.(dx .- h)) > tol
        throw(ArgumentError(
            "Coordinate vector must be uniformly spaced. " *
            "Max spacing deviation: $(maximum(abs.(dx .- h)))"))
    end
    return Float64(h)
end

"""
    _check_mubar_divisibility(N::Int, mubar::Int)

Verify that N is divisible by mubar. Throws `ArgumentError` with guidance if not.
"""
function _check_mubar_divisibility(N::Int, mubar::Int)
    if N % mubar != 0
        divisors = filter(m -> N % m == 0, 1:min(N, 10))
        throw(ArgumentError(
            "length(coordinate)=$N is not divisible by mubar=$mubar. " *
            "Choose mubar ∈ $divisors or resample data to a compatible size."))
    end
end

"""
    _infer_domain_bounds(x::AbstractVector{<:Real}, h::Float64) -> (Float64, Float64)

Infer domain bounds from regularly-spaced data points. For regular quadrature,
points are at cell midpoints, so the domain extends half a grid spacing beyond
the first and last points.
"""
function _infer_domain_bounds(x::AbstractVector{<:Real}, h::Float64)
    xmin = Float64(x[1]) - 0.5 * h
    xmax = Float64(x[end]) + 0.5 * h
    return xmin, xmax
end

"""
    _make_var_dict(vars::Vector{String}, nvars::Int) -> Dict{String, Int}

Build a variable name → index mapping. Auto-generates names `"v1"`, `"v2"`, …
if `vars` is empty.
"""
function _make_var_dict(vars::Vector{String}, nvars::Int)
    if isempty(vars)
        vars = ["v$i" for i in 1:nvars]
    end
    length(vars) == nvars || throw(ArgumentError(
        "Length of `vars` ($(length(vars))) must match number of data columns ($nvars)"))
    return Dict(vars[i] => i for i in 1:nvars)
end

"""
    _make_l_q_dict(l_q) -> Dict

Convert `l_q` to Dict format. Accepts a scalar (applied as `"default"`) or a Dict.
"""
_make_l_q_dict(l_q::Real) = Dict("default" => Float64(l_q))
_make_l_q_dict(l_q::Dict) = l_q

"""
    _expand_bc(bc_spec::Dict, var_dict::Dict) -> Dict

Expand a BC specification to a per-variable Dict. If `bc_spec` already has
variable-name keys, returns it directly. Otherwise, applies the BC spec to all
variables.

The convention is: `BCL = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R1T0)`
maps variable names to BC types. If the user passes a bare BC like `CubicBSpline.R0`,
it is applied to all variables.
"""
function _expand_bc(bc_spec::Dict, var_dict::Dict)
    var_names = keys(var_dict)
    # Check if bc_spec already has variable-name keys
    if any(k -> haskey(var_dict, k), keys(bc_spec))
        return bc_spec
    end
    # Apply same BC to all variables
    return Dict(name => bc_spec for name in var_names)
end

# ── 1D grid_from_regular_data ────────────────────────────────────────────

"""
    grid_from_regular_data(x, data; kwargs...) -> SpringsteelGrid
    grid_from_regular_data(x, y, data; kwargs...) -> SpringsteelGrid
    grid_from_regular_data(x, y, z, data; kwargs...) -> SpringsteelGrid

Create a CubicBSpline grid with `quadrature=:regular` from regularly-spaced data.

The coordinate vectors must be uniformly spaced. The number of points in each
dimension must be divisible by `mubar`.

# Dimension inference

Given N regularly-spaced points and target `mubar`, `num_cells = N ÷ mubar`.
Domain bounds are inferred from the coordinate vectors: since regular quadrature
places points at cell midpoints, the domain extends half a grid spacing beyond
the first and last data points.

# Data layout

`data` is always `(total_points, nvars)` where `total_points` is the product of
all coordinate vector lengths. For multi-dimensional data, the ordering is
i-outer, j-inner (j varies fastest), matching Springsteel's physical array layout.

# Keyword arguments
- `mubar::Int=3`: Quadrature points per cell
- `l_q=2.0`: Filter length (scalar or Dict)
- `BCL`, `BCR`: i-dimension boundary conditions (default `CubicBSpline.R0`)
- `BCU`, `BCD`: j-dimension boundary conditions (2D/3D only)
- `BCB`, `BCT`: k-dimension boundary conditions (3D only)
- `vars::Vector{String}=String[]`: Variable names (auto-generated if empty)

# Returns

A `SpringsteelGrid` with physical data populated in derivative slot 1.
Derivative slots are filled with `NaN` — call `spectralTransform!` followed by
`gridTransform!` to compute derivatives.

See also: [`grid_from_netcdf`](@ref), [`interpolate_to_grid`](@ref)
"""
function grid_from_regular_data(x::AbstractVector{<:Real}, data::AbstractMatrix{<:Real};
        mubar::Int=3, l_q=2.0,
        BCL::Dict=CubicBSpline.R0, BCR::Dict=CubicBSpline.R0,
        vars::Vector{String}=String[])

    N = length(x)
    nvars = size(data, 2)

    h = _check_uniform_spacing(x)
    _check_mubar_divisibility(N, mubar)
    size(data, 1) == N || throw(ArgumentError(
        "data must have $(N) rows (matching length(x)), got $(size(data, 1))"))

    num_cells = N ÷ mubar
    xmin, xmax = _infer_domain_bounds(x, h)
    var_dict = _make_var_dict(vars, nvars)
    l_q_dict = _make_l_q_dict(l_q)

    gp = SpringsteelGridParameters(
        geometry  = "R",
        iMin      = xmin,
        iMax      = xmax,
        num_cells = num_cells,
        mubar     = mubar,
        quadrature = :regular,
        BCL       = _expand_bc(BCL, var_dict),
        BCR       = _expand_bc(BCR, var_dict),
        l_q       = l_q_dict,
        vars      = var_dict,
    )

    grid = createGrid(gp)

    # Copy function values into slot 1; fill derivative slots with NaN
    grid.physical[:, :, 1] .= data
    for d in 2:size(grid.physical, 3)
        grid.physical[:, :, d] .= NaN
    end

    return grid
end

# ── 2D grid_from_regular_data ────────────────────────────────────────────

function grid_from_regular_data(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
        data::AbstractMatrix{<:Real};
        mubar::Int=3, l_q=2.0,
        BCL::Dict=CubicBSpline.R0, BCR::Dict=CubicBSpline.R0,
        BCU::Dict=CubicBSpline.R0, BCD::Dict=CubicBSpline.R0,
        vars::Vector{String}=String[])

    Nx, Ny = length(x), length(y)
    nvars = size(data, 2)

    hx = _check_uniform_spacing(x)
    hy = _check_uniform_spacing(y)
    _check_mubar_divisibility(Nx, mubar)
    _check_mubar_divisibility(Ny, mubar)
    size(data, 1) == Nx * Ny || throw(ArgumentError(
        "data must have $(Nx*Ny) rows (length(x)*length(y)), got $(size(data, 1))"))

    num_cells_i = Nx ÷ mubar
    xmin, xmax = _infer_domain_bounds(x, hx)
    ymin, ymax = _infer_domain_bounds(y, hy)
    var_dict = _make_var_dict(vars, nvars)
    l_q_dict = _make_l_q_dict(l_q)

    gp = SpringsteelGridParameters(
        geometry   = "RR",
        iMin       = xmin,
        iMax       = xmax,
        num_cells  = num_cells_i,
        mubar      = mubar,
        quadrature = :regular,
        BCL        = _expand_bc(BCL, var_dict),
        BCR        = _expand_bc(BCR, var_dict),
        jMin       = ymin,
        jMax       = ymax,
        jDim       = Ny,
        BCU        = _expand_bc(BCU, var_dict),
        BCD        = _expand_bc(BCD, var_dict),
        l_q        = l_q_dict,
        vars       = var_dict,
    )

    grid = createGrid(gp)

    grid.physical[:, :, 1] .= data
    for d in 2:size(grid.physical, 3)
        grid.physical[:, :, d] .= NaN
    end

    return grid
end

# ── 3D grid_from_regular_data ────────────────────────────────────────────

function grid_from_regular_data(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
        z::AbstractVector{<:Real}, data::AbstractMatrix{<:Real};
        mubar::Int=3, l_q=2.0,
        BCL::Dict=CubicBSpline.R0, BCR::Dict=CubicBSpline.R0,
        BCU::Dict=CubicBSpline.R0, BCD::Dict=CubicBSpline.R0,
        BCB::Dict=CubicBSpline.R0, BCT::Dict=CubicBSpline.R0,
        vars::Vector{String}=String[])

    Nx, Ny, Nz = length(x), length(y), length(z)
    nvars = size(data, 2)

    hx = _check_uniform_spacing(x)
    hy = _check_uniform_spacing(y)
    hz = _check_uniform_spacing(z)
    _check_mubar_divisibility(Nx, mubar)
    _check_mubar_divisibility(Ny, mubar)
    _check_mubar_divisibility(Nz, mubar)
    size(data, 1) == Nx * Ny * Nz || throw(ArgumentError(
        "data must have $(Nx*Ny*Nz) rows, got $(size(data, 1))"))

    num_cells_i = Nx ÷ mubar
    xmin, xmax = _infer_domain_bounds(x, hx)
    ymin, ymax = _infer_domain_bounds(y, hy)
    zmin, zmax = _infer_domain_bounds(z, hz)
    var_dict = _make_var_dict(vars, nvars)
    l_q_dict = _make_l_q_dict(l_q)

    gp = SpringsteelGridParameters(
        geometry   = "RRR",
        iMin       = xmin,
        iMax       = xmax,
        num_cells  = num_cells_i,
        mubar      = mubar,
        quadrature = :regular,
        BCL        = _expand_bc(BCL, var_dict),
        BCR        = _expand_bc(BCR, var_dict),
        jMin       = ymin,
        jMax       = ymax,
        jDim       = Ny,
        BCU        = _expand_bc(BCU, var_dict),
        BCD        = _expand_bc(BCD, var_dict),
        kMin       = zmin,
        kMax       = zmax,
        kDim       = Nz,
        BCB        = _expand_bc(BCB, var_dict),
        BCT        = _expand_bc(BCT, var_dict),
        l_q        = l_q_dict,
        vars       = var_dict,
    )

    grid = createGrid(gp)

    grid.physical[:, :, 1] .= data
    for d in 2:size(grid.physical, 3)
        grid.physical[:, :, d] .= NaN
    end

    return grid
end

# ── grid_from_netcdf ─────────────────────────────────────────────────────

"""
    grid_from_netcdf(filename::String; dim_names=nothing, var_names=nothing,
        kwargs...) -> SpringsteelGrid

Load regularly-spaced data from a NetCDF file into a CubicBSpline grid.

If `dim_names` is not specified, infers coordinate dimensions from the file
(variables whose names match dimension names, up to 3). If `var_names` is not
specified, reads all non-coordinate variables.

All remaining keyword arguments are forwarded to [`grid_from_regular_data`](@ref).

See also: [`grid_from_regular_data`](@ref), [`read_netcdf`](@ref)
"""
function grid_from_netcdf(filename::String;
        dim_names::Union{Nothing, Vector{String}}=nothing,
        var_names::Union{Nothing, Vector{String}}=nothing,
        kwargs...)

    NCDataset(filename, "r") do ds
        # ── Identify coordinate dimensions ────────────────────────────────
        if dim_names === nothing
            # Use dimensions that have a matching coordinate variable
            dim_names_found = String[]
            for (name, _) in ds.dim
                if haskey(ds, name)
                    push!(dim_names_found, name)
                end
            end
            if isempty(dim_names_found)
                throw(ArgumentError("No coordinate dimensions found in $filename"))
            end
            # Sort to maintain file order (NCDatasets preserves insertion order)
            dim_names_local = dim_names_found
        else
            dim_names_local = dim_names
        end

        ndims_spatial = length(dim_names_local)
        if ndims_spatial > 3
            throw(ArgumentError("At most 3 spatial dimensions supported, found $ndims_spatial"))
        end

        # ── Read coordinate vectors ───────────────────────────────────────
        coords = [Float64.(Array(ds[name])) for name in dim_names_local]

        # ── Identify data variables ───────────────────────────────────────
        if var_names === nothing
            coord_set = Set(dim_names_local)
            var_names_local = String[]
            for name in keys(ds)
                if !(name in coord_set)
                    push!(var_names_local, name)
                end
            end
        else
            var_names_local = var_names
        end

        if isempty(var_names_local)
            throw(ArgumentError("No data variables found in $filename"))
        end

        # ── Read and reshape data ─────────────────────────────────────────
        coord_sizes = [length(c) for c in coords]
        total_points = prod(coord_sizes)
        nvars = length(var_names_local)
        data = zeros(Float64, total_points, nvars)

        for (vi, vname) in enumerate(var_names_local)
            raw = Float64.(Array(ds[vname]))
            # Permute from Julia column-major (1st dim fastest) to
            # Springsteel convention (last dim fastest / i-outer j-inner)
            if ndims_spatial == 1
                data[:, vi] .= vec(raw)
            elseif ndims_spatial == 2
                # Julia: raw[x, y] with x fastest
                # Springsteel: i-outer, j-inner → j fastest
                data[:, vi] .= vec(permutedims(raw, (2, 1)))
            elseif ndims_spatial == 3
                # Julia: raw[x, y, z] with x fastest
                # Springsteel: k fastest, then j, then i
                data[:, vi] .= vec(permutedims(raw, (3, 2, 1)))
            end
        end

        # ── Delegate to grid_from_regular_data ────────────────────────────
        # Merge var_names into kwargs (override any 'vars' the user passed)
        merged_kwargs = Dict{Symbol, Any}(kwargs...)
        merged_kwargs[:vars] = var_names_local

        if ndims_spatial == 1
            return grid_from_regular_data(coords[1], data; merged_kwargs...)
        elseif ndims_spatial == 2
            return grid_from_regular_data(coords[1], coords[2], data; merged_kwargs...)
        else
            return grid_from_regular_data(coords[1], coords[2], coords[3], data;
                merged_kwargs...)
        end
    end
end


# ════════════════════════════════════════════════════════════════════════════
# Layer 2: Same-Geometry Interpolation
# ════════════════════════════════════════════════════════════════════════════

# ── Variable matching ────────────────────────────────────────────────────

"""
    _match_vars(source_vars::Dict, target_vars::Dict, requested) -> Vector{Tuple{String,String}}

Find variables present in both source and target grids. Returns a vector of
`(source_name, target_name)` pairs. Warns about unmatched variables.

If `requested` is `nothing`, matches all common variables. If a vector of names,
only matches those names.
"""
function _match_vars(source_vars::Dict, target_vars::Dict, requested)
    source_names = Set(keys(source_vars))
    target_names = Set(keys(target_vars))

    if requested === nothing
        common = intersect(source_names, target_names)
    else
        requested_set = Set(requested)
        common = intersect(requested_set, source_names, target_names)
        missing_src = setdiff(intersect(requested_set, target_names), source_names)
        if !isempty(missing_src)
            @warn "Requested variables not found in source grid: $(collect(missing_src))"
        end
    end

    unmatched_target = setdiff(target_names, source_names)
    if !isempty(unmatched_target) && requested === nothing
        @warn "Target variables not in source grid (will be NaN): $(collect(unmatched_target))"
    end

    return [(name, name) for name in sort(collect(common))]
end

# ── Bounds checking ──────────────────────────────────────────────────────

"""
    _check_bounds_1d(xmin, xmax, points) -> BitVector

Return a mask where `true` indicates the point is within `[xmin, xmax]`.
"""
function _check_bounds_1d(xmin::Float64, xmax::Float64, points::AbstractVector{Float64})
    return (points .>= xmin) .& (points .<= xmax)
end

"""
    _apply_oob!(result, mask, out_of_bounds)

Apply out-of-bounds handling to `result` based on `mask` (true = in bounds).
"""
function _apply_oob!(result::AbstractVector{Float64}, mask::BitVector, out_of_bounds)
    all(mask) && return
    if out_of_bounds === :error
        n_oob = count(.!mask)
        throw(DomainError(n_oob, "$n_oob target points are outside the source domain"))
    elseif out_of_bounds === :nan
        result[.!mask] .= NaN
    else
        result[.!mask] .= Float64(out_of_bounds)
    end
end

# ── Core interpolation ───────────────────────────────────────────────────

"""
    interpolate_to_grid(source::SpringsteelGrid, target::SpringsteelGrid;
        vars=nothing, out_of_bounds=:nan) -> Matrix{Float64}

Evaluate the source grid's spectral representation at the target grid's
physical points. Both grids must have the same geometry and basis types.

The source grid must have valid spectral coefficients — call
`spectralTransform!(source)` before interpolation.

# Arguments
- `source`: Source grid with populated `spectral` array
- `target`: Target grid defining the evaluation points
- `vars`: Variable names to interpolate (default: all common variables)
- `out_of_bounds`: How to handle target points outside the source domain.
  `:nan` (default), `:error`, or a numeric fill value.

# Returns
`Matrix{Float64}` of size `(target_phys_dim, n_target_vars)` with interpolated
function values. Unmatched target variables are filled with `NaN`.

See also: [`interpolate_to_grid!`](@ref), [`grid_from_regular_data`](@ref)
"""
function interpolate_to_grid(source::SpringsteelGrid{G,I,J,K},
                             target::SpringsteelGrid{G,I,J,K};
                             vars=nothing, out_of_bounds=:nan,
                             coordinate_map=nothing) where {G,I,J,K}

    # If a coordinate_map is provided, use the unstructured cross-geometry path
    if coordinate_map !== nothing
        t_pts = getGridpoints(target)
        src_pts = _map_points(coordinate_map, t_pts)
        sgp = source.params; tgp = target.params
        nvars_t = length(tgp.vars)
        phys_dim_t = size(target.physical, 1)
        result = fill(NaN, phys_dim_t, nvars_t)
        matched = _match_vars(sgp.vars, tgp.vars, vars)
        isempty(matched) && return result
        src_var_names = [sname for (sname, _) in matched]
        src_result = evaluate_unstructured(source, src_pts;
                                            vars=src_var_names, out_of_bounds=out_of_bounds)
        for (sname, tname) in matched
            sv = sgp.vars[sname]; tv = tgp.vars[tname]
            result[:, tv] .= src_result[:, sv]
        end
        return result
    end

    sgp = source.params
    tgp = target.params
    nvars_t = length(tgp.vars)
    phys_dim_t = size(target.physical, 1)

    result = fill(NaN, phys_dim_t, nvars_t)
    matched = _match_vars(sgp.vars, tgp.vars, vars)

    if isempty(matched)
        @warn "No matching variables found between source and target grids"
        return result
    end

    # Determine dimensionality
    i_active = !(source.ibasis isa NoBasisArray)
    j_active = !(source.jbasis isa NoBasisArray)
    k_active = !(source.kbasis isa NoBasisArray)

    if i_active && !j_active && !k_active
        _interpolate_1d!(result, source, target, matched, out_of_bounds)
    elseif i_active && j_active && !k_active
        _interpolate_2d_ij!(result, source, target, matched, out_of_bounds)
    elseif i_active && !j_active && k_active
        _interpolate_2d_ik!(result, source, target, matched, out_of_bounds)
    elseif i_active && j_active && k_active
        _interpolate_3d!(result, source, target, matched, out_of_bounds)
    else
        throw(ArgumentError("Unsupported grid dimensionality for interpolation"))
    end

    return result
end

"""
    interpolate_to_grid!(source::SpringsteelGrid, target::SpringsteelGrid;
        vars=nothing, out_of_bounds=:nan) -> SpringsteelGrid

Like [`interpolate_to_grid`](@ref), but writes the interpolated values directly
into `target.physical[:, :, 1]`. Derivative slots for interpolated variables are
filled with `NaN` to indicate they need recomputation via `spectralTransform!`
followed by `gridTransform!`.

Returns the modified `target` grid.
"""
function interpolate_to_grid!(source::SpringsteelGrid{G,I,J,K},
                              target::SpringsteelGrid{G,I,J,K};
                              vars=nothing, out_of_bounds=:nan) where {G,I,J,K}

    result = interpolate_to_grid(source, target; vars=vars, out_of_bounds=out_of_bounds)

    # Copy interpolated values into target physical array
    matched = _match_vars(source.params.vars, target.params.vars, vars)
    for (_, tname) in matched
        tv = target.params.vars[tname]
        target.physical[:, tv, 1] .= result[:, tv]
        # Set derivative slots to NaN for recomputation
        for d in 2:size(target.physical, 3)
            target.physical[:, tv, d] .= NaN
        end
    end

    return target
end

# ── 1D interpolation (spline only) ───────────────────────────────────────

function _interpolate_1d!(result::Matrix{Float64},
                          source::SpringsteelGrid, target::SpringsteelGrid,
                          matched::Vector{Tuple{String,String}}, out_of_bounds)

    sgp = source.params
    tgp = target.params

    # Get target evaluation points
    t_pts = collect(Float64, getGridpoints(target))

    # Bounds mask for spline dimension
    mask = _check_bounds_1d(sgp.iMin, sgp.iMax, t_pts)

    # Check for error mode upfront
    if out_of_bounds === :error && !all(mask)
        n_oob = count(.!mask)
        throw(DomainError(n_oob, "$n_oob target points are outside the source domain"))
    end

    # Only evaluate at in-bounds points to avoid spline domain errors
    ib_indices = findall(mask)
    pts_ib = t_pts[ib_indices]

    for (sname, tname) in matched
        sv = sgp.vars[sname]
        tv = tgp.vars[tname]

        sp = source.ibasis.data[1, sv]
        sp.b .= view(source.spectral, :, sv)
        SAtransform!(sp)

        if !isempty(pts_ib)
            u_ib = zeros(Float64, length(pts_ib))
            SItransform(sp, pts_ib, u_ib)
            for (k, idx) in enumerate(ib_indices)
                result[idx, tv] = u_ib[k]
            end
        end

        # Fill OOB values
        if !all(mask)
            fill_val = out_of_bounds === :nan ? NaN : Float64(out_of_bounds)
            for idx in findall(.!mask)
                result[idx, tv] = fill_val
            end
        end
    end
end

# ── 2D interpolation (i + j dimensions) ─────────────────────────────────

function _interpolate_2d_ij!(result::Matrix{Float64},
                             source::SpringsteelGrid, target::SpringsteelGrid,
                             matched::Vector{Tuple{String,String}}, out_of_bounds)

    sgp = source.params
    tgp = target.params
    b_iDim = sgp.b_iDim
    b_jDim = sgp.b_jDim

    # Get target evaluation points per dimension
    t_i_pts = _get_dim_points(target, :i)
    t_j_pts = _get_dim_points(target, :j)
    n_i = length(t_i_pts)
    n_j = length(t_j_pts)

    # Bounds checking for spline dimensions
    i_mask = _check_bounds_1d(sgp.iMin, sgp.iMax, t_i_pts)
    j_is_spline = source.jbasis isa SplineBasisArray
    j_mask = j_is_spline ? _check_bounds_1d(sgp.jMin, sgp.jMax, t_j_pts) : trues(n_j)

    for (sname, tname) in matched
        sv = sgp.vars[sname]
        tv = tgp.vars[tname]

        ibuf = zeros(Float64, n_i, b_jDim)

        # Step 1: i-direction evaluation at target i-points for each j-mode
        for l in 1:b_jDim
            r1 = (l - 1) * b_iDim + 1
            r2 = r1 + b_iDim - 1
            sp = source.ibasis.data[_ibasis_index(source, l, sv)...]
            sp.b .= view(source.spectral, r1:r2, sv)
            SAtransform!(sp)
            SItransform(sp, t_i_pts, view(ibuf, :, l))
        end

        # Step 2: j-direction evaluation at target j-points for each i output
        tmp = zeros(Float64, n_j)
        for xi in 1:n_i
            _eval_jdim!(source, ibuf, xi, sv, t_j_pts, tmp)

            flat = (xi - 1) * n_j + 1
            for ji in 1:n_j
                in_bounds = i_mask[xi] && j_mask[ji]
                if in_bounds
                    result[flat + ji - 1, tv] = tmp[ji]
                else
                    _set_oob_val!(result, flat + ji - 1, tv, out_of_bounds)
                end
            end
        end
    end
end

# ── 2D interpolation (i + k dimensions, e.g. RZ) ────────────────────────

function _interpolate_2d_ik!(result::Matrix{Float64},
                             source::SpringsteelGrid, target::SpringsteelGrid,
                             matched::Vector{Tuple{String,String}}, out_of_bounds)

    sgp = source.params
    tgp = target.params
    b_iDim = sgp.b_iDim
    b_kDim = sgp.b_kDim

    t_i_pts = _get_dim_points(target, :i)
    t_k_pts = _get_dim_points(target, :k)
    n_i = length(t_i_pts)
    n_k = length(t_k_pts)

    i_mask = _check_bounds_1d(sgp.iMin, sgp.iMax, t_i_pts)
    k_mask = _check_bounds_1d(sgp.kMin, sgp.kMax, t_k_pts)

    for (sname, tname) in matched
        sv = sgp.vars[sname]
        tv = tgp.vars[tname]

        ibuf = zeros(Float64, n_i, b_kDim)

        # Step 1: i-direction spline evaluation for each k-mode
        for z in 1:b_kDim
            r1 = (z - 1) * b_iDim + 1
            r2 = r1 + b_iDim - 1
            sp = source.ibasis.data[z, sv]
            sp.b .= view(source.spectral, r1:r2, sv)
            SAtransform!(sp)
            SItransform(sp, t_i_pts, view(ibuf, :, z))
        end

        # Step 2: k-direction Chebyshev evaluation for each i output
        cheb_col = source.kbasis.data[sv]
        tmp = zeros(Float64, n_k)
        for xi in 1:n_i
            # Set Chebyshev b-coefficients from ibuf
            for z in 1:b_kDim
                cheb_col.b[z] = ibuf[xi, z]
            end
            CAtransform!(cheb_col)
            _cheb_eval_pts!(cheb_col, t_k_pts, tmp)

            flat = (xi - 1) * n_k + 1
            for ki in 1:n_k
                in_bounds = i_mask[xi] && k_mask[ki]
                if in_bounds
                    result[flat + ki - 1, tv] = tmp[ki]
                else
                    _set_oob_val!(result, flat + ki - 1, tv, out_of_bounds)
                end
            end
        end
    end
end

# ── 3D interpolation (i + j + k dimensions) ─────────────────────────────

function _interpolate_3d!(result::Matrix{Float64},
                          source::SpringsteelGrid, target::SpringsteelGrid,
                          matched::Vector{Tuple{String,String}}, out_of_bounds)

    sgp = source.params
    tgp = target.params
    b_iDim = sgp.b_iDim
    b_jDim = sgp.b_jDim
    b_kDim = sgp.b_kDim

    t_i_pts = _get_dim_points(target, :i)
    t_j_pts = _get_dim_points(target, :j)
    t_k_pts = _get_dim_points(target, :k)
    n_i = length(t_i_pts)
    n_j = length(t_j_pts)
    n_k = length(t_k_pts)

    i_mask = _check_bounds_1d(sgp.iMin, sgp.iMax, t_i_pts)
    j_is_spline = source.jbasis isa SplineBasisArray
    j_mask = j_is_spline ? _check_bounds_1d(sgp.jMin, sgp.jMax, t_j_pts) : trues(n_j)
    k_is_spline = source.kbasis isa SplineBasisArray
    k_mask = k_is_spline ? _check_bounds_1d(sgp.kMin, sgp.kMax, t_k_pts) : trues(n_k)

    for (sname, tname) in matched
        sv = sgp.vars[sname]
        tv = tgp.vars[tname]

        # Step 1: i-direction evaluation → ibuf[n_i, b_jDim, b_kDim]
        ibuf = zeros(Float64, n_i, b_jDim * b_kDim)
        for jk in 1:(b_jDim * b_kDim)
            r1 = (jk - 1) * b_iDim + 1
            r2 = r1 + b_iDim - 1
            sp = _get_ibasis_3d(source, jk, sv)
            sp.b .= view(source.spectral, r1:r2, sv)
            SAtransform!(sp)
            SItransform(sp, t_i_pts, view(ibuf, :, jk))
        end

        # Step 2: j-direction evaluation → jbuf[n_i, n_j, b_kDim]
        jbuf = zeros(Float64, n_i, n_j, b_kDim)
        tmp_j = zeros(Float64, n_j)
        for xi in 1:n_i
            for kz in 1:b_kDim
                # Extract j-direction coefficients for this (i-point, k-mode)
                j_coeffs = zeros(Float64, b_jDim)
                for jl in 1:b_jDim
                    jk = (kz - 1) * b_jDim + jl  # spectral layout: k-outer, j-inner
                    j_coeffs[jl] = ibuf[xi, jk]
                end
                _eval_jdim_coeffs!(source, j_coeffs, sv, t_j_pts, tmp_j)
                jbuf[xi, :, kz] .= tmp_j
            end
        end

        # Step 3: k-direction evaluation → result
        tmp_k = zeros(Float64, n_k)
        for xi in 1:n_i
            for ji in 1:n_j
                _eval_kdim_coeffs!(source, view(jbuf, xi, ji, :), sv, t_k_pts, tmp_k)

                for ki in 1:n_k
                    flat = (xi - 1) * n_j * n_k + (ji - 1) * n_k + ki
                    in_bounds = i_mask[xi] && j_mask[ji] && k_mask[ki]
                    if in_bounds
                        result[flat, tv] = tmp_k[ki]
                    else
                        _set_oob_val!(result, flat, tv, out_of_bounds)
                    end
                end
            end
        end
    end
end

# ── Dimension helpers ────────────────────────────────────────────────────

"""Get per-dimension mish points from a target grid."""
function _get_dim_points(grid::SpringsteelGrid, dim::Symbol)
    if dim == :i
        return collect(Float64, grid.ibasis.data[1].mishPoints)
    elseif dim == :j
        return collect(Float64, grid.jbasis.data[1].mishPoints)
    elseif dim == :k
        return collect(Float64, grid.kbasis.data[1].mishPoints)
    end
end

"""Index helper for ibasis arrays (handles different dimensionalities)."""
function _ibasis_index(grid::SpringsteelGrid, l::Int, v::Int)
    if ndims(grid.ibasis.data) == 2
        return (l, v)
    else
        return (l, 1, v)  # 3D: ibasis.data[j_mode, k_mode, var]
    end
end

"""Get i-basis object for 3D grids where ibasis.data[j, k, v]."""
function _get_ibasis_3d(source::SpringsteelGrid, jk::Int, sv::Int)
    if ndims(source.ibasis.data) == 3
        b_kDim = source.params.b_kDim
        jl = div(jk - 1, b_kDim) + 1
        kz = mod(jk - 1, b_kDim) + 1
        return source.ibasis.data[jl, kz, sv]
    else
        return source.ibasis.data[jk, sv]
    end
end

"""Get a scratch Spline1D from jbasis.data, handling 2D and 3D storage."""
function _get_jbasis_scratch(source::SpringsteelGrid, sv::Int)
    if ndims(source.jbasis.data) == 3
        return source.jbasis.data[1, 1, sv]
    else
        return source.jbasis.data[1, sv]
    end
end

"""Get a scratch Spline1D from kbasis.data, handling 2D and 3D storage."""
function _get_kbasis_scratch(source::SpringsteelGrid, sv::Int)
    if ndims(source.kbasis.data) == 3
        return source.kbasis.data[1, 1, sv]
    else
        return source.kbasis.data[1, sv]
    end
end

"""Evaluate j-direction from intermediate buffer row."""
function _eval_jdim!(source::SpringsteelGrid, ibuf::Matrix{Float64},
                     xi::Int, sv::Int, t_j_pts::Vector{Float64},
                     out::Vector{Float64})
    if source.jbasis isa SplineBasisArray
        # Spline j-direction: set b-coefficients, B→A, evaluate
        scratch = _get_jbasis_scratch(source, sv)
        b_jDim = source.params.b_jDim
        for l in 1:b_jDim
            scratch.b[l] = ibuf[xi, l]
        end
        SAtransform!(scratch)
        SItransform(scratch, t_j_pts, out)
    elseif source.jbasis isa FourierBasisArray
        # Fourier j-direction: direct series evaluation from b-coefficients
        _fourier_eval_bcoeffs!(source.jbasis.data[1, sv].params,
                               view(ibuf, xi, :), t_j_pts, out)
    end
end

"""Evaluate j-direction from coefficient vector."""
function _eval_jdim_coeffs!(source::SpringsteelGrid, coeffs::Vector{Float64},
                            sv::Int, t_j_pts::Vector{Float64},
                            out::Vector{Float64})
    if source.jbasis isa SplineBasisArray
        scratch = _get_jbasis_scratch(source, sv)
        for l in eachindex(coeffs)
            scratch.b[l] = coeffs[l]
        end
        SAtransform!(scratch)
        SItransform(scratch, t_j_pts, out)
    elseif source.jbasis isa FourierBasisArray
        _fourier_eval_bcoeffs!(source.jbasis.data[1, sv].params,
                               coeffs, t_j_pts, out)
    end
end

"""Evaluate k-direction from coefficient vector."""
function _eval_kdim_coeffs!(source::SpringsteelGrid, coeffs::AbstractVector{Float64},
                            sv::Int, t_k_pts::Vector{Float64},
                            out::Vector{Float64})
    if source.kbasis isa SplineBasisArray
        scratch = _get_kbasis_scratch(source, sv)
        for l in eachindex(coeffs)
            scratch.b[l] = coeffs[l]
        end
        SAtransform!(scratch)
        SItransform(scratch, t_k_pts, out)
    elseif source.kbasis isa ChebyshevBasisArray
        col = source.kbasis.data[sv]
        b_kDim = source.params.b_kDim
        for z in 1:b_kDim
            col.b[z] = coeffs[z]
        end
        CAtransform!(col)
        _cheb_eval_pts!(col, t_k_pts, out)
    end
end

"""Direct Fourier series evaluation from b-coefficients at arbitrary points."""
function _fourier_eval_bcoeffs!(fp::Fourier.FourierParameters,
                                b::AbstractVector{Float64},
                                pts::Vector{Float64},
                                out::Vector{Float64})
    for i in eachindex(pts)
        θ = pts[i]
        val = b[1]
        for k = 1:fp.kmax
            val += 2.0 * b[k+1] * cos(k * θ) - 2.0 * b[fp.bDim-k+1] * sin(k * θ)
        end
        out[i] = val
    end
end

"""Set out-of-bounds value in result matrix."""
function _set_oob_val!(result::Matrix{Float64}, idx::Int, col::Int, out_of_bounds)
    if out_of_bounds === :error
        throw(DomainError(idx, "Target point at index $idx is outside the source domain"))
    elseif out_of_bounds === :nan
        result[idx, col] = NaN
    else
        result[idx, col] = Float64(out_of_bounds)
    end
end


# ════════════════════════════════════════════════════════════════════════════
# Layer 3: Cross-Geometry Interpolation
# ════════════════════════════════════════════════════════════════════════════

# ── Coordinate Mapping Functions ──────────────────────────────────────────

"""
    cartesian_to_cylindrical(x, y) -> (r, λ)

Map 2D Cartesian `(x, y)` to cylindrical `(r, λ)` with `λ ∈ [0, 2π)`.
"""
function cartesian_to_cylindrical(x::Float64, y::Float64)
    r = sqrt(x^2 + y^2)
    λ = mod(atan(y, x), 2π)
    return (r, λ)
end

"""
    cylindrical_to_cartesian(r, λ) -> (x, y)

Map cylindrical `(r, λ)` to 2D Cartesian `(x, y)`.
"""
function cylindrical_to_cartesian(r::Float64, λ::Float64)
    return (r * cos(λ), r * sin(λ))
end

"""
    cartesian_to_cylindrical_3d(x, y, z) -> (r, λ, z)

Map 3D Cartesian `(x, y, z)` to cylindrical `(r, λ, z)` with `λ ∈ [0, 2π)`.
"""
function cartesian_to_cylindrical_3d(x::Float64, y::Float64, z::Float64)
    r, λ = cartesian_to_cylindrical(x, y)
    return (r, λ, z)
end

"""
    cylindrical_to_cartesian_3d(r, λ, z) -> (x, y, z)

Map cylindrical `(r, λ, z)` to 3D Cartesian `(x, y, z)`.
"""
function cylindrical_to_cartesian_3d(r::Float64, λ::Float64, z::Float64)
    x, y = cylindrical_to_cartesian(r, λ)
    return (x, y, z)
end

"""
    cartesian_to_spherical(x, y, z) -> (R, θ, φ)

Map 3D Cartesian `(x, y, z)` to spherical `(R, θ, φ)` where `θ` is colatitude
`[0, π]` and `φ` is azimuth `[0, 2π)`.
"""
function cartesian_to_spherical(x::Float64, y::Float64, z::Float64)
    R = sqrt(x^2 + y^2 + z^2)
    θ = acos(clamp(z / max(R, eps()), -1.0, 1.0))
    φ = mod(atan(y, x), 2π)
    return (R, θ, φ)
end

"""
    spherical_to_cartesian(R, θ, φ) -> (x, y, z)

Map spherical `(R, θ, φ)` to 3D Cartesian `(x, y, z)`.
`θ` is colatitude, `φ` is azimuth.
"""
function spherical_to_cartesian(R::Float64, θ::Float64, φ::Float64)
    x = R * sin(θ) * cos(φ)
    y = R * sin(θ) * sin(φ)
    z = R * cos(θ)
    return (x, y, z)
end

"""
    cylindrical_to_spherical(r, λ, z) -> (R, θ, φ)

Map cylindrical `(r, λ, z)` to spherical `(R, θ, φ)`.
Azimuth is preserved: `φ = λ`.
"""
function cylindrical_to_spherical(r::Float64, λ::Float64, z::Float64)
    R = sqrt(r^2 + z^2)
    θ = acos(clamp(z / max(R, eps()), -1.0, 1.0))
    return (R, θ, λ)
end

"""
    spherical_to_cylindrical(R, θ, φ) -> (r, λ, z)

Map spherical `(R, θ, φ)` to cylindrical `(r, λ, z)`.
Azimuth is preserved: `λ = φ`.
"""
function spherical_to_cylindrical(R::Float64, θ::Float64, φ::Float64)
    r = R * sin(θ)
    z = R * cos(θ)
    return (r, φ, z)
end

# ── Lat/lon convenience functions ─────────────────────────────────────────

"""
    latlon_to_spherical(lon_deg, lat_deg) -> (θ, λ)

Convert geographic `(longitude, latitude)` in degrees to spherical
`(θ colatitude, λ azimuth)` in radians. `θ ∈ [0, π]`, `λ ∈ [0, 2π)`.
"""
function latlon_to_spherical(lon_deg::Float64, lat_deg::Float64)
    θ = deg2rad(90.0 - lat_deg)
    λ = deg2rad(mod(lon_deg, 360.0))
    return (θ, λ)
end

"""
    spherical_to_latlon(θ, λ) -> (lon_deg, lat_deg)

Convert spherical `(θ colatitude, λ azimuth)` in radians to geographic
`(longitude, latitude)` in degrees.
"""
function spherical_to_latlon(θ::Float64, λ::Float64)
    lat_deg = 90.0 - rad2deg(θ)
    lon_deg = rad2deg(λ)
    return (lon_deg, lat_deg)
end

# ── Mapping selection ─────────────────────────────────────────────────────

"""
    _get_cross_geometry_map(source, target, user_map) -> Function

Select the inverse coordinate mapping (target coords → source coords).
Uses `user_map` if provided, otherwise looks up a default based on geometry.
"""
function _get_cross_geometry_map(source::SpringsteelGrid, target::SpringsteelGrid,
                                  user_map)
    user_map !== nothing && return user_map

    SG = typeof(source).parameters[1]   # source geometry type
    TG = typeof(target).parameters[1]   # target geometry type

    s_ndims = _grid_ndims(source)
    t_ndims = _grid_ndims(target)

    # Cart <-> Cyl
    if SG === CartesianGeometry && TG === CylindricalGeometry
        if t_ndims == 2
            return (r, λ) -> cylindrical_to_cartesian(r, λ)
        else
            return (r, λ, z) -> cylindrical_to_cartesian_3d(r, λ, z)
        end
    elseif SG === CylindricalGeometry && TG === CartesianGeometry
        if t_ndims == 2
            return (x, y) -> cartesian_to_cylindrical(x, y)
        else
            return (x, y, z) -> cartesian_to_cylindrical_3d(x, y, z)
        end

    # Cart <-> Sph (3D only has natural default)
    # SLZ coordinate ordering: (θ=colatitude, λ=azimuth, z_R=radius/altitude)
    elseif SG === CartesianGeometry && TG === SphericalGeometry
        if t_ndims == 3
            # target SLZ (θ, λ, z_R) → source Cart (x, y, z)
            return (θ, λ, z_R) -> spherical_to_cartesian(z_R, θ, λ)
        else
            throw(ArgumentError(
                "No default coordinate mapping for 2D Cartesian → Spherical. " *
                "Provide a `coordinate_map` function, e.g. " *
                "`coordinate_map = (θ, λ) -> spherical_to_latlon(θ, λ)` " *
                "for lat/lon source grids."))
        end
    elseif SG === SphericalGeometry && TG === CartesianGeometry
        if s_ndims == 3
            # target Cart (x, y, z) → source SLZ (θ, λ, z_R)
            return (x, y, z) -> begin
                R, θ, φ = cartesian_to_spherical(x, y, z)
                (θ, φ, R)
            end
        else
            throw(ArgumentError(
                "No default coordinate mapping for 2D Spherical → Cartesian. " *
                "Provide a `coordinate_map` function specifying the projection."))
        end

    # Cyl <-> Sph
    elseif SG === CylindricalGeometry && TG === SphericalGeometry
        if t_ndims == 3
            # target SLZ (θ, λ, z_R) → source RLZ (r, λ_cyl, z_cyl)
            return (θ, λ, z_R) -> spherical_to_cylindrical(z_R, θ, λ)
        else
            throw(ArgumentError(
                "No default coordinate mapping for 2D Cylindrical → Spherical. " *
                "Provide a `coordinate_map` function."))
        end
    elseif SG === SphericalGeometry && TG === CylindricalGeometry
        if s_ndims == 3
            # target RLZ (r, λ, z) → source SLZ (θ, λ, z_R)
            return (r, λ, z) -> begin
                R, θ, φ = cylindrical_to_spherical(r, λ, z)
                (θ, φ, R)
            end
        else
            throw(ArgumentError(
                "No default coordinate mapping for 2D Spherical → Cylindrical. " *
                "Provide a `coordinate_map` function."))
        end

    else
        throw(ArgumentError(
            "No coordinate mapping available for $(SG) → $(TG). " *
            "Provide a `coordinate_map` function."))
    end
end

"""Count active dimensions of a grid."""
function _grid_ndims(grid::SpringsteelGrid)
    n = 0
    !(grid.ibasis isa NoBasisArray) && (n += 1)
    !(grid.jbasis isa NoBasisArray) && (n += 1)
    !(grid.kbasis isa NoBasisArray) && (n += 1)
    return n
end

# ── N-dimensional bounds checking ─────────────────────────────────────────

"""
    _check_bounds_nd(source::SpringsteelGrid, pts::AbstractMatrix{Float64}) -> BitVector

Check which points are within the source grid domain. Spline and Chebyshev
dimensions are bounded by `[min, max]`; Fourier dimensions are periodic
(always in bounds).
"""
function _check_bounds_nd(source::SpringsteelGrid, pts::AbstractMatrix{Float64})
    sgp = source.params
    mask = trues(size(pts, 1))
    col = 1

    if !(source.ibasis isa NoBasisArray)
        if source.ibasis isa SplineBasisArray
            mask .&= (pts[:, col] .>= sgp.iMin) .& (pts[:, col] .<= sgp.iMax)
        elseif source.ibasis isa ChebyshevBasisArray
            mask .&= (pts[:, col] .>= sgp.iMin) .& (pts[:, col] .<= sgp.iMax)
        end
        # FourierBasisArray: periodic, always in bounds
        col += 1
    end

    if !(source.jbasis isa NoBasisArray)
        if source.jbasis isa SplineBasisArray
            mask .&= (pts[:, col] .>= sgp.jMin) .& (pts[:, col] .<= sgp.jMax)
        elseif source.jbasis isa ChebyshevBasisArray
            mask .&= (pts[:, col] .>= sgp.jMin) .& (pts[:, col] .<= sgp.jMax)
        end
        col += 1
    end

    if !(source.kbasis isa NoBasisArray)
        if source.kbasis isa SplineBasisArray
            mask .&= (pts[:, col] .>= sgp.kMin) .& (pts[:, col] .<= sgp.kMax)
        elseif source.kbasis isa ChebyshevBasisArray
            mask .&= (pts[:, col] .>= sgp.kMin) .& (pts[:, col] .<= sgp.kMax)
        end
    end

    return mask
end

# ── Point mapping helper ──────────────────────────────────────────────────

"""Apply inverse coordinate mapping to each row of target_pts."""
function _map_points(inv_map, target_pts::AbstractMatrix{Float64})
    npts = size(target_pts, 1)
    ndims = size(target_pts, 2)
    mapped = similar(target_pts)
    for i in 1:npts
        if ndims == 1
            mapped[i, 1] = inv_map(target_pts[i, 1])
        elseif ndims == 2
            c1, c2 = inv_map(target_pts[i, 1], target_pts[i, 2])
            mapped[i, 1] = c1; mapped[i, 2] = c2
        elseif ndims == 3
            c1, c2, c3 = inv_map(target_pts[i, 1], target_pts[i, 2], target_pts[i, 3])
            mapped[i, 1] = c1; mapped[i, 2] = c2; mapped[i, 3] = c3
        end
    end
    return mapped
end

# ── Single-point evaluation helpers ───────────────────────────────────────

"""Evaluate Fourier series at a single point from B-coefficients."""
function _fourier_eval_single(fp::Fourier.FourierParameters,
                               b::AbstractVector{Float64}, θ::Float64)
    val = b[1]
    for k = 1:fp.kmax
        val += 2.0 * b[k+1] * cos(k * θ) - 2.0 * b[fp.bDim-k+1] * sin(k * θ)
    end
    return val
end

"""Evaluate Chebyshev expansion at a single point. `col.a` must be populated via `CAtransform!`."""
function _cheb_eval_single(col::Chebyshev.Chebyshev1D, z::Float64)
    cp = col.params
    N = cp.zDim
    a = col.a
    scale  = -0.5 * (cp.zmax - cp.zmin)
    offset =  0.5 * (cp.zmin + cp.zmax)
    ξ = clamp((z - offset) / scale, -1.0, 1.0)
    t = acos(ξ)
    val = a[1]
    for k in 2:(N - 1)
        val += 2.0 * a[k] * cos((k - 1) * t)
    end
    val += a[N] * cos((N - 1) * t)
    return val
end

"""Evaluate j-basis at a single point from coefficient vector."""
function _eval_single_jdim(source::SpringsteelGrid, coeffs::Vector{Float64},
                            sv::Int, y::Float64)
    if source.jbasis isa SplineBasisArray
        scratch = _get_jbasis_scratch(source, sv)
        for l in eachindex(coeffs)
            scratch.b[l] = coeffs[l]
        end
        SAtransform!(scratch)
        return CubicBSpline.SItransform(scratch.params, scratch.a, y, 0)
    elseif source.jbasis isa FourierBasisArray
        fp = _get_jfourier_params(source, sv)
        return _fourier_eval_single(fp, coeffs, y)
    end
end

"""Evaluate k-basis at a single point from coefficient vector."""
function _eval_single_kdim(source::SpringsteelGrid, coeffs::AbstractVector{Float64},
                            sv::Int, z::Float64)
    if source.kbasis isa SplineBasisArray
        scratch = _get_kbasis_scratch(source, sv)
        for l in eachindex(coeffs)
            scratch.b[l] = coeffs[l]
        end
        SAtransform!(scratch)
        return CubicBSpline.SItransform(scratch.params, scratch.a, z, 0)
    elseif source.kbasis isa ChebyshevBasisArray
        col = source.kbasis.data[sv]
        b_kDim = source.params.b_kDim
        for l in 1:b_kDim
            col.b[l] = coeffs[l]
        end
        CAtransform!(col)
        return _cheb_eval_single(col, z)
    end
end

"""Get FourierParameters from jbasis, handling 2D and 3D storage."""
function _get_jfourier_params(source::SpringsteelGrid, sv::Int)
    if ndims(source.jbasis.data) == 3
        return source.jbasis.data[1, 1, sv].params
    elseif ndims(source.jbasis.data) == 2
        return source.jbasis.data[1, sv].params
    else
        return source.jbasis.data[1].params
    end
end

# ── Unstructured evaluation: Cartesian grids ──────────────────────────────

"""Evaluate 1D spline grid at arbitrary points."""
function _eval_unstructured_1d_spline(source::SpringsteelGrid, pts_i::Vector{Float64}, sv::Int)
    b_iDim = source.params.b_iDim
    sp = source.ibasis.data[1, sv]
    sp.b .= view(source.spectral, 1:b_iDim, sv)
    SAtransform!(sp)
    u = zeros(Float64, length(pts_i))
    CubicBSpline.SItransform(sp, pts_i, u)
    return u
end

"""Evaluate 2D (i,j) Cartesian grid at unstructured points. Spectral layout: j-major."""
function _eval_unstructured_2d_ij(source::SpringsteelGrid, pts::AbstractMatrix{Float64}, sv::Int)
    sgp = source.params
    b_iDim = sgp.b_iDim
    b_jDim = sgp.b_jDim
    npts = size(pts, 1)
    result = zeros(Float64, npts)

    for n in 1:npts
        xi = pts[n, 1]
        yj = pts[n, 2]

        # Evaluate i-spline at xi for each j-mode
        j_coeffs = zeros(Float64, b_jDim)
        for l in 1:b_jDim
            r1 = (l - 1) * b_iDim + 1
            sp = source.ibasis.data[_ibasis_index(source, l, sv)...]
            sp.b .= view(source.spectral, r1:r1 + b_iDim - 1, sv)
            SAtransform!(sp)
            j_coeffs[l] = CubicBSpline.SItransform(sp.params, sp.a, xi, 0)
        end

        result[n] = _eval_single_jdim(source, j_coeffs, sv, yj)
    end
    return result
end

"""Evaluate 2D (i,k) Cartesian grid at unstructured points. Spectral layout: k-major."""
function _eval_unstructured_2d_ik(source::SpringsteelGrid, pts::AbstractMatrix{Float64}, sv::Int)
    sgp = source.params
    b_iDim = sgp.b_iDim
    b_kDim = sgp.b_kDim
    npts = size(pts, 1)
    result = zeros(Float64, npts)

    for n in 1:npts
        xi = pts[n, 1]
        zk = pts[n, 2]

        # Evaluate i-spline at xi for each k-mode
        k_coeffs = zeros(Float64, b_kDim)
        for z in 1:b_kDim
            r1 = (z - 1) * b_iDim + 1
            sp = source.ibasis.data[_ibasis_index(source, z, sv)...]
            sp.b .= view(source.spectral, r1:r1 + b_iDim - 1, sv)
            SAtransform!(sp)
            k_coeffs[z] = CubicBSpline.SItransform(sp.params, sp.a, xi, 0)
        end

        result[n] = _eval_single_kdim(source, k_coeffs, sv, zk)
    end
    return result
end

"""Evaluate 3D Cartesian grid at unstructured points.
Spectral layout: z-major, then j: idx = (z-1)*b_jDim*b_iDim + (l-1)*b_iDim + 1."""
function _eval_unstructured_3d_ijk(source::SpringsteelGrid, pts::AbstractMatrix{Float64}, sv::Int)
    sgp = source.params
    b_iDim = sgp.b_iDim
    b_jDim = sgp.b_jDim
    b_kDim = sgp.b_kDim
    npts = size(pts, 1)
    result = zeros(Float64, npts)

    for n in 1:npts
        xi = pts[n, 1]
        yj = pts[n, 2]
        zk = pts[n, 3]

        # Step 1: evaluate i-spline at xi for each (j,k) mode
        ibuf = zeros(Float64, b_jDim, b_kDim)
        for z in 1:b_kDim
            for l in 1:b_jDim
                r1 = (z - 1) * b_jDim * b_iDim + (l - 1) * b_iDim + 1
                sp = _get_ibasis_3d(source, (l - 1) * b_kDim + z, sv)
                sp.b .= view(source.spectral, r1:r1 + b_iDim - 1, sv)
                SAtransform!(sp)
                ibuf[l, z] = CubicBSpline.SItransform(sp.params, sp.a, xi, 0)
            end
        end

        # Step 2: evaluate j-basis at yj for each k-mode
        k_coeffs = zeros(Float64, b_kDim)
        j_coeffs = zeros(Float64, b_jDim)
        for z in 1:b_kDim
            for l in 1:b_jDim
                j_coeffs[l] = ibuf[l, z]
            end
            k_coeffs[z] = _eval_single_jdim(source, j_coeffs, sv, yj)
        end

        # Step 3: evaluate k-basis at zk
        result[n] = _eval_single_kdim(source, k_coeffs, sv, zk)
    end
    return result
end

# ── Unstructured evaluation: Spline+Fourier grids (RL, SL) ───────────────

"""
Evaluate 2D Spline+Fourier grid (RL or SL) at unstructured points.
Uses batched per-wavenumber spline evaluation for performance.

RL/SL spectral layout: wavenumber-interleaved (flat), p = k*2 (see TRAP-1).
- k=0: spectral[1:b_iDim]
- k≥1 cos: spectral[(2k-1)*b_iDim+1 : 2k*b_iDim]
- k≥1 sin: spectral[2k*b_iDim+1 : (2k+1)*b_iDim]
"""
# ── Cached ahat (γ-folded coefficients) for RL/RLZ unstructured eval ────────
# Avoids repeating the SAtransform! (γ-fold + banded solve) on every
# evaluate_unstructured call when the source spectral data is unchanged.
# The cache stores the post-SAtransform `a` coefficients per (grid, variable)
# and uses a hash of the spectral column to detect staleness.

struct _AhatCacheEntry
    spectral_hash::UInt64
    a_coeffs::Matrix{Float64}   # (bDim, n_stripes)
end

const _AHAT_CACHE = Dict{Tuple{UInt64, Int}, _AhatCacheEntry}()
const _AHAT_CACHE_LOCK = ReentrantLock()

function _get_ahat_cache_rl(source::SpringsteelGrid, sv::Int)
    gp = source.params
    b_iDim = gp.b_iDim
    kDim = gp.iDim + gp.patchOffsetL
    n_kslots = 1 + 2 * kDim
    key = (objectid(source), sv)
    spec_hash = hash(view(source.spectral, :, sv))

    entry = lock(_AHAT_CACHE_LOCK) do
        get(_AHAT_CACHE, key, nothing)
    end
    if entry !== nothing && entry.spectral_hash == spec_hash
        return entry.a_coeffs
    end

    a_cache = zeros(Float64, b_iDim, n_kslots)
    sp0 = source.ibasis.data[1, sv]
    sp0.b .= view(source.spectral, 1:b_iDim, sv)
    SAtransform!(sp0)
    a_cache[:, 1] .= sp0.a

    for k in 1:kDim
        p = k * 2
        p1c = (p - 1) * b_iDim + 1;  p2c = p * b_iDim
        p1s = p * b_iDim + 1;        p2s = (p + 1) * b_iDim

        spc = source.ibasis.data[2, sv]
        spc.b .= view(source.spectral, p1c:p2c, sv)
        SAtransform!(spc)
        a_cache[:, 2k] .= spc.a

        sps = source.ibasis.data[3, sv]
        sps.b .= view(source.spectral, p1s:p2s, sv)
        SAtransform!(sps)
        a_cache[:, 2k + 1] .= sps.a
    end

    lock(_AHAT_CACHE_LOCK) do
        _AHAT_CACHE[key] = _AhatCacheEntry(spec_hash, a_cache)
    end
    return a_cache
end

function _get_ahat_cache_rlz(source::SpringsteelGrid, sv::Int)
    gp = source.params
    b_iDim = gp.b_iDim
    b_kDim = gp.b_kDim
    kDim_wn = gp.iDim + gp.patchOffsetL
    n_kslots = 1 + 2 * kDim_wn
    total_stripes = b_kDim * n_kslots
    key = (objectid(source), sv)
    spec_hash = hash(view(source.spectral, :, sv))

    entry = lock(_AHAT_CACHE_LOCK) do
        get(_AHAT_CACHE, key, nothing)
    end
    if entry !== nothing && entry.spectral_hash == spec_hash
        return entry.a_coeffs
    end

    a_cache = zeros(Float64, b_iDim, total_stripes)

    for z_b in 1:b_kDim
        r1_base = (z_b - 1) * b_iDim * (1 + kDim_wn * 2) + 1
        r2_base = r1_base + b_iDim - 1
        stripe_offset = (z_b - 1) * n_kslots

        sp0 = source.ibasis.data[1, sv]
        sp0.b .= view(source.spectral, r1_base:r2_base, sv)
        SAtransform!(sp0)
        a_cache[:, stripe_offset + 1] .= sp0.a

        for k in 1:kDim_wn
            p = (k - 1) * 2
            p1 = r2_base + 1 + p * b_iDim;  p2 = p1 + b_iDim - 1

            spc = source.ibasis.data[2, sv]
            spc.b .= view(source.spectral, p1:p2, sv)
            SAtransform!(spc)
            a_cache[:, stripe_offset + 2k] .= spc.a

            p1 = p2 + 1;  p2 = p1 + b_iDim - 1
            sps = source.ibasis.data[3, sv]
            sps.b .= view(source.spectral, p1:p2, sv)
            SAtransform!(sps)
            a_cache[:, stripe_offset + 2k + 1] .= sps.a
        end
    end

    lock(_AHAT_CACHE_LOCK) do
        _AHAT_CACHE[key] = _AhatCacheEntry(spec_hash, a_cache)
    end
    return a_cache
end

function _clear_ahat_cache!()
    lock(_AHAT_CACHE_LOCK) do
        empty!(_AHAT_CACHE)
    end
end

# ── Scratch buffers for RL/RLZ unstructured eval ────────────────────────────

mutable struct _ScratchInterpRL
    ak::Matrix{Float64}
    r_pts::Vector{Float64}
    λ_pts::Vector{Float64}
    result::Vector{Float64}
    npts::Int
    n_kslots::Int
end

mutable struct _ScratchInterpRLZ
    spline_vals::Array{Float64, 3}
    r_pts::Vector{Float64}
    λ_pts::Vector{Float64}
    z_pts::Vector{Float64}
    result::Vector{Float64}
    npts::Int
    b_kDim::Int
    n_kslots::Int
end

const _INTERP_SCRATCH_RL  = Dict{UInt64, _ScratchInterpRL}()
const _INTERP_SCRATCH_RLZ = Dict{UInt64, _ScratchInterpRLZ}()

function _get_scratch_rl(grid_id::UInt64, npts::Int, n_kslots::Int)
    s = get(_INTERP_SCRATCH_RL, grid_id, nothing)
    if s !== nothing && s.npts >= npts && s.n_kslots >= n_kslots
        fill!(s.ak, 0.0)
        fill!(s.result, 0.0)
        return s
    end
    np = s === nothing ? npts : max(s.npts, npts)
    nk = s === nothing ? n_kslots : max(s.n_kslots, n_kslots)
    s = _ScratchInterpRL(zeros(Float64, np, nk), zeros(Float64, np),
                         zeros(Float64, np), zeros(Float64, np), np, nk)
    _INTERP_SCRATCH_RL[grid_id] = s
    return s
end

function _get_scratch_rlz(grid_id::UInt64, npts::Int, b_kDim::Int, n_kslots::Int)
    s = get(_INTERP_SCRATCH_RLZ, grid_id, nothing)
    if s !== nothing && s.npts >= npts && s.b_kDim >= b_kDim && s.n_kslots >= n_kslots
        fill!(s.spline_vals, 0.0)
        fill!(s.result, 0.0)
        return s
    end
    np = s === nothing ? npts : max(s.npts, npts)
    bk = s === nothing ? b_kDim : max(s.b_kDim, b_kDim)
    nk = s === nothing ? n_kslots : max(s.n_kslots, n_kslots)
    s = _ScratchInterpRLZ(zeros(Float64, np, bk, nk), zeros(Float64, np),
                          zeros(Float64, np), zeros(Float64, np),
                          zeros(Float64, np), np, bk, nk)
    _INTERP_SCRATCH_RLZ[grid_id] = s
    return s
end

function _eval_unstructured_rl(source::SpringsteelGrid, pts::AbstractMatrix{Float64}, sv::Int)
    gp = source.params
    b_iDim = gp.b_iDim
    kDim = gp.iDim + gp.patchOffsetL
    npts = size(pts, 1)
    n_kslots = 1 + 2 * kDim

    sc = _get_scratch_rl(objectid(source), npts, n_kslots)
    r_view = view(pts, :, 1)
    λ_view = view(pts, :, 2)

    a_cache = _get_ahat_cache_rl(source, sv)
    sp0 = source.ibasis.data[1, sv]
    for slot in 1:n_kslots
        CubicBSpline.SItransform(sp0.params, view(a_cache, :, slot),
                                 r_view, view(sc.ak, 1:npts, slot))
    end

    result = sc.result
    for n in 1:npts
        f = sc.ak[n, 1]
        λ = pts[n, 2]
        for k in 1:kDim
            f += 2.0 * (sc.ak[n, 2k] * cos(k * λ) + sc.ak[n, 2k + 1] * sin(k * λ))
        end
        result[n] = f
    end
    return view(result, 1:npts)
end

# ── Unstructured evaluation: Spline+Fourier+Chebyshev (RLZ, SLZ) ─────────

"""
Evaluate 3D Spline+Fourier+Chebyshev grid (RLZ or SLZ) at unstructured points.
Batched per-wavenumber per-z-mode spline evaluation.

RLZ/SLZ spectral layout: z-major, then wavenumber-interleaved.
Within each z-level block (TRAP-1: p = (k-1)*2 for k≥1):
- k=0: first b_iDim rows
- k≥1 cos: b_iDim + (k-1)*2*b_iDim + 1
- k≥1 sin: b_iDim + ((k-1)*2+1)*b_iDim + 1
"""
function _eval_unstructured_rlz(source::SpringsteelGrid, pts::AbstractMatrix{Float64}, sv::Int)
    gp = source.params
    b_iDim = gp.b_iDim
    b_kDim = gp.b_kDim
    kDim_wn = gp.iDim + gp.patchOffsetL
    npts = size(pts, 1)
    n_kslots = 1 + 2 * kDim_wn

    sc = _get_scratch_rlz(objectid(source), npts, b_kDim, n_kslots)
    r_view = view(pts, :, 1)

    a_cache = _get_ahat_cache_rlz(source, sv)
    sp0 = source.ibasis.data[1, sv]

    for z_b in 1:b_kDim
        stripe_offset = (z_b - 1) * n_kslots
        for slot in 1:n_kslots
            CubicBSpline.SItransform(sp0.params, view(a_cache, :, stripe_offset + slot),
                                     r_view, view(sc.spline_vals, 1:npts, z_b, slot))
        end
    end

    result = sc.result
    cheb_col = source.kbasis.data[sv]

    for n in 1:npts
        λ = pts[n, 2]
        z = pts[n, 3]

        for z_b in 1:b_kDim
            val = sc.spline_vals[n, z_b, 1]
            for k in 1:kDim_wn
                val += 2.0 * (sc.spline_vals[n, z_b, 2k] * cos(k * λ) +
                              sc.spline_vals[n, z_b, 2k + 1] * sin(k * λ))
            end
            cheb_col.b[z_b] = val
        end

        CAtransform!(cheb_col)
        result[n] = _cheb_eval_single(cheb_col, z)
    end
    return view(result, 1:npts)
end

# ── Unified unstructured evaluation dispatch ──────────────────────────────

"""
    evaluate_unstructured(source::SpringsteelGrid, pts::AbstractMatrix{Float64};
        vars=nothing, out_of_bounds=:nan) -> Matrix{Float64}

Evaluate the source grid's spectral representation at arbitrary points `pts`.

`pts` is `(N, ndims)` where columns are in `(i, j, k)` order matching the
source grid's coordinate system. The source grid must have valid spectral
coefficients — call `spectralTransform!` before evaluation.

# Arguments
- `vars`: variable names to evaluate (default: all)
- `out_of_bounds`: `:nan` (default), `:error`, or a numeric fill value

# Returns
`Matrix{Float64}` of size `(N, nvars)` with interpolated function values.

See also: [`interpolate_to_grid`](@ref), [`regularGridTransform`](@ref)
"""
function evaluate_unstructured(source::SpringsteelGrid, pts::AbstractMatrix{Float64};
                                vars=nothing, out_of_bounds=:nan)
    sgp = source.params
    nvars = length(sgp.vars)
    npts = size(pts, 1)
    result = fill(NaN, npts, nvars)

    # Variable selection
    if vars === nothing
        var_list = sort(collect(keys(sgp.vars)))
    else
        var_list = intersect(vars, keys(sgp.vars))
    end

    # Bounds checking
    mask = _check_bounds_nd(source, pts)
    if out_of_bounds === :error && !all(mask)
        n_oob = count(.!mask)
        throw(DomainError(n_oob, "$n_oob points are outside the source domain"))
    end

    ib_indices = findall(mask)
    isempty(ib_indices) && return result

    pts_ib = pts[ib_indices, :]

    # Determine grid type and dispatch
    i_active = !(source.ibasis isa NoBasisArray)
    j_active = !(source.jbasis isa NoBasisArray)
    k_active = !(source.kbasis isa NoBasisArray)

    i_spline  = source.ibasis isa SplineBasisArray
    j_spline  = source.jbasis isa SplineBasisArray
    j_fourier = source.jbasis isa FourierBasisArray
    k_cheb    = source.kbasis isa ChebyshevBasisArray
    k_spline  = source.kbasis isa SplineBasisArray

    for vname in var_list
        sv = sgp.vars[vname]

        vals = if i_active && !j_active && !k_active
            # 1D: R, Z, or L
            if i_spline
                _eval_unstructured_1d_spline(source, pts_ib[:, 1], sv)
            else
                error("Unstructured evaluation not implemented for this 1D basis type")
            end

        elseif i_active && j_active && !k_active
            # 2D: RR, RL, SL
            if i_spline && j_fourier
                _eval_unstructured_rl(source, pts_ib, sv)
            elseif i_spline && j_spline
                _eval_unstructured_2d_ij(source, pts_ib, sv)
            else
                error("Unstructured evaluation not implemented for this 2D basis combination")
            end

        elseif i_active && !j_active && k_active
            # 2D: RZ
            _eval_unstructured_2d_ik(source, pts_ib, sv)

        elseif i_active && j_active && k_active
            # 3D: RRR, RLZ, SLZ
            if i_spline && j_fourier && k_cheb
                _eval_unstructured_rlz(source, pts_ib, sv)
            elseif i_spline && j_spline && k_spline
                _eval_unstructured_3d_ijk(source, pts_ib, sv)
            else
                error("Unstructured evaluation not implemented for this 3D basis combination")
            end

        else
            error("Unsupported grid dimensionality for unstructured evaluation")
        end

        # Write in-bounds values to result
        for (k, idx) in enumerate(ib_indices)
            result[idx, sv] = vals[k]
        end

        # Fill OOB values
        if !all(mask) && out_of_bounds !== :nan
            fill_val = Float64(out_of_bounds)
            for idx in findall(.!mask)
                result[idx, sv] = fill_val
            end
        end
    end

    return result
end

# ── Cross-geometry interpolate_to_grid ────────────────────────────────────

"""
    interpolate_to_grid(source::SpringsteelGrid{G1,I1,J1,K1},
                        target::SpringsteelGrid{G2,I2,J2,K2};
                        vars=nothing, out_of_bounds=:nan,
                        coordinate_map=nothing) -> Matrix{Float64}

Cross-geometry interpolation. Maps target gridpoints to source coordinates
and evaluates the source spectral representation at those points.

Both grids may have different geometry types and basis types. The source grid
must have valid spectral coefficients.

# Arguments
- `coordinate_map`: function `(target_coords...) -> (source_coords...)`.
  If not provided, selects a default based on the geometry pair. Required for
  pairs without a natural default (e.g., 2D Cartesian → Spherical).
- `vars`: variable names to interpolate (default: all common variables)
- `out_of_bounds`: `:nan` (default), `:error`, or a numeric fill value

See also: [`evaluate_unstructured`](@ref), [`interpolate_to_grid!`](@ref)
"""
function interpolate_to_grid(source::SpringsteelGrid{G1,I1,J1,K1},
                             target::SpringsteelGrid{G2,I2,J2,K2};
                             vars=nothing, out_of_bounds=:nan,
                             coordinate_map=nothing) where {G1,I1,J1,K1,G2,I2,J2,K2}

    # Guard: if types match exactly, redirect to same-geometry method
    if G1 === G2 && I1 === I2 && J1 === J2 && K1 === K2
        return interpolate_to_grid(
            source::SpringsteelGrid{G1,I1,J1,K1},
            target::SpringsteelGrid{G1,I1,J1,K1};
            vars=vars, out_of_bounds=out_of_bounds)
    end

    sgp = source.params
    tgp = target.params
    nvars_t = length(tgp.vars)
    phys_dim_t = size(target.physical, 1)

    result = fill(NaN, phys_dim_t, nvars_t)
    matched = _match_vars(sgp.vars, tgp.vars, vars)
    isempty(matched) && return result

    # 1. Get target gridpoints (in target coordinate system)
    t_pts = getGridpoints(target)

    # 2. Select coordinate mapping
    inv_map = _get_cross_geometry_map(source, target, coordinate_map)

    # 3. Map target points to source coordinates
    src_pts = _map_points(inv_map, t_pts)

    # 4. Evaluate source at mapped points
    src_var_names = [sname for (sname, _) in matched]
    src_result = evaluate_unstructured(source, src_pts;
                                        vars=src_var_names, out_of_bounds=out_of_bounds)

    # 5. Copy to result with correct variable mapping
    for (sname, tname) in matched
        sv = sgp.vars[sname]
        tv = tgp.vars[tname]
        result[:, tv] .= src_result[:, sv]
    end

    return result
end

"""
    interpolate_to_grid!(source::SpringsteelGrid{G1,I1,J1,K1},
                         target::SpringsteelGrid{G2,I2,J2,K2};
                         vars=nothing, out_of_bounds=:nan,
                         coordinate_map=nothing) -> SpringsteelGrid

Cross-geometry in-place interpolation. Writes results to `target.physical[:, :, 1]`
and fills derivative slots with `NaN`.
"""
function interpolate_to_grid!(source::SpringsteelGrid{G1,I1,J1,K1},
                              target::SpringsteelGrid{G2,I2,J2,K2};
                              vars=nothing, out_of_bounds=:nan,
                              coordinate_map=nothing) where {G1,I1,J1,K1,G2,I2,J2,K2}

    # Guard: if types match exactly, redirect to same-geometry method
    if G1 === G2 && I1 === I2 && J1 === J2 && K1 === K2
        return interpolate_to_grid!(
            source::SpringsteelGrid{G1,I1,J1,K1},
            target::SpringsteelGrid{G1,I1,J1,K1};
            vars=vars, out_of_bounds=out_of_bounds)
    end

    result = interpolate_to_grid(source, target;
                                  vars=vars, out_of_bounds=out_of_bounds,
                                  coordinate_map=coordinate_map)
    target.physical[:, :, 1] .= result
    for d in 2:size(target.physical, 3)
        target.physical[:, :, d] .= NaN
    end
    return target
end

# ────────────────────────────────────────────────────────────────────────────
# R3X grid-level boundary interface
# ────────────────────────────────────────────────────────────────────────────

"""
    set_boundary_values!(grid::R_Grid, side::Symbol, var::String,
                         u0::Real, u1::Real, u2::Real)

Set inhomogeneous R3X boundary conditions on a 1D R grid.

# Arguments
- `grid`: 1D spline grid with R3X boundary condition
- `side`: `:left` or `:right`
- `var`: Variable name
- `u0, u1, u2`: Desired value, first derivative, and second derivative at the boundary

See also: [`CubicBSpline.R3X`](@ref), [`CubicBSpline.set_ahat_r3x!`](@ref)
"""
function set_boundary_values!(grid::R_Grid, side::Symbol, var::String,
                              u0::Real, u1::Real, u2::Real)
    v = grid.params.vars[var]
    CubicBSpline.set_ahat_r3x!(grid.ibasis.data[1, v], u0, u1, u2, side)
end

"""
    set_boundary_values!(grid::RR_Grid, side::Symbol, var::String,
                         u0::AbstractVector, u1::AbstractVector, u2::AbstractVector)

Set inhomogeneous R3X boundary conditions on a 2D RR grid (Spline×Spline).

Accepts physical-space arrays along the j-boundary, transforms them to j-spectral
space, and sets per-mode ahat on each i-spline.

# Arguments
- `grid`: 2D RR grid with R3X boundary condition on the i-dimension
- `side`: `:left` or `:right` (i-dimension boundary)
- `var`: Variable name
- `u0, u1, u2`: Vectors of length `jDim` with boundary values along the j-direction

See also: [`CubicBSpline.R3X`](@ref)
"""
function set_boundary_values!(grid::RR_Grid, side::Symbol, var::String,
                              u0::AbstractVector, u1::AbstractVector, u2::AbstractVector)
    v = grid.params.vars[var]
    b_jDim = size(grid.ibasis.data, 1)  # number of j spectral modes

    # Transform each BC array (u0, u1, u2) to j-spectral space using first j-spline
    bc_spectral = zeros(3, b_jDim)
    scratch_j = grid.jbasis.data[1, v]

    for (idx, bc_phys) in enumerate([u0, u1, u2])
        scratch_j.uMish .= bc_phys
        CubicBSpline.SBtransform!(scratch_j)
        CubicBSpline.SAtransform!(scratch_j)
        bc_spectral[idx, :] .= scratch_j.a[1:b_jDim]
    end

    # Set per-mode ahat on each i-spline
    for l in 1:b_jDim
        CubicBSpline.set_ahat_r3x!(grid.ibasis.data[l, v],
                                    bc_spectral[1,l], bc_spectral[2,l], bc_spectral[3,l], side)
    end
end

"""
    set_boundary_values!(grid::RZ_Grid, side::Symbol, var::String,
                         u0::AbstractVector, u1::AbstractVector, u2::AbstractVector)

Set inhomogeneous R3X boundary conditions on a 2D RZ grid (Spline×Chebyshev, i-dim only).

Accepts physical-space arrays along the k-boundary, transforms them to Chebyshev
spectral space, and sets per-mode ahat on each i-spline.

# Arguments
- `grid`: 2D RZ grid with R3X boundary condition on the i-dimension
- `side`: `:left` or `:right` (i-dimension boundary)
- `var`: Variable name
- `u0, u1, u2`: Vectors of length `kDim` with boundary values along the k-direction

See also: [`CubicBSpline.R3X`](@ref)
"""
function set_boundary_values!(grid::RZ_Grid, side::Symbol, var::String,
                              u0::AbstractVector, u1::AbstractVector, u2::AbstractVector)
    v = grid.params.vars[var]
    b_kDim = size(grid.ibasis.data, 1)  # number of Chebyshev spectral modes
    cheb = grid.kbasis.data[v]

    # Transform each BC array (u0, u1, u2) to Chebyshev spectral space
    bc_spectral = zeros(3, b_kDim)
    for (idx, bc_phys) in enumerate([u0, u1, u2])
        cheb.uMish .= bc_phys
        Chebyshev.CBtransform!(cheb)
        Chebyshev.CAtransform!(cheb)
        bc_spectral[idx, :] .= cheb.a[1:b_kDim]
    end

    # Set per-mode ahat on each i-spline
    for k in 1:b_kDim
        CubicBSpline.set_ahat_r3x!(grid.ibasis.data[k, v],
                                    bc_spectral[1,k], bc_spectral[2,k], bc_spectral[3,k], side)
    end
end

# ────────────────────────────────────────────────────────────────────────────
# Multi-patch grid connections via R3X coupling
# ────────────────────────────────────────────────────────────────────────────

"""
    COUPLING_MATRIX_2X

Exact 3×3 coupling matrix for 2:1 coarse-to-fine B-spline basis conversion.

Maps three primary (coarse) nodal amplitudes `[A_{outer}, A_{interface}, A_{inner}]`
to three secondary (fine) border coefficients `[a_{border-1}, a_{border}, a_{border+1}]`
at a patch interface where the fine grid cell width is exactly half the coarse.

The matrix arises from cubic B-spline basis evaluation at staggered node positions:
- `a_{border}` and `A_{interface}` are collocated at the interface point
- `a_{border±1}` sit at half-cell offsets relative to the coarse grid

No approximation — the values are exact for cubic B-splines with 2x refinement.
"""
const COUPLING_MATRIX_2X = [0.5   0.5   0.0;
                            0.125 0.75  0.125;
                            0.0   0.5   0.5]

"""
    COUPLING_MATRIX_1X

Identity 3×3 coupling matrix for 1:1 same-resolution B-spline basis conversion.

Used for domain decomposition where adjacent patches have the same cell width.
The border coefficients are copied directly without transformation.
"""
const COUPLING_MATRIX_1X = Float64[1 0 0; 0 1 0; 0 0 1]

"""
    _build_coupling_matrix(primary_DX, secondary_DX) -> Matrix{Float64}

Return the 3×3 coupling matrix for the given cell-width ratio.
Supports 1:1 (identity) and 2:1 (coarse-to-fine) ratios.
Throws an error for unsupported ratios.
"""
function _build_coupling_matrix(primary_DX::Float64, secondary_DX::Float64)
    ratio = primary_DX / secondary_DX
    tol = 1e-12 * max(primary_DX, secondary_DX)
    if abs(ratio - 1.0) < tol
        return copy(COUPLING_MATRIX_1X)
    elseif abs(ratio - 2.0) < tol
        return copy(COUPLING_MATRIX_2X)
    else
        throw(ArgumentError(
            "Only 1:1 and 2:1 coarse-to-fine ratios are supported, got $(ratio):1"))
    end
end

# ────────────────────────────────────────────────────────────────────────────
# PatchInterface
# ────────────────────────────────────────────────────────────────────────────

"""
    PatchInterface

Describes a directional connection between a primary (coarse/free) patch and a
secondary (fine/constrained) patch.  The primary patch has R0 (free) boundary
conditions at the interface, and the secondary has R3X to receive data.

After `gridTransform!` on the primary, call `update_interface!` to transfer
spectral coefficients to the secondary's `ahat` vector before transforming the
secondary.

# Fields
- `primary::SpringsteelGrid` — freely-evolving patch
- `secondary::SpringsteelGrid` — patch receiving boundary data via R3X
- `primary_side::Symbol` — `:left` or `:right`, which side of primary faces the interface
- `secondary_side::Symbol` — `:left` or `:right`, which side of secondary receives data
- `dimension::Symbol` — connected dimension (`:i` only for now)
- `coupling_matrix::Matrix{Float64}` — 3×3 basis conversion matrix
- `is_stacked::Bool` — `true` for interior (stacked) nest, `false` for hollow nest
- `primary_node_indices::Tuple{Int,Int,Int}` — indices into primary `.a` for coefficient extraction

See also: [`MultiPatchGrid`](@ref), [`update_interface!`](@ref),
[`PatchChain`](@ref), [`PatchEmbedded`](@ref)
"""
struct PatchInterface
    primary::SpringsteelGrid
    secondary::SpringsteelGrid
    primary_side::Symbol
    secondary_side::Symbol
    dimension::Symbol
    coupling_matrix::Matrix{Float64}
    is_stacked::Bool
    primary_node_indices::Tuple{Int,Int,Int}
end

# ── Validation helpers ─────────────────────────────────────────────────────

function _get_spline_DX(grid::SpringsteelGrid, dim::Symbol)
    if dim == :i
        return (grid.params.iMax - grid.params.iMin) / grid.params.num_cells
    else
        throw(ArgumentError("Only :i dimension is supported for patching, got :$dim"))
    end
end

function _get_domain_bounds(grid::SpringsteelGrid, dim::Symbol)
    if dim == :i
        return (grid.params.iMin, grid.params.iMax)
    else
        throw(ArgumentError("Only :i dimension is supported for patching, got :$dim"))
    end
end

function _get_spline_bDim(grid::SpringsteelGrid, dim::Symbol)
    if dim == :i
        return grid.params.b_iDim
    else
        throw(ArgumentError("Only :i dimension is supported for patching, got :$dim"))
    end
end

function _has_spline_basis(grid::SpringsteelGrid, dim::Symbol)
    if dim == :i
        return grid.ibasis isa SplineBasisArray
    elseif dim == :j
        return grid.jbasis isa SplineBasisArray
    else
        return false
    end
end

function _check_r3x_bc(grid::SpringsteelGrid, side::Symbol, dim::Symbol)
    if dim == :i
        bc_dict = (side == :left) ? grid.params.BCL : grid.params.BCR
    else
        throw(ArgumentError("Only :i dimension is supported for patching"))
    end
    for (key, val) in bc_dict
        if val isa Dict && haskey(val, "R3X")
            continue
        elseif val isa BoundaryConditions && bc_rank(val) == 3 && is_inhomogeneous(val)
            continue
        elseif val isa BoundaryConditions && bc_rank(val) == 3 && !is_inhomogeneous(val)
            # Homogeneous rank-3 (R3) — also acceptable since ahat zeros give R3 behavior
            continue
        elseif val isa Dict && haskey(val, "R3")
            # Legacy R3 dict — acceptable but ahat won't be used unless R3X
            continue
        else
            # Check if there's at least one R3X
        end
    end
    # Check that the "default" or all per-variable BCs are R3X
    has_r3x = false
    for (key, val) in bc_dict
        if val isa Dict && haskey(val, "R3X")
            has_r3x = true
        elseif val isa BoundaryConditions && bc_rank(val) == 3
            has_r3x = true
        end
    end
    if !has_r3x
        throw(ArgumentError(
            "Secondary patch must have R3X (rank-3) BC on the $side side of $dim dimension. " *
            "Got: $bc_dict"))
    end
end

function _check_r0_bc(grid::SpringsteelGrid, side::Symbol, dim::Symbol)
    if dim == :i
        bc_dict = (side == :left) ? grid.params.BCL : grid.params.BCR
    else
        throw(ArgumentError("Only :i dimension is supported for patching"))
    end
    for (key, val) in bc_dict
        if val isa Dict && (haskey(val, "R3X") || haskey(val, "R3") ||
                            haskey(val, "α1") || haskey(val, "α2"))
            throw(ArgumentError(
                "Primary patch must have R0 (free) BC on the $side side of $dim dimension. " *
                "Got constrained BC for variable '$key': $val"))
        elseif val isa BoundaryConditions && bc_rank(val) > 0
            throw(ArgumentError(
                "Primary patch must have R0 (free) BC on the $side side of $dim dimension. " *
                "Got rank-$(bc_rank(val)) BC for variable '$key'"))
        end
    end
end

function _compute_primary_node_index(grid::SpringsteelGrid, boundary_point::Float64, dim::Symbol)
    DX = _get_spline_DX(grid, dim)
    bounds = _get_domain_bounds(grid, dim)
    m = round(Int, (boundary_point - bounds[1]) / DX)
    array_idx = m + 2  # m=-1 → index 1, m=0 → index 2, etc.
    x_node = bounds[1] + m * DX
    if abs(x_node - boundary_point) > 1e-12 * DX
        throw(ArgumentError(
            "Secondary boundary at $boundary_point does not align with primary node at $x_node " *
            "(DX=$DX, offset=$(abs(x_node - boundary_point)))"))
    end
    bDim = _get_spline_bDim(grid, dim)
    if array_idx < 1 || array_idx > bDim
        throw(ArgumentError(
            "Primary node index $array_idx (m=$m) is out of bounds [1, $bDim]"))
    end
    return array_idx
end

"""
    PatchInterface(primary, secondary, primary_side, secondary_side, dimension;
                   is_stacked=false)

Construct a `PatchInterface` with full validation:
1. Both patches have `SplineBasisArray` on the connected dimension
2. Cell-width ratio is 2:1 (primary coarse, secondary fine)
3. Domain boundaries are aligned at the interface
4. Primary has R0 BC and secondary has R3X BC at the interface

For stacked (interior) nests, the primary node index at the interface is computed
automatically from domain alignment.
"""
function PatchInterface(primary::SpringsteelGrid, secondary::SpringsteelGrid,
                        primary_side::Symbol, secondary_side::Symbol,
                        dimension::Symbol; is_stacked::Bool=false)
    # Validate dimension
    if dimension != :i
        throw(ArgumentError("Only :i dimension is currently supported for patching"))
    end

    # Validate spline basis on both sides
    if !_has_spline_basis(primary, dimension)
        throw(ArgumentError("Primary patch must have SplineBasisArray on $dimension dimension"))
    end
    if !_has_spline_basis(secondary, dimension)
        throw(ArgumentError("Secondary patch must have SplineBasisArray on $dimension dimension"))
    end

    # Validate sides
    if !(primary_side in (:left, :right))
        throw(ArgumentError("primary_side must be :left or :right, got :$primary_side"))
    end
    if !(secondary_side in (:left, :right))
        throw(ArgumentError("secondary_side must be :left or :right, got :$secondary_side"))
    end

    # Validate cell-width ratio and build coupling matrix
    primary_DX = _get_spline_DX(primary, dimension)
    secondary_DX = _get_spline_DX(secondary, dimension)
    coupling_matrix = _build_coupling_matrix(primary_DX, secondary_DX)

    # Validate BCs
    if !is_stacked
        _check_r0_bc(primary, primary_side, dimension)
    end
    _check_r3x_bc(secondary, secondary_side, dimension)

    # Validate domain alignment and compute primary node indices
    p_bounds = _get_domain_bounds(primary, dimension)
    s_bounds = _get_domain_bounds(secondary, dimension)
    p_bDim = _get_spline_bDim(primary, dimension)

    if is_stacked
        # Stacked nest: secondary is interior to primary
        node_idx = _compute_primary_node_index(primary,
            (secondary_side == :left) ? s_bounds[1] : s_bounds[2], dimension)
        # Extract in ascending index order: [A_{X-1}, A_X, A_{X+1}]
        primary_node_indices = (node_idx - 1, node_idx, node_idx + 1)
    else
        # Hollow nest: interface is at domain boundary
        if primary_side == :right && secondary_side == :left
            # Primary's right boundary matches secondary's left boundary
            tol = 1e-12 * max(abs(p_bounds[2]), abs(s_bounds[1]), 1.0)
            if abs(p_bounds[2] - s_bounds[1]) > tol
                throw(ArgumentError(
                    "Domain mismatch: primary right boundary $(p_bounds[2]) != " *
                    "secondary left boundary $(s_bounds[1])"))
            end
            # Extract from primary right in ascending order: [A_{L-1}, A_L, A_{L+1}]
            primary_node_indices = (p_bDim - 2, p_bDim - 1, p_bDim)
        elseif primary_side == :left && secondary_side == :right
            # Primary's left boundary matches secondary's right boundary
            tol = 1e-12 * max(abs(p_bounds[1]), abs(s_bounds[2]), 1.0)
            if abs(p_bounds[1] - s_bounds[2]) > tol
                throw(ArgumentError(
                    "Domain mismatch: primary left boundary $(p_bounds[1]) != " *
                    "secondary right boundary $(s_bounds[2])"))
            end
            # Extract from primary left in ascending order: [A_{-1}, A_0, A_1]
            primary_node_indices = (1, 2, 3)
        else
            throw(ArgumentError(
                "For hollow nests, primary_side and secondary_side must be opposite " *
                "(right→left or left→right). Got primary=$primary_side, secondary=$secondary_side"))
        end
    end

    return PatchInterface(primary, secondary, primary_side, secondary_side,
                          dimension, coupling_matrix, is_stacked, primary_node_indices)
end

# ── Coefficient transfer ───────────────────────────────────────────────────

"""
    _extract_primary_coeffs(spline, indices) -> Vector{Float64}

Extract 3 spectral coefficients from a primary spline's `.a` vector.
"""
function _extract_primary_coeffs(spline::CubicBSpline.Spline1D,
                                 indices::Tuple{Int,Int,Int})
    return [spline.a[indices[1]], spline.a[indices[2]], spline.a[indices[3]]]
end

"""
    _write_interface_ahat!(spline, coeffs, side)

Write 3 coupled border coefficients into the secondary spline's `.ahat` vector.
- `side == :left` → write to `ahat[1:3]`
- `side == :right` → write to `ahat[end-2:end]`
"""
function _write_interface_ahat!(spline::CubicBSpline.Spline1D,
                                coeffs::Vector{Float64}, side::Symbol)
    if side == :left
        spline.ahat[1:3] .= coeffs
    elseif side == :right
        spline.ahat[end-2:end] .= coeffs
    else
        throw(ArgumentError("side must be :left or :right, got :$side"))
    end
end

"""
    update_interface!(iface::PatchInterface)

Transfer spectral coefficients from the primary patch to the secondary patch
at the interface.

Must be called after `gridTransform!` (or `SAtransform!`) on the primary and
before `gridTransform!` on the secondary.  The primary's `.a` coefficients
are extracted at the interface, multiplied by the 3×3 coupling matrix, and
written to the secondary's `.ahat` vector.

For 2D grids (e.g., RR), the transfer is performed independently for each
j-spectral mode.

# Example
```julia
gridTransform!(primary)
update_interface!(iface)
gridTransform!(secondary)
```

See also: [`PatchInterface`](@ref), [`multiGridTransform!`](@ref)
"""
function update_interface!(iface::PatchInterface)
    nvars = length(iface.secondary.params.vars)
    n_modes = size(iface.secondary.ibasis.data, 1)

    for v in 1:nvars
        for l in 1:n_modes
            primary_spline = iface.primary.ibasis.data[l, v]
            secondary_spline = iface.secondary.ibasis.data[l, v]

            A_coeffs = _extract_primary_coeffs(primary_spline, iface.primary_node_indices)
            a_border = iface.coupling_matrix * A_coeffs
            _write_interface_ahat!(secondary_spline, a_border, iface.secondary_side)
        end
    end
    return nothing
end

# ────────────────────────────────────────────────────────────────────────────
# MultiPatchGrid
# ────────────────────────────────────────────────────────────────────────────

"""
    MultiPatchGrid

Container for a set of connected patches and their interfaces.  Stores a
topologically sorted transform order so that primary patches are always
transformed before their dependents.

# Fields
- `patches::Vector{<:SpringsteelGrid}` — all patches in the multi-patch system
- `interfaces::Vector{PatchInterface}` — all inter-patch connections
- `transform_order::Vector{Vector{Int}}` — topological layers of patch indices

See also: [`PatchInterface`](@ref), [`multiGridTransform!`](@ref),
[`PatchChain`](@ref), [`PatchEmbedded`](@ref)
"""
struct MultiPatchGrid
    patches::Vector{<:SpringsteelGrid}
    interfaces::Vector{PatchInterface}
    transform_order::Vector{Vector{Int}}
end

"""
    _patch_index(mpg, grid) -> Int

Find the index of `grid` in `mpg.patches` by identity (===).
"""
function _patch_index(mpg::MultiPatchGrid, grid::SpringsteelGrid)
    for (i, p) in enumerate(mpg.patches)
        if p === grid
            return i
        end
    end
    throw(ArgumentError("Grid not found in MultiPatchGrid"))
end

"""
    _topological_sort(patches, interfaces) -> Vector{Vector{Int}}

Compute a topological ordering of patches based on interface dependencies.
Returns layers: patches in each layer can be transformed in parallel;
later layers depend on earlier ones.

Throws an error if a cycle is detected (a patch cannot be both primary and
secondary in a way that creates circular dependencies).
"""
function _topological_sort(patches::Vector{<:SpringsteelGrid},
                           interfaces::Vector{PatchInterface})
    n = length(patches)

    # Build index lookup
    idx_map = Dict{UInt, Int}()
    for (i, p) in enumerate(patches)
        idx_map[objectid(p)] = i
    end

    # Build dependency graph: secondary depends on primary
    deps = [Set{Int}() for _ in 1:n]
    for iface in interfaces
        p_idx = get(idx_map, objectid(iface.primary), 0)
        s_idx = get(idx_map, objectid(iface.secondary), 0)
        if p_idx == 0
            throw(ArgumentError("Interface primary patch not found in patches list"))
        end
        if s_idx == 0
            throw(ArgumentError("Interface secondary patch not found in patches list"))
        end
        push!(deps[s_idx], p_idx)
    end

    # Kahn's algorithm
    in_degree = [length(d) for d in deps]
    layers = Vector{Vector{Int}}()
    remaining = Set(1:n)

    while !isempty(remaining)
        # Find all patches with no remaining dependencies
        layer = Int[]
        for i in remaining
            if in_degree[i] == 0
                push!(layer, i)
            end
        end

        if isempty(layer)
            throw(ArgumentError(
                "Cycle detected in patch dependencies — cannot determine transform order"))
        end

        sort!(layer)
        push!(layers, layer)

        for i in layer
            delete!(remaining, i)
            # Remove this node from all dependents
            for j in remaining
                if i in deps[j]
                    in_degree[j] -= 1
                end
            end
        end
    end

    return layers
end

"""
    MultiPatchGrid(patches, interfaces)

Construct a `MultiPatchGrid` with automatic topological ordering.

All patches referenced by interfaces must be in the `patches` vector.
Throws an error if circular dependencies are detected.
"""
function MultiPatchGrid(patches::Vector{<:SpringsteelGrid},
                        interfaces::Vector{PatchInterface})
    transform_order = _topological_sort(patches, interfaces)
    return MultiPatchGrid(patches, interfaces, transform_order)
end

"""
    multiGridTransform!(mpg::MultiPatchGrid)

Perform a coupled inverse transform across all patches in topological order.

For each layer:
1. Call `gridTransform!` on all patches in the layer (independent)
2. Transfer interface coefficients for interfaces whose primary is in this layer

This ensures that primary patches are fully transformed before their
coefficients are read by `update_interface!`.

# Example
```julia
# Forward transform on all patches first
spectralTransform!(mpg.patches[1])
spectralTransform!(mpg.patches[2])
spectralTransform!(mpg.patches[3])

# Coupled inverse transform
multiGridTransform!(mpg)
```

See also: [`MultiPatchGrid`](@ref), [`update_interface!`](@ref)
"""
function multiGridTransform!(mpg::MultiPatchGrid)
    idx_map = Dict{UInt, Int}()
    for (i, p) in enumerate(mpg.patches)
        idx_map[objectid(p)] = i
    end

    for layer in mpg.transform_order
        # Transform all patches in this layer
        for idx in layer
            gridTransform!(mpg.patches[idx])
        end

        # Transfer interfaces where primary is in this layer
        for iface in mpg.interfaces
            p_idx = idx_map[objectid(iface.primary)]
            if p_idx in layer
                update_interface!(iface)
            end
        end
    end
    return nothing
end

# ────────────────────────────────────────────────────────────────────────────
# Topology constructors
# ────────────────────────────────────────────────────────────────────────────

"""
    PatchChain(grids; dimension=:i)

Create a `MultiPatchGrid` from a sequence of grids connected end-to-end.

Grids are provided in spatial order (e.g., left to right).  At each interface
between adjacent grids, the coarser grid is automatically selected as primary
(R0 BC side) and the finer grid as secondary (R3X BC side).  For 1:1 ratio
(same resolution), the left grid is primary by convention.

Supports asymmetric refinement chains like `8-4-2-1-2-4-8` DX, where the
primary/secondary direction flips at the finest grid.

# Arguments
- `grids::Vector{<:SpringsteelGrid}`: Two or more grids in spatial order.
- `dimension::Symbol=:i`: Connected dimension (`:i` only for now).

# BC requirements
- At each interface: the primary side must have R0, the secondary side R3X.
- End grids may have any user-chosen BCs on their outer (non-interface) edges.

# Example
```julia
# 3-grid chain: coarse — fine — coarse
mpg = PatchChain([left_grid, center_grid, right_grid])
multiGridTransform!(mpg)

# 7-grid chain: 8-4-2-1-2-4-8 DX
mpg = PatchChain([g8a, g4a, g2a, g1, g2b, g4b, g8b])
```

See also: [`PatchEmbedded`](@ref), [`PatchInterface`](@ref), [`MultiPatchGrid`](@ref)
"""
function PatchChain(grids::Vector{<:SpringsteelGrid}; dimension::Symbol=:i)
    n = length(grids)
    if n < 2
        throw(ArgumentError("PatchChain requires at least 2 grids, got $n"))
    end

    interfaces = PatchInterface[]
    for k in 1:(n-1)
        DX_left = _get_spline_DX(grids[k], dimension)
        DX_right = _get_spline_DX(grids[k+1], dimension)

        if DX_left >= DX_right
            # Left is coarser (or same resolution) → left is primary
            primary = grids[k]
            secondary = grids[k+1]
            primary_side = :right
            secondary_side = :left
        else
            # Right is coarser → right is primary
            primary = grids[k+1]
            secondary = grids[k]
            primary_side = :left
            secondary_side = :right
        end

        push!(interfaces, PatchInterface(primary, secondary,
                                         primary_side, secondary_side, dimension))
    end

    return MultiPatchGrid(grids, interfaces)
end

"""
    PatchEmbedded(grids; dimension=:i)

Create a `MultiPatchGrid` from a sequence of grids nested inside each other.

Grids are provided from outermost to innermost.  Each inner grid is spatially
contained within its predecessor and receives R3X boundary data from it on
both sides.  The outermost grid keeps its user-specified BCs.

All inner grids must have strictly finer resolution than their parent
(1:1 ratio is not allowed for embedded patches).

# Arguments
- `grids::Vector{<:SpringsteelGrid}`: Two or more grids, outermost first.
- `dimension::Symbol=:i`: Connected dimension (`:i` only for now).

# BC requirements
- Inner grids must have R3X on both sides in the connected dimension.
- The outermost grid may have any user-chosen BCs.

# Example
```julia
# 2-level embedding
mpg = PatchEmbedded([coarse_grid, fine_grid])

# 3-level embedding
mpg = PatchEmbedded([coarse_grid, medium_grid, fine_grid])
multiGridTransform!(mpg)
```

See also: [`PatchChain`](@ref), [`PatchInterface`](@ref), [`MultiPatchGrid`](@ref)
"""
function PatchEmbedded(grids::Vector{<:SpringsteelGrid}; dimension::Symbol=:i)
    n = length(grids)
    if n < 2
        throw(ArgumentError("PatchEmbedded requires at least 2 grids, got $n"))
    end

    interfaces = PatchInterface[]
    for k in 1:(n-1)
        outer = grids[k]
        inner = grids[k+1]

        # Verify strict refinement (no 1:1)
        DX_outer = _get_spline_DX(outer, dimension)
        DX_inner = _get_spline_DX(inner, dimension)
        ratio = DX_outer / DX_inner
        tol = 1e-12 * max(DX_outer, DX_inner)
        if abs(ratio - 1.0) < tol
            throw(ArgumentError(
                "PatchEmbedded requires refinement at each level. " *
                "Grids $k and $(k+1) have the same resolution (DX=$DX_outer)"))
        end

        # Create left and right interfaces (both stacked)
        push!(interfaces, PatchInterface(outer, inner, :right, :left, dimension;
                                         is_stacked=true))
        push!(interfaces, PatchInterface(outer, inner, :left, :right, dimension;
                                         is_stacked=true))
    end

    return MultiPatchGrid(grids, interfaces)
end

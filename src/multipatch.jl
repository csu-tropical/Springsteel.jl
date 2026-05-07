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
# Spline-coupling scheme trait
# ────────────────────────────────────────────────────────────────────────────

"""
    _is_reused_spline_grid(grid) -> Bool

Return `true` if the grid reuses spline objects across wavenumbers.
RL, RLZ, SL, SLZ grids have only 3 ibasis splines (k=0, k-real, k-imag)
shared across all wavenumbers, unlike R/RR/RRR which have dedicated splines
per spectral mode.
"""
function _is_reused_spline_grid(grid::SpringsteelGrid)
    return (grid.jbasis isa FourierBasisArray) && (grid.ibasis isa SplineBasisArray)
end

"""
    _spline_coupling_scheme(grid::SpringsteelGrid) -> Symbol

Return the spline-coupling scheme used by `gridTransform!` and
`update_interface!`:

- `:per_mode`   — dedicated splines per j/k spectral mode (R, RR, RRR, RZ).
- `:reused_2d`  — Fourier λ + reused splines, no Chebyshev (RL, SL).
- `:reused_3d`  — Fourier λ + reused splines + Chebyshev z (RLZ, SLZ).

Single extension point: a new spline-coupled geometry only needs a method on
this function, not on `compute_interface_payload` / `apply_interface_payload!`.
"""
function _spline_coupling_scheme(grid::SpringsteelGrid)
    if _is_reused_spline_grid(grid)
        return grid.kbasis isa NoBasisArray ? :reused_2d : :reused_3d
    else
        return :per_mode
    end
end

# ────────────────────────────────────────────────────────────────────────────
# PatchInterfaceMetadata, InterfacePayload, PatchInterface
# ────────────────────────────────────────────────────────────────────────────

"""
    PatchInterfaceMetadata

Side / coupling / topology data plus the minimum secondary descriptors needed
to size and dispatch interface payloads.  Safe to replicate across processes;
carries no grid pointers.

# Fields
- `primary_side::Symbol`, `secondary_side::Symbol`, `dimension::Symbol`
- `coupling_matrix::Matrix{Float64}` — 3×3 basis-conversion matrix
- `is_stacked::Bool`
- `primary_node_indices::Tuple{Int,Int,Int}`
- `scheme::Symbol` — `:per_mode | :reused_2d | :reused_3d`; both ends agree
- `nvars::Int`
- `n_modes::Int` — `:per_mode` mode count; `1` otherwise
- `kDim_couple::Int` — `min(kDim_p, kDim_s)` for `:reused_*`; `0` otherwise
- `b_kDim::Int` — `:reused_3d` Chebyshev levels; `0` otherwise
- `n_slots::Int` — secondary registry slot count (also the payload's slot dim)
- `b_iDim_p::Int`, `b_iDim_s::Int` — primary/secondary spline lengths along `:i`
- `kDim_s::Int` — secondary kDim along Fourier dim (for `:reused_*`)
"""
struct PatchInterfaceMetadata
    primary_side::Symbol
    secondary_side::Symbol
    dimension::Symbol
    coupling_matrix::Matrix{Float64}
    is_stacked::Bool
    primary_node_indices::Tuple{Int,Int,Int}
    scheme::Symbol
    nvars::Int
    n_modes::Int
    kDim_couple::Int
    b_kDim::Int
    n_slots::Int
    b_iDim_p::Int
    b_iDim_s::Int
    kDim_s::Int
end

"""
    InterfacePayload

Wire-format buffer carrying coupled border coefficients from a primary to a
secondary patch.  Layout is opaque to callers; consumed only by
[`apply_interface_payload!`](@ref).  Plain `Array{Float64}` for trivial
`Serialization.serialize` round-trip.

# Fields
- `scheme::Symbol` — `:per_mode | :reused_2d | :reused_3d`
- `side::Symbol`   — secondary side (`:left | :right`)
- `nvars::Int`
- `n_slots::Int`   — `n_modes` for `:per_mode`; total slot count otherwise
- `border::Array{Float64,3}` — `(3, n_slots, nvars)`
"""
struct InterfacePayload
    scheme::Symbol
    side::Symbol
    nvars::Int
    n_slots::Int
    border::Array{Float64,3}
end

function _allocate_payload(meta::PatchInterfaceMetadata)
    border = zeros(Float64, 3, meta.n_slots, meta.nvars)
    return InterfacePayload(meta.scheme, meta.secondary_side,
                            meta.nvars, meta.n_slots, border)
end

"""
    PatchInterface

Describes a directional connection between a primary (coarse/free) patch and a
secondary (fine/constrained) patch.  The primary patch has R0 (free) boundary
conditions at the interface, and the secondary has R3X to receive data.

After `gridTransform!` on the primary, call `update_interface!` to transfer
spectral coefficients to the secondary's `ahat` vector before transforming the
secondary.

# Fields
- `metadata::PatchInterfaceMetadata` — side / coupling / topology info
- `primary::SpringsteelGrid`   — freely-evolving patch
- `secondary::SpringsteelGrid` — patch receiving boundary data via R3X
- `_a_extract`, `_a_border`   — per-call 3-element scratch buffers
- `_payload_buf::InterfacePayload` — preallocated payload (zero-alloc path)

For backwards compatibility, the legacy fields `primary_side`, `secondary_side`,
`dimension`, `coupling_matrix`, `is_stacked`, `primary_node_indices` are forwarded
to `metadata` via `Base.getproperty`.

See also: [`MultiPatchGrid`](@ref), [`update_interface!`](@ref),
[`PatchChain`](@ref), [`PatchEmbedded`](@ref)
"""
struct PatchInterface{P<:SpringsteelGrid, S<:SpringsteelGrid}
    metadata::PatchInterfaceMetadata
    primary::P
    secondary::S
    # Per-call scratch buffers for the 3-element extract / coupling matvec
    _a_extract::Vector{Float64}
    _a_border::Vector{Float64}
    # Preallocated payload buffer to keep the single-process update_interface!
    # path zero-alloc (as recorded in project memory).
    _payload_buf::InterfacePayload
end

# Forward legacy field names (primary_side, coupling_matrix, etc.) onto
# metadata.  The compiler inlines this when the property name is a literal,
# so hot paths inside this file pay no extra cost.
function Base.getproperty(iface::PatchInterface, name::Symbol)
    if name === :primary_side    || name === :secondary_side       ||
       name === :dimension       || name === :coupling_matrix      ||
       name === :is_stacked      || name === :primary_node_indices
        return getfield(getfield(iface, :metadata), name)
    else
        return getfield(iface, name)
    end
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

    # Build metadata (carries all wire-relevant descriptors)
    p_scheme = _spline_coupling_scheme(primary)
    s_scheme = _spline_coupling_scheme(secondary)
    p_scheme === s_scheme || throw(ArgumentError(
        "Primary scheme :$p_scheme ≠ secondary scheme :$s_scheme"))

    nvars = length(secondary.params.vars)
    b_iDim_p = primary.params.b_iDim
    b_iDim_s = secondary.params.b_iDim

    if p_scheme === :per_mode
        n_modes = size(secondary.ibasis.data, 1)
        n_slots = n_modes
        kDim_couple = 0
        b_kDim = 0
        kDim_s = 0
    elseif p_scheme === :reused_2d
        n_modes = 1
        kDim_p = primary.params.iDim + primary.params.patchOffsetL
        kDim_s = secondary.params.iDim + secondary.params.patchOffsetL
        kDim_couple = min(kDim_p, kDim_s)
        b_kDim = 0
        n_slots = 2 + 2 * kDim_s
    else  # :reused_3d
        n_modes = 1
        kDim_p = primary.params.iDim + primary.params.patchOffsetL
        kDim_s = secondary.params.iDim + secondary.params.patchOffsetL
        kDim_couple = min(kDim_p, kDim_s)
        b_kDim = primary.params.b_kDim
        n_slots = b_kDim * (1 + 2 * kDim_s)
    end

    metadata = PatchInterfaceMetadata(
        primary_side, secondary_side, dimension,
        coupling_matrix, is_stacked, primary_node_indices,
        p_scheme, nvars, n_modes, kDim_couple, b_kDim,
        n_slots, b_iDim_p, b_iDim_s, kDim_s,
    )

    payload = _allocate_payload(metadata)

    return PatchInterface(metadata, primary, secondary,
                          zeros(Float64, 3), zeros(Float64, 3), payload)
end

# ── Coefficient transfer ───────────────────────────────────────────────────

"""
    _extract_primary_coeffs(spline, indices) -> Vector{Float64}

Extract 3 spectral coefficients from a primary spline's `.a` vector.
"""
function _extract_primary_coeffs!(buf::Vector{Float64},
                                  spline::CubicBSpline.Spline1D,
                                  indices::Tuple{Int,Int,Int})
    @inbounds buf[1] = spline.a[indices[1]]
    @inbounds buf[2] = spline.a[indices[2]]
    @inbounds buf[3] = spline.a[indices[3]]
    return buf
end

"""
    _write_interface_ahat!(spline, coeffs, side)

Write 3 coupled border coefficients into the secondary spline's `.ahat` vector.
- `side == :left` → write to `ahat[1:3]`
- `side == :right` → write to `ahat[end-2:end]`
"""
function _write_interface_ahat!(spline::CubicBSpline.Spline1D,
                                coeffs::AbstractVector{<:Real}, side::Symbol)
    if side == :left
        @inbounds spline.ahat[1] = coeffs[1]
        @inbounds spline.ahat[2] = coeffs[2]
        @inbounds spline.ahat[3] = coeffs[3]
    elseif side == :right
        n = length(spline.ahat)
        @inbounds spline.ahat[n - 2] = coeffs[1]
        @inbounds spline.ahat[n - 1] = coeffs[2]
        @inbounds spline.ahat[n]     = coeffs[3]
    else
        throw(ArgumentError("side must be :left or :right, got :$side"))
    end
end

# ────────────────────────────────────────────────────────────────────────────
# Distributed-friendly compute / apply split
# ────────────────────────────────────────────────────────────────────────────
#
# `compute_interface_payload[!]` reads only the primary grid and writes a
# self-contained `InterfacePayload`.  `apply_interface_payload!` reads only
# the secondary grid and the payload.  The payload is plain Float64 data —
# it can be sent across a process boundary via `Serialization.serialize`.
#
# `update_interface!(iface)` is now a thin wrapper that runs both halves
# in-process using the preallocated payload buffer on `iface`, preserving
# the legacy zero-allocation invariant.

# ── Compute kernels (write into payload.border) ────────────────────────────

function _fill_payload_per_mode!(border::Array{Float64,3},
                                 meta::PatchInterfaceMetadata,
                                 primary::SpringsteelGrid,
                                 sx::Vector{Float64},
                                 sb::Vector{Float64})
    nvars   = meta.nvars
    n_modes = meta.n_modes
    @inbounds for v in 1:nvars
        for l in 1:n_modes
            primary_spline = primary.ibasis.data[l, v]
            _extract_primary_coeffs!(sx, primary_spline, meta.primary_node_indices)
            mul!(sb, meta.coupling_matrix, sx)
            border[1, l, v] = sb[1]
            border[2, l, v] = sb[2]
            border[3, l, v] = sb[3]
        end
    end
    return nothing
end

function _fill_payload_reused_2d!(border::Array{Float64,3},
                                  meta::PatchInterfaceMetadata,
                                  primary::SpringsteelGrid,
                                  sx::Vector{Float64},
                                  sb::Vector{Float64})
    nvars       = meta.nvars
    b_iDim_p    = meta.b_iDim_p
    kDim_couple = meta.kDim_couple
    spec_stride = size(primary.spectral, 1)

    @inbounds for v in 1:nvars
        spec_col_off = (v - 1) * spec_stride

        # k=0 → registry slot 0 → payload column 1
        spline_k0 = primary.ibasis.data[1, v]
        copyto!(spline_k0.b, 1, primary.spectral, spec_col_off + 1, b_iDim_p)
        CubicBSpline.SAtransform!(spline_k0)
        _extract_primary_coeffs!(sx, spline_k0, meta.primary_node_indices)
        mul!(sb, meta.coupling_matrix, sx)
        border[1, 1, v] = sb[1]
        border[2, 1, v] = sb[2]
        border[3, 1, v] = sb[3]

        # k=1..kDim_couple, RL convention p = k*2
        for k in 1:kDim_couple
            p = k * 2

            # Real
            p1_r = (p - 1) * b_iDim_p + 1
            spline_real = primary.ibasis.data[2, v]
            copyto!(spline_real.b, 1, primary.spectral, spec_col_off + p1_r, b_iDim_p)
            CubicBSpline.SAtransform!(spline_real)
            _extract_primary_coeffs!(sx, spline_real, meta.primary_node_indices)
            mul!(sb, meta.coupling_matrix, sx)
            border[1, p + 1, v] = sb[1]
            border[2, p + 1, v] = sb[2]
            border[3, p + 1, v] = sb[3]

            # Imag
            p1_i = p * b_iDim_p + 1
            spline_imag = primary.ibasis.data[3, v]
            copyto!(spline_imag.b, 1, primary.spectral, spec_col_off + p1_i, b_iDim_p)
            CubicBSpline.SAtransform!(spline_imag)
            _extract_primary_coeffs!(sx, spline_imag, meta.primary_node_indices)
            mul!(sb, meta.coupling_matrix, sx)
            border[1, p + 2, v] = sb[1]
            border[2, p + 2, v] = sb[2]
            border[3, p + 2, v] = sb[3]
        end
    end
    return nothing
end

function _fill_payload_reused_3d!(border::Array{Float64,3},
                                  meta::PatchInterfaceMetadata,
                                  primary::SpringsteelGrid,
                                  sx::Vector{Float64},
                                  sb::Vector{Float64})
    nvars       = meta.nvars
    b_iDim_p    = meta.b_iDim_p
    b_kDim      = meta.b_kDim
    kDim_couple = meta.kDim_couple
    kDim_s      = meta.kDim_s
    slots_per_z = 1 + 2 * kDim_s
    kDim_p      = primary.params.iDim + primary.params.patchOffsetL
    spec_stride = size(primary.spectral, 1)

    @inbounds for v in 1:nvars
        spec_col_off = (v - 1) * spec_stride
        for z_b in 1:b_kDim
            base_p      = (z_b - 1) * b_iDim_p * (1 + kDim_p * 2)
            z_slot_base = (z_b - 1) * slots_per_z

            # k=0 → registry slot z_slot_base → payload column z_slot_base+1
            r1 = base_p + 1
            spline_k0 = primary.ibasis.data[1, v]
            copyto!(spline_k0.b, 1, primary.spectral, spec_col_off + r1, b_iDim_p)
            CubicBSpline.SAtransform!(spline_k0)
            _extract_primary_coeffs!(sx, spline_k0, meta.primary_node_indices)
            mul!(sb, meta.coupling_matrix, sx)
            border[1, z_slot_base + 1, v] = sb[1]
            border[2, z_slot_base + 1, v] = sb[2]
            border[3, z_slot_base + 1, v] = sb[3]

            # k=1..kDim_couple, RLZ convention p = (k-1)*2 (TRAP-1)
            r2 = base_p + b_iDim_p
            for k in 1:kDim_couple
                p = (k - 1) * 2

                # Real
                p1_r = r2 + 1 + p * b_iDim_p
                spline_real = primary.ibasis.data[2, v]
                copyto!(spline_real.b, 1, primary.spectral, spec_col_off + p1_r, b_iDim_p)
                CubicBSpline.SAtransform!(spline_real)
                _extract_primary_coeffs!(sx, spline_real, meta.primary_node_indices)
                mul!(sb, meta.coupling_matrix, sx)
                slot_real = z_slot_base + 1 + p
                border[1, slot_real + 1, v] = sb[1]
                border[2, slot_real + 1, v] = sb[2]
                border[3, slot_real + 1, v] = sb[3]

                # Imag
                p1_i = p1_r + b_iDim_p
                spline_imag = primary.ibasis.data[3, v]
                copyto!(spline_imag.b, 1, primary.spectral, spec_col_off + p1_i, b_iDim_p)
                CubicBSpline.SAtransform!(spline_imag)
                _extract_primary_coeffs!(sx, spline_imag, meta.primary_node_indices)
                mul!(sb, meta.coupling_matrix, sx)
                slot_imag = slot_real + 1
                border[1, slot_imag + 1, v] = sb[1]
                border[2, slot_imag + 1, v] = sb[2]
                border[3, slot_imag + 1, v] = sb[3]
            end
        end
    end
    return nothing
end

"""
    compute_interface_payload!(payload, meta, primary;
                               _a_extract, _a_border) -> InterfacePayload

In-place variant of [`compute_interface_payload`](@ref).  Zeroes `payload.border`
and writes coupled border coefficients from `primary` into it.  Used by
`update_interface!(iface)` to keep the single-process path zero-alloc.
"""
function compute_interface_payload!(payload::InterfacePayload,
                                    meta::PatchInterfaceMetadata,
                                    primary::SpringsteelGrid;
                                    _a_extract::Vector{Float64} = zeros(Float64, 3),
                                    _a_border::Vector{Float64}  = zeros(Float64, 3))
    payload.scheme === meta.scheme || throw(ArgumentError(
        "Payload scheme :$(payload.scheme) ≠ metadata scheme :$(meta.scheme)"))
    payload.n_slots == meta.n_slots || throw(ArgumentError(
        "Payload n_slots $(payload.n_slots) ≠ metadata n_slots $(meta.n_slots)"))
    fill!(payload.border, 0.0)
    if meta.scheme === :per_mode
        _fill_payload_per_mode!(payload.border, meta, primary, _a_extract, _a_border)
    elseif meta.scheme === :reused_2d
        _fill_payload_reused_2d!(payload.border, meta, primary, _a_extract, _a_border)
    else  # :reused_3d
        _fill_payload_reused_3d!(payload.border, meta, primary, _a_extract, _a_border)
    end
    return payload
end

"""
    compute_interface_payload(meta, primary; _a_extract, _a_border) -> InterfacePayload

Read coupled border coefficients from `primary` and return an
[`InterfacePayload`](@ref) that can be serialized and shipped to the process
that owns the secondary patch.  No secondary grid is required.

The returned payload is consumed by [`apply_interface_payload!`](@ref).
"""
function compute_interface_payload(meta::PatchInterfaceMetadata,
                                   primary::SpringsteelGrid;
                                   _a_extract::Vector{Float64} = zeros(Float64, 3),
                                   _a_border::Vector{Float64}  = zeros(Float64, 3))
    payload = _allocate_payload(meta)
    compute_interface_payload!(payload, meta, primary;
                               _a_extract=_a_extract, _a_border=_a_border)
    return payload
end

# ── Apply kernels (read payload.border, write secondary) ───────────────────
#
# Apply mirrors the legacy kernels exactly.  In particular, `spline.ahat` is
# *not* zero-filled before writing the 3 border floats: non-border positions
# are zero on first allocation and are the caller's invariant elsewhere.
# Embedded both-sides interfaces rely on this so that a left-side write
# followed by a right-side write deposits *both* sides' borders into the
# registry's per-wavenumber `ahat` buffer.

function _apply_payload_per_mode!(meta::PatchInterfaceMetadata,
                                  secondary::SpringsteelGrid,
                                  payload::InterfacePayload)
    side = meta.secondary_side
    @inbounds for v in 1:meta.nvars
        for l in 1:meta.n_modes
            sec_spline = secondary.ibasis.data[l, v]
            _write_interface_ahat!(sec_spline, view(payload.border, :, l, v), side)
        end
    end
    return nothing
end

function _apply_payload_reused_2d!(meta::PatchInterfaceMetadata,
                                   secondary::SpringsteelGrid,
                                   payload::InterfacePayload)
    side        = meta.secondary_side
    nvars       = meta.nvars
    kDim_couple = meta.kDim_couple
    n_slots_reg = meta.n_slots                 # 2 + 2*kDim_s
    @inbounds for v in 1:nvars
        sec_spline_k0 = secondary.ibasis.data[1, v]
        _write_interface_ahat!(sec_spline_k0, view(payload.border, :, 1, v), side)
        _set_wavenumber_ahat!(secondary, v, 0, sec_spline_k0.ahat, n_slots_reg)

        for k in 1:kDim_couple
            p = k * 2

            sec_spline_real = secondary.ibasis.data[2, v]
            _write_interface_ahat!(sec_spline_real,
                                   view(payload.border, :, p + 1, v), side)
            _set_wavenumber_ahat!(secondary, v, p,
                                  sec_spline_real.ahat, n_slots_reg)

            sec_spline_imag = secondary.ibasis.data[3, v]
            _write_interface_ahat!(sec_spline_imag,
                                   view(payload.border, :, p + 2, v), side)
            _set_wavenumber_ahat!(secondary, v, p + 1,
                                  sec_spline_imag.ahat, n_slots_reg)
        end
    end
    return nothing
end

function _apply_payload_reused_3d!(meta::PatchInterfaceMetadata,
                                   secondary::SpringsteelGrid,
                                   payload::InterfacePayload)
    side        = meta.secondary_side
    nvars       = meta.nvars
    b_kDim      = meta.b_kDim
    kDim_couple = meta.kDim_couple
    kDim_s      = meta.kDim_s
    slots_per_z = 1 + 2 * kDim_s
    n_slots_reg = meta.n_slots                 # b_kDim * slots_per_z
    @inbounds for v in 1:nvars
        for z_b in 1:b_kDim
            z_slot_base = (z_b - 1) * slots_per_z

            sec_spline_k0 = secondary.ibasis.data[1, v]
            _write_interface_ahat!(sec_spline_k0,
                                   view(payload.border, :, z_slot_base + 1, v), side)
            _set_wavenumber_ahat!(secondary, v, z_slot_base + 0,
                                  sec_spline_k0.ahat, n_slots_reg)

            for k in 1:kDim_couple
                p = (k - 1) * 2
                slot_real = z_slot_base + 1 + p

                sec_spline_real = secondary.ibasis.data[2, v]
                _write_interface_ahat!(sec_spline_real,
                                       view(payload.border, :, slot_real + 1, v), side)
                _set_wavenumber_ahat!(secondary, v, slot_real,
                                      sec_spline_real.ahat, n_slots_reg)

                slot_imag = slot_real + 1
                sec_spline_imag = secondary.ibasis.data[3, v]
                _write_interface_ahat!(sec_spline_imag,
                                       view(payload.border, :, slot_imag + 1, v), side)
                _set_wavenumber_ahat!(secondary, v, slot_imag,
                                      sec_spline_imag.ahat, n_slots_reg)
            end
        end
    end
    return nothing
end

"""
    apply_interface_payload!(meta, secondary, payload)

Apply a coupled-border `payload` to `secondary`.  Mirrors the writes that
[`update_interface!`](@ref) used to perform — `spline.ahat` borders for
per-mode grids, plus the per-wavenumber registry for reused-spline grids.

The payload's scheme/side/nvars/n_slots must match `meta`.
"""
function apply_interface_payload!(meta::PatchInterfaceMetadata,
                                  secondary::SpringsteelGrid,
                                  payload::InterfacePayload)
    payload.side    === meta.secondary_side || throw(ArgumentError(
        "Payload side :$(payload.side) ≠ metadata secondary_side :$(meta.secondary_side)"))
    payload.scheme  === meta.scheme || throw(ArgumentError(
        "Payload scheme :$(payload.scheme) ≠ metadata scheme :$(meta.scheme)"))
    payload.nvars   == meta.nvars || throw(ArgumentError(
        "Payload nvars $(payload.nvars) ≠ metadata nvars $(meta.nvars)"))
    payload.n_slots == meta.n_slots || throw(ArgumentError(
        "Payload n_slots $(payload.n_slots) ≠ metadata n_slots $(meta.n_slots)"))
    if meta.scheme === :per_mode
        _apply_payload_per_mode!(meta, secondary, payload)
    elseif meta.scheme === :reused_2d
        _apply_payload_reused_2d!(meta, secondary, payload)
    else  # :reused_3d
        _apply_payload_reused_3d!(meta, secondary, payload)
    end
    return nothing
end

"""
    update_interface!(iface::PatchInterface)

Transfer spectral coefficients from the primary patch to the secondary patch
at the interface.

Equivalent to `compute_interface_payload!` followed by `apply_interface_payload!`,
using the preallocated payload buffer on `iface`.  Single-process callers
should keep using `update_interface!`; distributed drivers should split the
two halves and ship the [`InterfacePayload`](@ref) over the wire.

Must be called after `gridTransform!` on the primary and before
`gridTransform!` on the secondary.

# Example
```julia
gridTransform!(primary)
update_interface!(iface)
gridTransform!(secondary)
```

See also: [`compute_interface_payload`](@ref), [`apply_interface_payload!`](@ref),
[`PatchInterface`](@ref), [`multiGridTransform!`](@ref)
"""
function update_interface!(iface::PatchInterface)
    meta = iface.metadata
    compute_interface_payload!(iface._payload_buf, meta, iface.primary;
                               _a_extract=iface._a_extract,
                               _a_border=iface._a_border)
    apply_interface_payload!(meta, iface.secondary, iface._payload_buf)
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
    interfaces::Vector{<:PatchInterface}
    transform_order::Vector{Vector{Int}}
    # Precomputed objectid → patch index for use by multiGridTransform!.
    # Built once at construction; avoids per-call Dict allocation.
    _patch_idx_map::Dict{UInt, Int}
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
                           interfaces::Vector{<:PatchInterface})
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
                        interfaces::Vector{<:PatchInterface})
    transform_order = _topological_sort(patches, interfaces)
    idx_map = Dict{UInt, Int}()
    for (i, p) in enumerate(patches)
        idx_map[objectid(p)] = i
    end
    return MultiPatchGrid(patches, interfaces, transform_order, idx_map)
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
    idx_map = mpg._patch_idx_map
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

    return MultiPatchGrid(grids, _narrow(interfaces))
end

_narrow(v::Vector{T}) where {T} = identity.(v)

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

    return MultiPatchGrid(grids, _narrow(interfaces))
end

# ────────────────────────────────────────────────────────────────────────────
# SpringsteelMultiGrid and createMultiGrid factory
# ────────────────────────────────────────────────────────────────────────────

"""
    SpringsteelMultiGrid <: AbstractMultiGrid

High-level multi-patch grid container, analogous to `SpringsteelGrid` for single
grids.  Wraps a [`MultiPatchGrid`](@ref) (low-level typed internals) plus the
construction config Dict for serialization and query.

Created by [`createMultiGrid`](@ref).

# Fields
- `config::Dict{Symbol, Any}` — construction configuration
- `mpg::MultiPatchGrid` — patches, interfaces, and topological transform order
"""
struct SpringsteelMultiGrid <: AbstractMultiGrid
    config::Dict{Symbol, Any}
    mpg::MultiPatchGrid
end

function spectralTransform!(mg::SpringsteelMultiGrid)
    for p in mg.mpg.patches
        spectralTransform!(p)
    end
    return nothing
end

function multiGridTransform!(mg::SpringsteelMultiGrid)
    multiGridTransform!(mg.mpg)
    return nothing
end

# ── Config validation ─────────────────────────────────────────────────────

const _REQUIRED_KEYS = Set([:topology, :geometry, :cells, :vars, :BCL, :BCR])
const _CYLINDRICAL_GEOMETRIES = Set(["RL", "RLZ", "SL", "SLZ"])

function _validate_multigrid_config(config::Dict{Symbol, Any})
    for key in _REQUIRED_KEYS
        haskey(config, key) || throw(ArgumentError("Missing required config key: :$key"))
    end
    topology = config[:topology]
    topology in (:chain, :embedded) ||
        throw(ArgumentError("Unknown topology: $topology. Must be :chain or :embedded"))

    if topology == :chain
        haskey(config, :boundaries) ||
            throw(ArgumentError("Chain topology requires :boundaries key"))
        boundaries = config[:boundaries]
        length(boundaries) >= 3 ||
            throw(ArgumentError("Chain requires at least 3 boundaries (2 patches), got $(length(boundaries))"))
        issorted(boundaries; lt = <) ||
            throw(ArgumentError("Boundaries must be strictly increasing"))
    elseif topology == :embedded
        haskey(config, :domains) ||
            throw(ArgumentError("Embedded topology requires :domains key"))
        domains = config[:domains]
        length(domains) >= 2 ||
            throw(ArgumentError("Embedded requires at least 2 domains, got $(length(domains))"))
    end

    cells = config[:cells]
    if cells isa Integer
        cells > 0 || throw(ArgumentError("cells must be positive, got $cells"))
    elseif cells isa Vector
        all(c -> c > 0, cells) || throw(ArgumentError("All cell counts must be positive"))
    else
        throw(ArgumentError("cells must be Int or Vector{Int}, got $(typeof(cells))"))
    end
end

# ── patchOffsetL computation ──────────────────────────────────────────────

function _compute_patch_offsets(geometry::String, boundaries::Vector{Float64},
                                cells_vec::Vector{Int}, mubar::Int)
    N = length(cells_vec)
    offsets = zeros(Int, N)
    if geometry in _CYLINDRICAL_GEOMETRIES
        cumulative = 0
        if boundaries[1] > 0
            DX_first = (boundaries[2] - boundaries[1]) / cells_vec[1]
            cumulative = round(Int, boundaries[1] / DX_first) * mubar
        end
        for k in 1:N
            offsets[k] = cumulative
            cumulative += cells_vec[k] * mubar
        end
    end
    return offsets
end

# ── Chain factory ─────────────────────────────────────────────────────────

function _create_chain(config::Dict{Symbol, Any})
    geometry = config[:geometry]::String
    boundaries = Float64.(config[:boundaries])
    N = length(boundaries) - 1
    cells_input = config[:cells]
    cells_vec = cells_input isa Integer ? fill(cells_input, N) : Int.(cells_input)
    length(cells_vec) == N ||
        throw(ArgumentError("cells vector length $(length(cells_vec)) != $N patches"))

    vars = config[:vars]::Dict
    BCL_outer = config[:BCL]::Dict
    BCR_outer = config[:BCR]::Dict
    mubar = get(config, :mubar, 3)::Int
    l_q = get(config, :l_q, Dict("default" => 2.0))

    # Compute DX per patch and validate ratios
    DX = [(boundaries[k+1] - boundaries[k]) / cells_vec[k] for k in 1:N]
    for k in 1:(N-1)
        ratio = max(DX[k], DX[k+1]) / min(DX[k], DX[k+1])
        tol = 1e-10 * max(DX[k], DX[k+1])
        (abs(ratio - 1.0) < tol || abs(ratio - 2.0) < tol) ||
            throw(ArgumentError(
                "Adjacent DX ratio between patches $k and $(k+1) is $(round(ratio, digits=4)):1, " *
                "must be 1:1 or 2:1. DX[$k]=$(DX[k]), DX[$(k+1)]=$(DX[k+1])"))
    end

    # Determine primary/secondary at each interface → assign BCs
    bc_left  = Vector{Dict}(undef, N)
    bc_right = Vector{Dict}(undef, N)
    for k in 1:N
        bc_left[k]  = Dict(key => NaturalBC() for key in keys(vars))
        bc_right[k] = Dict(key => NaturalBC() for key in keys(vars))
    end
    for k in 1:(N-1)
        if DX[k] >= DX[k+1]
            bc_left[k+1] = Dict(key => FixedBC() for key in keys(vars))
        else
            bc_right[k] = Dict(key => FixedBC() for key in keys(vars))
        end
    end

    # Override outer BCs with user-specified values
    bc_left[1]  = BCL_outer
    bc_right[N] = BCR_outer

    # Compute patchOffsetL for cylindrical/spherical geometries
    offsets = _compute_patch_offsets(geometry, boundaries, cells_vec, mubar)

    # Optional shared params
    kMin = get(config, :kMin, 0.0)
    kMax = get(config, :kMax, 0.0)
    kDim = get(config, :kDim, 0)
    BCB = get(config, :BCB, Dict{String,Any}())
    BCT = get(config, :BCT, Dict{String,Any}())
    quadrature = get(config, :quadrature, :gauss)
    fourier_filter = get(config, :fourier_filter, Dict{String,Any}())
    chebyshev_filter = get(config, :chebyshev_filter, Dict{String,Any}())
    spline_filter = get(config, :spline_filter, Dict{String,Any}())
    max_wavenumber = get(config, :max_wavenumber, Dict("default" => -1))

    grids = SpringsteelGrid[]
    for k in 1:N
        gp_kwargs = Dict{Symbol, Any}(
            :geometry       => geometry,
            :iMin           => boundaries[k],
            :iMax           => boundaries[k+1],
            :num_cells      => cells_vec[k],
            :mubar          => mubar,
            :quadrature     => quadrature,
            :l_q            => l_q,
            :BCL            => bc_left[k],
            :BCR            => bc_right[k],
            :vars           => vars,
            :max_wavenumber => max_wavenumber,
            :fourier_filter => fourier_filter,
            :chebyshev_filter => chebyshev_filter,
            :spline_filter  => spline_filter,
        )
        if offsets[k] > 0
            gp_kwargs[:patchOffsetL] = offsets[k]
        end
        if kDim > 0
            gp_kwargs[:kMin] = kMin
            gp_kwargs[:kMax] = kMax
            gp_kwargs[:kDim] = kDim
            if !isempty(BCB); gp_kwargs[:BCB] = BCB; end
            if !isempty(BCT); gp_kwargs[:BCT] = BCT; end
        end
        gp = SpringsteelGridParameters(; gp_kwargs...)
        push!(grids, createGrid(gp))
    end

    return PatchChain(grids)
end

# ── Embedded factory ──────────────────────────────────────────────────────

function _create_embedded(config::Dict{Symbol, Any})
    geometry = config[:geometry]::String
    domains = config[:domains]
    N = length(domains)
    cells_input = config[:cells]
    cells_vec = cells_input isa Integer ? fill(cells_input, N) : Int.(cells_input)
    length(cells_vec) == N ||
        throw(ArgumentError("cells vector length $(length(cells_vec)) != $N domains"))

    vars = config[:vars]::Dict
    BCL_outer = config[:BCL]::Dict
    BCR_outer = config[:BCR]::Dict
    mubar = get(config, :mubar, 3)::Int
    l_q = get(config, :l_q, Dict("default" => 2.0))

    # Validate containment
    for k in 2:N
        p_min, p_max = Float64(domains[k-1][1]), Float64(domains[k-1][2])
        c_min, c_max = Float64(domains[k][1]), Float64(domains[k][2])
        (c_min >= p_min && c_max <= p_max) ||
            throw(ArgumentError(
                "Domain $k ($c_min, $c_max) not contained in domain $(k-1) ($p_min, $p_max)"))
    end

    # Validate strict refinement
    DX = [(Float64(domains[k][2]) - Float64(domains[k][1])) / cells_vec[k] for k in 1:N]
    for k in 2:N
        ratio = DX[k-1] / DX[k]
        tol = 1e-10 * max(DX[k-1], DX[k])
        abs(ratio - 1.0) < tol &&
            throw(ArgumentError(
                "Embedded requires strict refinement. Domains $(k-1) and $k have same DX=$(DX[k])"))
    end

    # Assign BCs
    bc_left  = Vector{Dict}(undef, N)
    bc_right = Vector{Dict}(undef, N)
    bc_left[1]  = BCL_outer
    bc_right[1] = BCR_outer
    for k in 2:N
        bc_left[k]  = Dict(key => FixedBC() for key in keys(vars))
        bc_right[k] = Dict(key => FixedBC() for key in keys(vars))
    end

    # patchOffsetL — use outermost domain start as reference
    boundaries_flat = Float64[Float64(domains[k][1]) for k in 1:N]
    push!(boundaries_flat, Float64(domains[N][2]))
    offsets = _compute_patch_offsets(geometry, boundaries_flat, cells_vec, mubar)

    # Optional shared params
    kMin = get(config, :kMin, 0.0)
    kMax = get(config, :kMax, 0.0)
    kDim = get(config, :kDim, 0)
    BCB = get(config, :BCB, Dict{String,Any}())
    BCT = get(config, :BCT, Dict{String,Any}())
    quadrature = get(config, :quadrature, :gauss)
    fourier_filter = get(config, :fourier_filter, Dict{String,Any}())
    chebyshev_filter = get(config, :chebyshev_filter, Dict{String,Any}())
    spline_filter = get(config, :spline_filter, Dict{String,Any}())
    max_wavenumber = get(config, :max_wavenumber, Dict("default" => -1))

    grids = SpringsteelGrid[]
    for k in 1:N
        gp_kwargs = Dict{Symbol, Any}(
            :geometry       => geometry,
            :iMin           => Float64(domains[k][1]),
            :iMax           => Float64(domains[k][2]),
            :num_cells      => cells_vec[k],
            :mubar          => mubar,
            :quadrature     => quadrature,
            :l_q            => l_q,
            :BCL            => bc_left[k],
            :BCR            => bc_right[k],
            :vars           => vars,
            :max_wavenumber => max_wavenumber,
            :fourier_filter => fourier_filter,
            :chebyshev_filter => chebyshev_filter,
            :spline_filter  => spline_filter,
        )
        if offsets[k] > 0
            gp_kwargs[:patchOffsetL] = offsets[k]
        end
        if kDim > 0
            gp_kwargs[:kMin] = kMin
            gp_kwargs[:kMax] = kMax
            gp_kwargs[:kDim] = kDim
            if !isempty(BCB); gp_kwargs[:BCB] = BCB; end
            if !isempty(BCT); gp_kwargs[:BCT] = BCT; end
        end
        gp = SpringsteelGridParameters(; gp_kwargs...)
        push!(grids, createGrid(gp))
    end

    return PatchEmbedded(grids)
end

# ── Public factory ────────────────────────────────────────────────────────

"""
    createMultiGrid(config::Dict{Symbol, Any}) -> SpringsteelMultiGrid

Construct a multi-patch grid from a configuration dictionary.

Auto-computes interface BCs (NaturalBC for primary sides, FixedBC for secondary),
patchOffsetL for cylindrical/spherical geometries, and validates DX ratios.

# Required keys
- `:topology` — `:chain` or `:embedded`
- `:geometry` — `"R"`, `"RL"`, `"RLZ"`, `"SL"`, `"SLZ"`, `"RR"`, `"RRR"`
- `:cells` — `Int` (all equal) or `Vector{Int}` (per-patch)
- `:vars` — variable map
- `:BCL`, `:BCR` — outer BCs (per-variable Dict, required)

# Chain: `:boundaries => [x₁, x₂, ..., xₙ₊₁]` (N+1 values → N patches)
# Embedded: `:domains => [(min₁,max₁), ..., (minₙ,maxₙ)]` (outermost first)

# Example
```julia
mg = createMultiGrid(Dict(
    :topology   => :chain,
    :geometry   => "RL",
    :boundaries => [0.0, 50.0, 75.0],
    :cells      => 10,
    :vars       => Dict("u" => 1),
    :BCL        => Dict("u" => NaturalBC()),
    :BCR        => Dict("u" => NaturalBC())))
spectralTransform!(mg)
multiGridTransform!(mg)
```

See also: [`SpringsteelMultiGrid`](@ref), [`PatchChain`](@ref), [`PatchEmbedded`](@ref)
"""
function createMultiGrid(config::Dict{Symbol, Any};
                         embedded_in::Union{Nothing, SpringsteelGrid} = nothing)
    _validate_multigrid_config(config)
    topology = config[:topology]
    if topology == :chain
        mpg = _create_chain(config)
    elseif topology == :embedded
        mpg = _create_embedded(config)
    else
        throw(ArgumentError("Unknown topology: $topology"))
    end
    cfg = copy(config)
    if embedded_in !== nothing
        cfg[:embedded_in] = embedded_in
    end
    return SpringsteelMultiGrid(cfg, mpg)
end

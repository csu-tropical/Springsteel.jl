# ────────────────────────────────────────────────────────────────────────────
# Per-grid scratch buffer registry for transform inner loops
# ────────────────────────────────────────────────────────────────────────────
#
# `gridTransform!` and `spectralTransform!` need a handful of small loop-
# invariant scratch buffers (splineBuffer, ringBuffer, spline_scratch, tempcb,
# etc.) sized only by grid params. Allocating these at function entry costs
# 5–7 small allocations per call across the 3D geometries.
#
# This module caches them in a per-grid registry keyed by `objectid(grid)`,
# mirroring the existing `_WN_AHAT_REGISTRY` pattern. Each call to
# `gridTransform!` does a Dict lookup, returns a pre-built typed scratch
# struct, and uses its fields directly.
#
# Lifetime: cache entries persist for the lifetime of the grid and are not
# automatically freed when the grid is GC'd. Long-running programs that create
# many transient grids should call `_clear_transform_scratch!(grid)` manually.

const _TRANSFORM_SCRATCH = Dict{UInt, Any}()

# ── Per-geometry typed scratch structs ─────────────────────────────────────

# 2D Cartesian Spline×Spline (RR)
struct _ScratchRR
    splineBuffer::Matrix{Float64}     # [iDim, b_jDim]
    spline_scratch::Vector{Float64}   # [iDim] — for SI*xtransform writes
    tempsb::Matrix{Float64}           # [b_jDim, iDim] — spectralTransform stage 1
end

# 2D Cartesian Spline×Chebyshev (RZ)
struct _ScratchRZ
    splineBuffer::Matrix{Float64}     # [iDim, b_kDim]
    spline_scratch::Vector{Float64}   # [iDim]
    tempcb::Matrix{Float64}           # [b_kDim, iDim] — spectralTransform stage 1
end

# 3D Cartesian Spline×Spline×Spline (RRR)
struct _ScratchRRR
    splineBuffer_r::Array{Float64,3}      # [iDim, b_jDim, b_kDim]
    splineBuffer_l::Matrix{Float64}       # [jDim, b_kDim]
    splineBuffer_l_1st::Matrix{Float64}   # [jDim, b_kDim]
    splineBuffer_l_2nd::Matrix{Float64}   # [jDim, b_kDim]
    scratch_i::Vector{Float64}            # [iDim]
    scratch_j::Vector{Float64}            # [jDim]
    tempsb_z::Array{Float64,3}            # [b_kDim, iDim, jDim] — spectralTransform stage 1
    tempsb_l::Array{Float64,3}            # [b_jDim, b_kDim, iDim] — spectralTransform stage 2
end

# 2D Cylindrical Spline×Fourier (RL)
struct _ScratchRL
    spline_r::Matrix{Float64}         # [iDim, kDim*2 + 1]
    spline_rr::Matrix{Float64}        # [iDim, kDim*2 + 1]
    spline_scratch::Vector{Float64}   # [iDim]
end

# 3D Cylindrical Spline×Fourier×Chebyshev (RLZ)
struct _ScratchRLZ
    splineBuffer::Matrix{Float64}     # [iDim, 3]
    ringBuffer::Matrix{Float64}       # [max_lpoints, b_kDim]
    spline_scratch::Vector{Float64}   # [iDim]
    tempcb::Matrix{Float64}           # [b_kDim, max_lpoints] — spectralTransform stage 1
end

# 2D Spherical Spline×Fourier (SL) — same shape as RL
struct _ScratchSL
    spline_r::Matrix{Float64}
    spline_rr::Matrix{Float64}
    spline_scratch::Vector{Float64}
end

# 3D Spherical Spline×Fourier×Chebyshev (SLZ) — same shape as RLZ
struct _ScratchSLZ
    splineBuffer::Matrix{Float64}
    ringBuffer::Matrix{Float64}
    spline_scratch::Vector{Float64}
    tempcb::Matrix{Float64}
end

# ── Builders ───────────────────────────────────────────────────────────────

function _build_scratch(grid::_2DCartesianRR)
    p = grid.params
    return _ScratchRR(
        zeros(Float64, p.iDim, p.b_jDim),
        zeros(Float64, p.iDim),
        zeros(Float64, p.b_jDim, p.iDim),
    )
end

function _build_scratch(grid::_2DCartesianRZ)
    p = grid.params
    return _ScratchRZ(
        zeros(Float64, p.iDim, p.b_kDim),
        zeros(Float64, p.iDim),
        zeros(Float64, p.b_kDim, p.iDim),
    )
end

function _build_scratch(grid::_3DCartesianRRR)
    p = grid.params
    return _ScratchRRR(
        zeros(Float64, p.iDim, p.b_jDim, p.b_kDim),
        zeros(Float64, p.jDim, p.b_kDim),
        zeros(Float64, p.jDim, p.b_kDim),
        zeros(Float64, p.jDim, p.b_kDim),
        zeros(Float64, p.iDim),
        zeros(Float64, p.jDim),
        zeros(Float64, p.b_kDim, p.iDim, p.jDim),
        zeros(Float64, p.b_jDim, p.b_kDim, p.iDim),
    )
end

function _build_scratch(grid::_RLGrid)
    p = grid.params
    kDim = p.iDim + p.patchOffsetL
    return _ScratchRL(
        zeros(Float64, p.iDim, kDim * 2 + 1),
        zeros(Float64, p.iDim, kDim * 2 + 1),
        zeros(Float64, p.iDim),
    )
end

function _build_scratch(grid::_RLZGrid)
    p = grid.params
    kDim_wn = p.iDim + p.patchOffsetL
    max_lpoints = 4 + 4 * kDim_wn
    return _ScratchRLZ(
        zeros(Float64, p.iDim, 3),
        zeros(Float64, max_lpoints, p.b_kDim),
        zeros(Float64, p.iDim),
        zeros(Float64, p.b_kDim, max_lpoints),
    )
end

function _build_scratch(grid::_SLGrid)
    p = grid.params
    kDim = p.iDim + p.patchOffsetL
    return _ScratchSL(
        zeros(Float64, p.iDim, kDim * 2 + 1),
        zeros(Float64, p.iDim, kDim * 2 + 1),
        zeros(Float64, p.iDim),
    )
end

function _build_scratch(grid::_SLZGrid)
    p = grid.params
    # SLZ ring sizes vary with sin(θ); use the maximum across radii.
    max_lpoints = 0
    for r in 1:p.iDim
        max_lpoints = max(max_lpoints, grid.jbasis.data[r, 1].params.yDim)
    end
    return _ScratchSLZ(
        zeros(Float64, p.iDim, 3),
        zeros(Float64, max_lpoints, p.b_kDim),
        zeros(Float64, p.iDim),
        zeros(Float64, p.b_kDim, max_lpoints),
    )
end

# ── Typed accessors (one per geometry) ─────────────────────────────────────
#
# Each accessor returns the scratch struct with a concrete return type so that
# downstream field access in `gridTransform!` is fully type-stable. The cached
# value is stored as `Any`, so the `::T` assertion is what the optimiser uses
# to recover concrete typing past the Dict lookup.

for (gridtype, scratchtype) in (
        (:(_2DCartesianRR),  :_ScratchRR),
        (:(_2DCartesianRZ),  :_ScratchRZ),
        (:(_3DCartesianRRR), :_ScratchRRR),
        (:(_RLGrid),         :_ScratchRL),
        (:(_RLZGrid),        :_ScratchRLZ),
        (:(_SLGrid),         :_ScratchSL),
        (:(_SLZGrid),        :_ScratchSLZ),
    )
    @eval @inline function _scratch(grid::$gridtype)::$scratchtype
        id = objectid(grid)
        s = get(_TRANSFORM_SCRATCH, id, nothing)
        if s === nothing
            s = _build_scratch(grid)
            _TRANSFORM_SCRATCH[id] = s
        end
        return s::$scratchtype
    end
end

# Manual cleanup hook — call when a grid is no longer needed.
function _clear_transform_scratch!(grid::SpringsteelGrid)
    delete!(_TRANSFORM_SCRATCH, objectid(grid))
    return nothing
end

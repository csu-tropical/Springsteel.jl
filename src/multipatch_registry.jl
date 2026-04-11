# ────────────────────────────────────────────────────────────────────────────
# Per-wavenumber ahat registry for multi-patch coupling
# ────────────────────────────────────────────────────────────────────────────
#
# RL, RLZ, SL, SLZ grids reuse 3 spline objects (k=0, k-real, k-imag) for ALL
# wavenumbers during gridTransform.  The standard spline.ahat is shared across
# wavenumbers, but multi-patch coupling requires per-wavenumber ahat values.
#
# This registry stores per-wavenumber ahat buffers keyed by grid identity.
# update_interface! (in multipatch.jl) populates it; gridTransform (in
# transforms_cylindrical.jl etc.) reads from it; multiGridTransform! clears it.
#
# Layout: _WN_AHAT_REGISTRY[objectid(grid)] = Dict(v => Matrix(ahat_len, n_slots))
# where n_slots = 1 + 2*kDim, column 1 = k=0, columns 2k..2k+1 = k real/imag.

const _WN_AHAT_REGISTRY = Dict{UInt, Dict{Int, Matrix{Float64}}}()

"""
    _has_wavenumber_ahat(grid) -> Bool

Check if per-wavenumber ahat data exists for this grid in the registry.
"""
_has_wavenumber_ahat(grid::SpringsteelGrid) = haskey(_WN_AHAT_REGISTRY, objectid(grid))

"""
    _get_wavenumber_ahat(grid, v, slot) -> Vector{Float64}

Retrieve per-wavenumber ahat for variable `v` at spectral slot `slot`.
Slot 0 = k=0; for RL: slot 2k = wavenumber k real, slot 2k+1 = wavenumber k imag.
"""
function _get_wavenumber_ahat(grid::SpringsteelGrid, v::Int, slot::Int)
    return view(_WN_AHAT_REGISTRY[objectid(grid)][v], :, slot + 1)
end

"""
    _set_wavenumber_ahat!(grid, v, slot, ahat_vals, n_slots)

Store per-wavenumber ahat for variable `v` at spectral slot `slot`.
Creates the buffer with `n_slots` columns if it doesn't exist yet.

For RL: `n_slots = 2 + 2*kDim` (max slot = 2*kDim+1).
For RLZ: `n_slots = b_kDim * (1 + 2*kDim)` (Chebyshev levels × wavenumber slots).
"""
function _set_wavenumber_ahat!(grid::SpringsteelGrid, v::Int, slot::Int,
                               ahat_vals::AbstractVector{Float64}, n_slots::Int)
    id = objectid(grid)
    buf = get(_WN_AHAT_REGISTRY, id, nothing)
    if buf === nothing
        buf = Dict{Int, Matrix{Float64}}()
        _WN_AHAT_REGISTRY[id] = buf
    end
    M = get(buf, v, nothing)
    if M === nothing
        ahat_len = length(ahat_vals)
        M = zeros(Float64, ahat_len, n_slots)
        buf[v] = M
    end
    @inbounds for i in eachindex(ahat_vals)
        M[i, slot + 1] = ahat_vals[i]
    end
end

"""
    _clear_wavenumber_ahat!(grid)

Remove per-wavenumber ahat data for this grid from the registry.
"""
_clear_wavenumber_ahat!(grid::SpringsteelGrid) = delete!(_WN_AHAT_REGISTRY, objectid(grid))

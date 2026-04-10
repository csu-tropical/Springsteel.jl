# ────────────────────────────────────────────────────────────────────────────
# Spectral filtering framework
# ────────────────────────────────────────────────────────────────────────────
#
# Provides post-transform spectral coefficient filtering for Fourier and
# Chebyshev bases. Filters are specified per-variable in SpringsteelGridParameters
# via `fourier_filter` and `chebyshev_filter` Dicts, and applied automatically
# at the end of spectralTransform!.
#
# Filter types:
#   AbstractFilter       — abstract supertype
#   SpectralFilter       — boxcar/windowed wavenumber-domain filter
#   GaussianFilter       — Gaussian envelope filter
#
# Window functions:
#   :boxcar              — sharp cutoff (1 or 0)
#   :hann                — raised cosine taper
#   :lanczos             — sinc taper
#   :exponential         — Gaussian-like taper
#
# Must be included AFTER transforms_*.jl (uses grid type aliases).

# ════════════════════════════════════════════════════════════════════════════
# Filter types
# ════════════════════════════════════════════════════════════════════════════

"""
    AbstractFilter

Abstract supertype for all spectral filter types. Subtypes define how
spectral coefficients are weighted (kept, zeroed, or tapered) after the
forward transform.

See also: [`SpectralFilter`](@ref), [`GaussianFilter`](@ref), [`applyFilter!`](@ref)
"""
abstract type AbstractFilter end

"""
    SpectralFilter <: AbstractFilter

Wavenumber-domain filter with optional windowed transitions. Applies to
Fourier wavenumbers or Chebyshev mode numbers.

# Fields
- `low_pass::Int = -1`: Zero modes `k > low_pass`. Set to `-1` to disable.
- `high_pass::Int = 0`: Zero modes `k < high_pass`. Set to `0` to disable.
- `notch::Vector{Int} = Int[]`: Specific modes to zero (hard cutoff, ignores window).
- `window::Symbol = :boxcar`: Window function for taper at cutoff edges.
  Options: `:boxcar`, `:hann`, `:lanczos`, `:exponential`.
- `taper_width::Int = 0`: Number of modes over which the taper transitions from
  1 to 0 at each cutoff edge. With `taper_width=0`, all windows reduce to boxcar.

# Examples
```julia
# Remove wave 1 and everything above wave 50
SpectralFilter(low_pass=50, notch=[1])

# Band pass: keep only wavenumbers 5–20 with Lanczos taper over 3 modes
SpectralFilter(high_pass=5, low_pass=20, window=:lanczos, taper_width=3)

# Remove just the mean (wave 0)
SpectralFilter(notch=[0])
```

See also: [`GaussianFilter`](@ref), [`AbstractFilter`](@ref), [`applyFilter!`](@ref)
"""
Base.@kwdef struct SpectralFilter <: AbstractFilter
    low_pass::Int = -1
    high_pass::Int = 0
    notch::Vector{Int} = Int[]
    window::Symbol = :boxcar
    taper_width::Int = 0
end

"""
    GaussianFilter <: AbstractFilter

Gaussian envelope filter that multiplies spectral coefficients by
`exp(-(k/σ)^(2p))` where `k` is the wavenumber/mode number, `σ` is the
width parameter, and `p` is the order.

# Fields
- `sigma::Float64`: Width parameter in wavenumber/mode space. Modes at
  `k = sigma` are attenuated to `exp(-1)` for `order=1`.
- `order::Int = 1`: Filter order. Higher orders give a sharper transition.
  `order=1` is a standard Gaussian. The exponent is `2*order`.

# Examples
```julia
# Standard Gaussian with σ = 20
GaussianFilter(sigma=20.0)

# Sharper Gaussian (super-Gaussian) with order 3
GaussianFilter(sigma=20.0, order=3)
```

See also: [`SpectralFilter`](@ref), [`AbstractFilter`](@ref), [`applyFilter!`](@ref)
"""
Base.@kwdef struct GaussianFilter <: AbstractFilter
    sigma::Float64
    order::Int = 1
end

# ════════════════════════════════════════════════════════════════════════════
# Window functions
# ════════════════════════════════════════════════════════════════════════════

"""
    _window_weight(window::Symbol, t::Float64) -> Float64

Evaluate a window function at normalized position `t ∈ [0, 1]`.
`t = 0` is the passband edge (weight ≈ 1), `t = 1` is the stopband edge (weight ≈ 0).
"""
function _window_weight(window::Symbol, t::Float64)
    t = clamp(t, 0.0, 1.0)
    if window === :boxcar
        return t < 1.0 ? 1.0 : 0.0
    elseif window === :hann
        return 0.5 * (1.0 + cos(π * t))
    elseif window === :lanczos
        # sinc(t) = sin(πt)/(πt), well-defined at t=0
        if t ≈ 0.0
            return 1.0
        else
            return sin(π * t) / (π * t)
        end
    elseif window === :exponential
        return exp(-t^2 * 4.0)  # exp(-4) ≈ 0.018 at t=1
    else
        throw(ArgumentError("Unknown window function: $window. " *
            "Use :boxcar, :hann, :lanczos, or :exponential."))
    end
end

# ════════════════════════════════════════════════════════════════════════════
# Filter weight computation
# ════════════════════════════════════════════════════════════════════════════

"""
    _filter_weight(f::SpectralFilter, k::Int) -> Float64

Compute the multiplicative weight for wavenumber/mode `k` given a SpectralFilter.
Returns a value in [0, 1].
"""
function _filter_weight(f::SpectralFilter, k::Int)
    # Hard notch — always zero, regardless of window
    k in f.notch && return 0.0

    Δ = f.taper_width

    # High-pass: zero modes below high_pass
    if f.high_pass > 0 && k < f.high_pass
        if Δ > 0
            # Taper region: from (high_pass - Δ) to high_pass
            dist = f.high_pass - k  # distance into the stopband
            if dist > Δ
                return 0.0
            else
                t = Float64(dist) / Float64(Δ)
                return _window_weight(f.window, t)
            end
        else
            return 0.0
        end
    end

    # Low-pass: zero modes above low_pass
    if f.low_pass >= 0 && k > f.low_pass
        if Δ > 0
            # Taper region: from low_pass to (low_pass + Δ)
            dist = k - f.low_pass  # distance into the stopband
            if dist > Δ
                return 0.0
            else
                t = Float64(dist) / Float64(Δ)
                return _window_weight(f.window, t)
            end
        else
            return 0.0
        end
    end

    return 1.0
end

"""
    _filter_weight(f::GaussianFilter, k::Int) -> Float64

Compute the multiplicative weight for wavenumber/mode `k` given a GaussianFilter.
Returns `exp(-(k/σ)^(2*order))`.
"""
function _filter_weight(f::GaussianFilter, k::Int)
    return exp(-(Float64(k) / f.sigma)^(2 * f.order))
end

# ════════════════════════════════════════════════════════════════════════════
# Filter lookup helper
# ════════════════════════════════════════════════════════════════════════════

"""
    _get_filter(filter_dict::Dict, var_name::String) -> Union{AbstractFilter, Nothing}

Look up the filter for a variable. Returns `nothing` if no filter is defined.
Falls back to "default" key if the variable name is not found.
"""
function _get_filter(filter_dict::Dict, var_name::String)
    haskey(filter_dict, var_name) && return filter_dict[var_name]
    haskey(filter_dict, "default") && return filter_dict["default"]
    return nothing
end

"""
    _get_var_name(vars::Dict, var_idx::Int) -> String

Get variable name from index. Returns empty string if not found.
"""
function _get_var_name(vars::Dict, var_idx::Int)
    for (name, idx) in vars
        idx == var_idx && return name
    end
    return ""
end

# ════════════════════════════════════════════════════════════════════════════
# applyFilter! — main entry point
# ════════════════════════════════════════════════════════════════════════════

"""
    applyFilter!(grid::SpringsteelGrid) -> Nothing

Apply spectral filters to the grid's spectral coefficient array in-place.

Reads filter specifications from `grid.params.fourier_filter` and
`grid.params.chebyshev_filter` (both `Dict{String, AbstractFilter}` keyed by
variable name, with optional `"default"` fallback key).

For Fourier dimensions, the filter weight is applied to each wavenumber's
real and imaginary coefficient blocks. For Chebyshev dimensions, the filter
weight is applied to each polynomial mode's coefficient block.

This function is called automatically at the end of `spectralTransform!`
when filter Dicts are non-empty. It can also be called manually.

Does nothing for grids with only CubicBSpline dimensions (R, RR, RRR).

# Examples
```julia
gp = SpringsteelGridParameters(
    geometry = "RL", num_cells = 10,
    iMin = 0.0, iMax = 100.0,
    vars = Dict("u" => 1),
    BCL = Dict("u" => CubicBSpline.R0),
    BCR = Dict("u" => CubicBSpline.R0),
    fourier_filter = Dict("u" => SpectralFilter(low_pass=5, notch=[1])))
grid = createGrid(gp)
# ... populate physical, then:
spectralTransform!(grid)  # automatically calls applyFilter!
```

See also: [`SpectralFilter`](@ref), [`GaussianFilter`](@ref), [`spectralTransform!`](@ref)
"""
function applyFilter!(grid::SpringsteelGrid)
    gp = grid.params
    fourier_dict = gp.fourier_filter
    chebyshev_dict = gp.chebyshev_filter

    # Early exit if no filters defined
    isempty(fourier_dict) && isempty(chebyshev_dict) && return nothing

    # Dispatch to geometry-specific implementation
    _applyFilter_impl!(grid, fourier_dict, chebyshev_dict)
    return nothing
end

# ════════════════════════════════════════════════════════════════════════════
# Geometry-specific filter implementations
# ════════════════════════════════════════════════════════════════════════════

# ── No-op for pure spline grids (R, RR, RRR) ──────────────────────────────
function _applyFilter_impl!(grid::SpringsteelGrid{CartesianGeometry, <:SplineBasisArray, T, U},
                            fourier_dict::Dict, chebyshev_dict::Dict) where
                            {T <: Union{NoBasisArray, SplineBasisArray},
                             U <: Union{NoBasisArray, SplineBasisArray}}
    # Nothing to filter — all dimensions are CubicBSpline
    return nothing
end

# ── RZ (Spline × Chebyshev) — Chebyshev filtering only ────────────────────
# Spectral layout: spectral[(z-1)*b_iDim + j, v] for Chebyshev mode z, spline coeff j
function _applyFilter_impl!(grid::SpringsteelGrid{CartesianGeometry, <:SplineBasisArray, NoBasisArray, <:ChebyshevBasisArray},
                            fourier_dict::Dict, chebyshev_dict::Dict)
    isempty(chebyshev_dict) && return nothing

    gp = grid.params
    b_iDim = gp.b_iDim
    b_kDim = gp.b_kDim

    for (var_name, v) in gp.vars
        filt = _get_filter(chebyshev_dict, var_name)
        filt === nothing && continue

        for z in 1:b_kDim
            w = _filter_weight(filt, z - 1)  # mode 0 at z=1
            w ≈ 1.0 && continue
            r1 = (z - 1) * b_iDim + 1
            r2 = z * b_iDim
            if w ≈ 0.0
                grid.spectral[r1:r2, v] .= 0.0
            else
                grid.spectral[r1:r2, v] .*= w
            end
        end
    end
end

# ── RL / SL (Spline × Fourier) — Fourier filtering only ───────────────────
# RL spectral layout: k=0 at [1:b_iDim], k≥1 real at [(2k-1)*b_iDim+1:2k*b_iDim],
#                     k≥1 imag at [2k*b_iDim+1:(2k+1)*b_iDim]
# kDim = iDim + patchOffsetL
function _applyFilter_impl!(grid::SpringsteelGrid{G, <:SplineBasisArray, <:FourierBasisArray, NoBasisArray},
                            fourier_dict::Dict, chebyshev_dict::Dict) where
                            {G <: Union{CylindricalGeometry, SphericalGeometry}}
    isempty(fourier_dict) && return nothing

    gp = grid.params
    b_iDim = gp.b_iDim
    kDim = gp.iDim + gp.patchOffsetL  # max wavenumber

    for (var_name, v) in gp.vars
        filt = _get_filter(fourier_dict, var_name)
        filt === nothing && continue

        # k=0 (wavenumber 0)
        w0 = _filter_weight(filt, 0)
        if !(w0 ≈ 1.0)
            if w0 ≈ 0.0
                grid.spectral[1:b_iDim, v] .= 0.0
            else
                grid.spectral[1:b_iDim, v] .*= w0
            end
        end

        # k≥1: p = k*2 (RL convention)
        for k in 1:kDim
            w = _filter_weight(filt, k)
            w ≈ 1.0 && continue

            # Real part
            r1_real = (2*k - 1) * b_iDim + 1
            r2_real = 2*k * b_iDim
            # Imaginary part
            r1_imag = 2*k * b_iDim + 1
            r2_imag = (2*k + 1) * b_iDim

            if w ≈ 0.0
                grid.spectral[r1_real:r2_real, v] .= 0.0
                grid.spectral[r1_imag:r2_imag, v] .= 0.0
            else
                grid.spectral[r1_real:r2_real, v] .*= w
                grid.spectral[r1_imag:r2_imag, v] .*= w
            end
        end
    end
end

# ── RLZ / SLZ (Spline × Fourier × Chebyshev) — both filters ──────────────
# RLZ spectral layout: per z_b level (z_b = 1..b_kDim):
#   block_start = (z_b-1) * b_iDim * (1 + kDim_wn * 2)
#   k=0:      [block_start + 1 : block_start + b_iDim]
#   k≥1 real: [block_start + b_iDim + (k-1)*2*b_iDim + 1 : +b_iDim]
#   k≥1 imag: [block_start + b_iDim + (k-1)*2*b_iDim + b_iDim + 1 : +b_iDim]
function _applyFilter_impl!(grid::SpringsteelGrid{G, <:SplineBasisArray, <:FourierBasisArray, <:ChebyshevBasisArray},
                            fourier_dict::Dict, chebyshev_dict::Dict) where
                            {G <: Union{CylindricalGeometry, SphericalGeometry}}
    gp = grid.params
    b_iDim = gp.b_iDim
    b_kDim = gp.b_kDim
    kDim_wn = gp.iDim + gp.patchOffsetL  # max Fourier wavenumber
    block_size = b_iDim * (1 + kDim_wn * 2)

    for (var_name, v) in gp.vars
        f_filt = _get_filter(fourier_dict, var_name)
        c_filt = _get_filter(chebyshev_dict, var_name)
        (f_filt === nothing && c_filt === nothing) && continue

        for z_b in 1:b_kDim
            # Chebyshev weight for this z-level
            c_w = c_filt === nothing ? 1.0 : _filter_weight(c_filt, z_b - 1)

            block_start = (z_b - 1) * block_size

            # k=0
            f_w0 = f_filt === nothing ? 1.0 : _filter_weight(f_filt, 0)
            w0 = c_w * f_w0
            if !(w0 ≈ 1.0)
                r1 = block_start + 1
                r2 = block_start + b_iDim
                if w0 ≈ 0.0
                    grid.spectral[r1:r2, v] .= 0.0
                else
                    grid.spectral[r1:r2, v] .*= w0
                end
            end

            # k≥1: p = (k-1)*2 (RLZ convention)
            for k in 1:kDim_wn
                f_wk = f_filt === nothing ? 1.0 : _filter_weight(f_filt, k)
                w = c_w * f_wk
                w ≈ 1.0 && continue

                # Real part
                r1_real = block_start + b_iDim + (k - 1) * 2 * b_iDim + 1
                r2_real = r1_real + b_iDim - 1
                # Imaginary part
                r1_imag = r2_real + 1
                r2_imag = r1_imag + b_iDim - 1

                if w ≈ 0.0
                    grid.spectral[r1_real:r2_real, v] .= 0.0
                    grid.spectral[r1_imag:r2_imag, v] .= 0.0
                else
                    grid.spectral[r1_real:r2_real, v] .*= w
                    grid.spectral[r1_imag:r2_imag, v] .*= w
                end
            end
        end
    end
end

# ── Fallback for grids without filtering support ──────────────────────────
function _applyFilter_impl!(grid::SpringsteelGrid, fourier_dict::Dict, chebyshev_dict::Dict)
    # No filtering implemented for this grid type (L, LL, LLZ, Z, ZZ, ZZZ)
    # These grid types don't have spectralTransform! implementations yet
    return nothing
end

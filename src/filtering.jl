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
#   GaussianFilter       — Gaussian envelope (spectral) / Gaussian kernel (spline)
#   LanczosFilter        — Lanczos window (spectral) / Lanczos kernel (spline)
#
# Window functions:
#   :boxcar              — sharp cutoff (1 or 0)
#   :hann                — raised cosine taper
#   :lanczos             — sinc taper
#   :exponential         — Gaussian-like taper
#
# Spline-direction filtering operates as a physical-space convolution on the
# mish (quadrature) values BEFORE the SB transform fires.  Boundary handling
# is zero-extend + renormalise; the subsequent SA step re-imposes the BC via
# γ-folding and `ahat`, so user-configured BCs are preserved exactly.
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

Gaussian filter with dual interpretation:

- **Spectral path** (Fourier / Chebyshev): multiplies coefficients by
  `exp(-(k/σ)^(2*order))` where `k` is the wavenumber / mode number.
- **Spline path** (CubicBSpline): physical-space convolution with kernel
  `exp(-x²/(2σ²))` where `σ` is interpreted in cell-widths.

# Fields
- `sigma::Float64`: Width parameter.  On Fourier/Chebyshev: width in mode
  space.  On Spline: kernel width in cell widths (so `sigma=2` means the
  Gaussian's 1σ width equals 2 cell widths).
- `order::Int = 1`: (spectral path only) filter order.  Higher orders give a
  sharper transition.  The exponent is `2*order`.
- `n_sigma::Int = 4`: (spline path only) kernel half-width in units of `σ`;
  the convolution stencil is truncated beyond `n_sigma · σ`.

# Examples
```julia
# Standard Gaussian with σ = 20 (spectral path)
GaussianFilter(sigma=20.0)

# Sharper super-Gaussian with order 3 (spectral path)
GaussianFilter(sigma=20.0, order=3)

# Spline-direction Gaussian with 2-cell width (spline path)
GaussianFilter(sigma=2.0)
```

See also: [`LanczosFilter`](@ref), [`SpectralFilter`](@ref), [`applyFilter!`](@ref)
"""
Base.@kwdef struct GaussianFilter <: AbstractFilter
    sigma::Float64
    order::Int = 1
    n_sigma::Int = 4
end

"""
    LanczosFilter <: AbstractFilter

Lanczos-windowed filter with dual interpretation:

- **Spline path** (CubicBSpline): physical-space convolution with kernel
  `K(x) = sinc(x/h) · sinc(x/(a·h))` for `|x| < a·h`, zero outside, where
  `h = DX` (cell width) and `a` is the lobe count.
- **Spectral path** (Fourier / Chebyshev): convenience wrapper that delegates
  to `SpectralFilter(window=:lanczos, low_pass=low_pass, taper_width=a)`.

# Fields
- `a::Int = 3`: Lobe count for the spline kernel; doubles as taper width on
  the spectral path.
- `low_pass::Int = 0`: Spectral cutoff wavenumber.  Required (`> 0`) for
  Fourier/Chebyshev dispatch; ignored on the spline path (kernel scale is
  set by `a · DX`).

# Examples
```julia
# Spline-direction Lanczos with 3 lobes
LanczosFilter(a=3)

# Spectral Lanczos low-pass at k=10 with 3-mode taper
LanczosFilter(a=3, low_pass=10)
```

See also: [`GaussianFilter`](@ref), [`SpectralFilter`](@ref), [`applyFilter!`](@ref)
"""
Base.@kwdef struct LanczosFilter <: AbstractFilter
    a::Int = 3
    low_pass::Int = 0
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

"""
    _filter_weight(f::LanczosFilter, k::Int) -> Float64

Spectral-path weight for a `LanczosFilter` — equivalent to a
`SpectralFilter(window=:lanczos, low_pass=f.low_pass, taper_width=f.a)`.
"""
function _filter_weight(f::LanczosFilter, k::Int)
    f.low_pass <= 0 && throw(ArgumentError(
        "LanczosFilter on a spectral direction requires `low_pass > 0`. " *
        "Set `low_pass=<cutoff>` or use it on a spline direction."))
    sf = SpectralFilter(low_pass=f.low_pass, window=:lanczos, taper_width=f.a)
    return _filter_weight(sf, k)
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
# Spline-direction filtering: types, kernels, convolution helper
# ════════════════════════════════════════════════════════════════════════════

"""
    _resolve_spline_filter(sf::Dict, var_name::String, dir::Symbol) -> Union{AbstractFilter, Nothing}

Look up the per-(variable, direction) filter entry in a `spline_filter` Dict
of shape `Dict{String, Dict{Symbol, AbstractFilter}}`.  Resolution order:

1. `sf[var_name][dir]`
2. `sf[var_name][:default]`
3. `sf["default"][dir]`
4. `sf["default"][:default]`
5. `nothing`

Returns `nothing` if no rule matches.
"""
function _resolve_spline_filter(sf::Dict, var_name::String, dir::Symbol)
    isempty(sf) && return nothing
    if haskey(sf, var_name)
        inner = sf[var_name]
        if inner isa Dict
            haskey(inner, dir)      && return inner[dir]
            haskey(inner, :default) && return inner[:default]
        end
    end
    if haskey(sf, "default")
        inner = sf["default"]
        if inner isa Dict
            haskey(inner, dir)      && return inner[dir]
            haskey(inner, :default) && return inner[:default]
        end
    end
    return nothing
end

"""
    _gaussian_kernel(sigma_cells::Float64, dx::Float64) -> Function

Return a closure `K(δ) = exp(-(δ/σ)²/2) / (σ·√(2π))` evaluating the
Gaussian kernel at any signed offset `δ` (in physical units), where
`σ = sigma_cells · dx`.
"""
function _gaussian_kernel(sigma_cells::Float64, dx::Float64)
    σ = sigma_cells * dx
    inv2σ2 = 1.0 / (2.0 * σ * σ)
    norm   = 1.0 / (σ * sqrt(2π))
    return δ::Float64 -> norm * exp(-δ * δ * inv2σ2)
end

"""
    _lanczos_kernel(a::Int, dx::Float64) -> Function

Lobe-count-`a` Lanczos kernel `K(δ) = sinc(δ/h) · sinc(δ/(a·h))` for
`|δ| < a·h`, zero outside, where `h = dx`.
"""
function _lanczos_kernel(a::Int, dx::Float64)
    support = a * dx
    sinc_norm = (x::Float64) -> iszero(x) ? 1.0 : sin(π * x) / (π * x)
    return δ::Float64 -> begin
        if abs(δ) >= support
            return 0.0
        end
        x = δ / dx
        return sinc_norm(x) * sinc_norm(x / a)
    end
end

"""
    _spline_kernel(filter::AbstractFilter, dx::Float64) -> Tuple{Function, Float64}

Build the physical-space kernel closure `K` and its support radius
(half-width) for a spline-direction `filter`.  Returns `(K, support)`.

Throws on filter types that have no physical-space realisation
(`SpectralFilter` — boxcar/notch/window cutoffs require an FFT against an
orthogonal global basis).
"""
function _spline_kernel(filter::GaussianFilter, dx::Float64)
    K = _gaussian_kernel(filter.sigma, dx)
    support = filter.n_sigma * filter.sigma * dx
    return K, support
end

function _spline_kernel(filter::LanczosFilter, dx::Float64)
    K = _lanczos_kernel(filter.a, dx)
    support = filter.a * dx
    return K, support
end

function _spline_kernel(filter::SpectralFilter, dx::Float64)
    throw(ArgumentError(
        "SpectralFilter (boxcar / notch / window cutoffs) cannot be applied " *
        "to a spline direction — splines are a local basis. Use " *
        "GaussianFilter or LanczosFilter instead, or choose a Fourier or " *
        "Chebyshev basis if you need a hard spectral cutoff."))
end

"""
    _convolve_axis!(dst::AbstractVector, src::AbstractVector,
                     coords::AbstractVector, K, support::Float64) -> dst

Compute `dst[i] = Σⱼ K(coords[j] - coords[i]) · src[j] / W(i)` with
`W(i) = Σⱼ K(coords[j] - coords[i])` over `j` in the kernel support.
This is zero-extend + renormalise boundary handling: out-of-domain kernel
weight is dropped, and the in-domain weight is normalised so the kernel
preserves a uniform field exactly.

`coords` must be monotonically non-decreasing.  `dst` and `src` may not
alias.
"""
function _convolve_axis!(dst::AbstractVector{<:Real},
                          src::AbstractVector{<:Real},
                          coords::AbstractVector{<:Real},
                          K, support::Float64)
    n = length(src)
    @assert length(dst)    == n
    @assert length(coords) == n
    j_lo = 1
    @inbounds for i in 1:n
        x_i = coords[i]
        # advance lower bound while past-support on the left
        while j_lo <= n && coords[j_lo] < x_i - support
            j_lo += 1
        end
        acc  = 0.0
        wsum = 0.0
        j = j_lo
        while j <= n
            δ = coords[j] - x_i
            δ > support && break
            w = K(δ)
            acc  += w * src[j]
            wsum += w
            j += 1
        end
        dst[i] = wsum > 0.0 ? acc / wsum : src[i]
    end
    return dst
end

"""
    _filter_spline_uMish!(spline, filter::Union{AbstractFilter,Nothing},
                          scratch::AbstractVector) -> Nothing

Apply a physical-space spline filter to `spline.uMish` in-place.  No-op when
`filter === nothing`.  Uses `scratch` (length `length(spline.uMish)`) as a
temporary copy of the input, so the convolution writes back into `uMish`
without aliasing.
"""
function _filter_spline_uMish!(spline, filter::Nothing, scratch::AbstractVector)
    return nothing
end

function _filter_spline_uMish!(spline, filter::AbstractFilter,
                                scratch::AbstractVector)
    K, support = _spline_kernel(filter, spline.params.DX)
    n = length(spline.uMish)
    @assert length(scratch) >= n
    src = view(scratch, 1:n)
    @inbounds copyto!(src, spline.uMish)
    _convolve_axis!(spline.uMish, src, spline.mishPoints, K, support)
    return nothing
end

# ════════════════════════════════════════════════════════════════════════════
# Spline-axis filtering on physical[..., v, 1] strided views (Cartesian only)
# ════════════════════════════════════════════════════════════════════════════
#
# For the Cartesian geometries (R, RR, RRR, RZ) the i / j / k mish points
# are arranged on a regular grid, so we can convolve directly on a strided
# view of the physical array.  For Cylindrical / Spherical geometries the
# i-direction filter is applied inside `spectralTransform!` to the spline's
# uMish (post-Fourier, pre-SB) — see `_filter_spline_uMish!`.

"""
    _spline_dirs(geom::String) -> Tuple{Vararg{Symbol}}

Return the directions whose basis is a CubicBSpline for the given geometry
string.  Used by `_validate_spline_filter` and the dispatcher.

Geometries with no spline directions return `()`; querying the spline
filter on them is always a no-op.
"""
function _spline_dirs(geom::String)
    g = geom in ("Spline1D",)              ? "R"   :
        geom in ("Spline2D",)              ? "RR"  :
        geom in ("Spline3D", "Cylindrical","Samurai", "SphericalShell", "Sphere") ? geom : geom
    if g in ("R", "Spline1D")
        return (:i,)
    elseif g in ("RR", "Spline2D")
        return (:i, :j)
    elseif g in ("RRR", "Spline3D", "Samurai", "SphericalShell", "Sphere")
        return (:i, :j, :k)
    elseif g == "RZ"
        return (:i,)
    elseif g in ("RL", "RLZ", "SL", "SLZ", "Cylindrical", "Polar")
        return (:i,)
    else
        return ()
    end
end

"""
    _validate_spline_filter(sf::Dict, gp::SpringsteelGridParameters) -> Nothing

Validate a `spline_filter` Dict against a grid's variables and geometry.

Checks:
- outer keys are variable names in `gp.vars` or the literal `"default"`
- inner keys are `:i`, `:j`, `:k`, or `:default`
- inner values are `AbstractFilter` subtypes; `SpectralFilter` (boxcar/notch
  cutoffs) is rejected — a clear error tells the user to switch to
  `GaussianFilter` / `LanczosFilter`
- the named direction is actually a spline in the given geometry

Throws `ArgumentError` on the first violation.  No-op when `sf` is empty.
"""
function _validate_spline_filter(sf::Dict, gp::SpringsteelGridParameters)
    isempty(sf) && return nothing
    valid_dirs = (:i, :j, :k, :default)
    spline_dirs = _spline_dirs(gp.geometry)

    for (var_key, inner) in sf
        var_key isa AbstractString || throw(ArgumentError(
            "spline_filter outer keys must be variable-name strings or \"default\"; got $(typeof(var_key))"))
        var_key == "default" || haskey(gp.vars, String(var_key)) || throw(ArgumentError(
            "spline_filter references unknown variable \"$var_key\" (vars: $(collect(keys(gp.vars))))"))
        inner isa Dict || throw(ArgumentError(
            "spline_filter[\"$var_key\"] must be a Dict{Symbol,AbstractFilter}; got $(typeof(inner))"))
        for (dir, filt) in inner
            dir in valid_dirs || throw(ArgumentError(
                "spline_filter[\"$var_key\"] uses invalid direction `$(dir)`; valid: $(valid_dirs)"))
            filt isa AbstractFilter || throw(ArgumentError(
                "spline_filter[\"$var_key\"][$(dir)] must be an AbstractFilter; got $(typeof(filt))"))
            if filt isa SpectralFilter
                throw(ArgumentError(
                    "spline_filter[\"$var_key\"][$(dir)] is a SpectralFilter — boxcar/notch " *
                    "cutoffs are not supported on a spline direction. Use GaussianFilter " *
                    "or LanczosFilter, or apply this filter on a Fourier/Chebyshev direction."))
            end
            if dir !== :default && !(dir in spline_dirs)
                throw(ArgumentError(
                    "spline_filter[\"$var_key\"][$(dir)] specified, but direction $(dir) is " *
                    "not a CubicBSpline direction in geometry \"$(gp.geometry)\" " *
                    "(spline directions: $(spline_dirs))"))
            end
        end
    end
    return nothing
end

"""
    _filter_mish!(grid::SpringsteelGrid) -> Nothing

Apply physical-space spline filters to `grid.physical[..., v, 1]` (the
field-value slot) before any SB transform runs.  Reads filter specs from
`grid.params.spline_filter`.

This is a no-op when:
- `spline_filter` is empty,
- the grid has no Cartesian spline directions (RL/SL/RLZ/SLZ apply their
  i-direction filter inline inside `spectralTransform!`).

See also: [`_filter_spline_uMish!`](@ref) — used inline for radial filtering
on cylindrical / spherical grids.
"""
function _filter_mish!(grid::SpringsteelGrid)
    sf = grid.params.spline_filter
    isempty(sf) && return nothing
    _filter_mish_impl!(grid, sf)
    return nothing
end

# ── Fallback: any geometry without a Cartesian-mish dispatcher (RL/SL/RLZ/SLZ
#    fall here intentionally — their i-filter happens inline). ─────────────
function _filter_mish_impl!(grid::SpringsteelGrid, sf::Dict)
    return nothing
end

# ── R (1D Cartesian Spline) ────────────────────────────────────────────────
function _filter_mish_impl!(
        grid::SpringsteelGrid{CartesianGeometry, <:SplineBasisArray, NoBasisArray, NoBasisArray},
        sf::Dict)
    iDim = grid.params.iDim
    scratch = Vector{Float64}(undef, iDim)
    for (var_name, v) in grid.params.vars
        filt = _resolve_spline_filter(sf, var_name, :i)
        filt === nothing && continue
        spline = grid.ibasis.data[1, v]
        K, support = _spline_kernel(filt, spline.params.DX)
        col = view(grid.physical, 1:iDim, v, 1)
        @inbounds copyto!(scratch, col)
        _convolve_axis!(col, scratch, spline.mishPoints, K, support)
    end
    return nothing
end

# ── RR (2D Cartesian Spline×Spline) ────────────────────────────────────────
# Physical layout: flat = (r-1)*jDim + l   (r outer, l inner)
function _filter_mish_impl!(
        grid::SpringsteelGrid{CartesianGeometry, <:SplineBasisArray, <:SplineBasisArray, NoBasisArray},
        sf::Dict)
    iDim = grid.params.iDim
    jDim = grid.params.jDim

    nmax = max(iDim, jDim)
    scratch = Vector{Float64}(undef, nmax)
    col_buf = Vector{Float64}(undef, nmax)

    for (var_name, v) in grid.params.vars
        filt_i = _resolve_spline_filter(sf, var_name, :i)
        filt_j = _resolve_spline_filter(sf, var_name, :j)
        (filt_i === nothing && filt_j === nothing) && continue

        # ── i-direction: stride jDim, length iDim, for each l ────────────
        if filt_i !== nothing
            isp = grid.ibasis.data[1, v]
            K, support = _spline_kernel(filt_i, isp.params.DX)
            for l in 1:jDim
                @inbounds for r in 1:iDim
                    col_buf[r] = grid.physical[(r-1)*jDim + l, v, 1]
                end
                @inbounds copyto!(view(scratch, 1:iDim), view(col_buf, 1:iDim))
                _convolve_axis!(view(col_buf, 1:iDim), view(scratch, 1:iDim),
                                isp.mishPoints, K, support)
                @inbounds for r in 1:iDim
                    grid.physical[(r-1)*jDim + l, v, 1] = col_buf[r]
                end
            end
        end

        # ── j-direction: stride 1, length jDim, for each r ───────────────
        if filt_j !== nothing
            jsp = grid.jbasis.data[1, v]
            K, support = _spline_kernel(filt_j, jsp.params.DX)
            for r in 1:iDim
                base = (r - 1) * jDim
                row = view(grid.physical, (base+1):(base+jDim), v, 1)
                @inbounds copyto!(view(scratch, 1:jDim), row)
                _convolve_axis!(row, view(scratch, 1:jDim),
                                jsp.mishPoints, K, support)
            end
        end
    end
    return nothing
end

# ── RZ (2D Cartesian Spline×Chebyshev) — i-direction only ─────────────────
# Physical layout: flat = (r-1)*kDim + z
function _filter_mish_impl!(
        grid::SpringsteelGrid{CartesianGeometry, <:SplineBasisArray, NoBasisArray, <:ChebyshevBasisArray},
        sf::Dict)
    iDim = grid.params.iDim
    kDim = grid.params.kDim

    scratch = Vector{Float64}(undef, iDim)
    col_buf = Vector{Float64}(undef, iDim)

    for (var_name, v) in grid.params.vars
        filt_i = _resolve_spline_filter(sf, var_name, :i)
        filt_i === nothing && continue
        isp = grid.ibasis.data[1, v]
        K, support = _spline_kernel(filt_i, isp.params.DX)
        for z in 1:kDim
            @inbounds for r in 1:iDim
                col_buf[r] = grid.physical[(r-1)*kDim + z, v, 1]
            end
            @inbounds copyto!(scratch, col_buf)
            _convolve_axis!(col_buf, scratch, isp.mishPoints, K, support)
            @inbounds for r in 1:iDim
                grid.physical[(r-1)*kDim + z, v, 1] = col_buf[r]
            end
        end
    end
    return nothing
end

# ── RRR (3D Cartesian Spline×Spline×Spline) ────────────────────────────────
# Physical layout: flat = (r-1)*jDim*kDim + (l-1)*kDim + z
function _filter_mish_impl!(
        grid::SpringsteelGrid{CartesianGeometry, <:SplineBasisArray, <:SplineBasisArray, <:SplineBasisArray},
        sf::Dict)
    iDim = grid.params.iDim
    jDim = grid.params.jDim
    kDim = grid.params.kDim

    nmax = max(iDim, jDim, kDim)
    scratch = Vector{Float64}(undef, nmax)
    col_buf = Vector{Float64}(undef, nmax)

    for (var_name, v) in grid.params.vars
        filt_i = _resolve_spline_filter(sf, var_name, :i)
        filt_j = _resolve_spline_filter(sf, var_name, :j)
        filt_k = _resolve_spline_filter(sf, var_name, :k)
        (filt_i === nothing && filt_j === nothing && filt_k === nothing) && continue

        # ── i-direction (stride jDim*kDim, length iDim) ──────────────────
        if filt_i !== nothing
            isp = grid.ibasis.data[1, 1, v]
            K, support = _spline_kernel(filt_i, isp.params.DX)
            for l in 1:jDim, z in 1:kDim
                @inbounds for r in 1:iDim
                    col_buf[r] = grid.physical[(r-1)*jDim*kDim + (l-1)*kDim + z, v, 1]
                end
                @inbounds copyto!(view(scratch, 1:iDim), view(col_buf, 1:iDim))
                _convolve_axis!(view(col_buf, 1:iDim), view(scratch, 1:iDim),
                                isp.mishPoints, K, support)
                @inbounds for r in 1:iDim
                    grid.physical[(r-1)*jDim*kDim + (l-1)*kDim + z, v, 1] = col_buf[r]
                end
            end
        end

        # ── j-direction (stride kDim, length jDim) ───────────────────────
        if filt_j !== nothing
            jsp = grid.jbasis.data[1, 1, v]
            K, support = _spline_kernel(filt_j, jsp.params.DX)
            for r in 1:iDim, z in 1:kDim
                @inbounds for l in 1:jDim
                    col_buf[l] = grid.physical[(r-1)*jDim*kDim + (l-1)*kDim + z, v, 1]
                end
                @inbounds copyto!(view(scratch, 1:jDim), view(col_buf, 1:jDim))
                _convolve_axis!(view(col_buf, 1:jDim), view(scratch, 1:jDim),
                                jsp.mishPoints, K, support)
                @inbounds for l in 1:jDim
                    grid.physical[(r-1)*jDim*kDim + (l-1)*kDim + z, v, 1] = col_buf[l]
                end
            end
        end

        # ── k-direction (stride 1, length kDim) ──────────────────────────
        if filt_k !== nothing
            ksp = grid.kbasis.data[1, 1, v]
            K, support = _spline_kernel(filt_k, ksp.params.DX)
            for r in 1:iDim, l in 1:jDim
                base = (r-1)*jDim*kDim + (l-1)*kDim
                row = view(grid.physical, (base+1):(base+kDim), v, 1)
                @inbounds copyto!(view(scratch, 1:kDim), row)
                _convolve_axis!(row, view(scratch, 1:kDim),
                                ksp.mishPoints, K, support)
            end
        end
    end
    return nothing
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

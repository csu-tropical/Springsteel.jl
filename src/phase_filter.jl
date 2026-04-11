# ────────────────────────────────────────────────────────────────────────────
# PhaseFilter — shared, reusable azimuthal phase-shift + spectral cutoff filter
# ────────────────────────────────────────────────────────────────────────────
#
# Replaces the dense `phasefilter::Matrix{Float64}` field of `Fourier1D` (and
# the non-shifting `filter::Matrix{Float64}` of `Chebyshev1D`). The mathematical
# state per ring is O(kmax): a vector of precomputed cos/sin values for the
# azimuthal rotation plus the output layout parameters. The dense matmul
# previously used in FBtransform!/FAtransform! becomes an O(kmax) rotation
# loop applied directly to the R2HC output.
#
# A single `PhaseFilter` instance can be SHARED across rings with identical
# (yDim, kmax, bDim, phase shift) — notably across all `b_kDim` Chebyshev modes
# for a fixed θ-index in RLZ/SLZ. Chebyshev columns use a null phase shift
# (empty cos_k / sin_k); the apply loop skips the rotation step in that case.
#
# See `agent_files/plan_phasefilter_refactor.md` for design rationale.

"""
    PhaseFilter

Apply-once azimuthal phase shift and spectral cutoff for a single Fourier ring
or Chebyshev column. Stored as O(kmax) precomputed trig values rather than a
dense (yDim × bDim) matrix.

# Fields
- `yDim::Int` — physical azimuthal point count (R2HC scratch length).
- `kmax::Int` — effective max wavenumber kept. The apply loop walks `1..kmax`.
- `bDim::Int` — length of the filtered output vector `b`. Must satisfy
  `bDim ≥ 1 + 2*kmax` (equality for single-patch; strict `>` allowed when a
  caller reserves extra zero-padding slots).
- `cos_k::Vector{Float64}` — `cos(k·ymin)` for `k = 1..kmax`. Empty ⇒ no shift
  (Chebyshev and other non-periodic uses).
- `sin_k::Vector{Float64}` — `sin(k·ymin)` for `k = 1..kmax`. Must match
  `cos_k` length.

# Layout convention
The `b` vector mirrors FFTW's halfcomplex (R2HC) ordering, but with `bDim`:
- `b[1]` — DC term (`k = 0`)
- `b[k+1]` — real part of wavenumber `k`, for `k = 1..kmax`
- `b[bDim - k + 1]` — imaginary part of wavenumber `k`, for `k = 1..kmax`
- all other slots — zero (cubic-grid anti-aliasing / `max_wavenumber` cutoff)

Forward apply computes `b[k]` from the R2HC fft output via a rotation by
`-k·ymin`; inverse apply reverses it (rotation by `+k·ymin`).
"""
struct PhaseFilter
    yDim::Int
    kmax::Int
    bDim::Int
    cos_k::Vector{Float64}
    sin_k::Vector{Float64}
end

"""
    PhaseFilter(fp::FourierParameters) -> PhaseFilter

Construct a `PhaseFilter` matching the dense phasefilter/invphasefilter built
by [`calcPhaseFilter`](@ref) and [`calcInvPhaseFilter`](@ref) for the given
`FourierParameters`. Bit-identical to `b = phasefilter' * fft_scratch` within
Float64 precision.
"""
function PhaseFilter(fp::FourierParameters)
    cos_k = Vector{Float64}(undef, fp.kmax)
    sin_k = Vector{Float64}(undef, fp.kmax)
    @inbounds for k in 1:fp.kmax
        cos_k[k] = cos(k * fp.ymin)
        sin_k[k] = sin(k * fp.ymin)
    end
    return PhaseFilter(fp.yDim, fp.kmax, fp.bDim, cos_k, sin_k)
end

"""
    PhaseFilter(; yDim, kmax, bDim, ymin=0.0, shift=true) -> PhaseFilter

Low-level constructor. Set `shift=false` (or `ymin=0.0`) for a no-op phase
shift; the apply loop then only enforces the spectral cutoff.
"""
function PhaseFilter(; yDim::Int, kmax::Int, bDim::Int,
                     ymin::Float64 = 0.0, shift::Bool = true)
    if shift && ymin != 0.0
        cos_k = Vector{Float64}(undef, kmax)
        sin_k = Vector{Float64}(undef, kmax)
        @inbounds for k in 1:kmax
            cos_k[k] = cos(k * ymin)
            sin_k[k] = sin(k * ymin)
        end
    else
        cos_k = Float64[]
        sin_k = Float64[]
    end
    return PhaseFilter(yDim, kmax, bDim, cos_k, sin_k)
end

"""
    apply_phasefilter_forward!(b, fft_scratch, pf) -> b

In-place forward apply: read `fft_scratch` (length `pf.yDim`, FFTW R2HC layout),
write `b` (length `pf.bDim`). Combines the azimuthal phase-shift rotation
(if populated) and the spectral cutoff.

Mathematically equivalent to `mul!(b, old_phasefilter', fft_scratch)` but with
O(kmax) work instead of O(yDim · bDim).
"""
function apply_phasefilter_forward!(b::AbstractVector{Float64},
                                    fft_scratch::AbstractVector{Float64},
                                    pf::PhaseFilter)
    yDim = pf.yDim
    kmax = pf.kmax
    bDim = pf.bDim

    # DC term — untouched by phase shift.
    @inbounds b[1] = fft_scratch[1]

    if isempty(pf.cos_k)
        # No phase shift (Chebyshev / ymin=0 Fourier): pure R2HC passthrough
        # into the `b` layout with cutoff beyond kmax.
        @inbounds for k in 1:kmax
            b[k + 1]        = fft_scratch[k + 1]
            b[bDim - k + 1] = fft_scratch[yDim - k + 1]
        end
    else
        @inbounds for k in 1:kmax
            re = fft_scratch[k + 1]
            im = fft_scratch[yDim - k + 1]
            c  = pf.cos_k[k]
            s  = pf.sin_k[k]
            # Rotation by -k·ymin:  b_re = cos·re + sin·im ; b_im = -sin·re + cos·im
            b[k + 1]        =  c * re + s * im
            b[bDim - k + 1] = -s * re + c * im
        end
    end

    # Zero-fill any interior slots beyond the cutoff (only needed when
    # bDim > 1 + 2*kmax, e.g. if a caller reserves extra padding).
    if bDim > 1 + 2 * kmax
        @inbounds for i in (kmax + 2):(bDim - kmax)
            b[i] = 0.0
        end
    end

    return b
end

"""
    apply_phasefilter_inverse!(fft_scratch, b, pf) -> fft_scratch

In-place inverse apply: read `b` (length `pf.bDim`), write `fft_scratch`
(length `pf.yDim`, FFTW R2HC layout ready for the inverse FFT). Zero-fills
the tail beyond `kmax`.

Mathematically equivalent to `mul!(a, old_invphasefilter', b)`.
"""
function apply_phasefilter_inverse!(fft_scratch::AbstractVector{Float64},
                                    b::AbstractVector{Float64},
                                    pf::PhaseFilter)
    yDim = pf.yDim
    kmax = pf.kmax
    bDim = pf.bDim

    # DC term.
    @inbounds fft_scratch[1] = b[1]

    if isempty(pf.cos_k)
        @inbounds for k in 1:kmax
            fft_scratch[k + 1]        = b[k + 1]
            fft_scratch[yDim - k + 1] = b[bDim - k + 1]
        end
    else
        @inbounds for k in 1:kmax
            br = b[k + 1]
            bi = b[bDim - k + 1]
            c  = pf.cos_k[k]
            s  = pf.sin_k[k]
            # Rotation by +k·ymin (inverse of forward):
            #   a_re = cos·br - sin·bi
            #   a_im = sin·br + cos·bi
            fft_scratch[k + 1]        = c * br - s * bi
            fft_scratch[yDim - k + 1] = s * br + c * bi
        end
    end

    # Zero-fill middle slots beyond kmax (the wavenumbers the ring cannot
    # carry). The IFFT consumes the full yDim-length layout.
    @inbounds for i in (kmax + 2):(yDim - kmax)
        fft_scratch[i] = 0.0
    end

    return fft_scratch
end

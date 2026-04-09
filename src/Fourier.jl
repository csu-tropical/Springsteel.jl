module Fourier

using LinearAlgebra
using FFTW

export FourierParameters, Fourier1D
export FBtransform, FBtransform!, FAtransform, FAtransform!, FItransform!
export FBxtransform, FIxtransform, FIxxtransform
# Generic (no-prefix) wrappers for abstract 1D basis dispatch
export Btransform, Btransform!, Atransform, Atransform!, Itransform, Itransform!
export Ixtransform, Ixxtransform, IInttransform

#Define some convenient aliases
const real = Float64
const int = Int64
const uint = UInt64

# Only one boundary condition is meaningful for Fourier: periodic.
"""
Periodic boundary condition (Ooyama 2002, section 3e): the **only** valid BC
for the Fourier basis. Couples the left and right ends of the azimuthal domain
to simulate a cyclically continuous ring. All Fourier rings use this BC
automatically; it need not be specified explicitly in most APIs.
"""
const PERIODIC = Dict("PERIODIC" => 0)

# Define the ring parameters
"""
    FourierParameters

Immutable parameter struct (using `@kwdef`) for a 1D Fourier ring basis.

# Fields
- `ymin::Float64`: Azimuthal offset (radians) of the first grid point. Values other than
  `0.0` activate the phase-shift filter so that coefficients are referenced to a common
  zero-phase origin.
- `kmax::Int64`: Maximum retained wavenumber. All Fourier modes with wavenumber `k > kmax`
  are zeroed by the phase-filter matrix.
- `yDim::Int64`: Total number of physical grid points around the ring. Must be even for the
  real-to-halfcomplex FFT (`FFTW.R2HC`).
- `bDim::Int64`: Number of filtered Fourier B-coefficients. Formula: `bDim = 2*kmax + 1`
  (one constant term plus one cosine and one sine coefficient per wavenumber).

# Notes
- Unlike `CubicBSpline.SplineParameters`, there are no auto-computed fields;
  all four fields must be set explicitly at construction time.

# Example
```julia
fp = Fourier.FourierParameters(
    ymin = 0.0,
    kmax = 10,
    yDim = 64,
    bDim = 21   # 2*kmax + 1
)
```

See also: [`Fourier1D`](@ref)
"""
Base.@kwdef struct FourierParameters
    ymin::real = 0.0 # Offset for the position of the first grid point
    kmax::int = 0    # Maximum wavenumber allowed in the Fourier representation
    yDim::int = 0    # Number of points in the physical ring
    bDim::int = 0    # Number of Fourier coefficients after filtering to kmax
end

"""
    Fourier1D

One-dimensional Fourier ring object. Construct via `Fourier1D(fp::FourierParameters)`.

# Fields
- `params::FourierParameters`: Ring configuration (domain offset, `kmax`, grid size)
- `mishPoints::Vector{Float64}`: Evenly-spaced angles in `[ymin, ymin + 2π)` (half-open)
- `fftPlan::FFTW.r2rFFTWPlan`: Pre-measured real-to-halfcomplex forward FFT plan.
  **Do not serialise**; FFTW plans are not portable across processes.
- `ifftPlan::FFTW.r2rFFTWPlan`: Pre-measured halfcomplex-to-real inverse FFT plan.
- `phasefilter::Matrix{Float64}`: Size `(yDim, bDim)`. Combines a phase-shift rotation
  (aligns each wavenumber to `ymin = 0` reference) with anti-aliasing truncation above `kmax`.
- `invphasefilter::Matrix{Float64}`: Size `(bDim, yDim)`. Inverse phase shift plus
  zero-padding: maps the `bDim` filtered coefficients back to a `yDim`-length array for
  the inverse FFT.
- `uMish::Vector{Float64}`: Physical field values at the `yDim` ring points (mutable buffer).
- `b::Vector{Float64}`: Filtered B-coefficients of length `bDim` (result of [`FBtransform`](@ref)).
- `a::Vector{Float64}`: Zero-padded A-coefficients of length `yDim`, ready for inverse FFT
  (result of [`FAtransform`](@ref)).
- `ax::Vector{Float64}`: Working buffer of length `yDim` for derivative and integral
  coefficient computations.

# Notes
- Constructing `Fourier1D` calls `FFTW.plan_r2r` with `FFTW.PATIENT`, which performs
  benchmark measurements to find the fastest algorithm. First construction is slow;
  subsequent transforms using the same plan are fast.
- The struct is **not thread-safe** if `uMish`, `b`, `a`, or `ax` are mutated concurrently.
  Create one `Fourier1D` per thread for parallel workloads.

See also: [`FourierParameters`](@ref), [`FBtransform`](@ref), [`FAtransform`](@ref), [`FItransform`](@ref)
"""
struct Fourier1D
    # Parameters for the ring
    params::FourierParameters

    # Pre-calculated angular points
    mishPoints::Vector{real}

    # Measured FFTW Plan
    fftPlan::FFTW.r2rFFTWPlan
    ifftPlan::FFTW.r2rFFTWPlan

    # Phase-shift and filter matrix
    phasefilter::Matrix{real}
    invphasefilter::Matrix{real}

    # In this context, uMish is the physical values
    # b is the filtered Fourier coefficients
    # a is the Fourier coefficients with zeros padding up to physical size
    # ax is a buffer for derivative and integral coefficients
    uMish::Vector{real}
    b::Vector{real}
    a::Vector{real}
    ax::Vector{real}
    # Scratch buffer for in-place FBtransform!/FAtransform!. Length yDim,
    # holds the raw FFT output before applying the phase filter.
    _scratch_fft::Vector{real}
end

"""
    Fourier1D(fp::FourierParameters) -> Fourier1D

Construct a [`Fourier1D`](@ref) object from ring parameters.

Pre-computes and caches all state needed for repeated spectral transforms:
1. Evenly-spaced mish points via [`calcMishPoints`](@ref)
2. Phase-filter and inverse phase-filter matrices via [`calcPhaseFilter`](@ref) /
   [`calcInvPhaseFilter`](@ref)
3. Forward and inverse FFTW plans (measured with `FFTW.PATIENT` for maximum performance)
4. Zero-initialised working buffers `uMish`, `b`, `a`, `ax`

Reuse the constructed `Fourier1D` object across multiple transforms to amortise the
plan-measurement cost.

# Arguments
- `fp::FourierParameters`: Ring configuration

# Returns
- `Fourier1D`: Initialised ring object ready for transform calls

# Example
```julia
fp  = Fourier.FourierParameters(ymin=0.0, kmax=10, yDim=64, bDim=21)
ring = Fourier.Fourier1D(fp)
```

See also: [`FBtransform`](@ref), [`FAtransform`](@ref), [`FItransform`](@ref)
"""
function Fourier1D(fp::FourierParameters)

    # Calculate evenly spaced points along the ring
    mishPoints = calcMishPoints(fp)

    # Initialize the arrays to zero
    uMish = zeros(real,fp.yDim)
    b = zeros(real,fp.bDim)
    a = zeros(real,fp.yDim)
    ax = zeros(real,fp.yDim)
    
    # Plan the FFT
    fftPlan = FFTW.plan_r2r(a, FFTW.FFTW.R2HC, flags=FFTW.PATIENT)
    ifftPlan = FFTW.plan_r2r(a, FFTW.FFTW.HC2R, flags=FFTW.PATIENT)
    
    # Pre-calculate the phase filter matrix
    phasefilter = calcPhaseFilter(fp)
    invphasefilter = calcInvPhaseFilter(fp)

    # Scratch buffer for in-place FBtransform!/FAtransform!
    scratch_fft = zeros(real, fp.yDim)

    # Construct the Fourier1D ring object
    ring = Fourier1D(fp,mishPoints,fftPlan,ifftPlan,phasefilter,invphasefilter,
                     uMish,b,a,ax,scratch_fft)
    return ring
end

"""
    calcPhaseFilter(fp::FourierParameters) -> Matrix{Float64}

Build the phase-shift and anti-aliasing filter matrix of size `(yDim, bDim)`.

Combines two operations into a single matrix multiplication applied after the forward FFT:
1. **Phase shift**: rotates each wavenumber `k` by angle `−k·ymin` so that spectral
   coefficients are referenced to a common `ymin = 0` origin, enabling coherent
   aggregation along an orthogonal dimension (e.g. with cubic B-splines).
2. **Anti-aliasing truncation**: keeps only wavenumbers `0:kmax`; higher modes are not
   present in the `bDim`-column output.

Block structure of the matrix:
- `(1, 1)` → wavenumber 0 (constant term pass-through).
- For each `k = 1:kmax`: a 2×2 rotation block placed at rows `(k+1, yDim-k+1)`
  and columns `(k+1, bDim-k+1)`.

# Returns
- `Matrix{Float64}`: Phase-filter matrix of size `(yDim, bDim)`

See also: [`calcInvPhaseFilter`](@ref), [`FBtransform`](@ref)
"""
function calcPhaseFilter(fp::FourierParameters)

    #= Phase filter is a matrix that both shifts the phase and filters the Fourier coefficients
    The phase shifter aligns the coefficients to a common phase relative to ymin = 0
    The filter removes all high wavenumber coefficients above kmax that may be associated with aliasing or other unwanted small-scale features
    The combined filter and phase shift allows the amplitude of the coefficients to be aggregated along another dimension, for example with cubic bSplines =#
    phasefilter = zeros(real, fp.yDim, fp.bDim)
    phasefilter[1,1] = 1.0
    for k in 1:fp.kmax
        phasefilter[k+1,k+1] = cos(-k * fp.ymin)
        phasefilter[fp.yDim-k+1,k+1] = -sin(-k * fp.ymin)
        phasefilter[k+1,fp.bDim-k+1] = sin(-k * fp.ymin)
        phasefilter[fp.yDim-k+1,fp.bDim-k+1] = cos(-k * fp.ymin)
    end
    return phasefilter

end

"""
    calcInvPhaseFilter(fp::FourierParameters) -> Matrix{Float64}

Build the inverse phase-shift and zero-padding matrix of size `(bDim, yDim)`.

The inverse of [`calcPhaseFilter`](@ref). Applies angle `+k·ymin` to each wavenumber
to undo the phase shift, then zero-pads the `bDim` filtered coefficients back to a
`yDim`-length array for the subsequent inverse FFT.

Block structure mirrors `calcPhaseFilter` with opposite sign rotation.

# Returns
- `Matrix{Float64}`: Inverse phase-filter matrix of size `(bDim, yDim)`

See also: [`calcPhaseFilter`](@ref), [`FAtransform`](@ref)
"""
function calcInvPhaseFilter(fp::FourierParameters)

    # The inverse phase filter shifts the phase back to the original phase, and pads the Fourier coefficients with zeros back up to the full yDim space for the subsequent inverse FFT
    invphasefilter = zeros(real, fp.bDim, fp.yDim)
    invphasefilter[1,1] = 1.0
    for k in 1:fp.kmax
        invphasefilter[k+1,k+1] = cos(k * fp.ymin)
        invphasefilter[fp.bDim-k+1,k+1] = -sin(k * fp.ymin)
        invphasefilter[k+1,fp.yDim-k+1] = sin(k * fp.ymin)
        invphasefilter[fp.bDim-k+1,fp.yDim-k+1] = cos(k * fp.ymin)
    end
    return invphasefilter

end

"""
    calcMishPoints(fp::FourierParameters) -> Vector{Float64}

Compute the evenly-spaced grid point locations around the Fourier ring.

Points are placed at angles `ymin + 2π*(n-1)/yDim` for `n = 1:yDim`. The grid is
**half-open**: the last point is one step before `ymin + 2π`, so the starting angle is
not repeated when the ring closes.

# Returns
- `Vector{Float64}`: Ring angles of length `yDim`, beginning at `ymin`

See also: [`Fourier1D`](@ref), [`FBtransform`](@ref)
"""
function calcMishPoints(fp::FourierParameters)

    # Calculate the evenly spaced points around a ring for the FFT
    Nbasis = fp.yDim
    y = zeros(real,Nbasis)
    for n = 1:Nbasis
        y[n] = fp.ymin + (2 * π * (n-1) / Nbasis)
    end
    return y
end

"""
    FBtransform(fp::FourierParameters, fftPlan, phasefilter::Matrix{Float64}, uMish::Vector{Float64}) -> Vector{Float64}
    FBtransform(ring::Fourier1D, uMish::Vector{Float64}) -> Vector{Float64}
    FBtransform!(ring::Fourier1D)

Compute the forward Fourier transform (physical → filtered B-coefficients).

``b = \\text{phasefilter}^T \\cdot \\bigl(\\text{FFT}(u) \\,/\\, N\\bigr)``

The `FFTW.R2HC` (real-to-halfcomplex) output is divided by `yDim` so that an
amplitude-1 sinusoid in physical space maps to amplitude-1 in `b`. The phase-filter
matrix then applies the azimuthal phase shift and zeroes wavenumbers above `kmax`,
producing a compact `bDim`-length coefficient vector.

# Variants
- `FBtransform(fp, fftPlan, phasefilter, uMish)` — allocates; requires explicit plan and
  filter objects (useful when constructing per-radius rings with varying parameters)
- `FBtransform(ring, uMish)` — allocates using the `ring`'s cached plan and phase-filter
- `FBtransform!(ring)` — in-place; reads `ring.uMish`, writes `ring.b`

# Arguments
- `fp::FourierParameters`: Ring parameters
- `fftPlan`: Pre-measured `FFTW.r2rFFTWPlan` for the forward transform
- `phasefilter::Matrix{Float64}`: Phase-shift + truncation matrix from [`calcPhaseFilter`](@ref)
- `uMish::Vector{Float64}`: Physical field values at the `yDim` ring points

# Returns
- `Vector{Float64}`: Filtered B-coefficient vector of length `bDim` (allocating variants)

See also: [`FAtransform`](@ref), [`FItransform`](@ref)
"""
function FBtransform(fp::FourierParameters, fftPlan, phasefilter::Matrix{real}, uMish::Vector{real})

    # Do the forward Fourier transform, scale, and filter
    b = (fftPlan * uMish) ./ fp.yDim
    bfilter = (b' * phasefilter)'
    return bfilter
end

function FBtransform(ring::Fourier1D, uMish::Vector{real})
    b = (ring.fftPlan * uMish) ./ ring.params.yDim
    return (b' * ring.phasefilter)'
end

function FBtransform!(ring::Fourier1D)
    # In-place forward FFT, scale, and phase-filter using pre-allocated scratch.
    # _scratch_fft holds the raw FFT result; ring.b receives phasefilter' * scaled.
    mul!(ring._scratch_fft, ring.fftPlan, ring.uMish)
    scale = 1.0 / ring.params.yDim
    @. ring._scratch_fft *= scale
    # ring.b = phasefilter' * _scratch_fft (replaces (scratch' * phasefilter)' pattern)
    mul!(ring.b, ring.phasefilter', ring._scratch_fft)
    return ring.b
end

"""
    FAtransform(fp::FourierParameters, invphasefilter::Matrix{Float64}, b::Vector{Float64}) -> Vector{Float64}
    FAtransform(ring::Fourier1D, b::AbstractVector) -> Vector{Float64}
    FAtransform!(ring::Fourier1D)

Compute the FA transform: convert filtered B-coefficients to zero-padded A-coefficients
ready for the inverse FFT.

``a = \\text{invphasefilter}^T \\cdot b``

Applies the inverse phase-shift to undo the azimuthal offset introduced in
[`FBtransform`](@ref) and pads the `bDim`-length `b` back to a `yDim`-length `a`
(with zeros for wavenumbers above `kmax`).

# Variants
- `FAtransform(fp, invphasefilter, b)` — allocates; requires explicit inverse-filter object
- `FAtransform(ring, b)` — allocates using the `ring`'s cached `invphasefilter`
- `FAtransform!(ring)` — in-place; reads `ring.b`, writes `ring.a`

# Returns
- `Vector{Float64}`: Zero-padded A-coefficients of length `yDim` (allocating variants)

See also: [`FBtransform`](@ref), [`FItransform`](@ref)
"""
function FAtransform(fp::FourierParameters, invphasefilter::Matrix{real}, b::Vector{real})

    # Apply the inverse phasefilter to get padded Fourier coefficients for inverse FFT
    a = (b' * invphasefilter)'
    return a
end

function FAtransform(ring::Fourier1D, b::AbstractVector)
    return (b' * ring.invphasefilter)'
end

function FAtransform!(ring::Fourier1D)
    # In-place inverse phase-filter via mul!.
    # ring.a = invphasefilter' * ring.b (replaces (b' * invphasefilter)' pattern)
    mul!(ring.a, ring.invphasefilter', ring.b)
    return ring.a
end

"""
    FItransform(fp::FourierParameters, ifftPlan, a::Vector{Float64}) -> Vector{Float64}
    FItransform!(ring::Fourier1D)

Compute the inverse Fourier transform (A-coefficients → physical values).

``u = \\text{IFFT}(a)``

Performs a `FFTW.HC2R` (halfcomplex-to-real) inverse FFT on the `yDim`-length padded
A-coefficient vector. No additional scaling is required; the forward normalisation
applied in [`FBtransform`](@ref) ensures that the round-trip `FB → FA → FI` recovers
the original physical values.

# Variants
- `FItransform(fp, ifftPlan, a)` — allocates and returns `uMish`
- `FItransform!(ring)` — in-place; reads `ring.a`, writes `ring.uMish`

# Returns
- `Vector{Float64}`: Physical field values of length `yDim` (allocating variant)

See also: [`FBtransform`](@ref), [`FAtransform`](@ref)
"""
function FItransform(fp::FourierParameters, ifftPlan, a::Vector{real})

    # Do the inverse Fourier transform to get back physical values
    uMish = ifftPlan * a
    return uMish
end

function FItransform!(ring::Fourier1D)

    # Do the inverse transform to get back physical values in place
    ring.uMish .= ring.ifftPlan * ring.a
end

"""
    FIxcoefficients(fp::FourierParameters, a::Vector{Float64}, ax::Vector{real}) -> Vector{Float64}

Convert A-coefficients to first-derivative A-coefficients in-place.

Exploits the analytic Fourier differentiation rule: multiplying the cosine coefficient of
wavenumber `k` by `+k` and the sine coefficient by `−k` gives the derivative.
In the `FFTW.HC2R` halfcomplex layout:

- `ax[k+1]      = −k · a[yDim-k+1]`  (cosine component of `∂/∂θ`)
- `ax[yDim-k+1] = +k · a[k+1]`       (sine component of `∂/∂θ`)

The constant term (`k = 0`) is not modified (derivative of a constant is zero).
**Mutates** `ax` in-place and also returns it.

# Arguments
- `fp::FourierParameters`: Ring parameters (provides `kmax` and `yDim`)
- `a::Vector{Float64}`: Source A-coefficient vector of length `yDim`
- `ax::Vector{real}`: Pre-allocated output buffer of length `yDim`; overwritten on return

# Returns
- `ax`: Derivative A-coefficient vector

See also: [`FIxtransform`](@ref), [`FIxxtransform`](@ref)
"""
function FIxcoefficients(fp::FourierParameters, a::Vector{real}, ax::Vector{real})

    # Calculate the 1st derivative coefficients
    for k = 1:fp.kmax
        ax[k+1] = -k * a[fp.yDim-k+1]
        ax[fp.yDim-k+1] = k * a[k+1]
    end
    return ax
end

"""
    FIxtransform(fp::FourierParameters, ifftPlan, a::Vector{Float64}, ax::Vector{Float64}) -> Vector{Float64}
    FIxtransform(ring::Fourier1D) -> Vector{Float64}
    FIxtransform(ring::Fourier1D, ux::AbstractVector) -> AbstractVector

Evaluate the **first azimuthal derivative** ``\\partial u / \\partial \\theta`` in physical space.

Applies [`FIxcoefficients`](@ref) to convert A-coefficients to derivative coefficients,
then performs an inverse FFT.

# Variants
- `FIxtransform(fp, ifftPlan, a, ax)` — allocates; `ax` is overwritten as a side effect
- `FIxtransform(ring)` — allocates a new output vector; uses `ring.a` and `ring.ax`
- `FIxtransform(ring, ux)` — writes into pre-allocated `ux`; uses `ring.a` and `ring.ax`

# Returns
- `Vector{Float64}`: First-derivative values of length `yDim`

See also: [`FIxcoefficients`](@ref), [`FIxxtransform`](@ref), [`FItransform`](@ref)
"""
function FIxtransform(fp::FourierParameters, ifftPlan, a::Vector{real}, ax::Vector{real})

    # Do the inverse transform with derivative coefficients to get back physical values
    ux = ifftPlan * FIxcoefficients(fp,a,ax)
    return ux
end

function FIxtransform(ring::Fourier1D)

    # Do the inverse transform with derivative coefficients in place
    ux = ring.ifftPlan * FIxcoefficients(ring.params,ring.a,ring.ax)
    return ux
end

function FIxtransform(ring::Fourier1D, ux::AbstractVector)

    # Do the inverse transform with derivative coefficients in place with a preallocated buffer
    ux .= ring.ifftPlan * FIxcoefficients(ring.params,ring.a,ring.ax)
    return ux
end

"""
    FIxxtransform(ring::Fourier1D) -> Vector{Float64}

Evaluate the **second azimuthal derivative** ``\\partial^2 u / \\partial \\theta^2`` in physical space.

Applies [`FIxcoefficients`](@ref) twice: first to obtain first-derivative coefficients
`a'`, then again on `a'` to obtain `a''`. A `copy` of the intermediate buffer is used
to avoid overwriting the `ax` working buffer mid-computation. Performs one inverse FFT
on the final `a''` coefficients.

# Returns
- `Vector{Float64}`: Second-derivative values of length `yDim`

See also: [`FIxtransform`](@ref), [`FItransform`](@ref)
"""
function FIxxtransform(ring::Fourier1D)

    # Do the inverse transform with 2nd derivative coefficients
    ax = copy(FIxcoefficients(ring.params,ring.a,ring.ax))
    uxx = FIxtransform(ring.params, ring.ifftPlan, ax, ring.ax)
    return uxx
end

"""
    FIIntcoefficients(fp::FourierParameters, a::Vector{Float64}, aInt::Vector{Float64}, C0::Float64 = 0.0) -> Vector{Float64}

Convert A-coefficients to indefinite-integral A-coefficients in-place.

Exploits the analytic Fourier integration rule:

- `aInt[k+1]      =  a[yDim-k+1] / k`   (integral of ``k``-th cosine → sine/k)
- `aInt[yDim-k+1] = -a[k+1] / k`        (integral of ``k``-th sine → -cosine/k)
- `aInt[1]         = C0`                 (constant of integration)

**Mutates** `aInt` in-place and also returns it.

# Arguments
- `fp::FourierParameters`: Ring parameters (provides `kmax` and `yDim`)
- `a::Vector{Float64}`: Source A-coefficient vector of length `yDim`
- `aInt::Vector{Float64}`: Pre-allocated output buffer of length `yDim`; overwritten on return
- `C0::Float64`: Constant of integration (default `0.0`); sets the mean of the integral

# Returns
- `aInt`: Integral A-coefficient vector

See also: [`FIInttransform`](@ref)
"""
function FIIntcoefficients(fp::FourierParameters, a::Vector{real}, aInt::Vector{real}, C0::real = 0.0)

    # Calculate the integral coefficients, where C0 is an optional constant of integration
    aInt[1] = C0
    for k = 1:fp.kmax
        aInt[k+1] = a[fp.yDim-k+1] / k
        aInt[fp.yDim-k+1] = -a[k+1] / k
    end
    return aInt
end

"""
    FIInttransform(fp::FourierParameters, ifftPlan, a::Vector{Float64}, aInt::Vector{Float64}, C0::Float64 = 0.0) -> Vector{Float64}
    FIInttransform(ring::Fourier1D, C0::Float64 = 0.0) -> Vector{Float64}

Compute the **indefinite integral** of the Fourier expansion in physical space.

Applies [`FIIntcoefficients`](@ref) to convert A-coefficients to integral coefficients,
then performs an inverse FFT. The constant of integration `C0` sets the mean value of
the resulting field.

# Variants
- `FIInttransform(fp, ifftPlan, a, aInt, C0)` — allocates; `aInt` is overwritten
- `FIInttransform(ring, C0)` — convenience form using the cached `ring` internals

# Arguments
- `C0::Float64`: Constant of integration (default `0.0`)

# Returns
- `Vector{Float64}`: Integral field values of length `yDim`

See also: [`FIIntcoefficients`](@ref), [`FIxtransform`](@ref)
"""
function FIInttransform(fp::FourierParameters, ifftPlan, a::Vector{real}, aInt::Vector{real}, C0::real = 0.0)

    # Do the inverse transform with integral coefficients to get back physical values
    return ifftPlan * FIIntcoefficients(fp,a,aInt,C0)
end

function FIInttransform(ring::Fourier1D, C0::real = 0.0)

    # Do the inverse transform with integral coefficients in place
    return FIInttransform(ring.params,ring.ifftPlan,ring.a,ring.ax,C0)
end

# ---------------------------------------------------------------------------
# DFT matrix representations
# ---------------------------------------------------------------------------

"""
    dft_matrix(ring::Fourier1D) -> Matrix{Float64}

Build the ``(N \\times M)`` DFT evaluation matrix for the Fourier basis so that
`dft_matrix(ring) * ring.b` recovers the physical field values at the mish points.

Columns follow the B-coefficient layout produced by [`FBtransform`](@ref):
column 1 is the constant (wavenumber 0), columns `k+1` correspond to
``\\cos(k\\theta)`` and columns `bDim-k+1` to ``\\sin(k\\theta)`` for
`k = 1:kmax`. The FFTW normalisation convention means cosine columns carry
a factor of 2 and sine columns a factor of ``-2``.

See also: [`dft_1st_derivative`](@ref), [`dft_2nd_derivative`](@ref)
"""
function dft_matrix(ring::Fourier1D)
    fp = ring.params
    M = zeros(Float64, fp.yDim, fp.bDim)
    pts = ring.mishPoints
    for i = 1:fp.yDim
        # Column 1: constant term (weight 1)
        M[i, 1] = 1.0
        for k = 1:fp.kmax
            # Cosine column at index k+1 (weight 2 from FFTW normalisation)
            M[i, k+1] = 2.0 * cos(k * pts[i])
            # Sine column at index bDim-k+1 (weight -2 from FFTW normalisation)
            M[i, fp.bDim-k+1] = -2.0 * sin(k * pts[i])
        end
    end
    return M
end

"""
    dft_1st_derivative(ring::Fourier1D) -> Matrix{Float64}

Build the ``(N \\times M)`` first-derivative DFT matrix for the Fourier basis so that
`dft_1st_derivative(ring) * ring.b` gives ``\\partial u / \\partial \\theta`` at the
mish points.

Uses the analytic rules ``d/d\\theta[\\cos(k\\theta)] = -k\\sin(k\\theta)`` and
``d/d\\theta[\\sin(k\\theta)] = k\\cos(k\\theta)`` combined with the FFTW
normalisation factors.

See also: [`dft_matrix`](@ref), [`dft_2nd_derivative`](@ref)
"""
function dft_1st_derivative(ring::Fourier1D)
    fp = ring.params
    M = zeros(Float64, fp.yDim, fp.bDim)
    pts = ring.mishPoints
    for i = 1:fp.yDim
        # Column 1: derivative of constant = 0
        M[i, 1] = 0.0
        for k = 1:fp.kmax
            # d/dx [2*cos(kx)] = -2k*sin(kx) → cosine column
            M[i, k+1] = -2.0 * k * sin(k * pts[i])
            # d/dx [-2*sin(kx)] = -2k*cos(kx) → sine column
            M[i, fp.bDim-k+1] = -2.0 * k * cos(k * pts[i])
        end
    end
    return M
end

"""
    dft_2nd_derivative(ring::Fourier1D) -> Matrix{Float64}

Build the ``(N \\times M)`` second-derivative DFT matrix for the Fourier basis so
that `dft_2nd_derivative(ring) * ring.b` gives
``\\partial^2 u / \\partial \\theta^2`` at the mish points.

Uses the analytic rules ``d^2/d\\theta^2[\\cos(k\\theta)] = -k^2\\cos(k\\theta)``
and ``d^2/d\\theta^2[\\sin(k\\theta)] = -k^2\\sin(k\\theta)`` combined with the
FFTW normalisation factors.

See also: [`dft_matrix`](@ref), [`dft_1st_derivative`](@ref)
"""
function dft_2nd_derivative(ring::Fourier1D)
    fp = ring.params
    M = zeros(Float64, fp.yDim, fp.bDim)
    pts = ring.mishPoints
    for i = 1:fp.yDim
        M[i, 1] = 0.0
        for k = 1:fp.kmax
            # d²/dx² [2*cos(kx)] = -2k²*cos(kx) → cosine column
            M[i, k+1] = -2.0 * k^2 * cos(k * pts[i])
            # d²/dx² [-2*sin(kx)] = 2k²*sin(kx) → sine column
            M[i, fp.bDim-k+1] = 2.0 * k^2 * sin(k * pts[i])
        end
    end
    return M
end

"""
    FItransform_matrix(ring::Fourier1D, points::Vector{Float64}, derivative::Int=0) -> Matrix{Float64}

Build the Fourier evaluation matrix at arbitrary angular positions `points`.

Each row corresponds to one evaluation point and each column to one Fourier
coefficient (in FFTW halfcomplex order, matching [`dft_matrix`](@ref)).

# Arguments
- `ring::Fourier1D`: Fourier ring object (provides `params.kmax` and `params.bDim`)
- `points::Vector{Float64}`: Angular evaluation locations (radians)
- `derivative::Int`: Derivative order (0, 1, or 2; default `0`)

# Returns
- `Matrix{Float64}` of size `(length(points), bDim)`

See also: [`dft_matrix`](@ref), [`SItransform_matrix`](@ref), [`CItransform_matrix`](@ref)
"""
function FItransform_matrix(ring::Fourier1D, points::Vector{Float64}, derivative::Int=0)
    fp = ring.params
    np = length(points)
    M = zeros(Float64, np, fp.bDim)
    for i in 1:np
        θ = points[i]
        if derivative == 0
            M[i, 1] = 1.0
            for k = 1:fp.kmax
                M[i, k+1] = 2.0 * cos(k * θ)
                M[i, fp.bDim-k+1] = -2.0 * sin(k * θ)
            end
        elseif derivative == 1
            M[i, 1] = 0.0
            for k = 1:fp.kmax
                M[i, k+1] = -2.0 * k * sin(k * θ)
                M[i, fp.bDim-k+1] = -2.0 * k * cos(k * θ)
            end
        elseif derivative == 2
            M[i, 1] = 0.0
            for k = 1:fp.kmax
                M[i, k+1] = -2.0 * k^2 * cos(k * θ)
                M[i, fp.bDim-k+1] = 2.0 * k^2 * sin(k * θ)
            end
        else
            throw(ArgumentError("Derivative order $derivative not supported (use 0, 1, or 2)"))
        end
    end
    return M
end

# ---------------------------------------------------------------------------
# Generic wrappers (no "F" prefix) dispatching on Fourier1D
# These enable abstract 1D basis code that calls Btransform!, Itransform!, etc.
# without needing to know the underlying basis type.
# ---------------------------------------------------------------------------

"""Generic B-transform wrapper for `Fourier1D` (allocating). Delegates to [`FBtransform`](@ref)."""
Btransform(ring::Fourier1D, uMish::Vector{real}) = FBtransform(ring, uMish)

"""Generic in-place B-transform wrapper for `Fourier1D`. Delegates to `FBtransform!`."""
Btransform!(ring::Fourier1D) = FBtransform!(ring)

"""Generic A-transform wrapper for `Fourier1D` (allocating). Delegates to [`FAtransform`](@ref)."""
Atransform(ring::Fourier1D, b::AbstractVector) = FAtransform(ring, b)

"""Generic in-place A-transform wrapper for `Fourier1D`. Delegates to `FAtransform!`."""
Atransform!(ring::Fourier1D) = FAtransform!(ring)

"""Generic in-place I-transform wrapper for `Fourier1D`. Delegates to `FItransform!`."""
Itransform!(ring::Fourier1D) = FItransform!(ring)

"""Generic I-transform wrapper for `Fourier1D` (in-place output). Performs inverse FFT from `ring.a` and writes into `u`."""
function Itransform(ring::Fourier1D, u::AbstractVector)
    u .= ring.ifftPlan * ring.a
    return u
end

"""Generic Ix-transform wrapper for `Fourier1D` (allocating). Delegates to [`FIxtransform`](@ref)."""
Ixtransform(ring::Fourier1D) = FIxtransform(ring)

"""Generic Ix-transform wrapper for `Fourier1D` (in-place output). Delegates to [`FIxtransform`](@ref)."""
Ixtransform(ring::Fourier1D, ux::AbstractVector) = FIxtransform(ring, ux)

"""Generic Ixx-transform wrapper for `Fourier1D` (allocating). Delegates to [`FIxxtransform`](@ref)."""
Ixxtransform(ring::Fourier1D) = FIxxtransform(ring)

"""Generic indefinite-integral transform wrapper for `Fourier1D`. Delegates to [`FIInttransform`](@ref)."""
IInttransform(ring::Fourier1D, C0::real = 0.0) = FIInttransform(ring, C0)

"""Generic I-transform matrix wrapper for `Fourier1D`. Delegates to [`FItransform_matrix`](@ref)."""
Itransform_matrix(ring::Fourier1D, points::Vector{Float64}, derivative::Int=0) =
    FItransform_matrix(ring, points, derivative)

#Module end
end

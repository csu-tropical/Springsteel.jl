module Chebyshev

using LinearAlgebra
using FFTW

export ChebyshevParameters, Chebyshev1D
export CBtransform, CBtransform!, CAtransform, CAtransform!, CItransform!
export CBxtransform, CIxtransform, CIxxtransform, CIInttransform
# Generic (no-prefix) wrappers for abstract 1D basis dispatch
export Btransform, Btransform!, Atransform, Atransform!, Itransform, Itransform!
export Ixtransform, Ixxtransform, IInttransform

#Define some convenient aliases
const real = Float64
const int = Int64
const uint = UInt64

# Homogeneous boundary conditions following Ooyama (2002) rank/type nomenclature.
# The *rank* r is the number of constraints imposed at the boundary.
# The *type* t identifies which derivative is constrained
# (T0 = value, T1 = first derivative, T2 = second derivative).
# Not all B-spline BCs are available for Chebyshev; only R0 through R2 variants
# and R3 are supported. Inhomogeneous conditions are not yet implemented.

"""
Rank-0 boundary condition (Ooyama 2002, Eq. 3.2a): **no constraint** at the
boundary. All Chebyshev coefficients remain free. Use when no physical
condition needs to be enforced at the domain edge.
"""
const R0 = Dict("R0" => 0)

"""
Rank-1, type-0 boundary condition (Ooyama 2002, Eq. 3.2b): **zero field value**
at the boundary, ``u(z_0) = 0`` (homogeneous Dirichlet). Implemented via the
global affine (bottom-anchor) method.
"""
const R1T0 = Dict("α0" =>  0.0)

"""
Rank-1, type-1 boundary condition (Ooyama 2002, Eq. 3.2c): **zero first
derivative** at the boundary, ``u'(z_0) = 0`` (homogeneous Neumann).
Implemented via the Wang et al. (1993) global-coefficient method.
"""
const R1T1 = Dict("α1" =>  0.0)

"""
Rank-1, type-2 boundary condition (Ooyama 2002, Eq. 3.2d): **zero second
derivative** at the boundary, ``u''(z_0) = 0``.
"""
const R1T2 = Dict("α2" =>  0.0)

"""
Rank-2, type-1-0 boundary condition (Ooyama 2002, Eq. 3.2f): **zero value and
zero first derivative** at the boundary, ``u(z_0) = u'(z_0) = 0``.
Appropriate for a symmetrically reflecting boundary.
"""
const R2T10 = Dict("β1" => 0.0, "β2" => 0.0)

"""
Rank-2, type-2-0 boundary condition (Ooyama 2002, Eq. 3.2g): **zero value and
zero second derivative** at the boundary, ``u(z_0) = u''(z_0) = 0``.
Forces the field to be **antisymmetric** with respect to the boundary.
"""
const R2T20 = Dict("β1" => 0.0, "β2" => 0.0)

"""
Rank-3 boundary condition (Ooyama 2002, Eq. 3.2h): **zero value, zero first
derivative, and zero second derivative** at the boundary,
``u(z_0) = u'(z_0) = u''(z_0) = 0``. Eliminates all three border
coefficients. Relevant for domain-nesting inhomogeneous BCs (future work)."""
const R3 = Dict("R3" => 0)

"""
    ChebyshevParameters

Immutable parameter struct (using `@kwdef`) for a 1D Chebyshev column basis.

The physical domain `[zmin, zmax]` is sampled at `zDim` Chebyshev–Gauss–Lobatto (CGL)
points clustered near both endpoints:
``z_j = \\cos\\!\\left(\\frac{(j-1)\\pi}{N-1}\\right)``
mapped to the physical range. This gives spectral accuracy for smooth functions.

# Fields
- `zmin::Float64`: Bottom of the vertical domain (e.g. 0 m or a pressure level).
- `zmax::Float64`: Top of the vertical domain.
- `zDim::Int64`: Number of CGL nodes (physical grid points). Both endpoints are included.
- `bDim::Int64`: Number of retained Chebyshev modes. When `bDim < zDim` a sharp truncation
  matrix is applied; when `bDim == zDim` an exponential Eresman damping filter
  `exp(−36*(k/N)^36)` is used instead.
- `BCB::Dict`: Bottom boundary condition. One of [`R0`](@ref), [`R1T0`](@ref),
  [`R1T1`](@ref), [`R1T2`](@ref), [`R2T10`](@ref), [`R2T20`](@ref), [`R3`](@ref).
- `BCT::Dict`: Top boundary condition dict (same options as `BCB`).

# Notes
- Unlike `CubicBSpline.SplineParameters`, there are no auto-computed fields;
  all six must be set explicitly.
- `zDim` must be ≥ 2 (a DCT-I requires at least two endpoints).

# Example
```julia
cp = Chebyshev.ChebyshevParameters(
    zmin = 0.0, zmax = 10000.0,
    zDim = 25, bDim = 25,
    BCB = Chebyshev.R0, BCT = Chebyshev.R0
)
```

See also: [`Chebyshev1D`](@ref)
"""
Base.@kwdef struct ChebyshevParameters
    zmin::real = 0.0   # Minimum z in meters
    zmax::real = 0.0   # Maximum z in meters
    zDim::int = 0      # Nodal dimension
    bDim::int = 0      # Spectral dimension
    BCB::Dict = R0     # Bottom boundary condition
    BCT::Dict = R0     # Top boundary condition
end

"""
    Chebyshev1D

One-dimensional Chebyshev column object. Construct via `Chebyshev1D(cp::ChebyshevParameters)`.

# Fields
- `params::ChebyshevParameters`: Column configuration (domain bounds, grid size, BCs).
- `mishPoints::Vector{Float64}`: CGL points ordered **from `zmin` up to `zmax`** (increasing
  z), because `scale = -0.5*(zmax-zmin)` flips the cosine so that `cos(0)` maps to `zmin`
  (bottom) and `cos(π)` maps to `zmax` (top). Points are clustered near both endpoints.
- `gammaBC::Array{Float64}`: Spectral BC correction factor. A zero `Vector` for `R0/R0`
  (no correction); a rank-1 `Vector` or full `N×N` `Matrix` for Neumann/Robin combinations.
  The polymorphic `Array` type annotation covers both cases.
- `fftPlan::FFTW.r2rFFTWPlan`: Pre-measured `FFTW.REDFT00` (DCT-I) plan. DCT-I is symmetric
  around both endpoints, consistent with the CGL grid that includes both `zmin` and `zmax`.
  **Do not serialise**; FFTW plans are not portable.
- `filter::Matrix{Float64}`: Size `(bDim, zDim)`. Either a sharp truncation matrix
  (when `bDim < zDim`) or an exponential spectral damping matrix (when `bDim == zDim`).
- `uMish::Vector{Float64}`: Physical field values at the `zDim` CGL points (mutable buffer).
- `b::Vector{Float64}`: Filtered Chebyshev B-coefficients of length `bDim`.
- `a::Vector{Float64}`: BC-corrected A-coefficients of length `zDim`, ready for inverse DCT.
- `ax::Vector{Float64}`: Working buffer of length `zDim` for derivative/integral coefficients.

# Notes
- Constructing `Chebyshev1D` calls `FFTW.plan_r2r` with `FFTW.PATIENT`; first construction
  is slow but subsequent transforms are fast.
- The struct is **not thread-safe** if `uMish`, `b`, `a`, or `ax` are mutated concurrently.

See also: [`ChebyshevParameters`](@ref), [`CBtransform`](@ref), [`CAtransform`](@ref),
[`CItransform`](@ref)
"""
struct Chebyshev1D
    # Parameters for the column
    params::ChebyshevParameters

    # Pre-calculated Chebyshev–Gauss–Lobatto points (extrema of Chebyshev polynomials)
    mishPoints::Vector{real}

    # Vector or matrix that enforces boundary conditions. Concrete union enables
    # union splitting for the dispatch in CAtransform!; abstract `Array{real}`
    # would box every field access (~16 B per CAtransform! call).
    gammaBC::Union{Vector{Float64}, Matrix{Float64}}

    # Measured FFTW Plan
    fftPlan::FFTW.r2rFFTWPlan

    # Filter matrix
    filter::Matrix{real}

    # uMish is the physical values
    # b is the filtered Chebyshev coefficients without BCs
    # a is the padded Chebyshev coefficients with BCs
    # ax is a buffer for derivative and integral coefficients
    uMish::Vector{real}
    b::Vector{real}
    a::Vector{real}
    ax::Vector{real}
    # Scratch buffers for in-place CBtransform!/CAtransform!. _scratch_dct holds
    # the unfiltered DCT result of length zDim. _scratch_bfill holds the
    # zero-padded b vector of length zDim used in the BC fold. _scratch_ax holds
    # an intermediate derivative-coefficient buffer used by CIxxtransform so the
    # primary `ax` buffer is not clobbered between the two CIxcoefficients calls.
    _scratch_dct::Vector{real}
    _scratch_bfill::Vector{real}
    _scratch_ax::Vector{real}
end

"""
    Chebyshev1D(cp::ChebyshevParameters) -> Chebyshev1D

Construct a [`Chebyshev1D`](@ref) object from column parameters.

Pre-computes and caches all state needed for repeated spectral transforms:
1. CGL mish points via [`calcMishPoints`](@ref)
2. BC correction matrix/vector via [`calcGammaBC`](@ref)
3. A `FFTW.REDFT00` (DCT-I) plan measured with `FFTW.PATIENT`
4. A spectral filter matrix via [`calcFilterMatrix`](@ref)
5. Zero-initialised working buffers `uMish`, `b`, `a`, `ax`

Reuse the constructed `Chebyshev1D` object across transforms to amortise the
plan-measurement cost.

# Arguments
- `cp::ChebyshevParameters`: Column configuration

# Returns
- `Chebyshev1D`: Initialised column ready for transform calls

# Example
```julia
cp = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=1.0, zDim=25, bDim=25,
                                    BCB=Chebyshev.R0, BCT=Chebyshev.R0)
col = Chebyshev.Chebyshev1D(cp)
```

See also: [`CBtransform`](@ref), [`CAtransform`](@ref), [`CItransform`](@ref)
"""
function Chebyshev1D(cp::ChebyshevParameters)

    # Constructor for 1D Chebsyshev structure
    mishPoints = calcMishPoints(cp)
    gammaBC = calcGammaBC(cp)

    # Initialize the arrays
    uMish = zeros(real,cp.zDim)
    b = zeros(real,cp.bDim)
    a = zeros(real,cp.zDim)
    ax = zeros(real,cp.zDim)

    # Plan the FFT
    # From the FFTW documentation: FFTW_REDFT00 (DCT-I): even around j=0 and even around j=n-1.
    # If you specify a size-5 REDFT00 (DCT-I) of the data abcde, it corresponds to the DFT of the logical even array abcdedcb of size 8
    fftPlan = FFTW.plan_r2r(a, FFTW.REDFT00, flags=FFTW.PATIENT)

    # Pre-calculate the filter matrix
    filter = calcFilterMatrix(cp)

    # Scratch buffers for in-place transforms
    scratch_dct = zeros(real, cp.zDim)
    scratch_bfill = zeros(real, cp.zDim)
    scratch_ax = zeros(real, cp.zDim)

    # Construct a 1D Chebyshev column
    column = Chebyshev1D(cp,mishPoints,gammaBC,fftPlan,filter,uMish,b,a,ax,
                         scratch_dct,scratch_bfill,scratch_ax)
    return column
end

"""
    calcMishPoints(cp::ChebyshevParameters) -> Vector{Float64}

Compute the Chebyshev–Gauss–Lobatto (CGL) physical grid points for the column.

Points are placed at
``z_j = \\cos\\!\\left(\\frac{(j-1)\\pi}{N-1}\\right) \\cdot s + o``
where `s = -0.5*(zmax−zmin)` (negative scale) and `o = 0.5*(zmin+zmax)`. The negative
scale maps `cos(0) = 1` → `zmin` and `cos(π) = −1` → `zmax`, so the returned vector
runs **from `zmin` up to `zmax`** (increasing z, bottom to top). Points are clustered
near both endpoints according to the CGL distribution.

# Returns
- `Vector{Float64}`: CGL points of length `zDim`, from `zmin` to `zmax`

See also: [`Chebyshev1D`](@ref)
"""
function calcMishPoints(cp::ChebyshevParameters)

    # Calculate the physical Chebyshev points
    # The points are evenly spaced in the interval (0,π)
    # which are then mapped to the physical interval (-1,1) via the cosine function
    # and then scaled and offset to match the physical domain from zmin to zmax
    Nbasis = cp.zDim
    z = zeros(real,Nbasis)
    scale = -0.5 * (cp.zmax - cp.zmin)
    offset = 0.5 * (cp.zmin + cp.zmax)
    for n = 1:Nbasis
        z[n] = cos((n-1) * π / (Nbasis - 1)) * scale + offset
    end
    return z
end

"""
    calcFilterMatrix(cp::ChebyshevParameters) -> Matrix{Float64}

Build the spectral filter/truncation matrix of size `(bDim, zDim)` (when `bDim < zDim`)
or `(zDim, zDim)` with exponential damping (when `bDim == zDim`).

Two branches:
- **Sharp truncation** (`bDim < zDim`): returns the leading `bDim × zDim` rows of the
  identity, zeroing all Chebyshev modes above index `bDim`.
- **Spectral damping** (`bDim == zDim`): returns a diagonal matrix with entries
  ``\\text{filter}_{ii} = \\exp\\!\\left(-36 \\left(\\frac{i}{N}\\right)^{36}\\right)``,
  which leaves low modes essentially unchanged while strongly attenuating modes near the
  Nyquist limit (the Eresman/Boyd exponential filter).

# Returns
- `Matrix{Float64}`: Filter matrix applied after the forward DCT in `CBtransform!`

See also: [`CBtransform`](@ref), [`Chebyshev1D`](@ref)
"""
function calcFilterMatrix(cp::ChebyshevParameters)

    # Create a matrix to truncate the coefficients to bDim after Fourier transformation

    if cp.bDim < cp.zDim
        filter = Matrix(1.0I, cp.bDim, cp.zDim)
        return filter
    else
        filter = Matrix(1.0I, cp.zDim, cp.zDim)
        for i in 1:cp.zDim
            filter[i,i] = exp(-36.0 * (i/cp.zDim)^36)
        end
        return filter
    end
end

"""
    CBtransform(cp::ChebyshevParameters, fftPlan, uMish::Vector{Float64}) -> Vector{Float64}
    CBtransform(column::Chebyshev1D, uMish::Vector{Float64}) -> Vector{Float64}
    CBtransform!(column::Chebyshev1D)

Compute the forward Chebyshev transform (physical → filtered B-coefficients).

``b = \\text{filter} \\cdot \\frac{\\text{DCT}(u)}{2(N-1)}``

The `FFTW.REDFT00` (DCT-I) output is divided by `2*(zDim−1)` so that the discrete
orthogonality relation is satisfied and amplitude-1 polynomials map to amplitude-1
coefficients. The filter matrix then truncates to `bDim` modes or applies exponential
damping per [`calcFilterMatrix`](@ref).

# Variants
- `CBtransform(cp, fftPlan, uMish)` — allocates; requires explicit plan (useful for
  constructing columns with varying parameters without a full `Chebyshev1D` object)
- `CBtransform(column, uMish)` — allocates using `column`'s cached filter and plan
- `CBtransform!(column)` — in-place; reads `column.uMish`, writes `column.b`

# Arguments
- `cp::ChebyshevParameters`: Column parameters
- `fftPlan`: Pre-measured `FFTW.r2rFFTWPlan` for the DCT-I forward transform
- `uMish::Vector{Float64}`: Physical field values at the `zDim` CGL points

# Returns
- `Vector{Float64}`: Filtered B-coefficient vector of length `bDim` (allocating variants)

See also: [`CAtransform`](@ref), [`CItransform`](@ref)
"""
function CBtransform(cp::ChebyshevParameters, fftPlan, uMish::Vector{real})

    # Do the DCT transform and pre-scale the output based on the physical length
    b = (fftPlan * uMish) ./ (2 * (cp.zDim -1))
    return b[1:cp.bDim]
end

function CBtransform!(column::Chebyshev1D)
    # In-place forward DCT + filter, using pre-allocated scratch buffers.
    # _scratch_dct holds the raw DCT output (length zDim); column.b receives
    # the filtered result (length bDim).
    mul!(column._scratch_dct, column.fftPlan, column.uMish)
    scale = 1.0 / (2 * (column.params.zDim - 1))
    @. column._scratch_dct *= scale
    mul!(column.b, column.filter, column._scratch_dct)
    return column.b
end

function CBtransform(column::Chebyshev1D, uMish::Vector{real})

    # Do the DCT transform and pre-scale the output based on the physical length
    b = (column.fftPlan * uMish) ./ (2 * (column.params.zDim -1))
    return column.filter * b
end

"""
    CAtransform(cp::ChebyshevParameters, gammaBC, b::Vector{Float64}) -> Vector{Float64}
    CAtransform(column::Chebyshev1D, b::AbstractVector) -> Vector{Float64}
    CAtransform!(column::Chebyshev1D)

Apply boundary conditions and zero-pad B-coefficients to produce A-coefficients.

``a = b_{\\text{fill}} + \\texttt{gammaBC}^\\top \\cdot b_{\\text{fill}}``

where ``b_{\\text{fill}}`` is `b` zero-padded from `bDim` to `zDim`. The additive structure
means that `gammaBC = 0` (the `R0/R0` Dirichlet case) applies no correction. Julia's `*`
operator handles both vector and matrix `gammaBC` transparently.

The resulting `a` has length `zDim` and is ready for the inverse DCT in [`CItransform`](@ref).

# Variants
- `CAtransform(cp, gammaBC, b)` — allocates; requires explicit `gammaBC` (low-level form)
- `CAtransform(column, b)` — allocates using the `column`'s cached `gammaBC`
- `CAtransform!(column)` — in-place; reads `column.b`, writes `column.a`

# Returns
- `Vector{Float64}`: BC-corrected A-coefficients of length `zDim` (allocating variants)

See also: [`CBtransform`](@ref), [`CItransform`](@ref)
"""
function CAtransform(cp::ChebyshevParameters, gammaBC, b::Vector{real})

    # Apply the boundary conditions and pad the coefficients with zeros back to zDim
    bfill = [b ; zeros(Float64, cp.zDim-cp.bDim)]
    a = bfill .+ (gammaBC' * bfill)
    return a
end

function CAtransform(column::Chebyshev1D, b::AbstractVector)
    bfill = [b ; zeros(Float64, column.params.zDim - length(b))]
    return bfill .+ (column.gammaBC' * bfill)
end

function CAtransform!(column::Chebyshev1D)
    # In-place CA transform using pre-allocated scratch.
    # bfill = b padded with zeros up to zDim; a = bfill + gammaBC' * bfill.
    #
    # gammaBC may be a zero Vector (R0/R0), a rank-1 Vector (simple Neumann),
    # or a full Matrix (Wang global Neumann method). We dispatch by type to
    # avoid the matrix-build that would otherwise happen for Vector cases.
    bfill = column._scratch_bfill
    bDim = column.params.bDim
    zDim = column.params.zDim
    @inbounds for i in 1:bDim
        bfill[i] = column.b[i]
    end
    @inbounds for i in (bDim+1):zDim
        bfill[i] = 0.0
    end
    γ = column.gammaBC
    if γ isa Matrix
        # column.a = γ' * bfill, then add bfill in-place
        mul!(column.a, γ', bfill)
        @. column.a += bfill
    else
        # Vector or scalar gammaBC: fall back to broadcast (one alloc).
        column.a .= bfill .+ (γ' * bfill)
    end
    return column.a
end
    
"""
    CItransform(cp::ChebyshevParameters, fftPlan, a::Vector{Float64}) -> Vector{Float64}
    CItransform!(column::Chebyshev1D)

Compute the inverse Chebyshev transform (A-coefficients → physical values).

``u = \\text{DCT-I}(a)``

Performs the `FFTW.REDFT00` (DCT-I) inverse transform on the `zDim`-length A-coefficient
vector. No additional scaling is needed; the forward normalisation applied in
[`CBtransform`](@ref) ensures that the round-trip `CB → CA → CI` recovers the original
physical values.

# Variants
- `CItransform(cp, fftPlan, a)` — allocates and returns `uMish`
- `CItransform!(column)` — in-place; reads `column.a`, writes `column.uMish`

# Returns
- `Vector{Float64}`: Physical field values of length `zDim` (allocating variant)

See also: [`CBtransform`](@ref), [`CAtransform`](@ref)
"""
function CItransform(cp::ChebyshevParameters, fftPlan, a::Vector{real})

    # Do the inverse DCT transform to get back physical values
    uMish = fftPlan * a
    return uMish
end

function CItransform!(column::Chebyshev1D)

    # In-place inverse DCT transform to get back physical values
    mul!(column.uMish, column.fftPlan, column.a)
    return column.uMish
end


"""
    CIIntcoefficients(cp::ChebyshevParameters, a::Vector{Float64}, C0::Float64 = 0.0) -> Vector{Float64}

Convert A-coefficients to indefinite-integral A-coefficients via the Chebyshev recurrence.

``a^{\\text{int}}_k = \\frac{\\Delta z}{4} \\cdot \\frac{a_{k-1} - a_{k+1}}{k-1}, \\quad k = 2 \\ldots N-1``
``a^{\\text{int}}_N = \\frac{a_{N-1}}{N-1}``
``a^{\\text{int}}_1 = C_0 - 2\\sum_{k=2}^{N} a^{\\text{int}}_k``

where ``\\Delta z = z_{\\max} - z_{\\min}`` and `C0` is the optional constant of integration
(value of the integral at the reference point).

# Arguments
- `cp::ChebyshevParameters`: Column parameters
- `a::Vector{Float64}`: Source A-coefficient vector of length `zDim`
- `C0::Float64`: Constant of integration (default `0.0`)

# Returns
- `Vector{Float64}`: Integral A-coefficient vector (allocates)

See also: [`CIInttransform`](@ref)
"""
function CIIntcoefficients(cp::ChebyshevParameters, a::Vector{real}, C0::real = 0.0)

    # Calculate the integral coefficients using a recursive relationship
    # C0 is an optional constant of integration
    aInt = zeros(real,cp.zDim)
    sum = 0.0
    interval = -0.25 * (cp.zmax - cp.zmin)
    for k = 2:(cp.zDim-1)
        aInt[k] = interval * (a[k-1] - a[k+1]) / (k-1)
        sum += aInt[k]
    end
    aInt[cp.zDim] = a[cp.zDim-1]/(cp.zDim-1)
    sum += aInt[cp.zDim]
    aInt[1] = C0 - (2.0 * sum)
    return aInt
end

"""
    CIInttransform(cp::ChebyshevParameters, fftPlan, a::Vector{Float64}, C0::Float64 = 0.0) -> Vector{Float64}
    CIInttransform(column::Chebyshev1D, C0::Float64 = 0.0) -> Vector{Float64}

Compute the **indefinite integral** of the Chebyshev expansion in physical space.

Applies [`CIIntcoefficients`](@ref) to convert A-coefficients to integral coefficients,
then performs an inverse DCT-I. The constant of integration `C0` sets the reference value.

# Variants
- `CIInttransform(cp, fftPlan, a, C0)` — allocates; requires explicit plan
- `CIInttransform(column, C0)` — convenience form using the cached `column` internals

# Arguments
- `C0::Float64`: Constant of integration (default `0.0`)

# Returns
- `Vector{Float64}`: Integral field values of length `zDim`

See also: [`CIIntcoefficients`](@ref), [`CIxtransform`](@ref)
"""
function CIInttransform(cp::ChebyshevParameters, fftPlan, a::Vector{real}, C0::real = 0.0)

    # Do a transform to get the integral of the column values
    # C0 is an optional constant of integration
    aInt = CIIntcoefficients(cp,a,C0)
    uInt = fftPlan * aInt
    return uInt
end

function CIInttransform(column::Chebyshev1D, C0::real = 0.0)

    # Do a transform to get the integral of the column values
    uInt = CIInttransform(column.params, column.fftPlan, column.a, C0)
    return uInt
end

"""
    CIxcoefficients(cp::ChebyshevParameters, a::Vector{Float64}, ax::Vector{Float64}) -> Vector{Float64}

Convert A-coefficients to first-derivative A-coefficients via the Chebyshev recurrence.

The recurrence proceeds **backwards** from `k = zDim` to `k = 2`:
``a'_{k-1} = 2(k-1) a_k + a'_{k+1}``
then scaled by ``-1 / (0.5 \\cdot (z_{\\max} - z_{\\min}))`` to map from the reference
domain ``[-1,1]`` to the physical domain.

**Mutates** `ax` in-place and also returns it.

# Arguments
- `cp::ChebyshevParameters`: Column parameters
- `a::Vector{Float64}`: Source A-coefficient vector of length `zDim`
- `ax::Vector{Float64}`: Pre-allocated output buffer of length `zDim`; overwritten on return

# Returns
- `ax`: Derivative A-coefficient vector

See also: [`CIxtransform`](@ref), [`CIxxtransform`](@ref)
"""
function CIxcoefficients(cp::ChebyshevParameters, a::Vector{real}, ax::Vector{real})

    # Calculate the derivative coefficients using a recursive relationship
    k = cp.zDim
    ax[k-1] = (2.0 * (k-1) * a[k])
    for k = (cp.zDim-1):-1:2
        ax[k-1] = (2.0 * (k-1) * a[k]) + ax[k+1]
    end
    ax ./= (-0.5 * (cp.zmax - cp.zmin))
    return ax
end

"""
    CIxtransform(cp::ChebyshevParameters, fftPlan, a::Vector{Float64}, ax::Vector{Float64}) -> Vector{Float64}
    CIxtransform(column::Chebyshev1D) -> Vector{Float64}
    CIxtransform(column::Chebyshev1D, ux::AbstractVector) -> AbstractVector

Evaluate the **first vertical derivative** ``\\partial u / \\partial z`` in physical space.

Applies [`CIxcoefficients`](@ref) to convert A-coefficients to derivative coefficients,
then performs an inverse DCT-I.

# Variants
- `CIxtransform(cp, fftPlan, a, ax)` — allocates a new output vector; `ax` is overwritten
- `CIxtransform(column)` — allocates; convenience form using the cached `column` internals
- `CIxtransform(column, ux)` — writes into pre-allocated `ux` buffer and returns it

# Returns
- `Vector{Float64}`: First-derivative values of length `zDim`

See also: [`CIxcoefficients`](@ref), [`CIxxtransform`](@ref), [`CItransform`](@ref)
"""
function CIxtransform(cp::ChebyshevParameters, fftPlan, a::Vector{real}, ax::Vector{real})
    
    # Do the inverse transform to get back the first derivative in physical space
    ux = fftPlan * CIxcoefficients(cp,a,ax)
    return ux
end

function CIxtransform(column::Chebyshev1D)

    # Do the inverse transform to get back the first derivative in physical space
    CIxcoefficients(column.params, column.a, column.ax)
    ux = column.fftPlan * column.ax
    return ux
end

function CIxtransform(column::Chebyshev1D, ux::AbstractVector)
    CIxcoefficients(column.params, column.a, column.ax)
    mul!(ux, column.fftPlan, column.ax)
    return ux
end

"""
    CIxxtransform(column::Chebyshev1D) -> Vector{Float64}

Evaluate the **second vertical derivative** ``\\partial^2 u / \\partial z^2`` in physical space.

Applies [`CIxcoefficients`](@ref) twice: first on `column.a` to obtain first-derivative
coefficients `a'`, then again on `a'` to obtain `a''`. A `copy` of the intermediate `a'`
buffer is used to avoid overwriting the `ax` working buffer mid-computation. Performs one
inverse DCT-I on the final `a''` coefficients.

# Returns
- `Vector{Float64}`: Second-derivative values of length `zDim`

See also: [`CIxtransform`](@ref), [`CItransform`](@ref)
"""
function CIxxtransform(column::Chebyshev1D)

    # Do the inverse transform to get back the second derivative in physical space.
    # First derivative coefs are stashed in _scratch_ax so the second pass through
    # CIxcoefficients can write into column.ax without clobbering its own input.
    CIxcoefficients(column.params, column.a, column.ax)
    copyto!(column._scratch_ax, column.ax)
    CIxcoefficients(column.params, column._scratch_ax, column.ax)
    uxx = column.fftPlan * column.ax
    return uxx
end

function CIxxtransform(column::Chebyshev1D, uxx::AbstractVector)
    CIxcoefficients(column.params, column.a, column.ax)
    copyto!(column._scratch_ax, column.ax)
    CIxcoefficients(column.params, column._scratch_ax, column.ax)
    mul!(uxx, column.fftPlan, column.ax)
    return uxx
end


"""
    dct_matrix(Nbasis::Int64) -> Matrix{Float64}

Build the full DCT-I evaluation matrix of size `(Nbasis, Nbasis)`.

Entry `[i,j] = 2 cos((j-1) * (i-1)*π/(N-1))` with endpoint columns scaled by 0.5.
Useful for debugging transforms and constructing linear solvers directly in spectral space.

See also: [`dct_1st_derivative`](@ref), [`dct_2nd_derivative`](@ref)
"""
function dct_matrix(Nbasis::Int64)
    
    # Create a matrix with the DCT as basis functions
    # This function is used for debugging and also for linear solvers
    dct = zeros(Float64,Nbasis,Nbasis)
    for i = 1:Nbasis
        t = (i-1) * π / (Nbasis - 1)
        for j = 1:Nbasis
            dct[i,j] = 2*cos((j-1)*t)
        end
    end
    dct[:,1] *= 0.5
    dct[:,Nbasis] *= 0.5
    return dct
end

"""
    dct_1st_derivative(Nbasis::Int64, physical_length::Float64) -> Matrix{Float64}

Build the first-derivative matrix in DCT-I basis of size `(Nbasis, Nbasis)`.

Applies spectral differentiation analytically: entry `[i,j]` is the value of
``dT_n/dx`` at the `i`-th CGL node, scaled by `1 / (physical_length/4)` to convert
from the reference domain ``[-1,1]`` to the physical domain. Endpoint rows use the
limiting formula to avoid a `sin(t) = 0` singularity.

See also: [`dct_matrix`](@ref), [`dct_2nd_derivative`](@ref)
"""
function dct_1st_derivative(Nbasis::Int64, physical_length::Float64)

    # Create a 1st derivative matrix with the DCT as basis functions
    # This function is used for debugging and also for linear solvers
    dct = zeros(Float64,Nbasis,Nbasis)
    for i = 1:Nbasis
        t = (i-1) * π / (Nbasis - 1)
        for j = 1:Nbasis
            N = j-1
            if (i == 1)
                dct[i,j] = -N*N
            elseif (i == Nbasis)
                dct[i,j] = -N*N*(-1.0)^(N+1)
            else
                dct[i,j] = -N*sin(N*t)/sin(t)
            end
        end
    end
    return dct ./ (physical_length/4.0)
end

"""
    dct_2nd_derivative(Nbasis::Int64, physical_length::Float64) -> Matrix{Float64}

Build the second-derivative matrix in DCT-I basis of size `(Nbasis, Nbasis)`.

Analytic second spectral derivative, scaled by `1 / (physical_length^2/8)`. Endpoint rows
use the limiting formula. Useful for BVP solvers and debugging.

See also: [`dct_matrix`](@ref), [`dct_1st_derivative`](@ref)
"""
function dct_2nd_derivative(Nbasis::Int64, physical_length::Float64)

    # Create a 2nd derivative matrix with the DCT as basis functions
    # This function is used for debugging and also for linear solvers
    dct = zeros(Float64,Nbasis,Nbasis)
    for i = 1:Nbasis
        t = (i-1) * π / (Nbasis - 1)
        ct = cos(t)
        st = sin(t)
        for j = 1:Nbasis
            N = j-1
            if (i == 1)
                dct[i,j] = (N^4 - N^2)/3
            elseif (i == Nbasis)
                dct[i,j] = ((-1.0)^N)*(N^4 - N^2)/3
            else
                dct[i,j] = -N*N*cos(N*t)/(st*st) + N*sin(N*t)*ct/(st*st*st)
            end
        end
    end
    return dct ./ (physical_length^2/8.0)
end

"""
    CItransform_matrix(column::Chebyshev1D, points::Vector{Float64}, derivative::Int=0) -> Matrix{Float64}

Build the Chebyshev evaluation matrix at arbitrary physical locations `points`.

Maps physical coordinates to the reference domain via
`t = acos((z - offset) / scale)` where `scale = -(zmax - zmin)/2` and
`offset = (zmin + zmax)/2`, then evaluates Chebyshev polynomials (or their
derivatives) at the mapped locations.

Each row corresponds to one evaluation point and each column to one Chebyshev
coefficient, matching the convention of [`dct_matrix`](@ref).

# Arguments
- `column::Chebyshev1D`: Chebyshev column object (provides parameters and `zDim`)
- `points::Vector{Float64}`: Physical evaluation locations in `[zmin, zmax]`
- `derivative::Int`: Derivative order (0, 1, or 2; default `0`)

# Returns
- `Matrix{Float64}` of size `(length(points), zDim)`

See also: [`dct_matrix`](@ref), [`SItransform_matrix`](@ref), [`FItransform_matrix`](@ref)
"""
function CItransform_matrix(column::Chebyshev1D, points::Vector{Float64}, derivative::Int=0)
    cp = column.params
    N = cp.zDim
    np = length(points)
    scale  = -0.5 * (cp.zmax - cp.zmin)
    offset =  0.5 * (cp.zmin + cp.zmax)
    M = zeros(Float64, np, N)

    for i in 1:np
        ξ_raw = (points[i] - offset) / scale

        if derivative == 0
            # For evaluation, allow exact endpoints (no division by sin(t))
            ξ = clamp(ξ_raw, -1.0, 1.0)
            t = acos(ξ)
            # Match _cheb_eval_pts!: val = a[1] + Σ 2*a[k]*cos(...) + a[N]*cos(...)
            M[i, 1] = 1.0
            for k = 2:(N - 1)
                M[i, k] = 2.0 * cos((k - 1) * t)
            end
            if N > 1
                M[i, N] = cos((N - 1) * t)
            end

        elseif derivative == 1
            # For derivatives, clamp away from endpoints to avoid sin(t)=0
            ξ = clamp(ξ_raw, -1.0 + 1e-10, 1.0 - 1e-10)
            t = acos(ξ)
            # Match _cheb_dz_pts!: df/dz = dfdt / (-scale * sin(t))
            st = sin(t)
            inv_scale_st = 1.0 / (-scale * st)
            M[i, 1] = 0.0
            for k = 2:(N - 1)
                M[i, k] = -2.0 * (k - 1) * sin((k - 1) * t) * inv_scale_st
            end
            if N > 1
                M[i, N] = -(N - 1) * sin((N - 1) * t) * inv_scale_st
            end

        elseif derivative == 2
            ξ = clamp(ξ_raw, -1.0 + 1e-10, 1.0 - 1e-10)
            t = acos(ξ)
            # Match _cheb_dzz_pts!: d²f/dz² = (d²f/dt² * sin - df/dt * cos) / (s² sin³)
            st = sin(t)
            ct = cos(t)
            inv_s2s3 = 1.0 / (scale^2 * st^3)
            M[i, 1] = 0.0
            for k = 2:(N - 1)
                m = k - 1
                dv_dt = -2.0 * m * sin(m * t)
                d2v_dt2 = -2.0 * m^2 * cos(m * t)
                M[i, k] = (d2v_dt2 * st - dv_dt * ct) * inv_s2s3
            end
            if N > 1
                m = N - 1
                dv_dt = -m * sin(m * t)
                d2v_dt2 = -m^2 * cos(m * t)
                M[i, N] = (d2v_dt2 * st - dv_dt * ct) * inv_s2s3
            end
        else
            throw(ArgumentError("Derivative order $derivative not supported (use 0, 1, or 2)"))
        end
    end
    return M
end

"""
    calcGammaBCalt(cp::ChebyshevParameters)

Alternative BC matrix construction via DCT matrix inversion.

!!! warning "Deprecated"
    This function works only for Dirichlet BCs and is retained for reference.
    It does not handle Neumann BCs. Use [`calcGammaBC`](@ref) instead.
"""
function calcGammaBCalt(cp::ChebyshevParameters)
    
    # This works for Dirichelet BCs, but not for Neumann
    # It's also less efficient than the other methods, but not ready to delete this code yet
    Ndim = cp.zDim
    
    if (cp.BCB == R0) && (cp.BCT == R0)
        # Identity matrix
        gammaBC = Matrix(1.0I, Ndim, Ndim)
        return factorize(gammaBC)
    end
    
    # Create the BC matrix
    dctMatrix = calcDCTmatrix(Ndim)
    dctBC = calcDCTmatrix(Ndim)
    
    if haskey(cp.BCB,"α0")
        dctBC[:,1] .= cp.BCB["α0"]
    elseif haskey(cp.BCB,"α1")
        #Not implemented yet
    end

    if haskey(cp.BCT,"α0")
        dctBC[:,Ndim] .= cp.BCT["α0"]
    elseif haskey(cp.BCT,"α1")
        #Not implemented yet
    end
    
    gammaTranspose = dctMatrix' \ dctBC'    
    gammaBC = Matrix{Float64}(undef,25,25)
    gammaBC .= gammaTranspose'
    return gammaBC
end

function _bc_type(bc::Dict)::Symbol
    # Classify a BC dict by its key, ignoring the value.
    # This allows non-homogeneous BCs (e.g. Dict("α0" => 5.0)) to be treated
    # the same as homogeneous ones (R1T0 = Dict("α0" => 0.0)) for gammaBC computation.
    # The non-homogeneous value is handled separately by the solver via row replacement.
    if haskey(bc, "R0")
        return :R0
    elseif haskey(bc, "α0")
        return :R1T0
    elseif haskey(bc, "α1")
        return :R1T1
    else
        throw(DomainError(bc, "Unrecognized Chebyshev BC dict keys"))
    end
end

"""
    calcGammaBC(cp::ChebyshevParameters) -> Union{Vector{Float64}, Matrix{Float64}}

Compute the spectral BC correction term for `CAtransform`.

Following the Ooyama (2002) nomenclature, returns:
- A zero `Vector{Float64}` for `R0/R0` (Dirichlet at both ends — no correction needed)
- A rank-1 `Vector{Float64}` or rank-1 `Matrix{Float64}` for simple Neumann (R1T0/R0, R0/R1T0)
- A full `N×N` `Matrix{Float64}` for Neumann BCs using the Wang et al. (1993) global
  coefficient method (R1T1, R0/R1T1, R1T1/R0, R1T1/R1T1)
- Combined matrices for mixed-BC combinations

Neumann BCs via the global method reference:
> Wang, H., Lacroix, S., & Labrosse, G. (1993). *A Chebyshev collocation method for the
> Navier–Stokes equations with application to double-diffusive convection*.
> JCP **109**, 133. https://doi.org/10.1006/jcph.1993.1133

# Returns
- `Array{Float64}`: Vector or matrix depending on BC combination

# Notes
- Each supported BC combination is handled by a separate branch. Unsupported combinations
  throw a `DomainError`.

See also: [`CAtransform`](@ref), [`Chebyshev1D`](@ref)
"""
function calcGammaBC(cp::ChebyshevParameters)

    # Calculate a matrix to apply the Neumann and Dirichelet BCs
    # The nomenclature follows Ooyama (2002) to match the cubic b-spline designations
    Ndim = cp.zDim

    bcb = _bc_type(cp.BCB)
    bct = _bc_type(cp.BCT)

    if (bcb == :R0) && (bct == :R0)
        # No BCs
        gammaBC = zeros(Float64,Ndim)
        return gammaBC
    
    elseif (bcb == :R1T0) && (bct == :R0)
        #R1T0 bottom
        gammaBC = ones(Float64,Ndim)
        gammaBC[2:Ndim-1] *= 2.0
        gammaBC *= (-0.5 / (Ndim-1))
        return gammaBC
        
    elseif (bcb == :R1T1) && (bct == :R0)
        #R1T1 bottom
        # Global coefficient method (Wang et al. 1993) for Neumann BCs
        # https://doi.org/10.1006/jcph.1993.1133
        scaleFactor = 0.0
        gammaBC = zeros(Float64,Ndim,Ndim)
        c = ones(Float64,Ndim)
        c[1] *= 2.0
        c[Ndim] *= 2.0
        for i = 1:Ndim
            n = i-1
            scaleFactor += -(n * n) / (c[i] * (Ndim-1))
        end
        for i = 1:Ndim
            n = i -1
            for j = 1:Ndim
                gammaBC[i,j] = n * n / (scaleFactor * c[j] * (Ndim-1))
            end
        end
        return gammaBC
    
    elseif (bcb == :R0) && (bct == :R1T0)
        gammaBC = ones(Float64,Ndim,Ndim)
        for i = 1:Ndim
            for j = 1:Ndim
                gammaBC[i,j] *= -1.0* (-1.0)^(i-1) * (-1.0)^(j-1) / (Ndim-1)
            end
        end
        gammaBC[1,:] *= 0.5
        gammaBC[Ndim,:] *= 0.5
        return gammaBC
        
    elseif (bcb == :R0) && (bct == :R1T1)
        scaleFactor = 0.0
        gammaBC = zeros(Float64,Ndim,Ndim)
        c = ones(Float64,Ndim)
        c[1] *= 2.0
        c[Ndim] *= 2.0
        for i = 1:Ndim
            n = i-1
            scaleFactor += -(n * n) * (-1.0)^n * (-1.0)^(n+1) / (c[i] * (Ndim-1))
        end
        for i = 1:Ndim
            n = i -1
            for j = 1:Ndim
                gammaBC[i,j] = (-1.0)^(j+1) * (-1.0)^(n+1) * n * n / (scaleFactor * c[j] * (Ndim-1))
            end
        end
        return gammaBC
        
    elseif (bcb == :R1T0) && (bct == :R1T0)
        gammaBC = ones(Float64,Ndim,Ndim)
        for i = 1:Ndim
            for j = 1:Ndim
                gammaBC[i,j] *= -1.0* (-1.0)^(i-1) * (-1.0)^(j-1) / (Ndim-1)
            end
        end
        gammaBC[1,:] *= 0.5
        gammaBC[Ndim,:] *= 0.5

        gammaL = ones(Float64,Ndim)
        gammaL[2:Ndim-1] *= 2.0
        gammaL *= (-0.5 / (Ndim-1))

        for j = 1:Ndim
            gammaBC[:,j] += gammaL
        end
        return gammaBC  
        
    elseif (bcb == :R1T1) && (bct == :R1T1)
        scaleFactor = 0.0
        gammaBCB = zeros(Float64,Ndim,Ndim)
        c = ones(Float64,Ndim)
        c[1] *= 2.0
        c[Ndim] *= 2.0
        for i = 1:Ndim
            n = i-1
            scaleFactor += -(n * n) / (c[i] * (Ndim-1))
        end
        for i = 1:Ndim
            n = i -1
            for j = 1:Ndim
                gammaBCB[i,j] = n * n / (scaleFactor * c[j] * (Ndim-1))
            end
        end
        
        scaleFactor = 0.0
        gammaBCT = zeros(Float64,Ndim,Ndim)
        for i = 1:Ndim
            n = i-1
            scaleFactor += -(n * n) * (-1.0)^n * (-1.0)^(n+1) / (c[i] * (Ndim-1))
        end
        for i = 1:Ndim
            n = i -1
            for j = 1:Ndim
                gammaBCT[i,j] = (-1.0)^(j+1) * (-1.0)^(n+1) * n * n / (scaleFactor * c[j] * (Ndim-1))
            end
        end

        gammaBC = zeros(Float64,Ndim,Ndim)
        gammaBC .= gammaBCB + gammaBCT
        return gammaBC
    
    elseif (bcb == :R1T0) && (bct == :R1T1)
        c = ones(Float64,Ndim)
        c[1] *= 2.0
        c[Ndim] *= 2.0
        scaleFactor = 0.0
        gammaBCT = zeros(Float64,Ndim,Ndim)
        for i = 1:Ndim
            n = i-1
            scaleFactor += -(n * n) * (-1.0)^n * (-1.0)^(n+1) / (c[i] * (Ndim-1))
        end
        for i = 1:Ndim
            n = i -1
            for j = 1:Ndim
                gammaBCT[i,j] = (-1.0)^(j+1) * (-1.0)^(n+1) * n * n / (scaleFactor * c[j] * (Ndim-1))
            end
        end
        
        gammaBCB = ones(Float64,Ndim)
        gammaBCB[2:Ndim-1] *= 2.0
        gammaBCB *= (-0.5 / (Ndim-1))

        gammaBC = zeros(Float64,Ndim,Ndim)
        gammaBC .= gammaBCB .+ gammaBCT
        return gammaBC

    elseif (bcb == :R1T1) && (bct == :R1T0)
        c = ones(Float64,Ndim)
        c[1] *= 2.0
        c[Ndim] *= 2.0
        scaleFactor = 0.0
        gammaBCB = zeros(Float64,Ndim,Ndim)
        c = ones(Float64,Ndim)
        c[1] *= 2.0
        c[Ndim] *= 2.0
        for i = 1:Ndim
            n = i-1
            scaleFactor += -(n * n) / (c[i] * (Ndim-1))
        end
        for i = 1:Ndim
            n = i -1
            for j = 1:Ndim
                gammaBCB[i,j] = n * n / (scaleFactor * c[j] * (Ndim-1))
            end
        end

        gammaBCT = ones(Float64,Ndim,Ndim)
        for i = 1:Ndim
            for j = 1:Ndim
                gammaBCT[i,j] *= -1.0* (-1.0)^(i-1) * (-1.0)^(j-1) / (Ndim-1)
            end
        end
        gammaBCT[1,:] *= 0.5
        gammaBCT[Ndim,:] *= 0.5

        gammaBC = zeros(Float64,Ndim,Ndim)
        gammaBC .= gammaBCB .+ gammaBCT
        return gammaBC
    else
        bcs = "$(cp.BCB), $(cp.BCT)"
        throw(DomainError(bcs, "Chebyshev boundary condition combination not implemented"))
    end

end

"""
    bvp(N, u, ux, uxx, f, d0, d1, d2, scale, alpha, beta) -> Vector{Float64}

Modified Chebyshev boundary value problem solver (Boyd 2000).

Solves a general second-order BVP ``d_2 u'' + d_1 u' + d_0 u = f`` on ``[-1,1]`` with
Dirichlet boundary conditions `u(+1) = alpha`, `u(-1) = beta`. Uses the modified basis
functions from [`bvp_modified_basis`](@ref) to enforce homogeneous BCs automatically.
Primarily useful for testing the DCT matrix formulation against analytic solutions.

See also: [`bvp_modified_basis`](@ref), [`bvp_basis`](@ref)
"""
function bvp(N::Int64, u::Array{Float64}, ux::Array{Float64}, uxx::Array{Float64},
        f::Array{Float64}, d0::Array{Float64}, d1::Array{Float64}, d2::Array{Float64},
        scale::Float64 = 1.0, alpha::Float64 = 0.0, beta::Float64 = 0.0)

    # Modified Chebyshev boundary value problem solver from Boyd (2000)
    # Useful for testing the DCT matrices defined above
    nbasis = N-2
    xi = zeros(Float64, nbasis)
    g = zeros(Float64, nbasis)
    phi = zeros(Float64, nbasis)
    phix = zeros(Float64, nbasis)
    phixx = zeros(Float64, nbasis)
    h = zeros(Float64, nbasis, nbasis)
    # Compute the interior collocation points and forcing vector G
    for i in 1:nbasis
        xi[i] = cos(π*i/(nbasis+1))
        x = xi[i]
        b = alpha*(1-x)/2.0 + beta*(1+x)/2.0
        bx = (-alpha + beta)/2.0
        g[i] = f[i+1] - d0[i+1]*b - d1[i+1]*bx
    end

    # Compute the LHS square matrix
    for i in 1:nbasis
        x = xi[i]
        phi, phix, phixx = bvp_modified_basis(x, nbasis, phi, phix, phixx, scale)
        for j in 1:nbasis
            h[i,j] = d2[i+1]*phixx[j] + d1[i+1]*phix[j] + d0[i+1]*phi[j]
        end
    end

    # Solve the linear equation set
    aphi = h \ g

    # Transform back to physical space
    u[1] = beta
    u[N] = alpha
    ux[1] = (-alpha + beta)/2.0
    ux[N] = (-alpha + beta)/2.0
    uxx[1] = 0.0
    uxx[N] = 0.0
    x = 1.0
    phi, phix, phixx = bvp_modified_basis(x, nbasis, phi, phix, phixx, scale)
    for j in 1:nbasis
        u[1] = u[1] + (aphi[j] * phi[j])
        ux[1] = ux[1] + (aphi[j] * phix[j])
        uxx[1] = uxx[1] + (aphi[j] * phixx[j])
    end
    x = -1.0
    phi, phix, phixx = bvp_modified_basis(x, nbasis, phi, phix, phixx, scale)
    for j in 1:nbasis
        u[N] = u[N] + (aphi[j] * phi[j])
        ux[N] = ux[N] + (aphi[j] * phix[j])
        uxx[N] = uxx[N] + (aphi[j] * phixx[j])
    end
    for i in 1:nbasis
        x = xi[i]
        phi, phix, phixx = bvp_modified_basis(x, nbasis, phi, phix, phixx, scale)
        u[i+1] = alpha*(1-x)/2.0 + beta*(1+x)/2.0
        ux[i+1] = (-alpha + beta)/2.0
        uxx[i+1] = 0.0
        for j in 1:nbasis
            u[i+1] = u[i+1] + (aphi[j] * phi[j])
            ux[i+1] = ux[i+1] + (aphi[j] * phix[j])
            uxx[i+1] = uxx[i+1] + (aphi[j] * phixx[j])
        end
    end

    return u
end

"""
    bvp_modified_basis(x, nbasis, phi, phix, phixx, scale) -> (phi, phix, phixx)

Evaluate the modified Chebyshev basis functions and their derivatives at point `x`.

Modified basis functions satisfy homogeneous Dirichlet BCs at `±1`. Handles the
singularity at endpoints by using the limiting formulae. Returns in-place the arrays
`phi` (values), `phix` (first derivatives), `phixx` (second derivatives).

See also: [`bvp`](@ref)
"""
function bvp_modified_basis(x::Float64, nbasis::Int64, phi::Array{Float64}, phix::Array{Float64}, phixx::Array{Float64}, scale::Float64)

    if abs(x) < 1.0 # Avoid singularities at the endpoints
        t = acos(x)
        c = cos(t)
        s = sin(t)
        for i in 1:nbasis
            n = i+1
            tn = cos(n*t)
            tnt = -n * sin(n*t)
            tntt = -n * n * tn

            # Convert t-derivatives into x-derivatives
            tnx = -tnt/s
            tnxx = tntt/(s*s) - tnt*c/(s*s*s)

            # Convert to modified basis functions to enforce BCs
            if mod(n,2) == 0
                phi[i] = tn - 1.0
                phix[i] = tnx / scale
            else
                phi[i] = tn - x
                phix[i] = (tnx - 1.0) / scale
            end
            phixx[i] = tnxx / (scale * scale)
        end
    else
        for i in 1:nbasis
            phi[i] = 0.0
            n = i+1
            if mod(n,2) == 0
                phix[i] = sign(x)*n*n / scale
            else
                phix[i] = (n*n - 1) / scale
            end
            phixx[i] = sign(x)^n * n * n * ((n * n - 1.0)/3.0) / (scale * scale)
        end
    end

    return phi, phix, phixx
end

"""
    bvp_basis(x, nbasis, phi, phix, phixx) -> (phi, phix, phixx)

Evaluate the standard (unmodified) Chebyshev basis functions and their derivatives at `x`.

Used for debugging and verification; does not enforce BCs. See [`bvp_modified_basis`](@ref)
for the BC-enforcing variant.
"""
function bvp_basis(x::Float64, nbasis::Int64, phi::Array{Float64}, phix::Array{Float64}, phixx::Array{Float64})

    if abs(x) < 1.0 # Avoid singularities at the endpoints
        t = acos(x)
        c = cos(t)
        s = sin(t)
        for i in 1:nbasis
            n = i-1
            tn = cos(n*t)
            tnt = -n * sin(n*t)
            tntt = -n * n * tn

            # Convert t-derivatives into x-derivatives
            tnx = -tnt/s
            tnxx = tntt/(s*s) - tnt*c/(s*s*s)

            # Convert to modified basis functions to enforce BCs
            if mod(n,2) == 0
                phi[i] = tn #- 1.0
                phix[i] = tnx
            else
                phi[i] = tn #- x
                phix[i] = tnx #- 1.0
            end
            phixx[i] = tnxx
        end
    else
        t = acos(x)
        for i in 1:nbasis
            n = i-1
            phi[i] = cos(n*t)
            if x > 0.0
                phix[i] = -n*n
            else
                phix[i] = -n*n*(-1.0)^(n+1)
            end
            phixx[i] = ((-1.0)^n)*(n^4 - n^2)/3
        end
    end

    return phi, phix, phixx
end

# ---------------------------------------------------------------------------
# Generic wrappers (no "C" prefix) dispatching on Chebyshev1D
# These enable abstract 1D basis code that calls Btransform!, Itransform!, etc.
# without needing to know the underlying basis type.
# ---------------------------------------------------------------------------

"""Generic B-transform wrapper for `Chebyshev1D` (allocating). Delegates to [`CBtransform`](@ref)."""
Btransform(column::Chebyshev1D, uMish::Vector{real}) = CBtransform(column, uMish)

"""Generic in-place B-transform wrapper for `Chebyshev1D`. Delegates to `CBtransform!`."""
Btransform!(column::Chebyshev1D) = CBtransform!(column)

"""Generic A-transform wrapper for `Chebyshev1D` (allocating). Delegates to [`CAtransform`](@ref)."""
Atransform(column::Chebyshev1D, b::AbstractVector) = CAtransform(column, b)

"""Generic in-place A-transform wrapper for `Chebyshev1D`. Delegates to `CAtransform!`."""
Atransform!(column::Chebyshev1D) = CAtransform!(column)

"""Generic in-place I-transform wrapper for `Chebyshev1D`. Delegates to `CItransform!`."""
Itransform!(column::Chebyshev1D) = CItransform!(column)

"""Generic I-transform wrapper for `Chebyshev1D` (in-place output). Performs inverse DCT from `column.a` and writes into `u`."""
function Itransform(column::Chebyshev1D, u::AbstractVector)
    u .= column.fftPlan * column.a
    return u
end

"""Generic Ix-transform wrapper for `Chebyshev1D` (allocating). Delegates to [`CIxtransform`](@ref)."""
Ixtransform(column::Chebyshev1D) = CIxtransform(column)

"""Generic Ix-transform wrapper for `Chebyshev1D` (in-place output). Delegates to [`CIxtransform`](@ref)."""
Ixtransform(column::Chebyshev1D, ux::AbstractVector) = CIxtransform(column, ux)

"""Generic Ixx-transform wrapper for `Chebyshev1D` (allocating). Delegates to [`CIxxtransform`](@ref)."""
Ixxtransform(column::Chebyshev1D) = CIxxtransform(column)

"""Generic indefinite-integral transform wrapper for `Chebyshev1D`. Delegates to [`CIInttransform`](@ref)."""
IInttransform(column::Chebyshev1D, C0::real = 0.0) = CIInttransform(column, C0)

"""Generic I-transform matrix wrapper for `Chebyshev1D`. Delegates to [`CItransform_matrix`](@ref)."""
Itransform_matrix(column::Chebyshev1D, points::Vector{Float64}, derivative::Int=0) =
    CItransform_matrix(column, points, derivative)

#Module end
end #module

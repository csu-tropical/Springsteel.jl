module CubicBSpline

#= This module holds the functions for manipulating cubic B-splines.
The math and terminology closely follow Ooyama, K. V., 2002: The cubic-spline transform method: Basic definitions and tests in a 1d single domain. Mon. Wea. Rev., 130, 2392–2415. =#

using LinearAlgebra
using SparseArrays
using SuiteSparse

export SplineParameters, Spline1D
#export R0, R1T0, R1T1, R1T2, R2T10, R2T20, R3, PERIODIC
export SBtransform, SBtransform!, SAtransform!, SItransform!
export SAtransform, SBxtransform, SItransform, SIxtransform, SIxxtransform
export setMishValues

#Define some convenient aliases
const real = Float64
const int = Int64
const uint = UInt64
const ONESIXTH = 1.0/6.0
const FOURSIXTH = 4.0/6.0
const sqrt35 = sqrt(3.0/5.0)

# Homogeneous boundary conditions.
# Inhomogeneous conditions are not yet implemented.

"""Pinned boundary condition: value is zero at the boundary (`u = 0`)."""
const R0 = Dict("R0" => 0)

"""Robin BC type 0: `α₁·u + β₁·u' = 0` with α₁ = -4, β₁ = -1 (free-slip near wall)."""
const R1T0 = Dict("α1" => -4.0, "β1" => -1.0)

"""Robin BC type 1: zero first derivative at boundary (`u' = 0`, Neumann condition)."""
const R1T1 = Dict("α1" =>  0.0, "β1" =>  1.0)

"""Robin BC type 2: `α₁·u + β₁·u' = 0` with α₁ = 2, β₁ = -1."""
const R1T2 = Dict("α1" =>  2.0, "β1" => -1.0)

"""Rank-2 Robin BC type 1–0: `α₂·u + β₂·u'' = 0` with α₂ = 1, β₂ = -0.5."""
const R2T10 = Dict("α2" => 1.0, "β2" => -0.5)

"""Rank-2 Robin BC type 2–0: `α₂·u + β₂·u'' = 0` with α₂ = -1, β₂ = 0."""
const R2T20 = Dict("α2" => -1.0, "β2" => 0.0)

"""Rank-3 boundary condition: eliminates three edge coefficients (not commonly used)."""
const R3 = Dict("R3" => 0)

"""Periodic boundary condition: identifies left and right boundaries."""
const PERIODIC = Dict("PERIODIC" => 0)

# Fix the mish to 3 points
const mubar = 3
const gaussweight = [5.0/18.0, 8.0/18.0, 5.0/18.0]

"""
    SplineParameters

Immutable parameter struct (using `@kwdef`) for a 1D cubic B-spline basis.

# Fields
- `xmin::Float64`: Left boundary of the domain
- `xmax::Float64`: Right boundary of the domain
- `num_cells::Int64`: Number of spline cells; total physical gridpoints = `num_cells * mubar`
- `l_q::Float64`: Filter length scale (default `2.0`). Larger values produce smoother spectral fits
- `BCL::Dict`: Left boundary condition (one of [`R0`](@ref), [`R1T0`](@ref), [`R1T1`](@ref), [`R1T2`](@ref), [`R2T10`](@ref), [`R2T20`](@ref), [`R3`](@ref), [`PERIODIC`](@ref))
- `BCR::Dict`: Right boundary condition (same options as `BCL`)
- `DX::Float64`: Cell width, computed as `(xmax - xmin) / num_cells`
- `DXrecip::Float64`: Reciprocal of `DX`, precomputed for efficiency

# Example
```julia
sp = CubicBSpline.SplineParameters(
    xmin = 0.0,
    xmax = 100.0,
    num_cells = 20,
    l_q = 2.0,
    BCL = CubicBSpline.R0,
    BCR = CubicBSpline.R0
)
```

See also: [`Spline1D`](@ref)
"""
Base.@kwdef struct SplineParameters
    xmin::real = 0.0
    xmax::real = 0.0
    num_cells::int = 1
    l_q::real = 2.0
    BCL::Dict = R0
    BCR::Dict = R0
    DX::real = (xmax - xmin) / num_cells
    DXrecip::real = 1.0/DX
end

"""
    Spline1D

One-dimensional cubic B-spline object.  Construct via `Spline1D(sp::SplineParameters)`.

# Fields
- `params::SplineParameters`: Configuration (domain, cells, BCs, filter length)
- `gammaBC::Matrix{Float64}`: Boundary-condition projection matrix (maps interior to full coefficient space)
- `pq`: Full `(P + Q)` matrix used in the least-squares / variational solve
- `pqFactor::SuiteSparse.CHOLMOD.Factor{Float64}`: Sparse Cholesky factorisation of the open-form `(P + Q)` matrix for fast solves
- `mishDim::Int64`: Number of Gaussian quadrature (mish) points = `num_cells * mubar`
- `bDim::Int64`: Number of spectral coefficients = `num_cells + 3`
- `mishPoints::Vector{Float64}`: Physical locations of the mish points
- `uMish::Vector{Float64}`: Physical field values at mish points (mutable working buffer)
- `b::Vector{Float64}`: B-vector (result of SB transform, inner products ⟨φₘ, u⟩)
- `a::Vector{Float64}`: Spectral coefficient vector (result of SA transform)

# Notes
- Constructing `Spline1D` builds `gammaBC` and factorises the `(P + Q)` matrix, which is the
  computationally expensive step.  Reuse spline objects when possible.
- The `mubar = 3` Gauss–Legendre points per cell are fixed (Ooyama 2002).

See also: [`SplineParameters`](@ref), [`SBtransform`](@ref), [`SAtransform`](@ref), [`SItransform`](@ref)
"""
struct Spline1D
    params::SplineParameters
    gammaBC::Matrix{real}
    pq
    pqFactor::SuiteSparse.CHOLMOD.Factor{Float64}
    mishDim::int
    bDim::int
    mishPoints::Vector{real}
    uMish::Vector{real}
    b::Vector{real}
    a::Vector{real}
end

"""
    basis(sp::SplineParameters, m::Int64, x::Float64, derivative::Int64) -> Float64

Evaluate the cubic B-spline basis function φₘ(x) or one of its derivatives.

Follows the Ooyama (2002) compact-support cubic B-spline formulation. Each basis function
is nonzero only within two cell widths of node `m`.

# Arguments
- `sp::SplineParameters`: Spline configuration (domain and cell width)
- `m::Int64`: Node index.  Node positions are `xmin + m*DX` for `m = -1, 0, ..., num_cells+1`
- `x::Float64`: Physical evaluation point; must lie within `[sp.xmin, sp.xmax]`
- `derivative::Int64`: Order of derivative to evaluate:
  - `0` — function value
  - `1` — first derivative ∂φ/∂x
  - `2` — second derivative ∂²φ/∂x²
  - `3` — third derivative (piecewise constant, used for the Q-matrix smoothing term)

# Returns
- `Float64`: Value of φₘ(x) or its requested derivative at `x`

# Throws
- `DomainError` if `x` lies outside `[sp.xmin, sp.xmax]`
"""
function basis(sp::SplineParameters, m::int, x::real, derivative::int)
    
    b = 0.0
    if (x < sp.xmin) || (x > sp.xmax)
        throw(DomainError(x, "x outside spline domain"))
    end
    xm = sp.xmin + (m * sp.DX)
    delta = (x - xm) * sp.DXrecip
    z = abs(delta)
    if (z < 2.0)
        if (derivative == 0)
            z = 2.0 - z
            b = (z*z*z) * ONESIXTH
            z -= 1.0
            if (z > 0)
                b -= (z*z*z) * FOURSIXTH
           end
        elseif (derivative == 1)
            z = 2.0 - z
            b = (z*z) * ONESIXTH
            z -= 1.0
            if (z > 0)
                b -= (z*z) * FOURSIXTH
            end
            b *= ((delta > 0) ? -1.0 : 1.0) * 3.0 * sp.DXrecip
        elseif (derivative == 2)
            z = 2.0 - z
            b = z
            z -= 1.0
            if (z > 0)
                b -= z * 4
            end
            b *= sp.DXrecip * sp.DXrecip
        elseif (derivative == 3)
            if (z > 1.0)
                b = 1.0
            elseif (z < 1.0)
                b = -3.0
            end
            b *= ((delta > 0) ? -1.0 : 1.0) * sp.DXrecip * sp.DXrecip * sp.DXrecip
        end
    end
    return b
end

"""
    calcGammaBC(sp::SplineParameters) -> Matrix{Float64}

Build the boundary-condition projection matrix Γ.

The matrix maps the interior (free) spline coefficients to the full coefficient vector
of length `num_cells + 3`, encoding the homogeneous boundary conditions specified by
`sp.BCL` and `sp.BCR`.  It is used inside [`SAtransform`](@ref) to fold and unfold
boundary conditions during the spectral solve.

# Arguments
- `sp::SplineParameters`: Spline parameters including `BCL`, `BCR`, and `num_cells`

# Returns
- `Matrix{Float64}`: Projection matrix of size `(Minterior, num_cells+3)` where
  `Minterior = num_cells + 3 - rankL - rankR` and `rankL`/`rankR` are the number of
  constraints imposed by the left/right BCs.
"""
function calcGammaBC(sp::SplineParameters)
    if haskey(sp.BCL,"α1")
        rankL = 1
    elseif haskey(sp.BCL,"α2")
        rankL = 2
    elseif sp.BCL == R0
        rankL = 0
    elseif sp.BCL == R3
        rankL = 3
    elseif sp.BCL == PERIODIC
        rankL = 1
    end

    if haskey(sp.BCR,"α1")
        rankR = 1
    elseif haskey(sp.BCR,"α2")
        rankR = 2
    elseif sp.BCR == R0
        rankR = 0
    elseif sp.BCR == R3
        rankR = 3
    elseif sp.BCR == PERIODIC
        rankR = 2
    end

    Mdim = sp.num_cells + 3
    Minterior_dim = Mdim - rankL - rankR

    # Create the BC matrix
    gammaBC = zeros(real, Minterior_dim, Mdim)
    if haskey(sp.BCL,"α1")
        gammaBC[1,1] = sp.BCL["α1"]
        gammaBC[2,1] = sp.BCL["β1"]
        gammaBC[:,2:(Mdim-rankR)] = Matrix(1.0I, Minterior_dim, Minterior_dim)
    elseif haskey(sp.BCL,"α2")
        gammaBC[1,1] = sp.BCL["α2"]
        gammaBC[1,2] = sp.BCL["β2"]
        gammaBC[:,3:(Mdim-rankR)] = Matrix(1.0I, Minterior_dim, Minterior_dim)
    elseif sp.BCL == PERIODIC
        gammaBC[Minterior_dim,1] = 1.0
        gammaBC[:,2:(Mdim-rankR)] = Matrix(1.0I, Minterior_dim, Minterior_dim)
    elseif sp.BCL == R3
        gammaBC[:,4:(Mdim-rankR)] = Matrix(1.0I, Minterior_dim, Minterior_dim)
    else
        gammaBC[:,1:(Mdim-rankR)] = Matrix(1.0I, Minterior_dim, Minterior_dim)
    end

    if haskey(sp.BCR,"α1")
        gammaBC[Minterior_dim,Mdim] = sp.BCR["α1"]
        gammaBC[Minterior_dim-1,Mdim] = sp.BCR["β1"]
    elseif haskey(sp.BCR,"α2")
        gammaBC[Minterior_dim,Mdim] = sp.BCR["α2"]
        gammaBC[Minterior_dim,Mdim-1] = sp.BCR["β2"]
    elseif sp.BCR == PERIODIC
        gammaBC[1,Mdim-1] = 1.0
        gammaBC[2,Mdim] = 1.0
    end
    return gammaBC
end

"""
    calcPQfactor(sp::SplineParameters, gammaBC::Matrix{Float64}) -> (Matrix{Float64}, SuiteSparse.CHOLMOD.Factor)

Build the variational `(P + Q)` matrix and return its sparse Cholesky factorisation.

Following Ooyama (2002), the spectral fit minimises a penalised least-squares functional:

``J[a] = \\langle u - \\hat{u},\\, u - \\hat{u} \\rangle_P + \\varepsilon_q \\langle a''', a''' \\rangle_Q``

where P contains mass-matrix inner products of basis functions and Q contains inner products
of their third derivatives.  The smoothing weight is:

``\\varepsilon_q = \\left(\\frac{l_q \\cdot \\Delta x}{2\\pi}\\right)^6``

The open-form `(P + Q)` (with BCs folded in via `gammaBC`) is factorised with a sparse
Cholesky decomposition for repeated fast linear solves during [`SAtransform`](@ref).

# Arguments
- `sp::SplineParameters`: Spline parameters (domain, cells, filter length `l_q`)
- `gammaBC::Matrix{Float64}`: Boundary-condition projection matrix from [`calcGammaBC`](@ref)

# Returns
- `(pq::Matrix{Float64}, pqFactor)`: Full `(P + Q)` matrix and its Cholesky factor
"""
function calcPQfactor(sp::SplineParameters, gammaBC::Matrix{real})
    eps_q = ((sp.l_q*sp.DX)/(2*π))^6
    Mdim = sp.num_cells + 3

    # Create the P and Q matrices
    P = zeros(real, Mdim, Mdim)
    Q = zeros(real, Mdim, Mdim)

    for mi1 = 1:Mdim
        for mi2 = 1:Mdim
            if abs(mi1 - mi2) > 3
                continue
            end
            m1 = mi1 - 2
            m2 = mi2 - 2
            for mc = 0:(sp.num_cells-1)
                if (mc < (m1 - 2)) || (mc > (m1 + 1))
                    continue
                end
                if (mc < (m2 - 2)) || (mc > (m2 + 1))
                    continue
                end
                for mu = 1:mubar
                    i = mu + (mubar * mc)
                    x = sp.xmin + (mc * sp.DX) + sp.DX * ((mu/2.0 - 1.0) * sqrt35) + sp.DX * 0.5
                    pm1 = basis(sp, m1, x, 0)
                    qm1 = basis(sp, m1, x, 3)
                    pm2 = basis(sp, m2, x, 0)
                    qm2 = basis(sp, m2, x, 3)
                    P[mi1,mi2] += sp.DX * gaussweight[mu] * pm1 * pm2
                    Q[mi1,mi2] += sp.DX * gaussweight[mu] * eps_q * qm1 * qm2
                end
            end
        end
    end

    # Fold in the BCs to get open form
    PQ = Symmetric(P + Q)
    PQopen = Symmetric((gammaBC * P * gammaBC') + (gammaBC * Q * gammaBC'))
    PQsparse = sparse(PQopen)
    PQfactor = (cholesky(PQsparse))
    return PQ, PQfactor
end

"""
    calcMishPoints(sp::SplineParameters) -> Vector{Float64}

Compute the Gaussian quadrature ("mish") point locations.

For each of the `num_cells` cells, three Gauss–Legendre quadrature points are placed
at positions determined by the `sqrt(3/5)` abscissae (5-point rule mapped to each cell).
These are the physical-space locations where field values must be provided for the
spectral transform.

# Arguments
- `sp::SplineParameters`: Spline parameters (domain, cell width)

# Returns
- `Vector{Float64}`: Sorted vector of `num_cells * mubar` (= `num_cells * 3`) mish point
  coordinates in `[xmin, xmax]`

See also: [`Spline1D`](@ref), [`SBtransform`](@ref)
"""
function calcMishPoints(sp::SplineParameters)
    x = zeros(real,sp.num_cells*mubar)
    for mc = 0:(sp.num_cells-1)
        for mu = 1:mubar
            i = mu + (mubar * mc)
            x[i] = sp.xmin + (mc * sp.DX) + sp.DX * ((mu/2.0 - 1.0) * sqrt35) + sp.DX * 0.5
        end
    end
    return x
end

"""
    Spline1D(sp::SplineParameters) -> Spline1D

Construct a [`Spline1D`](@ref) object from spline parameters.

This constructor pre-computes and caches all objects needed for repeated spectral
transforms:
1. The boundary-condition projection matrix `gammaBC` via [`calcGammaBC`](@ref)
2. The `(P + Q)` matrix and its sparse Cholesky factorisation via [`calcPQfactor`](@ref)
3. The mish point locations via [`calcMishPoints`](@ref)
4. Zero-initialised working buffers `uMish`, `b`, and `a`

Reuse the constructed `Spline1D` object for multiple transforms to amortise the
Choleksy factorisation cost.

# Arguments
- `sp::SplineParameters`: Spline configuration

# Returns
- `Spline1D`: Initialised spline object ready for transform calls

# Example
```julia
sp = CubicBSpline.SplineParameters(
    xmin = 0.0, xmax = 1.0, num_cells = 10,
    BCL = CubicBSpline.R0, BCR = CubicBSpline.R0
)
spline = CubicBSpline.Spline1D(sp)
```

See also: [`SBtransform`](@ref), [`SAtransform`](@ref), [`SItransform`](@ref)
"""
function Spline1D(sp::SplineParameters)
    gammaBC = calcGammaBC(sp)
    pq, pqFactor = calcPQfactor(sp, gammaBC)

    mishDim = sp.num_cells*mubar
    mishPoints = calcMishPoints(sp)
    uMish = zeros(real,sp.num_cells*mubar)

    bDim = sp.num_cells + 3
    b = zeros(real,bDim)
    a = zeros(real,bDim)

    spline = Spline1D(sp,gammaBC,pq,pqFactor,mishDim,bDim,mishPoints,uMish,b,a)
    return spline
end

"""
    setMishValues(spline::Spline1D, uMish::Vector{Float64})

Copy physical field values into the spline's internal mish buffer.

Convenience wrapper that performs `spline.uMish .= uMish`.  Use before calling
`SBtransform!(spline)` when you prefer the in-place workflow.

# Arguments
- `spline::Spline1D`: Target spline object
- `uMish::Vector{Float64}`: Field values at the mish points (length `num_cells * 3`)

See also: [`SBtransform`](@ref)
"""
function setMishValues(spline::Spline1D, uMish::Vector{real})
    spline.uMish .= uMish
end

"""
    SBtransform(sp::SplineParameters, uMish::Vector{Float64}) -> Vector{Float64}
    SBtransform(spline::Spline1D,      uMish::Vector{Float64}) -> Vector{Float64}
    SBtransform!(spline::Spline1D)

Compute the **SB transform**: project physical field values at the mish points onto the
cubic B-spline basis to obtain the B-vector ``b_m = \\langle \\varphi_m,\\, u \\rangle``.

``b_m = \\sum_{\\text{cells}} \\Delta x \\sum_{\\mu} w_\\mu \\,\\varphi_m(x_\\mu)\\, u(x_\\mu)``

This is the first step of the two-step physical→spectral transform.  The full transform
continues with [`SAtransform`](@ref).

# Arguments  
- `sp` / `spline`: Spline parameters or a constructed `Spline1D`
- `uMish::Vector{Float64}`: Field values at the `num_cells * 3` mish points

# Returns
- `Vector{Float64}`: B-vector of length `num_cells + 3` (allocating versions)
- The in-place `SBtransform!` stores the result in `spline.b`

# Notes
- The boundary folds (gammaBC) are **not** applied here; they are applied inside
  [`SAtransform`](@ref).

See also: [`SAtransform`](@ref), [`SBxtransform`](@ref)
"""
function SBtransform(sp::SplineParameters, uMish::Vector{real})
    Mdim = sp.num_cells + 3
    b = zeros(real,Mdim)

    for mi = 1:Mdim
        m = mi - 2
        for mc = 0:(sp.num_cells-1)
            if (mc < (m - 2)) || (mc > (m + 1))
                continue
            end
            for mu = 1:mubar
                i = mu + (mubar * mc)
                x = sp.xmin + (mc * sp.DX) + sp.DX * ((mu/2.0 - 1.0) * sqrt35) + sp.DX * 0.5
                bm = basis(sp, m, x, 0)
                b[mi] += sp.DX * gaussweight[mu] * bm * uMish[i]
            end
        end
    end

    # Don't border fold it here, only in SA
    return b
end

function SBtransform(spline::Spline1D, uMish::Vector{real})

    b = SBtransform(spline.params,uMish)
    return b
end

function SBtransform!(spline::Spline1D)

    b = SBtransform(spline.params,spline.uMish)
    spline.b .= b
end

"""
    SBxtransform(sp::SplineParameters, uMish::Vector{Float64}, BCL::Float64, BCR::Float64) -> Vector{Float64}
    SBxtransform(spline::Spline1D,      uMish::Vector{Float64}, BCL::Float64, BCR::Float64) -> Vector{Float64}

Compute the **SBx transform**: B-vector of the derivative ``f'`` via integration by parts.

``b_m = \\bigl[\\varphi_m \\cdot f\\bigr]_{x_0}^{x_0'} - \\int \\varphi'_m(x)\\, f(x)\\, dx``

# Arguments
- `sp` / `spline`: Spline parameters or a constructed `Spline1D`
- `uMish::Vector{Float64}`: Field values at the `num_cells * 3` mish points
- `BCL::Float64`: Value of `f` at the left boundary `xmin` (not a boundary condition constant)
- `BCR::Float64`: Value of `f` at the right boundary `xmax`

# Returns
- `Vector{Float64}`: B-vector for `f'`, of length `num_cells + 3`

See also: [`SBtransform`](@ref), [`SAtransform`](@ref)
"""
function SBxtransform(sp::SplineParameters, uMish::Vector{real}, BCL::real, BCR::real)
    Mdim = sp.num_cells + 3
    b = zeros(real, Mdim)

    for mi = 1:Mdim
        m = mi - 2
        for mc = 0:(sp.num_cells-1)
            if (mc < (m - 2)) || (mc > (m + 1))
                continue
            end
            for mu = 1:mubar
                i = mu + (mubar * mc)
                x = sp.xmin + (mc * sp.DX) + sp.DX * ((mu/2.0 - 1.0) * sqrt35) + sp.DX * 0.5
                bm = basis(sp, m, x, 1)
                b[mi] += sp.DX * gaussweight[mu] * bm * uMish[i]
            end
        end
        bl = basis(sp, m, sp.xmin, 0) * BCL
        br = basis(sp, m, sp.xmax, 0) * BCR
        b[mi] = br - bl - b[mi]
    end

    return b
end

function SBxtransform(spline::Spline1D, uMish::Vector{real}, BCL::real, BCR::real)

    bx = SBxtransform(spline.params, uMish, BCL, BCR)
    return bx
end

"""
    SAtransform(sp::SplineParameters, gammaBC::Matrix{Float64}, pqFactor, b::Vector{Float64}) -> Vector{Float64}
    SAtransform(spline::Spline1D, b::AbstractVector) -> Vector{Float64}
    SAtransform!(spline::Spline1D)
    SAtransform(spline::Spline1D, b::Vector{Float64}, ahat::Vector{Float64}) -> Vector{Float64}

Compute the **SA transform**: convert a B-vector to spectral A-coefficients by solving
the variational system with boundary conditions.

``a = \\Gamma^T \\bigl[(\\Gamma (P+Q) \\Gamma^T)^{-1}\\, \\Gamma\\, b\\bigr]``

where ``\\Gamma`` is the boundary-condition projection matrix and `(P+Q)` is the
pre-computed (and pre-factored) variational matrix.

# Variants
- `SAtransform(sp, gammaBC, pqFactor, b)` — lowest-level call with all components explicit
- `SAtransform(spline, b)` — allocates and returns `a` using the cached spline internals
- `SAtransform!(spline)` — in-place; reads `spline.b`, writes `spline.a`
- `SAtransform(spline, b, ahat)` — inhomogeneous/incremental form; `ahat` is a background
  coefficient vector that carries inhomogeneous boundary values

# Returns
- `Vector{Float64}`: Spectral coefficient vector `a` of length `num_cells + 3`
  (allocating variants); the in-place `SAtransform!` stores in `spline.a`

See also: [`SBtransform`](@ref), [`SItransform`](@ref)
"""
function SAtransform(sp::SplineParameters, gammaBC::Matrix{Float64}, pqFactor, b::Vector{real})
    a = gammaBC' * (pqFactor \ (gammaBC * b))
    return a
end

function SAtransform(spline::Spline1D, b::AbstractVector)

    a = spline.gammaBC' * (spline.pqFactor \ (spline.gammaBC * b))
    return a
end

function SAtransform!(spline::Spline1D)

    # In-place version of the SA transform
    spline.a .= spline.gammaBC' * (spline.pqFactor \ (spline.gammaBC * spline.b))
end

function SAtransform(spline::Spline1D, b::Vector{real}, ahat::Vector{real})

    btilde = spline.gammaBC * (b - (spline.pq * ahat))
    a = (spline.gammaBC' * (spline.pqFactor \ btilde)) + ahat
    return a
end

"""
    SItransform(sp::SplineParameters, a::Vector{Float64}, x::Float64, derivative::Int64 = 0) -> Float64
    SItransform(sp::SplineParameters, a::Vector{Float64}, derivative::Int64 = 0) -> Vector{Float64}
    SItransform(sp::SplineParameters, a::Vector{Float64}, points::Vector{Float64}, derivative::Int64 = 0) -> Vector{Float64}
    SItransform(sp::SplineParameters, a::Vector{Float64}, points::Vector{Float64}, u::AbstractVector, derivative::Int64 = 0) -> Vector{Float64}
    SItransform!(spline::Spline1D)
    SItransform(spline::Spline1D, u::AbstractVector) -> Vector{Float64}
    SItransform(spline::Spline1D, points::Vector{Float64}, u::AbstractVector) -> Vector{Float64}

Compute the **SI transform**: evaluate the B-spline expansion at physical locations.

``u(x) = \\sum_m a_m\\, \\varphi_m(x)``

This is the spectral→physical step and is the inverse of the SB+SA pipeline.

# Variants
- Single point: `SItransform(sp, a, x, derivative)` — returns one `Float64`
- All mish points (allocating): `SItransform(sp, a, derivative)`
- Custom points (allocating): `SItransform(sp, a, points, derivative)`
- Custom points (in-place): `SItransform(sp, a, points, u, derivative)` — writes into `u`
- In-place at mish points: `SItransform!(spline)` — writes into `spline.uMish`
- Convenience at mish points: `SItransform(spline, u)`
- Convenience at custom points: `SItransform(spline, points, u)`

# Arguments
- `sp` / `spline`: Spline parameters or a `Spline1D` object
- `a::Vector{Float64}`: Spectral coefficient vector of length `num_cells + 3`
- `x` / `points`: Evaluation location(s); must lie within `[xmin, xmax]`
- `derivative::Int64`: Derivative order (default `0`; supports `0`, `1`, `2`)
- `u`: Pre-allocated output vector (written in-place where applicable)

# Returns
- Evaluated field values (and/or derivatives) at the requested points

See also: [`SAtransform`](@ref), [`SIxtransform`](@ref), [`SIxxtransform`](@ref)
"""
function SItransform(sp::SplineParameters, a::Vector{real}, x::real, derivative::int = 0)
    u = 0.0
    xm = ceil(int,(x - sp.xmin - (2.0 * sp.DX)) * sp.DXrecip)
    for m = xm:(xm + 3)
        if (m >= -1) && (m <= (sp.num_cells+1))
            mi = m + 2
            u += basis(sp, m, x, derivative) * a[mi]
        end
    end
    return u
end

function SItransform(sp::SplineParameters, a::Vector{real}, derivative::int = 0)

    u = zeros(real,sp.num_cells*mubar)
    for mc = 0:(sp.num_cells-1)
        for mu = 1:mubar
            i = mu + (mubar * mc)
            x = sp.xmin + (mc * sp.DX) + sp.DX * ((mu/2.0 - 1.0) * sqrt35) + sp.DX * 0.5
            for m = (mc-1):(mc+2)
                if (m >= -1) && (m <= (sp.num_cells+1))
                    mi = m + 2
                    u[i] += basis(sp, m, x, derivative) * a[mi]
                end
            end
        end
    end
    return u
end

function SItransform(sp::SplineParameters, a::Vector{real}, points::Vector{real}, derivative::int = 0)

    u = zeros(real,length(points))
    for i in eachindex(points)
        xm = ceil(int,(points[i] - sp.xmin - (2.0 * sp.DX)) * sp.DXrecip)
        for m = xm:(xm + 3)
            if (m >= -1) && (m <= (sp.num_cells+1))
                mi = m + 2
                u[i] += basis(sp, m, points[i], derivative) * a[mi]
            end
        end
    end
    return u
end

function SItransform(sp::SplineParameters, a::Vector{real}, points::Vector{real}, u::AbstractVector, derivative::int = 0)

    for i in eachindex(points)
        u[i] = 0.0
        xm = ceil(int,(points[i] - sp.xmin - (2.0 * sp.DX)) * sp.DXrecip)
        for m = xm:(xm + 3)
            if (m >= -1) && (m <= (sp.num_cells+1))
                mi = m + 2
                u[i] += basis(sp, m, points[i], derivative) * a[mi]
            end
        end
    end
    return u
end

function SItransform!(spline::Spline1D)

    # In-place SI transform
    u = SItransform(spline.params,spline.a,spline.mishPoints,spline.uMish)
    return u
end

function SItransform(spline::Spline1D, u::AbstractVector)

    u = SItransform(spline.params,spline.a,spline.mishPoints,u)
    return u
end

function SItransform(spline::Spline1D, points::Vector{real}, u::AbstractVector)

    u = SItransform(spline.params,spline.a,points,u)
    return u
end

"""
    SItransform_matrix(spline::Spline1D, points::Vector{Float64}, derivative::Int64 = 0) -> Matrix{Float64}

Build the SI transform as an explicit evaluation matrix (mainly for debugging and
linear-system solving).

Returns a matrix `M` of size `(num_cells*mubar, bDim)` such that `M * a ≈ u` at the
given `points`.  Each row corresponds to one point and each column to one spline basis
function.

# Arguments
- `spline::Spline1D`: Spline object (provides parameters and `bDim`)
- `points::Vector{Float64}`: Physical evaluation locations
- `derivative::Int64`: Derivative order (default `0`)

# Returns
- `Matrix{Float64}` of size `(length(points), bDim)`

See also: [`SItransform`](@ref)
"""
function SItransform_matrix(spline::Spline1D, points::Vector{Float64}, derivative::Int64 = 0)
    sp = spline.params
    u = zeros(Float64,sp.num_cells*mubar,spline.bDim)
    for i in eachindex(points)
        xm = ceil(Int64,(points[i] - sp.xmin - (2.0 * sp.DX)) * sp.DXrecip)
        for m = xm:(xm + 3)
            if (m >= -1) && (m <= (sp.num_cells+1))
                mi = m + 2
                u[i,mi] = basis(sp, m, points[i], derivative)
            end
        end
    end
    return u
end

"""
    SIxtransform(spline::Spline1D) -> Vector{Float64}
    SIxtransform(spline::Spline1D, uprime::AbstractVector) -> Vector{Float64}
    SIxtransform(spline::Spline1D, points::Vector{Float64}, uprime::AbstractVector) -> Vector{Float64}
    SIxtransform(sp::SplineParameters, a::Vector{Float64}, points::AbstractVector) -> Vector{Float64}

Evaluate the **first derivative** ``u'(x) = \\partial u / \\partial x`` of the B-spline
expansion at physical locations.

All variants delegate to `SItransform(..., derivative=1)`.

# Variants
- `SIxtransform(spline)` — allocates; evaluates at all mish points
- `SIxtransform(spline, uprime)` — in-place; `uprime` is written with derivatives at mish points
- `SIxtransform(spline, points, uprime)` — in-place at custom `points`
- `SIxtransform(sp, a, points)` — allocating at custom `points` given raw parameters

# Returns
- `Vector{Float64}`: First-derivative values at the requested points

See also: [`SItransform`](@ref), [`SIxxtransform`](@ref)
"""
function SIxtransform(spline::Spline1D)
    uprime = SItransform(spline.params,spline.a,spline.mishPoints,1)
    return uprime
end

function SIxtransform(spline::Spline1D, uprime::AbstractVector)

    uprime = SItransform(spline.params,spline.a,spline.mishPoints,uprime,1)
    return uprime
end

function SIxtransform(spline::Spline1D, points::Vector{real}, uprime::AbstractVector)

    uprime = SItransform(spline.params,spline.a,points,uprime,1)
    return uprime
end

function SIxtransform(sp::SplineParameters, a::Vector{real}, points::AbstractVector)

    uprime = SItransform(sp,a,points,1)
    return uprime
end

"""
    SIxxtransform(spline::Spline1D) -> Vector{Float64}
    SIxxtransform(spline::Spline1D, uprime2::AbstractVector) -> Vector{Float64}
    SIxxtransform(spline::Spline1D, points::Vector{Float64}, uprime2::AbstractVector) -> Vector{Float64}
    SIxxtransform(sp::SplineParameters, a::Vector{Float64}, points::Vector{Float64}) -> Vector{Float64}

Evaluate the **second derivative** ``u''(x) = \\partial^2 u / \\partial x^2`` of the
B-spline expansion at physical locations.

All variants delegate to `SItransform(..., derivative=2)`.

# Variants
- `SIxxtransform(spline)` — allocates; evaluates at all mish points
- `SIxxtransform(spline, uprime2)` — in-place; `uprime2` is written at mish points
- `SIxxtransform(spline, points, uprime2)` — in-place at custom `points`
- `SIxxtransform(sp, a, points)` — allocating at custom `points` given raw parameters

# Returns
- `Vector{Float64}`: Second-derivative values at the requested points

See also: [`SItransform`](@ref), [`SIxtransform`](@ref)
"""
function SIxxtransform(spline::Spline1D)
    uprime2 = SItransform(spline.params,spline.a,spline.mishPoints,2)
    return uprime2
end

function SIxxtransform(spline::Spline1D, uprime2::AbstractVector)

    uprime2 = SItransform(spline.params,spline.a,spline.mishPoints,uprime2,2)
    return uprime2
end

function SIxxtransform(spline::Spline1D, points::Vector{real}, uprime2::AbstractVector)

    uprime2 = SItransform(spline.params,spline.a,points,uprime2,2)
    return uprime2
end

function SIxxtransform(sp::SplineParameters, a::Vector{real}, points::Vector{real})

    uprime2 = SItransform(sp,a,points,2)
    return uprime2
end

#Module end
end
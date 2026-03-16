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
export SIIntcoefficients, SIInttransform
export setMishValues
export set_ahat_r3x!
# Generic (no-prefix) wrappers for abstract 1D basis dispatch
export Btransform, Btransform!, Bxtransform
export Atransform, Atransform!
export Itransform, Itransform!, Ixtransform, Ixxtransform, IInttransform

#Define some convenient aliases
const real = Float64
const int = Int64
const uint = UInt64
const ONESIXTH = 1.0/6.0
const FOURSIXTH = 4.0/6.0
const sqrt35 = sqrt(3.0/5.0)

# Homogeneous boundary conditions following Ooyama (2002).
# The *rank* r is the number of independent constraints imposed at the boundary,
# which removes r border coefficients from the spectral solve via the
# base-folding operator Γ.  The *type* t identifies which derivative is
# constrained (T0 = value, T1 = first derivative, T2 = second derivative).

"""
Rank-0 boundary condition (Ooyama 2002, Eq. 3.2a): **no constraint** at the
boundary. All border spline coefficients remain free variational parameters. Use
when no physical condition needs to be enforced at the domain edge (e.g. open
or internal boundaries in nested-domain applications).
"""
const R0 = Dict("R0" => 0)

"""
Rank-1, type-0 boundary condition (Ooyama 2002, Eq. 3.2b): **zero field value**
at the boundary, ``u(x_0) = 0`` (homogeneous Dirichlet). Removes one border
coefficient via the base-folding operator. Base-folding coefficients
(Ooyama 2002, Table 1): ``\\alpha_1 = -4``, ``\\beta_1 = -1``.
"""
const R1T0 = Dict("α1" => -4.0, "β1" => -1.0)

"""
Rank-1, type-1 boundary condition (Ooyama 2002, Eq. 3.2c): **zero first
derivative** at the boundary, ``u'(x_0) = 0`` (homogeneous Neumann). Removes
one border coefficient. Sufficient to enforce symmetry at a reflecting boundary.
Base-folding coefficients (Table 1): ``\\alpha_1 = 0``, ``\\beta_1 = 1``.
"""
const R1T1 = Dict("α1" =>  0.0, "β1" =>  1.0)

"""
Rank-1, type-2 boundary condition (Ooyama 2002, Eq. 3.2d): **zero second
derivative** at the boundary, ``u''(x_0) = 0``. Removes one border coefficient.
Base-folding coefficients (Table 1): ``\\alpha_1 = 2``, ``\\beta_1 = -1``.
"""
const R1T2 = Dict("α1" =>  2.0, "β1" => -1.0)

"""
Rank-2, type-1-0 boundary condition (Ooyama 2002, Eq. 3.2f): **zero value and
zero first derivative** at the boundary, ``u(x_0) = u'(x_0) = 0``. Removes two
border coefficients. Appropriate for a symmetrically reflecting boundary.
Base-folding coefficients (Table 1): ``\\alpha_2 = 1``, ``\\beta_2 = -0.5``.
"""
const R2T10 = Dict("α2" => 1.0, "β2" => -0.5)

"""
Rank-2, type-2-0 boundary condition (Ooyama 2002, Eq. 3.2g): **zero value and
zero second derivative** at the boundary, ``u(x_0) = u''(x_0) = 0``. Removes
two border coefficients. Forces the field to be **antisymmetric** with respect
to the boundary (the implicit field on the other side has opposite sign).
Base-folding coefficients (Table 1): ``\\alpha_2 = -1``, ``\\beta_2 = 0``.
"""
const R2T20 = Dict("α2" => -1.0, "β2" => 0.0)

"""
Rank-3 boundary condition (Ooyama 2002, Eq. 3.2h): **zero value, zero first
derivative, and zero second derivative** at the boundary,
``u(x_0) = u'(x_0) = u''(x_0) = 0``. Eliminates all three border
coefficients. The inhomogeneous variant R3X (where the three conditions carry
external data from an adjacent domain) is the primary tool for domain nesting
and will be implemented in a future release.
"""
const R3 = Dict("R3" => 0)

"""
Rank-3, inhomogeneous boundary condition (Ooyama 2002, section 3): **specified
value, first derivative, and second derivative** at the boundary,
``u(x_0) = u_0,\\; u'(x_0) = u_1,\\; u''(x_0) = u_2``. Eliminates all three
border coefficients, identical to R3 in the gammaBC matrix, but the SAtransform
uses a background coefficient vector `ahat` that carries the inhomogeneous data.
The primary use case is grid nesting, where an inner grid obtains boundary data
from an outer grid that changes each timestep. Set boundary values via
[`set_ahat_r3x!`](@ref).
"""
const R3X = Dict("R3X" => 0)

"""
Periodic boundary condition (Ooyama 2002, section 3e): couples the left and
right domain boundaries to simulate a cyclically continuous space. The border
basis functions are folded onto the interior so that the domain has exactly
`num_cells` independent coefficients. This is the only BC option for the
[`Fourier`](@ref Fourier_module) module and is also available for B-splines.
"""
const PERIODIC = Dict("PERIODIC" => 0)

# Default mish points per cell (kept for backward compatibility; prefer sp.mubar)
const mubar = 3
const gaussweight = [5.0/18.0, 8.0/18.0, 5.0/18.0]

# ── Quadrature rule tables ────────────────────────────────────────────────────
# Gauss-Legendre nodes on [-1, 1] and weights (sum to 2) for mubar 1–5.
# These are mapped to [0, 1] by _quadrature_rule.

const _GL_NODES = Dict{Int,Vector{Float64}}(
    1 => [0.0],
    2 => [-1.0/sqrt(3.0), 1.0/sqrt(3.0)],
    3 => [-sqrt(3.0/5.0), 0.0, sqrt(3.0/5.0)],
    4 => [-sqrt(3.0/7.0 + 2.0/7.0*sqrt(6.0/5.0)),
           -sqrt(3.0/7.0 - 2.0/7.0*sqrt(6.0/5.0)),
            sqrt(3.0/7.0 - 2.0/7.0*sqrt(6.0/5.0)),
            sqrt(3.0/7.0 + 2.0/7.0*sqrt(6.0/5.0))],
    5 => [-sqrt(5.0 + 2.0*sqrt(10.0/7.0))/3.0,
           -sqrt(5.0 - 2.0*sqrt(10.0/7.0))/3.0,
            0.0,
            sqrt(5.0 - 2.0*sqrt(10.0/7.0))/3.0,
            sqrt(5.0 + 2.0*sqrt(10.0/7.0))/3.0],
)
const _GL_WEIGHTS = Dict{Int,Vector{Float64}}(
    1 => [2.0],
    2 => [1.0, 1.0],
    3 => [5.0/9.0, 8.0/9.0, 5.0/9.0],
    4 => [(18.0 - sqrt(30.0))/36.0,
           (18.0 + sqrt(30.0))/36.0,
           (18.0 + sqrt(30.0))/36.0,
           (18.0 - sqrt(30.0))/36.0],
    5 => [(322.0 - 13.0*sqrt(70.0))/900.0,
           (322.0 + 13.0*sqrt(70.0))/900.0,
            128.0/225.0,
           (322.0 + 13.0*sqrt(70.0))/900.0,
           (322.0 - 13.0*sqrt(70.0))/900.0],
)

const _MAX_GAUSS_MUBAR = 5

"""
    _quadrature_rule(mubar::Int, quadrature::Symbol) -> (Vector{Float64}, Vector{Float64})

Return `(quadpoints, quadweights)` for the given `mubar` and quadrature type.

- `quadpoints`: positions within a cell as fractions of DX, in `[0, 1)`.
- `quadweights`: integration weights normalised so that `sum(quadweights) == 1`.
  Multiply by `DX` to get the physical integration weight.

Quadrature types:
- `:gauss` — Gauss-Legendre rule (mubar 1–$(_MAX_GAUSS_MUBAR) supported).
- `:regular` — equispaced midpoint rule (any mubar ≥ 1).
"""
function _quadrature_rule(mb::Int, quadrature::Symbol)
    if mb < 1
        throw(ArgumentError("mubar must be ≥ 1, got $mb"))
    end

    if quadrature == :gauss
        if mb > _MAX_GAUSS_MUBAR
            throw(ArgumentError(
                "Gauss-Legendre tables only available for mubar 1–$(_MAX_GAUSS_MUBAR), got $mb. " *
                "Use quadrature=:regular for larger mubar."))
        end
        # Map GL nodes from [-1, 1] to [0, 1] and scale weights to sum to 1
        gl_nodes = _GL_NODES[mb]
        gl_weights = _GL_WEIGHTS[mb]
        qpts = [(xi + 1.0) / 2.0 for xi in gl_nodes]
        qwts = [w / 2.0 for w in gl_weights]
        return qpts, qwts

    elseif quadrature == :regular
        # Equispaced midpoint rule: points at cell-local (2k-1)/(2N)
        qpts = [(2*k - 1) / (2*mb) for k in 1:mb]
        qwts = fill(1.0 / mb, mb)
        return qpts, qwts

    else
        throw(ArgumentError(
            "Unknown quadrature type: $quadrature. Use :gauss or :regular."))
    end
end

"""
    SplineParameters

Immutable parameter struct (using `@kwdef`) for a 1D cubic B-spline basis.

# Fields
- `xmin::Float64`: Left boundary of the domain
- `xmax::Float64`: Right boundary of the domain
- `num_cells::Int64`: Number of spline cells
- `mubar::Int64`: Number of quadrature (mish) points per cell (default `3`).
  This controls the physical grid density within each cell. The spectral resolution
  is always `bDim = num_cells + 3` regardless of `mubar`.
- `quadrature::Symbol`: Quadrature type (default `:gauss`).
  - `:gauss` — Gauss-Legendre quadrature (mubar 1–5). Optimal integration accuracy.
  - `:regular` — equispaced midpoint rule (any mubar ≥ 1). Produces a globally
    uniform physical grid with spacing `DX/mubar`.
- `l_q::Float64`: Filter length scale (default `2.0`). Larger values produce smoother spectral fits
- `BCL::Dict`: Left boundary condition (one of [`R0`](@ref), [`R1T0`](@ref), [`R1T1`](@ref), [`R1T2`](@ref), [`R2T10`](@ref), [`R2T20`](@ref), [`R3`](@ref), [`PERIODIC`](@ref))
- `BCR::Dict`: Right boundary condition (same options as `BCL`)
- `DX::Float64`: Cell width, computed as `(xmax - xmin) / num_cells`
- `DXrecip::Float64`: Reciprocal of `DX`, precomputed for efficiency
- `bDim::Int64`: Number of spectral coefficients, computed as `num_cells + 3`
- `mishDim::Int64`: Total physical (mish) gridpoints, computed as `num_cells * mubar`

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

# Regular grid with 2 points per cell
sp_reg = CubicBSpline.SplineParameters(
    xmin = 0.0, xmax = 100.0, num_cells = 20,
    mubar = 2, quadrature = :regular
)
```

See also: [`Spline1D`](@ref), [`_quadrature_rule`](@ref)
"""
Base.@kwdef struct SplineParameters
    xmin::real = 0.0
    xmax::real = 0.0
    num_cells::int = 1
    mubar::int = 3
    quadrature::Symbol = :gauss
    l_q::real = 2.0
    BCL::Dict = R0
    BCR::Dict = R0
    DX::real = (xmax - xmin) / num_cells
    DXrecip::real = 1.0/DX
    bDim::int = num_cells + 3
    mishDim::int = num_cells * mubar
end

function _validate_spline_params(sp::SplineParameters)
    if sp.mubar < 1
        throw(ArgumentError("mubar must be ≥ 1, got $(sp.mubar)"))
    end
    if sp.quadrature ∉ (:gauss, :regular)
        throw(ArgumentError(
            "Unknown quadrature type: $(sp.quadrature). Use :gauss or :regular."))
    end
    if sp.quadrature == :gauss && sp.mubar > _MAX_GAUSS_MUBAR
        throw(ArgumentError(
            "Gauss-Legendre tables only available for mubar 1–$(_MAX_GAUSS_MUBAR), got $(sp.mubar). " *
            "Use quadrature=:regular for larger mubar."))
    end
end

"""
    Spline1D

One-dimensional cubic B-spline object.  Construct via `Spline1D(sp::SplineParameters)`.

# Fields
- `params::SplineParameters`: Configuration (domain, cells, BCs, filter length, mubar, quadrature)
- `quadpoints::Vector{Float64}`: Cell-local quadrature positions in `[0, 1)` (length `mubar`)
- `quadweights::Vector{Float64}`: Quadrature weights normalised to sum to 1 (length `mubar`)
- `gammaBC::Matrix{Float64}`: Boundary-condition projection matrix (maps interior to full coefficient space)
- `pq`: Full `(P + Q)` matrix used in the least-squares / variational solve
- `pqFactor::SuiteSparse.CHOLMOD.Factor{Float64}`: Sparse Cholesky factorisation of the open-form `(P + Q)` matrix for fast solves
- `p1`: Full `(P⁽¹⁾ + Q)` matrix for the integral variational solve (same Q as `pq`, but P uses first-derivative basis ``\\varphi'_m``; see [`calcP1factor`](@ref))
- `p1Factor::SuiteSparse.CHOLMOD.Factor{Float64}`: Sparse Cholesky factorisation of the open-form `(P⁽¹⁾ + Q)` matrix, used by [`SIIntcoefficients`](@ref) for fast integration solves
- `mishPoints::Vector{Float64}`: Physical locations of the mish points (length `num_cells * mubar`)
- `uMish::Vector{Float64}`: Physical field values at mish points (mutable working buffer)
- `b::Vector{Float64}`: B-vector (result of SB transform, inner products ⟨φₘ, u⟩)
- `a::Vector{Float64}`: Spectral coefficient vector (result of SA transform)
- `ahat::Vector{Float64}`: Background coefficient vector for inhomogeneous R3X boundary conditions (length `bDim`). Initialised to zeros; set via [`set_ahat_r3x!`](@ref) for grid nesting.

# Notes
- Constructing `Spline1D` builds `gammaBC` and factorises the `(P + Q)` matrix, which is the
  computationally expensive step.  Reuse spline objects when possible.
- The number of quadrature points per cell (`mubar`) and the quadrature type
  (`:gauss` or `:regular`) are set in [`SplineParameters`](@ref).

See also: [`SplineParameters`](@ref), [`SBtransform`](@ref), [`SAtransform`](@ref), [`SItransform`](@ref)
"""
struct Spline1D
    params::SplineParameters
    quadpoints::Vector{real}
    quadweights::Vector{real}
    gammaBC::Matrix{real}
    pq
    pqFactor::SuiteSparse.CHOLMOD.Factor{Float64}
    p1
    p1Factor::SuiteSparse.CHOLMOD.Factor{Float64}
    mishPoints::Vector{real}
    uMish::Vector{real}
    b::Vector{real}
    a::Vector{real}
    ahat::Vector{real}
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
    elseif sp.BCL == R3 || haskey(sp.BCL, "R3X")
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
    elseif sp.BCR == R3 || haskey(sp.BCR, "R3X")
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
    elseif sp.BCL == R3 || haskey(sp.BCL, "R3X")
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
    qpts, qwts = _quadrature_rule(sp.mubar, sp.quadrature)
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
                for mu = 1:sp.mubar
                    i = mu + (sp.mubar * mc)
                    x = sp.xmin + mc * sp.DX + qpts[mu] * sp.DX
                    pm1 = basis(sp, m1, x, 0)
                    qm1 = basis(sp, m1, x, 3)
                    pm2 = basis(sp, m2, x, 0)
                    qm2 = basis(sp, m2, x, 3)
                    P[mi1,mi2] += sp.DX * qwts[mu] * pm1 * pm2
                    Q[mi1,mi2] += sp.DX * qwts[mu] * eps_q * qm1 * qm2
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
    calcP1factor(sp::SplineParameters, gammaBC::Matrix{Float64}) -> (Matrix{Float64}, SuiteSparse.CHOLMOD.Factor)

Build the integral variational `(P⁽¹⁾ + Q)` matrix and return its sparse Cholesky factorisation.

Follows Ooyama (2002) Eq. 4.24. Like [`calcPQfactor`](@ref), but the P matrix inner products
use the **first derivative** of the basis function rather than the function itself:

``P^{(1)}_{m_1 m_2} = \\sum_{\\text{cells}} \\Delta x \\sum_\\mu w_\\mu \\varphi'_{m_1}(x_\\mu)\\,\\varphi'_{m_2}(x_\\mu)``

The Q (smoothing) matrix is identical to the one in [`calcPQfactor`](@ref).  The factorised
system is used by [`SIIntcoefficients`](@ref) to find spectral coefficients of the
indefinite integral.

# Arguments
- `sp::SplineParameters`: Spline parameters (domain, cells, filter length `l_q`)
- `gammaBC::Matrix{Float64}`: Boundary-condition projection matrix from [`calcGammaBC`](@ref)

# Returns
- `(p1q::Matrix{Float64}, p1qFactor)`: Full `(P⁽¹⁾ + Q)` matrix and its Cholesky factor

See also: [`calcPQfactor`](@ref), [`SIIntcoefficients`](@ref)
"""
function calcP1factor(sp::SplineParameters, gammaBC::Matrix{real})
    qpts, qwts = _quadrature_rule(sp.mubar, sp.quadrature)
    eps_q = ((sp.l_q*sp.DX)/(2*π))^6
    Mdim = sp.num_cells + 3

    # Create the P1 matrix (first-derivative inner products) and Q matrix (same as calcPQfactor)
    P1 = zeros(real, Mdim, Mdim)
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
                for mu = 1:sp.mubar
                    i = mu + (sp.mubar * mc)
                    x = sp.xmin + mc * sp.DX + qpts[mu] * sp.DX
                    pm1 = basis(sp, m1, x, 1)  # first derivative of basis
                    pm2 = basis(sp, m2, x, 1)  # first derivative of basis
                    qm1 = basis(sp, m1, x, 3)
                    qm2 = basis(sp, m2, x, 3)
                    P1[mi1,mi2] += sp.DX * qwts[mu] * pm1 * pm2
                    Q[mi1,mi2]  += sp.DX * qwts[mu] * eps_q * qm1 * qm2
                end
            end
        end
    end

    # Fold in the BCs to get open form.
    # A small Tikhonov regularization (eps_lambda * I) is added to handle the null space of P1:
    # the constant function lies in null(P1) and null(Q) when BCs impose no constraint
    # (e.g. R0). This null space corresponds to the free constant of integration (C0), which
    # is set by the caller. The regularization does not affect the non-constant modes.
    eps_lambda = 1e-10
    P1Q = Symmetric(P1 + Q)
    P1Qopen = Symmetric((gammaBC * P1 * gammaBC') + (gammaBC * Q * gammaBC'))
    n = size(P1Qopen, 1)
    P1Qsparse = sparse(P1Qopen) + eps_lambda * sparse(I, n, n)
    P1Qfactor = cholesky(Symmetric(P1Qsparse))
    return P1Q, P1Qfactor
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
    qpts, _ = _quadrature_rule(sp.mubar, sp.quadrature)
    x = zeros(real, sp.mishDim)
    for mc = 0:(sp.num_cells-1)
        for mu = 1:sp.mubar
            i = mu + (sp.mubar * mc)
            x[i] = sp.xmin + mc * sp.DX + qpts[mu] * sp.DX
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
    _validate_spline_params(sp)
    qpts, qwts = _quadrature_rule(sp.mubar, sp.quadrature)

    gammaBC = calcGammaBC(sp)
    pq, pqFactor = calcPQfactor(sp, gammaBC)
    p1, p1Factor = calcP1factor(sp, gammaBC)

    mishPoints = calcMishPoints(sp)
    uMish = zeros(real,sp.mishDim)

    b = zeros(real,sp.bDim)
    a = zeros(real,sp.bDim)
    ahat = zeros(real,sp.bDim)

    spline = Spline1D(sp,qpts,qwts,gammaBC,pq,pqFactor,p1,p1Factor,mishPoints,uMish,b,a,ahat)
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
    qpts, qwts = _quadrature_rule(sp.mubar, sp.quadrature)
    Mdim = sp.num_cells + 3
    b = zeros(real,Mdim)

    for mi = 1:Mdim
        m = mi - 2
        for mc = 0:(sp.num_cells-1)
            if (mc < (m - 2)) || (mc > (m + 1))
                continue
            end
            for mu = 1:sp.mubar
                i = mu + (sp.mubar * mc)
                x = sp.xmin + mc * sp.DX + qpts[mu] * sp.DX
                bm = basis(sp, m, x, 0)
                b[mi] += sp.DX * qwts[mu] * bm * uMish[i]
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
    qpts, qwts = _quadrature_rule(sp.mubar, sp.quadrature)
    Mdim = sp.num_cells + 3
    b = zeros(real, Mdim)

    for mi = 1:Mdim
        m = mi - 2
        for mc = 0:(sp.num_cells-1)
            if (mc < (m - 2)) || (mc > (m + 1))
                continue
            end
            for mu = 1:sp.mubar
                i = mu + (sp.mubar * mc)
                x = sp.xmin + mc * sp.DX + qpts[mu] * sp.DX
                bm = basis(sp, m, x, 1)
                b[mi] += sp.DX * qwts[mu] * bm * uMish[i]
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
    SIIntcoefficients(sp::SplineParameters, gammaBC::Matrix{Float64}, p1Factor, uMish::Vector{Float64}) -> Vector{Float64}
    SIIntcoefficients(spline::Spline1D, uMish::Vector{Float64}) -> Vector{Float64}

Compute the **integral spectral coefficients** of the indefinite integral of `uMish`.

Follows Ooyama (2002) Eqs. 4.23–4.25.  Builds the right-hand side vector
``r_m = \\int \\varphi'_m(x)\\, f(x)\\,dx`` via quadrature, then solves the variational
system using the pre-factored ``(P^{(1)} + Q)`` matrix:

``\\hat{a}^{\\text{int}} = \\Gamma^T \\Bigl[(\\Gamma\\,(P^{(1)}+Q)\\,\\Gamma^T)^{-1}\\,\\Gamma\\, r\\Bigr]``

where ``P^{(1)}_{m_1 m_2} = \\int \\varphi'_{m_1}\\varphi'_{m_2}\\,dx``.

Note: ``r_m = -\\text{SBxtransform}(f, 0, 0)`` — the raw derivative-basis inner product without
boundary correction.  Boundary values of ``f`` are not required; the constant of integration
is applied separately via `C0` in [`SIInttransform`](@ref).

# Variants
- `SIIntcoefficients(sp, gammaBC, p1Factor, uMish)` — lowest-level; all components explicit
- `SIIntcoefficients(spline, uMish)` — convenience form using cached `spline` internals

# Arguments
- `sp` / `spline`: Spline parameters or a constructed `Spline1D`
- `uMish::Vector{Float64}`: Field values at the mish points (the function to be integrated)

# Returns
- `Vector{Float64}`: Spectral coefficient vector `aInt` of length `num_cells + 3`

See also: [`SIInttransform`](@ref), [`SBxtransform`](@ref), [`calcP1factor`](@ref)
"""
function SIIntcoefficients(sp::SplineParameters, gammaBC::Matrix{real}, p1Factor, uMish::Vector{real})
    # RHS = ∫ φ'_m f dx = -SBxtransform(f, 0, 0)
    bx = SBxtransform(sp, uMish, 0.0, 0.0)
    aInt = gammaBC' * (p1Factor \ (gammaBC * (-bx)))
    return aInt
end

function SIIntcoefficients(spline::Spline1D, uMish::Vector{real})
    # RHS = ∫ φ'_m f dx = -SBxtransform(f, 0, 0)
    bx = SBxtransform(spline.params, uMish, 0.0, 0.0)
    aInt = spline.gammaBC' * (spline.p1Factor \ (spline.gammaBC * (-bx)))
    return aInt
end

"""
    SIInttransform(sp::SplineParameters, gammaBC::Matrix{Float64}, p1Factor, uMish::Vector{Float64}, C0::Float64 = 0.0) -> Vector{Float64}
    SIInttransform(spline::Spline1D, uMish::Vector{Float64}, C0::Float64 = 0.0) -> Vector{Float64}

Compute the **indefinite integral** of the B-spline expansion in physical space.

Applies [`SIIntcoefficients`](@ref) (Ooyama 2002, Eqs. 4.23–4.25) to obtain the spectral
coefficients of the antiderivative, then evaluates the expansion at the mish points via
[`SItransform`](@ref).  The optional constant of integration `C0` is added uniformly to
every physical output value.

# Variants
- `SIInttransform(sp, gammaBC, p1Factor, uMish, C0)` — lowest-level call; all components explicit
- `SIInttransform(spline, uMish, C0)` — convenience form using cached `spline` internals

# Arguments
- `sp` / `spline`: Spline parameters or a constructed `Spline1D`
- `uMish::Vector{Float64}`: Field values at the mish points (the function to be integrated)
- `C0::Float64`: Constant of integration (default `0.0`); added uniformly to the output

# Returns
- `Vector{Float64}`: Integral field values of length `num_cells * 3` at the mish points

See also: [`SIIntcoefficients`](@ref), [`SItransform`](@ref), [`SIxtransform`](@ref)
"""
function SIInttransform(sp::SplineParameters, gammaBC::Matrix{real}, p1Factor, uMish::Vector{real}, C0::real = 0.0)
    aInt = SIIntcoefficients(sp, gammaBC, p1Factor, uMish)
    uInt = SItransform(sp, aInt)
    return uInt .+ C0
end

function SIInttransform(spline::Spline1D, uMish::Vector{real}, C0::real = 0.0)
    aInt = SIIntcoefficients(spline, uMish)
    uInt = SItransform(spline.params, aInt)
    return uInt .+ C0
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

"""
    _has_r3x(sp::SplineParameters) -> Bool

Return `true` if either boundary condition is R3X (inhomogeneous rank-3).
"""
_has_r3x(sp::SplineParameters) = haskey(sp.BCL, "R3X") || haskey(sp.BCR, "R3X")

function SAtransform!(spline::Spline1D)
    if _has_r3x(spline.params)
        btilde = spline.gammaBC * (spline.b .- (spline.pq * spline.ahat))
        spline.a .= (spline.gammaBC' * (spline.pqFactor \ btilde)) .+ spline.ahat
    else
        # In-place version of the SA transform
        spline.a .= spline.gammaBC' * (spline.pqFactor \ (spline.gammaBC * spline.b))
    end
end

function SAtransform(spline::Spline1D, b::Vector{real}, ahat::Vector{real})

    btilde = spline.gammaBC * (b - (spline.pq * ahat))
    a = (spline.gammaBC' * (spline.pqFactor \ btilde)) + ahat
    return a
end

# ── R3X inhomogeneous boundary support ───────────────────────────────────────

"""
    _border_matrix(sp::SplineParameters, side::Symbol) -> Matrix{Float64}

Build the 3×3 matrix relating boundary derivative values to border coefficients.

For `side == :left`, the three border basis functions are m = -1, 0, 1 (array
indices 1, 2, 3). For `side == :right`, they are m = N-1, N, N+1 (last 3 array
indices), where N = `num_cells`.

Row `d+1` of the matrix contains the `d`-th derivative of each border basis
function evaluated at the boundary point.

Returns an invertible 3×3 matrix `M` such that `M \\ [u₀, u₁, u₂]` gives the
border coefficients that produce the desired boundary values.
"""
function _border_matrix(sp::SplineParameters, side::Symbol)
    x = (side == :left) ? sp.xmin : sp.xmax
    border_ms = (side == :left) ? [-1, 0, 1] : [sp.num_cells-1, sp.num_cells, sp.num_cells+1]
    M = zeros(3, 3)
    for (col, m) in enumerate(border_ms)
        for deriv in 0:2
            M[deriv+1, col] = basis(sp, m, x, deriv)
        end
    end
    return M
end

"""
    set_ahat_r3x!(spline::Spline1D, u0::Real, u1::Real, u2::Real, side::Symbol)

Set the inhomogeneous R3X boundary conditions on `spline`.

Computes the border coefficients that produce the specified boundary values
`u(x₀) = u0`, `u'(x₀) = u1`, `u''(x₀) = u2` and stores them in `spline.ahat`.

# Arguments
- `spline::Spline1D`: Spline with R3X boundary condition on the specified side
- `u0::Real`: Desired field value at the boundary
- `u1::Real`: Desired first derivative at the boundary
- `u2::Real`: Desired second derivative at the boundary
- `side::Symbol`: `:left` or `:right`

# Example
```julia
sp = SplineParameters(xmin=0.0, xmax=1.0, num_cells=10,
                      BCL=CubicBSpline.R3X, BCR=CubicBSpline.R3X)
spline = Spline1D(sp)
set_ahat_r3x!(spline, 1.0, 0.0, 0.0, :left)   # u(0) = 1
set_ahat_r3x!(spline, 2.0, 0.0, 0.0, :right)  # u(1) = 2
```

See also: [`R3X`](@ref), [`_border_matrix`](@ref)
"""
function set_ahat_r3x!(spline::Spline1D, u0::Real, u1::Real, u2::Real, side::Symbol)
    M = _border_matrix(spline.params, side)
    border_coeffs = M \ [u0, u1, u2]
    if side == :left
        spline.ahat[1:3] .= border_coeffs
    elseif side == :right
        spline.ahat[end-2:end] .= border_coeffs
    else
        throw(ArgumentError("side must be :left or :right, got :$side"))
    end
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
    qpts, _ = _quadrature_rule(sp.mubar, sp.quadrature)

    u = zeros(real,sp.mishDim)
    for mc = 0:(sp.num_cells-1)
        for mu = 1:sp.mubar
            i = mu + (sp.mubar * mc)
            x = sp.xmin + mc * sp.DX + qpts[mu] * sp.DX
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
    u = zeros(Float64,length(points),spline.params.bDim)
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

"""
    spline_basis_matrix(spline::Spline1D; gammaBC=nothing) -> Matrix{Float64}

Build the ``(M \\times N)`` cubic B-spline evaluation matrix.

Entry `[i, j]` is the value of the `j`-th B-spline basis function
``\\varphi_j(x_i)`` at the `i`-th mish point. If `gammaBC` is provided,
returns the boundary-condition-folded form ``\\mathbf{M} \\boldsymbol{\\Gamma}^T``
with reduced column count.

# Keywords
- `gammaBC::Union{Matrix{Float64}, Nothing}`: If provided, fold boundary conditions
  to produce the reduced matrix of size `(mishDim, Minterior)`.

See also: [`spline_1st_derivative_matrix`](@ref), [`spline_2nd_derivative_matrix`](@ref)
"""
function spline_basis_matrix(spline::Spline1D; gammaBC::Union{Matrix{Float64}, Nothing}=nothing)
    sp = spline.params
    M = zeros(Float64, sp.mishDim, sp.bDim)
    for i in 1:sp.mishDim
        x = spline.mishPoints[i]
        xm = ceil(Int64, (x - sp.xmin - (2.0 * sp.DX)) * sp.DXrecip)
        for m = xm:(xm + 3)
            if (m >= -1) && (m <= (sp.num_cells + 1))
                mi = m + 2
                M[i, mi] = basis(sp, m, x, 0)
            end
        end
    end
    if gammaBC !== nothing
        return M * gammaBC'
    end
    return M
end

"""
    spline_1st_derivative_matrix(spline::Spline1D; gammaBC=nothing) -> Matrix{Float64}

Build the ``(M \\times N)`` cubic B-spline **first-derivative** matrix.

Entry `[i, j]` is the first derivative of the `j`-th B-spline basis function
``\\varphi'_j(x_i)`` at the `i`-th mish point. If `gammaBC` is provided,
returns the boundary-condition-folded form ``\\mathbf{M}_x \\boldsymbol{\\Gamma}^T``
with reduced column count.

# Keywords
- `gammaBC::Union{Matrix{Float64}, Nothing}`: If provided, fold boundary conditions
  to produce the reduced matrix of size `(mishDim, Minterior)`.

See also: [`spline_basis_matrix`](@ref), [`spline_2nd_derivative_matrix`](@ref)
"""
function spline_1st_derivative_matrix(spline::Spline1D; gammaBC::Union{Matrix{Float64}, Nothing}=nothing)
    sp = spline.params
    M = zeros(Float64, sp.mishDim, sp.bDim)
    for i in 1:sp.mishDim
        x = spline.mishPoints[i]
        xm = ceil(Int64, (x - sp.xmin - (2.0 * sp.DX)) * sp.DXrecip)
        for m = xm:(xm + 3)
            if (m >= -1) && (m <= (sp.num_cells + 1))
                mi = m + 2
                M[i, mi] = basis(sp, m, x, 1)
            end
        end
    end
    if gammaBC !== nothing
        return M * gammaBC'
    end
    return M
end

"""
    spline_2nd_derivative_matrix(spline::Spline1D; gammaBC=nothing) -> Matrix{Float64}

Build the ``(M \\times N)`` cubic B-spline **second-derivative** matrix.

Entry `[i, j]` is the second derivative of the `j`-th B-spline basis function
``\\varphi''_j(x_i)`` at the `i`-th mish point. If `gammaBC` is provided,
returns the boundary-condition-folded form ``\\mathbf{M}_{xx} \\boldsymbol{\\Gamma}^T``
with reduced column count.

# Keywords
- `gammaBC::Union{Matrix{Float64}, Nothing}`: If provided, fold boundary conditions
  to produce the reduced matrix of size `(mishDim, Minterior)`.

See also: [`spline_basis_matrix`](@ref), [`spline_1st_derivative_matrix`](@ref)
"""
function spline_2nd_derivative_matrix(spline::Spline1D; gammaBC::Union{Matrix{Float64}, Nothing}=nothing)
    sp = spline.params
    M = zeros(Float64, sp.mishDim, sp.bDim)
    for i in 1:sp.mishDim
        x = spline.mishPoints[i]
        xm = ceil(Int64, (x - sp.xmin - (2.0 * sp.DX)) * sp.DXrecip)
        for m = xm:(xm + 3)
            if (m >= -1) && (m <= (sp.num_cells + 1))
                mi = m + 2
                M[i, mi] = basis(sp, m, x, 2)
            end
        end
    end
    if gammaBC !== nothing
        return M * gammaBC'
    end
    return M
end

# ---------------------------------------------------------------------------
# Generic wrappers (no "S" prefix) dispatching on Spline1D
# These enable abstract 1D basis code that calls Btransform, Itransform, etc.
# without needing to know the underlying basis type.
# ---------------------------------------------------------------------------

"""Generic B-transform wrapper for `Spline1D`. Delegates to [`SBtransform`](@ref)."""
Btransform(spline::Spline1D, uMish::Vector{real}) = SBtransform(spline, uMish)

"""Generic in-place B-transform wrapper for `Spline1D`. Delegates to `SBtransform!`."""
Btransform!(spline::Spline1D) = SBtransform!(spline)

"""Generic Bx-transform wrapper for `Spline1D`. Delegates to [`SBxtransform`](@ref)."""
Bxtransform(spline::Spline1D, uMish::Vector{real}, BCL::real, BCR::real) =
    SBxtransform(spline, uMish, BCL, BCR)

"""Generic A-transform wrapper for `Spline1D` (allocating). Delegates to [`SAtransform`](@ref)."""
Atransform(spline::Spline1D, b::AbstractVector) = SAtransform(spline, b)

"""Generic A-transform wrapper for `Spline1D` (inhomogeneous form). Delegates to [`SAtransform`](@ref)."""
Atransform(spline::Spline1D, b::Vector{real}, ahat::Vector{real}) = SAtransform(spline, b, ahat)

"""Generic in-place A-transform wrapper for `Spline1D`. Delegates to `SAtransform!`."""
Atransform!(spline::Spline1D) = SAtransform!(spline)

"""Generic in-place I-transform wrapper for `Spline1D`. Delegates to `SItransform!`."""
Itransform!(spline::Spline1D) = SItransform!(spline)

"""Generic I-transform wrapper for `Spline1D` (mish points, in-place output). Delegates to [`SItransform`](@ref)."""
Itransform(spline::Spline1D, u::AbstractVector) = SItransform(spline, u)

"""Generic I-transform wrapper for `Spline1D` (custom points, in-place output). Delegates to [`SItransform`](@ref)."""
Itransform(spline::Spline1D, points::Vector{real}, u::AbstractVector) = SItransform(spline, points, u)

"""Generic Ix-transform wrapper for `Spline1D` (allocating, mish points). Delegates to [`SIxtransform`](@ref)."""
Ixtransform(spline::Spline1D) = SIxtransform(spline)

"""Generic Ix-transform wrapper for `Spline1D` (mish points, in-place output). Delegates to [`SIxtransform`](@ref)."""
Ixtransform(spline::Spline1D, uprime::AbstractVector) = SIxtransform(spline, uprime)

"""Generic Ix-transform wrapper for `Spline1D` (custom points, in-place output). Delegates to [`SIxtransform`](@ref)."""
Ixtransform(spline::Spline1D, points::Vector{real}, uprime::AbstractVector) =
    SIxtransform(spline, points, uprime)

"""Generic Ixx-transform wrapper for `Spline1D` (allocating, mish points). Delegates to [`SIxxtransform`](@ref)."""
Ixxtransform(spline::Spline1D) = SIxxtransform(spline)

"""Generic Ixx-transform wrapper for `Spline1D` (mish points, in-place output). Delegates to [`SIxxtransform`](@ref)."""
Ixxtransform(spline::Spline1D, uprime2::AbstractVector) = SIxxtransform(spline, uprime2)

"""Generic Ixx-transform wrapper for `Spline1D` (custom points, in-place output). Delegates to [`SIxxtransform`](@ref)."""
Ixxtransform(spline::Spline1D, points::Vector{real}, uprime2::AbstractVector) =
    SIxxtransform(spline, points, uprime2)

"""Generic indefinite-integral transform wrapper for `Spline1D`. Delegates to [`SIInttransform`](@ref)."""
IInttransform(spline::Spline1D, uMish::Vector{real}, C0::real = 0.0) =
    SIInttransform(spline, uMish, C0)

"""Generic I-transform matrix wrapper for `Spline1D`. Delegates to [`SItransform_matrix`](@ref)."""
Itransform_matrix(spline::Spline1D, points::Vector{Float64}, derivative::Int64=0) =
    SItransform_matrix(spline, points, derivative)

#Module end
end
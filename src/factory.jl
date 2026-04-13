# factory.jl — Grid factory for SpringsteelGrid{G,I,J,K}
#
# Provides:
#   • parse_geometry()         — geometry string → sentinel type tuple
#   • compute_derived_params() — fill in jDim/b_jDim/kDim/b_kDim
#   • createGrid(::SpringsteelGridParameters) — unified factory (all geometries)
#
# Must be included AFTER types.jl and basis_interface.jl.

# ────────────────────────────────────────────────────────────────────────────
# parse_geometry
# ────────────────────────────────────────────────────────────────────────────

"""
    parse_geometry(geometry::String) -> Tuple

    parse_geometry("R")             -> (CartesianGeometry(),   SplineBasisType(),    NoBasisType(),        NoBasisType())
    parse_geometry("Spline1D")      -> (CartesianGeometry(),   SplineBasisType(),    NoBasisType(),        NoBasisType())
    parse_geometry("RZ")            -> (CartesianGeometry(),   SplineBasisType(),    NoBasisType(),        ChebyshevBasisType())
    parse_geometry("RL")            -> (CylindricalGeometry(), SplineBasisType(),    FourierBasisType(),   NoBasisType())
    parse_geometry("Polar")         -> (CylindricalGeometry(), SplineBasisType(),    FourierBasisType(),   NoBasisType())  # alias for RL
    parse_geometry("RR")            -> (CartesianGeometry(),   SplineBasisType(),    SplineBasisType(),    NoBasisType())
    parse_geometry("Spline2D")      -> (CartesianGeometry(),   SplineBasisType(),    SplineBasisType(),    NoBasisType())
    parse_geometry("RLZ")           -> (CylindricalGeometry(), SplineBasisType(),    FourierBasisType(),   ChebyshevBasisType())
    parse_geometry("Cylindrical")   -> (CylindricalGeometry(), SplineBasisType(),    FourierBasisType(),   ChebyshevBasisType())  # alias for RLZ
    parse_geometry("RRR")           -> (CartesianGeometry(),   SplineBasisType(),    SplineBasisType(),    SplineBasisType())
    parse_geometry("Spline3D")      -> (CartesianGeometry(),   SplineBasisType(),    SplineBasisType(),    SplineBasisType())  # alias for RRR
    parse_geometry("Samurai")       -> (CartesianGeometry(),   SplineBasisType(),    SplineBasisType(),    SplineBasisType())  # alias for RRR
    parse_geometry("SL")            -> (SphericalGeometry(),   SplineBasisType(),    FourierBasisType(),   NoBasisType())
    parse_geometry("SphericalShell")-> (SphericalGeometry(),   SplineBasisType(),    FourierBasisType(),   NoBasisType())  # alias for SL
    parse_geometry("SLZ")           -> (SphericalGeometry(),   SplineBasisType(),    FourierBasisType(),   ChebyshevBasisType())
    parse_geometry("Sphere")        -> (SphericalGeometry(),   SplineBasisType(),    FourierBasisType(),   ChebyshevBasisType())  # alias for SLZ
    parse_geometry("L")             -> (CartesianGeometry(),   FourierBasisType(),   NoBasisType(),        NoBasisType())
    parse_geometry("Ring1D")        -> (CartesianGeometry(),   FourierBasisType(),   NoBasisType(),        NoBasisType())  # alias for L
    parse_geometry("LL")            -> (CartesianGeometry(),   FourierBasisType(),   FourierBasisType(),   NoBasisType())
    parse_geometry("Ring2D")        -> (CartesianGeometry(),   FourierBasisType(),   FourierBasisType(),   NoBasisType())  # alias for LL
    parse_geometry("LLZ")           -> (CartesianGeometry(),   FourierBasisType(),   FourierBasisType(),   ChebyshevBasisType())
    parse_geometry("DoublyPeriodic")-> (CartesianGeometry(),   FourierBasisType(),   FourierBasisType(),   ChebyshevBasisType())  # alias for LLZ
    parse_geometry("Z")             -> (CartesianGeometry(),   ChebyshevBasisType(), NoBasisType(),        NoBasisType())
    parse_geometry("Column1D")      -> (CartesianGeometry(),   ChebyshevBasisType(), NoBasisType(),        NoBasisType())  # alias for Z
    parse_geometry("ZZ")            -> (CartesianGeometry(),   ChebyshevBasisType(), ChebyshevBasisType(), NoBasisType())
    parse_geometry("Column2D")      -> (CartesianGeometry(),   ChebyshevBasisType(), ChebyshevBasisType(), NoBasisType())  # alias for ZZ
    parse_geometry("ZZZ")           -> (CartesianGeometry(),   ChebyshevBasisType(), ChebyshevBasisType(), ChebyshevBasisType())
    parse_geometry("Column3D")      -> (CartesianGeometry(),   ChebyshevBasisType(), ChebyshevBasisType(), ChebyshevBasisType())  # alias for ZZZ

Map a geometry string to a 4-tuple of sentinel type instances `(G, It, Jt, Kt)` used as
type parameters of [`SpringsteelGrid`](@ref):
- `G`  — geometry sentinel (`CartesianGeometry`, `CylindricalGeometry`, `SphericalGeometry`)
- `It` — i-dimension basis sentinel (always `SplineBasisType`)
- `Jt` — j-dimension basis sentinel or `NoBasisType`
- `Kt` — k-dimension basis sentinel or `NoBasisType`

# Arguments
- `geometry::String`: Canonical geometry identifiers: `"R"`, `"Spline1D"`, `"RZ"`, `"RL"`,
  `"RR"`, `"Spline2D"`, `"RLZ"`, `"RRR"`, `"SL"`, `"SLZ"`,
  `"L"`, `"LL"`, `"LLZ"`, `"Z"`, `"ZZ"`, `"ZZZ"`.
  Descriptive aliases: `"Polar"`, `"Cylindrical"`, `"Spline3D"`, `"Samurai"`,
  `"SphericalShell"`, `"Sphere"`, `"Ring1D"`, `"Ring2D"`, `"DoublyPeriodic"`,
  `"Column1D"`, `"Column2D"`, `"Column3D"`.

# Returns
A `Tuple{AbstractGeometry, AbstractBasisType, AbstractBasisType, AbstractBasisType}` of
singleton sentinel instances ready for use with `typeof()` in `SpringsteelGrid{G,I,J,K}`.

# Throws
- `DomainError` if `geometry` is not a recognised string.

# Example
```julia
G, It, Jt, Kt = parse_geometry("RL")
G isa CylindricalGeometry   # true
Jt isa FourierBasisType      # true
```

See also: [`SpringsteelGrid`](@ref), [`createGrid`](@ref)
"""
function parse_geometry(geometry::String)
    mapping = Dict(
        # ── Spline-based (original) ──────────────────────────────────────────────
        "R"            => (CartesianGeometry(),   SplineBasisType(),    NoBasisType(),        NoBasisType()),
        "Spline1D"     => (CartesianGeometry(),   SplineBasisType(),    NoBasisType(),        NoBasisType()),
        "RZ"           => (CartesianGeometry(),   SplineBasisType(),    NoBasisType(),        ChebyshevBasisType()),
        "RL"           => (CylindricalGeometry(), SplineBasisType(),    FourierBasisType(),   NoBasisType()),
        "Polar"        => (CylindricalGeometry(), SplineBasisType(),    FourierBasisType(),   NoBasisType()),
        "RR"           => (CartesianGeometry(),   SplineBasisType(),    SplineBasisType(),    NoBasisType()),
        "Spline2D"     => (CartesianGeometry(),   SplineBasisType(),    SplineBasisType(),    NoBasisType()),
        "RLZ"          => (CylindricalGeometry(), SplineBasisType(),    FourierBasisType(),   ChebyshevBasisType()),
        "Cylindrical"  => (CylindricalGeometry(), SplineBasisType(),    FourierBasisType(),   ChebyshevBasisType()),
        "RRR"          => (CartesianGeometry(),   SplineBasisType(),    SplineBasisType(),    SplineBasisType()),
        "Spline3D"     => (CartesianGeometry(),   SplineBasisType(),    SplineBasisType(),    SplineBasisType()),
        "Samurai"      => (CartesianGeometry(),   SplineBasisType(),    SplineBasisType(),    SplineBasisType()),
        "SL"           => (SphericalGeometry(),   SplineBasisType(),    FourierBasisType(),   NoBasisType()),
        "SphericalShell" => (SphericalGeometry(), SplineBasisType(),    FourierBasisType(),   NoBasisType()),
        "SLZ"          => (SphericalGeometry(),   SplineBasisType(),    FourierBasisType(),   ChebyshevBasisType()),
        "Sphere"       => (SphericalGeometry(),   SplineBasisType(),    FourierBasisType(),   ChebyshevBasisType()),
        # ── Fourier-based ────────────────────────────────────────────────────────
        "L"            => (CartesianGeometry(),   FourierBasisType(),   NoBasisType(),        NoBasisType()),
        "Ring1D"       => (CartesianGeometry(),   FourierBasisType(),   NoBasisType(),        NoBasisType()),
        "Fourier1D"    => (CartesianGeometry(),   FourierBasisType(),   NoBasisType(),        NoBasisType()),
        "LL"           => (CartesianGeometry(),   FourierBasisType(),   FourierBasisType(),   NoBasisType()),
        "Ring2D"       => (CartesianGeometry(),   FourierBasisType(),   FourierBasisType(),   NoBasisType()),
        "Fourier2D"    => (CartesianGeometry(),   FourierBasisType(),   FourierBasisType(),   NoBasisType()),
        "LLZ"          => (CartesianGeometry(),   FourierBasisType(),   FourierBasisType(),   ChebyshevBasisType()),
        "DoublyPeriodic" => (CartesianGeometry(), FourierBasisType(),   FourierBasisType(),   ChebyshevBasisType()),
        # ── Chebyshev-based ──────────────────────────────────────────────────────
        "Z"            => (CartesianGeometry(),   ChebyshevBasisType(), NoBasisType(),        NoBasisType()),
        "Column1D"     => (CartesianGeometry(),   ChebyshevBasisType(), NoBasisType(),        NoBasisType()),
        "Chebyshev1D"  => (CartesianGeometry(),   ChebyshevBasisType(), NoBasisType(),        NoBasisType()),
        "ZZ"           => (CartesianGeometry(),   ChebyshevBasisType(), ChebyshevBasisType(), NoBasisType()),
        "Column2D"     => (CartesianGeometry(),   ChebyshevBasisType(), ChebyshevBasisType(), NoBasisType()),
        "Chebyshev2D"  => (CartesianGeometry(),   ChebyshevBasisType(), ChebyshevBasisType(), NoBasisType()),
        "ZZZ"          => (CartesianGeometry(),   ChebyshevBasisType(), ChebyshevBasisType(), ChebyshevBasisType()),
        "Column3D"     => (CartesianGeometry(),   ChebyshevBasisType(), ChebyshevBasisType(), ChebyshevBasisType()),
        "Chebyshev3D"  => (CartesianGeometry(),   ChebyshevBasisType(), ChebyshevBasisType(), ChebyshevBasisType()),
    )
    haskey(mapping, geometry) ||
        throw(DomainError(geometry, "Unknown geometry for SpringsteelGrid: $geometry"))
    return mapping[geometry]
end

# ────────────────────────────────────────────────────────────────────────────
# Geometry alias normalisation
# ────────────────────────────────────────────────────────────────────────────

# Maps alias strings to their canonical geometry name used in createGrid
# and compute_derived_params dispatch.
const _GEOMETRY_ALIASES = Dict{String,String}(
    # ── Spline grid: descriptive name → canonical code ────────────────────────
    "Polar"          => "RL",
    "Cylindrical"    => "RLZ",
    "Spline3D"       => "RRR",
    "Samurai"        => "RRR",
    "SphericalShell" => "SL",
    "Sphere"         => "SLZ",
    # ── Fourier grid: descriptive name → canonical code ──────────────────────
    "Ring1D"         => "L",
    "Ring2D"         => "LL",
    "DoublyPeriodic" => "LLZ",
    # ── Chebyshev grid: descriptive name → canonical code ────────────────────
    "Column1D"       => "Z",
    "Column2D"       => "ZZ",
    "Column3D"       => "ZZZ",
)

"""Return the canonical geometry name, resolving any alias."""
_normalize_geometry(g::String) = get(_GEOMETRY_ALIASES, g, g)

# ────────────────────────────────────────────────────────────────────────────
# Internal dimension helpers
# ────────────────────────────────────────────────────────────────────────────

# Cylindrical Fourier ring dimensions at radial index ri
@inline function _cyl_ring_dims(ri::Int)
    lpoints = 4 + 4 * ri
    return lpoints, ri       # (lpoints, kmax_ring)
end

# Spherical Fourier ring dimensions at latitude theta
@inline function _sph_ring_dims(theta::Float64, max_ri::Int)
    kmax_ring = max(1, round(Int, sin(theta) * max_ri))
    lpoints   = 4 * kmax_ring
    return lpoints, kmax_ring
end

# jDim, b_jDim for cylindrical Fourier (RL, RLZ)
function _fourier_j_dims_cyl(gp::SpringsteelGridParameters)
    jDim, b_jDim = 0, 0
    for r in 1:gp.iDim
        ri = r + gp.patchOffsetL
        lp, km = _cyl_ring_dims(ri)
        jDim   += lp
        b_jDim += 1 + 2 * km
    end
    return jDim, b_jDim
end

# mishPoints for the i-direction, using first variable's BCs
function _i_mishpoints(gp::SpringsteelGridParameters)
    first_key = first(keys(gp.vars))
    tmp = Spline1D(SplineParameters(
        xmin     = gp.iMin,
        xmax     = gp.iMax,
        num_cells = gp.num_cells,
        mubar    = gp.mubar,
        quadrature = gp.quadrature,
        l_q      = 2.0,
        BCL      = _get_spline_bc(gp.BCL, first_key),
        BCR      = _get_spline_bc(gp.BCR, first_key)))
    return tmp.mishPoints
end

# jDim, b_jDim for spherical Fourier (SL, SLZ)
function _fourier_j_dims_sph(gp::SpringsteelGridParameters)
    theta  = _i_mishpoints(gp)
    max_ri = gp.iDim + gp.patchOffsetL
    jDim, b_jDim = 0, 0
    for t in theta
        lp, km = _sph_ring_dims(t, max_ri)
        jDim   += lp
        b_jDim += 1 + 2 * km
    end
    return jDim, b_jDim
end

# jDim, b_jDim for Cartesian Spline j (RR, RRR, Spline2D)
function _cartesian_j_dims(gp::SpringsteelGridParameters)
    if gp.jDim == 0
        dy = gp.jMax - gp.jMin
        dx = gp.iMax - gp.iMin
        nc_j = Int64(ceil(gp.num_cells * (dy / dx)))
    else
        nc_j = Int64(gp.jDim / gp.mubar)
    end
    return nc_j * gp.mubar, nc_j + 3
end

# kDim, b_kDim for Cartesian Spline k (RRR)
function _cartesian_k_dims(gp::SpringsteelGridParameters)
    if gp.kDim == 0
        dk = gp.kMax - gp.kMin
        dx = gp.iMax - gp.iMin
        nc_k = Int64(ceil(gp.num_cells * (dk / dx)))
    else
        nc_k = Int64(gp.kDim / gp.mubar)
    end
    return nc_k * gp.mubar, nc_k + 3
end

# Reconstruct SpringsteelGridParameters with updated j/k dims
function _update_gp(gp::SpringsteelGridParameters;
        jDim   = gp.jDim,   b_jDim  = gp.b_jDim,
        kDim   = gp.kDim,   b_kDim  = gp.b_kDim)
    return SpringsteelGridParameters(
        geometry       = gp.geometry,
        iMin           = gp.iMin,
        iMax           = gp.iMax,
        num_cells      = gp.num_cells,
        mubar          = gp.mubar,
        quadrature     = gp.quadrature,
        iDim           = gp.iDim,
        b_iDim         = gp.b_iDim,
        l_q            = gp.l_q,
        BCL            = gp.BCL,
        BCR            = gp.BCR,
        jMin           = gp.jMin,
        jMax           = gp.jMax,
        max_wavenumber = gp.max_wavenumber,
        jDim           = jDim,
        b_jDim         = b_jDim,
        BCU            = gp.BCU,
        BCD            = gp.BCD,
        kMin           = gp.kMin,
        kMax           = gp.kMax,
        kDim           = kDim,
        b_kDim         = b_kDim,
        BCB            = gp.BCB,
        BCT            = gp.BCT,
        vars           = gp.vars,
        fourier_filter = gp.fourier_filter,
        chebyshev_filter = gp.chebyshev_filter,
        spectralIndexL = gp.spectralIndexL,
        spectralIndexR = gp.spectralIndexR,
        patchOffsetL   = gp.patchOffsetL,
        patchOffsetR   = gp.patchOffsetR,
        tile_num       = gp.tile_num)
end

# ────────────────────────────────────────────────────────────────────────────
# compute_derived_params
# ────────────────────────────────────────────────────────────────────────────

"""
    compute_derived_params(gp::SpringsteelGridParameters) -> SpringsteelGridParameters

Compute derived dimension fields (`jDim`, `b_jDim`, `kDim`, `b_kDim`) and return a new,
fully-populated `SpringsteelGridParameters`.  For most geometries the i-dimension fields
are already set by the `@kwdef` defaults; only geometries that have variable or
domain-aspect-ratio-dependent j/k dimensions require recomputation.

| Geometry | What is derived |
|:-------- |:--------------- |
| `"R"`, `"Spline1D"`, `"RZ"` | Nothing — `jDim`/`kDim` stay at user-provided values |
| `"RL"`, `"RLZ"` | `jDim`, `b_jDim` from cylindrical ring formula `∑(4+4rᵢ)` |
| `"SL"`, `"SLZ"` | `jDim`, `b_jDim` from spherical sin(θ) ring formula |
| `"RR"`, `"Spline2D"` | `jDim`, `b_jDim` from domain aspect ratio |
| `"RRR"` | `jDim`, `b_jDim` AND `kDim`, `b_kDim` from domain aspect ratio |

See also: [`parse_geometry`](@ref), [`createGrid`](@ref)
"""
function compute_derived_params(gp::SpringsteelGridParameters)
    geom = _normalize_geometry(gp.geometry)

    if geom in ("R", "Spline1D", "RZ")
        return gp   # nothing to derive

    elseif geom in ("RL", "RLZ")
        jDim, b_jDim = _fourier_j_dims_cyl(gp)
        return _update_gp(gp; jDim=jDim, b_jDim=b_jDim)

    elseif geom in ("SL", "SLZ")
        jDim, b_jDim = _fourier_j_dims_sph(gp)
        return _update_gp(gp; jDim=jDim, b_jDim=b_jDim)

    elseif geom in ("RR", "Spline2D")
        jDim, b_jDim = _cartesian_j_dims(gp)
        return _update_gp(gp; jDim=jDim, b_jDim=b_jDim)

    elseif geom == "RRR"
        jDim, b_jDim = _cartesian_j_dims(gp)
        kDim, b_kDim = _cartesian_k_dims(gp)
        return _update_gp(gp; jDim=jDim, b_jDim=b_jDim, kDim=kDim, b_kDim=b_kDim)

    # Fourier-based (user supplies iDim/b_iDim/jDim/b_jDim/kDim/b_kDim directly)
    elseif geom in ("L", "LL", "LLZ")
        return gp

    # Chebyshev-based (user supplies iDim/b_iDim/jDim/b_jDim/kDim/b_kDim directly)
    elseif geom in ("Z", "ZZ", "ZZZ")
        return gp

    else
        return gp
    end
end

# ────────────────────────────────────────────────────────────────────────────
# BC lookup helper: per-variable key with "default" fallback
# ────────────────────────────────────────────────────────────────────────────
@inline function _get_bc(bc_dict::Dict, key)
    haskey(bc_dict, key) && return bc_dict[key]
    return bc_dict["default"]
end

# Typed BC helpers: look up the per-variable BC and convert to internal Dict
@inline _get_spline_bc(bc_dict::Dict, key) = _convert_bc(_get_bc(bc_dict, key), :spline)
@inline _get_chebyshev_bc(bc_dict::Dict, key) = _convert_bc(_get_bc(bc_dict, key), :chebyshev)

# ────────────────────────────────────────────────────────────────────────────
# BoundaryConditions → internal Dict conversion layer
# ────────────────────────────────────────────────────────────────────────────

"""Convert a `BoundaryConditions` to the CubicBSpline internal Dict."""
function _bc_to_spline_dict(bc::BoundaryConditions)
    bc.periodic && return CubicBSpline.PERIODIC

    r = bc_rank(bc)
    if r == 0
        return CubicBSpline.R0
    elseif r == 1
        if bc.u !== nothing
            return Dict("α1" => -4.0, "β1" => -1.0)       # Dirichlet (R1T0)
        elseif bc.du !== nothing
            return Dict("α1" => 0.0, "β1" => 1.0)          # Neumann (R1T1)
        elseif bc.d2u !== nothing
            return Dict("α1" => 2.0, "β1" => -1.0)         # 2nd deriv (R1T2)
        elseif bc.robin !== nothing
            α, β, _ = bc.robin
            λ = -β / α                                      # Ooyama λ parameter
            return Dict("α1" => -4.0/(3λ + 1),
                        "β1" => (3λ - 1)/(3λ + 1))         # Robin (R1T10(λ))
        end
    elseif r == 2
        if bc.u !== nothing && bc.du !== nothing
            return Dict("α2" => 1.0, "β2" => -0.5)         # R2T10
        elseif bc.u !== nothing && bc.d2u !== nothing
            return Dict("α2" => -1.0, "β2" => 0.0)         # R2T20
        else
            throw(ArgumentError(
                "Unsupported rank-2 BC combination for CubicBSpline: " *
                "du + d2u without u is not defined"))
        end
    elseif r == 3
        return is_inhomogeneous(bc) ? Dict("R3X" => 0) : Dict("R3" => 0)
    end
end

"""Convert a `BoundaryConditions` to the Chebyshev internal Dict."""
function _bc_to_chebyshev_dict(bc::BoundaryConditions)
    bc.periodic && throw(ArgumentError("PeriodicBC is not supported for Chebyshev basis"))
    bc.robin !== nothing && throw(ArgumentError("RobinBC is not supported for Chebyshev basis"))

    r = bc_rank(bc)
    if r == 0
        return Chebyshev.R0
    elseif r == 1
        if bc.u !== nothing
            return Dict("α0" => bc.u)           # R1T0 (Dirichlet)
        elseif bc.du !== nothing
            return Dict("α1" => bc.du)           # R1T1 (Neumann)
        elseif bc.d2u !== nothing
            return Dict("α2" => bc.d2u)          # R1T2 (2nd derivative)
        end
    else
        throw(ArgumentError("Rank-$r BCs are not yet implemented for Chebyshev basis"))
    end
end

"""Validate that a `BoundaryConditions` is periodic (the only valid Fourier BC)."""
function _validate_fourier_bc(bc::BoundaryConditions)
    if !bc.periodic
        throw(ArgumentError("Fourier basis requires PeriodicBC"))
    end
    return Fourier.PERIODIC
end

"""
    _convert_bc(bc, basis_type::Symbol)

Convert a boundary condition to the internal Dict representation for the given
basis type (`:spline`, `:chebyshev`, or `:fourier`).  Legacy Dict BCs are
passed through unchanged.
"""
function _convert_bc(bc::BoundaryConditions, basis_type::Symbol)
    if basis_type === :spline
        return _bc_to_spline_dict(bc)
    elseif basis_type === :chebyshev
        return _bc_to_chebyshev_dict(bc)
    elseif basis_type === :fourier
        return _validate_fourier_bc(bc)
    else
        throw(ArgumentError("Unknown basis type: $basis_type"))
    end
end

# Pass-through for legacy Dict BCs
_convert_bc(bc::Dict, ::Symbol) = bc

# ────────────────────────────────────────────────────────────────────────────
# Per-geometry internal creation functions
# ────────────────────────────────────────────────────────────────────────────

# 1D Cartesian (R, Spline1D)
function _create_cartesian_1d(gp::SpringsteelGridParameters)
    nvars   = length(values(gp.vars))
    splines = Array{Spline1D}(undef, 1, nvars)
    ibasis  = SplineBasisArray(splines)
    jbasis  = NoBasisArray()
    kbasis  = NoBasisArray()
    spectral = zeros(Float64, gp.b_iDim, nvars)
    physical = zeros(Float64, gp.iDim, nvars, 3)

    grid = SpringsteelGrid{CartesianGeometry, typeof(ibasis), typeof(jbasis), typeof(kbasis)}(
        gp, ibasis, jbasis, kbasis, spectral, physical)

    for key in keys(gp.vars)
        var_l_q = get(gp.l_q, key, get(gp.l_q, "default", 2.0))
        grid.ibasis.data[1, gp.vars[key]] = Spline1D(SplineParameters(
            xmin      = gp.iMin,
            xmax      = gp.iMax,
            num_cells = gp.num_cells,
            mubar     = gp.mubar,
            quadrature = gp.quadrature,
            l_q       = var_l_q,
            BCL       = _get_spline_bc(gp.BCL, key),
            BCR       = _get_spline_bc(gp.BCR, key)))
    end
    return grid
end

# 2D Cartesian Spline×Chebyshev (RZ)
function _create_cartesian_2d_rz(gp::SpringsteelGridParameters)
    nvars   = length(values(gp.vars))
    # ibasis: b_kDim splines per variable (one per Chebyshev mode)
    splines  = Array{Spline1D}(undef, gp.b_kDim, nvars)
    columns  = Array{Chebyshev1D}(undef, nvars)
    ibasis   = SplineBasisArray(splines)
    jbasis   = NoBasisArray()
    kbasis   = ChebyshevBasisArray(columns)

    spec_dim = gp.b_kDim * gp.b_iDim
    phys_dim = gp.iDim * gp.kDim
    spectral = zeros(Float64, spec_dim, nvars)
    physical = zeros(Float64, phys_dim, nvars, 5)

    grid = SpringsteelGrid{CartesianGeometry, typeof(ibasis), typeof(jbasis), typeof(kbasis)}(
        gp, ibasis, jbasis, kbasis, spectral, physical)

    for key in keys(gp.vars)
        v = gp.vars[key]
        var_l_q = get(gp.l_q, key, get(gp.l_q, "default", 2.0))
        # One spline per Chebyshev mode
        for z in 1:gp.b_kDim
            grid.ibasis.data[z, v] = Spline1D(SplineParameters(
                xmin      = gp.iMin,
                xmax      = gp.iMax,
                num_cells = gp.num_cells,
                mubar     = gp.mubar,
                quadrature = gp.quadrature,
                l_q       = var_l_q,
                BCL       = _get_spline_bc(gp.BCL, key),
                BCR       = _get_spline_bc(gp.BCR, key)))
        end
        grid.kbasis.data[v] = Chebyshev1D(ChebyshevParameters(
            zmin = gp.kMin,
            zmax = gp.kMax,
            zDim = gp.kDim,
            bDim = gp.b_kDim,
            BCB  = _get_chebyshev_bc(gp.BCB, key),
            BCT  = _get_chebyshev_bc(gp.BCT, key)))
    end
    return grid
end

# 2D Cartesian Spline×Spline (RR, Spline2D)
function _create_cartesian_2d_rr(gp::SpringsteelGridParameters)
    nvars   = length(values(gp.vars))
    # ibasis: b_jDim splines per variable (one per j spectral mode)
    splines = Array{Spline1D}(undef, gp.b_jDim, nvars)
    rings   = Array{Spline1D}(undef, gp.iDim, nvars)
    ibasis  = SplineBasisArray(splines)
    jbasis  = SplineBasisArray(rings)
    kbasis  = NoBasisArray()

    spec_dim = gp.b_iDim * gp.b_jDim
    phys_dim = gp.iDim * gp.jDim
    spectral = zeros(Float64, spec_dim, nvars)
    physical = zeros(Float64, phys_dim, nvars, 5)

    grid = SpringsteelGrid{CartesianGeometry, typeof(ibasis), typeof(jbasis), typeof(kbasis)}(
        gp, ibasis, jbasis, kbasis, spectral, physical)

    nc_j = Int64(gp.jDim / gp.mubar)
    for key in keys(gp.vars)
        v = gp.vars[key]
        var_l_q_i = get(gp.l_q, key, get(gp.l_q, "default", 2.0))
        var_l_q_j = get(gp.l_q, string(key, "_j"), var_l_q_i)
        for j in 1:gp.b_jDim
            grid.ibasis.data[j, v] = Spline1D(SplineParameters(
                xmin      = gp.iMin,
                xmax      = gp.iMax,
                num_cells = gp.num_cells,
                mubar     = gp.mubar,
                quadrature = gp.quadrature,
                l_q       = var_l_q_i,
                BCL       = _get_spline_bc(gp.BCL, key),
                BCR       = _get_spline_bc(gp.BCR, key)))
        end
        for r in 1:gp.iDim
            grid.jbasis.data[r, v] = Spline1D(SplineParameters(
                xmin      = gp.jMin,
                xmax      = gp.jMax,
                num_cells = nc_j,
                mubar     = gp.mubar,
                quadrature = gp.quadrature,
                l_q       = var_l_q_j,
                BCL       = _get_spline_bc(gp.BCD, key),
                BCR       = _get_spline_bc(gp.BCU, key)))
        end
    end
    return grid
end

# 3D Cartesian Spline×Spline×Spline (RRR)
function _create_cartesian_3d_rrr(gp::SpringsteelGridParameters)
    nvars   = length(values(gp.vars))
    splines  = Array{Spline1D}(undef, gp.b_jDim, gp.b_kDim, nvars)
    rings    = Array{Spline1D}(undef, gp.iDim, gp.b_kDim, nvars)
    columns  = Array{Spline1D}(undef, gp.iDim, gp.jDim, nvars)
    ibasis   = SplineBasisArray(splines)
    jbasis   = SplineBasisArray(rings)
    kbasis   = SplineBasisArray(columns)

    spec_dim = gp.b_iDim * gp.b_jDim * gp.b_kDim
    phys_dim = gp.iDim * gp.jDim * gp.kDim
    spectral = zeros(Float64, spec_dim, nvars)
    physical = zeros(Float64, phys_dim, nvars, 7)

    grid = SpringsteelGrid{CartesianGeometry, typeof(ibasis), typeof(jbasis), typeof(kbasis)}(
        gp, ibasis, jbasis, kbasis, spectral, physical)

    nc_j = Int64(gp.jDim / gp.mubar)
    nc_k = Int64(gp.kDim / gp.mubar)
    for key in keys(gp.vars)
        v = gp.vars[key]
        var_l_q_i = get(gp.l_q, key, get(gp.l_q, "default", 2.0))
        var_l_q_j = get(gp.l_q, string(key, "_j"), var_l_q_i)
        var_l_q_k = get(gp.l_q, string(key, "_k"), var_l_q_i)
        for j in 1:gp.b_jDim, z in 1:gp.b_kDim
            grid.ibasis.data[j, z, v] = Spline1D(SplineParameters(
                xmin      = gp.iMin, xmax = gp.iMax,
                num_cells = gp.num_cells, mubar = gp.mubar,
                quadrature = gp.quadrature, l_q = var_l_q_i,
                BCL       = _get_spline_bc(gp.BCL, key), BCR = _get_spline_bc(gp.BCR, key)))
        end
        for r in 1:gp.iDim, z in 1:gp.b_kDim
            grid.jbasis.data[r, z, v] = Spline1D(SplineParameters(
                xmin      = gp.jMin, xmax = gp.jMax,
                num_cells = nc_j, mubar = gp.mubar,
                quadrature = gp.quadrature, l_q = var_l_q_j,
                BCL       = _get_spline_bc(gp.BCD, key), BCR = _get_spline_bc(gp.BCU, key)))
        end
        for r in 1:gp.iDim, l in 1:gp.jDim
            grid.kbasis.data[r, l, v] = Spline1D(SplineParameters(
                xmin      = gp.kMin, xmax = gp.kMax,
                num_cells = nc_k, mubar = gp.mubar,
                quadrature = gp.quadrature, l_q = var_l_q_k,
                BCL       = _get_spline_bc(gp.BCB, key), BCR = _get_spline_bc(gp.BCT, key)))
        end
    end
    return grid
end

function _get_or_build_ring_shared(fp::FourierParameters)
    return Fourier.Fourier1D(fp)
end

# Internal helper: build cylindrical or spherical Fourier rings into pre-allocated array
function _fill_fourier_rings_cyl!(rings_arr, gp::SpringsteelGridParameters,
                                  var_kmax::Int, idx2::Int)
    iDim = gp.iDim
    for r in 1:iDim
        ri = r + gp.patchOffsetL
        lpoints, kmax_ring = _cyl_ring_dims(ri)
        if var_kmax >= 0; kmax_ring = min(var_kmax, ri); end
        dl     = 2π / lpoints
        offset = 0.5 * dl * (ri - 1)
        fp = FourierParameters(
            ymin  = offset,
            yDim  = lpoints,
            kmax  = kmax_ring,
            bDim  = ri * 2 + 1)
        rings_arr[r, idx2] = _get_or_build_ring_shared(fp)
    end
end

function _fill_fourier_rings_sph!(rings_arr, gp::SpringsteelGridParameters, mishpts,
                                  var_kmax::Int, idx2::Int)
    iDim   = gp.iDim
    max_ri = iDim + gp.patchOffsetL
    for r in 1:iDim
        theta = mishpts[r]
        lpoints, kmax_ring = _sph_ring_dims(theta, max_ri)
        if var_kmax >= 0; kmax_ring = min(var_kmax, kmax_ring); end
        dl     = 2π / lpoints
        offset = 0.5 * dl * r
        fp = FourierParameters(
            ymin  = offset,
            yDim  = lpoints,
            kmax  = kmax_ring,
            bDim  = 1 + 2 * kmax_ring)
        rings_arr[r, idx2] = _get_or_build_ring_shared(fp)
    end
end

# 2D Cylindrical Spline×Fourier (RL)
function _create_cylindrical_2d_rl(gp::SpringsteelGridParameters)
    nvars   = length(values(gp.vars))
    splines = Array{Spline1D}(undef, 3, nvars)   # 3 splines per var (k=0, cos, sin)
    rings   = Array{Fourier1D}(undef, gp.iDim, nvars)
    ibasis  = SplineBasisArray(splines)
    jbasis  = FourierBasisArray(rings)
    kbasis  = NoBasisArray()

    # Spectral layout: wavenumber-interleaved uniform blocks of b_iDim coefficients.
    # Total rows = b_iDim * (1 + 2*kDim) where kDim = iDim + patchOffsetL.
    # (Using b_jDim — the triangular sum of per-ring Fourier modes — over-allocates
    # for patches and under-allocates for tiles, producing OOB writes in spectralTransform!.)
    kDim_fourier = gp.iDim + gp.patchOffsetL
    spec_dim     = gp.b_iDim * (1 + 2 * kDim_fourier)
    spectral = zeros(Float64, spec_dim, nvars)
    physical = zeros(Float64, gp.jDim, nvars, 5)

    grid = SpringsteelGrid{CylindricalGeometry, typeof(ibasis), typeof(jbasis), typeof(kbasis)}(
        gp, ibasis, jbasis, kbasis, spectral, physical)


    for key in keys(gp.vars)
        v = gp.vars[key]
        var_l_q = get(gp.l_q, key, get(gp.l_q, "default", 2.0))
        for i in 1:3
            grid.ibasis.data[i, v] = Spline1D(SplineParameters(
                xmin      = gp.iMin,
                xmax      = gp.iMax,
                num_cells = gp.num_cells,
                mubar     = gp.mubar,
                quadrature = gp.quadrature,
                l_q       = var_l_q,
                BCL       = _get_spline_bc(gp.BCL, key),
                BCR       = _get_spline_bc(gp.BCR, key)))
        end
        var_kmax = get(gp.max_wavenumber, key, get(gp.max_wavenumber, "default", -1))
        _fill_fourier_rings_cyl!(grid.jbasis.data, gp, var_kmax, v)
    end
    return grid
end

# 3D Cylindrical Spline×Fourier×Chebyshev (RLZ)
function _create_cylindrical_3d_rlz(gp::SpringsteelGridParameters)
    nvars   = length(values(gp.vars))
    # ibasis: b_kDim splines per var (one per Chebyshev mode)
    splines  = Array{Spline1D}(undef, gp.b_kDim, nvars)
    # jbasis: rings indexed (iDim, b_kDim) — shared across vars
    rings    = Array{Fourier1D}(undef, gp.iDim, gp.b_kDim)
    columns  = Array{Chebyshev1D}(undef, nvars)
    ibasis   = SplineBasisArray(splines)
    jbasis   = FourierBasisArray(rings)
    kbasis   = ChebyshevBasisArray(columns)

    kDim_fourier = gp.iDim + gp.patchOffsetL
    spec_dim     = gp.b_kDim * gp.b_iDim * (1 + 2 * kDim_fourier)
    phys_dim     = gp.kDim * gp.jDim
    spectral = zeros(Float64, spec_dim, nvars)
    physical = zeros(Float64, phys_dim, nvars, 7)

    grid = SpringsteelGrid{CylindricalGeometry, typeof(ibasis), typeof(jbasis), typeof(kbasis)}(
        gp, ibasis, jbasis, kbasis, spectral, physical)

    for key in keys(gp.vars)
        v = gp.vars[key]
        var_l_q = get(gp.l_q, key, get(gp.l_q, "default", 2.0))
        for z in 1:gp.b_kDim
            grid.ibasis.data[z, v] = Spline1D(SplineParameters(
                xmin      = gp.iMin,
                xmax      = gp.iMax,
                num_cells = gp.num_cells,
                mubar     = gp.mubar,
                quadrature = gp.quadrature,
                l_q       = var_l_q,
                BCL       = _get_spline_bc(gp.BCL, key),
                BCR       = _get_spline_bc(gp.BCR, key)))
        end
        grid.kbasis.data[v] = Chebyshev1D(ChebyshevParameters(
            zmin = gp.kMin,
            zmax = gp.kMax,
            zDim = gp.kDim,
            bDim = gp.b_kDim,
            BCB  = _get_chebyshev_bc(gp.BCB, key),
            BCT  = _get_chebyshev_bc(gp.BCT, key)))
    end
    # Rings shared across vars: use default (or last) var_kmax
    var_kmax = get(gp.max_wavenumber, "default", -1)
    for key in keys(gp.vars)
        var_kmax = get(gp.max_wavenumber, key, var_kmax)
    end

    for b in 1:gp.b_kDim
        _fill_fourier_rings_cyl!(grid.jbasis.data, gp, var_kmax, b)
    end
    return grid
end

# 2D Spherical Spline×Fourier (SL)
function _create_spherical_2d_sl(gp::SpringsteelGridParameters)
    nvars   = length(values(gp.vars))
    mishpts = _i_mishpoints(gp)
    splines = Array{Spline1D}(undef, 3, nvars)
    rings   = Array{Fourier1D}(undef, gp.iDim, nvars)
    ibasis  = SplineBasisArray(splines)
    jbasis  = FourierBasisArray(rings)
    kbasis  = NoBasisArray()

    # Spectral layout: wavenumber-interleaved uniform blocks of b_iDim coefficients.
    # Total rows = b_iDim * (1 + 2*kDim) where kDim = iDim + patchOffsetL.
    # (Using b_jDim — the sum of per-ring Fourier modes — would under-allocate for
    # tiles near the poles where b_jDim < b_iDim*(1+2*kDim).)
    kDim_fourier = gp.iDim + gp.patchOffsetL
    spec_dim     = gp.b_iDim * (1 + 2 * kDim_fourier)
    spectral = zeros(Float64, spec_dim, nvars)
    physical = zeros(Float64, gp.jDim, nvars, 5)

    grid = SpringsteelGrid{SphericalGeometry, typeof(ibasis), typeof(jbasis), typeof(kbasis)}(
        gp, ibasis, jbasis, kbasis, spectral, physical)


    for key in keys(gp.vars)
        v = gp.vars[key]
        var_l_q = get(gp.l_q, key, get(gp.l_q, "default", 2.0))
        for i in 1:3
            grid.ibasis.data[i, v] = Spline1D(SplineParameters(
                xmin      = gp.iMin,
                xmax      = gp.iMax,
                num_cells = gp.num_cells,
                mubar     = gp.mubar,
                quadrature = gp.quadrature,
                l_q       = var_l_q,
                BCL       = _get_spline_bc(gp.BCL, key),
                BCR       = _get_spline_bc(gp.BCR, key)))
        end
        var_kmax = get(gp.max_wavenumber, key, get(gp.max_wavenumber, "default", -1))
        _fill_fourier_rings_sph!(grid.jbasis.data, gp, mishpts, var_kmax, v)
    end
    return grid
end

# 3D Spherical Spline×Fourier×Chebyshev (SLZ)
function _create_spherical_3d_slz(gp::SpringsteelGridParameters)
    nvars   = length(values(gp.vars))
    mishpts = _i_mishpoints(gp)
    splines  = Array{Spline1D}(undef, gp.b_kDim, nvars)
    rings    = Array{Fourier1D}(undef, gp.iDim, gp.b_kDim)
    columns  = Array{Chebyshev1D}(undef, nvars)
    ibasis   = SplineBasisArray(splines)
    jbasis   = FourierBasisArray(rings)
    kbasis   = ChebyshevBasisArray(columns)

    max_ri       = gp.iDim + gp.patchOffsetL
    kDim_fourier = max_ri
    spec_dim     = gp.b_kDim * gp.b_iDim * (1 + 2 * kDim_fourier)
    phys_dim     = gp.kDim * gp.jDim
    spectral = zeros(Float64, spec_dim, nvars)
    physical = zeros(Float64, phys_dim, nvars, 7)

    grid = SpringsteelGrid{SphericalGeometry, typeof(ibasis), typeof(jbasis), typeof(kbasis)}(
        gp, ibasis, jbasis, kbasis, spectral, physical)

    for key in keys(gp.vars)
        v = gp.vars[key]
        var_l_q = get(gp.l_q, key, get(gp.l_q, "default", 2.0))
        for z in 1:gp.b_kDim
            grid.ibasis.data[z, v] = Spline1D(SplineParameters(
                xmin      = gp.iMin,
                xmax      = gp.iMax,
                num_cells = gp.num_cells,
                mubar     = gp.mubar,
                quadrature = gp.quadrature,
                l_q       = var_l_q,
                BCL       = _get_spline_bc(gp.BCL, key),
                BCR       = _get_spline_bc(gp.BCR, key)))
        end
        grid.kbasis.data[v] = Chebyshev1D(ChebyshevParameters(
            zmin = gp.kMin,
            zmax = gp.kMax,
            zDim = gp.kDim,
            bDim = gp.b_kDim,
            BCB  = _get_chebyshev_bc(gp.BCB, key),
            BCT  = _get_chebyshev_bc(gp.BCT, key)))
    end
    var_kmax = get(gp.max_wavenumber, "default", -1)
    for key in keys(gp.vars)
        var_kmax = get(gp.max_wavenumber, key, var_kmax)
    end

    for b in 1:gp.b_kDim
        _fill_fourier_rings_sph!(grid.jbasis.data, gp, mishpts, var_kmax, b)
    end
    return grid
end

# ────────────────────────────────────────────────────────────────────────────
# Fourier-based and Chebyshev-based creation functions
# ────────────────────────────────────────────────────────────────────────────

# Helper: kmax for Fourier i-dimension (use max_wavenumber or derive from b_iDim)
@inline function _fourier_kmax_i(gp::SpringsteelGridParameters, key)
    kmax = get(gp.max_wavenumber, key, get(gp.max_wavenumber, "default", -1))
    kmax < 0 && (kmax = (gp.b_iDim - 1) ÷ 2)
    return kmax
end

# Helper: kmax for Fourier j-dimension
@inline function _fourier_kmax_j(gp::SpringsteelGridParameters, key)
    kmax = get(gp.max_wavenumber, string(key, "_j"), get(gp.max_wavenumber, "default", -1))
    kmax < 0 && (kmax = (gp.b_jDim - 1) ÷ 2)
    return kmax
end

# 1D Cartesian Fourier (L; aliases: Ring1D)
function _create_cartesian_1d_fourier(gp::SpringsteelGridParameters)
    nvars   = length(values(gp.vars))
    rings   = Array{Fourier1D}(undef, nvars)
    ibasis  = FourierBasisArray(rings)
    jbasis  = NoBasisArray()
    kbasis  = NoBasisArray()
    spectral = zeros(Float64, gp.b_iDim, nvars)
    physical = zeros(Float64, gp.iDim, nvars, 3)

    grid = SpringsteelGrid{CartesianGeometry, typeof(ibasis), typeof(jbasis), typeof(kbasis)}(
        gp, ibasis, jbasis, kbasis, spectral, physical)

    for key in keys(gp.vars)
        v      = gp.vars[key]
        kmax_i = _fourier_kmax_i(gp, key)
        grid.ibasis.data[v] = Fourier1D(FourierParameters(
            ymin = gp.iMin,
            yDim = gp.iDim,
            kmax = kmax_i,
            bDim = 2 * kmax_i + 1))
    end
    return grid
end

# 2D Cartesian Fourier×Fourier (LL; aliases: Ring2D)
function _create_cartesian_2d_fourier2d(gp::SpringsteelGridParameters)
    nvars   = length(values(gp.vars))
    i_rings = Array{Fourier1D}(undef, nvars)
    j_rings = Array{Fourier1D}(undef, nvars)
    ibasis  = FourierBasisArray(i_rings)
    jbasis  = FourierBasisArray(j_rings)
    kbasis  = NoBasisArray()
    spectral = zeros(Float64, gp.b_iDim * gp.b_jDim, nvars)
    physical = zeros(Float64, gp.iDim * gp.jDim, nvars, 5)

    grid = SpringsteelGrid{CartesianGeometry, typeof(ibasis), typeof(jbasis), typeof(kbasis)}(
        gp, ibasis, jbasis, kbasis, spectral, physical)

    for key in keys(gp.vars)
        v      = gp.vars[key]
        kmax_i = _fourier_kmax_i(gp, key)
        kmax_j = _fourier_kmax_j(gp, key)
        grid.ibasis.data[v] = Fourier1D(FourierParameters(
            ymin = gp.iMin,
            yDim = gp.iDim,
            kmax = kmax_i,
            bDim = 2 * kmax_i + 1))
        grid.jbasis.data[v] = Fourier1D(FourierParameters(
            ymin = gp.jMin,
            yDim = gp.jDim,
            kmax = kmax_j,
            bDim = 2 * kmax_j + 1))
    end
    return grid
end

# 3D Cartesian Fourier×Fourier×Chebyshev (LLZ; aliases: DoublyPeriodic)
function _create_cartesian_3d_doublyperiodic(gp::SpringsteelGridParameters)
    nvars   = length(values(gp.vars))
    i_rings = Array{Fourier1D}(undef, nvars)
    j_rings = Array{Fourier1D}(undef, nvars)
    columns = Array{Chebyshev1D}(undef, nvars)
    ibasis  = FourierBasisArray(i_rings)
    jbasis  = FourierBasisArray(j_rings)
    kbasis  = ChebyshevBasisArray(columns)
    spectral = zeros(Float64, gp.b_iDim * gp.b_jDim * gp.b_kDim, nvars)
    physical = zeros(Float64, gp.iDim * gp.jDim * gp.kDim, nvars, 7)

    grid = SpringsteelGrid{CartesianGeometry, typeof(ibasis), typeof(jbasis), typeof(kbasis)}(
        gp, ibasis, jbasis, kbasis, spectral, physical)

    for key in keys(gp.vars)
        v      = gp.vars[key]
        kmax_i = _fourier_kmax_i(gp, key)
        kmax_j = _fourier_kmax_j(gp, key)
        grid.ibasis.data[v] = Fourier1D(FourierParameters(
            ymin = gp.iMin,
            yDim = gp.iDim,
            kmax = kmax_i,
            bDim = 2 * kmax_i + 1))
        grid.jbasis.data[v] = Fourier1D(FourierParameters(
            ymin = gp.jMin,
            yDim = gp.jDim,
            kmax = kmax_j,
            bDim = 2 * kmax_j + 1))
        grid.kbasis.data[v] = Chebyshev1D(ChebyshevParameters(
            zmin = gp.kMin,
            zmax = gp.kMax,
            zDim = gp.kDim,
            bDim = gp.b_kDim,
            BCB  = _get_chebyshev_bc(gp.BCB, key),
            BCT  = _get_chebyshev_bc(gp.BCT, key)))
    end
    return grid
end

# 1D Cartesian Chebyshev (Z; aliases: Column1D)
# BCL/BCR map to bottom/top (zmin/zmax) boundary conditions.
function _create_cartesian_1d_chebyshev(gp::SpringsteelGridParameters)
    nvars   = length(values(gp.vars))
    columns = Array{Chebyshev1D}(undef, nvars)
    ibasis  = ChebyshevBasisArray(columns)
    jbasis  = NoBasisArray()
    kbasis  = NoBasisArray()
    spectral = zeros(Float64, gp.b_iDim, nvars)
    physical = zeros(Float64, gp.iDim, nvars, 3)

    grid = SpringsteelGrid{CartesianGeometry, typeof(ibasis), typeof(jbasis), typeof(kbasis)}(
        gp, ibasis, jbasis, kbasis, spectral, physical)

    for key in keys(gp.vars)
        v = gp.vars[key]
        grid.ibasis.data[v] = Chebyshev1D(ChebyshevParameters(
            zmin = gp.iMin,
            zmax = gp.iMax,
            zDim = gp.iDim,
            bDim = gp.b_iDim,
            BCB  = _get_chebyshev_bc(gp.BCL, key),
            BCT  = _get_chebyshev_bc(gp.BCR, key)))
    end
    return grid
end

# 2D Cartesian Chebyshev×Chebyshev (ZZ; aliases: Column2D)
# i-dim BCs from BCL/BCR; j-dim BCs from BCU/BCD.
function _create_cartesian_2d_chebyshev2d(gp::SpringsteelGridParameters)
    nvars   = length(values(gp.vars))
    # b_jDim i-columns per j spectral mode
    i_cols  = Array{Chebyshev1D}(undef, gp.b_jDim, nvars)
    j_cols  = Array{Chebyshev1D}(undef, nvars)
    ibasis  = ChebyshevBasisArray(i_cols)
    jbasis  = ChebyshevBasisArray(j_cols)
    kbasis  = NoBasisArray()
    spectral = zeros(Float64, gp.b_iDim * gp.b_jDim, nvars)
    physical = zeros(Float64, gp.iDim * gp.jDim, nvars, 5)

    grid = SpringsteelGrid{CartesianGeometry, typeof(ibasis), typeof(jbasis), typeof(kbasis)}(
        gp, ibasis, jbasis, kbasis, spectral, physical)

    for key in keys(gp.vars)
        v = gp.vars[key]
        for j in 1:gp.b_jDim
            grid.ibasis.data[j, v] = Chebyshev1D(ChebyshevParameters(
                zmin = gp.iMin,
                zmax = gp.iMax,
                zDim = gp.iDim,
                bDim = gp.b_iDim,
                BCB  = _get_chebyshev_bc(gp.BCL, key),
                BCT  = _get_chebyshev_bc(gp.BCR, key)))
        end
        grid.jbasis.data[v] = Chebyshev1D(ChebyshevParameters(
            zmin = gp.jMin,
            zmax = gp.jMax,
            zDim = gp.jDim,
            bDim = gp.b_jDim,
            BCB  = _get_chebyshev_bc(gp.BCD, key),
            BCT  = _get_chebyshev_bc(gp.BCU, key)))
    end
    return grid
end

# 3D Cartesian Chebyshev×Chebyshev×Chebyshev (ZZZ; aliases: Column3D)
# i-dim BCs from BCL/BCR; j-dim from BCU/BCD; k-dim from BCB/BCT.
function _create_cartesian_3d_chebyshev3d(gp::SpringsteelGridParameters)
    nvars  = length(values(gp.vars))
    i_cols = Array{Chebyshev1D}(undef, gp.b_jDim, gp.b_kDim, nvars)
    j_cols = Array{Chebyshev1D}(undef, gp.b_kDim, nvars)
    k_cols = Array{Chebyshev1D}(undef, nvars)
    ibasis = ChebyshevBasisArray(i_cols)
    jbasis = ChebyshevBasisArray(j_cols)
    kbasis = ChebyshevBasisArray(k_cols)
    spectral = zeros(Float64, gp.b_iDim * gp.b_jDim * gp.b_kDim, nvars)
    physical = zeros(Float64, gp.iDim * gp.jDim * gp.kDim, nvars, 7)

    grid = SpringsteelGrid{CartesianGeometry, typeof(ibasis), typeof(jbasis), typeof(kbasis)}(
        gp, ibasis, jbasis, kbasis, spectral, physical)

    for key in keys(gp.vars)
        v = gp.vars[key]
        for j in 1:gp.b_jDim, z in 1:gp.b_kDim
            grid.ibasis.data[j, z, v] = Chebyshev1D(ChebyshevParameters(
                zmin = gp.iMin, zmax = gp.iMax,
                zDim = gp.iDim, bDim = gp.b_iDim,
                BCB  = _get_chebyshev_bc(gp.BCL, key), BCT = _get_chebyshev_bc(gp.BCR, key)))
        end
        for z in 1:gp.b_kDim
            grid.jbasis.data[z, v] = Chebyshev1D(ChebyshevParameters(
                zmin = gp.jMin, zmax = gp.jMax,
                zDim = gp.jDim, bDim = gp.b_jDim,
                BCB  = _get_chebyshev_bc(gp.BCD, key), BCT = _get_chebyshev_bc(gp.BCU, key)))
        end
        grid.kbasis.data[v] = Chebyshev1D(ChebyshevParameters(
            zmin = gp.kMin, zmax = gp.kMax,
            zDim = gp.kDim, bDim = gp.b_kDim,
            BCB  = _get_chebyshev_bc(gp.BCB, key), BCT = _get_chebyshev_bc(gp.BCT, key)))
    end
    return grid
end

# ────────────────────────────────────────────────────────────────────────────
# Unified public factory
# ────────────────────────────────────────────────────────────────────────────

"""
    createGrid(gp::SpringsteelGridParameters) -> SpringsteelGrid

Create a [`SpringsteelGrid`](@ref) from a [`SpringsteelGridParameters`](@ref) specification.
This overload handles all geometry types and replaces the separate `create_*_Grid` functions.

# Arguments
- `gp::SpringsteelGridParameters`: Fully or partially specified grid configuration.
  Derived dimensions (`jDim`, `b_jDim`, `kDim`, `b_kDim`) are computed automatically by
  [`compute_derived_params`](@ref) if not provided.

# Returns
A `SpringsteelGrid{G, I, J, K}` where the type parameters are determined by `gp.geometry`:

| `geometry` | Returned type |
|:---------- |:------------- |
| `"R"`, `"Spline1D"` | `SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray}` |
| `"RZ"` | `SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, ChebyshevBasisArray}` |
| `"RL"` | `SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray}` |
| `"RR"`, `"Spline2D"` | `SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, NoBasisArray}` |
| `"RLZ"` | `SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}` |
| `"RRR"` | `SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, SplineBasisArray}` |
| `"SL"` | `SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray}` |
| `"SLZ"` | `SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}` |

# Example
```julia
gp = SpringsteelGridParameters(
    geometry = "RL",
    num_cells = 10,
    iMin = 0.0,
    iMax = 100.0,
    vars = Dict("u" => 1),
    BCL = Dict("u" => CubicBSpline.R0),
    BCR = Dict("u" => CubicBSpline.R0))

grid = createGrid(gp)
# typeof(grid) == RL_Grid (a type alias for SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray})
```

See also: [`SpringsteelGridParameters`](@ref), [`parse_geometry`](@ref),
[`compute_derived_params`](@ref), [`SpringsteelGrid`](@ref)
"""
function createGrid(gp::SpringsteelGridParameters)
    gp_final = compute_derived_params(gp)
    geom     = _normalize_geometry(gp_final.geometry)

    # ── Spline-based ─────────────────────────────────────────────────────────────
    if geom in ("R", "Spline1D")
        return _create_cartesian_1d(gp_final)
    elseif geom == "RZ"
        return _create_cartesian_2d_rz(gp_final)
    elseif geom in ("RR", "Spline2D")
        return _create_cartesian_2d_rr(gp_final)
    elseif geom == "RRR"
        return _create_cartesian_3d_rrr(gp_final)
    elseif geom == "RL"
        return _create_cylindrical_2d_rl(gp_final)
    elseif geom == "RLZ"
        return _create_cylindrical_3d_rlz(gp_final)
    elseif geom == "SL"
        return _create_spherical_2d_sl(gp_final)
    elseif geom == "SLZ"
        return _create_spherical_3d_slz(gp_final)
    # ── Fourier-based (canonical: L, LL, LLZ) ───────────────────────────────────
    elseif geom == "L"
        return _create_cartesian_1d_fourier(gp_final)
    elseif geom == "LL"
        return _create_cartesian_2d_fourier2d(gp_final)
    elseif geom == "LLZ"
        return _create_cartesian_3d_doublyperiodic(gp_final)
    # ── Chebyshev-based (canonical: Z, ZZ, ZZZ) ─────────────────────────────────
    elseif geom == "Z"
        return _create_cartesian_1d_chebyshev(gp_final)
    elseif geom == "ZZ"
        return _create_cartesian_2d_chebyshev2d(gp_final)
    elseif geom == "ZZZ"
        return _create_cartesian_3d_chebyshev3d(gp_final)
    else
        throw(DomainError(gp.geometry,
            "Unknown geometry for SpringsteelGrid: $(gp.geometry)"))
    end
end

# ────────────────────────────────────────────────────────────────────────────
# _subgrid_for_solution — narrow a grid's params to a subset of variables
# ────────────────────────────────────────────────────────────────────────────

# Narrow a per-variable Dict (keyed by variable name, with optional "default"
# fallback) to just the entries for `var_names`. Missing keys fall back to
# the "default" value if present. The "default" key itself is preserved so
# downstream lookups still work for any variables added later.
function _narrow_var_dict(d::Dict, var_names::Vector{String})
    out = empty(d)
    for name in var_names
        if haskey(d, name)
            out[name] = d[name]
        elseif haskey(d, "default")
            out[name] = d["default"]
        end
    end
    haskey(d, "default") && (out["default"] = d["default"])
    return out
end

"""
    _subgrid_for_solution(grid, var_names::Vector{String}) -> SpringsteelGrid

Construct a new `SpringsteelGrid` that shares geometry and basis parameters
with `grid` but whose `physical` / `spectral` arrays hold only the variables
named in `var_names`. Each name is renumbered to a 1-based slot in the
returned grid.

Used by `solve` to package non-mutating solution results: the returned grid
is independent of `grid` (its `physical` / `spectral` arrays are fresh
allocations), so writing solution values into it never touches the input
grid. Basis template caching means construction is sub-millisecond for
typical grid sizes.
"""
function _subgrid_for_solution(grid::SpringsteelGrid, var_names::Vector{String})
    isempty(var_names) && throw(ArgumentError("var_names must not be empty"))
    p = grid.params
    for name in var_names
        haskey(p.vars, name) || throw(ArgumentError(
            "Variable `$name` not found in grid.params.vars"))
    end
    new_vars = Dict{String, Int}(name => i for (i, name) in enumerate(var_names))

    new_p = SpringsteelGridParameters(
        geometry       = p.geometry,
        iMin           = p.iMin,
        iMax           = p.iMax,
        num_cells      = p.num_cells,
        mubar          = p.mubar,
        quadrature     = p.quadrature,
        iDim           = p.iDim,
        b_iDim         = p.b_iDim,
        l_q            = _narrow_var_dict(p.l_q, var_names),
        BCL            = _narrow_var_dict(p.BCL, var_names),
        BCR            = _narrow_var_dict(p.BCR, var_names),
        jMin           = p.jMin,
        jMax           = p.jMax,
        max_wavenumber = _narrow_var_dict(p.max_wavenumber, var_names),
        jDim           = p.jDim,
        b_jDim         = p.b_jDim,
        BCU            = _narrow_var_dict(p.BCU, var_names),
        BCD            = _narrow_var_dict(p.BCD, var_names),
        kMin           = p.kMin,
        kMax           = p.kMax,
        kDim           = p.kDim,
        b_kDim         = p.b_kDim,
        BCB            = _narrow_var_dict(p.BCB, var_names),
        BCT            = _narrow_var_dict(p.BCT, var_names),
        vars           = new_vars,
        fourier_filter = _narrow_var_dict(p.fourier_filter, var_names),
        chebyshev_filter = _narrow_var_dict(p.chebyshev_filter, var_names),
        spectralIndexL = p.spectralIndexL,
        spectralIndexR = p.spectralIndexR,
        patchOffsetL   = p.patchOffsetL,
        patchOffsetR   = p.patchOffsetR,
        tile_num       = p.tile_num,
        i_regular_out  = p.i_regular_out,
        j_regular_out  = p.j_regular_out,
        k_regular_out  = p.k_regular_out,
    )

    return createGrid(new_p)
end

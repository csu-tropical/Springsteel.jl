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

    parse_geometry("R")        -> (CartesianGeometry(),   SplineBasisType(), NoBasisType(),      NoBasisType())
    parse_geometry("Spline1D") -> (CartesianGeometry(),   SplineBasisType(), NoBasisType(),      NoBasisType())
    parse_geometry("RZ")       -> (CartesianGeometry(),   SplineBasisType(), NoBasisType(),      ChebyshevBasisType())
    parse_geometry("RL")       -> (CylindricalGeometry(), SplineBasisType(), FourierBasisType(), NoBasisType())
    parse_geometry("RR")       -> (CartesianGeometry(),   SplineBasisType(), SplineBasisType(),  NoBasisType())
    parse_geometry("Spline2D") -> (CartesianGeometry(),   SplineBasisType(), SplineBasisType(),  NoBasisType())
    parse_geometry("RLZ")      -> (CylindricalGeometry(), SplineBasisType(), FourierBasisType(), ChebyshevBasisType())
    parse_geometry("RRR")      -> (CartesianGeometry(),   SplineBasisType(), SplineBasisType(),  SplineBasisType())
    parse_geometry("SL")       -> (SphericalGeometry(),   SplineBasisType(), FourierBasisType(), NoBasisType())
    parse_geometry("SLZ")      -> (SphericalGeometry(),   SplineBasisType(), FourierBasisType(), ChebyshevBasisType())

Map a geometry string to a 4-tuple of sentinel type instances `(G, It, Jt, Kt)` used as
type parameters of [`SpringsteelGrid`](@ref):
- `G`  — geometry sentinel (`CartesianGeometry`, `CylindricalGeometry`, `SphericalGeometry`)
- `It` — i-dimension basis sentinel (always `SplineBasisType`)
- `Jt` — j-dimension basis sentinel or `NoBasisType`
- `Kt` — k-dimension basis sentinel or `NoBasisType`

# Arguments
- `geometry::String`: Geometry identifier. Valid values: `"R"`, `"Spline1D"`, `"RZ"`, `"RL"`,
  `"RR"`, `"Spline2D"`, `"RLZ"`, `"RRR"`, `"SL"`, `"SLZ"`.

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
        "R"       => (CartesianGeometry(),   SplineBasisType(), NoBasisType(),      NoBasisType()),
        "Spline1D"=> (CartesianGeometry(),   SplineBasisType(), NoBasisType(),      NoBasisType()),
        "RZ"      => (CartesianGeometry(),   SplineBasisType(), NoBasisType(),      ChebyshevBasisType()),
        "RL"      => (CylindricalGeometry(), SplineBasisType(), FourierBasisType(), NoBasisType()),
        "RR"      => (CartesianGeometry(),   SplineBasisType(), SplineBasisType(),  NoBasisType()),
        "Spline2D"=> (CartesianGeometry(),   SplineBasisType(), SplineBasisType(),  NoBasisType()),
        "RLZ"     => (CylindricalGeometry(), SplineBasisType(), FourierBasisType(), ChebyshevBasisType()),
        "RRR"     => (CartesianGeometry(),   SplineBasisType(), SplineBasisType(),  SplineBasisType()),
        "SL"      => (SphericalGeometry(),   SplineBasisType(), FourierBasisType(), NoBasisType()),
        "SLZ"     => (SphericalGeometry(),   SplineBasisType(), FourierBasisType(), ChebyshevBasisType()),
    )
    haskey(mapping, geometry) ||
        throw(DomainError(geometry, "Unknown geometry for SpringsteelGrid: $geometry"))
    return mapping[geometry]
end

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
        l_q      = 2.0,
        BCL      = gp.BCL[first_key],
        BCR      = gp.BCR[first_key]))
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
        nc_j = Int64(gp.jDim / CubicBSpline.mubar)
    end
    return nc_j * CubicBSpline.mubar, nc_j + 3
end

# kDim, b_kDim for Cartesian Spline k (RRR)
function _cartesian_k_dims(gp::SpringsteelGridParameters)
    if gp.kDim == 0
        dk = gp.kMax - gp.kMin
        dx = gp.iMax - gp.iMin
        nc_k = Int64(ceil(gp.num_cells * (dk / dx)))
    else
        nc_k = Int64(gp.kDim / CubicBSpline.mubar)
    end
    return nc_k * CubicBSpline.mubar, nc_k + 3
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
    geom = gp.geometry

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

    else
        return gp
    end
end

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

    grid = SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray}(
        gp, ibasis, jbasis, kbasis, spectral, physical)

    for key in keys(gp.vars)
        var_l_q = get(gp.l_q, key, get(gp.l_q, "default", 2.0))
        grid.ibasis.data[1, gp.vars[key]] = Spline1D(SplineParameters(
            xmin      = gp.iMin,
            xmax      = gp.iMax,
            num_cells = gp.num_cells,
            l_q       = var_l_q,
            BCL       = gp.BCL[key],
            BCR       = gp.BCR[key]))
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

    grid = SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, ChebyshevBasisArray}(
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
                l_q       = var_l_q,
                BCL       = gp.BCL[key],
                BCR       = gp.BCR[key]))
        end
        grid.kbasis.data[v] = Chebyshev1D(ChebyshevParameters(
            zmin = gp.kMin,
            zmax = gp.kMax,
            zDim = gp.kDim,
            bDim = gp.b_kDim,
            BCB  = gp.BCB[key],
            BCT  = gp.BCT[key]))
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

    grid = SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, NoBasisArray}(
        gp, ibasis, jbasis, kbasis, spectral, physical)

    nc_j = Int64(gp.jDim / CubicBSpline.mubar)
    for key in keys(gp.vars)
        v = gp.vars[key]
        var_l_q_i = get(gp.l_q, key, get(gp.l_q, "default", 2.0))
        var_l_q_j = get(gp.l_q, string(key, "_j"), var_l_q_i)
        for j in 1:gp.b_jDim
            grid.ibasis.data[j, v] = Spline1D(SplineParameters(
                xmin      = gp.iMin,
                xmax      = gp.iMax,
                num_cells = gp.num_cells,
                l_q       = var_l_q_i,
                BCL       = gp.BCL[key],
                BCR       = gp.BCR[key]))
        end
        for r in 1:gp.iDim
            grid.jbasis.data[r, v] = Spline1D(SplineParameters(
                xmin      = gp.jMin,
                xmax      = gp.jMax,
                num_cells = nc_j,
                l_q       = var_l_q_j,
                BCL       = gp.BCU[key],
                BCR       = gp.BCD[key]))
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

    grid = SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, SplineBasisArray}(
        gp, ibasis, jbasis, kbasis, spectral, physical)

    nc_j = Int64(gp.jDim / CubicBSpline.mubar)
    nc_k = Int64(gp.kDim / CubicBSpline.mubar)
    for key in keys(gp.vars)
        v = gp.vars[key]
        var_l_q_i = get(gp.l_q, key, get(gp.l_q, "default", 2.0))
        var_l_q_j = get(gp.l_q, string(key, "_j"), var_l_q_i)
        var_l_q_k = get(gp.l_q, string(key, "_k"), var_l_q_i)
        for j in 1:gp.b_jDim, z in 1:gp.b_kDim
            grid.ibasis.data[j, z, v] = Spline1D(SplineParameters(
                xmin      = gp.iMin, xmax = gp.iMax,
                num_cells = gp.num_cells, l_q = var_l_q_i,
                BCL       = gp.BCL[key], BCR = gp.BCR[key]))
        end
        for r in 1:gp.iDim, z in 1:gp.b_kDim
            grid.jbasis.data[r, z, v] = Spline1D(SplineParameters(
                xmin      = gp.jMin, xmax = gp.jMax,
                num_cells = nc_j, l_q = var_l_q_j,
                BCL       = gp.BCU[key], BCR = gp.BCD[key]))
        end
        for r in 1:gp.iDim, l in 1:gp.jDim
            grid.kbasis.data[r, l, v] = Spline1D(SplineParameters(
                xmin      = gp.kMin, xmax = gp.kMax,
                num_cells = nc_k, l_q = var_l_q_k,
                BCL       = gp.BCB[key], BCR = gp.BCT[key]))
        end
    end
    return grid
end

# Internal helper: build cylindrical or spherical Fourier rings into pre-allocated array
function _fill_fourier_rings_cyl!(rings_arr, gp::SpringsteelGridParameters, var_kmax::Int, idx2::Int)
    iDim = gp.iDim
    for r in 1:iDim
        ri = r + gp.patchOffsetL
        lpoints, kmax_ring = _cyl_ring_dims(ri)
        if var_kmax >= 0; kmax_ring = min(var_kmax, ri); end
        dl     = 2π / lpoints
        offset = 0.5 * dl * (ri - 1)
        rings_arr[r, idx2] = Fourier1D(FourierParameters(
            ymin  = offset,
            yDim  = lpoints,
            kmax  = kmax_ring,
            bDim  = ri * 2 + 1))
    end
end

function _fill_fourier_rings_sph!(rings_arr, gp::SpringsteelGridParameters, mishpts, var_kmax::Int, idx2::Int)
    iDim   = gp.iDim
    max_ri = iDim + gp.patchOffsetL
    for r in 1:iDim
        theta = mishpts[r]
        lpoints, kmax_ring = _sph_ring_dims(theta, max_ri)
        if var_kmax >= 0; kmax_ring = min(var_kmax, kmax_ring); end
        dl     = 2π / lpoints
        offset = 0.5 * dl * r
        rings_arr[r, idx2] = Fourier1D(FourierParameters(
            ymin  = offset,
            yDim  = lpoints,
            kmax  = kmax_ring,
            bDim  = 1 + 2 * kmax_ring))
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

    spectral = zeros(Float64, gp.b_jDim, nvars)
    physical = zeros(Float64, gp.jDim, nvars, 5)

    grid = SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray}(
        gp, ibasis, jbasis, kbasis, spectral, physical)

    for key in keys(gp.vars)
        v = gp.vars[key]
        var_l_q = get(gp.l_q, key, get(gp.l_q, "default", 2.0))
        for i in 1:3
            grid.ibasis.data[i, v] = Spline1D(SplineParameters(
                xmin      = gp.iMin,
                xmax      = gp.iMax,
                num_cells = gp.num_cells,
                l_q       = var_l_q,
                BCL       = gp.BCL[key],
                BCR       = gp.BCR[key]))
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

    grid = SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}(
        gp, ibasis, jbasis, kbasis, spectral, physical)

    for key in keys(gp.vars)
        v = gp.vars[key]
        var_l_q = get(gp.l_q, key, get(gp.l_q, "default", 2.0))
        for z in 1:gp.b_kDim
            grid.ibasis.data[z, v] = Spline1D(SplineParameters(
                xmin      = gp.iMin,
                xmax      = gp.iMax,
                num_cells = gp.num_cells,
                l_q       = var_l_q,
                BCL       = gp.BCL[key],
                BCR       = gp.BCR[key]))
        end
        grid.kbasis.data[v] = Chebyshev1D(ChebyshevParameters(
            zmin = gp.kMin,
            zmax = gp.kMax,
            zDim = gp.kDim,
            bDim = gp.b_kDim,
            BCB  = gp.BCB[key],
            BCT  = gp.BCT[key]))
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

    spectral = zeros(Float64, gp.b_jDim, nvars)
    physical = zeros(Float64, gp.jDim, nvars, 5)

    grid = SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray}(
        gp, ibasis, jbasis, kbasis, spectral, physical)

    for key in keys(gp.vars)
        v = gp.vars[key]
        var_l_q = get(gp.l_q, key, get(gp.l_q, "default", 2.0))
        for i in 1:3
            grid.ibasis.data[i, v] = Spline1D(SplineParameters(
                xmin      = gp.iMin,
                xmax      = gp.iMax,
                num_cells = gp.num_cells,
                l_q       = var_l_q,
                BCL       = gp.BCL[key],
                BCR       = gp.BCR[key]))
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

    grid = SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}(
        gp, ibasis, jbasis, kbasis, spectral, physical)

    for key in keys(gp.vars)
        v = gp.vars[key]
        var_l_q = get(gp.l_q, key, get(gp.l_q, "default", 2.0))
        for z in 1:gp.b_kDim
            grid.ibasis.data[z, v] = Spline1D(SplineParameters(
                xmin      = gp.iMin,
                xmax      = gp.iMax,
                num_cells = gp.num_cells,
                l_q       = var_l_q,
                BCL       = gp.BCL[key],
                BCR       = gp.BCR[key]))
        end
        grid.kbasis.data[v] = Chebyshev1D(ChebyshevParameters(
            zmin = gp.kMin,
            zmax = gp.kMax,
            zDim = gp.kDim,
            bDim = gp.b_kDim,
            BCB  = gp.BCB[key],
            BCT  = gp.BCT[key]))
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
# typeof(grid) == RL_Grid (once Phase 10 activates the alias)
# typeof(grid) == SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray}
```

See also: [`SpringsteelGridParameters`](@ref), [`parse_geometry`](@ref),
[`compute_derived_params`](@ref), [`SpringsteelGrid`](@ref)
"""
function createGrid(gp::SpringsteelGridParameters)
    gp_final = compute_derived_params(gp)
    geom     = gp_final.geometry

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
    else
        throw(DomainError(gp.geometry,
            "Unknown geometry for SpringsteelGrid: $(gp.geometry)"))
    end
end

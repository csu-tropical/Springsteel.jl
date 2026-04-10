# transforms_cartesian.jl — Spectral ↔ physical transforms for Cartesian SpringsteelGrids
#
# Covers:
#   • 1D Cartesian (SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray})
#   • 2D Cartesian (Spline×Spline = RR_Grid, and Spline×Chebyshev = RZ_Grid)
#   • 3D Cartesian (Spline×Spline×Spline = RRR_Grid)
#
# Provides:
#   • getGridpoints          — physical mish-point locations
#   • spectralTransform!     — physical → spectral (in-place, grid's own arrays)
#   • spectralTransform      — physical → spectral (explicit-array variant)
#   • gridTransform!         — spectral → physical + derivatives (in-place)
#   • gridTransform          — spectral → physical + derivatives (explicit-array variant)
#
# Transform order convention (forward): SBtransform on the i-dimension per variable.
# Transform order convention (inverse): SAtransform! → SItransform, SIxtransform, SIxxtransform.
#
# Must be included AFTER types.jl, basis_interface.jl, and factory.jl.

# ── Type alias for brevity ────────────────────────────────────────────────────
# 1D Cartesian: only the i-dimension is active; j and k slots are NoBasisArray.
const _1DCartesianGrid = SpringsteelGrid{CartesianGeometry, SplineBasisArray{2}, NoBasisArray, NoBasisArray}

# ────────────────────────────────────────────────────────────────────────────
# getGridpoints
# ────────────────────────────────────────────────────────────────────────────

"""
    getGridpoints(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray}) -> Vector{Float64}

Return the physical mish-point locations for the 1-D Cartesian spline grid.

All variables share the same radial domain, so the mish-points from the first
variable's `Spline1D` object are returned as the canonical gridpoint array.

# Arguments
- `grid`: A 1-D Cartesian [`SpringsteelGrid`](@ref) (type alias [`R_Grid`](@ref)).

# Returns
A `Vector{Float64}` of length `grid.params.iDim` containing the Gaussian quadrature
(`mish`) evaluation points, monotonically increasing from near `iMin` to near `iMax`.

# Example
```julia
gp   = SpringsteelGridParameters(geometry="R", num_cells=10, iMin=0.0, iMax=10.0,
                                  vars=Dict("u"=>1),
                                  BCL=Dict("u"=>CubicBSpline.R0),
                                  BCR=Dict("u"=>CubicBSpline.R0))
grid = createGrid(gp)
pts  = getGridpoints(grid)
length(pts) == grid.params.iDim   # true
```

See also: [`spectralTransform!`](@ref), [`gridTransform!`](@ref)
"""
function getGridpoints(grid::_1DCartesianGrid)
    return grid.ibasis.data[1, 1].mishPoints
end

# ────────────────────────────────────────────────────────────────────────────
# getRegularGridpoints — 1D Cartesian
# ────────────────────────────────────────────────────────────────────────────

"""
    getRegularGridpoints(grid::SpringsteelGrid{G, SplineBasisArray, NoBasisArray, NoBasisArray}) -> Vector{Float64}

Return `i_regular_out` uniformly-spaced output locations spanning `[iMin, iMax]`.

Unlike [`getGridpoints`](@ref), which returns the (unevenly-spaced) Gaussian mish
points, this function returns a regular grid suitable for visualisation or file I/O.
The number of points is `grid.params.i_regular_out` (default: `num_cells + 1`).

# Arguments
- `grid`: A 1-D Cartesian [`SpringsteelGrid`](@ref) (`R_Grid` / `Spline1D_Grid`).

# Returns
A `Vector{Float64}` of length `grid.params.i_regular_out` uniformly spaced from
`grid.params.iMin` to `grid.params.iMax`.

# Example
```julia
gp   = SpringsteelGridParameters(geometry="Spline1D", num_cells=10,
                                  iMin=0.0, iMax=10.0,
                                  vars=Dict("u"=>1),
                                  BCL=Dict("u"=>CubicBSpline.R0),
                                  BCR=Dict("u"=>CubicBSpline.R0))
grid = createGrid(gp)
pts  = getRegularGridpoints(grid)  # 11 evenly-spaced points in [0, 10]
```

See also: [`regularGridTransform`](@ref), [`getGridpoints`](@ref), [`write_grid`](@ref)
"""
function getRegularGridpoints(grid::_1DCartesianGrid)
    n  = grid.params.i_regular_out
    x0 = grid.params.iMin
    x1 = grid.params.iMax
    dx = (x1 - x0) / (n - 1)
    return [min(x0 + (i - 1) * dx, x1) for i in 1:n]
end

# ────────────────────────────────────────────────────────────────────────────
# regularGridTransform — 1D Cartesian
# ────────────────────────────────────────────────────────────────────────────

"""
    regularGridTransform(grid::SpringsteelGrid{G, SplineBasisArray, NoBasisArray, NoBasisArray}, gridpoints::AbstractVector{Float64}) -> Array{Float64}

Evaluate the B-spline representations at arbitrary output locations `gridpoints`,
returning field values and first/second derivatives.

Applies `SAtransform!` to the current `grid.spectral` coefficients (so the
grid's spectral array must be up-to-date), then evaluates the field and its
first and second derivatives at every point in `gridpoints`.

# Arguments
- `grid`: A 1-D Cartesian [`SpringsteelGrid`](@ref).
- `gridpoints`: Output evaluation locations; typically from
  [`getRegularGridpoints`](@ref) but any points within `[iMin, iMax]` are valid.

# Returns
An `Array{Float64}` of shape `(length(gridpoints), nvars, 3)` where the third axis is:
- `[:, :, 1]` — field values
- `[:, :, 2]` — first derivatives (∂f/∂r)
- `[:, :, 3]` — second derivatives (∂²f/∂r²)

# Example
```julia
spectralTransform!(grid)
reg_pts  = getRegularGridpoints(grid)
reg_phys = regularGridTransform(grid, reg_pts)
```

See also: [`getRegularGridpoints`](@ref), [`gridTransform!`](@ref), [`write_grid`](@ref)
"""
function regularGridTransform(grid::_1DCartesianGrid, gridpoints::AbstractVector{Float64})
    nvars   = length(grid.params.vars)
    gpts    = collect(Float64, gridpoints)
    physical = zeros(Float64, length(gpts), nvars, 3)
    for v in 1:nvars
        spline = grid.ibasis.data[1, v]
        spline.b .= view(grid.spectral, :, v)
        SAtransform!(spline)
        SItransform(spline, gpts, view(physical, :, v, 1))
        SIxtransform(spline, gpts, view(physical, :, v, 2))
        SIxxtransform(spline, gpts, view(physical, :, v, 3))
    end
    return physical
end

# ────────────────────────────────────────────────────────────────────────────

"""
    spectralTransform(grid, physical, spectral)

Explicit-array helper for the 1-D Cartesian forward transform.

Applies `SBtransform` to each variable's physical values and writes the
resulting B-spline coefficients into `spectral`.  Both `physical` and
`spectral` are caller-supplied arrays, so this is safe to use in distributed
workflows where the grid's own arrays are not used directly.

# Arguments
- `grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray}`:
  Grid providing the `Spline1D` basis objects.
- `physical::Array{Float64}`: Shape `(iDim, nvars, 3)`.  Only the `[:, :, 1]` slice
  (field values) is read; derivative slots are ignored.
- `spectral::Array{Float64}`: Destination array of shape `(b_iDim, nvars)`.

# Returns
`spectral` (mutated in-place).

See also: [`spectralTransform!`](@ref), [`gridTransform`](@ref)
"""
function spectralTransform(
        grid     :: _1DCartesianGrid,
        physical :: Array{real},
        spectral :: Array{real})
    nvars = size(spectral, 2)
    for v in 1:nvars
        b = SBtransform(grid.ibasis.data[1, v], physical[:, v, 1])
        spectral[:, v] .= b
    end
    return spectral
end

# ────────────────────────────────────────────────────────────────────────────
# spectralTransform! (in-place, uses grid's own arrays)
# ────────────────────────────────────────────────────────────────────────────

"""
    spectralTransform!(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray})

Transform field values from physical to spectral space for a 1-D Cartesian grid.

Reads `grid.physical[:, :, 1]` (field values for all variables), computes
B-spline coefficients via `SBtransform`, and writes results into `grid.spectral`.

# Arguments
- `grid`: A 1-D Cartesian [`SpringsteelGrid`](@ref).

# Returns
`grid.spectral` (the modified in-place spectral array).

# Example
```julia
# Fill physical values
pts = getGridpoints(grid)
for i in eachindex(pts)
    grid.physical[i, 1, 1] = sin(pts[i])
end

# Forward transform
spectralTransform!(grid)

# grid.spectral now holds B-spline coefficients
```

See also: [`gridTransform!`](@ref), [`spectralTransform`](@ref)
"""
function spectralTransform!(grid::_1DCartesianGrid)
    spectralTransform(grid, grid.physical, grid.spectral)
    applyFilter!(grid)
    return grid.spectral
end

# ────────────────────────────────────────────────────────────────────────────
# gridTransform (explicit-array helper)
# ────────────────────────────────────────────────────────────────────────────

"""
    gridTransform(grid, physical, spectral)

Explicit-array helper for the 1-D Cartesian inverse transform.

For each variable: solves `SAtransform!` (B → A coefficients), then evaluates
`SItransform`, `SIxtransform`, and `SIxxtransform` at the mish points, writing
the field value and its first and second derivatives into `physical[:, v, 1]`,
`physical[:, v, 2]`, and `physical[:, v, 3]` respectively.

# Arguments
- `grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray}`:
  Grid providing the `Spline1D` basis objects.
- `physical::Array{Float64}`: Destination array of shape `(iDim, nvars, 3)`.
  Slots `[:, :, 1]` (value), `[:, :, 2]` (∂/∂i), `[:, :, 3]` (∂²/∂i²) are all written.
- `spectral::Array{Float64}`: Source B-spline coefficient array of shape `(b_iDim, nvars)`.

# Returns
`physical` (mutated in-place).

See also: [`gridTransform!`](@ref), [`spectralTransform`](@ref)
"""
function gridTransform(
        grid     :: _1DCartesianGrid,
        physical :: Array{real},
        spectral :: Array{real})
    pts   = getGridpoints(grid)
    nvars = size(spectral, 2)
    for v in 1:nvars
        spline = grid.ibasis.data[1, v]
        spline.b .= view(spectral, :, v)
        SAtransform!(spline)
        SItransform(spline,   pts, view(physical, :, v, 1))
        SIxtransform(spline,  pts, view(physical, :, v, 2))
        SIxxtransform(spline, pts, view(physical, :, v, 3))
    end
    return physical
end

# ────────────────────────────────────────────────────────────────────────────
# gridTransform! (in-place, uses grid's own arrays)
# ────────────────────────────────────────────────────────────────────────────

"""
    gridTransform!(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray})

Transform from spectral space to physical space with derivatives for a 1-D Cartesian grid.

Reads `grid.spectral`, solves the B→A system, and evaluates the spline representation
at all mish points, writing values and derivatives into `grid.physical`.

After this call:
- `grid.physical[:, v, 1]` — field values
- `grid.physical[:, v, 2]` — first derivatives ∂f/∂i
- `grid.physical[:, v, 3]` — second derivatives ∂²f/∂i²

# Arguments
- `grid`: A 1-D Cartesian [`SpringsteelGrid`](@ref).

# Returns
`grid.physical` (the modified in-place physical array).

# Example
```julia
spectralTransform!(grid)     # forward transform first
gridTransform!(grid)         # inverse transform with derivatives

values      = grid.physical[:, 1, 1]
first_deriv = grid.physical[:, 1, 2]
second_deriv= grid.physical[:, 1, 3]
```

See also: [`spectralTransform!`](@ref), [`gridTransform`](@ref)
"""
function gridTransform!(grid::_1DCartesianGrid)
    gridTransform(grid, grid.physical, grid.spectral)
    return grid.physical
end

# ═══════════════════════════════════════════════════════════════════════════
# 2D Cartesian Transforms
# ═══════════════════════════════════════════════════════════════════════════

# ── Type aliases for brevity ──────────────────────────────────────────────

# 2D Cartesian Spline×Spline (RR):  i=Spline, j=Spline, k=NoBasis
const _2DCartesianRR = SpringsteelGrid{CartesianGeometry, SplineBasisArray{2}, SplineBasisArray{2}, NoBasisArray}

# 2D Cartesian Spline×Chebyshev (RZ): i=Spline, j=NoBasis, k=Chebyshev
const _2DCartesianRZ = SpringsteelGrid{CartesianGeometry, SplineBasisArray{2}, NoBasisArray, ChebyshevBasisArray{1}}

# ────────────────────────────────────────────────────────────────────────────
# getGridpoints — 2D grids
# ────────────────────────────────────────────────────────────────────────────

"""
    getGridpoints(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, NoBasisArray}) -> Matrix{Float64}
    getGridpoints(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, ChebyshevBasisArray}) -> Matrix{Float64}

Return a `(npoints, 2)` matrix of physical grid coordinates for a 2-D Cartesian grid.

Column 1 contains the i-direction (spline) coordinates; column 2 contains the j-direction
(spline, for RR) or k-direction (Chebyshev, for RZ) coordinates.  Points are ordered with
the j/k index varying fastest:  index `(r-1)*jDim + l` corresponds to `(r, l)`.

See also: [`spectralTransform!`](@ref), [`gridTransform!`](@ref)
"""
function getGridpoints(grid::_2DCartesianRR)
    iDim = grid.params.iDim
    jDim = grid.params.jDim
    pts  = zeros(Float64, iDim * jDim, 2)
    g = 1
    for r in 1:iDim
        xi = grid.ibasis.data[1, 1].mishPoints[r]
        for l in 1:jDim
            pts[g, 1] = xi
            pts[g, 2] = grid.jbasis.data[r, 1].mishPoints[l]
            g += 1
        end
    end
    return pts
end

function getGridpoints(grid::_2DCartesianRZ)
    iDim = grid.params.iDim
    kDim = grid.params.kDim
    pts  = zeros(Float64, iDim * kDim, 2)
    g = 1
    for r in 1:iDim
        xi = grid.ibasis.data[1, 1].mishPoints[r]
        for z in 1:kDim
            pts[g, 1] = xi
            pts[g, 2] = grid.kbasis.data[1].mishPoints[z]
            g += 1
        end
    end
    return pts
end

# ────────────────────────────────────────────────────────────────────────────
# spectralTransform / spectralTransform!  — 2D Cartesian Spline×Spline (RR)
# ────────────────────────────────────────────────────────────────────────────

"""
    spectralTransform(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, NoBasisArray}, physical, spectral)
    spectralTransform!(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, NoBasisArray})

Forward transform (physical → spectral) for a 2-D Cartesian Spline×Spline grid.

**Transform order** (physical → spectral):
1. j-direction `SBtransform!` for each i gridpoint → temporary buffer `[b_jDim, iDim]`
2. i-direction `SBtransform!` for each j spectral mode → `spectral[…, v]`

**Spectral layout**: consecutive `b_iDim`-element blocks, one per j-mode:
`spectral[(l-1)*b_iDim+1 : l*b_iDim, v]` holds the i-direction B-coefficients for
j-mode `l`.

**Physical layout**: `(r-1)*jDim + l` is the flat index for i-gridpoint `r`, j-gridpoint `l`.

See also: [`gridTransform!`](@ref)
"""
function spectralTransform(
        grid     :: _2DCartesianRR,
        physical :: Array{real},
        spectral :: Array{real})
    iDim  = grid.params.iDim
    jDim  = grid.params.jDim
    b_iDim = grid.params.b_iDim
    b_jDim = grid.params.b_jDim
    nvars = size(spectral, 2)
    tempsb = _scratch(grid).tempsb

    for v in 1:nvars
        # Step 1: j-direction transform for each i gridpoint
        for r in 1:iDim
            jsp = grid.jbasis.data[r, v]
            @inbounds for l in 1:jDim
                jsp.uMish[l] = physical[(r-1)*jDim + l, v, 1]
            end
            SBtransform!(jsp)
            @inbounds for k in 1:b_jDim
                tempsb[k, r] = jsp.b[k]
            end
        end

        # Step 2: i-direction transform for each j spectral coefficient
        for l in 1:b_jDim
            isp = grid.ibasis.data[l, v]
            @inbounds for r in 1:iDim
                isp.uMish[r] = tempsb[l, r]
            end
            SBtransform!(isp)
            r1 = (l-1)*b_iDim + 1
            @inbounds for k in 0:(b_iDim - 1)
                spectral[r1 + k, v] = isp.b[k + 1]
            end
        end
    end
    return spectral
end

"""
    spectralTransform!(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, NoBasisArray})

In-place forward transform for a 2-D Cartesian Spline×Spline grid.  Reads
`grid.physical[:, :, 1]` and writes `grid.spectral`.

See also: [`spectralTransform`](@ref), [`gridTransform!`](@ref)
"""
function spectralTransform!(grid::_2DCartesianRR)
    spectralTransform(grid, grid.physical, grid.spectral)
    applyFilter!(grid)
    return grid.spectral
end

# ────────────────────────────────────────────────────────────────────────────
# gridTransform / gridTransform!  — 2D Cartesian Spline×Spline (RR)
# ────────────────────────────────────────────────────────────────────────────

"""
    gridTransform(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, NoBasisArray}, physical, spectral)
    gridTransform!(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, NoBasisArray})

Inverse transform (spectral → physical + derivatives) for a 2-D Cartesian Spline×Spline grid.

**Transform order** (spectral → physical):
1. i-direction `SAtransform!` → `SItransform!` / `SIxtransform` / `SIxxtransform`
   for each j-mode → spline buffer `[iDim, b_jDim]`
2. j-direction `SAtransform!` → `SItransform!` / `SIxtransform` / `SIxxtransform`
   for each i gridpoint → physical slots

**Physical derivative layout** (5 slots):
- `physical[:, v, 1]` — field values
- `physical[:, v, 2]` — ∂f/∂i (first derivative, i-direction)
- `physical[:, v, 3]` — ∂²f/∂i² (second derivative, i-direction)
- `physical[:, v, 4]` — ∂f/∂j (first derivative, j-direction)
- `physical[:, v, 5]` — ∂²f/∂j² (second derivative, j-direction)

See also: [`spectralTransform!`](@ref)
"""
function gridTransform(
        grid     :: _2DCartesianRR,
        physical :: Array{real},
        spectral :: Array{real})
    iDim  = grid.params.iDim
    jDim  = grid.params.jDim
    b_iDim = grid.params.b_iDim
    b_jDim = grid.params.b_jDim
    nvars = size(spectral, 2)
    s = _scratch(grid)
    splineBuffer = s.splineBuffer
    spline_scratch = s.spline_scratch

    for v in 1:nvars
        for dr in 0:2
            # i-direction inverse transform per j-spectral mode
            for l in 1:b_jDim
                r1 = (l-1)*b_iDim + 1
                r2 = r1 + b_iDim - 1
                isp = grid.ibasis.data[l, v]
                copyto!(isp.b, view(spectral, r1:r2, v))
                SAtransform!(isp)
                if dr == 0
                    SItransform!(isp)
                    @inbounds for r in 1:iDim
                        splineBuffer[r, l] = isp.uMish[r]
                    end
                elseif dr == 1
                    SIxtransform(isp, spline_scratch)
                    @inbounds for r in 1:iDim
                        splineBuffer[r, l] = spline_scratch[r]
                    end
                else
                    SIxxtransform(isp, spline_scratch)
                    @inbounds for r in 1:iDim
                        splineBuffer[r, l] = spline_scratch[r]
                    end
                end
            end

            # j-direction inverse transform per i gridpoint
            for r in 1:iDim
                jsp = grid.jbasis.data[r, v]
                @inbounds for l in 1:b_jDim
                    jsp.b[l] = splineBuffer[r, l]
                end
                SAtransform!(jsp)
                SItransform!(jsp)
                l1 = (r-1)*jDim + 1
                l2 = l1 + jDim - 1
                if dr == 0
                    copyto!(view(physical, l1:l2, v, 1), jsp.uMish)
                    # Reuse jsp.uMish as scratch — its prior content was just copied above.
                    SIxtransform(jsp, jsp.uMish)
                    copyto!(view(physical, l1:l2, v, 4), jsp.uMish)
                    SIxxtransform(jsp, jsp.uMish)
                    copyto!(view(physical, l1:l2, v, 5), jsp.uMish)
                elseif dr == 1
                    copyto!(view(physical, l1:l2, v, 2), jsp.uMish)
                else
                    copyto!(view(physical, l1:l2, v, 3), jsp.uMish)
                end
            end
        end
    end
    return physical
end

"""
    gridTransform!(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, NoBasisArray})

In-place inverse transform for a 2-D Cartesian Spline×Spline grid.  Reads
`grid.spectral` and writes `grid.physical` (values + derivatives in all 5 slots).

See also: [`gridTransform`](@ref), [`spectralTransform!`](@ref)
"""
function gridTransform!(grid::_2DCartesianRR)
    gridTransform(grid, grid.physical, grid.spectral)
    return grid.physical
end

# ────────────────────────────────────────────────────────────────────────────
# spectralTransform / spectralTransform!  — 2D Cartesian Spline×Chebyshev (RZ)
# ────────────────────────────────────────────────────────────────────────────

"""
    spectralTransform(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, ChebyshevBasisArray}, physical, spectral)
    spectralTransform!(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, ChebyshevBasisArray})

Forward transform (physical → spectral) for a 2-D Cartesian Spline×Chebyshev grid.

**Transform order** (physical → spectral):
1. k-direction `CBtransform!` for each i gridpoint → temporary buffer `[b_kDim, iDim]`
2. i-direction `SBtransform!` for each k spectral mode → `spectral[…, v]`

**Spectral layout**: consecutive `b_iDim`-element blocks, one per k-mode:
`spectral[(z-1)*b_iDim+1 : z*b_iDim, v]` holds the i-direction B-coefficients for
Chebyshev mode `z`.

**Physical layout**: `(r-1)*kDim + z` is the flat index for i-gridpoint `r`, k-gridpoint `z`.

See also: [`gridTransform!`](@ref)
"""
function spectralTransform(
        grid     :: _2DCartesianRZ,
        physical :: Array{real},
        spectral :: Array{real})
    iDim  = grid.params.iDim
    kDim  = grid.params.kDim
    b_iDim = grid.params.b_iDim
    b_kDim = grid.params.b_kDim
    nvars = size(spectral, 2)
    tempcb = _scratch(grid).tempcb

    for v in 1:nvars
        # Step 1: k-direction (Chebyshev) transform for each i gridpoint
        kcol = grid.kbasis.data[v]
        for r in 1:iDim
            @inbounds for z in 1:kDim
                kcol.uMish[z] = physical[(r-1)*kDim + z, v, 1]
            end
            CBtransform!(kcol)
            @inbounds for k in 1:b_kDim
                tempcb[k, r] = kcol.b[k]
            end
        end

        # Step 2: i-direction (spline) transform for each k spectral mode
        for z in 1:b_kDim
            isp = grid.ibasis.data[z, v]
            @inbounds for r in 1:iDim
                isp.uMish[r] = tempcb[z, r]
            end
            SBtransform!(isp)
            r1 = (z-1)*b_iDim + 1
            @inbounds for k in 0:(b_iDim - 1)
                spectral[r1 + k, v] = isp.b[k + 1]
            end
        end
    end
    return spectral
end

"""
    spectralTransform!(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, ChebyshevBasisArray})

In-place forward transform for a 2-D Cartesian Spline×Chebyshev grid.  Reads
`grid.physical[:, :, 1]` and writes `grid.spectral`.

See also: [`spectralTransform`](@ref), [`gridTransform!`](@ref)
"""
function spectralTransform!(grid::_2DCartesianRZ)
    spectralTransform(grid, grid.physical, grid.spectral)
    applyFilter!(grid)
    return grid.spectral
end

# ────────────────────────────────────────────────────────────────────────────
# gridTransform / gridTransform!  — 2D Cartesian Spline×Chebyshev (RZ)
# ────────────────────────────────────────────────────────────────────────────

"""
    gridTransform(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, ChebyshevBasisArray}, physical, spectral)
    gridTransform!(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, ChebyshevBasisArray})

Inverse transform (spectral → physical + derivatives) for a 2-D Cartesian Spline×Chebyshev grid.

**Transform order** (spectral → physical):
1. i-direction `SAtransform!` → `SItransform!` / `SIxtransform` / `SIxxtransform`
   for each k-mode → spline buffer `[iDim, b_kDim]`
2. k-direction `CAtransform!` → `CItransform!` / `CIxtransform` / `CIxxtransform`
   for each i gridpoint → physical slots

**Physical derivative layout** (5 slots):
- `physical[:, v, 1]` — field values
- `physical[:, v, 2]` — ∂f/∂i (first derivative, i-direction / spline)
- `physical[:, v, 3]` — ∂²f/∂i² (second derivative, i-direction)
- `physical[:, v, 4]` — ∂f/∂k (first derivative, k-direction / Chebyshev)
- `physical[:, v, 5]` — ∂²f/∂k² (second derivative, k-direction)

See also: [`spectralTransform!`](@ref)
"""
function gridTransform(
        grid     :: _2DCartesianRZ,
        physical :: Array{real},
        spectral :: Array{real})
    iDim  = grid.params.iDim
    kDim  = grid.params.kDim
    b_iDim = grid.params.b_iDim
    b_kDim = grid.params.b_kDim
    nvars = size(spectral, 2)
    s = _scratch(grid)
    splineBuffer = s.splineBuffer
    spline_scratch = s.spline_scratch

    for v in 1:nvars
        for dr in 0:2
            # i-direction inverse transform per k-spectral mode
            for z in 1:b_kDim
                r1 = (z-1)*b_iDim + 1
                r2 = r1 + b_iDim - 1
                isp = grid.ibasis.data[z, v]
                copyto!(isp.b, view(spectral, r1:r2, v))
                SAtransform!(isp)
                if dr == 0
                    SItransform!(isp)
                    @inbounds for r in 1:iDim
                        splineBuffer[r, z] = isp.uMish[r]
                    end
                elseif dr == 1
                    SIxtransform(isp, spline_scratch)
                    @inbounds for r in 1:iDim
                        splineBuffer[r, z] = spline_scratch[r]
                    end
                else
                    SIxxtransform(isp, spline_scratch)
                    @inbounds for r in 1:iDim
                        splineBuffer[r, z] = spline_scratch[r]
                    end
                end
            end

            # k-direction inverse transform per i gridpoint
            kcol = grid.kbasis.data[v]
            for r in 1:iDim
                @inbounds for z in 1:b_kDim
                    kcol.b[z] = splineBuffer[r, z]
                end
                CAtransform!(kcol)
                CItransform!(kcol)
                z1 = (r-1)*kDim + 1
                z2 = z1 + kDim - 1
                if dr == 0
                    copyto!(view(physical, z1:z2, v, 1), kcol.uMish)
                    # Reuse kcol.uMish as scratch — its prior content was just copied above.
                    CIxtransform(kcol, kcol.uMish)
                    copyto!(view(physical, z1:z2, v, 4), kcol.uMish)
                    CIxxtransform(kcol, kcol.uMish)
                    copyto!(view(physical, z1:z2, v, 5), kcol.uMish)
                elseif dr == 1
                    copyto!(view(physical, z1:z2, v, 2), kcol.uMish)
                else
                    copyto!(view(physical, z1:z2, v, 3), kcol.uMish)
                end
            end
        end
    end
    return physical
end

"""
    gridTransform!(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, ChebyshevBasisArray})

In-place inverse transform for a 2-D Cartesian Spline×Chebyshev grid.  Reads
`grid.spectral` and writes `grid.physical` (values + derivatives in all 5 slots).

See also: [`gridTransform`](@ref), [`spectralTransform!`](@ref)
"""
function gridTransform!(grid::_2DCartesianRZ)
    gridTransform(grid, grid.physical, grid.spectral)
    return grid.physical
end

# ═══════════════════════════════════════════════════════════════════════════
# Regular-grid output — 2D Cartesian Spline×Spline (RR)
# ═══════════════════════════════════════════════════════════════════════════

"""
    getRegularGridpoints(grid::_2DCartesianRR) -> Matrix{Float64}

Return an `(n_i × n_j, 2)` matrix of uniformly-spaced `(x, y)` coordinates
for a 2-D Cartesian Spline×Spline (RR) grid.

Unlike [`getGridpoints`](@ref), which returns the unevenly-spaced Gaussian mish
points, this function returns a regular tensor-product grid for visualisation
and file I/O.

The output dimensions are controlled by [`SpringsteelGridParameters`](@ref) fields:
- `i_regular_out` — x-points in `[iMin, iMax]`   (default `num_cells + 1`)
- `j_regular_out` — y-points in `[jMin, jMax]`   (default `iDim * 2 + 1`)

Points are ordered x-outer, y-inner (y varies fastest), matching
[`regularGridTransform`](@ref).

See also: [`regularGridTransform`](@ref), [`getGridpoints`](@ref)
"""
function getRegularGridpoints(grid::_2DCartesianRR)
    n_i   = grid.params.i_regular_out
    n_j   = grid.params.j_regular_out
    i_pts = collect(LinRange(grid.params.iMin, grid.params.iMax, n_i))
    j_pts = collect(LinRange(grid.params.jMin, grid.params.jMax, n_j))
    pts   = zeros(Float64, n_i * n_j, 2)
    idx   = 1
    for i in 1:n_i
        for j in 1:n_j
            pts[idx, 1] = i_pts[i]
            pts[idx, 2] = j_pts[j]
            idx += 1
        end
    end
    return pts
end

"""
    regularGridTransform(grid::_2DCartesianRR, i_pts, j_pts) -> Array{Float64}
    regularGridTransform(grid::_2DCartesianRR, gridpoints)   -> Array{Float64}

Evaluate the RR spectral representation on a regular tensor-product `x × y` grid,
returning field values and all five derivatives.

`grid.spectral` must be populated (call [`spectralTransform!`](@ref) first).

**Algorithm** (two-level tensor-product):
1. Evaluate i-splines at `i_pts` for all j-modes → buffer `ibuf[n_i, b_jDim]`.
2. For each `i_out`, set j-spline b-coefficients from `ibuf[i_out, :]`, then
   evaluate at `j_pts`.

# Returns
`Array{Float64}` of shape `(n_i × n_j, nvars, 5)` — y varies fastest.  Derivative slots:
- `[:,:,1]` — `f(x, y)`
- `[:,:,2]` — `∂f/∂x`
- `[:,:,3]` — `∂²f/∂x²`
- `[:,:,4]` — `∂f/∂y`
- `[:,:,5]` — `∂²f/∂y²`

# Example
```julia
spectralTransform!(grid_rr)
reg_pts  = getRegularGridpoints(grid_rr)
reg_phys = regularGridTransform(grid_rr, reg_pts)
```

See also: [`getRegularGridpoints`](@ref), [`gridTransform!`](@ref)
"""
function regularGridTransform(grid::_2DCartesianRR,
                               i_pts::AbstractVector{Float64},
                               j_pts::AbstractVector{Float64})
    gp     = grid.params
    b_iDim = gp.b_iDim
    b_jDim = gp.b_jDim
    nvars  = length(gp.vars)
    n_i    = length(i_pts)
    n_j    = length(j_pts)
    i_vec  = collect(Float64, i_pts)
    j_vec  = collect(Float64, j_pts)

    physical = zeros(Float64, n_i * n_j, nvars, 5)

    for v in 1:length(gp.vars)
        ibuf = zeros(Float64, n_i, b_jDim)
        tmp  = zeros(Float64, n_j)
        for dr in 0:2
            # Step 1: i-direction spline evaluation at i_pts for each j-mode
            for l in 1:b_jDim
                r1 = (l - 1) * b_iDim + 1
                r2 = r1 + b_iDim - 1
                sp = grid.ibasis.data[l, v]
                sp.b .= view(grid.spectral, r1:r2, v)
                SAtransform!(sp)
                _spline_eval!(sp, i_vec, dr, view(ibuf, :, l))
            end

            # Step 2: j-direction evaluation for each i output point
            dl_range = (dr == 0) ? (0:2) : (0:0)
            for dl in dl_range
                slot = _rr_slot(dr, dl)
                slot == 0 && continue
                scratch = grid.jbasis.data[1, v]   # any row; all share the same j-params
                for xi in 1:n_i
                    for l in 1:b_jDim
                        scratch.b[l] = ibuf[xi, l]
                    end
                    SAtransform!(scratch)
                    _spline_eval!(scratch, j_vec, dl, tmp)
                    flat = (xi - 1) * n_j + 1
                    physical[flat:flat + n_j - 1, v, slot] .= tmp
                end
            end
        end
    end

    return physical
end

@inline function _rr_slot(dr::Int, dl::Int)
    if     dr == 0 && dl == 0; return 1
    elseif dr == 1 && dl == 0; return 2
    elseif dr == 2 && dl == 0; return 3
    elseif dr == 0 && dl == 1; return 4
    elseif dr == 0 && dl == 2; return 5
    else;  return 0; end
end

function regularGridTransform(grid::_2DCartesianRR, gridpoints::AbstractMatrix{Float64})
    i_pts = sort(unique(gridpoints[:, 1]))
    j_pts = sort(unique(gridpoints[:, 2]))
    return regularGridTransform(grid, i_pts, j_pts)
end

# ═══════════════════════════════════════════════════════════════════════════
# Regular-grid output — 2D Cartesian Spline×Chebyshev (RZ)
# ═══════════════════════════════════════════════════════════════════════════

"""
    getRegularGridpoints(grid::_2DCartesianRZ) -> Matrix{Float64}

Return an `(n_i × n_k, 2)` matrix of uniformly-spaced `(x, z)` coordinates
for a 2-D Cartesian Spline×Chebyshev (RZ) grid.

Unlike [`getGridpoints`](@ref), which returns the unevenly-spaced Gaussian and
Chebyshev mish points, this function returns a regular tensor-product grid for
visualisation and file I/O.

The output dimensions are controlled by [`SpringsteelGridParameters`](@ref) fields:
- `i_regular_out` — x-points in `[iMin, iMax]`   (default `num_cells + 1`)
- `k_regular_out` — z-points in `[kMin, kMax]`   (default `kDim + 1`)

Points are ordered x-outer, z-inner (z varies fastest), matching
[`regularGridTransform`](@ref).

See also: [`regularGridTransform`](@ref), [`getGridpoints`](@ref)
"""
function getRegularGridpoints(grid::_2DCartesianRZ)
    n_i   = grid.params.i_regular_out
    n_k   = grid.params.k_regular_out
    i_pts = collect(LinRange(grid.params.iMin, grid.params.iMax, n_i))
    k_pts = collect(LinRange(grid.params.kMin, grid.params.kMax, n_k))
    pts   = zeros(Float64, n_i * n_k, 2)
    idx   = 1
    for i in 1:n_i
        for k in 1:n_k
            pts[idx, 1] = i_pts[i]
            pts[idx, 2] = k_pts[k]
            idx += 1
        end
    end
    return pts
end

"""
    regularGridTransform(grid::_2DCartesianRZ, i_pts, k_pts) -> Array{Float64}
    regularGridTransform(grid::_2DCartesianRZ, gridpoints)   -> Array{Float64}

Evaluate the RZ spectral representation on a regular tensor-product `x × z` grid,
returning field values and all five derivatives.

`grid.spectral` must be populated (call [`spectralTransform!`](@ref) first).

**Algorithm** (two-level tensor-product):
1. Evaluate i-splines at `i_pts` for all Chebyshev modes → buffer `ibuf[n_i, b_kDim]`.
2. For each `i_out`, set Chebyshev b-coefficients from `ibuf[i_out, :]`, apply
   `CAtransform!`, then evaluate at `k_pts` using direct polynomial evaluation.

# Returns
`Array{Float64}` of shape `(n_i × n_k, nvars, 5)` — z varies fastest.  Derivative slots:
- `[:,:,1]` — `f(x, z)`
- `[:,:,2]` — `∂f/∂x`
- `[:,:,3]` — `∂²f/∂x²`
- `[:,:,4]` — `∂f/∂z`
- `[:,:,5]` — `∂²f/∂z²`

# Example
```julia
spectralTransform!(grid_rz)
reg_pts  = getRegularGridpoints(grid_rz)
reg_phys = regularGridTransform(grid_rz, reg_pts)
```

See also: [`getRegularGridpoints`](@ref), [`gridTransform!`](@ref)
"""
function regularGridTransform(grid::_2DCartesianRZ,
                               i_pts::AbstractVector{Float64},
                               k_pts::AbstractVector{Float64})
    gp     = grid.params
    b_iDim = gp.b_iDim
    b_kDim = gp.b_kDim
    nvars  = length(gp.vars)
    n_i    = length(i_pts)
    n_k    = length(k_pts)
    i_vec  = collect(Float64, i_pts)
    k_vec  = collect(Float64, k_pts)

    physical = zeros(Float64, n_i * n_k, nvars, 5)

    for v in 1:length(gp.vars)
        ibuf     = zeros(Float64, n_i, b_kDim)
        cheb_col = grid.kbasis.data[v]
        for dr in 0:2
            # Step 1: i-direction spline evaluation at i_pts for each Chebyshev mode
            for z in 1:b_kDim
                r1 = (z - 1) * b_iDim + 1
                r2 = r1 + b_iDim - 1
                sp = grid.ibasis.data[z, v]
                sp.b .= view(grid.spectral, r1:r2, v)
                SAtransform!(sp)
                _spline_eval!(sp, i_vec, dr, view(ibuf, :, z))
            end

            # Step 2: Chebyshev evaluation at k_pts for each i output point
            # (_cheb_eval_pts! / _cheb_dz_pts! / _cheb_dzz_pts! are defined in
            # transforms_cylindrical.jl, included after this file)
            dk_range = (dr == 0) ? (0:2) : (0:0)
            for dk in dk_range
                slot = _rz_slot(dr, dk)
                slot == 0 && continue
                for xi in 1:n_i
                    for z in 1:b_kDim
                        cheb_col.b[z] = ibuf[xi, z]
                    end
                    CAtransform!(cheb_col)
                    flat = (xi - 1) * n_k + 1
                    out  = view(physical, flat:flat + n_k - 1, v, slot)
                    if dk == 0
                        _cheb_eval_pts!(cheb_col, k_vec, out)
                    elseif dk == 1
                        _cheb_dz_pts!(cheb_col, k_vec, out)
                    else
                        _cheb_dzz_pts!(cheb_col, k_vec, out)
                    end
                end
            end
        end
    end

    return physical
end

@inline function _rz_slot(dr::Int, dk::Int)
    if     dr == 0 && dk == 0; return 1
    elseif dr == 1 && dk == 0; return 2
    elseif dr == 2 && dk == 0; return 3
    elseif dr == 0 && dk == 1; return 4
    elseif dr == 0 && dk == 2; return 5
    else;  return 0; end
end

function regularGridTransform(grid::_2DCartesianRZ, gridpoints::AbstractMatrix{Float64})
    i_pts = sort(unique(gridpoints[:, 1]))
    k_pts = sort(unique(gridpoints[:, 2]))
    return regularGridTransform(grid, i_pts, k_pts)
end

# ═══════════════════════════════════════════════════════════════════════════
# 3D Cartesian Transforms  (Spline×Spline×Spline = RRR)
# ═══════════════════════════════════════════════════════════════════════════

# ── Type alias for brevity ──────────────────────────────────────────────────

# 3D Cartesian Spline×Spline×Spline (RRR):  i=Spline, j=Spline, k=Spline
const _3DCartesianRRR = SpringsteelGrid{CartesianGeometry, SplineBasisArray{3}, SplineBasisArray{3}, SplineBasisArray{3}}

# ────────────────────────────────────────────────────────────────────────────
# getGridpoints — 3D Cartesian
# ────────────────────────────────────────────────────────────────────────────

"""
    getGridpoints(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, SplineBasisArray}) -> Matrix{Float64}

Return a `(npoints, 3)` matrix of physical `(x, y, z)` coordinates for a 3-D Cartesian
Spline×Spline×Spline grid.

Points are ordered with k varying fastest, then j, then i:
flat index `(r-1)*jDim*kDim + (l-1)*kDim + z` → `(r, l, z)`.

See also: [`spectralTransform!`](@ref), [`gridTransform!`](@ref)
"""
function getGridpoints(grid::_3DCartesianRRR)
    iDim = grid.params.iDim
    jDim = grid.params.jDim
    kDim = grid.params.kDim
    pts  = zeros(Float64, iDim * jDim * kDim, 3)
    g = 1
    for r in 1:iDim
        xi = grid.ibasis.data[1, 1, 1].mishPoints[r]
        for l in 1:jDim
            yj = grid.jbasis.data[r, 1, 1].mishPoints[l]
            for z in 1:kDim
                zk = grid.kbasis.data[r, l, 1].mishPoints[z]
                pts[g, 1] = xi
                pts[g, 2] = yj
                pts[g, 3] = zk
                g += 1
            end
        end
    end
    return pts
end

# ────────────────────────────────────────────────────────────────────────────
# spectralTransform / spectralTransform! — 3D Cartesian Spline×Spline×Spline
# ────────────────────────────────────────────────────────────────────────────

"""
    spectralTransform(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, SplineBasisArray}, physical, spectral)
    spectralTransform!(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, SplineBasisArray})

Forward transform (physical → spectral) for a 3-D Cartesian Spline×Spline×Spline grid (RRR).

**Transform order** (physical → spectral): k-direction first, then j, then i.
1. k-direction `SBtransform!` for each `(r, l)` gridpoint pair → `tempsb_z[b_kDim, iDim, jDim]`
2. j-direction `SBtransform!` for each `(r, z_coeff)` → `tempsb_l[b_jDim, b_kDim, iDim]`
3. i-direction `SBtransform!` for each `(l_coeff, z_coeff)` → `spectral`

**Spectral layout**: z-major, then j-spectral, then i-spectral:
`spectral[(z-1)*b_jDim*b_iDim + (l-1)*b_iDim + 1 : …]` holds the i-direction B-coefficients
for j-mode `l` and k-mode `z`.

**Physical layout**: `flat = (r-1)*jDim*kDim + (l-1)*kDim + z` for gridpoint `(r, l, z)`.

See also: [`gridTransform!`](@ref)
"""
function spectralTransform(
        grid     :: _3DCartesianRRR,
        physical :: Array{real},
        spectral :: Array{real})
    iDim   = grid.params.iDim
    jDim   = grid.params.jDim
    kDim   = grid.params.kDim
    b_iDim = grid.params.b_iDim
    b_jDim = grid.params.b_jDim
    b_kDim = grid.params.b_kDim
    nvars  = size(spectral, 2)
    s = _scratch(grid)
    tempsb_z = s.tempsb_z
    tempsb_l = s.tempsb_l

    for v in 1:size(spectral, 2)
        # Step 1: k-direction (Z) transform for each (r, l) gridpoint
        for r in 1:iDim
            for l in 1:jDim
                ksp = grid.kbasis.data[r, l, v]
                @inbounds for z in 1:kDim
                    ksp.uMish[z] = physical[(r-1)*jDim*kDim + (l-1)*kDim + z, v, 1]
                end
                SBtransform!(ksp)
                @inbounds for k in 1:b_kDim
                    tempsb_z[k, r, l] = ksp.b[k]
                end
            end
        end

        # Step 2: j-direction (L) transform for each (r, z_coeff)
        for z in 1:b_kDim
            for r in 1:iDim
                jsp = grid.jbasis.data[r, z, v]
                @inbounds for l in 1:jDim
                    jsp.uMish[l] = tempsb_z[z, r, l]
                end
                SBtransform!(jsp)
                @inbounds for k in 1:b_jDim
                    tempsb_l[k, z, r] = jsp.b[k]
                end
            end
        end

        # Step 3: i-direction (R) transform for each (l_coeff, z_coeff)
        for z in 1:b_kDim
            for l in 1:b_jDim
                isp = grid.ibasis.data[l, z, v]
                @inbounds for r in 1:iDim
                    isp.uMish[r] = tempsb_l[l, z, r]
                end
                SBtransform!(isp)
                idx = (z-1)*b_jDim*b_iDim + (l-1)*b_iDim + 1
                @inbounds for k in 0:(b_iDim - 1)
                    spectral[idx + k, v] = isp.b[k + 1]
                end
            end
        end
    end
    return spectral
end

"""
    spectralTransform!(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, SplineBasisArray})

In-place forward transform for a 3-D Cartesian Spline×Spline×Spline grid.  Reads
`grid.physical[:, :, 1]` and writes `grid.spectral`.

See also: [`spectralTransform`](@ref), [`gridTransform!`](@ref)
"""
function spectralTransform!(grid::_3DCartesianRRR)
    spectralTransform(grid, grid.physical, grid.spectral)
    applyFilter!(grid)
    return grid.spectral
end

# ────────────────────────────────────────────────────────────────────────────
# gridTransform / gridTransform! — 3D Cartesian Spline×Spline×Spline
# ────────────────────────────────────────────────────────────────────────────

"""
    gridTransform(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, SplineBasisArray}, physical, spectral)
    gridTransform!(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, SplineBasisArray})

Inverse transform (spectral → physical + derivatives) for a 3-D Cartesian grid (RRR).

**Transform order** (spectral → physical): i-direction first, then j, then k.
1. i-direction `SAtransform!` → `SItransform!` / `SIxtransform` / `SIxxtransform`
   for each `(l_coeff, z_coeff)` → `splineBuffer_r[iDim, b_jDim, b_kDim]`
2. j-direction `SAtransform!` → `SItransform!` for each `(r, z_coeff)` → `splineBuffer_l[jDim, b_kDim]`
   (computed per `r` to avoid BUG-3 stale overwrite)
3. k-direction `SAtransform!` → `SItransform!` for each `(r, l)` → physical slots

**Physical derivative layout** (7 slots):
- `physical[:, v, 1]` — field values
- `physical[:, v, 2]` — `∂f/∂i` (first i-derivative)
- `physical[:, v, 3]` — `∂²f/∂i²` (second i-derivative)
- `physical[:, v, 4]` — `∂f/∂j` (first j-derivative) — **BUG-2 fix** applied
- `physical[:, v, 5]` — `∂²f/∂j²` (second j-derivative)
- `physical[:, v, 6]` — `∂f/∂k` (first k-derivative)
- `physical[:, v, 7]` — `∂²f/∂k²` (second k-derivative)

**Bug fixes applied**:
- **BUG-2** (rrr_grid.jl §13.1): j-derivative slot assignment used wrong dimension; now computed
  via separate SIxtransform/SIxxtransform pass through the k-direction.
- **BUG-3** (rrr_grid.jl §13.1): j-direction buffer was overwritten in outer r-loop, leaving only
  the last radial point's data; now the k-transform runs nested inside the r-loop.

See also: [`spectralTransform!`](@ref)
"""
function gridTransform(
        grid     :: _3DCartesianRRR,
        physical :: Array{real},
        spectral :: Array{real})
    iDim   = grid.params.iDim
    jDim   = grid.params.jDim
    kDim   = grid.params.kDim
    b_iDim = grid.params.b_iDim
    b_jDim = grid.params.b_jDim
    b_kDim = grid.params.b_kDim
    nvars  = size(spectral, 2)

    # Per-grid scratch (cached): splineBuffer_r/l/l_1st/l_2nd, scratch_i, scratch_j
    s = _scratch(grid)
    splineBuffer_r     = s.splineBuffer_r
    splineBuffer_l     = s.splineBuffer_l
    splineBuffer_l_1st = s.splineBuffer_l_1st
    splineBuffer_l_2nd = s.splineBuffer_l_2nd
    scratch_i          = s.scratch_i
    scratch_j          = s.scratch_j

    for v in 1:size(spectral, 2)
        for dr in 0:2
            # ── Step 1: i-direction (R) inverse transform ─────────────────────
            for z in 1:b_kDim
                for l in 1:b_jDim
                    idx = (z-1)*b_jDim*b_iDim + (l-1)*b_iDim + 1
                    isp = grid.ibasis.data[l, z, v]
                    copyto!(isp.b, view(spectral, idx:idx+b_iDim-1, v))
                    SAtransform!(isp)
                    if dr == 0
                        SItransform!(isp)
                        @inbounds for r in 1:iDim
                            splineBuffer_r[r, l, z] = isp.uMish[r]
                        end
                    elseif dr == 1
                        SIxtransform(isp, scratch_i)
                        @inbounds for r in 1:iDim
                            splineBuffer_r[r, l, z] = scratch_i[r]
                        end
                    else
                        SIxxtransform(isp, scratch_i)
                        @inbounds for r in 1:iDim
                            splineBuffer_r[r, l, z] = scratch_i[r]
                        end
                    end
                end
            end

            # ── Steps 2+3: j and k direction transforms ──────────────────────
            # FIX BUG-3: nest k-transform inside the r-loop so splineBuffer_l
            #            is not overwritten and stale when the k-transform runs.
            for r in 1:iDim
                # j-direction (L) transform per z_coeff
                for z in 1:b_kDim
                    jsp = grid.jbasis.data[r, z, v]
                    @inbounds for l in 1:b_jDim
                        jsp.b[l] = splineBuffer_r[r, l, z]
                    end
                    SAtransform!(jsp)
                    # FIX: always call SItransform! — old code read stale uMish for dr≠0
                    SItransform!(jsp)
                    @inbounds for l in 1:jDim
                        splineBuffer_l[l, z] = jsp.uMish[l]
                    end

                    # Compute j-derivatives during dr==0 pass using the VALUE A-coefficients
                    if dr == 0
                        SIxtransform(jsp, scratch_j)
                        @inbounds for l in 1:jDim
                            splineBuffer_l_1st[l, z] = scratch_j[l]
                        end
                        SIxxtransform(jsp, scratch_j)
                        @inbounds for l in 1:jDim
                            splineBuffer_l_2nd[l, z] = scratch_j[l]
                        end
                    end
                end

                # k-direction (Z) transform per (r, l) gridpoint
                for l in 1:jDim
                    ksp = grid.kbasis.data[r, l, v]
                    @inbounds for zb in 1:b_kDim
                        ksp.b[zb] = splineBuffer_l[l, zb]
                    end
                    SAtransform!(ksp)
                    SItransform!(ksp)

                    i_flat = (r-1)*jDim*kDim + (l-1)*kDim + 1
                    i_flat_end = i_flat + kDim - 1
                    if dr == 0
                        copyto!(view(physical, i_flat:i_flat_end, v, 1), ksp.uMish)
                        # Reuse ksp.uMish — its prior content was just copied above.
                        SIxtransform(ksp, ksp.uMish)
                        copyto!(view(physical, i_flat:i_flat_end, v, 6), ksp.uMish)
                        SIxxtransform(ksp, ksp.uMish)
                        copyto!(view(physical, i_flat:i_flat_end, v, 7), ksp.uMish)

                        # FIX BUG-2: j-derivative slots via correct k-inverse-transform
                        # Slot 4: ∂f/∂j (first j-derivative)
                        @inbounds for zb in 1:b_kDim
                            ksp.b[zb] = splineBuffer_l_1st[l, zb]
                        end
                        SAtransform!(ksp)
                        SItransform!(ksp)
                        copyto!(view(physical, i_flat:i_flat_end, v, 4), ksp.uMish)

                        # Slot 5: ∂²f/∂j² (second j-derivative)
                        @inbounds for zb in 1:b_kDim
                            ksp.b[zb] = splineBuffer_l_2nd[l, zb]
                        end
                        SAtransform!(ksp)
                        SItransform!(ksp)
                        copyto!(view(physical, i_flat:i_flat_end, v, 5), ksp.uMish)

                    elseif dr == 1
                        copyto!(view(physical, i_flat:i_flat_end, v, 2), ksp.uMish)
                    else
                        copyto!(view(physical, i_flat:i_flat_end, v, 3), ksp.uMish)
                    end
                end
            end  # for r
        end  # for dr
    end  # for v

    return physical
end

"""
    gridTransform!(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, SplineBasisArray})

In-place inverse transform for a 3-D Cartesian Spline×Spline×Spline grid.  Reads
`grid.spectral` and writes `grid.physical` (values + derivatives in all 7 slots).

See also: [`gridTransform`](@ref), [`spectralTransform!`](@ref)
"""
function gridTransform!(grid::_3DCartesianRRR)
    gridTransform(grid, grid.physical, grid.spectral)
    return grid.physical
end

# ═══════════════════════════════════════════════════════════════════════════
# Regular-grid output — 3D Cartesian (RRR)
# ═══════════════════════════════════════════════════════════════════════════

"""
    getRegularGridpoints(grid::_3DCartesianRRR) -> Matrix{Float64}

Return an `(n_x × n_y × n_z, 3)` matrix of uniformly-spaced `(x, y, z)` coordinates
for a 3-D Cartesian Spline×Spline×Spline (RRR) grid.

Unlike [`getGridpoints`](@ref), which returns the unevenly-spaced Gaussian mish points,
this function returns a regular tensor-product grid for visualisation and file I/O.

The output dimensions are controlled by [`SpringsteelGridParameters`](@ref) fields:
- `i_regular_out` — x-points in `[iMin, iMax]`   (default `num_cells + 1`)
- `j_regular_out` — y-points in `[jMin, jMax]`   (default `iDim * 2 + 1`)
- `k_regular_out` — z-points in `[kMin, kMax]`   (default `kDim + 1`)

Points are ordered x-outer, y-middle, z-inner (z varies fastest), matching
[`regularGridTransform`](@ref).

See also: [`regularGridTransform`](@ref), [`getGridpoints`](@ref)
"""
function getRegularGridpoints(grid::_3DCartesianRRR)
    n_x   = grid.params.i_regular_out
    n_y   = grid.params.j_regular_out
    n_z   = grid.params.k_regular_out
    x_pts = collect(LinRange(grid.params.iMin, grid.params.iMax, n_x))
    y_pts = collect(LinRange(grid.params.jMin, grid.params.jMax, n_y))
    z_pts = collect(LinRange(grid.params.kMin, grid.params.kMax, n_z))
    pts   = zeros(Float64, n_x * n_y * n_z, 3)
    idx   = 1
    for i in 1:n_x
        for j in 1:n_y
            for k in 1:n_z
                pts[idx, 1] = x_pts[i]
                pts[idx, 2] = y_pts[j]
                pts[idx, 3] = z_pts[k]
                idx += 1
            end
        end
    end
    return pts
end

# Helper: apply spline inverse (or derivative) transform to arbitrary points
@inline function _spline_eval!(sp, pts, deriv, out)
    if deriv == 0; SItransform(sp, pts, out)
    elseif deriv == 1; SIxtransform(sp, pts, out)
    else; SIxxtransform(sp, pts, out); end
end

"""
    regularGridTransform(grid::_3DCartesianRRR, x_pts, y_pts, z_pts) -> Array{Float64}
    regularGridTransform(grid::_3DCartesianRRR, gridpoints)           -> Array{Float64}

Evaluate the RRR spectral representation on a regular tensor-product `x × y × z` grid.

`grid.spectral` must be populated (call [`spectralTransform!`](@ref) first).

**Algorithm** (three-level tensor-product):
1. Evaluate i-splines at `x_pts` for all `(j_coeff, k_coeff)` pairs → `ibuf[x, j, k]`.
2. For each `x_out`, evaluate j-splines at `y_pts` using `ibuf[x_out,:,k]` as B-coefficients.
3. For each `(x_out, y_out)`, evaluate k-splines at `z_pts` for the final field values.

The existing grid basis objects are reused as scratch buffers (their `.b` and `.a` fields
are overwritten; `grid.spectral` provides the authoritative spectral state).

# Returns
`Array{Float64}` of shape `(n_x × n_y × n_z, nvars, 7)` — z varies fastest.  Derivative
slots follow the same convention as [`gridTransform!`](@ref) for RRR:
- `[:,:,1]` — `f`, `[:,:,2]` — `∂f/∂x`, `[:,:,3]` — `∂²f/∂x²`
- `[:,:,4]` — `∂f/∂y`, `[:,:,5]` — `∂²f/∂y²`
- `[:,:,6]` — `∂f/∂z`, `[:,:,7]` — `∂²f/∂z²`

# Example
```julia
spectralTransform!(grid_rrr)
reg_pts  = getRegularGridpoints(grid_rrr)
reg_phys = regularGridTransform(grid_rrr, reg_pts)
```

See also: [`getRegularGridpoints`](@ref), [`gridTransform!`](@ref)
"""
function regularGridTransform(grid::_3DCartesianRRR,
                               x_pts::AbstractVector{Float64},
                               y_pts::AbstractVector{Float64},
                               z_pts::AbstractVector{Float64})
    gp     = grid.params
    b_iDim = gp.b_iDim
    b_jDim = gp.b_jDim
    b_kDim = gp.b_kDim
    nvars  = length(gp.vars)
    n_x    = length(x_pts)
    n_y    = length(y_pts)
    n_z    = length(z_pts)
    x_vec  = collect(Float64, x_pts)
    y_vec  = collect(Float64, y_pts)
    z_vec  = collect(Float64, z_pts)

    physical = zeros(Float64, n_x * n_y * n_z, nvars, 7)

    for v in 1:length(gp.vars)
        for dr in 0:2
            # ── Step 1: i-spline evaluation at x_pts ────────────────────────
            # ibuf[x_out, l_coeff, z_coeff]
            ibuf = zeros(Float64, n_x, b_jDim, b_kDim)
            for l in 1:b_jDim
                for z_b in 1:b_kDim
                    idx = (z_b - 1) * b_jDim * b_iDim + (l - 1) * b_iDim + 1
                    sp = grid.ibasis.data[l, z_b, v]
                    sp.b .= view(grid.spectral, idx:idx + b_iDim - 1, v)
                    SAtransform!(sp)
                    _spline_eval!(sp, x_vec, dr, view(ibuf, :, l, z_b))
                end
            end

            # ── Steps 2 & 3: j and k evaluations ───────────────────────────
            dl_range = (dr == 0) ? (0:2) : (0:0)
            for dl in dl_range
                # jbuf[x_out, y_out, z_coeff]
                jbuf = zeros(Float64, n_x, n_y, b_kDim)
                jsp  = grid.jbasis.data[1, 1, v]   # scratch j-spline (same params for all r/z)
                for xi in 1:n_x
                    for z_b in 1:b_kDim
                        for l in 1:b_jDim
                            jsp.b[l] = ibuf[xi, l, z_b]
                        end
                        SAtransform!(jsp)
                        _spline_eval!(jsp, y_vec, dl, view(jbuf, xi, :, z_b))
                    end
                end

                dk_range = (dr == 0 && dl == 0) ? (0:2) : (0:0)
                for dk in dk_range
                    slot = _rrr_regular_slot(dr, dl, dk)
                    slot == 0 && continue
                    ksp = grid.kbasis.data[1, 1, v]   # scratch k-spline
                    for xi in 1:n_x
                        for yj in 1:n_y
                            for z_b in 1:b_kDim
                                ksp.b[z_b] = jbuf[xi, yj, z_b]
                            end
                            SAtransform!(ksp)
                            flat = (xi - 1) * n_y * n_z + (yj - 1) * n_z + 1
                            _spline_eval!(ksp, z_vec, dk, view(physical, flat:flat + n_z - 1, v, slot))
                        end
                    end
                end
            end
        end   # dr
    end   # v

    return physical
end

@inline function _rrr_regular_slot(dr, dl, dk)
    if     dr == 0 && dl == 0 && dk == 0; return 1
    elseif dr == 1 && dl == 0 && dk == 0; return 2
    elseif dr == 2 && dl == 0 && dk == 0; return 3
    elseif dr == 0 && dl == 1 && dk == 0; return 4
    elseif dr == 0 && dl == 2 && dk == 0; return 5
    elseif dr == 0 && dl == 0 && dk == 1; return 6
    elseif dr == 0 && dl == 0 && dk == 2; return 7
    else;  return 0; end
end

function regularGridTransform(grid::_3DCartesianRRR, gridpoints::AbstractMatrix{Float64})
    x_pts = sort(unique(gridpoints[:, 1]))
    y_pts = sort(unique(gridpoints[:, 2]))
    z_pts = sort(unique(gridpoints[:, 3]))
    return regularGridTransform(grid, x_pts, y_pts, z_pts)
end

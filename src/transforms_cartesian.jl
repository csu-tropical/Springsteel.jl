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
        spline = grid.ibasis.data[1, v]
        @inbounds for i in eachindex(spline.uMish)
            spline.uMish[i] = physical[i, v, 1]
        end
        SBtransform!(spline)
        @inbounds for i in eachindex(spline.b)
            spectral[i, v] = spline.b[i]
        end
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
    _filter_mish!(grid)
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
    _filter_mish!(grid)
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
    _filter_mish!(grid)
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

# ════════════════════════════════════════════════════════════════════════════
# 2D Cartesian Spline×Spline with vertical in k (RiRk) — spline-i × spline-k.
# Structurally identical to RZ (one k-column per variable, shared across i),
# but the k-direction uses cubic B-spline transforms instead of Chebyshev.
# ════════════════════════════════════════════════════════════════════════════

const _2DCartesianRiRk = SpringsteelGrid{CartesianGeometry, SplineBasisArray{2}, NoBasisArray, SplineBasisArray{1}}

"""
    getGridpoints(grid::_2DCartesianRiRk) -> Matrix{Float64}

Return a `(iDim*kDim, 2)` matrix of physical grid coordinates. Column 1 is the
i-direction (spline) coordinate, column 2 the k-direction (spline) coordinate;
the k index varies fastest, matching the RZ layout `(r-1)*kDim + z`.
"""
function getGridpoints(grid::_2DCartesianRiRk)
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

"""
    spectralTransform(grid::_2DCartesianRiRk, physical, spectral)

Forward transform (physical → spectral) for a 2-D Cartesian Spline×Spline (k)
grid. Mirrors the RZ transform with the k-direction Chebyshev `CBtransform!`
replaced by the spline `SBtransform!`. Spectral layout is identical to RZ:
`b_kDim` consecutive `b_iDim`-element blocks, one per vertical spline mode.
"""
function spectralTransform(
        grid     :: _2DCartesianRiRk,
        physical :: Array{real},
        spectral :: Array{real})
    iDim  = grid.params.iDim
    kDim  = grid.params.kDim
    b_iDim = grid.params.b_iDim
    b_kDim = grid.params.b_kDim
    nvars = size(spectral, 2)
    tempcb = _scratch(grid).tempcb

    for v in 1:nvars
        # Step 1: k-direction (spline) transform for each i gridpoint
        kcol = grid.kbasis.data[v]
        for r in 1:iDim
            @inbounds for z in 1:kDim
                kcol.uMish[z] = physical[(r-1)*kDim + z, v, 1]
            end
            SBtransform!(kcol)
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
    spectralTransform!(grid::_2DCartesianRiRk)

In-place forward transform for a 2-D Cartesian Spline×Spline (k) grid.
"""
function spectralTransform!(grid::_2DCartesianRiRk)
    _filter_mish!(grid)
    spectralTransform(grid, grid.physical, grid.spectral)
    applyFilter!(grid)
    return grid.spectral
end

"""
    gridTransform(grid::_2DCartesianRiRk, physical, spectral)

Inverse transform (spectral → physical + derivatives) for a 2-D Cartesian
Spline×Spline (k) grid. Mirrors the RZ inverse transform; the k-direction
Chebyshev calls are replaced by their spline equivalents. Derivative slots:
1=value, 2=∂i, 3=∂²i, 4=∂k, 5=∂²k.
"""
function gridTransform(
        grid     :: _2DCartesianRiRk,
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
            # Inhomogeneous Neumann (R1T1X): one shared column spline serves every
            # radius, so the per-column wall derivative has to be installed into
            # `ahat` immediately before each column's SA solve. Hoisted BC lookup;
            # both flags are false for every other BC, so this costs one branch.
            wall_du = grid.kbasis.wall_du
            xL = haskey(kcol.params.BCL, "X1") && !isempty(wall_du)
            xR = haskey(kcol.params.BCR, "X1") && !isempty(wall_du)
            for r in 1:iDim
                @inbounds for z in 1:b_kDim
                    kcol.b[z] = splineBuffer[r, z]
                end
                xL && CubicBSpline.set_ahat_neumann!(kcol, wall_du[r, v, 1, dr+1], :left)
                xR && CubicBSpline.set_ahat_neumann!(kcol, wall_du[r, v, 2, dr+1], :right)
                SAtransform!(kcol)
                SItransform!(kcol)
                z1 = (r-1)*kDim + 1
                z2 = z1 + kDim - 1
                if dr == 0
                    copyto!(view(physical, z1:z2, v, 1), kcol.uMish)
                    # Reuse kcol.uMish as scratch — its prior content was just copied above.
                    SIxtransform(kcol, kcol.uMish)
                    copyto!(view(physical, z1:z2, v, 4), kcol.uMish)
                    SIxxtransform(kcol, kcol.uMish)
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
    gridTransform!(grid::_2DCartesianRiRk)

In-place inverse transform for a 2-D Cartesian Spline×Spline (k) grid.
"""
function gridTransform!(grid::_2DCartesianRiRk)
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
- `j_regular_out` — y-points in `[jMin, jMax]`   (default: y cells + 1)

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
# Regular-grid output — 2D Cartesian Spline×Spline in k (RiRk)
# ═══════════════════════════════════════════════════════════════════════════

"""
    getRegularGridpoints(grid::_2DCartesianRiRk) -> Matrix{Float64}

Return an `(n_i × n_k, 2)` matrix of uniformly-spaced `(x, z)` coordinates for a
2-D Cartesian Spline×Spline-in-k (RiRk) grid. Mirrors the RZ method — both place
the vertical in `k` — but here `k` is a cubic B-spline rather than Chebyshev.

Output dimensions come from [`SpringsteelGridParameters`](@ref):
- `i_regular_out` — x-points in `[iMin, iMax]`   (default `num_cells + 1`)
- `k_regular_out` — z-points in `[kMin, kMax]`   (default `kDim + 1`)

Points are ordered x-outer, z-inner (z varies fastest), matching
[`regularGridTransform`](@ref).

See also: [`regularGridTransform`](@ref), [`getGridpoints`](@ref)
"""
function getRegularGridpoints(grid::_2DCartesianRiRk)
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
    regularGridTransform(grid::_2DCartesianRiRk, i_pts, k_pts) -> Array{Float64}
    regularGridTransform(grid::_2DCartesianRiRk, gridpoints)   -> Array{Float64}

Evaluate the RiRk spectral representation on a regular tensor-product `x × z`
grid, returning field values and all five derivatives.

`grid.spectral` must be populated (call [`spectralTransform!`](@ref) first).

**Algorithm** (two-level tensor-product, identical structure to RZ but with the
k-direction Chebyshev replaced by a cubic B-spline):
1. Evaluate i-splines at `i_pts` for all k-modes → buffer `ibuf[n_i, b_kDim]`
   (k-major spectral layout, `r1 = (z-1)*b_iDim + 1`).
2. For each `i_out`, set k-spline b-coefficients from `ibuf[i_out, :]`, apply
   `SAtransform!`, then evaluate at `k_pts`.

# Returns
`Array{Float64}` of shape `(n_i × n_k, nvars, 5)` — z varies fastest. Derivative
slots: `[:,:,1]` `f(x,z)`, `[:,:,2]` `∂f/∂x`, `[:,:,3]` `∂²f/∂x²`,
`[:,:,4]` `∂f/∂z`, `[:,:,5]` `∂²f/∂z²`.

See also: [`getRegularGridpoints`](@ref), [`gridTransform!`](@ref)
"""
function regularGridTransform(grid::_2DCartesianRiRk,
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
        ibuf = zeros(Float64, n_i, b_kDim)
        tmp  = zeros(Float64, n_k)
        for dr in 0:2
            # Step 1: i-direction spline evaluation at i_pts for each k-mode
            for z in 1:b_kDim
                r1 = (z - 1) * b_iDim + 1
                r2 = r1 + b_iDim - 1
                sp = grid.ibasis.data[z, v]
                sp.b .= view(grid.spectral, r1:r2, v)
                SAtransform!(sp)
                _spline_eval!(sp, i_vec, dr, view(ibuf, :, z))
            end

            # Step 2: k-direction spline evaluation at k_pts for each i output
            dk_range = (dr == 0) ? (0:2) : (0:0)
            for dk in dk_range
                slot = _rz_slot(dr, dk)   # same (i-deriv, k-deriv) → slot mapping as RZ
                slot == 0 && continue
                scratch = grid.kbasis.data[v]
                for xi in 1:n_i
                    for z in 1:b_kDim
                        scratch.b[z] = ibuf[xi, z]
                    end
                    SAtransform!(scratch)
                    _spline_eval!(scratch, k_vec, dk, tmp)
                    flat = (xi - 1) * n_k + 1
                    physical[flat:flat + n_k - 1, v, slot] .= tmp
                end
            end
        end
    end

    return physical
end

function regularGridTransform(grid::_2DCartesianRiRk, gridpoints::AbstractMatrix{Float64})
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
    _filter_mish!(grid)
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
- `j_regular_out` — y-points in `[jMin, jMax]`   (default: y cells + 1)
- `k_regular_out` — z-points in `[kMin, kMax]`   (default: z cells + 1)

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

# ═══════════════════════════════════════════════════════════════════════════
# Pure-Chebyshev Cartesian transforms (Z / ZZ / ZZZ)
#
# These mirror the spline-based 1D / RZ / RRR transforms above, substituting the
# Chebyshev primitives (CBtransform!/CAtransform!/CItransform!/CIxtransform/
# CIxxtransform) for the spline ones. Index/slot conventions are identical:
#   • physical ordering is k-fastest (then j, then i)
#   • derivative slots: 1D→3 [f,∂i,∂²i]; 2D→5 [f,∂i,∂²i,∂j,∂²j];
#     3D→7 [f,∂i,∂²i,∂j,∂²j,∂k,∂²k]
#
# Structural note: unlike RRR (which carries per-physical-radius j/k basis
# objects), the ZZ/ZZZ factories build a nested *spectral* basis layout — the
# second/third bases are shared scratch columns (jbasis.data[z,v], kbasis.data[v]),
# one per spectral mode rather than per gridpoint. The transforms below index
# accordingly.
# ═══════════════════════════════════════════════════════════════════════════

# ── Type aliases ────────────────────────────────────────────────────────────
const _1DCartesianZ = SpringsteelGrid{CartesianGeometry, ChebyshevBasisArray{1}, NoBasisArray, NoBasisArray}
const _2DCartesianZZ = SpringsteelGrid{CartesianGeometry, ChebyshevBasisArray{2}, ChebyshevBasisArray{1}, NoBasisArray}
const _3DCartesianZZZ = SpringsteelGrid{CartesianGeometry, ChebyshevBasisArray{3}, ChebyshevBasisArray{2}, ChebyshevBasisArray{1}}

# Evaluate a Chebyshev column (whose A-coefficients are already populated via
# CAtransform!) and its derivatives at arbitrary points. Parallels `_spline_eval!`.
@inline function _cheb_eval!(col, pts, deriv, out)
    if deriv == 0
        _cheb_eval_pts!(col, pts, out)
    elseif deriv == 1
        _cheb_dz_pts!(col, pts, out)
    else
        _cheb_dzz_pts!(col, pts, out)
    end
end

# ───────────────────────────────────────────────────────────────────────────
# Z — 1D Cartesian Chebyshev
# ───────────────────────────────────────────────────────────────────────────

"""
    getGridpoints(grid::SpringsteelGrid{CartesianGeometry, ChebyshevBasisArray, NoBasisArray, NoBasisArray}) -> Vector{Float64}

Return the Chebyshev–Gauss–Lobatto mish points for a 1-D Cartesian Chebyshev grid
(`Z_Grid` / `Column1D_Grid`). All variables share the same domain, so the first
variable's column is canonical.
"""
function getGridpoints(grid::_1DCartesianZ)
    return grid.ibasis.data[1].mishPoints
end

"""
    getRegularGridpoints(grid::_1DCartesianZ) -> Vector{Float64}

Return `i_regular_out` uniformly-spaced output locations spanning `[iMin, iMax]`.
"""
function getRegularGridpoints(grid::_1DCartesianZ)
    n  = grid.params.i_regular_out
    x0 = grid.params.iMin
    x1 = grid.params.iMax
    dx = (x1 - x0) / (n - 1)
    return [min(x0 + (i - 1) * dx, x1) for i in 1:n]
end

"""
    spectralTransform(grid::_1DCartesianZ, physical, spectral)

Explicit-array forward transform for a 1-D Cartesian Chebyshev grid. Applies
`CBtransform!` per variable, writing B-coefficients into `spectral`.
"""
function spectralTransform(
        grid     :: _1DCartesianZ,
        physical :: Array{real},
        spectral :: Array{real})
    nvars = size(spectral, 2)
    for v in 1:nvars
        col = grid.ibasis.data[v]
        @inbounds for i in eachindex(col.uMish)
            col.uMish[i] = physical[i, v, 1]
        end
        CBtransform!(col)
        @inbounds for i in eachindex(col.b)
            spectral[i, v] = col.b[i]
        end
    end
    return spectral
end

"""
    spectralTransform!(grid::_1DCartesianZ)

In-place forward transform for a 1-D Cartesian Chebyshev grid.
"""
function spectralTransform!(grid::_1DCartesianZ)
    _filter_mish!(grid)
    spectralTransform(grid, grid.physical, grid.spectral)
    applyFilter!(grid)
    return grid.spectral
end

"""
    gridTransform(grid::_1DCartesianZ, physical, spectral)

Explicit-array inverse transform. For each variable: `CAtransform!` (B→A), then
`CItransform!` / `CIxtransform` / `CIxxtransform` at the mish points, writing the
value and first/second derivatives into slots 1/2/3.
"""
function gridTransform(
        grid     :: _1DCartesianZ,
        physical :: Array{real},
        spectral :: Array{real})
    nvars = size(spectral, 2)
    for v in 1:nvars
        col = grid.ibasis.data[v]
        copyto!(col.b, view(spectral, :, v))
        CAtransform!(col)
        CItransform!(col)
        @inbounds for i in eachindex(col.uMish)
            physical[i, v, 1] = col.uMish[i]
        end
        CIxtransform(col, view(physical, :, v, 2))
        CIxxtransform(col, view(physical, :, v, 3))
    end
    return physical
end

"""
    gridTransform!(grid::_1DCartesianZ)

In-place inverse transform for a 1-D Cartesian Chebyshev grid (slots 1/2/3).
"""
function gridTransform!(grid::_1DCartesianZ)
    gridTransform(grid, grid.physical, grid.spectral)
    return grid.physical
end

"""
    regularGridTransform(grid::_1DCartesianZ, gridpoints::AbstractVector{Float64}) -> Array{Float64}

Evaluate the Chebyshev representation (and first/second derivatives) at arbitrary
output locations. `grid.spectral` must be populated.
"""
function regularGridTransform(grid::_1DCartesianZ, gridpoints::AbstractVector{Float64})
    nvars    = length(grid.params.vars)
    gpts     = collect(Float64, gridpoints)
    physical = zeros(Float64, length(gpts), nvars, 3)
    for v in 1:nvars
        col = grid.ibasis.data[v]
        col.b .= view(grid.spectral, :, v)
        CAtransform!(col)
        _cheb_eval_pts!(col, gpts, view(physical, :, v, 1))
        _cheb_dz_pts!(col,  gpts, view(physical, :, v, 2))
        _cheb_dzz_pts!(col, gpts, view(physical, :, v, 3))
    end
    return physical
end

# ───────────────────────────────────────────────────────────────────────────
# ZZ — 2D Cartesian Chebyshev×Chebyshev (i, j active)
# Structurally the RZ transform with the second axis in the j-slot and the
# i-axis Chebyshev instead of spline.
# ───────────────────────────────────────────────────────────────────────────

"""
    getGridpoints(grid::_2DCartesianZZ) -> Matrix{Float64}

Return a `(iDim*jDim, 2)` matrix of `(x, y)` mish coordinates; j varies fastest,
flat index `(r-1)*jDim + l`.
"""
function getGridpoints(grid::_2DCartesianZZ)
    iDim = grid.params.iDim
    jDim = grid.params.jDim
    pts  = zeros(Float64, iDim * jDim, 2)
    g = 1
    for r in 1:iDim
        xi = grid.ibasis.data[1, 1].mishPoints[r]
        for l in 1:jDim
            pts[g, 1] = xi
            pts[g, 2] = grid.jbasis.data[1].mishPoints[l]
            g += 1
        end
    end
    return pts
end

"""
    spectralTransform(grid::_2DCartesianZZ, physical, spectral)

Forward transform. Step 1: j-direction `CBtransform!` per i gridpoint. Step 2:
i-direction `CBtransform!` per j-mode. Spectral layout: consecutive `b_iDim`
blocks per j-mode at `(l-1)*b_iDim+1`. Physical layout `(r-1)*jDim + l`.
"""
function spectralTransform(
        grid     :: _2DCartesianZZ,
        physical :: Array{real},
        spectral :: Array{real})
    iDim   = grid.params.iDim
    jDim   = grid.params.jDim
    b_iDim = grid.params.b_iDim
    b_jDim = grid.params.b_jDim
    nvars  = size(spectral, 2)
    tempcb = zeros(Float64, b_jDim, iDim)

    for v in 1:nvars
        # Step 1: j-direction (Chebyshev) transform for each i gridpoint
        jcol = grid.jbasis.data[v]
        for r in 1:iDim
            @inbounds for l in 1:jDim
                jcol.uMish[l] = physical[(r-1)*jDim + l, v, 1]
            end
            CBtransform!(jcol)
            @inbounds for k in 1:b_jDim
                tempcb[k, r] = jcol.b[k]
            end
        end

        # Step 2: i-direction (Chebyshev) transform for each j spectral mode
        for l in 1:b_jDim
            isp = grid.ibasis.data[l, v]
            @inbounds for r in 1:iDim
                isp.uMish[r] = tempcb[l, r]
            end
            CBtransform!(isp)
            r1 = (l-1)*b_iDim + 1
            @inbounds for k in 0:(b_iDim - 1)
                spectral[r1 + k, v] = isp.b[k + 1]
            end
        end
    end
    return spectral
end

"""
    spectralTransform!(grid::_2DCartesianZZ)

In-place forward transform for a 2-D Cartesian Chebyshev×Chebyshev grid.
"""
function spectralTransform!(grid::_2DCartesianZZ)
    _filter_mish!(grid)
    spectralTransform(grid, grid.physical, grid.spectral)
    applyFilter!(grid)
    return grid.spectral
end

"""
    gridTransform(grid::_2DCartesianZZ, physical, spectral)

Inverse transform. Step 1: i-direction `CAtransform!` → `CItransform!` /
`CIxtransform` / `CIxxtransform` per j-mode into `splineBuffer[iDim, b_jDim]`.
Step 2: j-direction inverse per i gridpoint into the 5 physical slots
([f, ∂i, ∂²i, ∂j, ∂²j]).
"""
function gridTransform(
        grid     :: _2DCartesianZZ,
        physical :: Array{real},
        spectral :: Array{real})
    iDim   = grid.params.iDim
    jDim   = grid.params.jDim
    b_iDim = grid.params.b_iDim
    b_jDim = grid.params.b_jDim
    nvars  = size(spectral, 2)
    splineBuffer = zeros(Float64, iDim, b_jDim)
    scratch_i    = zeros(Float64, iDim)

    for v in 1:nvars
        for dr in 0:2
            # i-direction inverse transform per j-spectral mode
            for l in 1:b_jDim
                r1 = (l-1)*b_iDim + 1
                r2 = r1 + b_iDim - 1
                isp = grid.ibasis.data[l, v]
                copyto!(isp.b, view(spectral, r1:r2, v))
                CAtransform!(isp)
                if dr == 0
                    CItransform!(isp)
                    @inbounds for r in 1:iDim
                        splineBuffer[r, l] = isp.uMish[r]
                    end
                elseif dr == 1
                    CIxtransform(isp, scratch_i)
                    @inbounds for r in 1:iDim
                        splineBuffer[r, l] = scratch_i[r]
                    end
                else
                    CIxxtransform(isp, scratch_i)
                    @inbounds for r in 1:iDim
                        splineBuffer[r, l] = scratch_i[r]
                    end
                end
            end

            # j-direction inverse transform per i gridpoint
            jcol = grid.jbasis.data[v]
            for r in 1:iDim
                @inbounds for l in 1:b_jDim
                    jcol.b[l] = splineBuffer[r, l]
                end
                CAtransform!(jcol)
                CItransform!(jcol)
                l1 = (r-1)*jDim + 1
                l2 = l1 + jDim - 1
                if dr == 0
                    copyto!(view(physical, l1:l2, v, 1), jcol.uMish)
                    # Reuse jcol.uMish as scratch — its prior content was just copied.
                    CIxtransform(jcol, jcol.uMish)
                    copyto!(view(physical, l1:l2, v, 4), jcol.uMish)
                    CIxxtransform(jcol, jcol.uMish)
                    copyto!(view(physical, l1:l2, v, 5), jcol.uMish)
                elseif dr == 1
                    copyto!(view(physical, l1:l2, v, 2), jcol.uMish)
                else
                    copyto!(view(physical, l1:l2, v, 3), jcol.uMish)
                end
            end
        end
    end
    return physical
end

"""
    gridTransform!(grid::_2DCartesianZZ)

In-place inverse transform for a 2-D Cartesian Chebyshev×Chebyshev grid (5 slots).
"""
function gridTransform!(grid::_2DCartesianZZ)
    gridTransform(grid, grid.physical, grid.spectral)
    return grid.physical
end

"""
    getRegularGridpoints(grid::_2DCartesianZZ) -> Matrix{Float64}

Return an `(i_regular_out × j_regular_out, 2)` matrix of uniformly-spaced `(x, y)`
coordinates; y varies fastest.
"""
function getRegularGridpoints(grid::_2DCartesianZZ)
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
    regularGridTransform(grid::_2DCartesianZZ, i_pts, j_pts) -> Array{Float64}
    regularGridTransform(grid::_2DCartesianZZ, gridpoints)   -> Array{Float64}

Evaluate the spectral representation on a regular `x × y` grid, returning values
and all five derivatives. Output shape `(n_i × n_j, nvars, 5)`, y varies fastest.
"""
function regularGridTransform(grid::_2DCartesianZZ,
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
        jcol = grid.jbasis.data[v]
        for dr in 0:2
            # Step 1: i-direction Chebyshev evaluation at i_pts for each j-mode
            for l in 1:b_jDim
                r1 = (l - 1) * b_iDim + 1
                r2 = r1 + b_iDim - 1
                sp = grid.ibasis.data[l, v]
                sp.b .= view(grid.spectral, r1:r2, v)
                CAtransform!(sp)
                _cheb_eval!(sp, i_vec, dr, view(ibuf, :, l))
            end

            # Step 2: j-direction Chebyshev evaluation at j_pts for each i output
            dj_range = (dr == 0) ? (0:2) : (0:0)
            for dj in dj_range
                slot = _rz_slot(dr, dj)
                slot == 0 && continue
                for xi in 1:n_i
                    for l in 1:b_jDim
                        jcol.b[l] = ibuf[xi, l]
                    end
                    CAtransform!(jcol)
                    flat = (xi - 1) * n_j + 1
                    out  = view(physical, flat:flat + n_j - 1, v, slot)
                    if dj == 0
                        _cheb_eval_pts!(jcol, j_vec, out)
                    elseif dj == 1
                        _cheb_dz_pts!(jcol, j_vec, out)
                    else
                        _cheb_dzz_pts!(jcol, j_vec, out)
                    end
                end
            end
        end
    end
    return physical
end

function regularGridTransform(grid::_2DCartesianZZ, gridpoints::AbstractMatrix{Float64})
    i_pts = sort(unique(gridpoints[:, 1]))
    j_pts = sort(unique(gridpoints[:, 2]))
    return regularGridTransform(grid, i_pts, j_pts)
end

# ───────────────────────────────────────────────────────────────────────────
# ZZZ — 3D Cartesian Chebyshev×Chebyshev×Chebyshev
# Mirrors RRR (including the BUG-2/BUG-3 inverse-transform structure), with the
# j/k bases as shared spectral-mode columns rather than per-radius objects.
# ───────────────────────────────────────────────────────────────────────────

"""
    getGridpoints(grid::_3DCartesianZZZ) -> Matrix{Float64}

Return a `(iDim*jDim*kDim, 3)` matrix of `(x, y, z)` mish coordinates; k varies
fastest, then j, then i: `(r-1)*jDim*kDim + (l-1)*kDim + z`.
"""
function getGridpoints(grid::_3DCartesianZZZ)
    iDim = grid.params.iDim
    jDim = grid.params.jDim
    kDim = grid.params.kDim
    pts  = zeros(Float64, iDim * jDim * kDim, 3)
    g = 1
    for r in 1:iDim
        xi = grid.ibasis.data[1, 1, 1].mishPoints[r]
        for l in 1:jDim
            yj = grid.jbasis.data[1, 1].mishPoints[l]
            for z in 1:kDim
                zk = grid.kbasis.data[1].mishPoints[z]
                pts[g, 1] = xi
                pts[g, 2] = yj
                pts[g, 3] = zk
                g += 1
            end
        end
    end
    return pts
end

"""
    spectralTransform(grid::_3DCartesianZZZ, physical, spectral)

Forward transform: k-direction `CBtransform!` first, then j, then i. Spectral
layout z-major: `(z-1)*b_jDim*b_iDim + (l-1)*b_iDim + 1`.
"""
function spectralTransform(
        grid     :: _3DCartesianZZZ,
        physical :: Array{real},
        spectral :: Array{real})
    iDim   = grid.params.iDim
    jDim   = grid.params.jDim
    kDim   = grid.params.kDim
    b_iDim = grid.params.b_iDim
    b_jDim = grid.params.b_jDim
    b_kDim = grid.params.b_kDim
    tempcb_z = zeros(Float64, b_kDim, iDim, jDim)
    tempcb_l = zeros(Float64, b_jDim, b_kDim, iDim)

    for v in 1:size(spectral, 2)
        # Step 1: k-direction transform for each (r, l) gridpoint
        kcol = grid.kbasis.data[v]
        for r in 1:iDim
            for l in 1:jDim
                @inbounds for z in 1:kDim
                    kcol.uMish[z] = physical[(r-1)*jDim*kDim + (l-1)*kDim + z, v, 1]
                end
                CBtransform!(kcol)
                @inbounds for k in 1:b_kDim
                    tempcb_z[k, r, l] = kcol.b[k]
                end
            end
        end

        # Step 2: j-direction transform for each (r, z_coeff)
        for z in 1:b_kDim
            jcol = grid.jbasis.data[z, v]
            for r in 1:iDim
                @inbounds for l in 1:jDim
                    jcol.uMish[l] = tempcb_z[z, r, l]
                end
                CBtransform!(jcol)
                @inbounds for k in 1:b_jDim
                    tempcb_l[k, z, r] = jcol.b[k]
                end
            end
        end

        # Step 3: i-direction transform for each (l_coeff, z_coeff)
        for z in 1:b_kDim
            for l in 1:b_jDim
                isp = grid.ibasis.data[l, z, v]
                @inbounds for r in 1:iDim
                    isp.uMish[r] = tempcb_l[l, z, r]
                end
                CBtransform!(isp)
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
    spectralTransform!(grid::_3DCartesianZZZ)

In-place forward transform for a 3-D Cartesian Chebyshev³ grid.
"""
function spectralTransform!(grid::_3DCartesianZZZ)
    _filter_mish!(grid)
    spectralTransform(grid, grid.physical, grid.spectral)
    applyFilter!(grid)
    return grid.spectral
end

"""
    gridTransform(grid::_3DCartesianZZZ, physical, spectral)

Inverse transform (7 slots): i-direction first, then j, then k. The k-transform
is nested inside the r-loop (BUG-3 fix) and the j-derivative slots are computed
via a separate k-inverse pass on the j-derivative coefficients (BUG-2 fix).
"""
function gridTransform(
        grid     :: _3DCartesianZZZ,
        physical :: Array{real},
        spectral :: Array{real})
    iDim   = grid.params.iDim
    jDim   = grid.params.jDim
    kDim   = grid.params.kDim
    b_iDim = grid.params.b_iDim
    b_jDim = grid.params.b_jDim
    b_kDim = grid.params.b_kDim

    splineBuffer_r     = zeros(Float64, iDim, b_jDim, b_kDim)
    splineBuffer_l     = zeros(Float64, jDim, b_kDim)
    splineBuffer_l_1st = zeros(Float64, jDim, b_kDim)
    splineBuffer_l_2nd = zeros(Float64, jDim, b_kDim)
    scratch_i          = zeros(Float64, iDim)
    scratch_j          = zeros(Float64, jDim)

    for v in 1:size(spectral, 2)
        for dr in 0:2
            # ── Step 1: i-direction inverse transform ─────────────────────────
            for z in 1:b_kDim
                for l in 1:b_jDim
                    idx = (z-1)*b_jDim*b_iDim + (l-1)*b_iDim + 1
                    isp = grid.ibasis.data[l, z, v]
                    copyto!(isp.b, view(spectral, idx:idx+b_iDim-1, v))
                    CAtransform!(isp)
                    if dr == 0
                        CItransform!(isp)
                        @inbounds for r in 1:iDim
                            splineBuffer_r[r, l, z] = isp.uMish[r]
                        end
                    elseif dr == 1
                        CIxtransform(isp, scratch_i)
                        @inbounds for r in 1:iDim
                            splineBuffer_r[r, l, z] = scratch_i[r]
                        end
                    else
                        CIxxtransform(isp, scratch_i)
                        @inbounds for r in 1:iDim
                            splineBuffer_r[r, l, z] = scratch_i[r]
                        end
                    end
                end
            end

            # ── Steps 2+3: j and k direction transforms (k nested in r-loop) ──
            for r in 1:iDim
                # j-direction transform per z_coeff
                for z in 1:b_kDim
                    jcol = grid.jbasis.data[z, v]
                    @inbounds for l in 1:b_jDim
                        jcol.b[l] = splineBuffer_r[r, l, z]
                    end
                    CAtransform!(jcol)
                    CItransform!(jcol)
                    @inbounds for l in 1:jDim
                        splineBuffer_l[l, z] = jcol.uMish[l]
                    end

                    if dr == 0
                        CIxtransform(jcol, scratch_j)
                        @inbounds for l in 1:jDim
                            splineBuffer_l_1st[l, z] = scratch_j[l]
                        end
                        CIxxtransform(jcol, scratch_j)
                        @inbounds for l in 1:jDim
                            splineBuffer_l_2nd[l, z] = scratch_j[l]
                        end
                    end
                end

                # k-direction transform per (r, l) gridpoint
                for l in 1:jDim
                    kcol = grid.kbasis.data[v]
                    @inbounds for zb in 1:b_kDim
                        kcol.b[zb] = splineBuffer_l[l, zb]
                    end
                    CAtransform!(kcol)
                    CItransform!(kcol)

                    i_flat     = (r-1)*jDim*kDim + (l-1)*kDim + 1
                    i_flat_end = i_flat + kDim - 1
                    if dr == 0
                        copyto!(view(physical, i_flat:i_flat_end, v, 1), kcol.uMish)
                        # Reuse kcol.uMish — its prior content was just copied above.
                        CIxtransform(kcol, kcol.uMish)
                        copyto!(view(physical, i_flat:i_flat_end, v, 6), kcol.uMish)
                        CIxxtransform(kcol, kcol.uMish)
                        copyto!(view(physical, i_flat:i_flat_end, v, 7), kcol.uMish)

                        # BUG-2 fix: j-derivative slots via k-inverse of j-deriv coeffs
                        @inbounds for zb in 1:b_kDim
                            kcol.b[zb] = splineBuffer_l_1st[l, zb]
                        end
                        CAtransform!(kcol)
                        CItransform!(kcol)
                        copyto!(view(physical, i_flat:i_flat_end, v, 4), kcol.uMish)

                        @inbounds for zb in 1:b_kDim
                            kcol.b[zb] = splineBuffer_l_2nd[l, zb]
                        end
                        CAtransform!(kcol)
                        CItransform!(kcol)
                        copyto!(view(physical, i_flat:i_flat_end, v, 5), kcol.uMish)
                    elseif dr == 1
                        copyto!(view(physical, i_flat:i_flat_end, v, 2), kcol.uMish)
                    else
                        copyto!(view(physical, i_flat:i_flat_end, v, 3), kcol.uMish)
                    end
                end
            end  # for r
        end  # for dr
    end  # for v
    return physical
end

"""
    gridTransform!(grid::_3DCartesianZZZ)

In-place inverse transform for a 3-D Cartesian Chebyshev³ grid (7 slots).
"""
function gridTransform!(grid::_3DCartesianZZZ)
    gridTransform(grid, grid.physical, grid.spectral)
    return grid.physical
end

"""
    getRegularGridpoints(grid::_3DCartesianZZZ) -> Matrix{Float64}

Return an `(i_regular_out × j_regular_out × k_regular_out, 3)` matrix of
uniformly-spaced `(x, y, z)` coordinates; z varies fastest, then y, then x.
"""
function getRegularGridpoints(grid::_3DCartesianZZZ)
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

"""
    regularGridTransform(grid::_3DCartesianZZZ, x_pts, y_pts, z_pts) -> Array{Float64}
    regularGridTransform(grid::_3DCartesianZZZ, gridpoints)           -> Array{Float64}

Evaluate the spectral representation on a regular `x × y × z` grid. Output shape
`(n_x × n_y × n_z, nvars, 7)`, z varies fastest. Slots follow the RRR convention.
"""
function regularGridTransform(grid::_3DCartesianZZZ,
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
            # ── Step 1: i-direction Chebyshev evaluation at x_pts ────────────
            ibuf = zeros(Float64, n_x, b_jDim, b_kDim)
            for l in 1:b_jDim
                for z_b in 1:b_kDim
                    idx = (z_b - 1) * b_jDim * b_iDim + (l - 1) * b_iDim + 1
                    sp = grid.ibasis.data[l, z_b, v]
                    sp.b .= view(grid.spectral, idx:idx + b_iDim - 1, v)
                    CAtransform!(sp)
                    _cheb_eval!(sp, x_vec, dr, view(ibuf, :, l, z_b))
                end
            end

            # ── Steps 2 & 3: j and k evaluations ─────────────────────────────
            dl_range = (dr == 0) ? (0:2) : (0:0)
            for dl in dl_range
                jbuf = zeros(Float64, n_x, n_y, b_kDim)
                jsp  = grid.jbasis.data[1, v]   # scratch column (shared params)
                for xi in 1:n_x
                    for z_b in 1:b_kDim
                        for l in 1:b_jDim
                            jsp.b[l] = ibuf[xi, l, z_b]
                        end
                        CAtransform!(jsp)
                        _cheb_eval!(jsp, y_vec, dl, view(jbuf, xi, :, z_b))
                    end
                end

                dk_range = (dr == 0 && dl == 0) ? (0:2) : (0:0)
                for dk in dk_range
                    slot = _rrr_regular_slot(dr, dl, dk)
                    slot == 0 && continue
                    ksp = grid.kbasis.data[v]   # scratch column
                    for xi in 1:n_x
                        for yj in 1:n_y
                            for z_b in 1:b_kDim
                                ksp.b[z_b] = jbuf[xi, yj, z_b]
                            end
                            CAtransform!(ksp)
                            flat = (xi - 1) * n_y * n_z + (yj - 1) * n_z + 1
                            _cheb_eval!(ksp, z_vec, dk, view(physical, flat:flat + n_z - 1, v, slot))
                        end
                    end
                end
            end
        end   # dr
    end   # v
    return physical
end

function regularGridTransform(grid::_3DCartesianZZZ, gridpoints::AbstractMatrix{Float64})
    x_pts = sort(unique(gridpoints[:, 1]))
    y_pts = sort(unique(gridpoints[:, 2]))
    z_pts = sort(unique(gridpoints[:, 3]))
    return regularGridTransform(grid, x_pts, y_pts, z_pts)
end

# ═══════════════════════════════════════════════════════════════════════════
# Fourier Cartesian transforms (L / LL / LLZ)
#
# These mirror the spline/Chebyshev Cartesian transforms above, substituting the
# Fourier primitives (FBtransform!/FAtransform!/FItransform!/FIxtransform/
# FIxxtransform) for the periodic axes. LLZ keeps a Chebyshev k-axis (reusing the
# Chebyshev primitives). Index/slot conventions are identical to the other
# families: k-fastest physical ordering, z-major spectral layout, and the 3/5/7
# derivative-slot layout.
#
# Structural note: each Fourier dimension carries a single `Fourier1D` ring per
# variable (ibasis.data[v]/jbasis.data[v]); the periodic domain is `[ymin, ymin+2π)`
# (period 2π, independent of iMax). The grid's own inverse FFT plan evaluates at
# the ring mish points; arbitrary-point (regular-grid) evaluation uses the analytic
# half-complex series, with the angle measured from the ring's `ymin`.
# ═══════════════════════════════════════════════════════════════════════════

# ── Type aliases ────────────────────────────────────────────────────────────
const _1DCartesianL = SpringsteelGrid{CartesianGeometry, FourierBasisArray{1}, NoBasisArray, NoBasisArray}
const _2DCartesianLL = SpringsteelGrid{CartesianGeometry, FourierBasisArray{1}, FourierBasisArray{1}, NoBasisArray}
const _3DCartesianLLZ = SpringsteelGrid{CartesianGeometry, FourierBasisArray{1}, FourierBasisArray{1}, ChebyshevBasisArray{1}}

# Evaluate a Fourier ring's half-complex series (from its b-coefficients) and its
# derivatives at arbitrary points. The forward phase filter references `b` to the
# absolute angle (it rotates by -k·ymin), so points are used directly — no offset.
# Matches `FItransform_matrix` exactly, allocation-free.
@inline function _fourier_eval!(ring, pts, deriv, out)
    fp = ring.params
    b  = ring.b
    @inbounds for i in eachindex(pts)
        θ = pts[i]
        val = 0.0
        if deriv == 0
            val = b[1]
            for k in 1:fp.kmax
                val += 2.0 * b[k+1] * cos(k*θ) - 2.0 * b[fp.bDim-k+1] * sin(k*θ)
            end
        elseif deriv == 1
            for k in 1:fp.kmax
                val += -2.0 * k * b[k+1] * sin(k*θ) - 2.0 * k * b[fp.bDim-k+1] * cos(k*θ)
            end
        else
            for k in 1:fp.kmax
                val += -2.0 * k^2 * b[k+1] * cos(k*θ) + 2.0 * k^2 * b[fp.bDim-k+1] * sin(k*θ)
            end
        end
        out[i] = val
    end
    return out
end

# ───────────────────────────────────────────────────────────────────────────
# L — 1D Cartesian Fourier
# ───────────────────────────────────────────────────────────────────────────

"""
    getGridpoints(grid::SpringsteelGrid{CartesianGeometry, FourierBasisArray, NoBasisArray, NoBasisArray}) -> Vector{Float64}

Return the evenly-spaced ring mish points for a 1-D Cartesian Fourier grid
(`L_Grid` / `Ring1D_Grid`), spanning `[iMin, iMin+2π)`.
"""
function getGridpoints(grid::_1DCartesianL)
    return grid.ibasis.data[1].mishPoints
end

"""
    getRegularGridpoints(grid::_1DCartesianL) -> Vector{Float64}

Return `i_regular_out` evenly-spaced output locations across the periodic domain
`[iMin, iMin+2π)`.
"""
function getRegularGridpoints(grid::_1DCartesianL)
    n  = grid.params.i_regular_out
    x0 = grid.params.iMin
    return [x0 + 2π * (i - 1) / n for i in 1:n]
end

"""
    spectralTransform(grid::_1DCartesianL, physical, spectral)

Explicit-array forward transform for a 1-D Cartesian Fourier grid. Applies
`FBtransform!` per variable, writing half-complex coefficients into `spectral`.
"""
function spectralTransform(
        grid     :: _1DCartesianL,
        physical :: Array{real},
        spectral :: Array{real})
    nvars = size(spectral, 2)
    for v in 1:nvars
        ring = grid.ibasis.data[v]
        @inbounds for i in eachindex(ring.uMish)
            ring.uMish[i] = physical[i, v, 1]
        end
        FBtransform!(ring)
        @inbounds for i in eachindex(ring.b)
            spectral[i, v] = ring.b[i]
        end
    end
    return spectral
end

"""
    spectralTransform!(grid::_1DCartesianL)

In-place forward transform for a 1-D Cartesian Fourier grid.
"""
function spectralTransform!(grid::_1DCartesianL)
    _filter_mish!(grid)
    spectralTransform(grid, grid.physical, grid.spectral)
    applyFilter!(grid)
    return grid.spectral
end

"""
    gridTransform(grid::_1DCartesianL, physical, spectral)

Explicit-array inverse transform. For each variable: `FAtransform!` (B→A), then
`FItransform!` / `FIxtransform` / `FIxxtransform` at the ring mish points, writing
value and first/second derivatives into slots 1/2/3.
"""
function gridTransform(
        grid     :: _1DCartesianL,
        physical :: Array{real},
        spectral :: Array{real})
    nvars = size(spectral, 2)
    for v in 1:nvars
        ring = grid.ibasis.data[v]
        copyto!(ring.b, view(spectral, :, v))
        FAtransform!(ring)
        FItransform!(ring)
        @inbounds for i in eachindex(ring.uMish)
            physical[i, v, 1] = ring.uMish[i]
        end
        FIxtransform(ring, view(physical, :, v, 2))
        FIxxtransform(ring, view(physical, :, v, 3))
    end
    return physical
end

"""
    gridTransform!(grid::_1DCartesianL)

In-place inverse transform for a 1-D Cartesian Fourier grid (slots 1/2/3).
"""
function gridTransform!(grid::_1DCartesianL)
    gridTransform(grid, grid.physical, grid.spectral)
    return grid.physical
end

"""
    regularGridTransform(grid::_1DCartesianL, gridpoints::AbstractVector{Float64}) -> Array{Float64}

Evaluate the Fourier representation (and first/second derivatives) at arbitrary
output locations. `grid.spectral` must be populated.
"""
function regularGridTransform(grid::_1DCartesianL, gridpoints::AbstractVector{Float64})
    nvars    = length(grid.params.vars)
    gpts     = collect(Float64, gridpoints)
    physical = zeros(Float64, length(gpts), nvars, 3)
    for v in 1:nvars
        ring = grid.ibasis.data[v]
        ring.b .= view(grid.spectral, :, v)
        _fourier_eval!(ring, gpts, 0, view(physical, :, v, 1))
        _fourier_eval!(ring, gpts, 1, view(physical, :, v, 2))
        _fourier_eval!(ring, gpts, 2, view(physical, :, v, 3))
    end
    return physical
end

# ───────────────────────────────────────────────────────────────────────────
# LL — 2D Cartesian Fourier×Fourier (i, j active)
# ───────────────────────────────────────────────────────────────────────────

"""
    getGridpoints(grid::_2DCartesianLL) -> Matrix{Float64}

Return a `(iDim*jDim, 2)` matrix of `(x, y)` ring coordinates; j varies fastest,
flat index `(r-1)*jDim + l`.
"""
function getGridpoints(grid::_2DCartesianLL)
    iDim = grid.params.iDim
    jDim = grid.params.jDim
    pts  = zeros(Float64, iDim * jDim, 2)
    g = 1
    for r in 1:iDim
        xi = grid.ibasis.data[1].mishPoints[r]
        for l in 1:jDim
            pts[g, 1] = xi
            pts[g, 2] = grid.jbasis.data[1].mishPoints[l]
            g += 1
        end
    end
    return pts
end

"""
    spectralTransform(grid::_2DCartesianLL, physical, spectral)

Forward transform. Step 1: j-direction `FBtransform!` per i gridpoint. Step 2:
i-direction `FBtransform!` per j-mode. Spectral layout: consecutive `b_iDim`
blocks per j-mode at `(l-1)*b_iDim+1`. Physical layout `(r-1)*jDim + l`.
"""
function spectralTransform(
        grid     :: _2DCartesianLL,
        physical :: Array{real},
        spectral :: Array{real})
    iDim   = grid.params.iDim
    jDim   = grid.params.jDim
    b_iDim = grid.params.b_iDim
    b_jDim = grid.params.b_jDim
    nvars  = size(spectral, 2)
    tempjb = zeros(Float64, b_jDim, iDim)

    for v in 1:nvars
        jring = grid.jbasis.data[v]
        for r in 1:iDim
            @inbounds for l in 1:jDim
                jring.uMish[l] = physical[(r-1)*jDim + l, v, 1]
            end
            FBtransform!(jring)
            @inbounds for k in 1:b_jDim
                tempjb[k, r] = jring.b[k]
            end
        end

        iring = grid.ibasis.data[v]
        for l in 1:b_jDim
            @inbounds for r in 1:iDim
                iring.uMish[r] = tempjb[l, r]
            end
            FBtransform!(iring)
            r1 = (l-1)*b_iDim + 1
            @inbounds for k in 0:(b_iDim - 1)
                spectral[r1 + k, v] = iring.b[k + 1]
            end
        end
    end
    return spectral
end

"""
    spectralTransform!(grid::_2DCartesianLL)

In-place forward transform for a 2-D Cartesian Fourier×Fourier grid.
"""
function spectralTransform!(grid::_2DCartesianLL)
    _filter_mish!(grid)
    spectralTransform(grid, grid.physical, grid.spectral)
    applyFilter!(grid)
    return grid.spectral
end

"""
    gridTransform(grid::_2DCartesianLL, physical, spectral)

Inverse transform (5 slots [f, ∂i, ∂²i, ∂j, ∂²j]). Step 1: i-direction inverse
per j-mode into `buffer[iDim, b_jDim]`. Step 2: j-direction inverse per i
gridpoint into the physical slots.
"""
function gridTransform(
        grid     :: _2DCartesianLL,
        physical :: Array{real},
        spectral :: Array{real})
    iDim   = grid.params.iDim
    jDim   = grid.params.jDim
    b_iDim = grid.params.b_iDim
    b_jDim = grid.params.b_jDim
    nvars  = size(spectral, 2)
    buffer    = zeros(Float64, iDim, b_jDim)
    scratch_i = zeros(Float64, iDim)

    for v in 1:nvars
        for dr in 0:2
            iring = grid.ibasis.data[v]
            for l in 1:b_jDim
                r1 = (l-1)*b_iDim + 1
                r2 = r1 + b_iDim - 1
                copyto!(iring.b, view(spectral, r1:r2, v))
                FAtransform!(iring)
                if dr == 0
                    FItransform!(iring)
                    @inbounds for r in 1:iDim
                        buffer[r, l] = iring.uMish[r]
                    end
                elseif dr == 1
                    FIxtransform(iring, scratch_i)
                    @inbounds for r in 1:iDim
                        buffer[r, l] = scratch_i[r]
                    end
                else
                    FIxxtransform(iring, scratch_i)
                    @inbounds for r in 1:iDim
                        buffer[r, l] = scratch_i[r]
                    end
                end
            end

            jring = grid.jbasis.data[v]
            for r in 1:iDim
                @inbounds for l in 1:b_jDim
                    jring.b[l] = buffer[r, l]
                end
                FAtransform!(jring)
                FItransform!(jring)
                l1 = (r-1)*jDim + 1
                l2 = l1 + jDim - 1
                if dr == 0
                    copyto!(view(physical, l1:l2, v, 1), jring.uMish)
                    # Reuse jring.uMish as scratch — its prior content was just copied.
                    FIxtransform(jring, jring.uMish)
                    copyto!(view(physical, l1:l2, v, 4), jring.uMish)
                    FIxxtransform(jring, jring.uMish)
                    copyto!(view(physical, l1:l2, v, 5), jring.uMish)
                elseif dr == 1
                    copyto!(view(physical, l1:l2, v, 2), jring.uMish)
                else
                    copyto!(view(physical, l1:l2, v, 3), jring.uMish)
                end
            end
        end
    end
    return physical
end

"""
    gridTransform!(grid::_2DCartesianLL)

In-place inverse transform for a 2-D Cartesian Fourier×Fourier grid (5 slots).
"""
function gridTransform!(grid::_2DCartesianLL)
    gridTransform(grid, grid.physical, grid.spectral)
    return grid.physical
end

"""
    getRegularGridpoints(grid::_2DCartesianLL) -> Matrix{Float64}

Return an `(i_regular_out × j_regular_out, 2)` matrix of evenly-spaced `(x, y)`
coordinates across the doubly-periodic domain; y varies fastest.
"""
function getRegularGridpoints(grid::_2DCartesianLL)
    n_i   = grid.params.i_regular_out
    n_j   = grid.params.j_regular_out
    x0    = grid.params.iMin
    y0    = grid.params.jMin
    i_pts = [x0 + 2π * (i - 1) / n_i for i in 1:n_i]
    j_pts = [y0 + 2π * (j - 1) / n_j for j in 1:n_j]
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
    regularGridTransform(grid::_2DCartesianLL, i_pts, j_pts) -> Array{Float64}
    regularGridTransform(grid::_2DCartesianLL, gridpoints)   -> Array{Float64}

Evaluate the spectral representation on a regular `x × y` grid, returning values
and all five derivatives. Output shape `(n_i × n_j, nvars, 5)`, y varies fastest.
"""
function regularGridTransform(grid::_2DCartesianLL,
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
        ibuf  = zeros(Float64, n_i, b_jDim)
        iring = grid.ibasis.data[v]
        jring = grid.jbasis.data[v]
        for dr in 0:2
            for l in 1:b_jDim
                r1 = (l - 1) * b_iDim + 1
                r2 = r1 + b_iDim - 1
                iring.b .= view(grid.spectral, r1:r2, v)
                _fourier_eval!(iring, i_vec, dr, view(ibuf, :, l))
            end

            dj_range = (dr == 0) ? (0:2) : (0:0)
            for dj in dj_range
                slot = _rz_slot(dr, dj)
                slot == 0 && continue
                for xi in 1:n_i
                    for l in 1:b_jDim
                        jring.b[l] = ibuf[xi, l]
                    end
                    flat = (xi - 1) * n_j + 1
                    _fourier_eval!(jring, j_vec, dj,
                                   view(physical, flat:flat + n_j - 1, v, slot))
                end
            end
        end
    end
    return physical
end

function regularGridTransform(grid::_2DCartesianLL, gridpoints::AbstractMatrix{Float64})
    i_pts = sort(unique(gridpoints[:, 1]))
    j_pts = sort(unique(gridpoints[:, 2]))
    return regularGridTransform(grid, i_pts, j_pts)
end

# ───────────────────────────────────────────────────────────────────────────
# LLZ — 3D Cartesian Fourier×Fourier×Chebyshev (doubly-periodic + vertical)
# Mirrors the ZZZ/RRR inverse structure (BUG-2/BUG-3 fixes) with i,j Fourier and
# the k-axis Chebyshev.
# ───────────────────────────────────────────────────────────────────────────

"""
    getGridpoints(grid::_3DCartesianLLZ) -> Matrix{Float64}

Return a `(iDim*jDim*kDim, 3)` matrix of `(x, y, z)` coordinates; k varies fastest,
then j, then i: `(r-1)*jDim*kDim + (l-1)*kDim + z`.
"""
function getGridpoints(grid::_3DCartesianLLZ)
    iDim = grid.params.iDim
    jDim = grid.params.jDim
    kDim = grid.params.kDim
    pts  = zeros(Float64, iDim * jDim * kDim, 3)
    g = 1
    for r in 1:iDim
        xi = grid.ibasis.data[1].mishPoints[r]
        for l in 1:jDim
            yj = grid.jbasis.data[1].mishPoints[l]
            for z in 1:kDim
                zk = grid.kbasis.data[1].mishPoints[z]
                pts[g, 1] = xi
                pts[g, 2] = yj
                pts[g, 3] = zk
                g += 1
            end
        end
    end
    return pts
end

"""
    spectralTransform(grid::_3DCartesianLLZ, physical, spectral)

Forward transform: k-direction `CBtransform!` first, then j (`FBtransform!`), then
i (`FBtransform!`). Spectral layout z-major: `(z-1)*b_jDim*b_iDim + (l-1)*b_iDim + 1`.
"""
function spectralTransform(
        grid     :: _3DCartesianLLZ,
        physical :: Array{real},
        spectral :: Array{real})
    iDim   = grid.params.iDim
    jDim   = grid.params.jDim
    kDim   = grid.params.kDim
    b_iDim = grid.params.b_iDim
    b_jDim = grid.params.b_jDim
    b_kDim = grid.params.b_kDim
    tempck = zeros(Float64, b_kDim, iDim, jDim)
    tempjb = zeros(Float64, b_jDim, b_kDim, iDim)

    for v in 1:size(spectral, 2)
        # Step 1: k-direction (Chebyshev) transform for each (r, l) gridpoint
        kcol = grid.kbasis.data[v]
        for r in 1:iDim
            for l in 1:jDim
                @inbounds for z in 1:kDim
                    kcol.uMish[z] = physical[(r-1)*jDim*kDim + (l-1)*kDim + z, v, 1]
                end
                CBtransform!(kcol)
                @inbounds for k in 1:b_kDim
                    tempck[k, r, l] = kcol.b[k]
                end
            end
        end

        # Step 2: j-direction (Fourier) transform for each (r, z_coeff)
        jring = grid.jbasis.data[v]
        for z in 1:b_kDim
            for r in 1:iDim
                @inbounds for l in 1:jDim
                    jring.uMish[l] = tempck[z, r, l]
                end
                FBtransform!(jring)
                @inbounds for k in 1:b_jDim
                    tempjb[k, z, r] = jring.b[k]
                end
            end
        end

        # Step 3: i-direction (Fourier) transform for each (l_coeff, z_coeff)
        iring = grid.ibasis.data[v]
        for z in 1:b_kDim
            for l in 1:b_jDim
                @inbounds for r in 1:iDim
                    iring.uMish[r] = tempjb[l, z, r]
                end
                FBtransform!(iring)
                idx = (z-1)*b_jDim*b_iDim + (l-1)*b_iDim + 1
                @inbounds for k in 0:(b_iDim - 1)
                    spectral[idx + k, v] = iring.b[k + 1]
                end
            end
        end
    end
    return spectral
end

"""
    spectralTransform!(grid::_3DCartesianLLZ)

In-place forward transform for a 3-D Cartesian Fourier×Fourier×Chebyshev grid.
"""
function spectralTransform!(grid::_3DCartesianLLZ)
    _filter_mish!(grid)
    spectralTransform(grid, grid.physical, grid.spectral)
    applyFilter!(grid)
    return grid.spectral
end

"""
    gridTransform(grid::_3DCartesianLLZ, physical, spectral)

Inverse transform (7 slots): i-direction (Fourier) first, then j (Fourier), then k
(Chebyshev). The k-transform is nested inside the r-loop (BUG-3 fix) and the
j-derivative slots are computed via a separate k-inverse pass (BUG-2 fix).
"""
function gridTransform(
        grid     :: _3DCartesianLLZ,
        physical :: Array{real},
        spectral :: Array{real})
    iDim   = grid.params.iDim
    jDim   = grid.params.jDim
    kDim   = grid.params.kDim
    b_iDim = grid.params.b_iDim
    b_jDim = grid.params.b_jDim
    b_kDim = grid.params.b_kDim

    buffer_r     = zeros(Float64, iDim, b_jDim, b_kDim)
    buffer_l     = zeros(Float64, jDim, b_kDim)
    buffer_l_1st = zeros(Float64, jDim, b_kDim)
    buffer_l_2nd = zeros(Float64, jDim, b_kDim)
    scratch_i    = zeros(Float64, iDim)
    scratch_j    = zeros(Float64, jDim)

    for v in 1:size(spectral, 2)
        for dr in 0:2
            # ── Step 1: i-direction (Fourier) inverse transform ───────────────
            iring = grid.ibasis.data[v]
            for z in 1:b_kDim
                for l in 1:b_jDim
                    idx = (z-1)*b_jDim*b_iDim + (l-1)*b_iDim + 1
                    copyto!(iring.b, view(spectral, idx:idx+b_iDim-1, v))
                    FAtransform!(iring)
                    if dr == 0
                        FItransform!(iring)
                        @inbounds for r in 1:iDim
                            buffer_r[r, l, z] = iring.uMish[r]
                        end
                    elseif dr == 1
                        FIxtransform(iring, scratch_i)
                        @inbounds for r in 1:iDim
                            buffer_r[r, l, z] = scratch_i[r]
                        end
                    else
                        FIxxtransform(iring, scratch_i)
                        @inbounds for r in 1:iDim
                            buffer_r[r, l, z] = scratch_i[r]
                        end
                    end
                end
            end

            # ── Steps 2+3: j (Fourier) and k (Chebyshev), k nested in r-loop ──
            for r in 1:iDim
                jring = grid.jbasis.data[v]
                for z in 1:b_kDim
                    @inbounds for l in 1:b_jDim
                        jring.b[l] = buffer_r[r, l, z]
                    end
                    FAtransform!(jring)
                    FItransform!(jring)
                    @inbounds for l in 1:jDim
                        buffer_l[l, z] = jring.uMish[l]
                    end

                    if dr == 0
                        FIxtransform(jring, scratch_j)
                        @inbounds for l in 1:jDim
                            buffer_l_1st[l, z] = scratch_j[l]
                        end
                        FIxxtransform(jring, scratch_j)
                        @inbounds for l in 1:jDim
                            buffer_l_2nd[l, z] = scratch_j[l]
                        end
                    end
                end

                kcol = grid.kbasis.data[v]
                for l in 1:jDim
                    @inbounds for zb in 1:b_kDim
                        kcol.b[zb] = buffer_l[l, zb]
                    end
                    CAtransform!(kcol)
                    CItransform!(kcol)

                    i_flat     = (r-1)*jDim*kDim + (l-1)*kDim + 1
                    i_flat_end = i_flat + kDim - 1
                    if dr == 0
                        copyto!(view(physical, i_flat:i_flat_end, v, 1), kcol.uMish)
                        # Reuse kcol.uMish — its prior content was just copied above.
                        CIxtransform(kcol, kcol.uMish)
                        copyto!(view(physical, i_flat:i_flat_end, v, 6), kcol.uMish)
                        CIxxtransform(kcol, kcol.uMish)
                        copyto!(view(physical, i_flat:i_flat_end, v, 7), kcol.uMish)

                        # BUG-2 fix: j-derivative slots via k-inverse of j-deriv coeffs
                        @inbounds for zb in 1:b_kDim
                            kcol.b[zb] = buffer_l_1st[l, zb]
                        end
                        CAtransform!(kcol)
                        CItransform!(kcol)
                        copyto!(view(physical, i_flat:i_flat_end, v, 4), kcol.uMish)

                        @inbounds for zb in 1:b_kDim
                            kcol.b[zb] = buffer_l_2nd[l, zb]
                        end
                        CAtransform!(kcol)
                        CItransform!(kcol)
                        copyto!(view(physical, i_flat:i_flat_end, v, 5), kcol.uMish)
                    elseif dr == 1
                        copyto!(view(physical, i_flat:i_flat_end, v, 2), kcol.uMish)
                    else
                        copyto!(view(physical, i_flat:i_flat_end, v, 3), kcol.uMish)
                    end
                end
            end  # for r
        end  # for dr
    end  # for v
    return physical
end

"""
    gridTransform!(grid::_3DCartesianLLZ)

In-place inverse transform for a 3-D Cartesian Fourier×Fourier×Chebyshev grid (7 slots).
"""
function gridTransform!(grid::_3DCartesianLLZ)
    gridTransform(grid, grid.physical, grid.spectral)
    return grid.physical
end

"""
    getRegularGridpoints(grid::_3DCartesianLLZ) -> Matrix{Float64}

Return an `(i_regular_out × j_regular_out × k_regular_out, 3)` matrix of `(x, y, z)`
coordinates: x, y evenly spaced across the periodic domain, z uniform in `[kMin, kMax]`;
z varies fastest, then y, then x.
"""
function getRegularGridpoints(grid::_3DCartesianLLZ)
    n_x   = grid.params.i_regular_out
    n_y   = grid.params.j_regular_out
    n_z   = grid.params.k_regular_out
    x0    = grid.params.iMin
    y0    = grid.params.jMin
    x_pts = [x0 + 2π * (i - 1) / n_x for i in 1:n_x]
    y_pts = [y0 + 2π * (j - 1) / n_y for j in 1:n_y]
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

"""
    regularGridTransform(grid::_3DCartesianLLZ, x_pts, y_pts, z_pts) -> Array{Float64}
    regularGridTransform(grid::_3DCartesianLLZ, gridpoints)           -> Array{Float64}

Evaluate the spectral representation on a regular `x × y × z` grid. Output shape
`(n_x × n_y × n_z, nvars, 7)`, z varies fastest. Slots follow the RRR convention.
"""
function regularGridTransform(grid::_3DCartesianLLZ,
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
        iring = grid.ibasis.data[v]
        jring = grid.jbasis.data[v]
        for dr in 0:2
            # ── Step 1: i-direction Fourier evaluation at x_pts ──────────────
            ibuf = zeros(Float64, n_x, b_jDim, b_kDim)
            for l in 1:b_jDim
                for z_b in 1:b_kDim
                    idx = (z_b - 1) * b_jDim * b_iDim + (l - 1) * b_iDim + 1
                    iring.b .= view(grid.spectral, idx:idx + b_iDim - 1, v)
                    _fourier_eval!(iring, x_vec, dr, view(ibuf, :, l, z_b))
                end
            end

            # ── Steps 2 & 3: j (Fourier) and k (Chebyshev) evaluations ───────
            dl_range = (dr == 0) ? (0:2) : (0:0)
            for dl in dl_range
                jbuf = zeros(Float64, n_x, n_y, b_kDim)
                for xi in 1:n_x
                    for z_b in 1:b_kDim
                        for l in 1:b_jDim
                            jring.b[l] = ibuf[xi, l, z_b]
                        end
                        _fourier_eval!(jring, y_vec, dl, view(jbuf, xi, :, z_b))
                    end
                end

                dk_range = (dr == 0 && dl == 0) ? (0:2) : (0:0)
                for dk in dk_range
                    slot = _rrr_regular_slot(dr, dl, dk)
                    slot == 0 && continue
                    kcol = grid.kbasis.data[v]
                    for xi in 1:n_x
                        for yj in 1:n_y
                            for z_b in 1:b_kDim
                                kcol.b[z_b] = jbuf[xi, yj, z_b]
                            end
                            CAtransform!(kcol)
                            flat = (xi - 1) * n_y * n_z + (yj - 1) * n_z + 1
                            out  = view(physical, flat:flat + n_z - 1, v, slot)
                            if dk == 0
                                _cheb_eval_pts!(kcol, z_vec, out)
                            elseif dk == 1
                                _cheb_dz_pts!(kcol, z_vec, out)
                            else
                                _cheb_dzz_pts!(kcol, z_vec, out)
                            end
                        end
                    end
                end
            end
        end   # dr
    end   # v
    return physical
end

function regularGridTransform(grid::_3DCartesianLLZ, gridpoints::AbstractMatrix{Float64})
    x_pts = sort(unique(gridpoints[:, 1]))
    y_pts = sort(unique(gridpoints[:, 2]))
    z_pts = sort(unique(gridpoints[:, 3]))
    return regularGridTransform(grid, x_pts, y_pts, z_pts)
end

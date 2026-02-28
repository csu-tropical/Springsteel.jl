# tiling.jl — 1-D tiling (i-dimension) for SpringsteelGrid
#
# Phase 8a: generic i-dimension tiling for all geometry types.
#
# Provides:
#   • calcTileSizes     — split a patch into tile sub-grids (Cartesian, Cylindrical, Spherical, fallback)
#   • num_columns       — number of j×k spectral columns (Cartesian grids not yet in transforms files)
#   • allocateSplineBuffer — allocate the radial derivative buffer for tile transforms
#   • getBorderSpectral — extract halo B-vector rows as a sparse matrix
#   • calcPatchMap      — SparseMatrixCSC marking inner (non-halo) tile region in patch spectral
#   • calcHaloMap       — SparseMatrixCSC marking 3-row halo between adjacent tiles
#   • sumSpectralTile!  — accumulate tile spectral into patch spectral
#   • setSpectralTile!  — zero patch then write tile spectral into it
#   • sumSharedSpectral — write tile interior + halo into a SharedArray
#   • splineTransform!  — SA (B→A) transform on shared spectral data (2-arg variant)
#   • tileTransform!    — A-coefficients → physical (4-arg variant)
#
# Must be included AFTER types.jl, factory.jl, transforms_cartesian.jl,
#   transforms_cylindrical.jl, and transforms_spherical.jl.
#
# ────────────────────────────────────────────────────────────────────────────────
# CONCURRENCY WARNINGS (see REFACTORING_PLAN.md §13.3):
#
#   RACE-1: sumSharedSpectral writes to SharedArray halo zones non-atomically.
#            Callers must serialize halo-zone writes across workers.
#
#   RACE-2: tileTransform! is @threads-safe ONLY because ibasis.data[:, v] and
#            physical[:, v, :] are fully independent per v. Do not share basis
#            objects across variables without reviewing this invariant.
#
#   RACE-3: splineTransform! reads from SharedArray. The caller must ensure
#            the shared array is fully populated before any parallel reads begin.
# ────────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# Internal helper: create a tile grid from patch params and new tile bounds
# ────────────────────────────────────────────────────────────────────────────

"""
    _create_tile_from_patch(patch, iMin, iMax, num_cells, spectralIndexL, tile_num) -> SpringsteelGrid

Internal: create a tile grid sharing the same geometry, basis type, BCs, and variables as
`patch`, but spanning the sub-domain `[iMin, iMax]` with `num_cells` spline cells.  The
`spectralIndexL` parameter determines where this tile's spectral coefficients map within the
patch spectral array.

All derived dimensions (`iDim`, `b_iDim`, `jDim`, `b_jDim`, `patchOffsetL`, etc.) are
computed automatically by [`createGrid`](@ref) / [`compute_derived_params`](@ref).
"""
function _create_tile_from_patch(patch::SpringsteelGrid,
                                  iMin::Float64, iMax::Float64,
                                  num_cells::int, spectralIndexL::int,
                                  tile_num::int)
    tile_gp = SpringsteelGridParameters(
        geometry       = patch.params.geometry,
        iMin           = iMin,
        iMax           = iMax,
        num_cells      = num_cells,
        l_q            = patch.params.l_q,
        BCL            = patch.params.BCL,
        BCR            = patch.params.BCR,
        jMin           = patch.params.jMin,
        jMax           = patch.params.jMax,
        max_wavenumber = patch.params.max_wavenumber,
        kMin           = patch.params.kMin,
        kMax           = patch.params.kMax,
        kDim           = patch.params.kDim,
        BCU            = patch.params.BCU,
        BCD            = patch.params.BCD,
        BCB            = patch.params.BCB,
        BCT            = patch.params.BCT,
        vars           = patch.params.vars,
        spectralIndexL = spectralIndexL,
        tile_num       = tile_num)
    return createGrid(tile_gp)
end

# ────────────────────────────────────────────────────────────────────────────
# calcTileSizes — Cartesian
# ────────────────────────────────────────────────────────────────────────────

"""
    calcTileSizes(patch::SpringsteelGrid{CartesianGeometry, SplineBasisArray, J, K}, num_tiles) -> Vector{SpringsteelGrid}
    calcTileSizes(patch::SpringsteelGrid{CylindricalGeometry, SplineBasisArray, J, K}, num_tiles) -> Vector{SpringsteelGrid}
    calcTileSizes(patch::SpringsteelGrid{SphericalGeometry, SplineBasisArray, J, K}, num_tiles) -> Vector{SpringsteelGrid}
    calcTileSizes(patch::SpringsteelGrid, num_tiles) -> Vector{SpringsteelGrid}

Split a patch grid into `num_tiles` sub-domain tile grids and return them as a vector.

Each tile is a fully-initialized `SpringsteelGrid` sharing the same geometry type, basis
functions, boundary conditions, and variable map as `patch`, but spanning a sub-domain
`[iMin_tile, iMax_tile]`.  The tile's `params.spectralIndexL` records its position within
the patch spectral array.

# Splitting strategies

| Geometry | Strategy |
|:-------- |:-------- |
| `CartesianGeometry` | Uniform: distribute physical gridpoints (`iDim`) evenly; each tile gets `⌈iDim/num_tiles⌉` or `⌊iDim/num_tiles⌋` gridpoints |
| `CylindricalGeometry` | Workload-balanced: distribute total Fourier ring gridpoints (`jDim`) as evenly as possible while maintaining ≥ 3 cells per tile |
| `SphericalGeometry` | Workload-balanced: same as Cylindrical using sin(θ)-weighted ring sizes |

All dispatch variants require the i-dimension to use a `SplineBasisArray` (spline basis is
required for domain decomposition).  The fallback (non-`SplineBasisArray` i-basis) returns a
single-element vector `[patch]` if `num_tiles == 1`, otherwise throws `DomainError`.

# Arguments
- `patch`: Full-domain patch grid
- `num_tiles::Int64`: Number of tiles to create

# Returns
`Vector{typeof(patch)}` of length `num_tiles`, each a valid `SpringsteelGrid` with:
- `params.num_cells >= 3` (minimum 3 spline cells per tile)
- `params.iDim >= 9` (minimum 9 physical gridpoints per tile)
- `params.spectralIndexL` set to its global offset within the patch spectral array

# Throws
- `DomainError` if the patch cannot be split into `num_tiles` tiles each with ≥ 3 cells
  (or ≥ 9 gridpoints for Cartesian), or if tiling is requested on a non-Spline i-dimension.

# Example
```julia
gp = SpringsteelGridParameters(geometry="R", num_cells=12, iMin=0.0, iMax=100.0,
                                 vars=Dict("u"=>1),
                                 BCL=Dict("u"=>CubicBSpline.R0),
                                 BCR=Dict("u"=>CubicBSpline.R0))
patch = createGrid(gp)
tiles = calcTileSizes(patch, 4)
length(tiles) == 4  # true
all(t.params.num_cells >= 3 for t in tiles)  # true
```

See also: [`calcPatchMap`](@ref), [`calcHaloMap`](@ref), [`allocateSplineBuffer`](@ref)
"""
function calcTileSizes(patch::SpringsteelGrid{CartesianGeometry, SplineBasisArray, J, K},
                        num_tiles::int) where {J, K}
    num_gridpoints = patch.params.iDim
    q, r_rem = divrem(num_gridpoints, num_tiles)
    tile_sizes = [i <= r_rem ? q+1 : q for i = 1:num_tiles]

    if any(x -> x < 9, tile_sizes)
        throw(DomainError(num_tiles,
            "Too many tiles for this grid (need at least 3 cells = 9 gridpoints per tile)"))
    end

    DX    = (patch.params.iMax - patch.params.iMin) / patch.params.num_cells
    iMins = zeros(Float64, num_tiles)
    iMaxs = zeros(Float64, num_tiles)
    nc    = zeros(Int64,   num_tiles)
    siL   = ones(Int64,    num_tiles)

    iMins[1] = patch.params.iMin
    nc[1]    = Int64(ceil(tile_sizes[1] / 3))
    iMaxs[1] = nc[1] * DX + iMins[1]

    for i in 2:num_tiles-1
        iMins[i] = iMaxs[i-1]
        nc[i]    = Int64(ceil(tile_sizes[i] / 3))
        iMaxs[i] = nc[i] * DX + iMins[i]
        siL[i]   = nc[i-1] + siL[i-1]
    end

    if num_tiles > 1
        iMins[num_tiles] = iMaxs[num_tiles-1]
        iMaxs[num_tiles] = patch.params.iMax
        siL[num_tiles]   = nc[num_tiles-1] + siL[num_tiles-1]
        nc[num_tiles]    = patch.params.num_cells - siL[num_tiles] + 1
    end

    tiles = Vector{typeof(patch)}(undef, num_tiles)
    for i in 1:num_tiles
        tiles[i] = _create_tile_from_patch(patch, iMins[i], iMaxs[i], nc[i], siL[i], i)
    end
    return tiles
end

# ────────────────────────────────────────────────────────────────────────────
# calcTileSizes — Cylindrical
# ────────────────────────────────────────────────────────────────────────────

function calcTileSizes(patch::SpringsteelGrid{CylindricalGeometry, SplineBasisArray, J, K},
                        num_tiles::int) where {J, K}
    if patch.params.num_cells / num_tiles < 3.0
        throw(DomainError(num_tiles,
            "Too many tiles for this grid (need at least 3 cells per tile)"))
    end

    DX    = (patch.params.iMax - patch.params.iMin) / patch.params.num_cells
    iMins = zeros(Float64, num_tiles)
    iMaxs = zeros(Float64, num_tiles)
    nc    = zeros(Int64,   num_tiles)
    siL   = ones(Int64,    num_tiles)

    if num_tiles == 1
        iMins[1] = patch.params.iMin
        iMaxs[1] = patch.params.iMax
        nc[1]    = patch.params.num_cells
    else
        # Compute lpoints per ring for workload balancing (identical to RL_Grid.calcTileSizes)
        lpoints = [4 + 4*(r + patch.params.patchOffsetL) for r in 1:patch.params.iDim]
        jDim    = patch.params.jDim
        q, r_rem = divrem(jDim, num_tiles)
        tile_targets = [i <= r_rem ? q+1 : q for i = 1:num_tiles]

        tile_count = zeros(Int64, num_tiles)
        nc[:] .= 0
        target = 1.0
        while any(nc .< 3) && target > 0.1
            t          = num_tiles
            cell_count = 0
            target    -= 0.1
            tile_count[:] .= 0
            nc[:] .= 0
            tile_min = Int64(floor(target * tile_targets[1]))
            for ri in patch.params.iDim:-1:1
                tile_count[t] += lpoints[ri]
                if ri % 3 == 0
                    cell_count += 1
                    if cell_count >= 3 && tile_count[t] >= tile_min
                        nc[t] = cell_count
                        cell_count = 0
                        t -= 1
                    end
                end
                if t == 0; break; end
            end
            nc[1] = patch.params.num_cells - sum(nc[2:end])
        end

        if any(nc .< 3)
            throw(DomainError(num_tiles,
                "Too many tiles for this grid (need at least 3 cells per tile)"))
        end

        iMins[1] = patch.params.iMin
        iMaxs[1] = nc[1] * DX + iMins[1]

        for i in 2:num_tiles-1
            iMins[i] = iMaxs[i-1]
            iMaxs[i] = nc[i] * DX + iMins[i]
            siL[i]   = nc[i-1] + siL[i-1]
        end

        iMins[num_tiles] = iMaxs[num_tiles-1]
        iMaxs[num_tiles] = patch.params.iMax
        siL[num_tiles]   = nc[num_tiles-1] + siL[num_tiles-1]
        nc[num_tiles]    = patch.params.num_cells - sum(nc[1:end-1])
    end

    tiles = Vector{typeof(patch)}(undef, num_tiles)
    for i in 1:num_tiles
        tiles[i] = _create_tile_from_patch(patch, iMins[i], iMaxs[i], nc[i], siL[i], i)
    end
    return tiles
end

# ────────────────────────────────────────────────────────────────────────────
# calcTileSizes — Spherical (same workload-balancing logic as Cylindrical)
# ────────────────────────────────────────────────────────────────────────────

function calcTileSizes(patch::SpringsteelGrid{SphericalGeometry, SplineBasisArray, J, K},
                        num_tiles::int) where {J, K}
    if patch.params.num_cells / num_tiles < 3.0
        throw(DomainError(num_tiles,
            "Too many tiles for this grid (need at least 3 cells per tile)"))
    end

    DX    = (patch.params.iMax - patch.params.iMin) / patch.params.num_cells
    iMins = zeros(Float64, num_tiles)
    iMaxs = zeros(Float64, num_tiles)
    nc    = zeros(Int64,   num_tiles)
    siL   = ones(Int64,    num_tiles)

    if num_tiles == 1
        iMins[1] = patch.params.iMin
        iMaxs[1] = patch.params.iMax
        nc[1]    = patch.params.num_cells
    else
        # Compute lpoints per ring using sin(θ) ring-size formula
        mishpts = patch.ibasis.data[1, 1].mishPoints
        max_ri  = patch.params.iDim + patch.params.patchOffsetL
        lpoints = [_sph_ring_dims(mishpts[r], max_ri)[1] for r in 1:patch.params.iDim]

        jDim    = patch.params.jDim
        q, r_rem = divrem(jDim, num_tiles)
        tile_targets = [i <= r_rem ? q+1 : q for i = 1:num_tiles]

        tile_count = zeros(Int64, num_tiles)
        nc[:] .= 0
        target = 1.0
        while any(nc .< 3) && target > 0.1
            t          = num_tiles
            cell_count = 0
            target    -= 0.1
            tile_count[:] .= 0
            nc[:] .= 0
            tile_min = Int64(floor(target * tile_targets[1]))
            for ri in patch.params.iDim:-1:1
                tile_count[t] += lpoints[ri]
                if ri % 3 == 0
                    cell_count += 1
                    if cell_count >= 3 && tile_count[t] >= tile_min
                        nc[t] = cell_count
                        cell_count = 0
                        t -= 1
                    end
                end
                if t == 0; break; end
            end
            nc[1] = patch.params.num_cells - sum(nc[2:end])
        end

        if any(nc .< 3)
            throw(DomainError(num_tiles,
                "Too many tiles for this grid (need at least 3 cells per tile)"))
        end

        iMins[1] = patch.params.iMin
        iMaxs[1] = nc[1] * DX + iMins[1]

        for i in 2:num_tiles-1
            iMins[i] = iMaxs[i-1]
            iMaxs[i] = nc[i] * DX + iMins[i]
            siL[i]   = nc[i-1] + siL[i-1]
        end

        iMins[num_tiles] = iMaxs[num_tiles-1]
        iMaxs[num_tiles] = patch.params.iMax
        siL[num_tiles]   = nc[num_tiles-1] + siL[num_tiles-1]
        nc[num_tiles]    = patch.params.num_cells - sum(nc[1:end-1])
    end

    tiles = Vector{typeof(patch)}(undef, num_tiles)
    for i in 1:num_tiles
        tiles[i] = _create_tile_from_patch(patch, iMins[i], iMaxs[i], nc[i], siL[i], i)
    end
    return tiles
end

# ────────────────────────────────────────────────────────────────────────────
# calcTileSizes — fallback (non-Spline i-basis)
# ────────────────────────────────────────────────────────────────────────────

function calcTileSizes(patch::SpringsteelGrid, num_tiles::int)
    num_tiles == 1 || throw(DomainError(num_tiles,
        "Tiling requires a Spline basis in the i-dimension"))
    return [patch]
end

# ────────────────────────────────────────────────────────────────────────────
# num_columns — Cartesian grids
# (Cylindrical and Spherical are defined in transforms_cylindrical.jl /
#  transforms_spherical.jl and return 0 for RL and SL.)
# ────────────────────────────────────────────────────────────────────────────

"""
    num_columns(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray}) -> Int

Return `1` for 1-D Cartesian grids.  The single radial spline corresponds to one
spectral "column" per variable.

See also: [`allocateSplineBuffer`](@ref)
"""
function num_columns(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray})
    return 1
end

"""
    num_columns(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, NoBasisArray}) -> Int

Return `b_jDim` for 2-D Cartesian Spline×Spline (RR) grids: one radial spline
column per j-spectral mode.
"""
function num_columns(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, NoBasisArray})
    return grid.params.b_jDim
end

"""
    num_columns(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, ChebyshevBasisArray}) -> Int

Return `b_kDim` for 2-D Cartesian Spline×Chebyshev (RZ) grids: one radial spline
column per Chebyshev vertical mode.
"""
function num_columns(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, ChebyshevBasisArray})
    return grid.params.b_kDim
end

"""
    num_columns(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, SplineBasisArray}) -> Int

Return `b_jDim * b_kDim` for 3-D Cartesian Spline×Spline×Spline (RRR) grids.
"""
function num_columns(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, SplineBasisArray})
    return grid.params.b_jDim * grid.params.b_kDim
end

# ────────────────────────────────────────────────────────────────────────────
# allocateSplineBuffer
# ────────────────────────────────────────────────────────────────────────────

"""
    allocateSplineBuffer(tile::SpringsteelGrid) -> Array{Float64}

Allocate the spline derivative buffer needed by [`tileTransform!`](@ref) for this tile type.

| Grid type | Buffer shape | Usage |
|:--------- |:------------ |:----- |
| 1-D Cartesian (`R`) | `(1,)` trivial | unused; included for API uniformity |
| 2-D Cylindrical (`RL`) | `(iDim, 2, nvars)` | real + imaginary radial evaluations |
| 2-D Spherical (`SL`) | `(iDim, 2, nvars)` | same as Cylindrical |
| 2-D Cartesian Spline×Spline (`RR`) | `(iDim, b_jDim, nvars)` | j-mode radial evaluations |
| 2-D Cartesian Spline×Chebyshev (`RZ`) | `(iDim, b_kDim, nvars)` | Chebyshev-mode radial evaluations |
| All other | `(1,)` trivial | fallback |

# Arguments
- `tile`: A `SpringsteelGrid` sub-domain tile (or the full patch for single-tile runs)

# Returns
`Array{Float64}` of the appropriate shape; all zeros.

See also: [`tileTransform!`](@ref), [`calcTileSizes`](@ref)
"""
function allocateSplineBuffer(tile::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray})
    return zeros(Float64, 1)
end

function allocateSplineBuffer(tile::SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray})
    return zeros(Float64, tile.params.iDim, 2, length(tile.params.vars))
end

function allocateSplineBuffer(tile::SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray})
    return zeros(Float64, tile.params.iDim, 2, length(tile.params.vars))
end

function allocateSplineBuffer(tile::SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, NoBasisArray})
    return zeros(Float64, tile.params.iDim, tile.params.b_jDim, length(tile.params.vars))
end

function allocateSplineBuffer(tile::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, ChebyshevBasisArray})
    return zeros(Float64, tile.params.iDim, tile.params.b_kDim, length(tile.params.vars))
end

# Generic fallback: 3D and any unspecialized tile types
function allocateSplineBuffer(tile::SpringsteelGrid)
    return zeros(Float64, 1)
end

# ────────────────────────────────────────────────────────────────────────────
# getBorderSpectral
# ────────────────────────────────────────────────────────────────────────────

"""
    getBorderSpectral(grid::SpringsteelGrid) -> SparseMatrixCSC{Float64, Int64}

Extract the rightmost 3 B-vector rows of the tile spectral array as a sparse matrix
keyed to the tile's global patch indices.  These rows overlap with the adjacent tile's
leftmost halo zone and must be accumulated (not exclusively written) in
[`sumSharedSpectral`](@ref).

For a tile with `b_iDim` spline coefficients and `spectralIndexR = spectralIndexL + b_iDim − 1`:
- Tile source rows: `b_iDim-2 : b_iDim`
- Patch destination rows: `spectralIndexR-2 : spectralIndexR`

For cylindrical/spherical grids the same 3-row extraction applies to every wavenumber block,
but only the wavenumber-0 block is implemented here (sufficient for 1-D halo transfer).

# Arguments
- `grid`: `SpringsteelGrid` tile (or patch for the no-halo single-tile case)

# Returns
`SparseMatrixCSC{Float64, Int64}` of size `(spectral_rows, nvars)` with 3 non-zero rows.

See also: [`sumSharedSpectral`](@ref), [`calcHaloMap`](@ref)
"""
function getBorderSpectral(grid::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray})
    n     = size(grid.spectral, 1)
    nvars = size(grid.spectral, 2)
    border_arr = zeros(Float64, n, nvars)
    biL = grid.params.spectralIndexR - 2
    biR = grid.params.spectralIndexR
    tiL = grid.params.b_iDim - 2
    tiR = grid.params.b_iDim
    if biL >= 1 && biR <= n && tiL >= 1
        border_arr[biL:biR, :] .= grid.spectral[tiL:tiR, :]
    end
    return sparse(border_arr)
end

# Generic fallback for Cylindrical, Spherical, and multi-D Cartesian grids:
# return a sparse zero matrix (avoids errors; callers that need halo extraction
# for full Cylindrical/Spherical tiling should use the specialized variants).
function getBorderSpectral(grid::SpringsteelGrid)
    n     = size(grid.spectral, 1)
    nvars = size(grid.spectral, 2)
    return spzeros(Float64, n, nvars)
end

# ────────────────────────────────────────────────────────────────────────────
# calcPatchMap
# ────────────────────────────────────────────────────────────────────────────

"""
    calcPatchMap(patch::SpringsteelGrid, tile::SpringsteelGrid) -> SparseMatrixCSC{Float64, Int64}

Return a sparse matrix marking the **inner (non-halo) region** of `tile` within the
`patch` spectral array.

The inner region is defined as `spectralIndexL : spectralIndexR - 3` in the 1-D (k=0
wavenumber) block.  The last 3 rows are the halo (overlap with the next tile) identified
by [`calcHaloMap`](@ref).  For cylindrical/spherical grids the same inner-region logic
applies to every wavenumber block.

# Returns
`SparseMatrixCSC{Float64, Int64}` of size `(spectral_rows, nvars)` with `1.0` entries
at the inner-region rows and `0.0` elsewhere.

# Note
The Cartesian 1-D method returns `nnz == (spectralIndexR-3 − spectralIndexL + 1) * nvars`;
the Cylindrical method returns `nnz == (b_iDim-4) * nvars * (2*kDim+1)`.

See also: [`calcHaloMap`](@ref), [`sumSharedSpectral`](@ref)
"""
function calcPatchMap(patch::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray},
                       tile::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray})
    n     = size(patch.spectral, 1)
    nvars = size(patch.spectral, 2)
    siL   = tile.params.spectralIndexL
    siR_inner = tile.params.spectralIndexR - 3
    map_arr = spzeros(Float64, n, nvars)
    if siL <= siR_inner && siR_inner <= n
        map_arr[siL:siR_inner, :] .= 1.0
    end
    return map_arr
end

function calcPatchMap(patch::SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray},
                       tile::SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray})
    n           = size(patch.spectral, 1)
    nvars       = size(patch.spectral, 2)
    # Max Fourier wavenumber representable by this tile
    kDim        = tile.params.iDim + tile.params.patchOffsetL
    siL         = tile.params.spectralIndexL
    patchStride = patch.params.b_iDim
    tileShare   = tile.params.b_iDim - 4  # inner rows (excluding 3-row halo)

    map_arr = spzeros(Float64, n, nvars)

    # Wavenumber 0
    p2 = siL + tileShare
    if p2 <= n
        map_arr[siL:p2, :] .= 1.0
    end

    for k in 1:kDim
        i = k * 2
        # Real part
        p1 = ((i-1)*patchStride) + siL
        p2 = p1 + tileShare
        if p1 >= 1 && p2 <= n
            map_arr[p1:p2, :] .= 1.0
        end
        # Imaginary part
        p1 = (i*patchStride) + siL
        p2 = p1 + tileShare
        if p1 >= 1 && p2 <= n
            map_arr[p1:p2, :] .= 1.0
        end
    end
    return map_arr
end

# Generic fallback (1-D slice approach for any unspecialized SpringsteelGrid)
function calcPatchMap(patch::SpringsteelGrid, tile::SpringsteelGrid)
    n          = size(patch.spectral, 1)
    nvars      = size(patch.spectral, 2)
    siL        = tile.params.spectralIndexL
    siR_inner  = tile.params.spectralIndexR - 3
    map_arr    = spzeros(Float64, n, nvars)
    if siL <= siR_inner && siR_inner <= n
        map_arr[siL:siR_inner, :] .= 1.0
    end
    return map_arr
end

# ────────────────────────────────────────────────────────────────────────────
# calcHaloMap
# ────────────────────────────────────────────────────────────────────────────

"""
    calcHaloMap(patch, tile1, tile2) -> SparseMatrixCSC{Float64, Int64}

Return a sparse matrix marking the **3-row halo (overlap) region** at the right boundary
of `tile1` within the `patch` spectral array.

The halo rows `spectralIndexR-2 : spectralIndexR` of `tile1` overlap with the first 3
spectral rows contributed by the adjacent `tile2`.  These rows must be **summed** (not
exclusively written) during distributed transforms.  See REFACTORING_PLAN.md §13.3 TRAP-2
for the off-by-one derivation (`-4` vs `-3`).

For cylindrical/spherical grids the same 3-row halo applies independently to every
wavenumber block.

# Arguments
- `patch`: Full-domain patch grid (provides spectral array dimensions)
- `tile1`: Left tile whose right-edge halo is identified
- `tile2`: Right tile (currently unused; included for API completeness and future validation)

# Returns
`SparseMatrixCSC{Float64, Int64}` of size `(spectral_rows, nvars)` with `1.0` at the
3 halo rows and `0.0` elsewhere.  For `nvars == 1`: `nnz == 3`.

# Concurrency
The halo rows identified here **must not be written simultaneously** by two workers.
Accumulate halo contributions serially or via explicit synchronisation.
See also [`sumSharedSpectral`](@ref).

See also: [`calcPatchMap`](@ref), [`getBorderSpectral`](@ref)
"""
function calcHaloMap(patch::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray},
                      tile1::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray},
                      tile2::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray})
    n     = size(patch.spectral, 1)
    nvars = size(patch.spectral, 2)
    hiL   = tile1.params.spectralIndexR - 2
    hiR   = tile1.params.spectralIndexR
    map_arr = spzeros(Float64, n, nvars)
    if hiL >= 1 && hiR <= n
        map_arr[hiL:hiR, :] .= 1.0
    end
    return map_arr
end

function calcHaloMap(patch::SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray},
                      tile1::SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray},
                      tile2::SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray})
    n           = size(patch.spectral, 1)
    nvars       = size(patch.spectral, 2)
    kDim        = tile1.params.iDim + tile1.params.patchOffsetL
    siL         = tile1.params.spectralIndexL
    patchStride = patch.params.b_iDim
    tileShare   = tile1.params.b_iDim - 3  # first halo index within tile block

    map_arr = spzeros(Float64, n, nvars)

    # Wavenumber 0 halo (3 rows)
    p1 = siL + tileShare
    if p1 >= 1 && p1+2 <= n
        map_arr[p1:p1+2, :] .= 1.0
    end

    for k in 1:kDim
        i = k * 2
        # Real part halo
        p1 = ((i-1)*patchStride) + siL + tileShare
        if p1 >= 1 && p1+2 <= n
            map_arr[p1:p1+2, :] .= 1.0
        end
        # Imaginary part halo
        p1 = (i*patchStride) + siL + tileShare
        if p1 >= 1 && p1+2 <= n
            map_arr[p1:p1+2, :] .= 1.0
        end
    end
    return map_arr
end

# Generic fallback (1-D halo approach)
function calcHaloMap(patch::SpringsteelGrid, tile1::SpringsteelGrid, tile2::SpringsteelGrid)
    n     = size(patch.spectral, 1)
    nvars = size(patch.spectral, 2)
    hiL   = tile1.params.spectralIndexR - 2
    hiR   = tile1.params.spectralIndexR
    map_arr = spzeros(Float64, n, nvars)
    if hiL >= 1 && hiR <= n
        map_arr[hiL:hiR, :] .= 1.0
    end
    return map_arr
end

# ────────────────────────────────────────────────────────────────────────────
# sumSpectralTile!
# ────────────────────────────────────────────────────────────────────────────

"""
    sumSpectralTile!(patch::SpringsteelGrid, tile::SpringsteelGrid) -> Array{Float64}

Accumulate `tile.spectral` into the corresponding rows of `patch.spectral`.

This is the **accumulation** variant: it **adds** (not overwrites) the tile B-vector
contributions.  Because the last 3 rows of each tile's spectral array overlap with the
next tile's first rows, multiple calls (one per tile) must be made before the full patch
spectral array is consistent.

# API variants

| Geometry | Accumulation pattern |
|:-------- |:--------------------- |
| 1-D Cartesian | `patch.spectral[siL:siR, :] .+= tile.spectral[:, :]` |
| 2-D Cylindrical | Wavenumber-interleaved blocks (see spectral layout in `transforms_cylindrical.jl`) |
| Generic | Same as 1-D Cartesian (safe for all other Cartesian types) |

# Concurrency note
Do NOT call this function concurrently from multiple threads/processes writing to
overlapping spectral rows. See REFACTORING_PLAN.md §13.3 RACE-1.

See also: [`setSpectralTile!`](@ref), [`sumSharedSpectral`](@ref)
"""
function sumSpectralTile!(patch::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray},
                           tile::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray})
    siL = tile.params.spectralIndexL
    siR = tile.params.spectralIndexR
    patch.spectral[siL:siR, :] .+= tile.spectral[:, :]
    return patch.spectral
end

function sumSpectralTile!(patch::SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray},
                           tile::SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray})
    # Max Fourier wavenumber covered by this tile (includes patchOffset rings beyond tile start)
    kDim = tile.params.iDim + tile.params.patchOffsetL
    siL  = tile.params.spectralIndexL
    siR  = tile.params.spectralIndexR

    # Wavenumber 0 block
    patch.spectral[siL:siR, :] .+= tile.spectral[1:tile.params.b_iDim, :]

    for k in 1:kDim
        p = k * 2
        t = k * 2
        # Real part
        pp1 = ((p-1)*patch.params.b_iDim) + siL
        tp1 = ((t-1)*tile.params.b_iDim) + 1
        patch.spectral[pp1:pp1+tile.params.b_iDim-1, :] .+= tile.spectral[tp1:tp1+tile.params.b_iDim-1, :]
        # Imaginary part
        pp1 = (p*patch.params.b_iDim) + siL
        tp1 = (t*tile.params.b_iDim) + 1
        patch.spectral[pp1:pp1+tile.params.b_iDim-1, :] .+= tile.spectral[tp1:tp1+tile.params.b_iDim-1, :]
    end
    return patch.spectral
end

# Generic fallback (all other Cartesian grid types, Spherical, etc.)
function sumSpectralTile!(patch::SpringsteelGrid, tile::SpringsteelGrid)
    siL = tile.params.spectralIndexL
    siR = tile.params.spectralIndexR
    patch.spectral[siL:siR, :] .+= tile.spectral[:, :]
    return patch.spectral
end

# ────────────────────────────────────────────────────────────────────────────
# setSpectralTile!
# ────────────────────────────────────────────────────────────────────────────

"""
    setSpectralTile!(patch::SpringsteelGrid, tile::SpringsteelGrid) -> Array{Float64}

Zero the full patch spectral array and write `tile.spectral` into the appropriate rows.

This is the **zero-then-write** variant, suitable for single-tile workflows or testing.
For multi-tile workflows use [`sumSpectralTile!`](@ref) after zeroing once.

# Concurrency note
Zeroes `patch.spectral` entirely before writing.  Do NOT call from multiple threads
simultaneously.

See also: [`sumSpectralTile!`](@ref)
"""
function setSpectralTile!(patch::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray},
                           tile::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray})
    patch.spectral[:] .= 0.0
    siL = tile.params.spectralIndexL
    siR = tile.params.spectralIndexR
    patch.spectral[siL:siR, :] .= tile.spectral[:, :]
    return patch.spectral
end

function setSpectralTile!(patch::SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray},
                           tile::SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray})
    patch.spectral[:] .= 0.0
    kDim = tile.params.iDim + tile.params.patchOffsetL
    siL  = tile.params.spectralIndexL
    siR  = tile.params.spectralIndexR

    # Wavenumber 0 block
    patch.spectral[siL:siR, :] .= tile.spectral[1:tile.params.b_iDim, :]

    for k in 1:kDim
        p = k * 2
        t = k * 2
        # Real part
        pp1 = ((p-1)*patch.params.b_iDim) + siL
        tp1 = ((t-1)*tile.params.b_iDim) + 1
        patch.spectral[pp1:pp1+tile.params.b_iDim-1, :] .= tile.spectral[tp1:tp1+tile.params.b_iDim-1, :]
        # Imaginary part
        pp1 = (p*patch.params.b_iDim) + siL
        tp1 = (t*tile.params.b_iDim) + 1
        patch.spectral[pp1:pp1+tile.params.b_iDim-1, :] .= tile.spectral[tp1:tp1+tile.params.b_iDim-1, :]
    end
    return patch.spectral
end

# Generic fallback
function setSpectralTile!(patch::SpringsteelGrid, tile::SpringsteelGrid)
    patch.spectral[:] .= 0.0
    siL = tile.params.spectralIndexL
    siR = tile.params.spectralIndexR
    patch.spectral[siL:siR, :] .= tile.spectral[:, :]
    return patch.spectral
end

# ────────────────────────────────────────────────────────────────────────────
# sumSharedSpectral
# ────────────────────────────────────────────────────────────────────────────

"""
    sumSharedSpectral(sharedSpectral, tile, patchMap, haloMap) -> Nothing

Write a tile's interior B-vectors into `sharedSpectral` (exclusive region) and accumulate
halo contributions (overlap region).

Uses the pre-computed `patchMap` to identify which rows are exclusive to this tile, and
`haloMap` to identify the 3 rows that overlap with the adjacent tile.

# Arguments
- `sharedSpectral::SharedArray{Float64}`: Shared memory array for the full-patch B-vector
- `tile::SpringsteelGrid`: Source tile providing B-coefficients and spectral index offsets
- `patchMap`: `SparseMatrixCSC` from [`calcPatchMap`](@ref) — exclusive interior region
- `haloMap`: `SparseMatrixCSC` from [`calcHaloMap`](@ref) — halo overlap region

# Concurrency warning (RACE-1)
The halo accumulation (`sharedSpectral[halo_rows, :] .+= ...`) is **non-atomic**.
If two adjacent tiles share halo rows, concurrent writes will produce incorrect results.
Callers **must serialise** halo-zone writes across workers.

See also: [`splineTransform!`](@ref), [`getBorderSpectral`](@ref)
"""
function sumSharedSpectral(sharedSpectral::SharedArray{real},
                             tile::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray},
                             patchMap::SparseArrays.SparseMatrixCSC{Float64, Int64},
                             haloMap::SparseArrays.SparseMatrixCSC{Float64, Int64})
    # Interior (exclusive) write — safe to parallelise across tiles
    siL = tile.params.spectralIndexL
    siR = tile.params.spectralIndexR - 3
    tiR = tile.params.b_iDim - 3
    sharedSpectral[siL:siR, :] .= tile.spectral[1:tiR, :]

    # Halo accumulation (non-atomic — serialise across adjacent tiles)
    biL = tile.params.spectralIndexL
    biR = biL + 2
    sharedSpectral[biL:biR, :] .+= Matrix(haloMap[biL:biR, :])
    return nothing
end

# Generic fallback
function sumSharedSpectral(sharedSpectral::SharedArray,
                             tile::SpringsteelGrid,
                             patchMap::SparseArrays.SparseMatrixCSC,
                             haloMap::SparseArrays.SparseMatrixCSC)
    siL = tile.params.spectralIndexL
    siR = tile.params.spectralIndexR - 3
    if siR >= siL
        sharedSpectral[siL:siR, :] .= tile.spectral[1:tile.params.b_iDim-3, :]
    end
    return nothing
end

# ────────────────────────────────────────────────────────────────────────────
# splineTransform!  (B → A)
# ────────────────────────────────────────────────────────────────────────────

"""
    splineTransform!(sharedSpectral, tile) -> Nothing

Apply the SA (B-vector → A-coefficient) transform for all splines in `tile` using data
from the shared spectral array.

After this call, `tile.spectral[:, v]` contains the A-coefficients for variable `v`
(ready for [`tileTransform!`](@ref) to evaluate physical values).

# Concurrency warning (RACE-3)
Reads from `sharedSpectral` (SharedArray).  The caller **must** ensure that
`sharedSpectral` is fully populated (all tiles have finished their
[`sumSharedSpectral`](@ref) calls) before invoking this function in parallel.

See also: [`tileTransform!`](@ref), [`sumSharedSpectral`](@ref)
"""
function splineTransform!(sharedSpectral::SharedArray{real},
                            tile::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray})
    for v in 1:length(tile.params.vars)
        tile.spectral[:, v] .= SAtransform(tile.ibasis.data[1, v],
                                            view(sharedSpectral, :, v))
    end
    return nothing
end

# ────────────────────────────────────────────────────────────────────────────
# tileTransform!  (A-coefficients → physical)
# ────────────────────────────────────────────────────────────────────────────

"""
    tileTransform!(sharedSpectral, tile, physical, spectral) -> Array{Float64}

Evaluate the field and its derivatives on `tile`'s gridpoints using pre-computed
A-coefficients stored in `spectral`.

Populates:
- `physical[:, v, 1]` — field values
- `physical[:, v, 2]` — first i-derivatives
- `physical[:, v, 3]` — second i-derivatives

# Thread safety (RACE-2)
Safe to parallelise over variables `v` because `ibasis.data[1, v]` and
`physical[:, v, :]` are fully independent per `v`.  Do **not** share basis objects
across variables without reviewing this invariant (see REFACTORING_PLAN.md §13.3 RACE-2).

# Arguments
- `sharedSpectral::SharedArray{Float64}`: Shared memory (not used directly; included for API uniformity)
- `tile::SpringsteelGrid`: Target tile whose `physical` array is written
- `physical::Array{Float64}`: Destination physical array (shape `(iDim, nvars, 3)`)
- `spectral::Array{Float64}`: Source A-coefficient array (pre-computed by [`splineTransform!`](@ref))

See also: [`splineTransform!`](@ref), [`gridTransform!`](@ref)
"""
function tileTransform!(sharedSpectral::SharedArray{real},
                          tile::SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray},
                          physical::Array{real},
                          spectral::Array{real})
    pts = getGridpoints(tile)
    Threads.@threads for v in 1:length(tile.params.vars)
        tile.ibasis.data[1, v].a .= view(spectral, :, v)
        SItransform(tile.ibasis.data[1, v],   pts, view(physical, :, v, 1))
        SIxtransform(tile.ibasis.data[1, v],  pts, view(physical, :, v, 2))
        SIxxtransform(tile.ibasis.data[1, v], pts, view(physical, :, v, 3))
    end
    return physical
end

# ════════════════════════════════════════════════════════════════════════════
# Phase 10: Convenience wrappers / backward-compat helpers
# ════════════════════════════════════════════════════════════════════════════

"""
    gridTransform!(patch, tile)

Evaluate `patch`'s spectral representation at `tile`'s gridpoints and write the
result into `tile.physical` (values and first + second i-derivatives).

This 2-argument form is the primary cross-grid evaluation interface.  It reads
`patch.spectral`, runs the B→A transform on the patch's own spline objects, then
evaluates at the tile gridpoints.

See also: [`gridTransform!`](@ref), [`spectralTransform!`](@ref)
"""
function gridTransform!(patch::_1DCartesianGrid, tile::_1DCartesianGrid)
    tile_pts = getGridpoints(tile)
    nvars = size(patch.spectral, 2)
    for v in 1:nvars
        sp = patch.ibasis.data[1, v]
        sp.b .= view(patch.spectral, :, v)
        SAtransform!(sp)
        SItransform(sp,   tile_pts, view(tile.physical, :, v, 1))
        SIxtransform(sp,  tile_pts, view(tile.physical, :, v, 2))
        SIxxtransform(sp, tile_pts, view(tile.physical, :, v, 3))
    end
    return tile.physical
end

"""
    gridTransform!(patchSplines, patchSpectral, patchParams, tile, splineBuffer)

5-argument explicit form of the cross-grid inverse transform.  Reads B-coefficients
from `patchSpectral`, uses `patchSplines` for the SA transform, and evaluates at
`tile`'s gridpoints.  `splineBuffer` is accepted for API uniformity but is unused
for 1-D grids.

See also: [`gridTransform!`](@ref)
"""
function gridTransform!(patchSplines::SplineBasisArray,
                         patchSpectral::Array{real},
                         patchParams::SpringsteelGridParameters,
                         tile::_1DCartesianGrid,
                         splineBuffer::Array)
    tile_pts = getGridpoints(tile)
    nvars = size(patchSpectral, 2)
    for v in 1:nvars
        sp = patchSplines.data[1, v]
        sp.b .= view(patchSpectral, :, v)
        SAtransform!(sp)
        SItransform(sp,   tile_pts, view(tile.physical, :, v, 1))
        SIxtransform(sp,  tile_pts, view(tile.physical, :, v, 2))
        SIxxtransform(sp, tile_pts, view(tile.physical, :, v, 3))
    end
    return tile.physical
end

"""
    allocateSplineBuffer(patch, tile) -> Array

2-argument form: allocate a spline buffer sized for `tile`.
Delegates to the 1-argument form `allocateSplineBuffer(tile)`.

See also: [`allocateSplineBuffer`](@ref)
"""
function allocateSplineBuffer(patch::SpringsteelGrid, tile::SpringsteelGrid)
    return allocateSplineBuffer(tile)
end

"""
    calcHaloMap(patch, tile) -> SparseMatrixCSC{Float64, Int64}

2-argument convenience form of `calcHaloMap`.  Calls the 3-argument internal
implementation with `tile` used as both left and right neighbours (sufficient
for the halo-map computation, which only reads `tile1`).

See also: [`calcHaloMap`](@ref)
"""
function calcHaloMap(patch::SpringsteelGrid, tile::SpringsteelGrid)
    return calcHaloMap(patch, tile, tile)
end

"""
    sumSpectralTile(dest, tileSpectral, siL, siR)

Lower-level accumulation: add `tileSpectral` into `dest[siL:siR, :]`.

This is the array-only variant (no `SpringsteelGrid` objects required).
Multiple calls accumulate (sum) contributions.

See also: [`sumSpectralTile!`](@ref)
"""
function sumSpectralTile(dest::Array{real}, tileSpectral::Array{real},
                          siL::int, siR::int)
    dest[siL:siR, :] .+= tileSpectral[:, :]
    return dest
end

"""
    setSpectralTile(dest, patchParams, tile)

3-argument array form: zero `dest` then write `tile.spectral` into the rows
`tile.params.spectralIndexL : tile.params.spectralIndexR`.

See also: [`setSpectralTile!`](@ref)
"""
function setSpectralTile(dest::Array{real}, patchParams::SpringsteelGridParameters,
                          tile::SpringsteelGrid)
    dest[:] .= 0.0
    siL = tile.params.spectralIndexL
    siR = tile.params.spectralIndexR
    dest[siL:siR, :] .= tile.spectral[:, :]
    return dest
end

# ════════════════════════════════════════════════════════════════════════════
# Phase 8b: Multi-Dimensional Tiling
# ════════════════════════════════════════════════════════════════════════════
#
# Extends tiling to Cartesian Spline×Spline (RR) and Spline×Spline×Spline
# (RRR) grids where multiple dimensions can be partitioned simultaneously.
#
# The spectral array for a 2D Spline×Spline grid is laid out as:
#   flat_index = (j_mode - 1) * b_iDim + i_mode
# A multi-dim tile covers a contiguous i-mode range [siL_i, siR_i] for every
# j-mode block it owns [sjL_j, sjR_j].  Face halos (3 modes wide) exist on
# every tiled-dimension boundary; corner halos (3×3, 3×3×3) arise where two
# or three tiled dimensions meet.
#
# CONCURRENCY note: The same RACE-1/RACE-3 warnings from 1D tiling apply.
# Halo zones in multi-dim extend along dimension borders AND dimension corners.
# ════════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────
# Internal helper: split a spline domain into n_tiles sub-domains (by cells)
#
# Returns: (nc_vec, xMins, xMaxs, siL_vec)
#   nc_vec  — number of cells per tile
#   xMins   — physical left boundary of each tile
#   xMaxs   — physical right boundary of each tile
#   siL_vec — 1-based spectral index (within whole patch) for each tile
# ────────────────────────────────────────────────────────────────────────────
function _split_cells_1d(xMin::Float64, xMax::Float64, nc_total::Int64,
                           n_tiles::Int64, min_cells::Int64=3)
    if nc_total / n_tiles < min_cells
        throw(DomainError(n_tiles,
            "Too many tiles: need at least $min_cells cells per tile (have $nc_total cells)"))
    end
    DX = (xMax - xMin) / nc_total
    q, r = divrem(nc_total, n_tiles)
    nc   = [i <= r ? q+1 : q for i in 1:n_tiles]

    xMins = zeros(Float64, n_tiles)
    xMaxs = zeros(Float64, n_tiles)
    siL   = ones(Int64, n_tiles)

    xMins[1] = xMin
    xMaxs[1] = nc[1] * DX + xMins[1]
    for idx in 2:n_tiles-1
        xMins[idx] = xMaxs[idx-1]
        xMaxs[idx] = nc[idx] * DX + xMins[idx]
        siL[idx]   = siL[idx-1] + nc[idx-1]
    end
    if n_tiles > 1
        xMins[n_tiles] = xMaxs[n_tiles-1]
        xMaxs[n_tiles] = xMax
        siL[n_tiles]   = siL[n_tiles-1] + nc[n_tiles-1]
        nc[n_tiles]    = nc_total - siL[n_tiles] + 1
    end
    return nc, xMins, xMaxs, siL
end

# ────────────────────────────────────────────────────────────────────────────
# calcTileSizes — 2D Cartesian Spline×Spline (RR)  [NamedTuple overload]
# ────────────────────────────────────────────────────────────────────────────

"""
    calcTileSizes(patch::SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, NoBasisArray},
                  tile_spec::NamedTuple) -> Vector{SpringsteelGrid}

Multi-dimensional tile decomposition for a 2-D Cartesian Spline×Spline grid (RR or
Spline2D).  Returns a flat `Vector` of `ni × nj` tile sub-grids.

`tile_spec` is a `NamedTuple` with optional keys `:i` (number of i-tiles, default 1)
and `:j` (number of j-tiles, default 1).  The resulting tiles are ordered with j
varying fastest: `tiles[(ti-1)*nj + tj]` corresponds to i-strip `ti`, j-strip `tj`.

Each tile is a fully-initialised `SpringsteelGrid{CartesianGeometry, SplineBasisArray,
SplineBasisArray, NoBasisArray}` spanning the sub-domain
`[iMin_ti, iMax_ti] × [jMin_tj, jMax_tj]`.

# Splitting strategy
Both i and j dimensions are split by cell count (`nc ≥ 3` per tile).  The i-direction
spectral index (`spectralIndexL`) for each tile records its position within the patch
i-spectral array; j-tiles have independent j-spectral layouts within each tile.

# Throws
- `DomainError` if any tile would have fewer than 3 cells in any tiled dimension.

# See also
[`calcTileSizes(::SpringsteelGrid, ::Int64)`](@ref),
[`calcPatchMap_multidim`](@ref), [`calcHaloMap_multidim`](@ref)
"""
function calcTileSizes(patch::_2DCartesianRR, tile_spec::NamedTuple)
    ni = get(tile_spec, :i, 1)
    nj = get(tile_spec, :j, 1)

    if ni == 1 && nj == 1
        return [patch]
    end
    if nj == 1
        return calcTileSizes(patch, ni)   # delegate to 1D Cartesian
    end

    nc_i_total = patch.params.num_cells
    nc_j_total = patch.params.b_jDim - 3

    # ── i-direction splitting (by gridpoints, matching 1D Cartesian logic) ─
    # Minimum is 3 cells = 9 gridpoints; split gridpoints then back-convert.
    i_gpts = patch.params.iDim
    if i_gpts / ni < 9.0
        throw(DomainError(ni,
            "Too many i-tiles for this grid (need ≥ 9 i-gridpoints per tile)"))
    end
    DX_cell = (patch.params.iMax - patch.params.iMin) / nc_i_total
    qi, ri   = divrem(i_gpts, ni)
    i_sizes  = [k <= ri ? qi+1 : qi for k in 1:ni]   # gridpoints per i-tile

    iMins  = zeros(Float64, ni)
    iMaxs  = zeros(Float64, ni)
    nc_i   = zeros(Int64, ni)
    siL_i  = ones(Int64, ni)

    iMins[1] = patch.params.iMin
    nc_i[1]  = Int64(ceil(i_sizes[1] / 3))
    iMaxs[1] = nc_i[1] * DX_cell + iMins[1]
    for idx in 2:ni-1
        iMins[idx] = iMaxs[idx-1]
        nc_i[idx]  = Int64(ceil(i_sizes[idx] / 3))
        iMaxs[idx] = nc_i[idx] * DX_cell + iMins[idx]
        siL_i[idx] = siL_i[idx-1] + nc_i[idx-1]
    end
    if ni > 1
        iMins[ni] = iMaxs[ni-1]
        iMaxs[ni] = patch.params.iMax
        siL_i[ni] = siL_i[ni-1] + nc_i[ni-1]
        nc_i[ni]  = nc_i_total - siL_i[ni] + 1
    end

    # ── j-direction splitting (by cell count) ─────────────────────────────
    nc_j_vec, jMins, jMaxs, _ = _split_cells_1d(
        patch.params.jMin, patch.params.jMax, nc_j_total, nj, 3)

    # ── Build Cartesian-product tile array ─────────────────────────────────
    tiles = Vector{typeof(patch)}(undef, ni * nj)
    t = 1
    for ti in 1:ni
        for tj in 1:nj
            tile_gp = SpringsteelGridParameters(
                geometry       = "RR",
                iMin           = iMins[ti],
                iMax           = iMaxs[ti],
                num_cells      = nc_i[ti],
                jMin           = jMins[tj],
                jMax           = jMaxs[tj],
                jDim           = nc_j_vec[tj] * CubicBSpline.mubar,
                l_q            = patch.params.l_q,
                BCL            = patch.params.BCL,
                BCR            = patch.params.BCR,
                BCU            = patch.params.BCU,
                BCD            = patch.params.BCD,
                vars           = patch.params.vars,
                spectralIndexL = siL_i[ti],
                tile_num       = t)
            tiles[t] = createGrid(tile_gp)
            t += 1
        end
    end
    return tiles
end

# ────────────────────────────────────────────────────────────────────────────
# calcTileSizes — 3D Cartesian Spline×Spline×Spline (RRR)  [NamedTuple overload]
# ────────────────────────────────────────────────────────────────────────────

"""
    calcTileSizes(patch::SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, SplineBasisArray},
                  tile_spec::NamedTuple) -> Vector{SpringsteelGrid}

Multi-dimensional tile decomposition for a 3-D Cartesian Spline×Spline×Spline grid (RRR).
Returns a flat `Vector` of `ni × nj × nk` tile sub-grids.

`tile_spec` accepts optional keys `:i`, `:j`, `:k` (default 1 each).  Tiles are ordered
with k varying fastest: `tiles[(ti-1)*nj*nk + (tj-1)*nk + tk]`.

# Throws
- `DomainError` if any tile would have fewer than 3 cells in any tiled dimension.

# See also
[`calcTileSizes(::_2DCartesianRR, ::NamedTuple)`](@ref)
"""
function calcTileSizes(patch::_3DCartesianRRR, tile_spec::NamedTuple)
    ni = get(tile_spec, :i, 1)
    nj = get(tile_spec, :j, 1)
    nk = get(tile_spec, :k, 1)

    if ni == 1 && nj == 1 && nk == 1
        return [patch]
    end
    if nj == 1 && nk == 1
        return calcTileSizes(patch, ni)   # delegate to 1D Cartesian
    end

    nc_i_total = patch.params.num_cells
    nc_j_total = patch.params.b_jDim - 3
    nc_k_total = patch.params.b_kDim - 3

    # ── i-direction splitting ───────────────────────────────────────────────
    i_gpts = patch.params.iDim
    if i_gpts / ni < 9.0
        throw(DomainError(ni,
            "Too many i-tiles for this grid (need ≥ 9 i-gridpoints per tile)"))
    end
    DX_cell   = (patch.params.iMax - patch.params.iMin) / nc_i_total
    qi, ri    = divrem(i_gpts, ni)
    i_sizes   = [k <= ri ? qi+1 : qi for k in 1:ni]

    iMins  = zeros(Float64, ni)
    iMaxs  = zeros(Float64, ni)
    nc_i   = zeros(Int64, ni)
    siL_i  = ones(Int64, ni)

    iMins[1] = patch.params.iMin
    nc_i[1]  = Int64(ceil(i_sizes[1] / 3))
    iMaxs[1] = nc_i[1] * DX_cell + iMins[1]
    for idx in 2:ni-1
        iMins[idx] = iMaxs[idx-1]
        nc_i[idx]  = Int64(ceil(i_sizes[idx] / 3))
        iMaxs[idx] = nc_i[idx] * DX_cell + iMins[idx]
        siL_i[idx] = siL_i[idx-1] + nc_i[idx-1]
    end
    if ni > 1
        iMins[ni] = iMaxs[ni-1]
        iMaxs[ni] = patch.params.iMax
        siL_i[ni] = siL_i[ni-1] + nc_i[ni-1]
        nc_i[ni]  = nc_i_total - siL_i[ni] + 1
    end

    # ── j-direction splitting ───────────────────────────────────────────────
    nc_j_vec, jMins, jMaxs, _ = _split_cells_1d(
        patch.params.jMin, patch.params.jMax, nc_j_total, nj, 3)

    # ── k-direction splitting ───────────────────────────────────────────────
    nc_k_vec, kMins, kMaxs, _ = _split_cells_1d(
        patch.params.kMin, patch.params.kMax, nc_k_total, nk, 3)

    # ── Build Cartesian-product tile array ─────────────────────────────────
    tiles = Vector{typeof(patch)}(undef, ni * nj * nk)
    t = 1
    for ti in 1:ni
        for tj in 1:nj
            for tk in 1:nk
                tile_gp = SpringsteelGridParameters(
                    geometry       = "RRR",
                    iMin           = iMins[ti],
                    iMax           = iMaxs[ti],
                    num_cells      = nc_i[ti],
                    jMin           = jMins[tj],
                    jMax           = jMaxs[tj],
                    jDim           = nc_j_vec[tj] * CubicBSpline.mubar,
                    kMin           = kMins[tk],
                    kMax           = kMaxs[tk],
                    kDim           = nc_k_vec[tk] * CubicBSpline.mubar,
                    l_q            = patch.params.l_q,
                    BCL            = patch.params.BCL,
                    BCR            = patch.params.BCR,
                    BCU            = patch.params.BCU,
                    BCD            = patch.params.BCD,
                    BCB            = patch.params.BCB,
                    BCT            = patch.params.BCT,
                    vars           = patch.params.vars,
                    spectralIndexL = siL_i[ti],
                    tile_num       = t)
                tiles[t] = createGrid(tile_gp)
                t += 1
            end
        end
    end
    return tiles
end

# ────────────────────────────────────────────────────────────────────────────
# calcTileSizes — generic NamedTuple fallback
# ────────────────────────────────────────────────────────────────────────────

"""
    calcTileSizes(patch::SpringsteelGrid{G, SplineBasisArray, J, K}, tile_spec::NamedTuple) -> Vector{SpringsteelGrid}

Generic NamedTuple dispatch for grids whose i-dimension uses a Spline basis.

For grids where only the i-dimension is tiled (`tile_spec = (i=N,)`) this delegates to
the existing 1-D [`calcTileSizes`](@ref) overload.  If `:j` or `:k` counts exceed 1, but
the corresponding basis is not a `SplineBasisArray`, a `DomainError` is thrown — tiling
requires a Spline basis in the tiled dimension.

This is the entry point for calls such as `calcTileSizes(rl_patch, (i=4,))` (single
i-strip tiling via NamedTuple) or `calcTileSizes(rl_patch, (i=2, j=2))` (error: j is
Fourier, not Spline).

# Throws
- `DomainError` if `:j > 1` and `J` is not `SplineBasisArray`, or `:k > 1` and `K` is
  not `SplineBasisArray`.
"""
function calcTileSizes(patch::SpringsteelGrid{G, SplineBasisArray, J, K},
                        tile_spec::NamedTuple) where {G, J, K}
    ni = get(tile_spec, :i, 1)
    nj = get(tile_spec, :j, 1)
    nk = get(tile_spec, :k, 1)

    # Validate: can only tile a dimension that uses the Spline basis
    if nj > 1 && !(J <: SplineBasisArray)
        throw(DomainError(nj,
            "Cannot tile j-dimension: j-basis is $(J), not SplineBasisArray"))
    end
    if nk > 1 && !(K <: SplineBasisArray)
        throw(DomainError(nk,
            "Cannot tile k-dimension: k-basis is $(K), not SplineBasisArray"))
    end

    # Delegate i-only tiling to the existing Int64 dispatch
    return calcTileSizes(patch, ni)
end

# ────────────────────────────────────────────────────────────────────────────
# calcPatchMap_multidim
# ────────────────────────────────────────────────────────────────────────────

"""
    calcPatchMap_multidim(patch::SpringsteelGrid, tile::SpringsteelGrid) -> SparseMatrixCSC

Compute a `SparseMatrixCSC` marking the non-halo (interior) spectral rows of `patch`
that belong exclusively to `tile` in a multi-dimensional tiling context.

For a 2-D Cartesian Spline×Spline grid the spectral array is laid out as
`spectral[(j_mode - 1) * b_iDim + i_mode, v]`.  The interior i-mode range for the
tile is `[siL, siR - 3]` (excludes the 3-mode halo on the right boundary), and the
interior j-mode range is `[1, tile.params.b_jDim - 3]` within the tile's own spectral
array.  This function marks the corresponding rows in the **patch** spectral array.

The returned sparse matrix has shape `(size(patch.spectral, 1), length(patch.params.vars))`
with nonzero value 1.0 at each (row, var) that is exclusively owned by this tile.

For a 1-D (non-multi-dim) tile, this reduces to the same result as [`calcPatchMap`](@ref).

See also: [`calcHaloMap_multidim`](@ref), [`calcPatchMap`](@ref)
"""
function calcPatchMap_multidim(patch::SpringsteelGrid, tile::SpringsteelGrid)
    patch_rows = size(patch.spectral, 1)
    nvars      = size(patch.spectral, 2)
    pmap = spzeros(Float64, patch_rows, nvars)

    siL   = tile.params.spectralIndexL
    b_iDim_patch = patch.params.b_iDim
    b_jDim_tile  = tile.params.b_jDim   # 0 for 1D grids

    if b_jDim_tile == 0 || b_jDim_tile == 1
        # 1D case: fall through to row-based inner region
        siR_inner = tile.params.spectralIndexR - 3
        for v in 1:nvars
            for row in siL:siR_inner
                pmap[row, v] = 1.0
            end
        end
        return pmap
    end

    # 2D+ case: mark (j_block, i_range) pairs in patchSpectral
    # tile's inner i-range in patch coordinates: siL .. spectralIndexR - 3
    siR_inner    = tile.params.spectralIndexR - 3
    b_jDim_inner = b_jDim_tile - 3   # j inner modes (excl. 3-mode j halo)

    for jt in 1:b_jDim_inner
        # j-mode jt in the tile maps to j-mode jt in the patch for the
        # tile's j sub-domain (the tile has its own j spectral array).
        # Flat row in patch: (jt-1)*b_iDim_patch + siL .. siR_inner
        jt_offset = (jt - 1) * b_iDim_patch
        for v in 1:nvars
            for row in siL:siR_inner
                patch_row = jt_offset + row
                if 1 <= patch_row <= patch_rows
                    pmap[patch_row, v] = 1.0
                end
            end
        end
    end
    return pmap
end

# ────────────────────────────────────────────────────────────────────────────
# calcHaloMap_multidim
# ────────────────────────────────────────────────────────────────────────────

"""
    calcHaloMap_multidim(patch::SpringsteelGrid, tile_L::SpringsteelGrid, tile_R::SpringsteelGrid) -> SparseMatrixCSC

Compute the **face halo** `SparseMatrixCSC` between two adjacent tiles that share an
i-direction boundary in a multi-dimensional tiling context.

The halo region consists of the 3 i-spectral modes at the right boundary of `tile_L`
(indices `spectralIndexR - 2 : spectralIndexR` in patch coordinates) for all j-mode
blocks owned by `tile_L`.

The returned sparse matrix has shape `(size(patch.spectral, 1), length(patch.params.vars))`
with nonzero value 1.0 at each halo (row, var) entry.

Corner halos (where two or three tiled dimensions overlap) require calling this function
per face and intersecting the results — or using dedicated corner-halo logic for
performance.  See REFACTORING_PLAN.md §6 for the full multi-dim halo design.

# Arguments
- `patch`: Full-domain patch grid (provides spectral layout)
- `tile_L`: Left tile (halo is on its right i-boundary)
- `tile_R`: Right tile (not used directly; present for API symmetry with 1-D version)

See also: [`calcPatchMap_multidim`](@ref), [`calcHaloMap`](@ref)
"""
function calcHaloMap_multidim(patch::SpringsteelGrid, tile_L::SpringsteelGrid,
                                tile_R::SpringsteelGrid)
    patch_rows   = size(patch.spectral, 1)
    nvars        = size(patch.spectral, 2)
    hmap         = spzeros(Float64, patch_rows, nvars)

    biL          = tile_L.params.spectralIndexR - 2
    biR          = tile_L.params.spectralIndexR
    b_iDim_patch = patch.params.b_iDim
    b_jDim_tile  = tile_L.params.b_jDim

    if b_jDim_tile == 0 || b_jDim_tile == 1
        # 1D case: 3 contiguous halo rows
        for v in 1:nvars
            for row in biL:biR
                if 1 <= row <= patch_rows
                    hmap[row, v] = 1.0
                end
            end
        end
        return hmap
    end

    # 2D+ case: halo extends across all j-mode blocks of tile_L
    for jt in 1:b_jDim_tile
        jt_offset = (jt - 1) * b_iDim_patch
        for v in 1:nvars
            for row in biL:biR
                patch_row = jt_offset + row
                if 1 <= patch_row <= patch_rows
                    hmap[patch_row, v] = 1.0
                end
            end
        end
    end
    return hmap
end

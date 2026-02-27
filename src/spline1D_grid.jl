# Functions for the Spline Grid

"""
    Spline1D_Grid <: SpringsteelGrid

One-dimensional spectral grid using cubic B-spline basis functions.

# Fields
- `params::SpringsteelGridParameters`: Grid configuration including domain bounds, resolution, and boundary conditions
- `splines::Array{Spline1D}`: Array of 1D spline objects, one per variable
- `spectral::Array{Float64}`: Spectral coefficients array with dimensions `(b_iDim, vars)`
- `physical::Array{Float64}`: Physical space values with dimensions `(iDim, vars, 3)` where the last dimension stores `[value, derivative, second_derivative]`

# Description
`Spline1D_Grid` provides a one-dimensional spectral representation using cubic B-splines. The grid supports:
- Multiple variables with independent boundary conditions
- Variable-specific filter lengths (l_q parameter)
- Efficient spectral transforms between physical and spectral space
- Domain tiling for parallel/distributed computing
- Automatic computation of derivatives up to second order

# Example
```julia
using Springsteel

# Create a simple 1D grid
gp = SpringsteelGridParameters(
    geometry = "Spline1D",
    iMin = 0.0,
    iMax = 10.0,
    num_cells = 20,
    BCL = Dict("u" => CubicBSpline.R0),
    BCR = Dict("u" => CubicBSpline.R0),
    vars = Dict("u" => 1)
)

grid = createGrid(gp)

# Set values in physical space
gridpoints = getGridpoints(grid)
for i in eachindex(gridpoints)
    grid.physical[i, 1, 1] = sin(gridpoints[i])
end

# Transform to spectral space
spectralTransform!(grid)

# Transform back to physical space (with derivatives)
gridTransform!(grid)

# Access derivatives
values = grid.physical[:, 1, 1]
first_derivatives = grid.physical[:, 1, 2]
second_derivatives = grid.physical[:, 1, 3]
```

See also: [`create_Spline1D_Grid`](@ref), [`SpringsteelGridParameters`](@ref), [`spectralTransform!`](@ref), [`gridTransform!`](@ref)
"""
struct Spline1D_Grid <: SpringsteelGrid
    params::SpringsteelGridParameters
    splines::Array{Spline1D}
    spectral::Array{Float64}
    physical::Array{Float64}
end

"""
    create_Spline1D_Grid(gp::SpringsteelGridParameters) -> Spline1D_Grid

Create a one-dimensional spectral grid with cubic B-spline basis functions.

# Arguments
- `gp::SpringsteelGridParameters`: Grid parameters specifying domain, resolution, boundary conditions, and variables

# Returns
- `Spline1D_Grid`: Initialized grid with allocated spline objects and arrays

# Description
Constructs an `Spline1D_Grid` by:
1. Allocating spectral and physical space arrays based on grid dimensions
2. Creating a `Spline1D` object for each variable with specified boundary conditions
3. Supporting variable-specific filter lengths via `gp.l_q` dictionary

The resulting grid is ready for spectral transforms and derivative computation.

# Example
```julia
gp = SpringsteelGridParameters(
    geometry = "Spline1D",
    iMin = -1.0,
    iMax = 1.0,
    num_cells = 50,
    BCL = Dict("u" => CubicBSpline.R0),
    BCR = Dict("u" => CubicBSpline.R0),
    vars = Dict("u" => 1),
    l_q = Dict("u" => 3.0)  # Custom filter length for u
)

grid = create_Spline1D_Grid(gp)
```

See also: [`Spline1D_Grid`](@ref), [`SpringsteelGridParameters`](@ref)
"""
function create_Spline1D_Grid(gp::SpringsteelGridParameters)

    # Create a 1-D grid with bSplines as basis
    splines = Array{Spline1D}(undef,1,length(values(gp.vars)))
    spectral = zeros(Float64, gp.b_iDim, length(values(gp.vars)))
    physical = zeros(Float64, gp.iDim, length(values(gp.vars)), 3)
    grid = Spline1D_Grid(gp, splines, spectral, physical)
    for key in keys(gp.vars)
        # Allow for filter length to be variable specific
        var_l_q = 2.0
        if haskey(gp.l_q,key)
            var_l_q = gp.l_q[key]
        end
        grid.splines[1,gp.vars[key]] = Spline1D(SplineParameters(
            xmin = gp.iMin,
            xmax = gp.iMax,
            num_cells = gp.num_cells,
            l_q = var_l_q,
            BCL = gp.BCL[key],
            BCR = gp.BCR[key]))
    end
    return grid
end

"""
    calcTileSizes(patch::Spline1D_Grid, num_tiles::Int64) -> Matrix{Float64}

Compute tile geometry for distributing a `Spline1D_Grid` patch across `num_tiles` workers.

Tiles subdivide the patch radially (along the I/radial direction). Each tile must contain
at least 3 spline cells (9 gridpoints) to ensure sufficient spectral overlap at tile
boundaries.

# Arguments
- `patch::Spline1D_Grid`: The full-domain patch to be split
- `num_tiles::Int64`: Number of tiles (workers) to split into

# Returns
- `Matrix{Float64}` of size `(5, num_tiles)` with rows:
  1. `iMins` — left domain boundary of each tile
  2. `iMaxs` — right domain boundary of each tile
  3. `num_cells` — number of spline cells in each tile
  4. `spectralIndicesL` — left spectral index in the patch spectral array (1-based)
  5. `tile_sizes` — number of physical gridpoints in each tile

# Throws
- `DomainError` if any tile would have fewer than 9 gridpoints (< 3 cells)

See also: [`calcPatchMap`](@ref), [`calcHaloMap`](@ref)
"""
function calcTileSizes(patch::Spline1D_Grid, num_tiles::int)
    num_gridpoints = patch.params.iDim
    q,r = divrem(num_gridpoints, num_tiles)
    tile_sizes = [i <= r ? q+1 : q for i = 1:num_tiles]
    if any(x->x<9, tile_sizes)
        throw(DomainError(0, "Too many tiles for this grid (need at least 3 cells in the direction)"))
    end

    # Calculate the dimensions and set the parameters
    DX = (patch.params.iMax - patch.params.iMin) / patch.params.num_cells

    iMins = zeros(Float64,num_tiles)
    iMaxs = zeros(Float64,num_tiles)
    num_cells = zeros(Int64,num_tiles)
    spectralIndicesL = ones(Int64,num_tiles)

    # First tile starts on the patch boundary
    iMins[1] = patch.params.iMin
    num_cells[1] = Int64(ceil(tile_sizes[1] / 3))
    iMaxs[1] = (num_cells[1] * DX) + iMins[1]
    # Implicit spectralIndicesL = 1

    for i = 2:num_tiles-1
        iMins[i] = iMaxs[i-1]
        num_cells[i] = Int64(ceil(tile_sizes[i] / 3))
        iMaxs[i] = (num_cells[i] * DX) + iMins[i]
        spectralIndicesL[i] = num_cells[i-1] + spectralIndicesL[i-1]
    end

    # Last tile ends on the patch boundary
    if num_tiles > 1
        iMins[num_tiles] = iMaxs[num_tiles-1]
        iMaxs[num_tiles] = patch.params.iMax
        spectralIndicesL[num_tiles] = num_cells[num_tiles-1] + spectralIndicesL[num_tiles-1]
        num_cells[num_tiles] = patch.params.num_cells - spectralIndicesL[num_tiles] + 1
    end

    tile_params = vcat(iMins', iMaxs', num_cells', spectralIndicesL', tile_sizes')
    return tile_params
end

"""
    getGridpoints(grid::Spline1D_Grid) -> Vector{Float64}

Return the physical locations of all gridpoints in the Spline1D direction.

# Arguments
- `grid::Spline1D_Grid`: The grid object

# Returns
- `Vector{Float64}`: Array of gridpoint locations (mish points) in the left to right direction

# Description
Returns the Gaussian quadrature `mish` points where the physical field values are defined.
For cubic B-splines with `num_cells` cells, there are `3*num_cells` interior gridpoints
plus boundary points determined by the boundary conditions.

# Example
```julia
grid = createGrid(gp)
points = getGridpoints(grid)

# Use gridpoints to initialize field
for (i, r) in enumerate(points)
    grid.physical[i, 1, 1] = exp(-r^2)
end
```

See also: [`Spline1D_Grid`](@ref)
"""
function getGridpoints(grid::Spline1D_Grid)

    # Return an array of the gridpoint locations
    return grid.splines[1].mishPoints
end

"""
    spectralTransform!(grid::Spline1D_Grid) -> Array{Float64}

Transform field values from physical space to spectral (B-spline coefficient) space.

# Arguments
- `grid::Spline1D_Grid`: Grid containing physical values in `grid.physical[:, :, 1]`

# Returns
- `Array{Float64}`: Spectral coefficients (also stored in `grid.spectral`)

# Description
Performs a spectral transform for all variables in the grid, computing the B-spline
coefficients that represent the physical field. The transform uses a cubic B-spline
basis with the boundary conditions specified in the grid parameters.

The function modifies `grid.spectral` in-place and returns the spectral array.
Only the field values `grid.physical[:, :, 1]` are used; derivatives are ignored.

# Example
```julia
# Set physical values
for i in 1:length(gridpoints)
    grid.physical[i, 1, 1] = sin(2π * gridpoints[i])
end

# Transform to spectral space
coeffs = spectralTransform!(grid)

# Spectral coefficients are now in grid.spectral
```

See also: [`gridTransform!`](@ref), [`Spline1D_Grid`](@ref)
"""
function spectralTransform!(grid::Spline1D_Grid)
    
    # Transform from the grid to spectral space
    # For 1D grid, the only varying dimension is the variable name
    spectral = spectralTransform(grid, grid.physical, grid.spectral)
    return spectral
end

"""
    spectralTransform(grid::Spline1D_Grid, physical::Array{Float64}, spectral::Array{Float64}) -> Array{Float64}

Non-mutating internal helper: SB transform of `physical[:, :, 1]` into `spectral`.

Differs from [`spectralTransform!`](@ref) in that the physical and spectral arrays are
passed explicitly (useful for tiled/distributed workflows where the grid's own arrays
are not used directly).  Writes into `spectral` and returns it.

# Arguments
- `grid::Spline1D_Grid`: Grid (used for spline objects)
- `physical::Array{Float64}`: Physical values array of shape `(iDim, nvars, 3)`
- `spectral::Array{Float64}`: Destination spectral array of shape `(b_iDim, nvars)`

See also: [`spectralTransform!`](@ref)
"""
function spectralTransform(grid::Spline1D_Grid, physical::Array{real}, spectral::Array{real})
    for i in eachindex(grid.splines)
        b = SBtransform(grid.splines[i], physical[:,i,1])
        
        # Assign the spectral array
        spectral[:,i] .= b
    end
end

#function spectralxTransform(grid::Spline1D_Grid, physical::Array{real}, spectral::Array{real})
#    
#    # Transform from the grid to spectral space
#    # For 1D grid, the only varying dimension is the variable name
#    # Need to use a R0 BC for this!
#    Fspline = Spline1D(SplineParameters(iMin = grid.params.iMin, 
#            iMax = grid.params.iMax,
#            num_cells = grid.params.num_cells, 
#            BCL = CubicBSpline.R0, 
#            BCR = CubicBSpline.R0))
#
#    for i in eachindex(grid.splines)
#        b = SBtransform(Fspline, physical[:,i,1])
#        a = SAtransform(Fspline, b)
#        Fx = SIxtransform(Fspline, a)
#        bx = SBtransform(Fspline, Fx)
#        
#        # Assign the spectral array
#        spectral[:,i] .= bx
#    end
#end

function spectralxTransform(grid::Spline1D_Grid, physical::Array{real}, spectral::Array{real})

    # Not implemented

end

"""
    gridTransform!(grid::Spline1D_Grid) -> Array{Float64}

Transform from spectral (B-spline coefficient) space to physical space with derivatives.

# Arguments
- `grid::Spline1D_Grid`: Grid containing spectral coefficients in `grid.spectral`

# Returns
- `Array{Float64}`: Physical values and derivatives (also stored in `grid.physical`)

# Description
Performs the inverse spectral transform, evaluating the B-spline representation at all
gridpoints. For each variable, computes:
- `grid.physical[:, var, 1]`: Field values
- `grid.physical[:, var, 2]`: First derivatives (∂/∂x)
- `grid.physical[:, var, 3]`: Second derivatives (∂²/∂x²)

The transform uses the spectral coefficients in `grid.spectral` and the B-spline basis
functions defined by the grid parameters.

# Example
```julia
# After spectral transform or spectral space operations
spectralTransform!(grid)

# Transform back to physical space
gridTransform!(grid)

# Access values and derivatives
values = grid.physical[:, 1, 1]
derivatives = grid.physical[:, 1, 2]
second_derivatives = grid.physical[:, 1, 3]
```

See also: [`spectralTransform!`](@ref), [`Spline1D_Grid`](@ref)
"""
function gridTransform!(grid::Spline1D_Grid)
    
    # Transform from the spectral to grid space
    # For 1D grid, the only varying dimension is the variable name
    physical = gridTransform(grid, grid.physical, grid.spectral)
    return physical
end

"""
    gridTransform(grid::Spline1D_Grid, physical::Array{Float64}, spectral::Array{Float64}) -> Array{Float64}

Non-mutating internal helper: SA + SI transform from `spectral` into `physical`.

Differs from [`gridTransform!`](@ref) in that both arrays are passed explicitly.
For each variable, performs the `SAtransform!` followed by `SItransform`,
`SIxtransform`, and `SIxxtransform` to populate all three derivative layers.

# Arguments
- `grid::Spline1D_Grid`: Grid object (used for spline objects and gridpoints)
- `physical::Array{Float64}`: Destination physical array of shape `(iDim, nvars, 3)`
- `spectral::Array{Float64}`: Source spectral array of shape `(b_iDim, nvars)`

# Returns
- The modified `physical` array

See also: [`gridTransform!`](@ref)
"""
function gridTransform(grid::Spline1D_Grid, physical::Array{real}, spectral::Array{real})
    for i in eachindex(grid.splines)
        grid.splines[i].b .= view(spectral,:,i)
        SAtransform!(grid.splines[i])
        
        # Assign the grid array
        SItransform(grid.splines[i],getGridpoints(grid),view(physical,:,i,1))
        SIxtransform(grid.splines[i],getGridpoints(grid),view(physical,:,i,2))
        SIxxtransform(grid.splines[i],getGridpoints(grid),view(physical,:,i,3))
    end
    
    return physical 
end

"""
    gridTransform!(patch::Spline1D_Grid, tile::Spline1D_Grid) -> Array{Float64}

Evaluate the patch spectral representation at the tile's gridpoints.

Uses the **patch** spline objects and spectral coefficients to evaluate the field at the
**tile** mish points, writing into `tile.physical`.  This is the cross-tile variant used
in distributed/parallel workflows where the patch and tile have different domains.

# Arguments
- `patch::Spline1D_Grid`: Full-domain grid holding assembled spectral coefficients
- `tile::Spline1D_Grid`: Sub-domain tile whose `physical` array will be written

# Returns
- `tile.physical` (shape `(tile.iDim, nvars, 3)`)

See also: [`gridTransform!`](@ref), [`tileTransform!`](@ref)
"""
function gridTransform!(patch::Spline1D_Grid, tile::Spline1D_Grid)
    for i in eachindex(patch.splines)
        patch.splines[i].b .= view(patch.spectral,:,i)
        SAtransform!(patch.splines[i])

        # Assign to the tile grid
        SItransform(patch.splines[i],getGridpoints(tile),view(tile.physical,:,i,1))
        SIxtransform(patch.splines[i],getGridpoints(tile),view(tile.physical,:,i,2))
        SIxxtransform(patch.splines[i],getGridpoints(tile),view(tile.physical,:,i,3))
    end

    return tile.physical
end

"""
    gridTransform!(patchSplines, patchSpectral, pp, tile, splineBuffer)

Worker-callable variant performing the B→A→physical transform for a tile.

Accepts the patch splines and spectral arrays directly (without a `Spline1D_Grid` wrapper)
for compatibility with the distributed transform pipeline.  `pp` and `splineBuffer` are
accepted for API compatibility but are not used in the 1D case.

# Arguments
- `patchSplines::Array{Spline1D}`: Spline objects for the full-domain patch
- `patchSpectral::Array{Float64}`: Patch B-vector spectral array
- `pp::SpringsteelGridParameters`: Patch grid parameters (unused in 1D; retained for API compatibility)
- `tile::Spline1D_Grid`: Target tile whose `physical` array is written
- `splineBuffer::Array{Float64}`: Derivative buffer (unused in 1D; retained for API compatibility)

# Returns
- `tile.physical`

See also: [`tileTransform!`](@ref), [`splineTransform!`](@ref)
"""
function gridTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::SpringsteelGridParameters, tile::Spline1D_Grid, splineBuffer::Array{Float64})

    for i in eachindex(patchSplines)
        patchSplines[i].b .= view(patchSpectral,:,i)
        SAtransform!(patchSplines[i])

        # Assign to the tile grid
        SItransform(patchSplines[i],getGridpoints(tile),view(tile.physical,:,i,1))
        SIxtransform(patchSplines[i],getGridpoints(tile),view(tile.physical,:,i,2))
        SIxxtransform(patchSplines[i],getGridpoints(tile),view(tile.physical,:,i,3))
    end

    return tile.physical
end

"""
    tileTransform!(patchSplines, patchSpectral, pp, tile, splineBuffer)

Fast physical-space evaluation using pre-computed A-coefficients.

Differs from [`gridTransform!`](@ref) (the worker-callable variant) in that `patchSpectral`
is assumed to already contain the **A-coefficients** (output of SA transform) rather than
B-vectors.  Skips the SA solve step, making it faster when A-coefficients are already
available (e.g. in the second pass of a split-step integration).

# Arguments
- `patchSplines::Array{Spline1D}`: Spline objects for the patch
- `patchSpectral::Array{Float64}`: A-coefficient array (already SA-transformed)
- `pp::SpringsteelGridParameters`: Patch parameters (unused in 1D; retained for API compatibility)
- `tile::Spline1D_Grid`: Target tile whose `physical` array is written
- `splineBuffer::Array{Float64}`: Derivative buffer (unused in 1D; retained for API compatibility)

# Returns
- `tile.physical`

See also: [`gridTransform!`](@ref), [`splineTransform!`](@ref)
"""
function tileTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::SpringsteelGridParameters, tile::Spline1D_Grid, splineBuffer::Array{Float64})

    for i in eachindex(patchSplines)
        patchSplines[i].a .= view(patchSpectral,:,i)

        # Assign to the tile grid
        SItransform(patchSplines[i],getGridpoints(tile),view(tile.physical,:,i,1))
        SIxtransform(patchSplines[i],getGridpoints(tile),view(tile.physical,:,i,2))
        SIxxtransform(patchSplines[i],getGridpoints(tile),view(tile.physical,:,i,3))
    end

    return tile.physical
end

"""
    splineTransform!(patchSplines, patchSpectral, pp, sharedSpectral, tile)

Assemble per-tile B-vectors from a shared array into patch A-coefficients (SA step).

This is the spectral-assembly step in the distributed transform pipeline. Because the
SB transform produces overlapping B-vector contributions in the halo region, this step
must be performed with exclusive access to the shared array (cannot be parallelised).
Once assembled, call [`tileTransform!`](@ref) on each worker to complete the physical
evaluation.

# Arguments
- `patchSplines::Array{Spline1D}`: Patch spline objects (one per variable)
- `patchSpectral::Array{Float64}`: Output array; receives A-coefficients for each variable
- `pp::SpringsteelGridParameters`: Patch parameters (unused in 1D; retained for API compatibility)
- `sharedSpectral::SharedArray{Float64}`: Shared array holding the summed B-vectors from all tiles
- `tile::Spline1D_Grid`: Tile (unused in 1D; retained for API compatibility)

See also: [`tileTransform!`](@ref), [`sumSharedSpectral`](@ref)
"""
function splineTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::SpringsteelGridParameters, sharedSpectral::SharedArray{Float64},tile::Spline1D_Grid)
    for i in eachindex(patchSplines)
        patchSpectral[:,i] .= SAtransform(patchSplines[i], view(sharedSpectral,:,i))
    end
end

"""
    sumSpectralTile!(patch::Spline1D_Grid, tile::Spline1D_Grid) -> Array{Float64}
    sumSpectralTile(spectral_patch, spectral_tile, spectralIndexL, spectralIndexR) -> Array{Float64}

Accumulate a tile's B-vector contribution into the patch spectral array.

Adds `tile.spectral` (or `spectral_tile`) into the corresponding rows of `patch.spectral`
(or `spectral_patch`) identified by `[spectralIndexL:spectralIndexR]`.  Because B-vectors
from adjacent tiles overlap in the halo region, multiple tiles must be summed before
performing the SA transform.

# Variants
- `sumSpectralTile!(patch, tile)` — convenience wrapper; reads indices from `tile.params`
- `sumSpectralTile(spectral_patch, spectral_tile, spectralIndexL, spectralIndexR)` — lower-level

# Returns
- The modified patch spectral array

See also: [`setSpectralTile!`](@ref), [`sumSharedSpectral`](@ref)
"""
function sumSpectralTile!(patch::Spline1D_Grid, tile::Spline1D_Grid)
    return spectral
end

function sumSpectralTile(spectral_patch::Array{real}, spectral_tile::Array{real},
                         spectralIndexL::int, spectralIndexR::int)

    # Add the tile b's to the patch
    spectral_patch[spectralIndexL:spectralIndexR,:] =
        spectral_patch[spectralIndexL:spectralIndexR,:] .+ spectral_tile[:,:]
    return spectral_patch
end

"""
    setSpectralTile!(patch::Spline1D_Grid, tile::Spline1D_Grid) -> Array{Float64}
    setSpectralTile(patchSpectral, pp, tile) -> Array{Float64}

Zero the patch spectral array and write a single tile's B-vector contribution into it.

Equivalent to zeroing the patch spectral array and then calling [`sumSpectralTile!`](@ref)
with one tile.  Useful when only one tile contributes (e.g. single-tile patches or testing).

# Variants
- `setSpectralTile!(patch, tile)` — mutates `patch.spectral` in-place using a 4-argument
  lower-level call (reads indices from `tile.params`)
- `setSpectralTile(patchSpectral, pp, tile)` — clears `patchSpectral`, then inserts
  `tile.spectral` at the tile's spectral indices; `pp` is unused in 1D (API compatibility)

# Returns
- The modified patch spectral array

See also: [`sumSpectralTile!`](@ref)
"""
function setSpectralTile!(patch::Spline1D_Grid, tile::Spline1D_Grid)
    return spectral
end

function setSpectralTile(patchSpectral::Array{real}, pp::SpringsteelGridParameters, tile::Spline1D_Grid)

    # pp::SpringsteelGridParameters is patch parameters, but this is not needed for 1D case
    # It is retained for compatibility with calling function for more complex cases

    # Clear the patch
    patchSpectral[:] .= 0.0

    spectralIndexL = tile.params.spectralIndexL
    spectralIndexR = tile.params.spectralIndexR

    # Add the tile b's to the patch
    patchSpectral[spectralIndexL:spectralIndexR,:] .= tile.spectral[:,:]
    return patchSpectral
end

"""
    sumSharedSpectral(sharedSpectral, borderSpectral, pp, tile)

Write a tile's interior B-vectors into the shared array and add the halo contributions.

Designed for the lock-free portion of the distributed SB transform pipeline:
- Interior B-vector rows (indices `spectralIndexL : spectralIndexR-3`) are written
  exclusively (no overlap with other workers).
- The three halo rows at the left boundary (`spectralIndexL : spectralIndexL+2`) are
  **summed** with `borderSpectral` (contributions from the left-adjacent tile).

# Arguments
- `sharedSpectral::SharedArray{Float64}`: Shared memory array accumulating the full-patch B-vector
- `borderSpectral::SparseMatrixCSC{Float64, Int64}`: Sparse matrix of halo B-vector contributions
  from the right end of this tile (obtained via [`getBorderSpectral`](@ref))
- `pp::SpringsteelGridParameters`: Patch parameters (unused in 1D; retained for API compatibility)
- `tile::Spline1D_Grid`: Source tile providing B-vectors and spectral index offsets

See also: [`getBorderSpectral`](@ref), [`splineTransform!`](@ref)
"""
function sumSharedSpectral(sharedSpectral::SharedArray{real}, borderSpectral::SparseArrays.SparseMatrixCSC{Float64, Int64}, pp::SpringsteelGridParameters, tile::Spline1D_Grid)

    # Indices of sharedArray that won't be touched by other workers
    siL = tile.params.spectralIndexL
    siR = tile.params.spectralIndexR - 3

    # Indices of tile spectral that map to shared
    tiL = 1
    tiR = tile.params.b_iDim-3

    # Indices of border matrix
    biL = tile.params.spectralIndexL
    biR = biL + 2

    # Set the tile b's in the shared array with the local values
    sharedSpectral[siL:siR,:] .= tile.spectral[tiL:tiR,:]

    # Sum with the border values
    #sharedSpectral[:,:] .= sum([sharedSpectral, borderSpectral])
    sharedSpectral[biL:biR,:] .= sharedSpectral[biL:biR,:] + borderSpectral[biL:biR,:]

    return nothing
end

"""
    getBorderSpectral(pp, tile, patchSpectral) -> SparseMatrixCSC{Float64, Int64}

Extract the tile's rightmost halo B-vectors as a sparse matrix for sending to adjacent workers.

The last three B-vector rows of a tile (indices `b_iDim-2 : b_iDim`) capture basis
functions that overlap with the next tile's domain.  This function packages those rows
into a sparse matrix keyed to the correct global patch indices so that the receiving
worker can add them via [`sumSharedSpectral`](@ref).

# Arguments
- `pp::SpringsteelGridParameters`: Patch parameters (unused in 1D; retained for API compatibility)
- `tile::Spline1D_Grid`: Source tile
- `patchSpectral::Array{Float64}`: Pre-allocated working buffer (patch-sized); zeroed and reused

# Returns
- `SparseMatrixCSC{Float64, Int64}`: Sparse matrix containing the 3 halo B-vector rows

See also: [`sumSharedSpectral`](@ref), [`calcHaloMap`](@ref)
"""
function getBorderSpectral(pp::SpringsteelGridParameters, tile::Spline1D_Grid, patchSpectral::Array{Float64})

    # Clear the local border matrix that will be sent to other workers
    patchSpectral[:] .= 0.0

    # Indices of border matrix
    biL = tile.params.spectralIndexR - 2
    biR = biL + 2

    # Indices of tile to shared
    tiL = tile.params.b_iDim-2
    tiR = tile.params.b_iDim

    # Add the b's to the border matrix
    patchSpectral[biL:biR,:] .= tile.spectral[tiL:tiR,:]

    return sparse(patchSpectral)
end

"""
    calcPatchMap(patch::Spline1D_Grid, tile::Spline1D_Grid) -> (BitArray, SubArray)

Return Boolean index masks identifying the interior (non-halo) region of a tile within
the patch spectral array.

Used in the distributed transform to determine which elements of the shared spectral
array can be written exclusively by this worker (i.e. without conflict with adjacent tiles).

# Arguments
- `patch::Spline1D_Grid`: Full-domain patch
- `tile::Spline1D_Grid`: Sub-domain tile

# Returns
- `patchMap::BitArray`: `true` at patch spectral indices `[spectralIndexL : spectralIndexR-3, :]`
  (interior rows belonging exclusively to this tile)
- `view(tile.spectral, tileView)`: Corresponding view into the tile's spectral array

See also: [`calcHaloMap`](@ref), [`calcTileSizes`](@ref)
"""
function calcPatchMap(patch::Spline1D_Grid, tile::Spline1D_Grid)
    tileView = falses(size(tile.spectral))

    # Indices of sharedArray that won't be touched by other workers
    siL = tile.params.spectralIndexL
    siR = tile.params.spectralIndexR - 3

    patchMap[siL:siR,:] .= true

    # Indices of tile spectral that map to shared
    tiL = 1
    tiR = tile.params.b_iDim-3

    tileView[tiL:tiR, :] .= true

    return patchMap, view(tile.spectral, tileView)
end

"""
    calcHaloMap(patch::Spline1D_Grid, tile::Spline1D_Grid) -> (BitArray, SubArray)

Return Boolean index masks identifying the halo (overlap) region at the right edge of a
tile within the patch spectral array.

The halo consists of the three rightmost B-vector rows that overlap with the adjacent
tile.  These must be summed (not exclusively written) during distributed transforms.

# Arguments
- `patch::Spline1D_Grid`: Full-domain patch
- `tile::Spline1D_Grid`: Sub-domain tile

# Returns
- `haloMap::BitArray`: `true` at patch spectral indices `[spectralIndexR-2 : spectralIndexR, :]`
- `view(tile.spectral, haloView)`: Corresponding view into the last 3 rows of the tile spectral array

See also: [`calcPatchMap`](@ref), [`getBorderSpectral`](@ref)
"""
function calcHaloMap(patch::Spline1D_Grid, tile::Spline1D_Grid)
    haloView = falses(size(tile.spectral))

    # Indices of border matrix
    hiL = tile.params.spectralIndexR - 2
    hiR = hiL + 2

    haloMap[hiL:hiR,:] .= true

    # Indices of tile to shared
    tiL = tile.params.b_iDim-2
    tiR = tile.params.b_iDim

    haloView[tiL:tiR,:] .= true

    return haloMap, view(tile.spectral, haloView)
end

"""
    allocateSplineBuffer(patch::Spline1D_Grid, tile::Spline1D_Grid) -> Array{Float64}

Allocate the spline derivative buffer required for multi-dimensional transform pipelines.

In 1D this buffer is not used; the function returns a zero-element array. It exists to
provide a uniform interface across all grid types (`Spline1D`, `Spline2D`, etc.) so that
higher-level tiling code can call `allocateSplineBuffer` without branching on grid type.

# Arguments
- `patch::Spline1D_Grid`: Full-domain patch (unused in 1D)
- `tile::Spline1D_Grid`: Sub-domain tile (unused in 1D)

# Returns
- `zeros()` — an empty `Array{Float64, 0}` (placeholder)

See also: [`tileTransform!`](@ref), [`splineTransform!`](@ref)
"""
function allocateSplineBuffer(patch::Spline1D_Grid, tile::Spline1D_Grid)
    return zeros()
end

"""
    num_columns(grid::Spline1D_Grid) -> Int64

Return the number of vertical columns in the grid.

Always returns `0` for `Spline1D_Grid` because there is no vertical (Z) dimension.
This function provides a uniform interface across grid types.

See also: [`Spline1D_Grid`](@ref)
"""
function num_columns(grid::Spline1D_Grid)
    return 0
end

"""
    getRegularGridpoints(grid::Spline1D_Grid) -> Vector{Float64}

Return evenly-spaced output gridpoint locations spanning the domain.

Unlike [`getGridpoints`](@ref) which returns the (unevenly-spaced) Gaussian mish points,
this function returns `i_regular_out` uniformly-spaced locations from `iMin` to `iMax`.
The number of points is set by `grid.params.i_regular_out` (default: `num_cells + 1`,
which places one point per cell plus the right boundary).

Typically used with [`regularGridTransform`](@ref) to produce output on a regular grid
suitable for visualisation or file I/O.

# Arguments
- `grid::Spline1D_Grid`: The grid object

# Returns
- `Vector{Float64}`: `i_regular_out` uniformly-spaced x-coordinates in `[iMin, iMax]`

See also: [`regularGridTransform`](@ref), [`getGridpoints`](@ref)
"""
function getRegularGridpoints(grid::Spline1D_Grid)
    num_gridpoints = grid.params.i_regular_out
    gridpoints = zeros(Float64, num_gridpoints)
    x_incr = (grid.params.iMax - grid.params.iMin) / (num_gridpoints - 1)
    for x_i = 1:num_gridpoints
        x = grid.params.iMin + (x_i-1)*x_incr
        if x > grid.params.iMax
            x = grid.params.iMax
        end
        gridpoints[x_i] = x
    end
    return gridpoints
end

"""
    regularGridTransform(grid::Spline1D_Grid, gridpoints::Array{Float64}) -> Array{Float64}

Evaluate the B-spline representation at arbitrary (e.g. regularly-spaced) output locations.

Performs the SA transform on the current `grid.spectral` coefficients and then evaluates
the field and its first and second derivatives at every point in `gridpoints`.

# Arguments
- `grid::Spline1D_Grid`: Grid with current spectral coefficients in `grid.spectral`
- `gridpoints::Array{Float64}`: Output evaluation locations; typically from
  [`getRegularGridpoints`](@ref) but any points within `[iMin, iMax]` are valid

# Returns
- `Array{Float64}` of shape `(length(gridpoints), nvars, 3)` where the third axis is:
  - `[:, :, 1]` — field values
  - `[:, :, 2]` — first derivatives (\u2202/\u2202x)
  - `[:, :, 3]` — second derivatives (\u2202\u00b2/\u2202x\u00b2)

# Example
```julia
spectralTransform!(grid)
regPoints = getRegularGridpoints(grid)
regPhysical = regularGridTransform(grid, regPoints)
```

See also: [`getRegularGridpoints`](@ref), [`gridTransform!`](@ref)
"""
function regularGridTransform(grid::Spline1D_Grid, gridpoints::Array{Float64})

    physical = zeros(Float64, length(gridpoints), 
        length(values(grid.params.vars)),3)
    
    for i in eachindex(grid.splines)
        grid.splines[i].b .= view(grid.spectral,:,i)
        SAtransform!(grid.splines[i])
        
        # Assign the grid array
        SItransform(grid.splines[i],gridpoints,view(physical,:,i,1))
        SIxtransform(grid.splines[i],gridpoints,view(physical,:,i,2))
        SIxxtransform(grid.splines[i],gridpoints,view(physical,:,i,3))
    end
    
    return physical
end


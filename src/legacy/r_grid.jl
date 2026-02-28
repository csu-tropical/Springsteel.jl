# Functions for the R Grid

"""
    R_Grid <: AbstractGrid

One-dimensional spectral grid using cubic B-spline basis functions in the radial (R) direction.

# Fields
- `params::GridParameters`: Grid configuration including domain bounds, resolution, and boundary conditions
- `splines::Array{Spline1D}`: Array of 1D spline objects, one per variable
- `spectral::Array{Float64}`: Spectral coefficients array with dimensions `(b_rDim, vars)`
- `physical::Array{Float64}`: Physical space values with dimensions `(rDim, vars, 3)` where the last dimension stores `[value, derivative, second_derivative]`

# Description
`R_Grid` provides a one-dimensional spectral representation using cubic B-splines. The grid supports:
- Multiple variables with independent boundary conditions
- Variable-specific filter lengths (l_q parameter)
- Efficient spectral transforms between physical and spectral space
- Domain tiling for parallel/distributed computing
- Automatic computation of derivatives up to second order

# Example
```julia
using Springsteel

# Create a simple 1D grid
gp = GridParameters(
    geometry = "R",
    xmin = 0.0,
    xmax = 10.0,
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

See also: [`create_R_Grid`](@ref), [`GridParameters`](@ref), [`spectralTransform!`](@ref), [`gridTransform!`](@ref)
"""
struct R_Grid <: AbstractGrid
    params::GridParameters
    splines::Array{Spline1D}
    spectral::Array{Float64}
    physical::Array{Float64}
end

"""
    create_R_Grid(gp::GridParameters) -> R_Grid

Create a one-dimensional spectral grid with cubic B-spline basis functions.

# Arguments
- `gp::GridParameters`: Grid parameters specifying domain, resolution, boundary conditions, and variables

# Returns
- `R_Grid`: Initialized grid with allocated spline objects and arrays

# Description
Constructs an `R_Grid` by:
1. Allocating spectral and physical space arrays based on grid dimensions
2. Creating a `Spline1D` object for each variable with specified boundary conditions
3. Supporting variable-specific filter lengths via `gp.l_q` dictionary

The resulting grid is ready for spectral transforms and derivative computation.

# Example
```julia
gp = GridParameters(
    geometry = "R",
    xmin = -1.0,
    xmax = 1.0,
    num_cells = 50,
    BCL = Dict("temperature" => CubicBSpline.R0, "velocity" => CubicBSpline.R1),
    BCR = Dict("temperature" => CubicBSpline.R0, "velocity" => CubicBSpline.R1),
    vars = Dict("temperature" => 1, "velocity" => 2),
    l_q = Dict("temperature" => 3.0)  # Custom filter length for temperature
)

grid = create_R_Grid(gp)
```

See also: [`R_Grid`](@ref), [`GridParameters`](@ref)
"""
function create_R_Grid(gp::GridParameters)

    # Create a 1-D grid with bSplines as basis
    splines = Array{Spline1D}(undef,1,length(values(gp.vars)))
    spectral = zeros(Float64, gp.b_rDim, length(values(gp.vars)))
    physical = zeros(Float64, gp.rDim, length(values(gp.vars)), 3)
    grid = R_Grid(gp, splines, spectral, physical)
    for key in keys(gp.vars)
        # Allow for filter length to be variable specific
        var_l_q = 2.0
        if haskey(gp.l_q,key)
            var_l_q = gp.l_q[key]
        end
        grid.splines[1,gp.vars[key]] = Spline1D(SplineParameters(
            xmin = gp.xmin,
            xmax = gp.xmax,
            num_cells = gp.num_cells,
            l_q = var_l_q,
            BCL = gp.BCL[key],
            BCR = gp.BCR[key]))
    end
    return grid
end

function calcTileSizes(patch::R_Grid, num_tiles::int)

    # Calculate the appropriate tile size for the given patch
    num_gridpoints = patch.params.rDim
    q,r = divrem(num_gridpoints, num_tiles)
    tile_sizes = [i <= r ? q+1 : q for i = 1:num_tiles]
    if any(x->x<9, tile_sizes)
        throw(DomainError(0, "Too many tiles for this grid (need at least 3 cells in R direction)"))
    end

    # Calculate the dimensions and set the parameters
    DX = (patch.params.xmax - patch.params.xmin) / patch.params.num_cells

    xmins = zeros(Float64,num_tiles)
    xmaxs = zeros(Float64,num_tiles)
    num_cells = zeros(Int64,num_tiles)
    spectralIndicesL = ones(Int64,num_tiles)

    # First tile starts on the patch boundary
    xmins[1] = patch.params.xmin
    num_cells[1] = Int64(ceil(tile_sizes[1] / 3))
    xmaxs[1] = (num_cells[1] * DX) + xmins[1]
    # Implicit spectralIndicesL = 1

    for i = 2:num_tiles-1
        xmins[i] = xmaxs[i-1]
        num_cells[i] = Int64(ceil(tile_sizes[i] / 3))
        xmaxs[i] = (num_cells[i] * DX) + xmins[i]
        spectralIndicesL[i] = num_cells[i-1] + spectralIndicesL[i-1]
    end

    # Last tile ends on the patch boundary
    if num_tiles > 1
        xmins[num_tiles] = xmaxs[num_tiles-1]
        xmaxs[num_tiles] = patch.params.xmax
        spectralIndicesL[num_tiles] = num_cells[num_tiles-1] + spectralIndicesL[num_tiles-1]
        num_cells[num_tiles] = patch.params.num_cells - spectralIndicesL[num_tiles] + 1
    end

    tile_params = vcat(xmins', xmaxs', num_cells', spectralIndicesL', tile_sizes')
    return tile_params
end

"""
    getGridpoints(grid::R_Grid) -> Vector{Float64}

Return the physical locations of all gridpoints in the R direction.

# Arguments
- `grid::R_Grid`: The grid object

# Returns
- `Vector{Float64}`: Array of gridpoint locations (mesh points) in the radial direction

# Description
Returns the mesh points (collocation points) where the physical field values are defined.
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

See also: [`R_Grid`](@ref)
"""
function getGridpoints(grid::R_Grid)

    # Return an array of the gridpoint locations
    return grid.splines[1].mishPoints
end

"""
    spectralTransform!(grid::R_Grid) -> Array{Float64}

Transform field values from physical space to spectral (B-spline coefficient) space.

# Arguments
- `grid::R_Grid`: Grid containing physical values in `grid.physical[:, :, 1]`

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

See also: [`gridTransform!`](@ref), [`R_Grid`](@ref)
"""
function spectralTransform!(grid::R_Grid)
    
    # Transform from the grid to spectral space
    # For R grid, the only varying dimension is the variable name
    spectral = spectralTransform(grid, grid.physical, grid.spectral)
    return spectral
end

function spectralTransform(grid::R_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the grid to spectral space
    # For R grid, the only varying dimension is the variable name
    for i in eachindex(grid.splines)
        b = SBtransform(grid.splines[i], physical[:,i,1])
        
        # Assign the spectral array
        spectral[:,i] .= b
    end
end

#function spectralxTransform(grid::R_Grid, physical::Array{real}, spectral::Array{real})
#    
#    # Transform from the grid to spectral space
#    # For R grid, the only varying dimension is the variable name
#    # Need to use a R0 BC for this!
#    Fspline = Spline1D(SplineParameters(xmin = grid.params.xmin, 
#            xmax = grid.params.xmax,
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

function spectralxTransform(grid::R_Grid, physical::Array{real}, spectral::Array{real})

    # Not implemented

end

"""
    gridTransform!(grid::R_Grid) -> Array{Float64}

Transform from spectral (B-spline coefficient) space to physical space with derivatives.

# Arguments
- `grid::R_Grid`: Grid containing spectral coefficients in `grid.spectral`

# Returns
- `Array{Float64}`: Physical values and derivatives (also stored in `grid.physical`)

# Description
Performs the inverse spectral transform, evaluating the B-spline representation at all
gridpoints. For each variable, computes:
- `grid.physical[:, var, 1]`: Field values
- `grid.physical[:, var, 2]`: First derivatives (∂/∂r)
- `grid.physical[:, var, 3]`: Second derivatives (∂²/∂r²)

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

See also: [`spectralTransform!`](@ref), [`R_Grid`](@ref)
"""
function gridTransform!(grid::R_Grid)
    
    # Transform from the spectral to grid space
    # For R grid, the only varying dimension is the variable name
    physical = gridTransform(grid, grid.physical, grid.spectral)
    return physical
end

function gridTransform(grid::R_Grid, physical::Array{real}, spectral::Array{real})

    # Transform from the spectral to grid space
    # For R grid, the only varying dimension is the variable name
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

function gridTransform!(patch::R_Grid, tile::R_Grid)

    # Transform from the spectral to grid space
    # For R grid, the only varying dimension is the variable name
    # Have to use the patch spline and spectral array
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

function gridTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::GridParameters, tile::R_Grid, splineBuffer::Array{Float64})

    # Transform from the spectral to grid space
    # For R grid, the only varying dimension is the variable name
    # Have to use the patch spline and spectral array

    # pp::GridParameters is patch parameters, but this is not needed for 1D case
    # SplineBuffer holds radial derivatives, but this is not needed for 1D case
    # They are retained for compatibility with calling function for more complex cases

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

function tileTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::GridParameters, tile::R_Grid, splineBuffer::Array{Float64})

    # Transform from the spectral to grid space
    # For R grid, the only varying dimension is the variable name
    # Have to use the patch spline and spectral array

    # pp::GridParameters is patch parameters, but this is not needed for 1D case
    # SplineBuffer holds radial derivatives, but this is not needed for 1D case
    # They are retained for compatibility with calling function for more complex cases

    for i in eachindex(patchSplines)
        patchSplines[i].a .= view(patchSpectral,:,i)

        # Assign to the tile grid
        SItransform(patchSplines[i],getGridpoints(tile),view(tile.physical,:,i,1))
        SIxtransform(patchSplines[i],getGridpoints(tile),view(tile.physical,:,i,2))
        SIxxtransform(patchSplines[i],getGridpoints(tile),view(tile.physical,:,i,3))
    end

    return tile.physical
end

function splineTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::GridParameters, sharedSpectral::SharedArray{Float64},tile::R_Grid)

    # Do a partial transform from B to A for splines only
    for i in eachindex(patchSplines)
        patchSpectral[:,i] .= SAtransform(patchSplines[i], view(sharedSpectral,:,i))
    end
end

function sumSpectralTile!(patch::R_Grid, tile::R_Grid)

    spectral = sumSpectralTile(patch.spectral, tile.spectral, tile.params.spectralIndexL, tile.params.spectralIndexR)
    return spectral
end

#function sumSpectralTile(spectral_patch::Array{real}, spectral_tile::Array{real},
#                         spectralIndexL::int, spectralIndexR::int)
#
#    # Add the tile b's to the patch
#    spectral_patch[spectralIndexL:spectralIndexR,:] =
#        spectral_patch[spectralIndexL:spectralIndexR,:] .+ spectral_tile[:,:]
#    return spectral_patch
#end

function setSpectralTile!(patch::R_Grid, tile::R_Grid)

    spectral = setSpectralTile(patch.spectral, tile.spectral, tile.params.spectralIndexL, tile.params.spectralIndexR)
    return spectral
end

function setSpectralTile(patchSpectral::Array{real}, pp::GridParameters, tile::R_Grid)

    # pp::GridParameters is patch parameters, but this is not needed for 1D case
    # It is retained for compatibility with calling function for more complex cases

    # Clear the patch
    patchSpectral[:] .= 0.0

    spectralIndexL = tile.params.spectralIndexL
    spectralIndexR = tile.params.spectralIndexR

    # Add the tile b's to the patch
    patchSpectral[spectralIndexL:spectralIndexR,:] .= tile.spectral[:,:]
    return patchSpectral
end

function sumSharedSpectral(sharedSpectral::SharedArray{real}, borderSpectral::SparseArrays.SparseMatrixCSC{Float64, Int64}, pp::GridParameters, tile::R_Grid)

    # pp::GridParameters is patch parameters, but this is not needed for 1D case
    # It is retained for compatibility with calling function for more complex cases

    # Indices of sharedArray that won't be touched by other workers
    siL = tile.params.spectralIndexL
    siR = tile.params.spectralIndexR - 3

    # Indices of tile spectral that map to shared
    tiL = 1
    tiR = tile.params.b_rDim-3

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

function getBorderSpectral(pp::GridParameters, tile::R_Grid, patchSpectral::Array{Float64})

    # pp::GridParameters is patch parameters, but this is not needed for 1D case
    # It is retained for compatibility with calling function for more complex cases

    # Clear the local border matrix that will be sent to other workers
    patchSpectral[:] .= 0.0

    # Indices of border matrix
    biL = tile.params.spectralIndexR - 2
    biR = biL + 2

    # Indices of tile to shared
    tiL = tile.params.b_rDim-2
    tiR = tile.params.b_rDim

    # Add the b's to the border matrix
    patchSpectral[biL:biR,:] .= tile.spectral[tiL:tiR,:]

    return sparse(patchSpectral)
end

function calcPatchMap(patch::R_Grid, tile::R_Grid)

    patchMap = falses(size(patch.spectral))
    tileView = falses(size(tile.spectral))

    # Indices of sharedArray that won't be touched by other workers
    siL = tile.params.spectralIndexL
    siR = tile.params.spectralIndexR - 3

    patchMap[siL:siR,:] .= true

    # Indices of tile spectral that map to shared
    tiL = 1
    tiR = tile.params.b_rDim-3

    tileView[tiL:tiR, :] .= true

    return patchMap, view(tile.spectral, tileView)
end

function calcHaloMap(patch::R_Grid, tile::R_Grid)

    haloMap = falses(size(patch.spectral))
    haloView = falses(size(tile.spectral))

    # Indices of border matrix
    hiL = tile.params.spectralIndexR - 2
    hiR = hiL + 2

    haloMap[hiL:hiR,:] .= true

    # Indices of tile to shared
    tiL = tile.params.b_rDim-2
    tiR = tile.params.b_rDim

    haloView[tiL:tiR,:] .= true

    return haloMap, view(tile.spectral, haloView)
end

function allocateSplineBuffer(patch::R_Grid, tile::R_Grid)

    # Not needed for R Grid
    return zeros()
end

function num_columns(grid::R_Grid)

    return 0
end

function getRegularGridpoints(grid::R_Grid)

    # Return an array of regular gridpoint locations spaced at r_grid_incr intervals
    num_gridpoints = Int64(ceil((grid.params.xmax - grid.params.xmin) / grid.params.r_incr_out)) + 1
    gridpoints = zeros(Float64, num_gridpoints)
    for r = 1:num_gridpoints
        r_m = grid.params.xmin + (r-1)*grid.params.r_incr_out
        if r_m > grid.params.xmax
            r_m = grid.params.xmax
        end
        gridpoints[r] = r_m
    end
    return gridpoints
end

function regularGridTransform(grid::R_Grid, gridpoints::Array{Float64})

    # Transform from the spectral to grid space
    # For R grid, the only varying dimension is the variable name

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


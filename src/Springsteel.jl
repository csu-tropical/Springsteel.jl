module Springsteel

# Functions to define a spectral grid
abstract type AbstractGrid end
abstract type SpringsteelGrid end

using CSV
using DataFrames
using SharedArrays
using SparseArrays

#Define some convenient aliases
const real = Float64
const int = Int64
const uint = UInt64

# These are declared as submodules to avoid namespace clashes with each other and other packages
include("CubicBSpline.jl")
include("Fourier.jl")
include("Chebyshev.jl")
using .CubicBSpline, .Fourier, .Chebyshev

export AbstractGrid, GridParameters
export SpringsteelGrid, SpringsteelGridParameters
export CubicBSpline, SplineParameters, Spline1D
export SBtransform, SBtransform!, SAtransform!, SItransform!
export SAtransform, SBxtransform, SItransform, SIxtransform, SIxxtransform
export setMishValues

export FourierParameters, Fourier1D
export FBtransform, FBtransform!, FAtransform!, FItransform!
export FBxtransform, FIxtransform, FIxxtransform

export Chebyshev, ChebyshevParameters, Chebyshev1D
export CBtransform, CBtransform!, CAtransform!, CItransform!
export CBxtransform, CIxtransform, CIxxtransform, CIInttransform

export SplineParameters, Spline1D
export createGrid, getGridpoints, calcTileSizes
export read_physical_grid, write_grid
export calcPatchMap, calcHaloMap, allocateSplineBuffer, num_columns
export spectralTransform!, splineTransform!, tileTransform!, gridTransform!
export regularGridTransform, getRegularGridpoints

"""
    GridParameters

Configuration structure for spectral grid construction using mixed basis functions.

# Fields

## Geometry and Dimensions
- `geometry::String = "R"`: Grid geometry type. Options: `"R"`, `"RZ"`, `"RL"`, `"RR"`, `"RLZ"`, `"RRR"`

## Radial (R) Direction (Cubic B-splines)
- `xmin::Float64 = 0.0`: Minimum radial coordinate
- `xmax::Float64 = 0.0`: Maximum radial coordinate  
- `num_cells::Int64 = 0`: Number of cubic B-spline cells in R direction
- `rDim::Int64`: Number of physical gridpoints (auto: `num_cells * 3`)
- `b_rDim::Int64`: Number of spectral coefficients (auto: `num_cells + 3`)
- `l_q::Dict = Dict("default" => 2.0)`: Filter length parameter (per variable)
- `BCL::Dict`: Left boundary condition dictionary (per variable)
- `BCR::Dict`: Right boundary condition dictionary (per variable)

## Azimuthal (L) Direction (Fourier or Cubic B-splines)
- `ymin::Float64 = 0.0`: Minimum azimuthal coordinate
- `ymax::Float64 = 2π`: Maximum azimuthal coordinate
- `kmax::Dict = Dict("default" => -1)`: Maximum wavenumber (Fourier), -1 for ring-specific
- `lDim::Int64 = 0`: Number of physical gridpoints (auto-calculated from aspect ratio)
- `b_lDim::Int64 = 0`: Number of spectral modes/coefficients
- `BCU::Dict = Fourier.PERIODIC`: Upper boundary condition (Fourier grids)
- `BCD::Dict = Fourier.PERIODIC`: Lower boundary condition (Fourier grids)

## Vertical (Z) Direction (Chebyshev or Cubic B-splines)
- `zmin::Float64 = 0.0`: Minimum vertical coordinate
- `zmax::Float64 = 0.0`: Maximum vertical coordinate
- `zDim::Int64 = 0`: Number of physical gridpoints
- `b_zDim::Int64`: Number of spectral coefficients (auto-calculated)
- `BCB::Dict = Chebyshev.R0`: Bottom boundary condition
- `BCT::Dict = Chebyshev.R0`: Top boundary condition

## Variables and Output
- `vars::Dict = Dict("u" => 1)`: Variable name to index mapping

## Tiling Parameters (for distributed computing)
- `spectralIndexL::Int64 = 1`: Left spectral index for tile
- `spectralIndexR::Int64`: Right spectral index for tile (auto)
- `patchOffsetL::Int64`: Left patch offset in gridpoints (auto)
- `patchOffsetR::Int64`: Right patch offset in gridpoints (auto)
- `tile_num::Int64 = 0`: Tile number identifier

## Regular Output Grid
- `r_regular_out::Int64`: Radial points for regular output (auto: `num_cells + 1`)
- `l_regular_out::Int64`: Azimuthal points for regular output (auto: `rDim*2 + 1`)
- `z_regular_out::Int64`: Vertical points for regular output (auto: `zDim + 1`)

# Description
`GridParameters` uses the `@kwdef` macro, allowing keyword-based construction with defaults.
Many fields are auto-calculated from other parameters, simplifying grid setup.

# Boundary Condition Options
- **CubicBSpline**: `R0` (value), `R1` (derivative), `R2` (second derivative)
- **Fourier**: `PERIODIC`
- **Chebyshev**: `R0`, `R1`, `R2`

# Example: 1D Radial Grid
```julia
gp = GridParameters(
    geometry = "R",
    xmin = 0.0,
    xmax = 10.0,
    num_cells = 20,
    BCL = Dict("temperature" => CubicBSpline.R0),
    BCR = Dict("temperature" => CubicBSpline.R0),
    vars = Dict("temperature" => 1)
)
```

# Example: 2D Grid with Different Basis Functions
```julia
# RR grid: B-splines in both R and L
gp_rr = GridParameters(
    geometry = "RR",
    xmin = 1.0,
    xmax = 5.0,
    num_cells = 30,
    ymin = 0.0,
    ymax = 10.0,
    BCL = Dict("u" => CubicBSpline.R0),
    BCR = Dict("u" => CubicBSpline.R0),
    vars = Dict("u" => 1)
)

# RL grid: B-splines in R, Fourier in L
gp_rl = GridParameters(
    geometry = "RL",
    xmin = 1.0,
    xmax = 5.0,
    num_cells = 30,
    kmax = Dict("vorticity" => 100),
    BCL = Dict("vorticity" => CubicBSpline.R0),
    BCR = Dict("vorticity" => CubicBSpline.R0),
    vars = Dict("vorticity" => 1)
)
```

# Example: 3D Grid
```julia
gp_rrr = GridParameters(
    geometry = "RRR",
    xmin = 0.0,
    xmax = 5.0,
    num_cells = 25,
    ymin = 0.0,
    ymax = 10.0,
    zmin = -2.0,
    zmax = 2.0,
    BCL = Dict("w" => CubicBSpline.R0),
    BCR = Dict("w" => CubicBSpline.R0),
    vars = Dict("w" => 1)
)
```

See also: [`createGrid`](@ref), [`R_Grid`](@ref)
"""
Base.@kwdef struct SpringsteelGridParameters
    geometry::String = "1D"
    iMin::real = 0.0
    iMax::real = 0.0
    num_cells::int = 0
    iDim::int = num_cells * CubicBSpline.mubar
    b_iDim::int = num_cells + 3
    l_q::Dict = Dict("default" => 2.0)
    BCL::Dict = CubicBSpline.R0
    BCR::Dict = CubicBSpline.R0
    jMin::real = 0.0
    jMax::real = 2 * π
    max_wavenumber::Dict = Dict("default" => -1) # Default is -1 to indicate ring specific
    jDim::int = 0
    b_jDim::int = 0
    BCU::Dict = Fourier.PERIODIC
    BCD::Dict = Fourier.PERIODIC
    kMin::real = 0.0
    kMax::real = 0.0
    kDim::int = 0
    b_kDim::int = min(kDim, floor(((2 * kDim) - 1) / 3) + 1)
    BCB::Dict = Chebyshev.R0
    BCT::Dict = Chebyshev.R0
    vars::Dict = Dict("u" => 1)
    # Patch indices
    spectralIndexL::int = 1
    spectralIndexR::int = spectralIndexL + b_iDim - 1
    patchOffsetL::int = (spectralIndexL - 1) * 3
    patchOffsetR::int = patchOffsetL + iDim
    tile_num::int = 0
    # The default i increment is the number of spline cells
    i_regular_out::int = num_cells + 1
    # The default j_increment is the maximum number of wavenumbers on the outermost ring
    j_regular_out::int = (iDim*2) + 1
    k_regular_out::int = kDim + 1
end

"""
    GridParameters

Legacy immutable parameter struct (using `@kwdef`) for configuring spectral grids.
Specifies geometry type, domain bounds, resolution, boundary conditions, and variables.

# Fields
- `geometry`: Grid geometry string, e.g. `"R"`, `"RL"`, `"RZ"`, `"RLZ"`
- `xmin`, `xmax`: Radial domain bounds
- `num_cells`: Number of radial spline cells
- `rDim`: Physical gridpoints in radial direction (`num_cells * mubar`)
- `b_rDim`: Spectral coefficients in radial direction (`num_cells + 3`)
- `l_q`: Filter length Dict for B-splines (default `Dict("default" => 2.0)`)
- `BCL`, `BCR`: Left/right radial boundary condition Dicts
- `ymin`, `ymax`: Azimuthal domain bounds (default `0` to `2π`)
- `kmax`: Maximum Fourier wavenumber Dict
- `zmin`, `zmax`, `zDim`: Vertical domain bounds and gridpoints
- `BCB`, `BCT`: Bottom/top vertical boundary condition Dicts
- `vars`: Variable name-to-index mapping Dict

See also: [`createGrid`](@ref), [`R_Grid`](@ref)
"""
Base.@kwdef struct GridParameters
    geometry::String = "R"
    xmin::real = 0.0
    xmax::real = 0.0
    num_cells::int = 0
    rDim::int = num_cells * CubicBSpline.mubar
    b_rDim::int = num_cells + 3
    l_q::Dict = Dict("default" => 2.0)
    BCL::Dict = CubicBSpline.R0
    BCR::Dict = CubicBSpline.R0
    ymin::real = 0.0
    ymax::real = 2 * π
    kmax::Dict = Dict("default" => -1) # Default is -1 to indicate ring specific
    lDim::int = 0
    b_lDim::int = 0
    BCU::Dict = Fourier.PERIODIC
    BCD::Dict = Fourier.PERIODIC
    zmin::real = 0.0
    zmax::real = 0.0
    zDim::int = 0
    b_zDim::int = min(zDim, floor(((2 * zDim) - 1) / 3) + 1)
    BCB::Dict = Chebyshev.R0
    BCT::Dict = Chebyshev.R0
    vars::Dict = Dict("u" => 1)
    # Patch indices
    spectralIndexL::int = 1
    spectralIndexR::int = spectralIndexL + b_rDim - 1
    patchOffsetL::int = (spectralIndexL - 1) * 3
    patchOffsetR::int = patchOffsetL + rDim
    tile_num::int = 0
    r_regular_out::int = num_cells + 1
    # The default l_increment is the maximum number of wavenumbers on the outermost ring
    # The code will probably break if you change this for RL or RLZ grids
    l_regular_out::int = (rDim*2) + 1
    z_regular_out::int = zDim + 1
end

# Include functions for implemented grids
include("r_grid.jl")
include("spline1D_grid.jl")
include("rz_grid.jl")
include("rl_grid.jl")
include("rr_grid.jl")
include("spline2D_grid.jl")
include("rlz_grid.jl")
include("rrr_grid.jl")

# Not yet implemented
struct Z_Grid <: AbstractGrid
    params::GridParameters
    columns::Array{Chebyshev1D}
    spectral::Array{Float64}
    physical::Array{Float64}
end

# I/O routines
include("io.jl")

"""
    createGrid(gp::GridParameters) -> AbstractGrid

Factory function to create a spectral grid based on geometry specification.

# Arguments
- `gp::GridParameters`: Grid configuration specifying geometry, domain, resolution, and boundary conditions

# Returns
- `AbstractGrid`: Concrete grid type based on `gp.geometry`:
  - `"R"` → [`R_Grid`](@ref): 1D radial grid with cubic B-splines
  - `"RZ"` → `RZ_Grid`: 2D cylindrical grid (B-splines × Chebyshev)
  - `"RL"` → `RL_Grid`: 2D cylindrical grid (B-splines × Fourier)
  - `"RR"` → `RR_Grid`: 2D grid with B-splines in both directions
  - `"RLZ"` → `RLZ_Grid`: 3D cylindrical grid (B-splines × Fourier × Chebyshev)
  - `"RRR"` → `RRR_Grid`: 3D Cartesian grid with B-splines in all directions

# Throws
- `DomainError`: If `gp.geometry` is `"Z"` (not yet implemented) or unrecognized

# Description
This is the primary entry point for creating spectral grids in Springsteel. The function
dispatches to specialized constructors based on the geometry type, each implementing the
appropriate combination of basis functions:

- **Cubic B-splines**: Used for smooth, compact support in physical space
- **Fourier**: Used for periodic azimuthal direction
- **Chebyshev**: Used for non-periodic vertical direction with spectral accuracy

All grids support multiple variables, spectral transforms, derivative computation,
and domain tiling for parallel/distributed computing.

# Example: 1D Grid
```julia
gp = GridParameters(
    geometry = "R",
    xmin = 0.0,
    xmax = 10.0,
    num_cells = 30,
    BCL = Dict("u" => CubicBSpline.R0),
    BCR = Dict("u" => CubicBSpline.R0),
    vars = Dict("u" => 1)
)
grid = createGrid(gp)
```

# Example: 2D Grid with B-splines
```julia
gp = GridParameters(
    geometry = "RR",
    xmin = 1.0,
    xmax = 5.0,
    num_cells = 40,
    ymin = 0.0,
    ymax = 8.0,
    BCL = Dict("vorticity" => CubicBSpline.R0),
    BCR = Dict("vorticity" => CubicBSpline.R1),
    vars = Dict("vorticity" => 1)
)
grid = createGrid(gp)
```

# Example: 3D Grid
```julia
gp = GridParameters(
    geometry = "RRR",
    xmin = -5e5,
    xmax = 5e5,
    num_cells = 50,
    ymin = -5e5,
    ymax = 5e5,
    zmin = 0.0,
    zmax = 2e4,
    BCL = Dict("w" => CubicBSpline.R0),
    BCR = Dict("w" => CubicBSpline.R0),
    vars = Dict("w" => 1)
)
grid = createGrid(gp)
```

See also: [`GridParameters`](@ref), [`R_Grid`](@ref)
"""
function createGrid(gp::GridParameters)

    # Call the respective grid factory
    if gp.geometry == "R"
        # R grid
        grid = create_R_Grid(gp)
        return grid

    elseif gp.geometry == "Spline1D"
        # Spline1D grid
        grid = create_Spline1D_Grid(gp)
        return grid

    elseif gp.geometry == "RZ"
        # RZ grid
        grid = create_RZ_Grid(gp)
        return grid

    elseif gp.geometry == "RL"
        # RL grid
        grid = create_RL_Grid(gp)
        return grid

    elseif gp.geometry == "RR"
        # RR grid
        grid = create_RR_Grid(gp)
        return grid

    elseif gp.geometry == "RLZ"
        # RLZ grid
        grid = create_RLZ_Grid(gp)
        return grid
        
    elseif gp.geometry == "RRR"
        # RRR grid
        grid = create_RRR_Grid(gp)
        return grid

    elseif gp.geometry == "Z"
        # Z grid
        throw(DomainError(0, "Z column model not implemented yet"))
    else
        # Unknown grid
        throw(DomainError(0, "Unknown geometry"))
    end
    
end

"""
    createGrid(gp::SpringsteelGridParameters) -> SpringsteelGrid

Factory function to create a spectral grid using the new SpringsteelGridParameters type.

# Arguments
- `gp::SpringsteelGridParameters`: Grid configuration for generalized geometries

# Returns
- `SpringsteelGrid`: Concrete grid type based on `gp.geometry`:
  - `"Spline1D"` → [`Spline1D_Grid`](@ref): 1D grid with cubic B-splines

# Description
This overload of `createGrid` accepts the newer `SpringsteelGridParameters` type which uses
generalized coordinate names (i, j, k) instead of geometry-specific names (r, l, z).
This allows for more flexible grid definitions that aren't tied to specific physical interpretations.

# Example
```julia
gp = SpringsteelGridParameters(
    geometry = "Spline1D",
    iMin = 0.0,
    iMax = 10.0,
    num_cells = 30,
    BCL = Dict("u" => CubicBSpline.R0),
    BCR = Dict("u" => CubicBSpline.R0),
    vars = Dict("u" => 1)
)
grid = createGrid(gp)
```

See also: [`SpringsteelGridParameters`](@ref), [`Spline1D_Grid`](@ref), [`createGrid(::GridParameters)`](@ref)
"""
function createGrid(gp::SpringsteelGridParameters)

    # Call the respective grid factory
    if gp.geometry == "Spline1D"
        # Spline1D grid
        grid = create_Spline1D_Grid(gp)
        return grid
    
    elseif gp.geometry == "Spline2D"
        # Spline2D grid
        grid = create_Spline2D_Grid(gp)
        return grid
    
    else
        # Unknown geometry for SpringsteelGridParameters
        throw(DomainError(0, "Unknown geometry for SpringsteelGridParameters: $(gp.geometry)"))
    end
    
end

# Module end
end

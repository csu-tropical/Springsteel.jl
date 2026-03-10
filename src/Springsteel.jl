module Springsteel

# Functions to define a spectral grid
abstract type AbstractGrid end

using CSV
using Dates
using DataFrames
using JLD2
using NCDatasets
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

export FourierParameters, Fourier1D, Fourier
export FBtransform, FBtransform!, FAtransform!, FItransform!
export FBxtransform, FIxtransform, FIxxtransform, FIInttransform
export IInttransform

export Chebyshev, ChebyshevParameters, Chebyshev1D
export CBtransform, CBtransform!, CAtransform!, CItransform!
export CBxtransform, CIxtransform, CIxxtransform, CIInttransform

export SplineParameters, Spline1D
export createGrid, getGridpoints, calcTileSizes
export read_physical_grid, write_grid, check_grid_dims
export save_grid, load_grid, write_netcdf, read_netcdf
export calcPatchMap, calcHaloMap, allocateSplineBuffer, num_columns
export calcPatchMap_multidim, calcHaloMap_multidim
export sumSpectralTile!, setSpectralTile!, getBorderSpectral, sumSharedSpectral
export spectralTransform!, splineTransform!, tileTransform!, gridTransform!
export regularGridTransform, getRegularGridpoints
export applyFilter!

# Filter type exports
export AbstractFilter, SpectralFilter, GaussianFilter

# Unified type system exports
export AbstractGeometry, CartesianGeometry, CylindricalGeometry, SphericalGeometry
export AbstractBasisType, SplineBasisType, FourierBasisType, ChebyshevBasisType, NoBasisType
export SplineBasisArray, FourierBasisArray, ChebyshevBasisArray, NoBasisArray
export SL_Grid, SLZ_Grid
export R_Grid, RL_Grid, RZ_Grid, RR_Grid, RLZ_Grid, RRR_Grid, Spline1D_Grid, Spline2D_Grid
# Geometry-name aliases for spline grids
export Polar_Grid, Cylindrical_Grid, Spline3D_Grid, Samurai_Grid, SphericalShell_Grid, Sphere_Grid
# Fourier-based grids: canonical L / LL / LLZ, descriptive Ring1D / Ring2D / DoublyPeriodic
export L_Grid, LL_Grid, LLZ_Grid
export Ring1D_Grid, Ring2D_Grid, DoublyPeriodic_Grid
# Chebyshev-based grids: canonical Z / ZZ / ZZZ, descriptive Column1D / Column2D / Column3D
export Z_Grid, ZZ_Grid, ZZZ_Grid
export Column1D_Grid, Column2D_Grid, Column3D_Grid

# Factory exports
export parse_geometry, compute_derived_params, convert_to_springsteel_params
export num_deriv_slots

# Basis interface exports
export gridpoints, spectral_dim, physical_dim

# Solver framework exports
export OperatorTerm, AbstractSolverBackend, LocalLinearBackend, OptimizationBackend
export SpringsteelProblem, SpringsteelSolution
export operator_matrix, assemble_operator, assemble_from_equation
export solve, solver_gridpoints

# Interpolation framework exports
export grid_from_regular_data, grid_from_netcdf
export interpolate_to_grid, interpolate_to_grid!
export evaluate_unstructured
export cartesian_to_cylindrical, cylindrical_to_cartesian
export cartesian_to_cylindrical_3d, cylindrical_to_cartesian_3d
export cartesian_to_spherical, spherical_to_cartesian
export cylindrical_to_spherical, spherical_to_cylindrical
export latlon_to_spherical, spherical_to_latlon

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
    mubar::int = 3
    quadrature::Symbol = :gauss
    iDim::int = num_cells * mubar
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
    # Spectral filters (per-variable, keyed by variable name or "default")
    fourier_filter::Dict = Dict()
    chebyshev_filter::Dict = Dict()
    # Patch indices
    spectralIndexL::int = 1
    spectralIndexR::int = spectralIndexL + b_iDim - 1
    patchOffsetL::int = (spectralIndexL - 1) * mubar
    patchOffsetR::int = patchOffsetL + iDim
    tile_num::int = 0
    # The default i increment is the number of spline cells
    i_regular_out::int = num_cells + 1
    # The default j_increment is the maximum number of wavenumbers on the outermost ring
    j_regular_out::int = (iDim*2) + 1
    k_regular_out::int = kDim + 1
end

# ── Unified type system ────────────────────────────────────────────────────────────
# Must be included after SpringsteelGridParameters (used as field type in SpringsteelGrid)
include("types.jl")
include("basis_interface.jl")

# ── Grid factory ──────────────────────────────────────────────────────────────────
# Must be included after types.jl (uses geometry/basis sentinel types)
include("factory.jl")
# deprecated.jl included below (after GridParameters is defined)

# ── Cartesian and Cylindrical/Spherical transforms ───────────────────────────
# Must be included after factory.jl (uses _RLGrid and related aliases)
include("transforms_cartesian.jl")


# Must be included after factory.jl (uses _RLGrid alias → SpringsteelGrid{...})
include("transforms_cylindrical.jl")

# Must be included after transforms_cylindrical.jl (shares spectral layout conventions)
include("transforms_spherical.jl")

# ── 1D Tiling ───────────────────────────────────────────────────────────────────────
# Must be included after transforms_*.jl (uses num_columns from Cylindrical/Spherical files)
include("tiling.jl")

# ── Solver framework ──────────────────────────────────────────────────────────────────
# Must be included after basis modules and factory.jl
include("solver.jl")

# ── Interpolation framework ──────────────────────────────────────────────────────────
# Must be included after transforms_*.jl (uses _cheb_eval_pts!) and factory.jl
include("interpolation.jl")

# ── Filtering framework ─────────────────────────────────────────────────────────────
# Must be included after transforms_*.jl (uses grid type aliases)
include("filtering.jl")


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
    mubar::int = 3
    quadrature::Symbol = :gauss
    rDim::int = num_cells * mubar
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
    # Spectral filters (per-variable, keyed by variable name or "default")
    fourier_filter::Dict = Dict()
    chebyshev_filter::Dict = Dict()
    # Patch indices
    spectralIndexL::int = 1
    spectralIndexR::int = spectralIndexL + b_rDim - 1
    patchOffsetL::int = (spectralIndexL - 1) * mubar
    patchOffsetR::int = patchOffsetL + rDim
    tile_num::int = 0
    r_regular_out::int = num_cells + 1
    # The default l_increment is the maximum number of wavenumbers on the outermost ring
    # The code will probably break if you change this for RL or RLZ grids
    l_regular_out::int = (rDim*2) + 1
    z_regular_out::int = zDim + 1
end

# Backward-compatibility helpers (require GridParameters to be defined)
include("deprecated.jl")

# I/O routines
include("io.jl")

# Module end
end

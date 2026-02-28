# Springsteel.jl AI Agent Guide

## Project Overview
Springsteel is a semi-spectral grid engine for numerical modeling that combines cubic B-splines, Fourier, and Chebyshev basis functions. It serves as the computational "steel" foundation for Scythe.jl (numerical model) and Daisho.jl (data analysis/assimilation).

**Core Philosophy**: Mixed spectral methods with domain decomposition support for distributed computing.

## Architecture

### Grid Type Hierarchy
All grids inherit from `AbstractGrid` and follow this structure:
```julia
struct {R,RL,RZ,RLZ}_Grid <: AbstractGrid
    params::GridParameters
    splines::Array{Spline1D}      # Radial (R) dimension
    rings::Array{Fourier1D}        # Azimuthal (L) dimension (RL, RLZ only)
    columns::Array{Chebyshev1D}    # Vertical (Z) dimension (RZ, RLZ only)
    spectral::Array{Float64}       # Spectral coefficients
    physical::Array{Float64}       # Physical space values
end
```

**Dimensions nomenclature**:
- `R` = radial (1D)
- `RL` = radius-lambda/azimuth (2D cylindrical)
- `RZ` = radius-height (2D axisymmetric)
- `RLZ` = radius-lambda-height (3D cylindrical)

### Submodules
Basis functions are separate submodules to avoid namespace collisions:
- `CubicBSpline` - implements Ooyama (2002) cubic B-spline transform method
- `Fourier` - Fourier basis for azimuthal direction
- `Chebyshev` - Chebyshev polynomials for vertical direction

Each exports transform functions: `{S,F,C}Btransform` (physical→spectral), `{S,F,C}Itransform` (spectral→physical), and derivative transforms (`*xtransform`, `*xxtransform`).

## Key Patterns

### Grid Creation Workflow
1. Define `GridParameters` with geometry, domain bounds, cells, boundary conditions (BCs), and variables
2. Call `createGrid(grid_params)` which dispatches to `create_{R,RL,RZ,RLZ}_Grid()`
3. Use `getGridpoints(grid)` to extract physical mesh locations

**Critical**: GridParameters uses immutable structs. To modify parameters (e.g., in RL_Grid creation), create a new instance with updated fields.

### Boundary Conditions
BCs are defined as Dict constants in each basis module:
- **CubicBSpline**: `R0`, `R1T0`, `R1T1`, `R1T2`, `R2T10`, `R2T20`, `R3`, `PERIODIC`
- **Fourier**: `PERIODIC` (typically the only option)
- **Chebyshev**: `R0` (and others)

BCs are specified per-variable in GridParameters: `BCL`/`BCR` (left/right radial), `BCB`/`BCT` (bottom/top vertical).

### Transform Pipeline
**Physical → Spectral**: `spectralTransform!(grid)` 
- R_Grid: applies `SBtransform` (cubic B-spline)
- RL_Grid: `SBtransform` on each ring, then `FBtransform` (Fourier) per radius
- RZ_Grid: `CBtransform` (Chebyshev) on columns, then `SBtransform` on splines
- RLZ_Grid: combines all three

**Spectral → Physical**: `gridTransform!(grid)`
- Inverse operations using `*Itransform` functions

All transforms modify in-place (note `!` convention).

### Variable-Specific Parameters
Grid parameters support per-variable customization:
- `l_q` - filter length for B-splines (default 2.0)
- `kmax` - maximum Fourier wavenumber (default: ring-dependent)

Check with `haskey(gp.l_q, key)` before accessing variable-specific values.

### Distributed Computing
Grids support radial decomposition via "tiles" with halo regions:
- `calcTileSizes(patch, num_tiles)` computes tile dimensions
- Tiles must have ≥3 cells in R direction (≥9 gridpoints)
- Functions: `calcPatchMap`, `calcHaloMap`, `allocateSplineBuffer`
- Tile transforms: `tileTransform!`, `splineTransform!` with shared arrays

## Critical Constants
- `mubar = 3` - fixed mish (basis evaluation) points per cell
- `rDim = num_cells * mubar` - total physical gridpoints in radial direction
- `b_rDim = num_cells + 3` - spectral coefficients in radial direction

## I/O Patterns
- **Input**: `read_physical_grid(file, grid)` from CSV (uses DataFrames)
- **Output**: `write_grid(grid, output_dir, tag)` writes spectral, physical, and gridded CSVs
- Dimension checking via `check_grid_dims()` dispatches on grid type

## Development Workflow
- **Setup**: `julia --project` or `export JULIA_PROJECT=/path/to/Springsteel.jl`
- **Install**: `] dev /path/to/Springsteel.jl` (development) or `] activate; instantiate` (static)
- **Test**: Load with `using Springsteel` (no formal test suite yet; see `notebooks/Springsteel_tests.ipynb`)
- **Dependencies**: CSV, DataFrames, FFTW, LinearAlgebra, SharedArrays, SparseArrays, SuiteSparse

## Common Pitfalls
- **RL/RLZ grids**: `lDim` is computed, not specified (varies by radius). Set `l_regular_out = (rDim*2) + 1` for output.
- **Indexing**: Radial spectral indices use `spectralIndexL`, `spectralIndexR` with patch offsets (`patchOffsetL`).
- **Physical array**: 3rd dimension holds derivatives - `[:,:,1]` is value, `[:,:,2]` first derivative, etc.
- **Type aliases**: Code uses `real = Float64`, `int = Int64` for brevity.

## File Map
- `src/Springsteel.jl` - main module, grid factory
- `src/{CubicBSpline,Fourier,Chebyshev}.jl` - basis function submodules
- `src/{r,rl,rz,rlz}_grid.jl` - grid implementations
- `src/io.jl` - CSV input/output
- `Project.toml` - dependencies

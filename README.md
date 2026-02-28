# Springsteel.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://csu-tropical.github.io/Springsteel.jl/dev/)
[![Build Status](https://github.com/csu-tropical/Springsteel.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/csu-tropical/Springsteel.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Springsteel is a semi-spectral grid engine that uses a mixture of cubic B-spline, Fourier, and Chebyshev basis functions to represent physical variables and their spatial derivatives. The name comes from an amalgamation of "spectral grid engine" and is the name of a particular type of steel used to make swords and other blades. Things made of [spring steel](https://en.wikipedia.org/wiki/Spring_steel) tend to return to their original shape even after deformation, which is analogous to goals of this software to seamlessly transform between spectral and physical space within a variety of different grid geometries. This package provides the "steel" for the [Scythe.jl](https://github.com/mmbell/Scythe.jl) numerical model and the [Daisho.jl](https://github.com/csu-tropical/Daisho.jl) data analysis and assimilation software. Springsteel currently supports a variety of different grids in 1, 2, or 3 dimensions in Cartesian, cylindrical, or spherical geometry using a mix of cubic B-spline, Fourier, or Chebyshev basis functions.

Springsteel has some limited distributed grid capabilities. The grid can be decomposed if B-splines are used as one or more of the basis functions into a user-specified number of "tiles" with a halo at the tile interfaces. This decomposition is useful for parallel processing.

### Installation

After cloning this repository, start Julia using Springsteel.jl as the project directory. This can be done on the command line using `julia --project` or set the JULIA_PROJECT environmental variable:

`export JULIA_PROJECT=/path/to/Springsteel.jl`

To install Springsteel, in the REPL, go into Package mode by pressing `]`. You will see the REPL change color and indicate `pkg` mode. 

If you are actively developing or modifying Springsteel then you can install the module using `dev /path/to/Springsteel.jl` in `pkg` mode. This will update the module as changes are made to the code. You should see the dependencies being installed, and then the Springsteel package will be precompiled. Exit Package mode with ctrl-C.

If you wish to just install a static version of the latest code, run `activate` to activate the package environment. Then, run `instantiate` to install the necessary dependencies. Exit Package mode with ctrl-C.

Test to make sure the precompilation was successful by running `using Springsteel` in the REPL. If everything is successful then you should get no errors and it will just move to a new line.

### Springsteel API

The full API documentation is available at **[https://csu-tropical.github.io/Springsteel.jl/dev/](https://csu-tropical.github.io/Springsteel.jl/dev/)** and covers:

- **[Tutorial](https://csu-tropical.github.io/Springsteel.jl/dev/tutorial/)** — end-to-end worked examples for each grid type
- **[Springsteel Grid API](https://csu-tropical.github.io/Springsteel.jl/dev/springsteel_grid/)** — full reference for all exported functions
- **[CubicBSpline](https://csu-tropical.github.io/Springsteel.jl/dev/cubicbspline/)**, **[Fourier](https://csu-tropical.github.io/Springsteel.jl/dev/fourier/)**, **[Chebyshev](https://csu-tropical.github.io/Springsteel.jl/dev/chebyshev/)** — low-level basis function modules
- **[Developer Notes](https://csu-tropical.github.io/Springsteel.jl/dev/developer_notes/)** — spectral layout conventions, tiling design, and known traps
- **[Testing Guide](https://csu-tropical.github.io/Springsteel.jl/dev/testing_guide/)** — how to run and extend the test suite

#### Quick start

All grids are created from a `SpringsteelGridParameters` struct followed by `createGrid`. A minimal 1-D radial grid:

```julia
using Springsteel

gp = SpringsteelGridParameters(
    geometry  = "R",
    iMin      = 0.0,
    iMax      = 100.0,
    num_cells = 20,
    BCL       = Dict("u" => CubicBSpline.R0),
    BCR       = Dict("u" => CubicBSpline.R0),
    vars      = Dict("u" => 1))

grid = createGrid(gp)
pts  = getGridpoints(grid)          # physical gridpoint locations
```

The `geometry` keyword selects the grid type. Supported values:

| `geometry` | Dimensions | Bases |
|:-----------|:----------:|:------|
| `"R"` | 1D | Spline |
| `"RR"` | 2D Cartesian | Spline × Spline |
| `"RZ"` | 2D Cartesian | Spline × Chebyshev |
| `"RRR"` | 3D Cartesian | Spline × Spline × Spline |
| `"RL"` | 2D Cylindrical | Spline × Fourier |
| `"RLZ"` | 3D Cylindrical | Spline × Fourier × Chebyshev |
| `"SL"` | 2D Spherical | Spline × Fourier (sin θ rings) |
| `"SLZ"` | 3D Spherical | Spline × Fourier × Chebyshev |

#### Core API

All grid types share the same function signatures, dispatched via Julia's multiple dispatch:

```julia
spectralTransform!(grid)            # physical → spectral (in-place)
gridTransform!(grid)                # spectral → physical (in-place)

getGridpoints(grid)                 # native (non-uniform) gridpoint locations
getRegularGridpoints(grid)          # uniform tensor-product output grid
regularGridTransform(grid, pts)     # evaluate spectral field on regular grid

write_grid(grid, output_dir, tag)   # write spectral + physical CSVs
read_physical_grid(file, grid)      # read physical CSV into grid
```

For distributed computing, each grid supports radial decomposition into tiles with halo overlap:

```julia
tiles = calcTileSizes(patch, n)     # decompose into n radial tiles
buf   = allocateSplineBuffer(tile)  # allocate per-tile work buffer
pmap  = calcPatchMap(patch, tile)   # sparse map of interior spectral rows
hmap  = calcHaloMap(patch, t1, t2)  # sparse map of 3-row halo region
sumSpectralTile!(patch, tile)       # accumulate tile spectral into patch
setSpectralTile!(patch, tile)       # zero patch then write tile spectral
```

The supertype of all grids is `AbstractGrid`; the concrete parametric type is
`SpringsteelGrid{Geometry, IBasis, JBasis, KBasis}` which can be used to write
geometry-agnostic code via multiple dispatch.

### Future plans
Support for CF-compliant NetCDF input and output will be added in the near future. Dedicated tiling implementations for 3D grids (RLZ, SLZ) and regular-grid output for 2D Cartesian grids (RR, RZ) are planned. Support for grid nesting using the cubic B-splines will be added in future versions. Interested users are welcome to contribute to improve the grid engine. Stay tuned for more functionality!

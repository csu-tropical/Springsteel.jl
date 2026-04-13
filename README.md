# Springsteel.jl

<p align="center">
  <img src="images/springsteel.png" alt="Springsteel.jl" width="320"/>
</p>

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Stable: [![Stable Build Status](https://github.com/csu-tropical/Springsteel.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/csu-tropical/Springsteel.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://csu-tropical.github.io/Springsteel.jl/stable/)

Development: [![Development Build Status](https://github.com/csu-tropical/Springsteel.jl/actions/workflows/CI.yml/badge.svg?branch=development)](https://github.com/csu-tropical/Springsteel.jl/actions/workflows/CI.yml?query=branch%3Adevelopment) [![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://csu-tropical.github.io/Springsteel.jl/dev/)

[![codecov](https://codecov.io/gh/csu-tropical/Springsteel.jl/graph/badge.svg?token=DM683UZRQM)](https://codecov.io/gh/csu-tropical/Springsteel.jl)

Springsteel is a semi-spectral grid engine that uses a mixture of cubic B-spline, Fourier, and Chebyshev basis functions to represent physical variables and their spatial derivatives on 1D, 2D, and 3D grids in Cartesian, cylindrical, or spherical geometry. The name comes from an amalgamation of "spectral grid engine" and is the name of a particular type of steel used to make swords and other blades. Things made of [spring steel](https://en.wikipedia.org/wiki/Spring_steel) tend to return to their original shape even after deformation, which is analogous to goals of this software to seamlessly transform between spectral and physical space within a variety of different grid geometries. This package provides the "steel" for the [Scythe.jl](https://github.com/mmbell/Scythe.jl) numerical model and the [Daisho.jl](https://github.com/csu-tropical/Daisho.jl) data analysis and assimilation software.

## What's in the box

- **Spectral grids** in Cartesian (`R`, `RR`, `RRR`, `RZ`), cylindrical (`RL`, `RLZ`), and spherical (`SL`, `SLZ`) geometries, plus pure Fourier (`L`, `LL`, `LLZ`) and pure Chebyshev (`Z`, `ZZ`, `ZZZ`) variants.
- **Per-variable boundary conditions** via a basis-agnostic `BoundaryConditions` type system (`DirichletBC`, `NeumannBC`, `RobinBC`, periodic, R3X, …).
- **Linear solver framework** with an operator-algebra DSL (`(∂_x^2 + ∂_y^2) * u => :f`), dense / sparse / Krylov backends, preconditioners, and block multi-variable systems. Workspace-cached for fast repeated solves in time-stepping loops.
- **Grid-to-grid interpolation** including data import (regular arrays, NetCDF), same-geometry tensor-product interpolation, and cross-geometry interpolation via coordinate maps.
- **Grid relocation** — reproject cylindrical grids to a new center for vortex-tracking workflows, with in-place time-stepping support and multiple boundary strategies.
- **Multi-patch grids** — chain or embed B-spline patches with exact R3X coupling at interfaces for non-uniform resolution (e.g., fine eyewall + coarse environment in a hurricane model).
- **Tiled parallelism** — decompose B-spline grids into tiles with halo overlap for shared- or distributed-memory parallelism.
- **Spectral filtering** — per-variable wavenumber-domain filters (box-car, Hann, Lanczos, Gaussian) for Fourier and Chebyshev directions.
- **I/O** — JLD2 for full grid state round-trips, NetCDF for CF-compliant regular-grid output, and CSV for human-readable native-grid dumps.

## Installation

Springsteel is a registered Julia package. To install, open the Julia REPL, enter Package mode by pressing `]`, and run:

```
pkg> add Springsteel
```

Or equivalently from Julia code:

```julia
using Pkg
Pkg.add("Springsteel")
```

For development work, clone the repo and `pkg> dev /path/to/Springsteel.jl` to track local changes.

## Quick start

All grids are created from a `SpringsteelGridParameters` struct followed by `createGrid`. A minimal 1D radial grid:

```julia
using Springsteel

gp = SpringsteelGridParameters(
    geometry  = "R",
    iMin      = 0.0,
    iMax      = 100.0,
    num_cells = 20,
    vars      = Dict("u" => 1),
    BCL       = Dict("u" => NaturalBC()),
    BCR       = Dict("u" => NaturalBC()),
)
grid = createGrid(gp)

pts = getGridpoints(grid)
grid.physical[:, 1, 1] .= exp.(-(pts ./ 20.0).^2)

spectralTransform!(grid)    # physical → spectral
gridTransform!(grid)         # spectral → physical + derivatives
```

After `gridTransform!`, `grid.physical[:, 1, 1]` holds the reconstructed values, `[:, 1, 2]` the first derivative, and `[:, 1, 3]` the second.

For a walkthrough that builds on this through boundary conditions, the solver DSL, interpolation, relocation, and multi-patch grids, see the [Tutorial](https://csu-tropical.github.io/Springsteel.jl/dev/tutorial/).

## Documentation

Full documentation lives at **[csu-tropical.github.io/Springsteel.jl/dev/](https://csu-tropical.github.io/Springsteel.jl/dev/)**. Start here:

| Page | What you'll find |
|:-----|:-----------------|
| [Tutorial](https://csu-tropical.github.io/Springsteel.jl/dev/tutorial/) | End-to-end walkthrough, grid types, solver, interpolation, relocation, I/O |
| [SpringsteelGrid](https://csu-tropical.github.io/Springsteel.jl/dev/springsteel_grid/) | Core grid type, factory, transforms, tiling, I/O reference |
| [Boundary Conditions](https://csu-tropical.github.io/Springsteel.jl/dev/boundary_conditions/) | `BoundaryConditions` type system and convenience constructors |
| [Solver Framework](https://csu-tropical.github.io/Springsteel.jl/dev/solver/) | Operator algebra DSL, `solve` / `solve!`, backends, block systems |
| [Multi-Patch Grids](https://csu-tropical.github.io/Springsteel.jl/dev/multipatch/) | Chain / embedded topologies with R3X coupling |
| [Interpolation](https://csu-tropical.github.io/Springsteel.jl/dev/interpolation/) | Data import, same-geometry and cross-geometry interpolation |
| [Grid Relocation](https://csu-tropical.github.io/Springsteel.jl/dev/relocation/) | RL/RLZ coordinate-shift for vortex-tracking workflows |
| [Spectral Filtering](https://csu-tropical.github.io/Springsteel.jl/dev/filtering/) | Wavenumber-domain filters for Fourier and Chebyshev bases |
| [CubicBSpline](https://csu-tropical.github.io/Springsteel.jl/dev/cubicbspline/) / [Fourier](https://csu-tropical.github.io/Springsteel.jl/dev/fourier/) / [Chebyshev](https://csu-tropical.github.io/Springsteel.jl/dev/chebyshev/) | Low-level basis module reference |
| [Contributing](https://csu-tropical.github.io/Springsteel.jl/dev/contributing/) | Dev workflow, test conventions, architecture invariants, roadmap |

## Contributing

Issues and pull requests are welcome. Before opening a PR, please read the [Contributing](https://csu-tropical.github.io/Springsteel.jl/dev/contributing/) page — it covers running the test suite, adding test groups, the architecture invariants that must not be "cleaned up" (spectral layouts, tile halo arithmetic, per-geometry index formulas), and the v1.1+ roadmap items that are open for contributions.

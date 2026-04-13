```@meta
CurrentModule = Springsteel
```

# Springsteel.jl

```@raw html
<p align="center">
  <img src="assets/logo.png" alt="Springsteel.jl" width="320"/>
</p>
```

Documentation for [Springsteel](https://github.com/csu-tropical/Springsteel.jl).

## Overview

Springsteel is a semi-spectral grid engine for numerical modelling in Julia. It provides
a unified `SpringsteelGrid{G, I, J, K}` parametric type that supports **any combination**
of three basis function families in up to three dimensions:

- **Cubic B-splines** ([CubicBSpline](cubicbspline.md)) — compact support, ideal for the radial/Cartesian direction
- **Fourier series** ([Fourier](fourier.md)) — periodic domains such as azimuthal coordinates
- **Chebyshev polynomials** ([Chebyshev](chebyshev.md)) — non-periodic bounded domains such as vertical coordinates

Springsteel serves as the computational foundation for
[Scythe.jl](https://github.com/mmbell/Scythe.jl) (numerical model) and
[Daisho.jl](https://github.com/mmbell/Daisho.jl) (data analysis/assimilation).

### Supported Grid Types

| Geometry String | Basis Functions | Type Alias | Coordinate System |
|:---|:---|:---|:---|
| `"R"` / `"Spline1D"` | Spline | [`R_Grid`](@ref) | Cartesian 1D |
| `"RR"` / `"Spline2D"` | Spline × Spline | [`RR_Grid`](@ref) | Cartesian 2D |
| `"RZ"` | Spline × Chebyshev | [`RZ_Grid`](@ref) | Cartesian 2D (axisymmetric) |
| `"RL"` | Spline × Fourier | [`RL_Grid`](@ref) | Cylindrical 2D (polar) |
| `"RRR"` | Spline × Spline × Spline | [`RRR_Grid`](@ref) | Cartesian 3D |
| `"RLZ"` | Spline × Fourier × Chebyshev | [`RLZ_Grid`](@ref) | Cylindrical 3D |
| `"SL"` | Spline × Fourier | [`SL_Grid`](@ref) | Spherical 2D |
| `"SLZ"` | Spline × Fourier × Chebyshev | [`SLZ_Grid`](@ref) | Spherical 3D |

## Installation

Springsteel.jl is not yet registered in the Julia General registry. Install directly
from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/mmbell/Springsteel.jl")
```

Or in development mode (for contributors):

```bash
cd /path/to/Springsteel.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### Dependencies

CSV, DataFrames, FFTW, LinearAlgebra, SharedArrays, SparseArrays, SuiteSparse.

## Quick Start

```julia
using Springsteel

# 1. Configure the grid
gp = SpringsteelGridParameters(
    geometry  = "R",
    iMin      = 0.0,
    iMax      = 100.0,
    num_cells = 30,
    BCL       = Dict("u" => NaturalBC()),
    BCR       = Dict("u" => NaturalBC()),
    vars      = Dict("u" => 1)
)

# 2. Build the grid
grid = createGrid(gp)

# 3. Fill the physical array
pts = getGridpoints(grid)
for i in eachindex(pts)
    grid.physical[i, 1, 1] = exp(-(pts[i] / 20.0)^2)
end

# 4. Forward transform (physical → spectral)
spectralTransform!(grid)

# 5. Inverse transform (spectral → physical + derivatives)
gridTransform!(grid)

# grid.physical[:, 1, 1]  → reconstructed values
# grid.physical[:, 1, 2]  → first derivative  ∂f/∂x
# grid.physical[:, 1, 3]  → second derivative ∂²f/∂x²
```

### Derivative Slot Layout

| Dimensions | Slots | Contents |
|:---|:---|:---|
| 1D | 3 | `[f, ∂f/∂i, ∂²f/∂i²]` |
| 2D | 5 | `[f, ∂f/∂i, ∂²f/∂i², ∂f/∂j, ∂²f/∂j²]` |
| 3D | 7 | `[f, ∂f/∂i, ∂²f/∂i², ∂f/∂j, ∂²f/∂j², ∂f/∂k, ∂²f/∂k²]` |

## Documentation Contents

### User guides

- **[Tutorial](tutorial.md)** — step-by-step examples with 1D, 2D, and 3D grids
- **[SpringsteelGrid](springsteel_grid.md)** — core grid type, `createGrid`, transforms, tiling, and I/O
- **[Boundary Conditions](boundary_conditions.md)** — `BoundaryConditions` type system and convenience constructors
- **[Solver Framework](solver.md)** — operator algebra DSL, `solve` / `solve!`, backends, block systems
- **[Multi-Patch Grids](multipatch.md)** — chained and embedded B-spline patches with R3X coupling
- **[Interpolation](interpolation.md)** — data import, same-geometry and cross-geometry interpolation
- **[Grid Relocation](relocation.md)** — RL/RLZ coordinate-shift for vortex-tracking workflows
- **[Spectral Filtering](filtering.md)** — wavenumber-domain filters for Fourier and Chebyshev bases

### Basis modules

- **[CubicBSpline](cubicbspline.md)** — cubic B-spline basis module
- **[Fourier](fourier.md)** — Fourier basis module
- **[Chebyshev](chebyshev.md)** — Chebyshev polynomial basis module

### For contributors

- **[Contributing](contributing.md)** — development workflow, test conventions, architecture invariants, and v1.1+ roadmap

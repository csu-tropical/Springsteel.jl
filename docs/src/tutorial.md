```@meta
CurrentModule = Springsteel
```

# Tutorial

This tutorial walks through the core Springsteel workflow — build a
grid, transform between physical and spectral space, inspect
derivatives — and then shows how the same grid plugs into the solver,
interpolation, relocation, multi-patch, and I/O subsystems.

The goal is to get you from zero to solving a real problem in one
reading. Each subsystem has its own reference page with deeper details:

- [SpringsteelGrid](springsteel_grid.md) — full grid-type catalog
- [Boundary Conditions](boundary_conditions.md)
- [Solver Framework](solver.md)
- [Interpolation](interpolation.md)
- [Grid Relocation](relocation.md)
- [Multi-Patch Grids](multipatch.md)
- [Spectral Filtering](filtering.md)

---

## 1. Your first grid

The simplest Springsteel grid uses cubic B-splines in a single
dimension. `SpringsteelGridParameters` holds every configuration value;
`createGrid` turns it into a live [`SpringsteelGrid`](@ref):

```julia
using Springsteel

gp = SpringsteelGridParameters(
    geometry  = "R",
    iMin      = -50.0,
    iMax      =  50.0,
    num_cells = 30,
    vars      = Dict("gauss" => 1),
    BCL       = Dict("gauss" => NaturalBC()),
    BCR       = Dict("gauss" => NaturalBC()),
)
grid = createGrid(gp)
```

The grid's type is `SpringsteelGrid{CartesianGeometry, SplineBasisArray,
NoBasisArray, NoBasisArray}`, aliased as `R_Grid`. Its storage layout:

| Array            | Shape                         | Contents |
|:-----------------|:------------------------------|:---------|
| `grid.physical`  | `(iDim, n_vars, n_deriv)`     | Function values and derivatives |
| `grid.spectral`  | `(b_iDim, n_vars)`            | B-spline coefficients |

For 1D grids there are 3 derivative slots per variable: `[f, ∂f/∂x,
∂²f/∂x²]`. A 2D grid has 5 slots; 3D has 7.

The `num_cells` determines how many cells the domain is divided into. This is the _spectral mesh_ and determines the grid spacing between Cubic B-spline nodes. The actual physical grid, or _mish_ is finer than this, with a default split into 3 Gaussian quadrature points for each cell. With the default smoothing, the nominal _resolution_ of the grid is therefore close to the nodal spacing but the actual _grid spacing_ is 3 times finer.

## 2. Fill, transform, inspect

Use [`getGridpoints`](@ref) to pull the gridpoint coordinates, fill
`grid.physical[:, var, 1]` with your function values, then run
[`spectralTransform!`](@ref) and [`gridTransform!`](@ref) for a full
round trip:

```julia
pts = getGridpoints(grid)
σ   = 10.0
grid.physical[:, 1, 1] .= exp.(-(pts ./ σ).^2)

spectralTransform!(grid)    # physical → spectral
gridTransform!(grid)         # spectral → physical + derivatives

# Reconstructed values
original   = exp.(-(pts ./ σ).^2)
max_err    = maximum(abs.(grid.physical[:, 1, 1] .- original))    # ≈ 1e-6

# Analytic derivatives
dgauss_dx  = @. -(2 * pts / σ^2) * exp(-(pts / σ)^2)
d2gauss_dx = @. ((2 / σ^2) * (2 * pts^2 / σ^2 - 1)) * exp(-(pts / σ)^2)
@assert maximum(abs.(grid.physical[:, 1, 2] .- dgauss_dx)) < 1e-5
@assert maximum(abs.(grid.physical[:, 1, 3] .- d2gauss_dx)) < 1e-4
```

`gridTransform!` always fills every derivative slot — you don't ask
for a first or second derivative separately, they come as a group.

## 3. Boundary conditions

Constraints at the domain edges are specified per-variable in direction
dicts (`BCL`, `BCR`, `BCD`, `BCU`, `BCB`, `BCT`). The convenience
constructors in [Boundary Conditions](boundary_conditions.md) cover
every case:

```julia
# Dirichlet u = 0 on both ends
BCL = Dict("u" => DirichletBC())
BCR = Dict("u" => DirichletBC())

# Inhomogeneous Dirichlet u(x_max) = 1
BCR = Dict("u" => DirichletBC(1.0))

# Mixed — Dirichlet on left, Neumann (zero flux) on right
BCL = Dict("u" => DirichletBC())
BCR = Dict("u" => NeumannBC())

# Robin αu + βu' = γ
BCR = Dict("T" => RobinBC(1.0, 0.2, 293.15))
```

An older specification that maps to Ooyama (2002) nomenclature `Dict("u" => CubicBSpline.R0)` form still works and can
be mixed in the same spec, but new code should use the
`BoundaryConditions` constructors — they validate and the
physical meaning is obvious at the call site.

## 4. 2D grid with Fourier and multiple variables

A 2D polar grid uses splines in radius and Fourier in azimuth:

```julia
gp = SpringsteelGridParameters(
    geometry   = "RL",
    iMin       = 0.0, iMax = 50.0,
    num_cells  = 20,
    vars       = Dict("u" => 1, "v" => 2),
    BCL        = Dict("default" => NaturalBC()),
    BCR        = Dict("default" => DirichletBC()),
)
grid = createGrid(gp)
```

`Dict("default" => ...)` applies the same BC to every variable — use
explicit keys to override per variable. Fourier dimensions don't take
a user BC spec; they're implicitly periodic.

`getGridpoints` on a 2D grid returns an `(N, 2)` matrix:

```julia
pts   = getGridpoints(grid)       # N × 2: columns are [r, λ]
rvals = pts[:, 1]
λvals = pts[:, 2]

# Fill a Gaussian vortex in u and an m=2 pattern in v
grid.physical[:, 1, 1] .= exp.(-(rvals ./ 15).^2)
grid.physical[:, 2, 1] .= exp.(-(rvals ./ 15).^2) .* cos.(2 .* λvals)
spectralTransform!(grid)
gridTransform!(grid)
```

Each variable has its own slot in the trailing dimension of `physical`
(integers from `vars`), and transforms process them independently — you
never have to loop over variables yourself.

Other frequently-used 2D / 3D geometries:

| Geometry | Basis            | When to use                         |
|:---------|:-----------------|:------------------------------------|
| `RR`     | Spline × Spline  | 2D Cartesian plane                  |
| `RZ`     | Spline × Chebyshev | 2D axisymmetric slab                |
| `RLZ`    | Spline × Fourier × Chebyshev | 3D cylindrical (hurricane-like) |
| `RRR`    | Spline × Spline × Spline | 3D Cartesian volume               |
| `SL` / `SLZ` | Spline × Fourier (× Chebyshev) | Spherical shell / sphere  |

See [`SpringsteelGrid`](springsteel_grid.md) for the complete list of
grid types and their type aliases.

## 5. Solving a boundary value problem

The v1.0 solver framework speaks operator-algebra notation. Build an
unknown [`SpringsteelField`](@ref SpringsteelField), combine derivative
atoms into an `OperatorExpr`, and pair it with an RHS.

For a 1D Poisson problem, configure the grid with two variables — one
for the unknown `u` and one for the RHS `f`:

```julia
gp = SpringsteelGridParameters(
    geometry  = "Z",
    iMin = 0.0, iMax = 1.0, iDim = 25, b_iDim = 25,
    vars = Dict("u" => 1, "f" => 2),
    BCL  = Dict("u" => DirichletBC(), "f" => NaturalBC()),
    BCR  = Dict("u" => DirichletBC(), "f" => NaturalBC()),
)
grid = createGrid(gp)

# 1D Poisson: u''(x) = f(x) with f(x) = -π² sin(πx)  →  u(x) = sin(πx)
pts = solver_gridpoints(grid, "u")
grid.physical[:, 2, 1] .= -π^2 .* sin.(π .* pts)    # write f (slot 2)

u    = SpringsteelField(grid, "u")                   # unknown
prob = SpringsteelProblem(grid, ∂ᵢ^2 * u => :f)      # pulls RHS from :f

solve!(prob)
# Solution is now in grid.physical[:, 1, 1]; the original RHS is still
# in grid.physical[:, 2, 1].
max_err = maximum(abs.(grid.physical[:, 1, 1] .- sin.(π .* pts)))
@assert max_err < 1e-10
```

Why two variables? The RHS `=> :f` tells the solver to read
`grid.physical[:, f_idx, 1]` at every `solve!` call, and the writeback
puts the solution into `grid.physical[:, u_idx, 1]`. Separating the
two means the solve is non-destructive with respect to the RHS and
you can solve repeatedly with updated RHS values — the canonical
time-stepping pattern:

```julia
# For a time-stepping loop:
# while stepping
#     grid.physical[:, 2, 1] .= new_rhs     # refresh f
#     solve!(prob)                           # writes u into slot 1
# end
```

It's technically legal to use a single grid variable as both the
unknown and the RHS (`vars = Dict("u" => 1)` with `∂ᵢ^2 * u => :u`),
and a one-shot solve will still return the correct answer. But the
writeback then overwrites the RHS with the solution, so any second
`solve!` on that problem would read the *previous solution* as the new
RHS — rarely what you want. Use the two-variable pattern unless you
have a specific reason not to.

`solve` is the non-mutating twin — it writes the result into an
independent narrowed grid and returns a [`SpringsteelSolution`](@ref)
whose `physical` and `coefficients` are views into that grid:

```julia
sol = solve(prob)
sol.physical         # view over the solution values
sol.coefficients     # view over the spectral coefficients
sol.grid             # independent SpringsteelGrid — pass to write_grid, etc.
```

`sol` is a snapshot; mutating `grid` after the call doesn't change it.

Derivative atoms: `∂ᵢ`, `∂ⱼ`, `∂ₖ` (generic) and `∂_x`, `∂_y`, `∂_z`,
`∂_r`, `∂_θ`, `∂_λ` (physical — resolved against the grid's geometry).
Build 2D operators by summing:

```julia
L = ∂_x^2 + ∂_y^2                    # Cartesian Laplacian
L = ∂_r^2 + (1/r) * ∂_r                # cylindrical radial Laplacian
prob = SpringsteelProblem(grid, L * u => :f)
```

Backends: `:auto` (default, sparse LU for mixed-basis grids, dense for
pure Chebyshev), `:dense`, `:sparse`, `:krylov`. See
[Solver Framework](solver.md) for preconditioners, block systems, and
the full operator algebra reference.

## 6. Importing data from a regular array

[`grid_from_regular_data`](@ref) lifts raw numpy-style array data onto
a Springsteel grid and runs the forward transform in one call:

```julia
x = collect(0.0:0.05:1.0)              # 21 points, uniform
data = @. exp(-(x - 0.5)^2 / 0.01)

grid = grid_from_regular_data(x, data;
    mubar = 1,
    BCL   = DirichletBC(),
    BCR   = DirichletBC(),
    vars  = Dict("u" => 1),
)
```

Coordinate vectors must be uniformly spaced and (for `mubar ≠ 1`) have
lengths compatible with the internal quadrature. 2D and 3D overloads
take `(x, y, data)` or `(x, y, z, data)` respectively.

For netCDF inputs, [`grid_from_netcdf`](@ref) handles dimension
permutation and per-variable import:

```julia
grid = grid_from_netcdf("rainfall.nc";
    dim_names = ("lon", "lat"),
    var_names = ["precip_mm"],
    BCL = NaturalBC(), BCR = NaturalBC(),
    BCD = NaturalBC(), BCU = NaturalBC(),
)
```

## 7. Interpolating between grids

Once you have a spectral source grid, [`interpolate_to_grid!`](@ref)
moves data to any target grid of the same geometry:

```julia
coarse = grid_from_regular_data(x, data; mubar=1, BCL=DirichletBC(), BCR=DirichletBC(), vars=Dict("u"=>1))

fine_gp = SpringsteelGridParameters(
    geometry = "R", iMin = 0.0, iMax = 1.0, num_cells = 40,
    vars = Dict("u" => 1),
    BCL  = Dict("u" => DirichletBC()),
    BCR  = Dict("u" => DirichletBC()),
)
fine = createGrid(fine_gp)

interpolate_to_grid!(coarse, fine)       # fills fine.physical from coarse spectrum
```

For cross-geometry interpolation (e.g., a Cartesian `RR` analysis into
a cylindrical `RL` hurricane grid) pass a `coordinate_map` callback; see
[Interpolation](interpolation.md).

To evaluate a grid at arbitrary points without building a target grid:

```julia
pts = hcat(rand(100) .* 50.0,            # r
           rand(100) .* 2π)               # λ
vals = evaluate_unstructured(rl_grid, pts; vars=["u"])
```

## 8. Relocating a cylindrical grid

For vortex-tracking workflows where you want the grid origin to follow
a moving center, [`relocate_grid!`](@ref) reprojects all spectral data
to a new Cartesian center in place:

```julia
gp = SpringsteelGridParameters(
    geometry = "RL",
    iMin = 0.0, iMax = 100.0, num_cells = 30,
    vars = Dict("u" => 1),
    BCL = Dict("u" => NaturalBC()), BCR = Dict("u" => NaturalBC()),
)
grid = createGrid(gp)
# ... fill a vortex into grid.physical and spectralTransform!(grid) ...

relocate_grid!(grid, (5.0, 3.0); boundary = :azimuthal_mean)
grid_center(grid)     # → (5.0, 3.0); cumulative across calls
```

Available boundary strategies for the OOB annulus: `:nan`, `:nearest`,
`:azimuthal_mean` (default), `:bc_respecting`. `taper_width=N` smooths
the transition over the last `N` rings. Repeated in-place calls track
the cumulative center shift via [`grid_center`](@ref).

See [Grid Relocation](relocation.md) for the full strategy table,
per-radius fast path details, and multi-patch relocation.

## 9. Multi-patch grids

Chain or embed B-spline patches to cover a large domain with
non-uniform resolution — e.g., a fine inner mesh for a hurricane eyewall
and a coarser outer mesh for the environment:

```julia
mg = createMultiGrid(Dict(
    :topology   => :chain,
    :geometry   => "RL",
    :boundaries => [0.0, 50.0, 100.0],   # 2 patches
    :cells      => [20, 10],              # 2:1 ratio (inner fine)
    :vars       => Dict("u" => 1),
    :BCL        => Dict("u" => NaturalBC()),
    :BCR        => Dict("u" => NaturalBC()),
))

# Fill physical data on every patch, then run the whole thing as one
for patch in mg.mpg.patches
    pts = getGridpoints(patch)
    patch.physical[:, 1, 1] .= exp.(-(pts[:, 1] ./ 15).^2)
end
spectralTransform!(mg)
multiGridTransform!(mg)
```

`multiGridTransform!` handles interface synchronisation — the primary
(outer) patch runs first, `update_interface!` pushes its border
coefficients into the secondary (inner) patch's R3X `ahat` vector, and
then the inner patch transforms. Grid and spectral transforms have no temporary memory allocation overhead, just CPU time.

See [Multi-Patch Grids](multipatch.md) for embedded topology, the 2:1
half-gridpoint constraint, and the R3X coupling matrices.

## 10. Saving and loading grids

Springsteel ships two persistence formats.

### JLD2 — full grid state

[`save_grid`](@ref) / [`load_grid`](@ref) serialise the complete grid
(physical, spectral, basis parameters) to a zstd-compressed JLD2 file.
Reloaded grids are ready to transform immediately — FFTW plans are
reconstructed on load from cached templates.

```julia
save_grid("vortex.jld2", grid)
grid2 = load_grid("vortex.jld2")

spectralTransform!(grid2)    # works, no fresh createGrid needed
gridTransform!(grid2)
```

JLD2 is the format to use for checkpointing, restarting simulations,
and passing spectral state between processes.

### NetCDF — CF-compliant physical output

[`write_grid`](@ref) writes `grid.physical` to a NetCDF file with CF
coordinate metadata suitable for xarray, NCO, CDO, and most visualisation
tools:

```julia
write_grid("vortex.nc", grid; vars = ["u"])
```

The companion [`read_netcdf`](@ref) / [`grid_from_netcdf`](@ref) bring
external data in — `read_netcdf` into plain arrays, `grid_from_netcdf`
straight onto a new `SpringsteelGrid`.

Both functions support variable subsets, attribute metadata, and
per-direction regular output grids via `regularGridTransform`.

## 11. Summary: the core workflow

```julia
using Springsteel

# 1. Configure
gp = SpringsteelGridParameters(
    geometry = "R",
    iMin = 0.0, iMax = 1.0, num_cells = 30,
    vars = Dict("u" => 1),
    BCL  = Dict("u" => DirichletBC()),
    BCR  = Dict("u" => DirichletBC()),
)

# 2. Build
grid = createGrid(gp)

# 3. Fill physical data
pts = getGridpoints(grid)
grid.physical[:, 1, 1] .= sin.(π .* pts)

# 4. Forward transform
spectralTransform!(grid)

# 5. Inverse transform (fills values + derivatives)
gridTransform!(grid)

# 6. (Optional) solve, interpolate, relocate, or write out
```

From here, the feature-specific pages walk through more details
of each subsystem.

```@meta
CurrentModule = Springsteel
```

# Interpolation

Springsteel's interpolation framework moves data between grids of
different resolutions, domains, and geometries while preserving the
spectral representation as much as possible. It has three layers:

1. **Data import** — lift regularly-gridded data (from NetCDF, array
   inputs, external model output) onto a freshly constructed
   `SpringsteelGrid`.
2. **Same-geometry interpolation** — move data between two grids with
   matching `{G, I, J, K}` type parameters (e.g., `RR` → `RR`).
3. **Cross-geometry interpolation** — move data between grids of
   different geometries (e.g., `RR` → `RL`, `SL` → `RLZ`), using
   coordinate mappings to link the physical coordinate systems.

The fourth capability layered on top — **unstructured evaluation** at
arbitrary Cartesian / cylindrical / spherical points — is the building
block for both cross-geometry interpolation and user-facing tools like
[`evaluate_unstructured`](@ref).

## Layer 1 — Data import

If you're starting from raw arrays of values on a regular grid,
`grid_from_regular_data` wraps the whole setup: it creates a
`SpringsteelGrid` with the right dimensions and fills the physical
array. Derivative slots are left `NaN` — call `spectralTransform!` and
`gridTransform!` yourself when you need them.

```@docs
grid_from_regular_data
```

The signature is overloaded for 1D, 2D, and 3D inputs. Coordinate
vectors must be **uniformly spaced** (the factory checks this).

### Sampling conventions

`samples` states where your coordinates sit relative to the grid cells:

| `samples` | coordinates are | domain | `num_cells` default |
|:--|:--|:--|:--|
| `:nodal` (default) | cell **boundaries**, endpoint-inclusive | `[x[1], x[end]]` | `length(x) - 1` |
| `:midpoint` | cell **midpoints** | `[x[1] - h/2, x[end] + h/2]` | `length(x) ÷ mubar` |

`:nodal` is the layout of most gridded datasets and of everything
[`write_netcdf`](@ref) produces, so it is the default. Because the input
points become the cell boundaries, a file written by `write_netcdf` reads
back with its `num_cells` and domain recovered **exactly**, for any cell
count.

!!! warning "The default changed in v1.2.0"
    Coordinates were read as midpoints before v1.2.0. If you omit `samples`
    on a call whose result changed silently — no `num_cells`, and every
    coordinate length divisible by `mubar` — you get a one-time warning
    naming both conventions. Pass `samples = :midpoint` to keep the old
    reading, or `samples = :nodal` to accept the new one and silence it.

    Calls whose lengths were *not* divisible by `mubar` raised an
    `ArgumentError` before, so they changed from error to success and are
    not warned about.

!!! note "Loading regular data always interpolates"
    Values are *projected* onto the grid's quadrature points, not assigned
    to them, and that is unavoidable rather than a design choice: with
    `mubar = 1` the mish sits on cell midpoints, and with `mubar ≥ 2` the
    `:gauss` mish is not uniformly spaced, so no input length ever makes it
    a plain assignment.

    The consequence is a liberating one — **`mubar` and the input length
    are independent**. There is no divisibility requirement under `:nodal`,
    and `num_cells` can be set explicitly to build a grid coarser or finer
    than the data.

    The projection fits a cubic B-spline whose own quadrature points are
    your input coordinates, then evaluates it on the target mish. It is far
    more accurate than linear interpolation (roughly 10× at coarse
    resolution, 250× at fine) and converges at cubic order. Accuracy at the
    natural `write_netcdf` resolution is about 0.4% of field amplitude,
    improving rapidly with more input points.

    For an **exact** round trip, use [`save_grid`](@ref) / [`load_grid`](@ref),
    which store the spectral coefficients directly.

`:midpoint` preserves the historical behaviour exactly: the input *is* the
`:regular` mish, values are assigned rather than projected, and the length
must be divisible by `mubar`.

```julia
using Springsteel

x = collect(0.0:0.05:1.0)        # 21 nodes spanning [0, 1]
data = @. exp(-(x - 0.5)^2 / 0.01)

grid = grid_from_regular_data(x, data;
    mubar = 3,                   # quadrature points per cell
    BCL   = DirichletBC(),
    BCR   = DirichletBC(),
    vars  = ["u"],
)
# 20 cells spanning exactly [0, 1]; mubar is unconstrained by length(x)

# Build a deliberately coarser grid from the same data
coarse = grid_from_regular_data(x, data; num_cells = 5, vars = ["u"])
```

BCs may be bare `BoundaryConditions` (applied to every variable) or
per-variable Dicts (`Dict("u" => DirichletBC(), "v" => NaturalBC())`).
Derivative slots are left as `NaN` to catch accidental use before the
spectral round trip completes.

### NetCDF

```@docs
grid_from_netcdf
```

Reads a NetCDF file and builds a grid from one or more variables. You
pass a `Vector{String}` of dimension names in i/j/k order and a list of
variable names (or `nothing` to import every variable). The factory
handles dimension permutation so Springsteel's i-outer / k-inner layout
is respected regardless of how the file was written.

```julia
grid = grid_from_netcdf("rainfall.nc";
    dim_names = ["lon", "lat"],
    var_names = ["precip_mm"],
    BCL = NaturalBC(), BCR = NaturalBC(),
    BCD = NaturalBC(), BCU = NaturalBC(),
)
```

### Files with a time axis

`grid_from_netcdf` builds a **single spatial grid**, so a time axis is never
treated as a spatial dimension. It is detected, excluded from the grid
dimensions, and sliced out of each data variable. `time_index` chooses the step
(1-based):

```julia
# One step: time_index is optional. This includes anything written by
# write_netcdf(grid; time = t).
grid = grid_from_netcdf("analysis.nc")

# Several steps: time_index is required — a slice is never chosen for you.
grid = grid_from_netcdf("forecast.nc"; time_index = 6)
```

Omitting `time_index` on a file with more than one step raises an `ArgumentError`
naming the time variable and the valid range, rather than silently picking one.

A time axis is recognised from its CF metadata — a `units` attribute of the form
`"<unit> since <origin>"`, `standard_name = "time"`, `axis = "T"`, or a value
NCDatasets has already decoded to a date/time type. The slice is taken at the
time dimension's position in each variable's own dimension list, so it does not
matter whether the file stores `(time, y, x)` or `(y, x, time)`.

!!! note "Limitations"
    - Only one step is loaded. To read every step, or to keep the time
      coordinate itself, use [`read_netcdf`](@ref).
    - The selected time value is not carried onto the returned grid.
    - A coordinate merely *named* `time` or `t` that carries no CF metadata is
      still excluded, but with a warning — it cannot be distinguished from a
      spatial axis with an unfortunate name. Add CF `units` if it is a time axis,
      or rename it if it is not.

!!! note "Round-tripping `write_netcdf` output"
    `write_netcdf` emits `num_cells + 1` endpoint-inclusive nodes, which is
    exactly the `:nodal` convention, so its output reads back with the geometry,
    cell count and domain recovered exactly — for every cell count, with no
    `mubar` constraint. Field *values* are interpolated onto the quadrature
    points (see above), so the round trip is faithful but not bit-exact; use
    [`save_grid`](@ref) / [`load_grid`](@ref) when you need exactness.

## Layer 2 — Same-geometry interpolation

When source and target grids share the same `{G, I, J, K}` parameters
(same geometry, same basis types), interpolation reuses the basis
evaluation machinery directly — there are no coordinate transformations
needed. The source's spectral coefficients are evaluated at the
target's gridpoints via tensor-product B-spline / Fourier / Chebyshev
basis matrices.

```@docs
interpolate_to_grid
interpolate_to_grid!
```

The bang version writes into `target.physical`; the non-bang version
returns the values as a matrix.

Out-of-bounds handling is controlled by `out_of_bounds`:

| Value        | Behaviour                                 |
|:-------------|:------------------------------------------|
| `:nan`       | (default) fill OOB target points with NaN |
| `:error`     | throw an `ArgumentError`                  |
| `<Number>`   | fill OOB points with the given constant   |

Variable matching is by name — the intersection of source and target
`vars` dicts is interpolated; unmatched target variables are left
unchanged and a warning is printed so you don't silently miss data.

```julia
# Interpolate from a coarse RR grid to a fine RR grid
coarse = createGrid(gp_coarse)
fine   = createGrid(gp_fine)
# ... fill coarse.physical and spectralTransform!(coarse) ...
interpolate_to_grid!(coarse, fine)
```

## Layer 3 — Cross-geometry interpolation

When source and target live on different geometries (e.g., a Cartesian
`RR` analysis grid → a cylindrical `RL` hurricane grid) there's no way
to share a basis evaluation path — the target gridpoints live in a
different coordinate system from the source. The framework bridges this
by converting each target point's coordinates into the source's frame,
then evaluating the source spectrally at those unstructured points.

```@docs
interpolate_to_grid(source::SpringsteelGrid, target::SpringsteelGrid; kwargs...)
```

Default coordinate mappings are provided for the common 3D cases:

- `Cartesian ↔ Cylindrical` (2D and 3D)
- `Cartesian ↔ Spherical` (3D)
- `Cylindrical ↔ Spherical` (3D)

For 2D Cartesian ↔ spherical, 2D cylindrical ↔ spherical, and `RR` ↔
`SL` there is no natural default — you must pass an explicit
`coordinate_map` function that takes a matrix of target points
`pts[:, dims]` and returns the corresponding source-frame points.

```julia
# RR → RL: Cartesian analysis to cylindrical hurricane grid centred
#          at (x0, y0).
rl_grid = interpolate_to_grid(rr_grid, rl_target;
    coordinate_map = pts -> begin
        r = pts[:, 1]; λ = pts[:, 2]
        hcat(x0 .+ r .* cos.(λ), y0 .+ r .* sin.(λ))
    end,
)
```

If your source and target grid happen to have matching
`{G, I, J, K}` parameters but you want to go through a coordinate
mapping anyway (e.g., to shift the origin), pass the mapping explicitly
and the framework will route through the unstructured path rather than
the fast tensor-product path.

## Unstructured point evaluation

The workhorse underlying cross-geometry interpolation is
`evaluate_unstructured`, which evaluates a `SpringsteelGrid` at a set of
arbitrary points in its native coordinate system:

```@docs
evaluate_unstructured
```

Dispatches by basis: 1D B-spline (`R`), 2D B-spline × B-spline (`RR`),
2D B-spline × Chebyshev (`RZ`), 3D B-spline × B-spline × B-spline
(`RRR`), batched-Fourier `RL` / `SL` (2D cylindrical / spherical), and
3D batched Fourier + Chebyshev `RLZ` / `SLZ`. The input points are a
matrix with one row per point and columns matching the grid's
coordinate order:

```julia
# Evaluate an RL grid at 1000 random polar points
npts = 1000
pts  = hcat(rand(npts) .* 50.0,          # r
            rand(npts) .* 2π)             # λ
vals = evaluate_unstructured(rl_grid, pts; vars=["u"])
```

Points that fall outside the source grid's bounds are filtered before
evaluation (B-spline evaluation errors on out-of-domain inputs), and
the returned values at those rows follow the `out_of_bounds` policy
(`:nan` by default).

Per-call caching of the γ-folded `ahat` stripes keeps repeated
evaluations against the same spectral state allocation-free — `RL`
interpolation allocations dropped from 25 MB to 494 kB between v0.2 and
v1.0, and `RLZ` from 27 MB to 75 kB, via this cache. You don't need to
manage it; it keys on `(objectid, variable)` and invalidates on the
spectral column hash changing.

## Coordinate transforms

```@docs
cartesian_to_cylindrical
cylindrical_to_cartesian
cartesian_to_cylindrical_3d
cylindrical_to_cartesian_3d
cartesian_to_spherical
spherical_to_cartesian
cylindrical_to_spherical
spherical_to_cylindrical
latlon_to_spherical
spherical_to_latlon
```

These helpers are used internally by the default cross-geometry
mappings, but you can call them directly when building a custom
`coordinate_map`. Note the spherical convention: the returned tuple is
`(θ, λ, r)` where `θ` is colatitude (0 at the north pole) and `λ` is
azimuth — this matches Springsteel's `SLZ` grid layout. The
`latlon_to_spherical(lon_deg, lat_deg)` helper adapts the more common
lon/lat geographic order.

## Setting boundary values

Some workflows need to pin boundary values (e.g., interpolating a
coarse field into the border of a fine domain without touching the
interior). `set_boundary_values!` is the per-geometry helper for that.

```@docs
set_boundary_values!
```

## See also

- [SpringsteelGrid](springsteel_grid.md) — grid construction + geometry
  type parameters
- [Grid Relocation](relocation.md) — closely related: RL/RLZ grid shift
  via the same unstructured evaluation path
- [Tutorial](tutorial.md) — worked examples using imported data

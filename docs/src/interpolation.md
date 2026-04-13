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
`SpringsteelGrid` with the right dimensions, fills the physical array,
and runs `spectralTransform!` so the result is ready for transforms,
derivatives, or further interpolation.

```@docs
grid_from_regular_data
```

The signature is overloaded for 1D, 2D, and 3D inputs. Coordinate
vectors must be **uniformly spaced** (the factory checks this), and
their length must be divisible by `mubar` (default 3) so the input
aligns to the internal Gauss–Legendre quadrature nodes. Domain bounds
are inferred as `[x[1] - h/2, x[end] + h/2]` — regular-midpoint
convention, which matches the layout of typical NetCDF and raw-array
data sources.

```julia
using Springsteel

x = collect(0.0:0.05:1.0)        # 21 points, h = 0.05
data = @. exp(-(x - 0.5)^2 / 0.01)

grid = grid_from_regular_data(x, data;
    mubar = 1,                   # regular (non-Gauss) quadrature
    BCL   = DirichletBC(),
    BCR   = DirichletBC(),
    vars  = Dict("u" => 1),
)

# grid is now a SpringsteelGrid that you can pass to downstream code
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
pass a tuple of dimension names in i/j/k order and a list of variable
names (or `nothing` to import every variable). The factory handles
dimension permutation so Springsteel's i-outer / k-inner layout is
respected regardless of how the file was written.

```julia
grid = grid_from_netcdf("rainfall.nc";
    dim_names = ("lon", "lat"),
    var_names = ["precip_mm"],
    BCL = NaturalBC(), BCR = NaturalBC(),
    BCD = NaturalBC(), BCU = NaturalBC(),
)
```

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

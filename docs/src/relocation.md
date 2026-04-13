```@meta
CurrentModule = Springsteel
```

# Grid Relocation

Grid relocation shifts the origin of a cylindrical grid (`RL` or `RLZ`)
to a new Cartesian center, reprojecting all spectral data onto the new
coordinate system. The most common use case is **vortex tracking**:
keep a hurricane's eye centred on the grid origin even as the storm
translates through its environment.

Relocation is available for 2D cylindrical (`RL`) and 3D cylindrical
(`RLZ`) grids. Cartesian and spherical grids are not currently
supported. A companion notebook at
`notebooks/Springsteel_relocation_tutorial.ipynb` walks through the
full API with plots.

## The idea

For a cylindrical grid with center `c_old = (x₀, y₀)` in Cartesian
space, a point at radius `r` and azimuth `λ` maps to Cartesian
coordinates `(x₀ + r cos λ, y₀ + r sin λ)`. To relocate the grid to
`c_new = c_old + (Δx, Δy)`, each new-grid point `(r', λ')` must pull
its value from the source grid at the corresponding point in the
source's coordinate frame:

```
x_src = r' cos λ' + Δx
y_src = r' sin λ' + Δy
r_src = √(x_src² + y_src²)
λ_src = atan2(y_src, x_src)
```

The source grid is then evaluated at `(r_src, λ_src)` (and optionally
`z`) via [`evaluate_unstructured`](@ref). This is a natural fit for the
spectral representation — no interpolation error beyond the source's
native truncation.

## Basic usage

```@docs
relocate_grid
relocate_grid!
```

```julia
using Springsteel

# Build a cylindrical grid centred at the origin
gp = SpringsteelGridParameters(
    geometry = "RL",
    iMin = 0.0, iMax = 100.0, num_cells = 30,
    vars = Dict("u" => 1),
    BCL = Dict("u" => NaturalBC()),
    BCR = Dict("u" => NaturalBC()),
)
grid = createGrid(gp)
# ... fill grid.physical with a vortex ...
spectralTransform!(grid)

# Non-mutating: get a fresh grid shifted by (5, 3) km
new_grid = relocate_grid(grid, (5.0, 3.0))

# In-place: reuse the same grid, cumulative center tracked automatically
relocate_grid!(grid, (5.0, 3.0))
grid_center(grid)   # → (5.0, 3.0)
relocate_grid!(grid, (2.0, 0.0))
grid_center(grid)   # → (7.0, 3.0)
```

The mutating `relocate_grid!` is the one you want inside a
time-stepping loop. It reuses the grid's storage (snapshotting source
data before evaluation so the operation is safe) and tracks the
cumulative center shift in a per-grid registry.

```@docs
grid_center
```

`grid_center` returns `(0.0, 0.0)` for grids that have never been
relocated. It reflects the cumulative displacement from the grid's
original construction origin, not the absolute Cartesian position.

## Boundary strategies

Some new-grid points will map to source radii beyond `iMax` — they
lie outside the source's valid domain. Four strategies control how
those out-of-bounds points are filled:

| Strategy            | Behaviour                                              |
|:--------------------|:-------------------------------------------------------|
| `:nan`              | Fill OOB points with `NaN`                             |
| `:nearest`          | Clamp source radius to `iMax`, evaluate there          |
| `:azimuthal_mean`   | **(default)** Use only the `k=0` (azimuthal-mean) coefficient of the source |
| `:bc_respecting`    | Extrapolate using the grid's radial BCs (naive path)   |

`:azimuthal_mean` is the best default for atmospheric vortex tracking:
it preserves the background environment (slowly-varying in azimuth) at
the outer edge, avoids ringing artefacts, and has no NaN propagation
problems downstream. `:nan` is useful when you want to detect OOB
regions explicitly. `:nearest` gives a physically-plausible clamp for
smooth fields; `:bc_respecting` is the most expensive and least
commonly needed — it goes through the naive per-point evaluation path
instead of the fast per-radius path.

## Taper zone

For smooth transitions between the full-evaluation interior and the
OOB fill at the outer edge, pass `taper_width > 0`. The last
`taper_width` radial rings cosine-blend between the evaluated value and
the azimuthal-mean fill, eliminating the discontinuity at the OOB
boundary:

```julia
relocate_grid!(grid, (5.0, 3.0);
    boundary    = :azimuthal_mean,
    taper_width = 3,
)
```

Zero (default) applies the OOB strategy sharply at `iMax`. A value of
2–4 cells is usually enough for smooth fields.

## Per-radius fast path

`relocate_grid` / `relocate_grid!` dispatch on a **per-radius fast
path** that batches B-spline evaluations across each ring. For a 30×30
RL grid this is roughly 14× faster than the naive `evaluate_unstructured`
dispatch; for RLZ the gap is wider because the Chebyshev column
evaluation is reused across all points of a ring. Typical
time-stepping relocations on 30 × 120 RL grids complete in under a
millisecond warm.

The fast path runs for every boundary strategy except `:bc_respecting`,
which requires the naive per-point path to handle BC-based
extrapolation correctly.

## Multi-patch grid relocation

For a `SpringsteelMultiGrid` — a nest of `RL` / `RLZ` patches — call
`relocate_grid!` on the container. The entire nest shifts as a unit,
and the outer patch provides OOB lookup data to the inner patches when
their remapped points cross what was previously the inner → outer
boundary:

```julia
mg = createMultiGrid(Dict(
    :topology   => :embedded,
    :geometry   => "RL",
    :domains    => [(0.0, 100.0), (0.0, 50.0)],
    :cells      => [10, 20],
    :vars       => Dict("u" => 1),
    :BCL        => Dict("u" => NaturalBC()),
    :BCR        => Dict("u" => NaturalBC()),
))
# ... fill and spectralTransform!(mg) ...

relocate_grid!(mg, (2.5, 0.0); boundary=:azimuthal_mean)
grid_center(mg)   # delegates to the innermost patch
```

When the multigrid is embedded in a still-larger outer environment
grid, pass that outer grid in the configuration and the relocation
will fall back to it for points outside every patch in the nest. This
is set up by [`createMultiGrid`](@ref) via the `embedded_in` kwarg; see
that function's docs for the full config.

### Snap quantization

For embedded multigrids, the shift `(Δx, Δy)` is snapped to the
nearest multiple of the outer grid's radial node spacing so that the
inner patch's half-gridpoint offset constraint is preserved across
relocations. This is automatic — you don't need to compute the snap
quantum yourself, but be aware that a requested `0.3 km` shift on a
`1 km`-resolution outer grid may become `0.5 km` or `0 km` depending on
the snap. Inspect with `grid_center(mg)` after the call.

## See also

- [`evaluate_unstructured`](@ref) — the low-level point evaluation
  path under the hood
- [Multi-Patch Grids](multipatch.md) — multigrid topology and
  construction
- [Interpolation](interpolation.md) — related cross-grid tools
- `notebooks/Springsteel_relocation_tutorial.ipynb` — worked examples
  with plots

```@meta
CurrentModule = Springsteel
```

# Multi-Patch Grids

Springsteel supports grids composed of multiple B-spline patches linked
by exact basis conversion coupling at their interfaces. A multi-patch grid lets you

- **chain** patches at different resolutions along a direction — useful
  for hurricanes, where a fine-resolution inner mesh handles the eyewall
  and a coarser outer mesh handles the environment;
- **embed** a fine patch inside a coarser outer patch — useful for
  nesting a limited-area high-resolution domain inside a larger global
  or environmental grid.

All interface data transfer happens through the existing spectral
transforms — no interpolation, no re-projection, no loss.

## How coupling works

Each patch is a normal [`SpringsteelGrid`](@ref). What makes a set of
patches into a multi-patch grid is a list of [`PatchInterface`](@ref)
records, each describing one connection:

- a **primary** patch with `NaturalBC` on the shared side — it runs
  freely, like any standalone grid;
- a **secondary** patch with `FixedBC` (R3X) on the shared side — it
  receives spectral boundary values from the primary at each transform
  step;
- a **coupling matrix** that converts three consecutive primary B-spline
  coefficients into three consecutive secondary border coefficients.

The coupling matrix depends only on the cell-width ratio between the two
patches. Springsteel currently supports two ratios:

- **1:1** — same cell width; the border coefficients are copied directly.
  See [`COUPLING_MATRIX_1X`](@ref).
- **2:1** — the fine patch has half the coarse cell width. The exact 3×3
  matrix is derived from cubic B-spline basis evaluation at the staggered
  node positions. See [`COUPLING_MATRIX_2X`](@ref).

Any other ratio throws at construction time. The 2:1 case is exact for
cubic B-splines — there is no interpolation error introduced at the
interface, only the inherent basis truncation of each patch.

```@docs
PatchInterface
MultiPatchGrid
COUPLING_MATRIX_1X
COUPLING_MATRIX_2X
```

## Transform ordering

For a forward transform on a multi-patch grid, the order is:

1. `spectralTransform!` each patch independently.
2. Done — spectral coefficients are now consistent across interfaces
   provided the input physical data was itself interface-consistent.

For an inverse transform, the order matters:

1. `gridTransform!` the primary side of each interface first (the side
   with `NaturalBC`).
2. Call [`update_interface!`](@ref) on each interface to copy the
   primary's border coefficients into the secondary's `ahat` vector.
3. `gridTransform!` the secondary side — it now reconstructs physical
   values using the just-copied interface data through the R3X
   (inhomogeneous BC) code path.

[`multiGridTransform!`](@ref) handles this ordering automatically and is
the entry point most user code should call.

```@docs
update_interface!
multiGridTransform!
```

Both `update_interface!` and `multiGridTransform!` on the parameterized
`MultiPatchGrid` container are zero-allocation at steady state — the
`PatchInterface{P,S}` parameters bind the concrete grid types so the
dispatch path is fully specialised.

## Building a multi-patch grid

The high-level factory is [`createMultiGrid`](@ref), which takes a
configuration dict that describes topology, geometry, patch boundaries,
and per-variable BCs. It auto-computes interface BCs (`NaturalBC` on
primary sides, `FixedBC` on secondary sides), derives `patchOffsetL` for
cylindrical/spherical geometries, and validates cell-width ratios.

```@docs
createMultiGrid
SpringsteelMultiGrid
```

### Chain topology

A chain links N patches along the radial / i direction. You pass N+1
boundary coordinates; adjacent patches share an interface at each
interior boundary.

```julia
using Springsteel

mg = createMultiGrid(Dict(
    :topology   => :chain,
    :geometry   => "RL",
    :boundaries => [0.0, 50.0, 100.0],   # 2 patches: [0, 50] and [50, 100]
    :cells      => [20, 10],              # 2:1 ratio (inner fine, outer coarse)
    :vars       => Dict("u" => 1),
    :BCL        => Dict("u" => NaturalBC()),
    :BCR        => Dict("u" => NaturalBC()),
))
```

The inner patch (cells 20, width 2.5) is the secondary; the outer patch
(cells 10, width 5.0) is the primary. Physical data flows outer → inner
at the interface.

For equal-resolution chains, pass a scalar `:cells`:

```julia
mg = createMultiGrid(Dict(
    :topology   => :chain,
    :geometry   => "R",
    :boundaries => [0.0, 25.0, 50.0, 75.0],   # 3 patches
    :cells      => 10,
    :vars       => Dict("u" => 1),
    :BCL        => Dict("u" => DirichletBC()),
    :BCR        => Dict("u" => DirichletBC()),
))
```

### Embedded topology

Embedded stacks fine patches *inside* a coarser outer patch. You pass
domain bounds for each nested level, outermost first. The outer patch
runs freely over the entire domain; the inner patches overwrite it
inside their subdomains.

```julia
mg = createMultiGrid(Dict(
    :topology   => :embedded,
    :geometry   => "RL",
    :domains    => [(0.0, 100.0),     # outer
                    (0.0,  50.0)],    # inner (2:1 refinement)
    :cells      => [10, 20],
    :vars       => Dict("u" => 1),
    :BCL        => Dict("u" => NaturalBC()),
    :BCR        => Dict("u" => NaturalBC()),
))
```

The outer patch provides background data everywhere; the inner patch
receives R3X interface values at its outer edge from the outer patch and
runs independently inside.

### Cell-width / refinement constraint

The fine patch must be at the **half-gridpoint** of the coarse patch
for the 2:1 exact BC to work. `createMultiGrid` enforces this by
computing `patchOffsetL` automatically for cylindrical and spherical
geometries, but when you're hand-building patches via
[`PatchChain`](@ref) / [`PatchEmbedded`](@ref) you are responsible for
making sure adjacent patches are correctly aligned. The factory will
throw if `DX_primary / DX_secondary` isn't 1.0 or 2.0 within tolerance.

## Low-level factories

If you need more control than `createMultiGrid` offers — for example
mixing refinement ratios on different interfaces, or wiring up interfaces
by hand on pre-built patches — use these direct factories:

```@docs
PatchChain
PatchEmbedded
```

Both take a vector of pre-constructed `SpringsteelGrid`s and build the
interface list, validating DX ratios and auto-computing coupling
matrices.

## Relocating a multi-patch grid

The entire nest can be relocated as a unit with
[`relocate_grid!`](@ref):

```julia
relocate_grid!(mg, (Δx, Δy); boundary=:azimuthal_mean)
```

For embedded multigrids the outer patch provides OOB lookup data to the
inner patches when they cross the old → new domain boundary. See
[Grid Relocation](relocation.md) for the full relocation story.

## See also

- [Boundary Conditions](boundary_conditions.md) — `NaturalBC` / `FixedBC`
  are what drive the primary / secondary roles
- [SpringsteelGrid](springsteel_grid.md) — single-patch grid type
- [Grid Relocation](relocation.md) — relocating a multi-patch nest
- [Developer Notes / Contributing](contributing.md) — TRAP-1..4 cover
  multi-patch spectral layout invariants

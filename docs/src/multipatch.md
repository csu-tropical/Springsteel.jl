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

### Supported geometries

`createMultiGrid` decomposes along the **i** direction only; any j/k axes are shared
and are copied unchanged into every patch.

| Geometry family | Notes |
|:---|:---|
| `R`, `RL`, `RLZ`, `SL`, `SLZ`, `RZ` | Cylindrical / spherical / Chebyshev-vertical, `patchOffsetL` auto-derived where applicable |
| `RR`, `RRR`, `RiRk` | Cartesian spline j/k axes must be given **explicitly** — a spline axis has no sensible default domain, so `_validate_multigrid_config` requires it. `BCU`/`BCD`/`BCB`/`BCT` default to natural |
| `RLR`, `SLR` | Spline vertical in the k slot; included in the cylindrical offset path |

2-D and 3-D *patch* decomposition (splitting j or k as well as i) is not yet supported.

!!! note "Spline j/k axes on `RR`/`RRR`"
    Before v1.1, `createMultiGrid` advertised `RR`/`RRR` but never plumbed their spline
    j/k axes, so a patch silently inherited the `@kwdef` default `jMax = 2π` with
    periodic Fourier BCs on what is actually a cubic B-spline axis. Supplying the axes
    is now mandatory and validated.

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

## Two-way nesting

The interface machinery above is one-way: the primary patch donates its border trio and
the secondary patch reads it. A two-way nest adds a **subcycling child** (which advances
several short steps per parent step) and **fine → coarse feedback** (the parent's
tendencies see the child's solution). Both follow DeMaria et al. (1992) and Ooyama (2001).

### Temporal interpolation for a subcycling child

A child taking `n` substeps per parent step needs a boundary condition at each substep,
but the parent only supplies one at each of its own steps. [`lerp_payload!`](@ref)
linearly interpolates two [`InterfacePayload`](@ref)s in time and is bitwise-exact at the
endpoints, so substep 0 and substep `n` reproduce the parent's donated trio exactly:

```julia
# Parent bracketing payloads p0 (at t) and p1 (at t + Δt_parent)
for j in 0:n-1
    θ = j / n
    lerp_payload!(payload, p0, p1, θ)
    apply_interface_payload!(meta, child, payload)   # meta::PatchInterfaceMetadata
    step!(child, Δt_child)
end
```

`lerp_payload!` validates that both endpoints match `dest` in scheme, side, and variable
count, and `apply_interface_payload!` re-validates the payload against the interface
metadata, so a mismatched pairing throws rather than writing the wrong border.

### Collar interfaces

Feedback must not be routed through the child's boundary condition. A rank-3 spline's
border trio *is* its `ahat`, so writing the child's own solution back into the parent's
border and re-donating it is degenerate — the interface freezes. The `test/nesting_support.jl`
suite carries an explicit anti-freeze regression against this.

The correct construction is a **collar**: extend the parent patch one cell past the
nominal junction, into the child's domain, so that the trio the child's R3X boundary reads
consists of *interior*, freely-fitted parent amplitudes. This is the existing
`PatchInterface(...; is_stacked=true)` path — no new interface type is needed.

### Fine → coarse feedback

Feedback then goes through the **tendencies**, not through a boundary condition
(DeMaria et al. 1992, eq. 2.22): the coarse patch's collar quadrature points are evaluated
on the fine grid, and the results are injected into the coarse Galerkin loads. The
per-geometry evaluators for that injection are:

| Function | Grids | Returns |
|:---|:---|:---|
| [`evaluate_grid_ipoints`](@ref) (and `!`) | `R`, `RiRk` | Values at arbitrary i-direction points, in `grid.physical`'s variable/derivative-slice layout |
| [`evaluate_grid_points`](@ref) | `RL` | Value plus ∂r, ∂²r, ∂λ, ∂²λ at arbitrary `(r, λ)` |
| [`evaluate_grid_points`](@ref) | `RLR` | Whole vertical columns at arbitrary `(r, λ)` and arbitrary `z`, 7 slices, z-fastest per column |

The `RL` and `RLR` forms take an optional per-point `kmax` wavenumber truncation. This is
the azimuthal analogue of ring transmissibility: an injected value must not carry
wavenumbers the target ring cannot support. All three are registry-aware, so a nested
patch's R3X borders are honoured during evaluation.

### Nest-annulus grids

An `RL` patch may set `patchOffsetL` explicitly (leaving `spectralIndexL` patch-relative)
so that its ring point counts and wavenumber support follow the **global** ring numbering
rather than its own. `_create_tile_from_patch` composes the patch's `patchOffsetL` into
each of its tiles; for an ordinary patch this composition is the identity.

!!! warning "Tiled transforms and the wavenumber registry"
    On the tiled (distributed) path, the per-wavenumber coupled-border coefficients that
    `apply_interface_payload!` registers must be reloaded for every wavenumber block. A
    transform that reuses one spline object across wavenumbers without reloading its
    `ahat` will silently carry the last-applied wavenumber's border on every mode. This
    was fixed for `RL` and `RLR` in v1.1; it is the failure mode to suspect first if a
    nested run diverges from its single-grid control while the single-tile case matches.

```@docs
lerp_payload!
InterfacePayload
PatchInterfaceMetadata
compute_interface_payload
compute_interface_payload!
apply_interface_payload!
evaluate_grid_ipoints
evaluate_grid_points
```

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

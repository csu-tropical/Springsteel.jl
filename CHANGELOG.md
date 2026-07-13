# Changelog

All notable changes to Springsteel.jl are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`Thermodynamics.potential_temperature(p_Pa, rho_d)`** — a new two-argument method giving the
  dry potential temperature from the `(p, ρ_d)` pair, `θ_d ≡ (p₀^κ/R_d)·p^(1−κ)/ρ_d`. It is the
  same quantity as the existing `potential_temperature(s, rho_d, q_v)`, but needs no entropy
  inversion, so total-energy equation sets that carry pressure prognostically can diagnose θ
  without a spectral transform. Purely additive: a new signature on an existing generic, with no
  change to the three-argument method. Note `p_Pa` is in **pascals** (matching
  `PressureReferenceState.pbar`), not the hPa used by the saturation functions.
- **Explicit cell counts for every cubic B-spline direction.**
  `SpringsteelGridParameters` gains `num_cells_i` (an alias for `num_cells`),
  `num_cells_j`, and `num_cells_k`. A spline axis is defined by its cell count
  (`Dim = cells * mubar`, `bDim = cells + 3`), so this lets the j- and k-directions of
  `RR`, `RRR`, `RiRk`, `RLR`, and `SLR` be specified exactly, the way `num_cells` already
  specified the i-direction. The resolved counts are stored on the returned parameters.
- **`iDim`-only "reverse mode" now works** for spline-i geometries: supplying `iDim`
  without a cell count back-derives `num_cells = iDim ÷ mubar` and rebuilds `b_iDim`,
  `spectralIndexR`, and `patchOffsetR`. Previously this produced a malformed grid that
  failed with a `PosDefException` inside a factorization.
- Documented the previously implicit sizing rules: the spline `Dim`/`bDim` formulas, and
  the j/k auto-default (uniform nodal spacing with the i-direction, rounded up).

### Changed

- **Six of the `Dict` fields on `SpringsteelGridParameters` are now concretely typed.**
  `vars` becomes `Dict{String,Int64}`, `l_q` becomes `Dict{String,Float64}`,
  `max_wavenumber` becomes `Dict{String,Int64}`, `fourier_filter` and `chebyshev_filter`
  become `Dict{String,AbstractFilter}`, and `spline_filter` becomes
  `Dict{String,Dict{Symbol,AbstractFilter}}`. They were previously declared as a bare
  `Dict`, i.e. `Dict{Any,Any}` at the type level.

  The one that matters is `vars`. Because it was abstract, `vars["p"]` returned `Any`, so
  every slot index a downstream model pulled out of it was boxed and every array access
  built from that index was type-unstable. The declaration is the only thing that was ever
  abstract — at runtime the dict was already `Dict{String,Int64}` on every path — so
  narrowing it de-boxes those lookups without changing a single value.

  **This is not a breaking change.** Every construction site already infers the narrowed
  type, and anything wider (`Dict{String,Any}`, an empty `Dict()`, an `Int` where a
  `Float64` is wanted) is coerced by `convert` at construction exactly as before. Archives
  written before this change load unchanged: JLD2 narrows each field as it rebuilds the
  struct. A `test/fixtures/widened_dicts_grid.jld2` fixture — written while the fields were
  still untyped, and deliberately holding `Dict{String,Any}` values — pins that behaviour,
  and `test/fixtures/make_fixtures.jl` now documents how the fixtures are produced.

  The six boundary-condition fields (`BCL`, `BCR`, `BCU`, `BCD`, `BCB`, `BCT`) are
  **deliberately left as a bare `Dict`**. They must accept both the legacy Dict-valued BC
  constants (`CubicBSpline.R0` is a `Dict{String,Int64}`, `R1T0` a `Dict{String,Float64}`)
  and the newer `BoundaryConditions` struct; those have no common supertype but `Any`, so
  no concrete value type can hold them. They are read at grid-construction time only and so
  carry none of the performance payoff. See `agent_files/project_bc_type_unification.md`.
- `_validate_spline_filter` no longer hand-checks the *shape* of a `spline_filter` — that a
  key is a `String`, that the inner value is a `Dict`, that the leaf is an `AbstractFilter`.
  The field's declared type now enforces all three, so malformed input is rejected at
  construction as a `MethodError`/`InexactError` rather than reaching the validator and
  raising an `ArgumentError`. The *semantic* checks are unchanged and still raise
  `ArgumentError`: unknown variable name, invalid direction, a `SpectralFilter` on a spline
  direction, or a direction that is not a spline in the given geometry.
- A spline `iDim`/`jDim`/`kDim` that is not a multiple of `mubar` now raises an
  `ArgumentError` naming the field, `mubar`, and the nearest valid cell counts. It
  previously raised `InexactError: Int64(16.666666666666668)`, naming nothing. Chebyshev
  and Fourier axes are unaffected.
- Supplying contradictory cell/gridpoint counts for the same axis (e.g. `num_cells_j=7`
  with `jDim=20`) is now an `ArgumentError` rather than being silently resolved.
- The j/k auto-default emits a warning when the domain length is not an integer multiple
  of the i-cell width, since the rounding then leaves `DX_j != DX_i`. Commensurate
  domains — the overwhelmingly common case — stay silent.
- `save_grid` stamps `format_version = "1.1"` into the archive. Archives written before the
  new parameter fields existed are still upgraded on load. Note that this upgrade is driven
  by *type dispatch*, not by the version tag: `save_grid` serialises the whole `params`
  struct, so when a field is missing from an archive JLD2 hands back a reconstructed type,
  and `_upgrade_params` rebuilds it through the keyword constructor, letting the new fields
  take their defaults. `load_grid` does not read `format_version` at all — it is written for
  forensics and for future use, and nothing currently branches on it.

### Fixed

- **`grid_from_regular_data` (and `grid_from_netcdf`, which forwards to it) never accepted
  `BoundaryConditions` structs.** Its BC keyword arguments were annotated `::Dict`, so a bare
  `DirichletBC()` — which is not a `Dict` — raised a `MethodError`. The struct-BC API simply
  never reached these entry points, and the calls documented in `docs/src/tutorial.md` and
  `docs/src/interpolation.md` did not run. The BC arguments now take the new `BCSpec` union
  (`Union{Dict, BoundaryConditions}`), and a bare struct BC is broadcast across all variables
  exactly as a bare legacy constant already was. Both spellings work on every axis of the 1D,
  2D, and 3D methods; legacy `Dict` callers are unaffected.
- `grid_from_regular_data` now accepts single-variable `data` as a plain vector, reshaping it
  to `(total_points, 1)`. Only an `AbstractMatrix` method existed, so the vector form shown in
  the documentation raised a `MethodError` regardless of which BC spelling was used.
- Corrected the `grid_from_regular_data` examples in `docs/src/tutorial.md` and
  `docs/src/interpolation.md`, which additionally passed `vars` as a `Dict("u" => 1)` where the
  signature takes a `Vector{String}`. All three documented examples now execute verbatim.
- `_create_tile_from_patch` did not propagate a patch's spline j/k cell counts, so a tile
  re-derived them from its own narrower i-domain. This was latent (the auto-default is
  tile-invariant) but would have produced mismatched coefficient-array shapes across
  tiles as soon as a patch set `num_cells_j` explicitly.

## [1.1.0] - 2026-06-27

`v1.0.0 → v1.1.0` is a MINOR bump: every change is a backward-compatible addition
(new geometry, two new submodules, new transform/tiling methods, performance) or a
fix. No public API was removed or changed in a breaking way.

### Added

- **First-class grid transforms for six previously solver-only geometries** —
  `Z`, `ZZ`, `ZZZ` (pure-Chebyshev Cartesian) and `L`, `LL`, `LLZ` (Fourier, and
  Fourier × Fourier × Chebyshev). Each gains the full transform surface
  (`getGridpoints`, `spectralTransform`/`spectralTransform!`,
  `gridTransform`/`gridTransform!`, `getRegularGridpoints`, and both
  `regularGridTransform` overloads). These grids now round-trip and emit
  regular-grid output like the spline families; previously they raised
  `MethodError` for every transform entry point and only worked through the solver
  assembly path.
- **RiRk geometry** — a 2-D Cartesian grid with a cubic-B-spline vertical
  (`Spline-i × Spline-k`; exports `RiRk_Grid`, and `RiRj_Grid` as an alias of
  `RR`). Full support: forward/inverse transforms, tiling, regular-grid output
  (`getRegularGridpoints`/`regularGridTransform`), NetCDF/CSV IO, and inhomogeneous
  R3X boundary conditions (`set_boundary_values!`). Transforms are zero-allocation.
- **`Thermodynamics` submodule** — basis-agnostic atmospheric thermodynamics
  (physical equation of state + diagnostics), shared by Springsteel-grid clients;
  imported explicitly, e.g. `using Springsteel.Thermodynamics`.
- **Hydrostatic reference-state module** — physical-density reference (base) state
  types and builders: `AbstractReferenceState`, `DryReferenceState`,
  `MoistReferenceState`, `CondensateReferenceState`, with hydrostatic builders and
  a physical-format exact reference builder (including condensate).
- **RZ tiling transforms** — tiled transform support for the `RZ`
  (Spline × Chebyshev) geometry.
- **Geometry-aware regular-output sizing** — regular-grid output dimensions are now
  derived per geometry and preserve explicitly-provided values.
- **Chebyshev solver parity test** — the unified solver is checked against the
  legacy `Chebyshev.bvp` reference; documentation grid-type tables and the v1.1
  roadmap were refreshed (RiRk documented; "available but untested" grid claims
  corrected).
- **New test groups:** `thermodynamics`, `reference_state` (added to the suite and
  CI).

### Performance

- **Batched Cartesian unstructured evaluation** — `evaluate_unstructured` on the
  `RR`, `RZ`, `RiRk`, and `RRR` spline grids now hoists the per-stripe
  `SAtransform!` out of the per-output-point loop and batch-evaluates into a
  per-call buffer. Allocation count is now constant in the number of query points
  (was ~2 per point), removing a multi-threaded GC-pressure hazard. Roughly
  **17–21× faster** for `RR`/`RZ` and **~12×** for `RRR` at 1000 query points.
  Results are unchanged.

### Fixed

- Reference-state builder: copy the `Itransform!` result and recompute saturation
  pressure (correctness fix in the new reference-state path).
- RZ tiling: restored physical-column `num_columns` semantics.
- Test-suite hygiene: an operator-algebra testset titled "ZZ grid" actually
  constructed a `ZZZ` grid — title corrected.

### Maintenance

- CI: added the `reference_state` and `thermodynamics` test groups; TagBot now
  receives the token and SSH key.
- Removed stale top-level dev scripts (`test_ci.sh`, `test_regular_grid.jl`,
  `.actrc`).

## [1.0.0] - 2025

Initial stable release: the unified `SpringsteelGrid{G, I, J, K}` type system over
mixed CubicBSpline / Fourier / Chebyshev bases, the solver framework, grid-to-grid
interpolation, spectral filtering, multi-patch grid connections, basis-template
caching, and grid relocation. See the Git tag history for releases prior to 1.1.0.

[1.1.0]: https://github.com/csu-tropical/Springsteel.jl/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/csu-tropical/Springsteel.jl/releases/tag/v1.0.0

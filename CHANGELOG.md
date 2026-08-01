# Changelog

All notable changes to Springsteel.jl are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **`[compat] julia` corrected from `"1.9"` to `"1.10"` — it was never satisfiable.**
  `Krylov = "0.10.6"` requires `julia >= 1.10`, so `Pkg.add("Springsteel")` on Julia 1.9
  failed with `Unsatisfiable requirements detected for package Krylov`. The declared
  floor has been wrong for the whole v1.x line (v1.0.0 carries the same
  `Krylov = "0.10.6 - 0.10"` bound), so this drops no support that ever worked — it makes
  the manifest honest about what already resolved. Found by General's AutoMerge, which
  resolves on the lowest compatible version; nothing in this repo was testing the floor.

### Maintenance

- **CI now exercises the compat floor.** A `minimum-julia` job resolves, loads, and runs
  the full suite on the lowest supported Julia. The rest of the matrix runs on `'1'`
  (latest stable), so the declared lower bound was previously a claim nothing checked —
  which is exactly how the unsatisfiable `julia = "1.9"` above survived two releases.
  Because `Manifest.toml` is gitignored, `Pkg.instantiate()` resolves from scratch in that
  job, so the resolve itself is the test. It is one job rather than a second pass over all
  23 groups, and is deliberately not gated on `parity-check` so a group-name typo cannot
  hide a dependency-resolution regression.
- **Three test-suite portability problems, found by that new job on its first run.** The
  package itself is fine on Julia 1.10; the suite was not:
  - `test/operator_algebra.jl` did `const OperatorTerm = Springsteel.OperatorTerm`, but
    `OperatorTerm` is exported, so `using Springsteel` already bound it. Rebinding an
    imported name is an error on 1.10 (`cannot assign a value to imported variable`). The
    redundant `const` is removed; `_lower`, which really is unexported, stays.
  - `test/grids.jl` used `Base.infer_return_type`, which is Julia 1.11+. It now falls back
    to `Base.return_types` on older versions.
  - `test/basis.jl` asserted `(@allocations Chebyshev.Ixtransform(col, buf)) == 0`. Julia
    1.11+ elides that last allocation but 1.10 does not, so the strict zero is now pinned
    only where it holds, with a `<= 1` bound on 1.10 that still catches a regression into
    per-element allocation. Note this qualifies the v1.1.0 claim that the generic in-place
    `Ix`/`Ixx` forms "allocate nothing once warm" — true on 1.11+, one allocation on 1.10.

## [1.1.0] - 2026-08-01

`v1.0.0 → v1.1.0` is a MINOR bump: every change is a backward-compatible addition
(new geometries, two new submodules, new boundary-condition and constrained-fit
families, new transform/tiling/nesting methods, performance) or a fix. No public API
was removed or changed in a breaking way.

Two fixes **change results** for code that was already exercising the affected paths,
and are called out inline below: unstructured evaluation of azimuthally asymmetric
`RL`/`RLZ`/`RLR` fields (a sine-sign error), and `num_columns(::RRR_Grid)` (which
returned a spectral block count rather than physical columns).

### Added

#### Geometries, sizing, and grid transforms

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
- **Geometry-aware regular-output sizing** — regular-grid output dimensions are now
  derived per geometry and preserve explicitly-provided values.

#### Boundary conditions and constrained fits

- **`CubicBSpline.R1T1X` — rank-1 inhomogeneous Neumann carried in the affine `ahat`
  offset.** A rigid wall in a compressible atmosphere needs a state-dependent pressure
  condition (`w` Dirichlet at the wall ⟹ `dp′/dz = −g·ρ_t′` there exactly) that no
  homogeneous BC supplies, but the semi-implicit acoustic solve eliminates `w` and so
  needs the `R1T1` subspace to stay operator-consistent. `R1T1X` uses the **same**
  `gammaBC` as `R1T1` — hence the same admissible subspace and solver stability — with
  the boundary derivative carried in `ahat`. `du = 0` reproduces the homogeneous fit
  bitwise, so nothing that does not opt in can change. Adds `set_ahat_neumann!`,
  `set_wall_derivatives!`, and `_has_ahat` (formerly `_has_r3x`, now covering both
  inhomogeneous families). `SplineBasisArray` gains `wall_du`, indexed
  `[column, variable, side, dr+1]`: a k-basis shares one spline across every column, so
  the per-column wall data cannot live on the spline and is installed inside the
  transform's column loop (both the `RiRk` `gridTransform` and `tileTransform`). The `dr`
  axis is load-bearing — a 2-D transform differentiates in `i` *before* fitting in `k`,
  so the `dr = 1/2` passes need `d(wall)/di` and `d²(wall)/di²`. New `test/r1t1x.jl`,
  run under the existing `r3x` group.

- **Positivity-constrained cubic-B-spline fits for positive-definite fields.**
  `SAtransform_bounded!` imposes a per-coefficient lower bound on the SA solve, selected
  per variable and per leg through the new `SpringsteelGridParameters.positivity` field
  (e.g. `positivity = Dict("q" => Dict(:k => 0.0))`). Because the B-spline basis is
  non-negative, a componentwise bound on the coefficients bounds the *reconstruction*
  everywhere by the convex-hull property — not merely at the mish points. The clip is
  conservative: the mass a column must shed is redistributed within that column, and
  anything that cannot be placed is accumulated into `bound_shortfall` rather than
  silently created.

  Two rules govern which legs to bound. **Sufficiency** — bounding the *last* leg puts
  the field above the bound everywhere the model evaluates it, since every earlier
  direction has already been collapsed to a physical coordinate. **Feasibility** —
  bounding the *earlier* legs is what keeps the last one solvable: `Σₘ bₘ = ∫u` by
  partition of unity and `Σₘ aₘwₘ = ∫u` because the `l_q` penalty has zero third
  derivative, so a componentwise non-negative `b` entering a leg guarantees that leg's
  column mass is non-negative and hence always conservatively fixable.

  Supporting API: `basis_integrals`, `bound_shortfall`, `set_lower_bound!`,
  `clear_lower_bound!`, `set_lower_bound_from_profile!`, and a `lower` keyword on
  `Spline1D`. Constraints are rejected rather than mis-applied where the box cannot
  express positivity: `R1T0`/`R3` boundary conditions (a slaved coefficient is not in the
  box), R3X borders (the trio is pinned to the parent's donated `ahat`), Fourier and
  Chebyshev directions (no convex-hull property), and a *nonzero* `:i` bound on an
  intermediate leg (which fits inner products, not the field, so only a zero bound
  carries through unchanged).

#### Tiling, multipatch, and nesting

- **RZ tiling transforms** — tiled transform support for the `RZ`
  (Spline × Chebyshev) geometry.
- **Tiled `splineTransform!` for the `RLR` and `SLR` geometries.** The 3-argument
  `(sharedSpectral, patch, tile)` form existed for `RZ`/`RiRk`, `RL`, `SL`, `RLZ`, and
  `SLZ` but not `RLR`/`SLR`, so any distributed run on those geometries failed with a
  `MethodError` at the first spectral sync. Both use the `RLZ` per-`z_b` block layout
  (k = 0 block, then real/imag pairs per azimuthal wavenumber) over the spline vertical's
  `b_kDim` blocks.
- **`createMultiGrid` now supports i-direction multipatch for `RR`, `RRR`, and `RiRk`.**
  It previously advertised `RR`/`RRR` but never plumbed their spline j/k axes, so an `RR`
  patch silently inherited the `@kwdef` `jMax = 2π` with periodic Fourier BCs on a cubic
  B-spline j-axis. The shared spline axes are now threaded through `_create_chain` and
  `_create_embedded` (only `i` is decomposed, so j/k are read once and copied into every
  patch) and are *required* by `_validate_multigrid_config`, since a spline axis has no
  sensible default domain; `BCU`/`BCD`/`BCB`/`BCT` default to natural. 2-D/3-D patch
  decomposition remains deferred.
- **Temporal-nesting support for two-way nested models** (DeMaria et al. 1992;
  Ooyama 2001):
  - **`lerp_payload!(dest, p0, p1, θ)`** linearly interpolates two
    [`InterfacePayload`]s in time, bitwise-exact at the endpoints. A subcycling
    child patch applies the interpolated parent trio at each of its substeps
    between the parent's bracketing steps.
  - **`evaluate_grid_ipoints(grid, xq)`** (and the in-place `!` form) evaluates a
    grid's spectral representation at arbitrary i-direction points with the same
    variable/derivative-slice layout as `grid.physical` (1-D spline `R` and
    `RiRk` grids). This is the fine→coarse feedback primitive: the coarse patch's
    collar quadrature points are evaluated on the fine grid and injected into the
    coarse Galerkin loads, per DeMaria et al. (1992) eq. 2.22 — feedback through
    the tendencies, not through a boundary condition.
  - **Collar interfaces** — a parent patch extended one cell past the nominal
    junction into the child's domain, so the trio the child's R3X boundary reads
    consists of interior, freely-fitted parent amplitudes — are covered by the
    existing `PatchInterface(...; is_stacked=true)` path; the new
    `test/nesting_support.jl` suite documents the construction and adds an
    anti-freeze regression against BC-based (dual-R3X) feedback, which is
    degenerate (a rank-3 spline's border trio is identically its `ahat`).
  - **`evaluate_grid_points(grid::RL_Grid, pts; kmax)`** — values plus
    ∂r/∂²r/∂λ/∂²λ at arbitrary `(r, λ)` points for radially-nested RL models,
    with an optional per-point wavenumber truncation (the azimuthal analogue of
    ring transmissibility: injected values must not exceed the target ring's
    supported wavenumbers). Registry-aware via `_get_ahat_cache_rl`.
  - **Nest-annulus grids**: an RL patch may set `patchOffsetL` explicitly (with
    `spectralIndexL` left patch-relative) so its ring point counts and
    wavenumber support follow the global ring numbering, and
    `_create_tile_from_patch` now composes the patch's own `patchOffsetL` into
    its tiles (identity for ordinary patches).
- **`evaluate_grid_points(grid::RLR_Grid, pts, zq; kmax)`** — the fine→coarse collar
  primitive for radially-nested `RLR` models. Evaluates whole vertical columns at
  arbitrary `(r, λ)` locations and arbitrary `z`, returning the 7-slice physical layout
  z-fastest per column, with optional per-column azimuthal transmissibility truncation.
  Radial spline evaluations are deduped per distinct radius, and each column pays one
  z-solve per horizontal-derivative family. `RLR` and `SLR` are also added to
  `_CYLINDRICAL_GEOMETRIES` for the `createMultiGrid` offset path.

#### Thermodynamics and reference states

- **`Thermodynamics` submodule** — basis-agnostic atmospheric thermodynamics
  (physical equation of state + diagnostics), shared by Springsteel-grid clients;
  imported explicitly, e.g. `using Springsteel.Thermodynamics`.
- **Hydrostatic reference-state module** — physical-density reference (base) state
  types and builders: `AbstractReferenceState`, `DryReferenceState`,
  `MoistReferenceState`, `CondensateReferenceState`, with hydrostatic builders and
  a physical-format exact reference builder (including condensate).
- **`PressureReferenceState`** — a pressure-based hydrostatic reference for total-energy
  equation sets, storing `p` [Pa], partial and total densities, and derived temperature
  (EOS), total energy density `E_t = ρ_d·e_i + ρ_t·g·z` (BF02 internal energy), and
  supersaturation density `Q_ss = ρ_v − ρ_v,sat(T, p)`. Balance is the direct
  `dp/dz = −ρ_t·g`, so no entropy/log-density refinement is needed. Builders
  `exact_pressure_reference_state` (reads `z p ρ_d ρ_v ρ_c`; saturated input columns give
  `Q_ssbar ≡ 0`) and `calculate_pressure_reference_state` (sounding path, spectral
  fixed-point integration of `dp/dz = −g·ρ_t`); accessors `ref_pressure`, `ref_rho_t`,
  `ref_total_energy`, `ref_qss`; new thermodynamics helpers `rho_v_sat(Tk, phPa)` and
  `internal_energy_bf02(Tk, q_v, q_l)`.
- **`sigmabar` — the entropy-density (`ρ_d·s`) profile** — added to the dry, moist, and
  condensate reference states with a `ref_sigma` accessor, supporting entropy-density
  equation sets.
- **`Thermodynamics.potential_temperature(p_Pa, rho_d)`** — a new two-argument method giving the
  dry potential temperature from the `(p, ρ_d)` pair, `θ_d ≡ (p₀^κ/R_d)·p^(1−κ)/ρ_d`. It is the
  same quantity as the existing `potential_temperature(s, rho_d, q_v)`, but needs no entropy
  inversion, so total-energy equation sets that carry pressure prognostically can diagnose θ
  without a spectral transform. Purely additive: a new signature on an existing generic, with no
  change to the three-argument method. Note `p_Pa` is in **pascals** (matching
  `PressureReferenceState.pbar`), not the hPa used by the saturation functions.

#### Transform surface

- **Generic in-place `Ixtransform`/`Ixxtransform`.** The basis-native in-place derivative
  evaluations existed on both bases, but the cross-basis generic surface offered only the
  allocating 1-argument forms, so per-column hot paths allocated a fresh derivative vector
  on every call. Adds the Chebyshev generic in-place `Ixxtransform` wrapper and the
  two-argument top-level delegations for both bases; tests assert the in-place forms match
  the allocating ones bit-for-bit and allocate nothing once warm.

#### Tests

- **Chebyshev solver parity test** — the unified solver is checked against the
  legacy `Chebyshev.bvp` reference; documentation grid-type tables and the v1.1
  roadmap were refreshed (RiRk documented; "available but untested" grid claims
  corrected).
- **New test groups:** `thermodynamics`, `reference_state`, `cell_counts`,
  `nesting_support`, and `positivity` (all added to the suite and the CI matrix).

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
  carry none of the performance payoff.
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

### Performance

- **Batched Cartesian unstructured evaluation** — `evaluate_unstructured` on the
  `RR`, `RZ`, `RiRk`, and `RRR` spline grids now hoists the per-stripe
  `SAtransform!` out of the per-output-point loop and batch-evaluates into a
  per-call buffer. Allocation count is now constant in the number of query points
  (was ~2 per point), removing a multi-threaded GC-pressure hazard. Roughly
  **17–21× faster** for `RR`/`RZ` and **~12×** for `RRR` at 1000 query points.
  Results are unchanged.

### Fixed

#### Unstructured evaluation and coefficient caches

- **Unstructured RL/RLZ evaluation flipped every sine (odd-in-λ) component.**
  `_eval_unstructured_rl`/`_eval_unstructured_rlz` (behind `evaluate_unstructured`,
  `interpolate_to_grid`, and grid relocation) synthesized the Fourier series with
  `+ aI·sin(kλ)`, but the FFTW halfcomplex "imag" slots hold the **negative**
  sine sums — so any azimuthally-asymmetric field was evaluated as its mirror
  image. The synthesis now subtracts the sine term, matching the `HC2R` inverse
  used by `FItransform!`; a new mish-point test pins the convention against
  `gridTransform!` (the existing interpolation/relocation tests passed under
  either sign and never caught this). **This changes results** for any consumer
  that evaluated asymmetric RL/RLZ fields at unstructured points.
- **Unstructured `RLR` evaluation flipped every sine (odd-in-λ) component.**
  `_eval_unstructured_rlr` had the same defect, and was invisible to the existing
  axisymmetric test. **This changes results** for any consumer that evaluated
  azimuthally asymmetric `RLR` fields at unstructured points.
- **`_get_ahat_cache_rlz` never reloaded the per-wavenumber coupled-border registry**, so
  unstructured and collar evaluation of a nested patch pinned its R3X borders to zero. It
  now mirrors `gridTransform(_RLRGrid)`'s registry loads.

#### Tiled transforms

- **The tiled RL b→a solves now reload the coupled per-wavenumber border.**
  The 2- and 3-argument `splineTransform!(…, ::RL_Grid)` reused the three
  k0/real/imag spline objects across all wavenumbers without reloading their
  `ahat` from the multi-patch registry, so an R3X-coupled RL patch carried the
  *last-applied* wavenumber's border on every mode (only k=0 was right). Both
  methods — and the `_get_ahat_cache_rl` coefficient cache behind the
  unstructured evaluators — now mirror `gridTransform`'s per-wavenumber registry
  loads.
- **The tiled `RLR` `splineTransform!` ignored the R3X per-wavenumber `ahat` registry.**
  Both `RLR` methods (single-tile 2-argument and patch/tile 3-argument) solved every
  `(z_b, k)` block without reloading the per-wavenumber coupled-border coefficients that
  `apply_interface_payload!` registers, so a nested `RLR` patch's R3X borders were
  silently ignored on every worker-loop solve — the RL tiled-transform bug reincarnated on
  the `RLR` path. Field evidence: a 2-patch nested `RLR` run diverged from its
  bitwise-matched nested-axisymmetric twin by O(1) relative within 300 s, while the
  single-grid control matched to 1e-10 over the same integration. Both methods now reload
  slots `z_slot_base + 0 / 1+p / 2+p` (`p = (k−1)*2`), matching `gridTransform(_RLRGrid)`.
- **The out-of-place `SAtransform(spline, b)` now honors the R3X `ahat`.** This
  allocating form is what every 3-argument (tiled) `splineTransform!` method uses
  for the b→a solve on the patch splines, and it ignored `spline.ahat` — so a
  rank-3-coupled (R3X) patch lost its donated border trio on the distributed tile
  path, with the border silently pinned to zero while the in-place `SAtransform!`
  honored the coupling. The rank-3 inhomogeneous path now matches `SAtransform!`
  exactly; a regression test asserts allocating == in-place for an R3X spline
  with nonzero `ahat`.
- **The tiled 3-D `splineTransform!` used the wrong patch stride on inner tiles.** The
  3-argument method for the z-major 3-D layouts (`RLZ`/`SLZ`, and the newly added `RLR`)
  computed the *patch* z-level base index with the *tile's* wavenumber block count.
  Sub-blocks are aligned by wavenumber, but an inner tile carries fewer wavenumbers than
  the patch (its `kDim` follows its outermost ring), so every z level past the first read
  the wrong shared rows on inner tiles. Single-tile runs (tile = patch) are unchanged —
  the strides coincide. A multi-tile b→a round-trip test for `RLZ`/`RLR`/`SLR` seeds
  wavenumber-1 content and reconstructs the physical field on every tile of a 3-way
  radial decomposition; it was verified to fail against the pre-fix stride math.
- `_create_tile_from_patch` did not propagate a patch's spline j/k cell counts, so a tile
  re-derived them from its own narrower i-domain. This was latent (the auto-default is
  tile-invariant) but would have produced mismatched coefficient-array shapes across
  tiles as soon as a patch set `num_cells_j` explicitly.

#### Multipatch coupling and column semantics

- **`num_columns(::RRR_Grid)` returned a spectral block count, not physical columns.** It
  returned `b_jDim * b_kDim`, while every other grid with a vertical dimension returns the
  number of physical z-fastest columns (`iDim` for `RZ`/`RiRk`, `jDim` for
  `RLZ`/`RLR`/`SLZ`/`SLR`). Model drivers advance the state one vertical column at a time
  and stride by `kDim`, so the `RRR` value must be `iDim * jDim`. No caller used the old
  value.
- RZ tiling: restored physical-column `num_columns` semantics.
- **The `RRR` `:per_mode` inter-patch coupling only coupled `b_jDim` of `b_jDim*b_kDim`
  i-splines.** The fill/apply kernels iterated `size(ibasis.data, 1)` modes and indexed
  `data[l, v]`, which for a 3-D `RRR` `ibasis` (`b_jDim, b_kDim, nvars`) misses the
  `b_kDim` factor and collapses onto the `k = 1` slice via partial linear indexing. The
  kernels now use `n_modes = length(ibasis.data) ÷ nvars` with
  `reshape(ibasis.data, :, nvars)`, so `RR` (2-D) and `RRR` (3-D) share one index —
  column-major order makes the flattened index exact. This was structurally broken
  independent of `createMultiGrid`. The test helper `_snapshot_patch` collapsed the same
  way and so never checked `RRR`'s `k > 1` modes in the round trip; it is fixed, and an
  explicit per-k-plane coupling assertion was added.

#### Integration and reference states

- **The generic `IInttransform` was mis-anchored on spline columns.** The basis-agnostic
  `IInttransform(spline, [uMish,] C0)` wrappers delegated to `SIInttransform`, whose
  Ooyama (2002) antiderivative coefficients carry their own gauge (zero near the domain
  center) with `C0` added uniformly, while the Chebyshev `IInttransform` anchors the result
  to `C0` at `zmin`. Cross-basis callers meaning "`C0` = value at the bottom" — the
  hydrostatic reference-state integrations, and boundary-layer `w`-from-divergence — got a
  silently mis-anchored profile: on a 20 km column,
  `calculate_pressure_reference_state` returned a 2247 hPa surface pressure from a
  1014.8 hPa sounding. The generic wrappers now evaluate the antiderivative at `xmin` and
  shift so the result equals `C0` there, matching Chebyshev. The spline-native
  `SIInttransform`/`SIInttransform!` keep their original gauge and are untouched.
- **The hydrostatic reference-state sweep did not converge, and the pressure
  antiderivative was thrown away.** Both defects are behind `hydrostatic = true` and
  bitwise inert when off.
  - The fixed-point sweep in `calculate_pressure_reference_state` *oscillates* before it
    settles, and a hard-coded `for _ in 1:5` stopped it mid-swing. On a 50-cell / 25 km TC
    grid with the Dunion moist-tropical sounding the lid pressure ran
    1912 → 4010 → 1512 → 3550 → 2273 → … → 2707 Pa, so the returned `p` was 16% below
    convergence and the returned `(p, ρ_d)` pair was mutually inconsistent by 27%. The
    sweep now iterates to 1e-12 or throws.
  - `_pressure_reference` built `p` by `IInttransform` of the fitted `−g·ρ_t` and then
    discarded that spline, re-fitting its *values*. A 0.03% value-fit error on 1e5 Pa is
    ~30 Pa, which across a 500 m cell is ~0.06 Pa/m — 10–17% of `g·ρ_t` where `p` is
    small. `_hydrostatic_pressure_profile` now keeps the antiderivative for the value slot
    and snaps the derivative slots to `−g·ρ_tbar`, so discrete hydrostatic balance is
    exact. Measured on the TC grid, `−(dp/dz + g·ρ_t)/ρ_t` went from −2.0e−01 at 0.06 km /
    +1.67e+00 at 22.6 km to 0.0 at every level. Since `_pressure_reference` is the shared
    entry point for both the calculate and exact builders, the values-only `.ref` round
    trip now carries a balanced reference with no file-format change: `p` is re-integrated
    from `ρ_t`, and the file's pressure is only the anchor and the accuracy check
    (rejected above 1%).
- Reference-state builder: copy the `Itransform!` result and recompute saturation
  pressure (correctness fix in the new reference-state path).

#### IO and grid construction

- **`read_netcdf` threw on files with a CF time coordinate.** A time axis
  written by `write_netcdf(...; time=t)` (units `"seconds since ..."`, a
  `calendar`) is decoded by NCDatasets to `DateTime`, which the reader tried to
  store in a `Dict{String, Vector{Float64}}` — a `convert` error. The
  `"coordinates"`/`"variables"` containers are now `Any`-valued, so the decoded
  time is preserved (`data["coordinates"]["time"]::Vector{DateTime}`) alongside
  the `Float64` spatial coordinates.
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

#### Test-suite hygiene

- An operator-algebra testset titled "ZZ grid" actually constructed a `ZZZ` grid — title
  corrected.

### Maintenance

- CI: added the `reference_state`, `thermodynamics`, `cell_counts`, `nesting_support`,
  and `positivity` test groups to the matrix; TagBot now receives the token and SSH key.
- CI gained a `parity-check` job (`.github/scripts/check_test_group_parity.jl`) that the
  test matrix `needs:`, so a group wired into `test/runtests.jl` but missing from the
  `CI.yml` matrix fails fast with a named diff instead of silently going unrun.
- Removed stale top-level dev scripts (`test_ci.sh`, `test_regular_grid.jl`,
  `.actrc`).

### Documentation

- **New page: Thermodynamics & Reference States.** The `Thermodynamics` submodule and the
  reference-state module had no documentation page at all; `Thermodynamics` is now also in
  `makedocs(modules=...)`, so its docstrings appear in the manual.
- The `R3X`/`R1T1X` inhomogeneous boundary families, positivity-constrained fits, and
  two-way nesting (subcycling, collar interfaces, the fine→coarse evaluators) are now
  documented narratively, not just in docstrings.
- `RLR`, `SLR`, `RiRk`, and `RiRj_Grid` were missing from the grid-type reference table.
- **`load_grid`'s docstring was never attached to `load_grid`.** It was separated from the
  function by the `_upgrade_params` definitions, so Julia bound it to `_upgrade_params` —
  an internal function — and the text rendered nowhere. `SVDLinearBackend` and `BCSpec`
  had docstrings but no `@docs` block. The manual now has no missing-docs, unresolved
  cross-reference, or unincluded-docstring warnings.
- The docs `size_threshold` was raised from 300 KB to 512 KB. `springsteel_grid.md`
  renders to ~285 KB, so the old ceiling was 15 KB from failing the build outright.
- Corrected the `grid_from_netcdf` example in `docs/src/interpolation.md`, which passed
  `dim_names` as a tuple where the signature takes a `Vector{String}` — the documented
  call raised a `TypeError`.

### Known limitations

Both are pre-existing (not v1.1 regressions), both fail loudly, and both are targeted
at v1.1.1.

- **`grid_from_netcdf` does not support a time axis**
  ([#22](https://github.com/csu-tropical/Springsteel.jl/issues/22)). It builds a single
  spatial grid and has no notion of a time dimension. A CF time axis is decoded to
  `DateTime` and raises `MethodError: no method matching Float64(::DateTime)` — including
  on files written by `write_netcdf(grid; time=t)`. Passing `dim_names` alone does not
  work around it: the data-variable inference excludes only the caller's chosen dims, so
  the time variable is picked up as a data variable and hits the same conversion. Both
  `dim_names` and `var_names` must be given. A time axis carrying no CF `units` attribute
  decodes to `Float64` and is **silently adopted as a spatial dimension**, fitting a
  spline through time with no error or warning. Multi-timestep files cannot be loaded at
  all. `read_netcdf` handles CF time correctly and is the workaround for reading such
  files.
- **`write_netcdf` output does not round-trip through `grid_from_netcdf` for most cell
  counts** ([#24](https://github.com/csu-tropical/Springsteel.jl/issues/24)).
  `write_netcdf` emits `num_cells + 1` regular gridpoints, while `grid_from_regular_data`
  requires the coordinate length be a multiple of `mubar`; with the default `mubar = 3`
  those agree only when `num_cells ≡ 2 (mod 3)`. Setting `i_regular_out` (and the j/k
  equivalents) to a multiple of `mubar` produces readable output.

## [1.0.0] - 2025

Initial stable release: the unified `SpringsteelGrid{G, I, J, K}` type system over
mixed CubicBSpline / Fourier / Chebyshev bases, the solver framework, grid-to-grid
interpolation, spectral filtering, multi-patch grid connections, basis-template
caching, and grid relocation. See the Git tag history for releases prior to 1.1.0.

[Unreleased]: https://github.com/csu-tropical/Springsteel.jl/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/csu-tropical/Springsteel.jl/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/csu-tropical/Springsteel.jl/releases/tag/v1.0.0

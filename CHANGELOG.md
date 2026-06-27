# Changelog

All notable changes to Springsteel.jl are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

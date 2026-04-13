```@meta
CurrentModule = Springsteel
```

# Contributing

This page is for people working on the Springsteel source — whether
that's adding a feature, fixing a bug, writing a test, or extending
the solver framework. It covers the development workflow, test
conventions, architecture invariants you should know before touching
the transform pipeline, and the v1.1+ roadmap.

User-facing documentation (how to use the package) lives in the other
pages; this one is about the *inside*.

## Getting started

Clone the repo and install in development mode:

```bash
git clone https://github.com/csu-tropical/Springsteel.jl
cd Springsteel.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

Run the full test suite once to verify your environment:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

The suite takes ~90 seconds cold, ~60 seconds warm, and passes 12,000+
tests. If anything fails on a clean clone, that's a bug worth filing.

## Running tests

### Full suite

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

### A single test group

During development, run only the group you're touching for fast
iteration. Use the `TEST_GROUP` environment variable:

```bash
julia --project -e 'ENV["TEST_GROUP"]="solver"; using Pkg; Pkg.test()'
```

Available groups (from `test/runtests.jl`):

```
basis            banded_cholesky  grids            transforms
tiling           io               solver           mubar
interpolation    filtering        r3x              bc
multipatch       tile_multipatch  operator_algebra solver_problem
basis_cache      relocation
```

Each group corresponds to one file under `test/`. The top-level
`runtests.jl` is just a dispatcher.

### Bench scripts

Performance regressions are checked manually (not in CI — runner
variance is too flaky). See `bench/README.md` for the workflow:

```bash
# Snapshot baseline from main before your change
git checkout main
julia --project bench/bench_grids.jl bench/baseline/grids_pre.csv
julia --project bench/bench_interpolation.jl bench/baseline/interp_pre.csv

# Make your changes, re-run, diff
git checkout my-feature
julia --project bench/bench_grids.jl bench/baseline/grids_post.csv
diff bench/baseline/grids_pre.csv bench/baseline/grids_post.csv
```

A >20% slowdown on any row or any new allocation is a red flag —
investigate before merging.

## Test conventions

Test files don't include `using Springsteel` or `using Test` — those are
loaded once in `runtests.jl` so that individual files can be included
directly without re-importing. Each test file has one top-level
`@testset` wrapping the sub-testsets:

```julia
@testset "Solver Tests" begin
    @testset "Operator Matrix Assembly" begin
        # ...
    end
    @testset "SpringsteelProblem" begin
        # ...
    end
end
```

When adding a new test file:

1. Create `test/<name>.jl` with the above shape.
2. Add one dispatch line to `test/runtests.jl`:
   `TEST_GROUP in ("all", "<name>") && include("<name>.jl")`.
3. Prefer deterministic inputs over randomness in assertions. If you do
   use RNG, print the seed.
4. Use analytic solutions as ground truth. For BVPs, tests that check
   `sol.physical ≈ u_analytic` to 1e-3..1e-8 (depending on grid size)
   are the gold standard.
5. Don't mock the basis modules or the solver — run them end-to-end.
   Mocks of numerical code drift from reality; the suite's value comes
   from its fidelity.

## Docstring conventions

Every exported symbol gets a triple-quoted docstring directly above its
definition. Keep them substantive — describe the *why*, the inputs and
outputs, the edge cases, and any non-obvious invariants.

```julia
"""
    assemble_from_equation(grid, var; d0=nothing, d_ii=nothing, ...)

Assemble a multi-dimensional operator matrix via Kronecker products.
...
"""
function assemble_from_equation(grid, var; kwargs...)
    # ...
end
```

Rules:

- Lead with the signature line; Documenter.jl uses it for cross-refs.
- Use `# Fields`, `# Arguments`, `# Returns`, `# Example` sections as
  needed.
- Cross-link with `[`otherfunction`](@ref)` — the Documenter build
  checks these and warns on breakage.
- Do not reference deprecated or removed APIs. The audit from commit
  `1871ef9` onwards caught `sol.info`, `prob.operator`, and
  `field.grid` in stale docstrings; don't let them creep back.
- Internal helpers (`_foo_bar`) don't need docstrings unless the logic
  is non-obvious.

## Architecture invariants

These are the things you should know *before* touching the transform
pipeline, the tiling code, or the multipatch machinery. They're
intentionally non-uniform across grid types — resist the urge to
"clean them up" until you understand why they differ.

### Spectral array layouts

Every grid stores spectral data in `grid.spectral[row, var]`. The row
ordering scheme **differs by grid type** and must not be unified. RL
and RLZ arrays look similar but use different wavenumber offset
formulas; see TRAP-1 below.

**1D grids**

| Grid | Layout           | Total rows |
|:-----|:-----------------|:-----------|
| `R`  | `[1:b_iDim, v]`  | `b_iDim`   |
| `Z`  | `[1:b_iDim, v]`  | `b_iDim`   |
| `L`  | `[1:b_iDim, v]`  | `b_iDim`   |

**2D grids**

| Grid | Layout | Total rows |
|:-----|:-------|:-----------|
| `RR` / `LL` | row-major: `r1 = (l-1)*b_iDim + 1` | `b_jDim * b_iDim` |
| `RZ`        | z-major: `r1 = (z-1)*b_iDim + 1`  | `b_kDim * b_iDim` |
| `RL`        | wavenumber-interleaved flat: `[k=0 | k=1 real | k=1 imag | …]` | `b_iDim * (1 + 2*kDim)` |
| `SL`        | same as RL                         | `b_iDim * (1 + 2*kDim)` |

**3D grids**

| Grid | Layout | Total rows |
|:-----|:-------|:-----------|
| `RRR` / `ZZZ` / `LLZ` | z-major, then row-major L: `idx = (z-1)*b_jDim*b_iDim + (l-1)*b_iDim + 1` | `b_kDim * b_jDim * b_iDim` |
| `RLZ` | z-major, then wavenumber-interleaved: `r1 = (z-1)*b_iDim*(1+kDim_wn*2) + 1` | `b_kDim * b_iDim * (1 + kDim_wn*2)` |
| `SLZ` | same as RLZ | `b_kDim * b_iDim * (1 + kDim_wn*2)` |

### TRAP-1: RL vs RLZ wavenumber offset formulas differ

**Files**: `transforms_cylindrical.jl`, `transforms_spherical.jl`.

```
RL  spectralTransform:  p = k*2     for k ≥ 1
RLZ spectralTransform:  p = (k-1)*2 for k ≥ 1
```

Both are correct — they refer to different quantities.

- **RL / SL (2D)**: The spectral array is a single flat interleaved
  vector. The block index of the real part of wavenumber `k` is `2k`
  (1-based), hence `p = k*2`.
- **RLZ / SLZ (3D)**: The array is z-major. *Within each z-level* the
  k=0 block comes first; the offset to wavenumber `k ≥ 1` measured from
  just after the k=0 block is `(k-1)*2` blocks, hence `p = (k-1)*2`.

The two implementations were written independently. Do not harmonise
these formulas without rewriting one transform end-to-end.

### TRAP-2: Halo boundary arithmetic uses `-4`, not `-3`

**File**: `tiling.jl` (`calcHaloMap`).

The tile overlap width is exactly 3 B-spline coefficients. With
`b_iDim = num_cells + 3`, the last *owned* coefficient of a tile is at
row `b_iDim - 4` (1-based), and the 3 shared halo coefficients are at
`b_iDim-3 : b_iDim-1`.

`tileShare = b_iDim - 4` is correct. Changing to `-3` gives 2-wide
overlap (misses one shared coefficient); `-5` gives 4-wide (double-
counts one). Either breaks the assembled spline.

### TRAP-3: Tiled cylindrical transform reuses `patchSplines[1, v]`

**File**: `tiling.jl`.

In the tiled cylindrical transform, `patchSplines[1, v]` is reused for
wavenumber 0 and **overwritten** for each subsequent wavenumber. Only
one wavenumber is live in the buffer at any moment. Don't allocate
separate `Spline1D` objects per wavenumber — the reuse pattern is
necessary for memory efficiency on large tiled cylindrical grids.

### TRAP-4: Physical index formulas are grid-specific

**Files**: `transforms_cylindrical.jl`, `transforms_cartesian.jl`.

RL and RLZ grids have **variable-length Fourier rings** (ring size
`4 + 4*(r + patchOffsetL)` for RL, sin(θ)-based for SL) and cannot use
a simple product formula.

| Grid | Physical index formula | Notes |
|:-----|:-----------------------|:------|
| `R`   | `physical[r, v, d]`               | direct |
| `RL`  | running: `l1 = l2+1; l2 = l1+3+4*ri` | variable rings |
| `RR`  | `flat = (r-1)*jDim + l`             | regular |
| `RZ`  | `flat = (r-1)*kDim + z`             | regular |
| `RLZ` | running: accumulate `lpoints * b_kDim` per ring | variable rings |
| `RRR` | `flat = (r-1)*jDim*kDim + (l-1)*kDim + z` | regular |

Don't generalise these.

### RACE-1: `SharedArray` halo summation is non-atomic

**File**: `tiling.jl` (`sumSharedSpectral`).

`shared_spectral[idx] += tile_spectral[local_idx]` is a
read-modify-write on a `SharedArray`. Two workers operating on adjacent
tiles whose halo zones overlap will race.

Current callers serialise halo-zone writes. If you extend distributed
execution, either preserve serialisation or add explicit barriers /
atomic operations.

### RACE-2: `tileTransform!` per-variable threading invariant

**File**: `tiling.jl` (`tileTransform!`).

The `Threads.@threads for v in 1:length(pp.vars)` loop is safe *only*
because:

1. `ibasis.data[1, v]` objects are fully independent per variable.
2. `physical[:, v, :]` slices are non-overlapping.
3. No shared mutable state is read or written across `v`.

Any refactor that shares a basis across variables or changes the
physical array layout introduces a race.

### RACE-3: `splineTransform!` needs fully-populated `SharedArray`

**File**: `tiling.jl` (`splineTransform!`).

Read-only access to a `SharedArray` is safe *only after* all workers
have finished writing. There's no internal barrier — the caller must
guarantee completeness before parallel reads.

### Critical constants

Derived from the Ooyama (2002) B-spline formulation. Any deviation
produces wrong results.

| Name | Value | Meaning |
|:-----|:------|:--------|
| `mubar`          | 3                 | Gauss-Legendre quadrature points per B-spline cell |
| `iDim`           | `num_cells * mubar` | Total physical gridpoints in the i-direction |
| `b_iDim`         | `num_cells + 3`     | Number of spectral B-spline coefficients |
| `lpoints`        | `4 + 4*ri`          | Fourier ring size at radius index `ri` (RL / RLZ) |
| `halo_width`     | 3                 | Shared B-spline coefficients in tile overlap |
| `min_tile_cells` | 3                 | Minimum B-spline cells per tile (≥ 9 gridpoints) |

### Transform ordering reference

**Physical → spectral (forward)**

| Grid  | Stage order |
|:------|:------------|
| `R`   | SBtransform |
| `RL`  | FBtransform per ring → SBtransform per wavenumber |
| `RR`  | SBtransform in j (per i) → SBtransform in i (per j) |
| `RZ`  | CBtransform per column → SBtransform per z-coefficient |
| `RLZ` | CBtransform per column → FBtransform per ring×z → SBtransform per wavenumber×z |
| `RRR` | SBtransform k → SBtransform j → SBtransform i |
| `SL`  | FBtransform per sin(θ) ring → SBtransform per wavenumber |
| `SLZ` | CBtransform per column → FBtransform per ring×z → SBtransform per wavenumber×z |

Inverse `gridTransform!` runs the stages in reverse.

### Fourier indexing invariants

- **Half-complex convention**: imaginary part of wavenumber `k` is
  stored at `bDim - k + 1` within the Fourier coefficient array.
- **Ring azimuthal offset**: `offset = 0.5 * dl * (ri - 1)` (phase
  staggering between rings; handled by `calcPhaseFilter`).
- **Per-ring kmax**: for cylindrical grids `kmax_ring = r + patchOffsetL`;
  for spherical grids `kmax_ring = ring.params.kmax`.

### Tiling

Tiling is restricted to the i-dimension only. B-splines are the only
basis with compact local support; Fourier and Chebyshev bases are
global and cannot be partitioned with finite halos.

| Geometry | Tiling |
|:---------|:-------|
| Cartesian + Spline i | Full 1-D tiling |
| Cylindrical (RL, RLZ) | 1-D in i; all wavenumber blocks present per tile |
| Spherical (SL, SLZ)   | 1-D in i (same as cylindrical) |
| No Spline i          | No tiling (single-tile metadata only) |

Multi-D tiling for pure-spline Cartesian grids is implemented in
`tiling.jl` but not yet exercised in production.

## Performance notes

Transform and interpolation allocation expectations after the v1.0
refactor (`bench/bench_grids.jl`, `bench/bench_interpolation.jl`):

- Every multi-D grid (`R`, `RR`, `RZ`, `RL`, `SL`, `RRR`, `RLZ`, `SLZ`)
  has **zero allocations per call** for `gridTransform!` and
  `spectralTransform!` at steady state.
- `SAtransform!` pipelines scale O(Mdim) via the structured GammaBC
  operator.
- `createGrid` is sub-millisecond warm via the basis template cache
  (cold construction of a CONUS-10km RR grid was 1.9 s; warm is 1.8 ms
  — a 1000× speedup).
- `update_interface!` is zero alloc after the `PatchInterface{P,S}`
  parameterisation.
- Repeated `solve!` on a prebuilt `SpringsteelProblem` is 1.6 μs per
  call at N=25 Chebyshev Poisson, allocating only the inevitable ~400 B
  for the backsolve return.
- Interpolation hot paths use cached γ-folded ahat stripes
  (`_AHAT_CACHE`) and per-grid scratch registries
  (`_ScratchInterpRL` / `_ScratchInterpRLZ`). `RL` unstructured
  evaluation dropped 24.8 MB → 494 kB; `RLZ` dropped 27 MB → 75 kB.

Don't regress these. `bench/bench_grids.jl` runs in ~30 s and
`bench/bench_interpolation.jl` in ~2 min. Any non-zero allocation on a
transform or >20 % slowdown is a bug.

## v1.1+ roadmap

Items deferred from the v1.0 perf and feature arc, parked as candidates
for future releases. All of these are scoped and have no API-breaking
implications in v1.0.

- **Multi-threading outer transform loops** — requires per-thread
  scratch restructure in `_solve_*` paths and the grid transforms.
  Tier 1 in the old perf doc. Biggest potential win for large grids.
- **Fully unify `MultiPatchGrid` dispatch** — parameterise
  `MultiPatchGrid` on its patch / interface types to eliminate the
  residual ~2 kB `multiGridTransform!` allocation from heterogeneous
  vector dispatch. Small but annoying.
- **Non-cylindrical grid relocation** — `relocate_grid!` currently
  supports `RL` and `RLZ` only. Cartesian and spherical relocations are
  feasible via the same `evaluate_unstructured` machinery but would
  need a coordinate mapping layer and new boundary strategies.
- **Precompute `SAtransform` application matrix** (Tier 3 #9) — a
  per-wavenumber cache; only a win at `Mdim < 20`.
- **Combined `SBtransform + SAtransform` matrix** (Tier 3 #10) — O(nc²)
  memory growth, so only usable for small-to-moderate grids; big win
  when applicable.
- **Cache-friendly physical / spectral layout** (Tier 3 #11) — high
  risk refactor that reorders the physical array for better cache
  locality on wide multi-var grids. Would need a clear downstream
  workload driving it.
- **Additional grid types are "available but untested"**: `Z`, `ZZ`,
  `ZZZ`, `L`, `LL`, `LLZ`. They build and transform, but coverage is
  thin compared to the RR / RL / RLZ production paths. Filling in test
  suites for these is a good v1.1 contribution that won't break any
  existing API.

## See also

- [`bench/README.md`](https://github.com/csu-tropical/Springsteel.jl/tree/main/bench) — manual perf workflow
- `CLAUDE.md` in the repo root — project-specific agent instructions
- `agent_files/` (gitignored) — planning docs and scratch notes

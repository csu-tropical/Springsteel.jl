```@meta
CurrentModule = Springsteel
```

# Developer Notes

This page collects architectural decisions, invariants, known pitfalls, and
concurrency rules that every contributor and AI agent must understand before
modifying the transform pipeline or tiling infrastructure.  These notes are
derived from careful analysis of the original legacy grid implementations and
capture decisions that are *intentionally* non-uniform across grid types — do
not "clean them up."

---

## 1. Spectral Array Layouts

Every grid type stores spectral data in a `spectral[row, var]` matrix.  The
row-ordering scheme differs by grid type and **must not be unified** — the RL
and RLZ arrays look similar but use different wavenumber offset formulas for a
valid reason (see [TRAP-1](@ref trap1) below).

### 1D grids

| Grid | Layout (per variable) | Total spectral rows |
|:---- |:--------------------- |:------------------- |
| `R`  | `spectral[1:b_iDim, v]` | `b_iDim` |
| `Z`  | `spectral[1:b_iDim, v]` | `b_iDim` |
| `L`  | `spectral[1:b_iDim, v]` | `b_iDim` |

### 2D grids

| Grid | Layout (per variable) | Total spectral rows |
|:---- |:--------------------- |:------------------- |
| `RR` / `LL` | Row-major: `r1 = (l-1)*b_iDim + 1` | `b_jDim * b_iDim` |
| `RZ`        | Z-major: `r1 = (z-1)*b_iDim + 1` | `b_kDim * b_iDim` |
| `RL`        | Wavenumber-interleaved (flat): `[k=0 block \| k=1 real \| k=1 imag \| ...]` | `b_iDim * (1 + 2*kDim)` |
| `SL`        | Same as RL | `b_iDim * (1 + 2*kDim)` |

**RL wavenumber indexing**: the real part of wavenumber `k ≥ 1` starts at row
`(2k-1)*b_iDim + 1`, and the imaginary part at `2k*b_iDim + 1`.  In the code
this is expressed as `p = k * 2` where `p` is the 1-based block index of the
real part (so rows `= (p-1)*b_iDim+1 : p*b_iDim`).

### 3D grids

| Grid | Layout (per variable) | Total spectral rows |
|:---- |:--------------------- |:------------------- |
| `RRR` / `ZZZ` / `LLZ` | Z-major, then row-major L: `idx = (z-1)*b_jDim*b_iDim + (l-1)*b_iDim + 1` | `b_kDim * b_jDim * b_iDim` |
| `RLZ` | Z-major, then wavenumber-interleaved: `r1 = (z-1)*b_iDim*(1+kDim_wn*2) + 1` | `b_kDim * b_iDim * (1 + kDim_wn*2)` |
| `SLZ` | Same as RLZ | `b_kDim * b_iDim * (1 + kDim_wn*2)` |

**RLZ / SLZ wavenumber indexing**: the spectral array is Z-major.  Within each
z-level block (starting at row `r1` above), the k=0 spline block occupies the
first `b_iDim` rows and then wavenumber `k ≥ 1` real starts at `b_iDim + (k-1)*2*b_iDim`.
In code this is `p = (k-1) * 2` — the real part is at row
`r1 + b_iDim + p*b_iDim`.

---

## 2. [TRAP-1](@id trap1) — RL vs RLZ wavenumber offset formulas differ

**Severity**: medium.  **Files**: `transforms_cylindrical.jl`, `transforms_spherical.jl`.

```
RL  spectralTransform:  p = k*2     for k ≥ 1
RLZ spectralTransform:  p = (k-1)*2 for k ≥ 1
```

Both are **correct** — they refer to different quantities.

- **RL / SL** (2-D): The spectral array is a single flat interleaved vector.
  The block index of the real part of wavenumber `k` counted from position 0 is
  `2k - 1` (0-based) = `2k` (1-based block number, hence `p = k*2`).

- **RLZ / SLZ** (3-D): The array is Z-major.  *Within each z-level* the k=0
  block comes first; the offset to wavenumber `k ≥ 1` measured from *just after
  the k=0 block* is `(k-1) * 2` blocks, hence `p = (k-1)*2`.  The absolute row
  within the z-level block is therefore `b_iDim + p*b_iDim`.

**Why they differ**: The two implementations were written independently (RL
preceded RLZ).  The RL convention counts from the absolute start of the spectral
array; the RLZ convention counts from the end of the k=0 block within a z-level.
Both are internally consistent — changing either formula breaks the corresponding
grid.

**Danger**: If someone "harmonises" these formulas, one transform will silently
produce wrong spectral data.

---

## 3. [TRAP-2](@id trap2) — Halo boundary arithmetic: `-4` vs `-3`

**Severity**: medium.  **File**: `tiling.jl` (`calcHaloMap`).

The tile overlap width is exactly **3 B-spline coefficients**.  With
`b_iDim = num_cells + 3`, the last fully *owned* coefficient of a tile is at
row index `b_iDim - 4` (1-based), and the 3 shared halo coefficients are at
rows `b_iDim - 3 : b_iDim - 1` (but the tile's last stored index is `b_iDim`).

The partition formula `tileShare = b_iDim - 4` is correct.  Changing it to
`-3` means 2-wide overlap (misses one shared coefficient); changing to `-5` means
4-wide (double-counts one).  Either way the assembled spline is wrong.

---

## 4. [TRAP-3](@id trap3) — RL tiled `splineTransform!` reuses `patchSplines[1,v]`

**Severity**: medium.  **File**: `tiling.jl`.

In the tiled cylindrical transform, `patchSplines[1, v]` is reused for
wavenumber-0 and **overwritten** for each subsequent wavenumber (`patchSplines[2, v]`
and `patchSplines[3, v]` carry real/imaginary parts, also reused per wavenumber).
This is intentional: only one wavenumber is live in the buffer at any moment.

Do **not** allocate separate `Spline1D` objects per wavenumber.  The reuse
pattern is necessary for memory efficiency in large tiled cylindrical grids.

---

## 5. [TRAP-4](@id trap4) — Physical array flattening formulas are grid-specific

**Severity**: medium.  **Files**: `transforms_cylindrical.jl`, `transforms_cartesian.jl`.

Each grid type has its own physical-array index formula.  These differ because
RL and RLZ grids have **variable-length Fourier rings** (ring size =
`4 + 4 * (r + patchOffsetL)` for RL, sin(θ)-based for SL) and cannot use a
simple product formula.

| Grid | Physical index formula | Notes |
|:---- |:---------------------- |:----- |
| `R`  | `physical[r, v, d]` | Direct |
| `RL` | Running counter: `l1 = l2+1; l2 = l1+3+4*ri` | Variable rings |
| `RR` | `flat = (r-1)*jDim + l` | Regular |
| `RZ` | `flat = (r-1)*kDim + z` | Regular |
| `RLZ`| Running counter: accumulate `lpoints * b_kDim` per ring | Variable rings |
| `RRR`| `flat = (r-1)*jDim*kDim + (l-1)*kDim + z` | Regular |

Do **not** generalise these into a single product formula.

---

## 6. [RACE-1](@id race1) — SharedArray halo summation is non-atomic

**Severity**: critical for distributed execution.  **File**: `tiling.jl` (`sumSharedSpectral`).

```julia
shared_spectral[idx] += tile_spectral[local_idx]   # NOT ATOMIC
```

This read-modify-write on a `SharedArray` is **not thread/process safe**.  If
two workers call `sumSharedSpectral` simultaneously on adjacent tiles whose halo
zones overlap, the result is a data race.

**Requirement**: Either serialise halo-zone writes across workers (current
assumption), or add explicit barriers / atomic operations around halo
accumulation.  This constraint must be documented at every call site.

---

## 7. [RACE-2](@id race2) — `tileTransform!` thread safety invariant

**Severity**: low if invariant is maintained.  **File**: `tiling.jl` (`tileTransform!`).

The `Threads.@threads for v in 1:length(pp.vars)` loop is safe **only because**:

1. `ibasis.data[1, v]` objects are fully independent per variable `v`.
2. `physical[:, v, :]` slices are non-overlapping (v is the trailing index).
3. No shared mutable state is read or written across `v` in the loop body.

Any refactoring that shares a basis object across variables or changes the
physical array layout would introduce a race.

---

## 8. [RACE-3](@id race3) — `splineTransform!` requires fully-populated SharedArray

**Severity**: low if caller serialises.  **File**: `tiling.jl` (`splineTransform!`).

`splineTransform!` reads from the shared spectral array.  Read-only access is
safe on `SharedArray` **only after** all workers have finished writing.  There
is no internal barrier — the caller must guarantee the array is complete before
any parallel reads begin.

---

## 9. Critical Constants

These values are invariants derived from the Ooyama (2002) B-spline formulation.
Any deviation produces wrong results.

| Name | Value | Meaning |
|:---- |:----- |:------- |
| `mubar` | 3 | Gauss-Legendre quadrature points per B-spline cell |
| `iDim` | `num_cells * mubar` | Total physical gridpoints in the i-direction |
| `b_iDim` | `num_cells + 3` | Number of spectral B-spline coefficients |
| `lpoints` | `4 + 4*ri` | Fourier ring size at radius index `ri` (RL/RLZ) |
| `halo_width` | 3 | Shared B-spline coefficients in tile overlap |
| `min_tile_cells` | 3 | Minimum B-spline cells per tile (≥ 9 gridpoints) |

---

## 10. Transform Ordering Reference

### Physical → Spectral (forward) order

| Grid | Stage order |
|:---- |:----------- |
| `R`  | SBtransform |
| `RL` | FBtransform per ring → SBtransform per wavenumber (k=0 and ±k) |
| `RR` | SBtransform in j (per i) → SBtransform in i (per j) |
| `RZ` | CBtransform per column → SBtransform per z-coefficient |
| `RLZ`| CBtransform per column → FBtransform per ring×z → SBtransform per wavenumber×z |
| `RRR`| SBtransform k → SBtransform j → SBtransform i |
| `SL` | FBtransform per sin(θ) ring → SBtransform per wavenumber |
| `SLZ`| CBtransform per column → FBtransform per ring×z → SBtransform per wavenumber×z |

Inverse (`gridTransform!`) runs these stages in reverse.

### Fourier indexing invariants

- **Half-complex convention**: imaginary part of wavenumber `k` is stored at
  `bDim - k + 1` within the Fourier coefficient array.
- **Ring azimuthal offset**: `offset = 0.5 * dl * (ri - 1)` (phase staggering
  between rings; handled by `calcPhaseFilter`).
- **Per-ring kmax**: for cylindrical grids `kmax_ring = r + patchOffsetL`; for
  spherical grids `kmax_ring = ring.params.kmax` (baked into the `Fourier1D`
  object at construction time).

---

## 11. Known Bugs Intentionally Fixed

The following bugs existed in the pre-refactoring legacy grid files.  The
current implementation fixes all of them; regression tests are present in
`test/runtests.jl`.

### BUG-1 — RLZ `gridTransform` kDim mismatch (fixed)

The old `gridTransform` used `kDim = grid.params.rDim` (missing `patchOffsetL`),
so higher wavenumbers were silently discarded during the inverse transform when a
grid had `patchOffsetL > 0`.  **Fix**: use `kDim_wn = iDim + patchOffsetL`
everywhere, matching `spectralTransform`.

### BUG-2 — RRR derivative slot 4 dimension mismatch (fixed)

In the old code the j-direction derivative was written to a slice expected to
hold `kDim` values but received `jDim` values instead, causing silent array
corruption when `jDim ≠ kDim`.  **Fix**: Derivative assignments use the correct
target dimension throughout the refactored `RRR` transform.

### BUG-3 — RRR `splineBuffer_l` stale data (fixed)

The old i-direction buffer was overwritten inside the j-loop, so only the last
j-index's data survived when the k-direction transform ran.  **Fix**: The buffer
is indexed `[jDim, b_kDim, iDim]` so each i-position's data is consumed before
being overwritten.

### BUG-4 — RZ tiled `gridTransform!` referenced undefined `grid` (fixed)

The tiled inverse transform referred to `grid.params` instead of `patch.params`,
causing `UndefVarError` on the first call.  **Fix**: correct variable names.

### BUG-5 — RL tile `tileTransform!` kDim omitted `patchOffsetL` (fixed)

Same class as BUG-1: the tiled RL inverse transform used `kDim = pp.rDim`
instead of `kDim = pp.rDim + pp.patchOffsetL`.  **Fix**: use
`kDim = iDim + patchOffsetL` everywhere.

---

## 12. Multi-Dimensional Tiling Design Notes

Tiling is **restricted to the i-dimension only**.  The mathematical reason is
that only B-splines (used in the i-dimension) have compact local support, making
overlapping halo widths well-defined.  Fourier and Chebyshev bases are global
and cannot be partitioned with finite halos.

For all-Spline Cartesian grids (e.g., `RRR`), multi-dimensional tiling is
theoretically possible because the spectral array is a regular tensor product.
A 2-D tile operation decomposes as "tile in i, then for each i-tile, tile in j."
Corner halo zones (3×3 blocks) must be summed exactly once.  This is implemented
in `tiling.jl` but not yet exercised in production.

**Geometry-specific tiling**:

| Geometry | Tiling support |
|:-------- |:-------------- |
| `CartesianGeometry` + `SplineBasisArray` i-dim | Full 1-D tiling |
| `CylindricalGeometry` (RL, RLZ) | 1-D tiling in i; wavenumber blocks must all be present in each tile |
| `SphericalGeometry` (SL, SLZ) | 1-D tiling in i (same as cylindrical) |
| Any geometry without Spline i-basis | No tiling (single-tile metadata only) |

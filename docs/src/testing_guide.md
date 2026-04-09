# Springsteel.jl Comprehensive Test Suite Guide

## Overview

This document describes the testing strategy developed for `CubicBSpline.jl` and `spline1D_grid.jl`, and provides step-by-step instructions for Claude to replicate a similarly comprehensive test suite for any other basis function module (e.g., `Fourier.jl`, `Chebyshev.jl`) and its associated grid file (e.g., `rl_grid.jl`, `rz_grid.jl`, `rlz_grid.jl`).

The test suite uses Julia's built-in `Test` standard library with 3392 tests passing as of the current implementation.

---

## Part 1: Test File Structure

Tests are split into separate files by functional area, loaded by the top-level
`test/runtests.jl` runner. Each file can be run independently via the
`TEST_GROUP` environment variable for faster iteration during development.

### 1.1 Test Runner (`test/runtests.jl`)

```julia
using Springsteel
using Test
using Dates
using JLD2
using NCDatasets
using SharedArrays
using SparseArrays
using DataFrames

const TEST_GROUP = get(ENV, "TEST_GROUP", "all")

@testset "Springsteel.jl" begin
    TEST_GROUP in ("all", "basis")      && include("basis.jl")
    TEST_GROUP in ("all", "grids")      && include("grids.jl")
    TEST_GROUP in ("all", "transforms") && include("transforms.jl")
    TEST_GROUP in ("all", "tiling")     && include("tiling.jl")
    TEST_GROUP in ("all", "io")         && include("io.jl")
    TEST_GROUP in ("all", "compat")     && include("compat.jl")
    TEST_GROUP in ("all", "solver")     && include("solver.jl")
end
```

### 1.2 Test File Responsibilities

| File | Contents | Approximate tests |
|:-----|:---------|:-----------------|
| `test/basis.jl` | Low-level basis module tests (CubicBSpline, Fourier, Chebyshev): parameter structs, 1D object construction, basis function evaluation, all transform dispatch variants, matrix representations, BC types | ~1200 |
| `test/grids.jl` | Grid creation tests for all geometries via `SpringsteelGridParameters`: dimensions, variable counts, multi-variable support | ~1600 |
| `test/transforms.jl` | Grid-level roundtrip accuracy, derivative accuracy, boundary condition enforcement, regular grid transforms for all grid types | ~2100 |
| `test/tiling.jl` | Distributed computing: `calcTileSizes`, `calcPatchMap`, `calcHaloMap`, `splineTransform!`, `tileTransform!`, spectral tile assembly | ~700 |
| `test/io.jl` | CSV, JLD2, and NetCDF I/O roundtrips | ~800 |
| `test/solver.jl` | Solver framework: operator matrix assembly, `SpringsteelProblem` construction, `solve()` for linear and nonlinear problems, multi-dimensional BVP tests | new |

### 1.3 Running Individual Test Groups

During development, run only the relevant group for fast iteration:

```bash
# Run only solver tests (~5-10 seconds vs ~40 seconds for full suite)
julia --project -e 'ENV["TEST_GROUP"]="solver"; using Pkg; Pkg.test()'

# Or directly without Pkg.test():
julia --project -e 'using Springsteel, Test; include("test/solver.jl")'

# Run only basis tests (for matrix representation work)
julia --project -e 'ENV["TEST_GROUP"]="basis"; using Pkg; Pkg.test()'

# Run everything (CI and final verification)
julia --project -e 'using Pkg; Pkg.test()'
```

### 1.4 Adding a New Test File

When adding a new test file (e.g., `test/solver.jl`):

1. Create the file with a top-level `@testset` that wraps all sub-testsets:
   ```julia
   @testset "Solver Tests" begin
       @testset "Operator Matrix Assembly" begin ... end
       @testset "SpringsteelProblem" begin ... end
       @testset "Linear Solver" begin ... end
   end
   ```
2. Add the include to `test/runtests.jl` with a new `TEST_GROUP` key:
   ```julia
   TEST_GROUP in ("all", "solver") && include("solver.jl")
   ```
3. Verify the new file runs independently before running the full suite.
4. The file should NOT include `using Springsteel` or `using Test` — these are
   already loaded by `runtests.jl`. The file is `include()`-d inside the
   `@testset "Springsteel.jl"` block.

### 1.5 Internal Testset Organization

Within each file, tests are organized in nested `@testset` blocks:

```julia
@testset "CubicBSpline Tests" begin     # basis function module low-level
    @testset "SplineParameters auto-computed fields" begin ... end
    @testset "Spline1D construction" begin ... end
    ...
end
```

The `CubicBSpline Tests` section tests the **module internals** (structs, math, transform API). The `Spline1D_Grid Tests` section tests the **grid-level integration** of those transforms with `SpringsteelGridParameters` and `SpringsteelGrid`. Separate them clearly in the test file.

---

## Part 2: CubicBSpline Module Test Pattern

### 2.1 Parameter Struct Tests

Every basis module defines a parameter struct (e.g., `SplineParameters`, `FourierParameters`, `ChebyshevParameters`). Test auto-computed derived fields:

```julia
@testset "{Module}Parameters auto-computed fields" begin
    sp = {ParamStruct}(xmin=0.0, xmax=10.0, num_cells=20)
    @test sp.DX ≈ 0.5           # (xmax - xmin) / num_cells
    @test sp.DXrecip ≈ 2.0      # 1 / DX
    # Test a second configuration to verify the formula is general
    sp2 = {ParamStruct}(xmin=-5.0, xmax=5.0, num_cells=10)
    @test sp2.DX ≈ 1.0
    @test sp2.DXrecip ≈ 1.0
end
```

### 2.2 Basis Object Construction Test

Every module has a constructor `{Basis}1D(params)`. Test that:
- Array dimensions match the mathematical formulas:
  - `mishDim` = `num_cells * mubar` (= `num_cells * 3` for B-splines)
  - `bDim` = `num_cells + 3` (for B-splines); equivalent for Fourier/Chebyshev
- Internal arrays are allocated with the right lengths
- Mish points lie strictly inside the domain and are monotonically increasing

```julia
@testset "{Basis}1D construction" begin
    sp = {ParamStruct}(xmin=0.0, xmax=10.0, num_cells=20, BCL=..., BCR=...)
    obj = {Basis}1D(sp)
    @test obj.mishDim == 60
    @test obj.bDim == 23
    @test length(obj.mishPoints) == 60
    @test length(obj.b) == 23
    @test length(obj.a) == 23
    @test obj.mishPoints[1] > sp.xmin
    @test obj.mishPoints[end] < sp.xmax
    @test all(diff(obj.mishPoints) .> 0)
end
```

### 2.3 Basis Function Tests (CubicBSpline specific)

For `CubicBSpline`, test the `basis()` function directly (it is internal but exported via `CubicBSpline.basis`):
- Peak value at the node center (`delta=0`): should equal `4/6`
- First derivative is zero at peak by symmetry
- Basis is zero at support boundary (`|delta| = 2`)
- Second derivative at peak matches formula
- Domain error is thrown for x outside `[xmin, xmax]`

For Fourier and Chebyshev, the equivalent internal evaluation functions (if any) should be similarly spot-checked.

### 2.4 setMishValues Test

```julia
@testset "setMishValues" begin
    obj = {Basis}1D(sp)
    u = sin.(obj.mishPoints)
    setMishValues(obj, u)
    @test obj.uMish ≈ u
    setMishValues(obj, zeros(length(u)))
    @test all(obj.uMish .== 0.0)
end
```

### 2.5 Forward Transform ({X}Btransform) Tests

Every module exports `{X}Btransform` in multiple dispatch flavors. Test **all** of them:

| Variant | Signature | Test |
|---------|-----------|------|
| `SplineParameters` only | `SBtransform(sp, uMish)` | Returns vector of length `bDim`, non-zero |
| `Spline1D` | `SBtransform(spline, uMish)` | Output identical to `SplineParameters` variant |
| In-place | `SBtransform!(spline)` (uses `spline.uMish`) | `spline.b` matches the two variants above |

```julia
@testset "{X}Btransform variants" begin
    obj = {Basis}1D(sp)
    u_vals = sin.(obj.mishPoints)

    b1 = {X}Btransform(sp, u_vals)
    @test length(b1) == obj.bDim
    @test maximum(abs.(b1)) > 0.0

    b2 = {X}Btransform(obj, u_vals)
    @test b1 ≈ b2

    setMishValues(obj, u_vals)
    {X}Btransform!(obj)
    @test obj.b ≈ b1
end
```

### 2.6 Derivative-of-Function B Transform (SBxtransform) Tests

`SBxtransform` computes the B vector of the derivative $f'$ using integration by parts:
$$b_m = \int \phi_m f' dx = [\phi_m f]_{x_0}^{x_0'} - \int \phi_m' f dx$$

Arguments `BCL = f(x_{min})` and `BCR = f(x_{max})` are the **function boundary values** (not BC-type dicts).

Four test cases cover the two dispatch variants, zero and non-zero boundary values, and the full SA+SI pipeline:

| Test | Function | BCs | Key assertion |
|------|----------|-----|---------------|
| A — linear (exact) | `f(x) = x`, `f'=1` | R0 | `SBxt(f, 0, L) ≈ SBt(1)` to machine precision (`< 1e-12`); 3-pt Gauss is exact for degree-1 polynomial |
| B — smooth, zero BCs | `f(x) = sin(πx/L)`, `f'=(π/L)cos` | R0 | `< 1e-10` with 40 cells; boundary terms vanish, residual is quadrature approximation error |
| C — periodic, zero BCs | `f(x) = sin(x)`, `f'=cos(x)` | PERIODIC | `< 1e-8` with 40 cells; higher frequency → larger quadrature error than Test B |
| D — full pipeline | same as C | PERIODIC | SA+SI recovers `cos(x)` to `< 0.01` |

```julia
@testset "SBxtransform variants" begin
    # Test A: linear f(x) = x — exact in quadrature
    sp_lin   = SplineParameters(xmin=0.0, xmax=10.0, num_cells=20,
                                BCL=CubicBSpline.R0, BCR=CubicBSpline.R0)
    spline_lin = Spline1D(sp_lin)
    pts_lin    = spline_lin.mishPoints
    bx_lin     = SBxtransform(sp_lin, pts_lin, 0.0, 10.0)          # SplineParameters variant
    bx_spline  = SBxtransform(spline_lin, pts_lin, 0.0, 10.0)      # Spline1D variant
    b_dir      = SBtransform(sp_lin, ones(length(pts_lin)))
    @test maximum(abs.(bx_lin .- b_dir)) < 1e-12    # machine precision for L1 polynomial
    @test bx_spline ≈ bx_lin

    # Test B: sin(π*x/L) — quadrature error for non-polynomial
    sp_sin = SplineParameters(xmin=0.0, xmax=10.0, num_cells=40, ...)
    bx_sin = SBxtransform(sp_sin, f_sin, 0.0, 0.0)
    @test maximum(abs.(bx_sin .- SBtransform(sp_sin, fp_sin))) < 1e-10

    # Test C: PERIODIC sin(x)
    sp_per = SplineParameters(xmin=0.0, xmax=2π, num_cells=40,
                              BCL=CubicBSpline.PERIODIC, BCR=CubicBSpline.PERIODIC)
    bx_per = SBxtransform(sp_per, sin.(pts_per), 0.0, 0.0)
    @test maximum(abs.(bx_per .- SBtransform(sp_per, cos.(pts_per)))) < 1e-8

    # Test D: SA+SI pipeline
    spline_per.b .= bx_per
    SAtransform!(spline_per)
    u_recov = zeros(length(pts_per))
    SItransform(spline_per, pts_per, u_recov)
    @test maximum(abs.(u_recov .- cos.(pts_per))) < 0.01
end
```

**Important tolerance guidance:**
The integration-by-parts identity holds *exactly* in Gauss quadrature only when both integrands ($\phi_m f'$ and $\phi_m' f$) are polynomials of degree $\leq 5$ within each cell. For cubic B-splines (degree 3) this requires $f$ to have degree $\leq 2$. For smooth but non-polynomial $f$, the residual error scales with the magnitude of $f^{(6)}$ and $DX^6$. Looser tolerances are therefore expected and appropriate for transcendental functions.

### 2.7 Coefficient Solve ({X}Atransform) Tests

Test all dispatch variants for `{X}Atransform`:

| Variant | Signature | Test |
|---------|-----------|------|
| 4-arg (params) | `SAtransform(sp, gammaBC, pqFactor, b)` | Returns finite vector |
| `Spline1D` | `SAtransform(spline, b)` | Same result |
| In-place | `SAtransform!(spline)` | `spline.a` matches |
| Inhomogeneous | `SAtransform(spline, b, ahat)` | With `ahat=zeros` equals standard result |

### 2.8 Inverse Transform ({X}Itransform) Tests

The inverse transform has the most dispatch variants. Systematically test each one. Use a smooth function whose exact transforms are known (e.g., `sin(π*x/L)` with R0 BCs, or `sin(x)` with PERIODIC BCs):

| Variant | Signature | Test |
|---------|-----------|------|
| Scalar single point | `SItransform(sp, a, x_scalar, 0)` | Matches analytic value at that point |
| All mish points (new vector) | `SItransform(sp, a, 0)` | Length = `mishDim`, array ≈ analytic |
| Arbitrary points (new vector) | `SItransform(sp, a, pts, 0)` | Matches mish-point result |
| In-place with params | `SItransform(sp, a, pts, u_buf, 0)` | Fills buffer, ≈ above |
| In-place via `!` | `SItransform!(spline)` | `spline.uMish` ≈ mish-point result |
| Via `Spline1D` (new buffer) | `SItransform(spline, u_buf)` | ≈ mish-point result |
| Via `Spline1D` + points | `SItransform(spline, pts, u_buf)` | ≈ mish-point result |

### 2.9 Matrix Representation Test

`SItransform_matrix` allocates a transform matrix `M` such that `M * a ≈ u`. Test that:
- Size is `(mishDim, bDim)`
- `M * a` gives the same result as `SItransform(sp, a)`

**Important caveat for CubicBSpline**: the `points` argument must have length exactly `mishDim` (the function allocates `rows = mishDim` regardless of `length(points)`).

### 2.10 Derivative Transform Tests ({X}Ixtransform, {X}Ixxtransform)

For each derivative transform, test all dispatch variants using a function with known derivatives (e.g., `sin(x)` with PERIODIC BCs has `cos(x)` as first derivative and `-sin(x)` as second):

| Variant | {X}Ixtransform | {X}Ixxtransform |
|---------|----------------|-----------------|
| No-arg (returns new vector) | ✓ | ✓ |
| Pre-allocated AbstractVector | ✓ | ✓ |
| Points + pre-allocated | ✓ | ✓ |
| SplineParameters form | ✓ | ✓ |

Tolerance for derivative accuracy with ~30 cells: `< 0.01` for first and second derivatives.

### 2.11 Boundary Condition Type Tests

Test that **every exported BC constant** can be used to construct a `Spline1D` and produce a finite round-trip. The BC constants for CubicBSpline are:
`R0`, `R1T0`, `R1T1`, `R1T2`, `R2T10`, `R2T20`, `R3`, `PERIODIC`

Match each test function to its BC:
- **`R0`** (no constraint) — any smooth function works; confirm `all(isfinite.(a))` after SA transform.
- **`R1T0`** (Dirichlet, `u = 0`) — use a function that *vanishes* at both boundaries (e.g., `sin(π*x/L)`) so the physical BC is naturally satisfied.
- **`R1T1`** (Neumann, `u' = 0`) — use a function whose *first derivative vanishes* at both boundaries (e.g., `cos(π*x/L)` on `[0,L]`).
- **`R1T2`** (zero second derivative) — use a function whose second derivative vanishes at the boundaries.
- **`R2T10`**, **`R2T20`** — use functions satisfying both conditions simultaneously; the vanishing-value `sin(π*x/L)` with ≥2-cell padding works.
- **`R3`** on both sides — removes 6 DOF total, so use ≥6 cells.
- **`PERIODIC`** — use a smooth periodic function (e.g., `sin(x)` on `[0, 2π]`).

---

## Part 3: Spline1D_Grid (SpringsteelGrid) Test Pattern

### 3.1 Grid Creation Tests

**Single variable:**
```julia
gp = SpringsteelGridParameters(
    geometry = "Spline1D",
    iMin = 0.0, iMax = 10.0, num_cells = 10,
    BCL = Dict("u" => CubicBSpline.R0),
    BCR = Dict("u" => CubicBSpline.R0),
    vars = Dict("u" => 1)
)
grid = createGrid(gp)

@test typeof(grid) <: Springsteel.Spline1D_Grid
@test grid.params.iDim == 30          # num_cells * mubar
@test grid.params.b_iDim == 13        # num_cells + 3
@test size(grid.physical, 1) == grid.params.iDim
@test size(grid.spectral, 1) == grid.params.b_iDim
@test size(grid.physical, 2) == 1     # 1 variable
@test size(grid.physical, 3) == 3     # value, di, dii
```

**Multiple variables:** verify `size(grid.physical, 2) == N` and `size(grid.spectral, 2) == N`.

### 3.2 Gridpoints Test

```julia
gridpoints = getGridpoints(grid)
@test length(gridpoints) == grid.params.iDim
@test gridpoints[1] >= gp.iMin
@test gridpoints[end] <= gp.iMax
@test all(diff(gridpoints) .> 0)
```

### 3.3 Round-trip Accuracy Tests

Use PERIODIC BCs and `sin(2π*x/L)`:
```julia
max_error = maximum(abs.(reconstructed .- original))
@test max_error < 1e-4
```

Use `R0` (free boundary, no constraint) BCs and a cubic polynomial (`x^3 - 2x^2 + x + 1`):
```julia
@test max_error < 1e-3   # slightly relaxed — border coefficients are unconstrained, so edge accuracy is lower
```

### 3.4 Derivative Tests

Use PERIODIC BCs with `sin(x)` (30+ cells):
- First derivative: `cos(x)` with tolerance `0.01`
- Second derivative: `-sin(x)` with tolerance `0.01`

### 3.5 Boundary Condition Test

Use `R1T0` (Dirichlet, zero-value) BCs with `sin(π*(x - xmin)/(xmax - xmin))`,
which vanishes at both endpoints by construction.

**Important**: `grid.physical[1]` and `grid.physical[end]` are inner mish-points, not
the domain boundary — they are the first and last Gauss–Legendre quadrature nodes, located
a fraction of `DX` inside the domain. To verify the BC is actually enforced, evaluate the
spline *exactly at* `xmin` and `xmax` using `regularGridTransform`:

```julia
spectralTransform!(grid)
gridTransform!(grid)

# Evaluate AT the boundary endpoints to confirm R1T0 enforcement
boundary_vals = regularGridTransform(grid, [gp.xmin, gp.xmax])
@test abs(boundary_vals[1, 1, 1]) < 1e-10  # u(xmin) = 0 (exactly, by BC)
@test abs(boundary_vals[2, 1, 1]) < 1e-10  # u(xmax) = 0 (exactly, by BC)
```

Do **not** use `R0` (free boundary, no constraint) for this test — `R0` imposes no
condition on endpoint values, so boundary zeros would only hold accidentally for
functions that happen to vanish there.

For `R1T1` (Neumann, zero first derivative), use `cos(π*(x-xmin)/(xmax-xmin))` and
verify that the first derivative at the boundary is near zero:

```julia
boundary_derivs = regularGridTransform(grid_r1t1, [gp.xmin, gp.xmax])
@test abs(boundary_derivs[1, 1, 2]) < 1e-6  # u'(xmin) ≈ 0
@test abs(boundary_derivs[2, 1, 2]) < 1e-6  # u'(xmax) ≈ 0
```

### 3.6 Variable-Specific Filter Length Test

Provide `l_q = Dict("u" => 1.5, "v" => 3.0)` and verify:
```julia
@test grid.splines[1, 1].params.l_q == 1.5
@test grid.splines[1, 2].params.l_q == 3.0
```

### 3.7 Regular Grid Transform Tests

Use a Gaussian `u = exp(-(x/σ)²)` centered in the domain with compact support well inside boundaries:
- `regularGridTransform(grid, reg_pts)` should produce values, first, and second derivatives matching the analytic Gaussian.
- Tolerances: values `< 1e-5`, first derivatives `< 1e-4`, second derivatives `< 1e-4` (with ≥100 cells and σ wide relative to DX).
- Also test `getRegularGridpoints(grid)` returns the correct number, matches domain extents, and is monotonically increasing.

### 3.8 Patch-to-Tile Transform Tests

The simplest derivation-independent approach: fit a Gaussian to the full patch, then evaluate at tile gridpoints. Tile should be a strict sub-domain of the patch.

```julia
# patch: iMin=-50, iMax=50, 100 cells
# tile:  iMin=-50, iMax=0,  50 cells (left half)
gridTransform!(patch, tile)
# verify values, first + second derivatives vs analytic
```

### 3.9 Distributed Computing Infrastructure Tests

These test the internal machinery for domain decomposition. Use a Gaussian on a patch (100 cells) and a tile (50-cell right-half sub-domain).

**calcTileSizes:**
- Test with 2 and 3 tiles: verify `iMin` of first tile = patch `iMin`, `iMax` of last tile = patch `iMax`
- `sum(num_cells) == patch.num_cells`
- `spectralIndexL[1] == 1`, `spectralIndexL[2] == num_cells[1] + 1`
- `sum(tile_sizes) == patch.iDim`
- Too many tiles should throw `DomainError`

**gridTransform! (5-arg patchSplines variant):**
```julia
gridTransform!(patch.splines, patch.spectral, patch.params, tile, splineBuffer)
```
Verify values and derivatives against analytic Gaussian.

**splineTransform! / tileTransform! workflow:**
```julia
sharedB = SharedArray{Float64}(size(patch.spectral))
sharedB[:,:] .= patch.spectral          # fill shared array with B coefficients
patchA = zeros(size(patch.spectral))
splineTransform!(patch.splines, patchA, patch.params, sharedB, tile)  # B→A
splineBuffer = allocateSplineBuffer(patch, tile)
tileTransform!(patch.splines, patchA, patch.params, tile, splineBuffer)  # A→physical
# verify tile.physical against analytic Gaussian
```

**sumSpectralTile:**
- Fill `tile.spectral` with `collect(1.0:b_iDim)`
- Call `sumSpectralTile(buf, tile.spectral, siL, siR)` → verify values copied to `buf[siL:siR]`
- Call again → values doubled (accumulation)
- Call `sumSpectralTile!(patch, tile)` → verify `patch.spectral[siL:siR]` set correctly

**setSpectralTile:**
- Call `setSpectralTile(ps, patch.params, tile)` → verify buffer is zeroed, then tile values written at `[siL:siR]`
- Note: `setSpectralTile!` has a known call-signature mismatch → `@test_throws MethodError`

**getBorderSpectral:**
- Border indices: `biL = spectralIndexR - 2`, `biR = spectralIndexR`
- Tool indices in tile: `tiL = b_iDim - 2`, `tiR = b_iDim`
- Verify `nnz(border) == 3` (exactly 3 non-zero entries for 1 variable)

**calcPatchMap:**
- Inner region: `siL = spectralIndexL`, `siR = spectralIndexR - 3`
- Verify `all(patchMap[siL:siR, :])`, `!any(patchMap[siR+1:end, :])`
- `length(tileView) == siR - siL + 1`

**calcHaloMap:**
- Halo region: `hiL = spectralIndexR - 2`, `hiR = spectralIndexR` (3 rows)
- Verify `all(haloMap[hiL:hiR, :])`, `count(haloMap) == 3`, `length(haloView) == 3`

**sumSharedSpectral:**
- Build `borderSpectral` via `getBorderSpectral`
- After `sumSharedSpectral(sharedSpectral, borderSpectral, pp, tile)`:
  - `sharedSpectral[siL:siR]` matches `tile.spectral[1:(siR-siL+1)]`
  - All values are finite

**num_columns / allocateSplineBuffer:**
```julia
@test Springsteel.num_columns(grid) == 0         # 1D grid has no columns
buf = allocateSplineBuffer(grid, grid)
@test isa(buf, Array)                             # trivial but returns an array
```

---

## Part 4: Functions NOT Requiring Tests

These functions may be skipped or only tested indirectly:

| Function | Reason |
|----------|--------|
| `calcGammaBC` | Internal; covered by `Spline1D` construction + any SA transform |
| `calcPQfactor` | Internal; covered by `Spline1D` construction |
| `calcMishPoints` | Internal; mish points verified in construction test |
| `spectralxTransform` | Body is empty (`# Not implemented`) |
| `spectralTransform` (2-arg, non-`!`) | Called internally by `spectralTransform!`; covered by round-trip |
| `gridTransform` (2-arg, non-`!`) | Called internally by `gridTransform!`; covered by derivative tests |

---

## Part 5: Test Function Reference — Analytic Test Cases

Use these analytic test cases consistently across all basis modules:

### Round-trip accuracy (tolerances scale with resolution)
| Test function | BC type | Expected max error |
|---------------|---------|--------------------|
| `sin(2π*x/L)` | `PERIODIC` | `< 1e-4` with 20 cells |
| `x^3 - 2x^2 + x + 1` | `R0` (free boundary, no constraint) | `< 1e-3` with 10 cells |

### Derivative accuracy
| Test function | Derivatives | BC type | Tolerance |
|---------------|-------------|---------|-----------|
| `sin(x)` on `[0, 2π]` | `cos(x)`, `-sin(x)` | PERIODIC | `< 0.01` with 30 cells |

### Regular grid / patch-tile accuracy (high-precision)
| Test function | Domain | Cells | Tolerance |
|---------------|--------|-------|-----------|
| Gaussian `exp(-(x/σ)²)`, σ=20 | `[-50, 50]` | 100 | values `< 1e-5`, derivatives `< 1e-4` |

The Gaussian is ideal because it has analytic derivatives of all orders, is non-trivial, and with σ=20 on a 100-cell domain gives high accuracy without being resolution-sensitive.

---

## Part 6: Replication Instructions for Other Basis Functions

### For `Fourier.jl` / `rl_grid.jl`

1. **Module tests** (`Fourier Tests` testset):
   - `FourierParameters` auto-computed fields (e.g., `DX`, `DXrecip`, `kmax`)
   - `Fourier1D` construction: `mishDim`, `bDim` (2 * `kmax` + 1 modes), allocated arrays, ring points monotonically increasing
   - `FBtransform` variants: `FourierParameters`, `Fourier1D`, in-place `FBtransform!`
   - `FAtransform` variants: standard, in-place `FAtransform!`
   - `FItransform` variants: all dispatch flavors (same list as Section 2.7)
   - `FIxtransform` and `FIxxtransform`: all variants with `sin(x)` test
   - BC type: `PERIODIC` only

2. **Grid tests** (`RL_Grid Tests` testset):
   - Grid creation: verify `lDim` auto-calculated, `b_lDim` matches Fourier mode count
   - Note: `lDim` varies by radius (it is computed, not set). Use `l_regular_out = rDim*2 + 1` for output
   - Multi-variable, getGridpoints, round-trip, derivatives, kmax per variable
   - Tile tests: `calcTileSizes`, patch-to-tile, splineTransform/tileTransform, spectral index maps
   - `num_columns` should return 0 for 2D RL grid

### For `Chebyshev.jl` / `rz_grid.jl`

1. **Module tests** (`Chebyshev Tests` testset):
   - `ChebyshevParameters` struct fields
   - `Chebyshev1D` construction
   - `CBtransform`, `CAtransform!`, `CItransform`, `CIxtransform`, `CIxxtransform`, `CIInttransform` (integration transform)
   - BC types: `R0` (free boundary), `R1T0` (Dirichlet, zero value), `R1T1` (Neumann, zero first derivative), `R1T2` (zero second derivative), `R2T10`, `R2T20`, `R3`

2. **Grid tests** (`RZ_Grid Tests` testset):
   - Grid creation: both radial (B-spline) and vertical (Chebyshev) dimensions
   - `physical` array has 3rd dimension = 5 (value, ∂/∂r, ∂²/∂r², ∂/∂z, ∂²/∂z²); physical indexing is r-outer/z-inner: `flat = (r-1)*zDim + z`
   - Round-trip with 2D function `sin(r) * cos(z)`
   - Derivative tests in both r and z directions
   - `num_columns` returns `zDim` (number of Chebyshev columns)

### For `rlz_grid.jl` (3D RLZ)

Combine the 1D tests for all three directions. The physical array 3rd dimension encodes all mixed partial derivatives. Test after a 3D round-trip that:
- Pure radial derivatives match analytic
- Pure azimuthal derivatives match analytic
- Pure vertical derivatives match analytic

---

## Part 7: Test Quality Checklist

Before finalizing a new test suite, verify:

- [ ] Every exported function has at least one direct `@test`
- [ ] Every dispatch variant (different call signatures) is explicitly exercised
- [ ] In-place (`!`) variants are tested separately from non-mutating variants
- [ ] At least one known-answer test (analytic function with exact derivatives) per transform chain
- [ ] All BC types are enumerated and constructed without errors
- [ ] Distributed computing functions (`calcTileSizes`, `calcPatchMap`, `calcHaloMap`, spectral tile assembly) have dedicated tests
- [ ] Known bugs / unimplemented features are documented with `@test_throws` or a comment, not silently skipped
- [ ] Tolerances are documented alongside the number of cells used, so future resolution changes don't silently break tests
- [ ] `num_columns` and `allocateSplineBuffer` are tested for every grid type even if trivial
---

## Part 8: Solver Test Patterns (`test/solver.jl`)

Solver tests live in `test/solver.jl` and cover the operator matrix assembly,
`SpringsteelProblem` type, and `solve()` function. This section provides patterns
for agents implementing solver steps.

### 8.1 Matrix Representation Tests (Basis Module Level)

Matrix representation tests for individual basis modules go in `test/basis.jl`
(not `test/solver.jl`) since they test basis-module internals:

```julia
@testset "Fourier matrix representations" begin
    fp = FourierParameters(ymin=0.0, yDim=32, bDim=11, kmax=5)
    ring = Fourier1D(fp)
    M = Fourier.dft_matrix(ring)
    Mx = Fourier.dft_1st_derivative(ring)
    Mxx = Fourier.dft_2nd_derivative(ring)

    @test size(M) == (fp.yDim, fp.bDim)
    @test size(Mx) == (fp.yDim, fp.bDim)
    @test size(Mxx) == (fp.yDim, fp.bDim)

    # Consistency: M * a ≈ FItransform for a known signal
    ring.uMish .= sin.(ring.mishPoints)
    FBtransform!(ring)
    FAtransform!(ring)
    u_inv = ring.fftPlan * ring.a     # manual inverse
    @test M * ring.a[1:fp.bDim] ≈ u_inv atol=1e-10

    # Derivative: d/dx sin(x) = cos(x)
    cos_analytic = cos.(ring.mishPoints)
    @test Mx * ring.a[1:fp.bDim] ≈ cos_analytic atol=1e-8
end
```

### 8.2 Operator Assembly Tests (Solver Level)

Operator assembly tests go in `test/solver.jl`:

```julia
@testset "Operator Matrix Assembly" begin
    @testset "1D operator_matrix" begin
        gp = SpringsteelGridParameters(
            geometry = "R", num_cells = 10,
            iMin = 0.0, iMax = 1.0,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0))
        grid = createGrid(gp)

        M0 = operator_matrix(grid, :i, 0)    # evaluation matrix
        M2 = operator_matrix(grid, :i, 2)    # 2nd derivative
        @test size(M0, 1) == grid.params.iDim
        @test size(M0, 2) == grid.params.b_iDim
    end

    @testset "2D Kronecker assembly" begin
        gp = SpringsteelGridParameters(
            geometry = "RZ", num_cells = 5,
            iMin = 0.0, iMax = 1.0,
            kMin = 0.0, kMax = 1.0, kDim = 10, b_kDim = 10,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0),
            BCB = Dict("u" => Chebyshev.R1T0),
            BCT = Dict("u" => Chebyshev.R1T0))
        grid = createGrid(gp)

        # Laplacian: ∂²/∂r² ⊗ I_z + I_r ⊗ ∂²/∂z²
        terms = [
            OperatorTerm(2, 0, 0, 1.0),   # ∂²/∂i²
            OperatorTerm(0, 0, 2, 1.0),   # ∂²/∂k²
        ]
        L = assemble_operator(grid, terms, "u")
        n_phys = grid.params.iDim * grid.params.kDim
        n_spec = grid.params.b_iDim * grid.params.b_kDim
        @test size(L) == (n_phys, n_spec)
    end
end
```

### 8.3 Linear Solver Tests

Linear solver tests verify known analytic solutions:

```julia
@testset "Linear Solver" begin
    @testset "1D Chebyshev Poisson" begin
        # Solve u'' = f where u(x) = sin(πx), f = -π²sin(πx)
        # Domain [0, 1], Dirichlet BCs u(0) = u(1) = 0
        gp = SpringsteelGridParameters(
            geometry = "Z", kMin = 0.0, kMax = 1.0,
            kDim = 25, b_kDim = 25,
            vars = Dict("u" => 1),
            BCB = Dict("u" => Chebyshev.R1T0),
            BCT = Dict("u" => Chebyshev.R1T0))
        grid = createGrid(gp)
        pts = getGridpoints(grid)

        # Assemble operator and RHS
        L = assemble_operator(grid, [OperatorTerm(0, 0, 2, 1.0)], "u")
        f = -π^2 .* sin.(π .* pts)

        prob = SpringsteelProblem(grid; operator=L, rhs=f)
        sol = solve(prob)

        u_analytic = sin.(π .* pts)
        @test sol.converged == true
        @test maximum(abs.(sol.physical .- u_analytic)) < 1e-8
    end
end
```

### 8.4 Multi-dimensional Solver Tests

For 2D and 3D problems, use separable analytic solutions:

```julia
@testset "2D Poisson on RZ_Grid" begin
    # u(r,z) = sin(πr) * sin(πz), ∇²u = -2π²sin(πr)sin(πz)
    # Dirichlet BCs on all boundaries
    ...
    @test maximum(abs.(sol.physical .- u_analytic)) < 1e-6
end
```

**Key principle**: Always use functions whose exact solution is known analytically.
Verify spectral convergence by testing at two resolutions when feasible.

### 8.5 Running Solver Tests During Development

For fast iteration when implementing solver steps:

```bash
# Run ONLY solver tests (fastest feedback loop)
cd /path/to/Springsteel.jl
julia --project -e '
    using Springsteel, Test
    @testset "Solver dev" begin
        include("test/solver.jl")
    end
'

# Run solver + basis tests (for matrix representation work)
julia --project -e '
    ENV["TEST_GROUP"] = "solver"
    using Pkg; Pkg.test()
'

# Full suite (final verification before committing)
julia --project -e 'using Pkg; Pkg.test()'
```

### 8.6 Solver Test Quality Checklist

- [ ] Every solver type (`SpringsteelProblem`, `SpringsteelSolution`, backend sentinels) has a construction test
- [ ] `solve()` is tested with at least one analytic solution per basis type (Spline, Fourier, Chebyshev)
- [ ] Multi-dimensional operator assembly is tested for at least one 2D grid
- [ ] `assemble_from_equation` results match manual `assemble_operator` results
- [ ] Error cases are tested (`@test_throws` for singular operators, missing fields, etc.)
- [ ] Cached factorisation is verified (solve twice with different RHS, check both correct)
- [ ] If Optimization.jl tests exist, they are conditionally loaded:
  ```julia
  if isdefined(Main, :Optimization)
      @testset "Optimization.jl backend" begin ... end
  end
  ```
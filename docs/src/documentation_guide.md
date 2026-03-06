# Springsteel.jl Documentation Guide

## Overview

This document describes the documentation strategy established for `CubicBSpline.jl` and
`spline1D_grid.jl`, and provides step-by-step instructions for replicating similarly
complete documentation for any other basis function module (`Fourier.jl`, `Chebyshev.jl`)
and its associated grid file (`rl_grid.jl`, `rz_grid.jl`, `rlz_grid.jl`).

The documentation uses Julia's [Documenter.jl](https://documenter.juliadocs.org/) standard
library. Every public symbol should have a docstring; internal helpers should have docstrings
too if they contain non-obvious math or multiple dispatch variants.

See `CubicBSpline.jl` and `spline1D_grid.jl` for the canonical reference implementation.

---

## Part 1: Docstring Format Conventions

### 1.1 Standard Docstring Template

Every function docstring follows this layout (omit sections that don't apply):

```julia
"""
    FunctionName(arg1::Type1, arg2::Type2) -> ReturnType
    FunctionName(arg1::Type1, arg2::Type2, arg3::Type3) -> ReturnType

One-sentence summary of what the function does.

Longer explanation if the math, algorithm, or behaviour needs it. Use KaTeX-compatible
LaTeX with double-backslash escaping for inline math:
``u(x) = \\sum_m a_m \\varphi_m(x)``

# Variants (only for multi-dispatch families)
- `FunctionName(arg1, arg2)` — what this variant does
- `FunctionName(arg1, arg2, arg3)` — what this other variant does

# Arguments
- `arg1::Type1`: Description (units if applicable)
- `arg2::Type2`: Description. Special values: `0` means …, `−1` means …

# Returns
- `ReturnType`: Description of what is returned. Note any in-place variants.

# Notes (optional — for pitfalls, performance advice, or implementation details)
- Note 1
- Note 2

# Example (required for public structs and major entry-point functions)
```julia
obj = MyType(param = value)
result = myFunction(obj, data)
```

See also: [`RelatedFunction`](@ref), [`OtherType`](@ref)
"""
function FunctionName(...)
```

**Key rules:**

- Multi-signature banners: list all dispatch variants in the top banner when they share the
  same logical purpose (e.g., all `SBtransform` overloads share one docstring).
- The `!` (in-place) variant **must** appear in the banner alongside the allocating form.
  Never write separate docstrings for the only difference is mutability.
- `@ref` cross-links: only link to symbols that have their own docstring. Using `@ref` on
  an undocumented symbol will fail the `makedocs` cross-reference check.
- LaTeX: wrap inline math in double backtick `\`\`` ... `\`\`` (Documenter renders these as
  KaTeX). For display math use a block with `\`\`math` ... `\`\``.
- BC constants: a one-line docstring is sufficient. Place it immediately before the `const`
  definition using `"""..."""` on a single line.

### 1.2 Struct Docstrings

Structs (both parameter structs and composite data structs) need a `@kwdef` docstring that
covers all fields. Follow this pattern, matching the `SplineParameters` / `CubicBSpline.Spline1D`
examples in `CubicBSpline.jl`:

```julia
"""
    MyParameters

Brief description of what this struct configures.

# Fields
- `field1::Type`: What it controls. Auto-computed: `field2 / field3`
- `field2::Type`: ...
- `BCL::Dict`: Left boundary condition dict. One of [`R0`](@ref), [`R1T0`](@ref), ...

# Example
```julia
p = MyParameters(xmin=0.0, xmax=10.0, num_cells=20)
```

See also: [`MyObject`](@ref)
"""
Base.@kwdef struct MyParameters
    ...
end
```

---

## Part 2: Documenting Fourier.jl

### 2.1 Boundary Condition Constant

`Fourier.jl` exports only one BC constant: `PERIODIC`. Give it a one-line docstring:

```julia
"""Periodic boundary condition: the only valid BC for Fourier basis functions."""
const PERIODIC = Dict("PERIODIC" => 0)
```

### 2.2 FourierParameters Struct

`FourierParameters` uses `@kwdef` with four fields. Key documentation points:
- `ymin` is the **azimuthal offset** (angle in radians) of the first grid point, not a
  domain boundary in the usual sense. Values other than `0.0` invoke the phase-shift filter.
- `yDim` is the total number of physical points around the ring. It must be even for the
  real-to-halfcomplex FFT (`FFTW.R2HC`).
- `bDim` is the number of Fourier B-coefficients after filtering out wavenumbers above
  `kmax`. The formula is `bDim = 2*kmax + 1` (one constant term plus sin+cos per wavenumber).
- `kmax` sets the maximum retained wavenumber. Wavenumbers above `kmax` are zeroed by the
  phase-filter matrix.

Unlike `SplineParameters`, `FourierParameters` has **no auto-computed fields**. Its fields
are all set explicitly at construction time.

### 2.3 Fourier1D Struct

`Fourier1D` has more internal state than `Spline1D` because it stores pre-measured FFTW
plans. Key documentation points:

| Field | Type | Notes |
|-------|------|-------|
| `mishPoints` | `Vector{Float64}` | Evenly-spaced angles in `[ymin, ymin + 2π)` |
| `fftPlan` | `FFTW.r2rFFTWPlan` | Pre-measured real-to-halfcomplex plan; **do not serialise** |
| `ifftPlan` | `FFTW.r2rFFTWPlan` | Pre-measured halfcomplex-to-real plan |
| `phasefilter` | `Matrix{Float64}` | Size `(yDim, bDim)`; shifts phase and zeroes wavenumbers > kmax |
| `invphasefilter` | `Matrix{Float64}` | Size `(bDim, yDim)`; inverse phase shift + zero-pad to yDim |
| `uMish` | `Vector{Float64}` | Physical values (length `yDim`) |
| `b` | `Vector{Float64}` | Filtered B-coefficients (length `bDim`) |
| `a` | `Vector{Float64}` | Zero-padded A-coefficients ready for inverse FFT (length `yDim`) |
| `ax` | `Vector{Float64}` | Working buffer for derivative/integral coefficients (length `yDim`) |

Mention in the docstring that constructing `Fourier1D` calls `FFTW.plan_r2r` with
`FFTW.PATIENT` (slower first construction, faster subsequent transforms), and that the
resulting struct is **not thread-safe** if `uMish`, `b`, `a`, `ax` are mutated concurrently.

### 2.4 Internal Setup Functions

These are not exported but should still have docstrings (they are referenced from the struct
docstring and appear in `@autodocs`):

**`calcMishPoints(fp::FourierParameters)`**
- Points are `ymin + 2π*(n-1)/yDim` for `n = 1:yDim`.
- The ring is **half-open**: the last point is one step before `ymin + 2π`, so it does not
  repeat the starting point when the ring closes.

**`calcPhaseFilter(fp::FourierParameters)`**
- Returns a `(yDim, bDim)` matrix. Column `k+1` aligns cosine and sine components of
  wavenumber `k` to a common phase reference at `ymin = 0`. Wavenumbers above `kmax` are
  simply not present (their columns are absent).
- Document the block structure: the `(1,1)` element handles wavenumber 0; for each
  `k = 1:kmax` a 2×2 rotation block is placed at rows `k+1, yDim-k+1` and columns `k+1, bDim-k+1`.

**`calcInvPhaseFilter(fp::FourierParameters)`**
- Inverse (transpose-of-rotation) of `phasefilter`. Size `(bDim, yDim)`.
- Also performs zero-padding: maps the `bDim` filtered coefficients back to `yDim`-length
  array for the inverse FFT.

### 2.5 Transform Function Families

Fourier has three transform stages (FB → FA → FI) analogous to SB → SA → SI in CubicBSpline:

**`FBtransform` — physical → B-coefficients**

```julia
"""
    FBtransform(fp::FourierParameters, fftPlan, phasefilter, uMish) -> Vector{Float64}
    FBtransform!(ring::Fourier1D)

Compute the forward Fourier transform (physical → filtered B-coefficients).
...
"""
```

- Document the normalisation: output of `FFTW.R2HC` is divided by `yDim` so that
  amplitude 1 in physical space maps to amplitude 1 in `b`.
- Note that the `FourierParameters` variant requires explicit `fftPlan` and `phasefilter`
  objects; the `Fourier1D` variant uses the cached copies in the struct.

**`FAtransform` / `FAtransform!` — B-coefficients → padded A-coefficients**

- Applies the inverse phase filter: `a = invphasefilter * b` (after un-batching).
- The result `a` is `yDim`-length (zero-padded beyond `bDim`) and ready for the inverse FFT.

**`FItransform` / `FItransform!` — A-coefficients → physical values**

- Performs the `FFTW.HC2R` inverse transform.
- **No scaling needed here** (the forward scaling in `FBtransform` handles it).

**`FIxtransform` — first derivative in physical space**

Unlike the CubicBSpline approach (which differentiates basis functions directly), the Fourier
derivative is computed analytically in spectral space:

```
a'[k+1]     = −k · a_s[k]     (cosine component of d/dθ)
a'[yDim-k+1] = +k · a_c[k]   (sine component of d/dθ)
```

where `a_c[k]` and `a_s[k]` are the cosine and sine coefficients at wavenumber `k`,
stored at indices `k+1` and `yDim-k+1` in the half-complex layout.

Document all three dispatch variants:
- `FIxtransform(fp, ifftPlan, a, ax)` — allocates; uses `FIxcoefficients` then inverse FFT
- `FIxtransform(ring::Fourier1D)` — allocates new output vector
- `FIxtransform(ring::Fourier1D, ux::AbstractVector)` — in-place into `ux`

**`FIxxtransform` — second derivative** 

Applies `FIxcoefficients` twice (once to get `a'`, then again to get `a''`) and performs
one inverse FFT. Document the `copy` of `ax` needed to avoid overwriting the buffer mid-computation.

**`FIInttransform` / `FIIntcoefficients` — indefinite integral**

The Fourier module also provides an **integral** transform (CubicBSpline does not).
The constant of integration `C0` sets the mean:
- `aInt[k+1]   =  a_s[k] / k`  (integral of sine component)
- `aInt[yDim-k+1] = -a_c[k] / k` (integral of cosine component)

Document the two variants of `FIInttransform`: one taking raw parameters + plan, one taking
`ring::Fourier1D` with an optional `C0::Float64 = 0.0` keyword argument.

### 2.6 Internal Helper `FIxcoefficients`

Not exported but used by both `FIxtransform` and `FIxxtransform`. Docstring should explain
the half-complex index layout and the in-place mutation of `ax`.

---

## Part 3: Documenting Chebyshev.jl

### 3.1 Boundary Condition Constants

Chebyshev has seven BC constants. The naming follows the Ooyama (2002) rank/type
convention used by `CubicBSpline`, with identical physical meanings:

| Constant | Dict key(s) | Rank | Physical meaning |
|----------|-------------|:----:|------------------|
| `R0` | `"R0"` | 0 | No constraint (free boundary) |
| `R1T0` | `"α0"` | 1 | Zero field value at boundary, ``u(z_0) = 0`` (Dirichlet) |
| `R1T1` | `"α1"` | 1 | Zero first derivative, ``u'(z_0) = 0`` (Neumann); uses Wang et al. (1993) method |
| `R1T2` | `"α2"` | 1 | Zero second derivative, ``u''(z_0) = 0`` |
| `R2T10` | `"β1"`, `"β2"` | 2 | Zero value and zero first derivative, ``u = u' = 0`` |
| `R2T20` | `"β1"`, `"β2"` | 2 | Zero value and zero second derivative, ``u = u'' = 0`` |
| `R3` | `"R3"` | 3 | Zero value, first, and second derivative, ``u = u' = u'' = 0`` |

Write one-line docstrings for each, keeping the same style as the Fourier `PERIODIC` example.

### 3.2 ChebyshevParameters Struct

Six fields, all explicitly set (no auto-computation):

| Field | Description |
|-------|-------------|
| `zmin` | Bottom of the vertical domain (e.g. 0 m or surface pressure level) |
| `zmax` | Top of the vertical domain |
| `zDim` | Number of Chebyshev–Gauss–Lobatto (CGL) nodes |
| `bDim` | Number of retained spectral modes (≤ `zDim`; if `< zDim` a sharp Eresman filter is applied) |
| `BCB` | Bottom boundary condition dict |
| `BCT` | Top boundary condition dict |

Document the CGL point distribution: the domain `[zmin, zmax]` is mapped to `[−1, 1]`
via `z = cos((n−1)π/(zDim−1))`, so points are clustered near both endpoints. Contrast with
the uniform spacing of Fourier mish points.

Also document the filter behaviour: when `bDim == zDim`, a spectral damping filter
`exp(−36 * (k/N)^36)` is applied to suppress aliasing; when `bDim < zDim`, a sharp
truncation matrix is used instead.

### 3.3 Chebyshev1D Struct

Key differences from `Spline1D`:
- `gammaBC` is **polymorphic**: it is a `Vector{Float64}` for simple `R0` BCs and a
  `Matrix{Float64}` for Neumann/Robin BCs. The `CAtransform` functions handle both cases.
  (The `Array{real}` type annotation in the struct covers both.)
- `filter` is a `Matrix{Float64}` that both truncates to `bDim` and applies the exponential
  spectral damping.
- `a` and `ax` are **`zDim`-length** (not `bDim`-length) because the inverse FFT step
  always works on the full Chebyshev grid.

Include a note about the DCT convention: `FFTW.REDFT00` (DCT-I) is used, which is symmetric
around both endpoints. This requires that the physical grid includes both endpoints (unlike
the half-open Fourier ring).

### 3.4 Internal Setup Functions

**`calcMishPoints(cp::ChebyshevParameters)`**
- Returns CGL points ordered **from `zmin` up to `zmax`** (increasing z, bottom-to-top)
  because `scale = -0.5*(zmax-zmin)` (negative) flips the cosine mapping so that
  `cos(0) = 1` → `zmin` (bottom) and `cos(π) = -1` → `zmax` (top). The returned vector
  is therefore already in natural bottom-to-top order and does **not** need to be reversed.

**`calcFilterMatrix(cp::ChebyshevParameters)`**
- Two branches: truncation (`bDim < zDim`) or spectral damping (`bDim == zDim`).
- Document the Eresman damping formula: `filter[i,i] = exp(−36*(i/N)^36)`, which leaves
  low modes unchanged but strongly damps modes near the Nyquist limit.

**`calcGammaBC(cp::ChebyshevParameters)`**
- The most complex setup function. Returns a `Vector` (for R0/R0), a rank-1 Matrix, or a
  full `N×N` Matrix depending on BC combination. Document each branch:
  - `R0/R0` → zero vector (no correction needed)
  - `R1T0/R0` → column vector using the global affine method
  - `R1T1/R0` and `R0/R1T1` → full matrix using Wang et al. (1993)
  - Other combinations analogously
- Reference Wang, H., Lacroix, S., & Labrosse, G. (1993), JCP 109, 133. DOI 10.1006/jcph.1993.1133.

**`calcGammaBCalt`**: Document as a deprecated/alternative Dirichlet-only implementation.
Note in the docstring that it does not handle Neumann BCs and is retained for reference only.

**`dct_matrix`, `dct_1st_derivative`, `dct_2nd_derivative`**
- Matrix representations of the DCT and its derivatives. Useful for debugging and linear
  solvers. Small standalone docstrings suffice; reference their use in testing.

### 3.5 Transform Function Families

**`CBtransform` — physical → B-coefficients**

Three variants:
- `CBtransform(cp, fftPlan, uMish)` — allocates; DCT + division by `2*(zDim-1)` + truncation
- `CBtransform!(column)` — in-place; DCT + filter matrix applied to `column.b`
- `CBtransform(column, uMish)` — allocates but uses the struct's stored filter

Document the DCT-I normalisation: the `2*(zDim-1)` divisor arises from the DCT-I
convention where endpoint values are counted once (not twice) in the discrete orthogonality
relation.

**`CAtransform` / `CAtransform!` — B-coefficients → A-coefficients with BCs**

The BC application is a **spectral correction**: `a = bfill + gammaBC' * bfill`, where
`bfill` is `b` zero-padded from `bDim` to `zDim`. This additive structure means that
`gammaBC = 0` (the `R0/R0` case) means no correction is applied. Document the polymorphism:
the `gammaBC' * bfill` product is valid whether `gammaBC` is a scalar zero vector
(dot-product broadcasts to zero) or a full matrix, because Julia's `*` handles both.

**`CItransform` / `CItransform!` — A-coefficients → physical values**

Simple inverse DCT-I via `fftPlan * a`. Note that no rescaling is needed here.

**`CIxtransform` — first derivative**

Computed via the Chebyshev recurrence:
```
a'[k-1] = 2(k-1)*a[k] + a'[k+1]    (k = zDim → 2, backwards)
a'[0] and a'[1] handled separately
```
then scaled by `−1 / (0.5*(zmax − zmin))` to map from the reference domain `[−1,1]` to the
physical domain. Document both variants:
- `CIxtransform(cp, fftPlan, a, ax)` — uses `CIxcoefficients` then DCT; `ax` is overwritten
- `CIxtransform(column::Chebyshev1D)` — single-dispatch convenience form

**`CIxxtransform` — second derivative**

Applies `CIxcoefficients` twice (once to `a`, once to `a'`). Document the `copy` of the
intermediate derivative coefficients to avoid aliasing with the `ax` buffer.

**`CIInttransform` / `CIIntcoefficients` — indefinite integral**

Chebyshev integration via the recurrence:
```
aInt[k] = 0.5*(zmax − zmin)/2 * (a[k-1] − a[k+1]) / (k-1)    (k = 2:zDim-1)
aInt[1]  = C0 − 2 * sum(aInt[2:end])
```
Document `C0::Float64 = 0.0` as the constant of integration (value of the integral at the
top boundary). Reference `CIInttransform` in the docs for `FIInttransform` to show the
consistent integral-transform pattern across modules.

---

## Part 4: Documenting RL_Grid (rl_grid.jl)

### 4.1 Struct Docstring

`RL_Grid` stores **three** splines per variable (not one):

| Index | Purpose |
|-------|---------|
| `splines[1, v]` | Wavenumber 0 (monopole); azimuthally symmetric |
| `splines[2, v]` | Cosine components; radial BCs may differ from wavenumber 0 |
| `splines[3, v]` | Sine components; radial BCs same as splines[2] |

The `rings` array has shape `(rDim, nvars)` — one `Fourier1D` ring per radial gridpoint
per variable, because `kmax` and `yDim` vary with radius (larger rings have more points).

The `physical` array has a **5-element third dimension**:
- `[:, v, 1]` — field values
- `[:, v, 2]` — radial derivative ∂f/∂r
- `[:, v, 3]` — second radial derivative ∂²f/∂r²
- `[:, v, 4]` — azimuthal derivative ∂f/∂λ
- `[:, v, 5]` — second azimuthal derivative ∂²f/∂λ²

Note that the stored azimuthal derivatives are the **raw** Fourier derivatives ∂f/∂λ and
∂²f/∂λ², *not* the geometric (1/r) ∂f/∂λ counterparts. Any 1/r scaling required for
the physical equations must be applied by the caller.

The spectral array is **1D** (`b_lDim × nvars`) because the combined radial-Fourier spectrum
is stored in a flattened layout. Document `b_lDim` as the total number of spectral coefficients
across all radial gridpoints and Fourier modes.

### 4.2 `create_RL_Grid` Docstring

Critical implementation notes to include:
1. `lDim` and `b_lDim` are **computed** inside the constructor by iterating over all radial
   gridpoints and summing ring sizes. They cannot be specified in advance and are not in
   `GridParameters`. The `GridParameters` struct is rebult `gp2` with the computed values inserted.
2. `kmax` default of `−1` triggers the ring-specific maximum: `kmax = ri` where `ri` is the
   1-based radial index offset by `patchOffsetL`. Override with `kmax = Dict("u" => 10)`.
3. Three `Spline1D` objects are created even though they share the same BC. This is because
   the wavenumber-0 wind components have different physical definitions at r=0 compared to
   higher modes.

### 4.3 Key Differences from Spline1D_Grid

Document these differences prominently, either in the struct docstring or a top-of-file comment:

| Aspect | Spline1D_Grid | RL_Grid |
|--------|--------------|---------|
| Basis functions | Spline only | Spline (radial) + Fourier (azimuthal) |
| physical 3rd dimension | 3 (value, d, dd) | 5 (value, dr, dλ, drr, dλλ) |
| Splines per variable | 1 | 3 (wavenumber 0 + cos + sin) |
| Ring count | 0 (no rings) | `rDim` rings per variable |
| lDim | set by user | auto-computed in constructor |
| Spectral layout | simple `(b_iDim, nvars)` | flattened `(b_lDim, nvars)` |

### 4.4 Transform Functions

Follow the same docstring patterns as `spline1D_grid.jl`. Note the additional complexity:
- `spectralTransform!` must iterate over rings (doing `FBtransform!`), then iterate over
  radii (doing `SBtransform!` on the aggregated Fourier modes).
- `gridTransform!` runs `SAtransform!` then evaluates at all ring points via `FAtransform!`
  and `FIxtransform`, `FIxxtransform`.
- Document the spectral index arithmetic: `b_lDim` entries are laid out in a specific order
  (wavenumber 0 splines first, then cos/sin splines interleaved). Show the index formula.

---

## Part 5: Documenting RZ_Grid (rz_grid.jl)

### 5.1 Struct Docstring

`RZ_Grid` is a 2D radius-height grid:
- `splines` has shape `(b_zDim, nvars)` — one `Spline1D` per Chebyshev spectral mode per
  variable. These splines operate in the **transformed spectral-vertical space**, not in
  the original physical vertical coordinates.
- `columns` has shape `(nvars,)` — one `Chebyshev1D` per variable.
- `physical` has shape `(rDim * zDim, nvars, 5)` — **r-outer, z-inner** flattened layout
  (index = `(r-1)*zDim + z`) with 5 derivative slots: `[value, ∂f/∂r, ∂²f/∂r², ∂f/∂z, ∂²f/∂z²]`.
- `spectral` has shape `(b_zDim * b_rDim, nvars)` — flattened 2D spectral layout.

### 5.2 `create_RZ_Grid` Docstring

Document the two-loop construction:
1. One `Chebyshev1D` column per variable (uses `zmin, zmax, zDim, b_zDim, BCB, BCT`).
2. `b_zDim` `Spline1D` objects per variable (one per Chebyshev output mode). These all
   share the same radial domain `[xmin, xmax]`, cell count, and boundary conditions.

### 5.3 Key Differences from Spline1D_Grid

| Aspect | Spline1D_Grid | RZ_Grid |
|--------|--------------|---------|
| Dimensions | 1D (radial) | 2D (radial × vertical) |
| Basis | Spline | Spline (r) + Chebyshev (z) |
| physical 3rd dimension | 3 | 5 |
| physical index 1st dim | `iDim` | `rDim * zDim` (flattened) |
| `num_columns` | 0 | `zDim` |

### 5.4 Transform Functions

The 2D transform has two stages in each direction:
- **physical → spectral**: first `CBtransform!` each column (z → Chebyshev), then
  `SBtransform!` each radial spline (r → B-spline) for each Chebyshev mode.
- **spectral → physical**: first `SAtransform!` each radial spline, then `CAtransform!`
  and `CItransform` / `CIxtransform` / `CIxxtransform` to get vertical values + derivatives.

Document `num_columns(grid::RZ_Grid)` — returns `zDim` (number of Chebyshev columns) and
exists for API compatibility with grid-type-agnostic code.

---

## Part 6: Modifying docs/make.jl and docs/src/

### 6.1 Adding a New Module to makedocs

When adding docstrings to a submodule (e.g. `Fourier` or `Chebyshev`), add it to the
`modules` list in `docs/make.jl`:

```julia
makedocs(;
    modules=[Springsteel, Springsteel.CubicBSpline, Springsteel.Fourier, Springsteel.Chebyshev],
    ...
)
```

Failure to include a module means `@docs` blocks referencing its symbols will silently
produce "docstring not found" errors.

### 6.2 Creating a Documentation Page

Create `docs/src/{module_name}.md` following the structure used in `docs/src/cubicbspline.md`:

````markdown
```@meta
CurrentModule = Springsteel
```

# ModuleName

Brief description paragraph.

## Boundary Condition Constants

```@docs
ModuleName.R0
ModuleName.PERIODIC
```

## Parameter and Data Structures

```@docs
ModuleName.ModuleParameters
ModuleName.Module1D
ModuleName.Module1D(::ModuleName.ModuleParameters)
```

## Internal Setup Functions

```@docs
ModuleName.calcMishPoints
ModuleName.calcGammaBC
...
```

## Transform Functions

```@docs
ModuleName.XBtransform
ModuleName.XAtransform
ModuleName.XItransform
ModuleName.XIxtransform
ModuleName.XIxxtransform
```
````

Then add the page to the `pages` list in `make.jl`:

```julia
pages=[
    "Home" => "index.md",
    "CubicBSpline" => "cubicbspline.md",
    "Fourier"      => "fourier.md",
    "Chebyshev"    => "chebyshev.md",
    "Testing Guide" => "testing_guide.md",
],
```

### 6.3 Using @autodocs vs. @docs

- **`@docs`** (explicit list): use for submodule pages where you want to control grouping
  into sections (BC constants, structs, transforms). This is what `cubicbspline.md` uses.
- **`@autodocs`**: use on the home `index.md` to catch all top-level `Springsteel` exports
  automatically as a fallback. It respects `Modules = [...]` filtering.

Do **not** put the same symbol in both `@docs` and `@autodocs` blocks — Documenter will
warn about duplicate docstrings.

### 6.4 Avoiding Cross-Reference Failures

Every `[`SymbolName`](@ref)` in a docstring must resolve to a documented symbol in the
`modules` list. Common failure cases:

| Pattern | Fix |
|---------|-----|
| `SBtransform!` linked with `@ref` but the `!` form shares a docstring with `SBtransform` | Use backtick-only ( `` `SBtransform!(spline)` `` ) instead of `@ref` for the bang variant |
| Cross-module link e.g. `[`CubicBSpline.R0`](@ref)` from inside Fourier docstring | The target module must be in the `modules` list, and the full qualified name must be used |
| Link to a private function | Remove the `@ref`; use plain code style instead |

---

## Part 7: Module-by-Module Function Reference

Use these tables when documenting to ensure no function is missed.

### Fourier.jl

| Function | Exported? | Docstring Priority |
|----------|-----------|--------------------|
| `FourierParameters` | yes | full (struct) |
| `Fourier1D` (struct) | yes | full (struct) |
| `Fourier1D(fp)` (constructor) | yes | full |
| `calcMishPoints` | no | brief |
| `calcPhaseFilter` | no | medium (non-trivial math) |
| `calcInvPhaseFilter` | no | brief |
| `FBtransform`, `FBtransform!` | yes | full (multi-dispatch) |
| `FAtransform`, `FAtransform!` | yes | full |
| `FItransform`, `FItransform!` | yes | full |
| `FIxcoefficients` | no | brief (documents the half-complex index layout) |
| `FIxtransform` (3 variants) | yes | full (multi-dispatch) |
| `FIxxtransform` | yes | full |
| `FIIntcoefficients` | no | brief |
| `FIInttransform` (2 variants) | yes | full |

### Chebyshev.jl

| Function | Exported? | Docstring Priority |
|----------|-----------|--------------------|
| `R0`, `R1T0`, `R1T1`, `R1T2`, `R2T10`, `R2T20`, `R3` | yes (implied) | one-line each |
| `ChebyshevParameters` | yes | full (struct) |
| `Chebyshev1D` (struct) | yes | full (struct) |
| `Chebyshev1D(cp)` (constructor) | yes | full |
| `calcMishPoints` | no | medium (CGL ordering note) |
| `calcFilterMatrix` | no | medium (two-branch documentation) |
| `calcGammaBC` | no | full (most complex helper) |
| `calcGammaBCalt` | no | brief (deprecated note) |
| `dct_matrix` | no | brief |
| `dct_1st_derivative` | no | brief |
| `dct_2nd_derivative` | no | brief |
| `CBtransform` (3 variants), `CBtransform!` | yes | full |
| `CAtransform`, `CAtransform!` | yes | full |
| `CItransform`, `CItransform!` | yes | full |
| `CIIntcoefficients` | no | brief |
| `CIInttransform` (2 variants) | yes | full |
| `CIxcoefficients` | no | brief (recurrence formula) |
| `CIxtransform` (2 variants) | yes | full |
| `CIxxtransform` | yes | full |

### rl_grid.jl / rz_grid.jl (grid files)

For grid files, document every function following the `spline1D_grid.jl` pattern. Priority:

| Function type | Docstring priority |
|--------------|-------------------|
| Struct (`RL_Grid`, `RZ_Grid`) | full |
| `create_*_Grid` | full |
| `getGridpoints` | full |
| `spectralTransform!` | full |
| `gridTransform!` (all variants) | full |
| `tileTransform!`, `splineTransform!` | full |
| `calcTileSizes`, `calcPatchMap`, `calcHaloMap` | full |
| `sumSpectralTile!`, `setSpectralTile!` | full |
| `sumSharedSpectral`, `getBorderSpectral` | full |
| `allocateSplineBuffer` | brief (API-compat note) |
| `num_columns` | brief |
| `getRegularGridpoints` | full |
| `regularGridTransform` | full |
| Non-`!` internal helpers (`spectralTransform`, `gridTransform`) | brief |

---

## Part 8: Documentation Quality Checklist

Before finalising documentation for a new module or grid file, verify:

- [ ] Every **exported** symbol has a docstring with `# Arguments`, `# Returns`, and at least one `# Example`
- [ ] Every **multi-dispatch family** uses a single shared docstring with all variant signatures in the banner
- [ ] Every `[`Symbol`](@ref)` cross-link resolves without error (`makedocs` will report failures)
- [ ] BC constants each have a one-line docstring (even if trivial)
- [ ] Parameter/data structs document all fields including auto-computed ones
- [ ] Non-obvious math is explained inline with a short LaTeX formula
- [ ] Index ordering and dimension conventions are stated explicitly for 2D/3D arrays
- [ ] `make.jl` includes the new submodule in the `modules` list
- [ ] The new `.md` page is added to the `pages` list in `make.jl`
- [ ] `makedocs` runs to completion with no errors (only the deployment warning is acceptable)
- [ ] Deprecated or unimplemented functions note their status explicitly in the docstring

---

## Part 9: Math Notation Reference

Use these consistent conventions throughout all docstrings:

| Concept | LaTeX notation |
|---------|---------------|
| Basis function | `\\varphi_m(x)` (B-spline), `\\phi_k(\\theta)` (Fourier) |
| Spectral coefficients | `a_m` (B-spline), `\\hat{u}_k` (Fourier) |
| B-vector inner product | `b_m = \\langle \\varphi_m, u \\rangle` |
| Forward transform | `\\mathcal{B}[u]` or `SB[u]`, `FB[u]`, `CB[u]` |
| Inverse transform | `\\mathcal{I}[a]` |
| Integration by parts | `[\\varphi_m f]_{x_0}^{x_1} - \\int \\varphi_m' f\\, dx` |
| Filter weight | `\\varepsilon_q = \\left(\\frac{l_q \\Delta x}{2\\pi}\\right)^6` |
| Derivative order in SI | `\\partial^n / \\partial x^n` with `n = 0, 1, 2` |
| Chebyshev polynomial | `T_n(z)` |
| CGL node | `z_j = \\cos((j-1)\\pi/(N-1))` |
| Linear operator | `\\mathcal{L}[u] = f` |
| Cost functional | `J(u, p) = \\int \\mathcal{F}(u, u', u'', p)\\, dx` |
| Kronecker product | `A \\otimes B` |
| Laplacian | `\\nabla^2 u` |
| Operator matrix | `\\mathbf{L}` (bold for matrices) |

All LaTeX in docstrings must use **double** backslashes (`\\sum`, `\\varphi`) because Julia
string parsing consumes one level of escaping before Documenter sees them.

---

## Part 10: Documenting the Solver Framework (`solver.jl`)

### 10.1 Solver Type Docstrings

The solver framework introduces composite types that bundle several components. Follow these
patterns for consistent documentation.

**Abstract Backend Type:**
```julia
"""
    AbstractSolverBackend

Abstract supertype for solver backend sentinel types. Concrete subtypes select the
algorithm used by [`solve`](@ref).

See also: [`LocalLinearBackend`](@ref), [`OptimizationBackend`](@ref)
"""
abstract type AbstractSolverBackend end
```

**SpringsteelProblem:**
```julia
"""
    SpringsteelProblem{B <: AbstractSolverBackend}

Composite type bundling a [`SpringsteelGrid`](@ref), linear operator or cost
functional, and solver backend specification.

# Fields
- `grid::AbstractGrid`: The discretised domain.
- `operator::Union{Matrix{Float64}, Nothing}`: Assembled linear operator (for linear problems).
- `rhs::Union{Vector{Float64}, Nothing}`: Right-hand side vector (for linear problems).
- `cost::Union{Function, Nothing}`: Cost functional ``J(u, p)`` (for nonlinear/optimization problems).
- `parameters::Dict{String, Any}`: Problem parameters passed to the solver.
- `backend::B`: Solver backend sentinel.

# Example
```julia
prob = SpringsteelProblem(grid; operator=L, rhs=f)
sol = solve(prob)
```

See also: [`solve`](@ref), [`SpringsteelSolution`](@ref), [`assemble_operator`](@ref)
"""
```

### 10.2 Operator Assembly Docstrings

Matrix assembly functions should document:
1. What the returned matrix represents mathematically
2. The relationship between matrix dimensions and grid dimensions
3. How boundary conditions are incorporated

```julia
"""
    operator_matrix(grid::SpringsteelGrid, dim::Symbol, order::Int) -> Matrix{Float64}

Extract a 1D operator matrix for dimension `dim` (`:i`, `:j`, or `:k`) at derivative
`order` (0 = evaluation, 1 = first derivative, 2 = second derivative).

Dispatches to the appropriate basis module's matrix function based on the grid's
basis type for that dimension (Spline → `spline_basis_matrix`, Fourier →
`dft_matrix`, Chebyshev → `dct_matrix`).

# Arguments
- `grid::SpringsteelGrid`: The grid whose basis objects provide the matrix
- `dim::Symbol`: Which dimension (`:i`, `:j`, `:k`)
- `order::Int`: Derivative order (0, 1, or 2)

# Returns
- `Matrix{Float64}`: Size `(physical_dim, spectral_dim)` for the requested dimension

See also: [`assemble_operator`](@ref), [`OperatorTerm`](@ref)
"""
```

### 10.3 solve() Docstring

The `solve()` function has multiple dispatches (one per backend). Use a single
shared docstring with a multi-signature banner:

```julia
"""
    solve(prob::SpringsteelProblem{LocalLinearBackend}) -> SpringsteelSolution
    solve(prob::SpringsteelProblem{OptimizationBackend}) -> SpringsteelSolution

Solve the problem defined by `prob` using the specified backend.

For `LocalLinearBackend`: factorises the operator matrix and solves
``\\mathbf{L} \\mathbf{a} = \\mathbf{f}`` via LU decomposition.

For `OptimizationBackend`: minimises the cost functional ``J(u, p)``
using the algorithm specified in the backend (requires `Optimization.jl`).

# Returns
- [`SpringsteelSolution`](@ref) containing the spectral coefficients,
  physical-space solution, convergence flag, and solver diagnostics.

See also: [`SpringsteelProblem`](@ref), [`assemble_operator`](@ref)
"""
```

### 10.4 Matrix Representation Docstrings (Basis Modules)

New matrix representation functions added to basis modules should follow the
existing `dct_matrix` pattern in `Chebyshev.jl`:

```julia
"""
    dft_matrix(ring::Fourier1D) -> Matrix{Float64}

Build the ``(N \\times M)`` DFT evaluation matrix for the Fourier basis.

Entry `[i, j]` is the value of the `j`-th Fourier basis function at the
`i`-th mish point. Columns are ordered as ``[1, \\cos(x), \\sin(x),
\\cos(2x), \\sin(2x), \\ldots]`` up to wavenumber `kmax`.

Useful for debugging transforms and constructing linear solvers directly
in spectral space.

See also: [`dft_1st_derivative`](@ref), [`dft_2nd_derivative`](@ref)
"""
```

### 10.5 Documentation Page Structure (`docs/src/solver.md`)

The solver documentation page should follow this structure:

````markdown
```@meta
CurrentModule = Springsteel
```

# Solver Framework

## Overview

Brief description of the solver framework and its two backends.

## Matrix Representations

```@docs
CubicBSpline.spline_basis_matrix
CubicBSpline.spline_1st_derivative_matrix
CubicBSpline.spline_2nd_derivative_matrix
Fourier.dft_matrix
Fourier.dft_1st_derivative
Fourier.dft_2nd_derivative
Chebyshev.dct_matrix
Chebyshev.dct_1st_derivative
Chebyshev.dct_2nd_derivative
```

## Operator Assembly

```@docs
OperatorTerm
operator_matrix
assemble_operator
assemble_from_equation
```

## Problem Definition

```@docs
SpringsteelProblem
SpringsteelSolution
LocalLinearBackend
OptimizationBackend
```

## Solving

```@docs
solve
```
````

Add the page to `docs/make.jl`:
```julia
pages=[
    ...
    "Solver" => "solver.md",
    ...
]
```

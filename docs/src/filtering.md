```@meta
CurrentModule = Springsteel
```

# Spectral Filtering

Springsteel supports post-transform spectral coefficient filtering for
Fourier and Chebyshev bases. Filters are specified per-variable on the
grid parameters and applied automatically at the end of every
`spectralTransform!` — your physical arrays and derivative slots see the
filtered field, not the raw projection.

Filtering is useful for

- **removing specific azimuthal or vertical modes** you know aren't
  physical (e.g. wave 0 for a mean-subtracted analysis, wave 1 for a
  translation-removed vortex);
- **smoothing** fields before differentiating, to avoid aliasing or
  small-scale noise contaminating the derivatives;
- **band-passing** in wavenumber space to isolate scales of interest.

For B-spline directions, the spectral-coefficient filters above are a
no-op (splines are a local, non-orthogonal basis with BC structure baked
into the coefficients). Use the `spline_filter` parameter — see
[Spline filtering](#Spline-filtering) — to apply a smoothing filter
on a CubicBSpline direction; it operates as a physical-space convolution
on the mish before SB. B-spline interior smoothness can also be tuned at
the basis level via `l_q`.

## Filter types

Two filter types cover most needs. Both are subtypes of
[`AbstractFilter`](@ref).

```@docs
AbstractFilter
```

### `SpectralFilter` — wavenumber-domain

```@docs
SpectralFilter
```

`SpectralFilter` is a parameterised combination of low-pass, high-pass,
and notch cuts with an optional tapered transition at the passband
edges. Fields:

| Field         | Default    | Meaning                                          |
|:--------------|:-----------|:-------------------------------------------------|
| `low_pass`    | `-1`       | Zero modes `k > low_pass`; `-1` disables         |
| `high_pass`   | `0`        | Zero modes `k < high_pass`; `0` disables         |
| `notch`       | `Int[]`    | Hard-cutoff list of specific modes to zero       |
| `window`      | `:boxcar`  | Taper window: `:boxcar`, `:hann`, `:lanczos`, `:exponential` |
| `taper_width` | `0`        | Mode count over which window transitions 1 → 0   |

With `taper_width = 0` every window reduces to boxcar (sharp cutoff).
For smooth transitions, increase `taper_width` — typical values are
3–10 modes. Hann is the most forgiving taper; Lanczos has a slightly
sharper transition; exponential is near-Gaussian.

```julia
# Hard low-pass at k=50, plus remove wave 1
SpectralFilter(low_pass=50, notch=[1])

# Band-pass 5 ≤ k ≤ 20 with Lanczos taper over 3 modes on each edge
SpectralFilter(high_pass=5, low_pass=20, window=:lanczos, taper_width=3)

# Remove just the mean (k=0)
SpectralFilter(notch=[0])
```

### `GaussianFilter` — smooth envelope

```@docs
GaussianFilter
```

`GaussianFilter` has two modes of action:

- On a Fourier or Chebyshev direction it multiplies each spectral
  coefficient by `exp(-(k/σ)^(2p))`, where `σ` is a width parameter
  (modes at `k = σ` are attenuated to `e⁻¹` for `order = 1`) and `p` is
  the order.
- On a CubicBSpline direction, when supplied via `spline_filter`, it
  acts as a physical-space convolution with kernel
  `exp(-x²/(2σ²))`. Here `σ` is in cell widths.

```julia
# Spectral envelope on a Fourier/Chebyshev direction
GaussianFilter(sigma=20.0)            # standard
GaussianFilter(sigma=20.0, order=3)   # super-Gaussian, sharper cutoff

# Physical-space smoothing on a spline direction (2-cell-wide Gaussian)
GaussianFilter(sigma=2.0)
```

### `LanczosFilter` — windowed sinc

```@docs
LanczosFilter
```

`LanczosFilter` is a thin convenience wrapper around a Lanczos-windowed
sinc:

- On a CubicBSpline direction it convolves with the kernel
  `K(x) = sinc(x/h) · sinc(x/(a·h))` for `|x| < a·h`, zero outside,
  where `h` is the cell width and `a` is the lobe count.
- On a Fourier or Chebyshev direction it delegates to a
  `SpectralFilter(window=:lanczos, low_pass=low_pass, taper_width=a)`.

```julia
# Spline path: 3-lobe Lanczos kernel
LanczosFilter(a=3)

# Spectral path: low-pass at k=10 with Lanczos taper of 3 modes
LanczosFilter(a=3, low_pass=10)
```

## Attaching filters to a grid

Filters live in two per-variable dicts on
[`SpringsteelGridParameters`](@ref): `fourier_filter` for Fourier
dimensions and `chebyshev_filter` for Chebyshev dimensions. The keys
match the grid's variable names, with an optional `"default"`
fallback:

```julia
gp = SpringsteelGridParameters(
    geometry = "RLZ",
    iMin = 0.0, iMax = 100.0, num_cells = 30,
    kMin = 0.0, kMax = 10.0, kDim = 32,
    vars = Dict("u" => 1, "v" => 2),
    BCL  = Dict("default" => NaturalBC()),
    BCR  = Dict("default" => NaturalBC()),
    BCB  = Dict("default" => DirichletBC()),
    BCT  = Dict("default" => DirichletBC()),

    # Remove wave 0 from u (zonal mean), band-pass v (azimuthal waves 1–5)
    fourier_filter = Dict(
        "u" => SpectralFilter(notch=[0]),
        "v" => SpectralFilter(high_pass=1, low_pass=5),
    ),
    # Smooth the vertical structure of both
    chebyshev_filter = Dict(
        "default" => GaussianFilter(sigma=20.0, order=2),
    ),
)
grid = createGrid(gp)
```

Variables without an entry fall back to `"default"`, and variables
with no filter at all are left unfiltered.

## Spline filtering

Spline directions live on a local, non-orthogonal basis with boundary
conditions baked into the coefficients, so a wavenumber-domain envelope
is not the right tool. `spline_filter` instead takes a per-(variable,
direction) Dict mapping each spline axis to a `GaussianFilter` or
`LanczosFilter`. It is applied as a physical-space convolution on the
mish (quadrature points) immediately before the SB transform:

```
spectralTransform!(grid):
    _filter_mish!(grid)            # spline filter convolves on physical[..., v, 1]
    SB transforms (per direction)
    SA transforms (per direction)
    applyFilter!(grid)             # spectral filters on Fourier / Chebyshev
```

Boundary conditions are preserved by construction: filtering happens
upstream of SB / SA, so γ-folding and `ahat` re-impose the configured BC
after the convolution. Boundary handling for the kernel itself is
zero-extend + renormalise (kernel weight outside the domain reweights
the in-domain contributions, preserving the DC component).

The dict shape is `Dict{String, Dict{Symbol, AbstractFilter}}`. Outer
keys are variable names (or `"default"`); inner keys are `:i`, `:j`,
`:k`, or `:default` (a var-wide override that applies to every spline
direction the geometry has).

```julia
gp = SpringsteelGridParameters(
    geometry = "RR",
    iMin = 0.0, iMax = 100.0, num_cells = 30,
    jMin = 0.0, jMax = 100.0,
    vars = Dict("u" => 1, "v" => 2),
    BCL = Dict("default" => NaturalBC()),
    BCR = Dict("default" => NaturalBC()),
    BCU = Dict("default" => NaturalBC()),
    BCD = Dict("default" => NaturalBC()),

    spline_filter = Dict(
        # Per-direction Gaussian on u
        "u" => Dict(
            :i => GaussianFilter(sigma=2.0),
            :j => GaussianFilter(sigma=4.0),
        ),
        # Same Lanczos kernel on every spline direction of v
        "v" => Dict(:default => LanczosFilter(a=3)),
    ),
)
grid = createGrid(gp)
```

`SpectralFilter` (boxcar / notch / hard-window cutoffs) is rejected on a
spline direction — those require an FFT against a global orthogonal
basis. Choose a Fourier or Chebyshev basis if you need a hard spectral
cutoff.

## Application

```@docs
applyFilter!
```

`applyFilter!` runs automatically at the end of every
`spectralTransform!` — you don't normally call it directly. If you
modify `grid.spectral` by hand and want to re-apply the configured
filters, call `applyFilter!(grid)` manually. The dispatch is
geometry-aware; it walks the per-variable filter dicts and applies the
right window / envelope to each spectral block.

## Anti-aliasing vs filtering

Filtering via `fourier_filter` / `chebyshev_filter` is **orthogonal to
max_wavenumber**. The `max_wavenumber` setting controls the physical
grid resolution's Nyquist cap and acts as a hard anti-aliasing cutoff
at the transform level — that's how many modes the grid can resolve at
all. Filters act on the modes that survive anti-aliasing, selecting
which of those to keep, weight, or suppress.

Use `max_wavenumber` to cap the resolved spectrum for performance or
accuracy reasons; use a filter to shape the spectrum within that cap
for physical reasons.

## See also

- [SpringsteelGrid](springsteel_grid.md) — where filter dicts live in
  grid parameters
- [Fourier](fourier.md) / [Chebyshev](chebyshev.md) — underlying basis
  modules
- [Solver Framework](solver.md) — filtering interacts with BC handling;
  the solver sees the filtered operator

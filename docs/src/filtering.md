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

Filtering is a **no-op for pure B-spline grids** (`R`, `RR`, `RRR`) —
B-spline smoothness is controlled at the basis level via `l_q` rather
than at the spectral-coefficient level. Filter specs on pure-spline
grids are accepted but silently ignored.

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

`GaussianFilter` multiplies each spectral coefficient by
`exp(-(k/σ)^(2p))`, where `σ` is a width parameter (modes at `k = σ`
are attenuated to `e⁻¹` for `order = 1`) and `p` is the order. Higher
orders approach a boxcar with a smooth transition — useful when you
want aggressive smoothing without ringing:

```julia
# Standard Gaussian with width σ = 20
GaussianFilter(sigma=20.0)

# Super-Gaussian order 3 — sharper cutoff, still smooth
GaussianFilter(sigma=20.0, order=3)
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

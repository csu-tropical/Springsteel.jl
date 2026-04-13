```@meta
CurrentModule = Springsteel
```

# Fourier

The `Fourier` submodule implements the 1D Fourier ring basis used for
the azimuthal direction in cylindrical-coordinate grids (`RL_Grid`,
`RLZ_Grid`) and for purely periodic grids (`L`, `LL`, `LLZ`). It uses
[FFTW.jl](https://juliamath.github.io/FFTW.jl/latest/) real-to-
halfcomplex transforms (`FFTW.R2HC` / `FFTW.HC2R`) with a pre-measured
plan for maximum throughput.

## Mathematical Overview

For a real-valued function $u$ on the circle $[0, 2\pi)$ sampled at
$N$ equally spaced points $\phi_n = 2\pi n / N$, the discrete Fourier
series writes $u$ as a sum of sines and cosines up to the Nyquist
wavenumber $\lfloor N/2 \rfloor$:

```math
u(\phi) \;=\; \frac{a_0}{2}
          + \sum_{k=1}^{\lfloor N/2 \rfloor - 1}
              \bigl(a_k \cos(k\phi) + b_k \sin(k\phi)\bigr)
          + \frac{a_{N/2}}{2}\cos\!\left(\tfrac{N}{2}\phi\right),
```

where the coefficient pair $(a_k, b_k)$ captures the $k$-th azimuthal
wavenumber. The last term is the Nyquist mode and is real-valued
only when $N$ is even. Because $u$ is real, the full complex spectrum
is conjugate-symmetric and only $\lfloor N/2\rfloor + 1$ independent
coefficients need to be stored — this is the **half-complex**
(`R2HC`) layout FFTW uses.

Derivatives in the spectral domain are exact and local: differentiating
$\cos(k\phi)$ gives $-k\sin(k\phi)$ and differentiating $\sin(k\phi)$
gives $k\cos(k\phi)$, so the $n$-th derivative operator is a diagonal
multiplication by $(ik)^n$ on the complex spectrum (modulo the
real-valued storage reshuffle). This is why Fourier bases are the
natural choice for periodic directions: exactness plus FFT-speed
evaluation.

Springsteel stores the half-complex coefficients in the order
`[a_0, a_1, a_2, ..., a_{N/2}, b_{N/2-1}, ..., b_2, b_1]` — the
standard FFTW `R2HC` layout. Internally the cylindrical and spherical
grids further rearrange these per-ring coefficients into the larger
`grid.spectral` array using the wavenumber-interleaved layout
documented in [Contributing](contributing.md).

### References

- Wikipedia: [Discrete Fourier transform](https://en.wikipedia.org/wiki/Discrete_Fourier_transform)
- Wikipedia: [Fourier series](https://en.wikipedia.org/wiki/Fourier_series)
- Press et al., *Numerical Recipes*, 3rd ed., Chapter 12
- FFTW manual: [One-dimensional DFTs of real data](https://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html)

## Boundary Condition Constants

```@docs
Fourier.PERIODIC
```

## Parameter and Data Structures

```@docs
Fourier.FourierParameters
Fourier.Fourier1D
Fourier.PhaseFilter
```

## Internal Setup Functions

```@docs
Fourier.calcMishPoints
Fourier.calcPhaseFilter
Fourier.calcInvPhaseFilter
Fourier.apply_phasefilter_forward!
Fourier.apply_phasefilter_inverse!
```

## Transform Functions

```@docs
Fourier.FBtransform
Fourier.FAtransform
Fourier.FItransform
Fourier.FIxcoefficients
Fourier.FIxtransform
Fourier.FIxxtransform
Fourier.FIIntcoefficients
Fourier.FIInttransform
```

## Matrix Representations

```@docs
Fourier.dft_matrix
Fourier.dft_1st_derivative
Fourier.dft_2nd_derivative
```

## Generic Wrappers

No-prefix wrappers that delegate to the `F`-prefixed functions above, enabling
basis-type-agnostic code.

```@docs
Fourier.Btransform
Fourier.Btransform!
Fourier.Atransform
Fourier.Atransform!
Fourier.Itransform
Fourier.Itransform!
Fourier.Ixtransform
Fourier.Ixxtransform
Fourier.IInttransform
```

```@meta
CurrentModule = Springsteel
```

# Fourier

The `Fourier` submodule implements the 1D Fourier ring basis used for the azimuthal
direction in cylindrical-coordinate grids (`RL_Grid`, `RLZ_Grid`).  It uses
[FFTW.jl](https://juliamath.github.io/FFTW.jl/latest/) real-to-halfcomplex transforms
(`FFTW.R2HC` / `FFTW.HC2R`) with a pre-measured plan for maximum throughput.

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

```@meta
CurrentModule = Springsteel
```

# Chebyshev

The `Chebyshev` submodule implements the 1D Chebyshev spectral transform used for the
vertical direction in cylindrical-coordinate grids (`RZ_Grid`, `RLZ_Grid`). It uses
[FFTW.jl](https://juliamath.github.io/FFTW.jl/latest/) DCT-I transforms (`FFTW.REDFT00`)
with a pre-measured plan for maximum throughput. Grid points are placed at
Chebyshev–Gauss–Lobatto (CGL) nodes, giving spectral accuracy for smooth functions.

## Boundary Condition Constants

```@docs
Chebyshev.R0
Chebyshev.R1T0
Chebyshev.R1T1
Chebyshev.R1T2
Chebyshev.R2T10
Chebyshev.R2T20
Chebyshev.R3
```

## Parameter and Data Structures

```@docs
Chebyshev.ChebyshevParameters
Chebyshev.Chebyshev1D
```

## Internal Setup Functions

```@docs
Chebyshev.calcMishPoints
Chebyshev.calcFilterMatrix
Chebyshev.calcGammaBC
Chebyshev.calcGammaBCalt
```

## Transform Functions

```@docs
Chebyshev.CBtransform
Chebyshev.CAtransform
Chebyshev.CItransform
Chebyshev.CIxcoefficients
Chebyshev.CIxtransform
Chebyshev.CIxxtransform
Chebyshev.CIIntcoefficients
Chebyshev.CIInttransform
```

## Generic Wrappers

No-prefix wrappers that delegate to the `C`-prefixed functions above, enabling
basis-type-agnostic code.

```@docs
Chebyshev.Btransform!
Chebyshev.Atransform!
Chebyshev.Itransform!
Chebyshev.Ixtransform
Chebyshev.Ixxtransform
Chebyshev.IInttransform
```

## DCT Matrix Utilities

```@docs
Chebyshev.dct_matrix
Chebyshev.dct_1st_derivative
Chebyshev.dct_2nd_derivative
```

## BVP Solver Utilities

```@docs
Chebyshev.bvp
Chebyshev.bvp_modified_basis
Chebyshev.bvp_basis
```


```@meta
CurrentModule = Springsteel
```

# CubicBSpline

The `CubicBSpline` submodule implements the Ooyama (2002) cubic B-spline transform
method used as the radial basis function throughout Springsteel.

## Boundary Condition Constants

```@docs
CubicBSpline.R0
CubicBSpline.R1T0
CubicBSpline.R1T1
CubicBSpline.R1T2
CubicBSpline.R2T10
CubicBSpline.R2T20
CubicBSpline.R3
CubicBSpline.PERIODIC
```

## Parameter and Data Structures

```@docs
CubicBSpline.SplineParameters
CubicBSpline.Spline1D
CubicBSpline.Spline1D(::CubicBSpline.SplineParameters)
```

## Internal Basis and Setup Functions

```@docs
CubicBSpline.basis
CubicBSpline.calcGammaBC
CubicBSpline.calcPQfactor
CubicBSpline.calcMishPoints
CubicBSpline.setMishValues
```

## Transform Functions

```@docs
CubicBSpline.SBtransform
CubicBSpline.SBxtransform
CubicBSpline.SAtransform
CubicBSpline.SItransform
CubicBSpline.SItransform_matrix
CubicBSpline.SIxtransform
CubicBSpline.SIxxtransform
```

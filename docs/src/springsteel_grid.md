```@meta
CurrentModule = Springsteel
```

# SpringsteelGrid

The `SpringsteelGrid` type is the unified parametric grid struct that replaces the
legacy per-geometry structs (`R_Grid`, `RL_Grid`, `RZ_Grid`, etc.) while remaining
backward-compatible via type aliases.

## Geometry Types

```@docs
AbstractGeometry
CartesianGeometry
CylindricalGeometry
SphericalGeometry
```

## Basis Array Types

```@docs
SplineBasisArray
FourierBasisArray
ChebyshevBasisArray
NoBasisArray
```

## Grid Parameters

```@docs
SpringsteelGridParameters
GridParameters
```

## Grid Struct and Aliases

```@docs
SpringsteelGrid
```

## Grid Factory

```@docs
createGrid
parse_geometry
compute_derived_params
```

## Transforms

```@docs
spectralTransform!
gridTransform!
```

## Tiling

```@docs
calcTileSizes
calcPatchMap
calcHaloMap
allocateSplineBuffer
num_columns
sumSpectralTile!
setSpectralTile!
getBorderSpectral
sumSharedSpectral
splineTransform!
tileTransform!
```

## I/O

```@docs
getGridpoints
getRegularGridpoints
regularGridTransform
write_grid
read_physical_grid
check_grid_dims
```

## Grid Type Reference

The table below lists every grid type available in Springsteel together with its
canonical `geometry` string, descriptive alias strings, and the corresponding
`SpringsteelGrid` type alias.

### Spline-based grids

| Canonical `geometry` | Basis (i × j × k) | Primary type alias | Descriptive aliases |
|:-------------------- |:------------------ |:------------------ |:------------------- |
| `"R"` | Spline | `R_Grid` | `Spline1D_Grid` |
| `"RZ"` | Spline × Chebyshev | `RZ_Grid` | — |
| `"RL"` | Spline × Fourier (cyl.) | `RL_Grid` | `Polar_Grid` |
| `"RR"` | Spline × Spline | `RR_Grid` | `Spline2D_Grid` |
| `"RLZ"` | Spline × Fourier × Chebyshev (cyl.) | `RLZ_Grid` | `Cylindrical_Grid` |
| `"RRR"` | Spline × Spline × Spline | `RRR_Grid` | `Spline3D_Grid`, `Samurai_Grid` |
| `"SL"` | Spline × Fourier (sph.) | `SL_Grid` | `SphericalShell_Grid` |
| `"SLZ"` | Spline × Fourier × Chebyshev (sph.) | `SLZ_Grid` | `Sphere_Grid` |

### Fourier-based grids

| Canonical `geometry` | Basis (i × j × k) | Primary type alias | Descriptive aliases |
|:-------------------- |:------------------ |:------------------ |:------------------- |
| `"L"` | Fourier | `L_Grid` | `Ring1D_Grid` |
| `"LL"` | Fourier × Fourier | `LL_Grid` | `Ring2D_Grid` |
| `"LLZ"` | Fourier × Fourier × Chebyshev | `LLZ_Grid` | `DoublyPeriodic_Grid` |

### Chebyshev-based grids

| Canonical `geometry` | Basis (i × j × k) | Primary type alias | Descriptive aliases |
|:-------------------- |:------------------ |:------------------ |:------------------- |
| `"Z"` | Chebyshev | `Z_Grid` | `Column1D_Grid` |
| `"ZZ"` | Chebyshev × Chebyshev | `ZZ_Grid` | `Column2D_Grid` |
| `"ZZZ"` | Chebyshev × Chebyshev × Chebyshev | `ZZZ_Grid` | `Column3D_Grid` |

### Alias naming convention

All geometry strings are normalised though a two-step look-up:

1. **Descriptive aliases** (right-hand column in the tables above) are defined in
   `_GEOMETRY_ALIASES` and map to the **canonical** short code on the left.
2. The canonical code then determines which creation function is called.

The rule is: *the more descriptive name is always the alias; the short code is
always the canonical target.*  For example, `"Ring1D"` → `"L"`,
`"DoublyPeriodic"` → `"LLZ"`, `"Polar"` → `"RL"`.

```@docs
L_Grid
LL_Grid
LLZ_Grid
Ring1D_Grid
Ring2D_Grid
DoublyPeriodic_Grid
Z_Grid
ZZ_Grid
ZZZ_Grid
Column1D_Grid
Column2D_Grid
Column3D_Grid
```

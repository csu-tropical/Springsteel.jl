```@meta
CurrentModule = Springsteel
```

# SpringsteelGrid

The `SpringsteelGrid` type is the unified parametric grid struct introduced in the
Phase 1–9 refactoring.  It replaces the legacy per-geometry structs (`R_Grid`,
`RL_Grid`, `RZ_Grid`, etc.) while remaining backward-compatible via type aliases.

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

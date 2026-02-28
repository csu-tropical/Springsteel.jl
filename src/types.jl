# types.jl — Unified parametric type system for SpringsteelGrid
#
# Defines:
#   • Geometry sentinel types  (AbstractGeometry and subtypes)
#   • Basis sentinel types     (AbstractBasisType and subtypes)
#   • Typed basis containers   (SplineBasisArray, FourierBasisArray, etc.)
#   • SpringsteelGrid{G,I,J,K} unified parametric struct
#   • num_deriv_slots dispatch
#   • New type aliases (SL_Grid, SLZ_Grid)
#   • Commented-out aliases for existing grid types (activated in Phase 10)

# ────────────────────────────────────────────────────────────────────────────
# Geometry sentinel types
# ────────────────────────────────────────────────────────────────────────────

"""
    AbstractGeometry

Abstract supertype for all geometry sentinel types used as type parameters of
[`SpringsteelGrid`](@ref).  Dispatch on the concrete subtype selects transform
logic and ring-size formulas appropriate for each geometry.

See also: [`CartesianGeometry`](@ref), [`CylindricalGeometry`](@ref),
[`SphericalGeometry`](@ref)
"""
abstract type AbstractGeometry end

"""
    CartesianGeometry <: AbstractGeometry

Geometry sentinel for grids whose dimensions are all Cartesian (non-periodic,
tensor-product).  Used by `R_Grid`, `RR_Grid`, `RZ_Grid`, `RRR_Grid`.
Transform logic is a simple tensor-product pipeline (separable in every
dimension).

See also: [`AbstractGeometry`](@ref), [`CylindricalGeometry`](@ref)
"""
struct CartesianGeometry <: AbstractGeometry end

"""
    CylindricalGeometry <: AbstractGeometry

Geometry sentinel for grids that include a periodic azimuthal (Fourier)
dimension whose ring size varies linearly with radius:
``\\text{ring\\_size}(r_i) = 4 + 4 r_i``.
Used by `RL_Grid` (2-D) and `RLZ_Grid` (3-D).

See also: [`AbstractGeometry`](@ref), [`SphericalGeometry`](@ref)
"""
struct CylindricalGeometry <: AbstractGeometry end

"""
    SphericalGeometry <: AbstractGeometry

Geometry sentinel for grids that include a periodic azimuthal (Fourier)
dimension whose ring size follows a sin(θ) distribution along the latitudinal
i-dimension.  Used by `SL_Grid` (2-D) and `SLZ_Grid` (3-D).

See also: [`AbstractGeometry`](@ref), [`CylindricalGeometry`](@ref)
"""
struct SphericalGeometry <: AbstractGeometry end

# ────────────────────────────────────────────────────────────────────────────
# Basis sentinel types
# ────────────────────────────────────────────────────────────────────────────

"""
    AbstractBasisType

Abstract supertype for basis sentinel singletons.  These are used only as type
parameters in [`SpringsteelGrid`](@ref) dispatch; they carry no data.

See also: [`SplineBasisType`](@ref), [`FourierBasisType`](@ref),
[`ChebyshevBasisType`](@ref), [`NoBasisType`](@ref)
"""
abstract type AbstractBasisType end

"""
    SplineBasisType <: AbstractBasisType

Basis sentinel indicating the cubic B-spline basis
([`CubicBSpline.Spline1D`](@ref)) is active in a given dimension.
"""
struct SplineBasisType  <: AbstractBasisType end

"""
    FourierBasisType <: AbstractBasisType

Basis sentinel indicating the Fourier basis
([`Fourier.Fourier1D`](@ref)) is active in a given dimension.
"""
struct FourierBasisType <: AbstractBasisType end

"""
    ChebyshevBasisType <: AbstractBasisType

Basis sentinel indicating the Chebyshev basis
([`Chebyshev.Chebyshev1D`](@ref)) is active in a given dimension.
"""
struct ChebyshevBasisType <: AbstractBasisType end

"""
    NoBasisType <: AbstractBasisType

Basis sentinel indicating that a dimension slot is unused.  A grid whose
k-dimension has `NoBasisType` is at most 2-D.
"""
struct NoBasisType <: AbstractBasisType end

# ────────────────────────────────────────────────────────────────────────────
# Typed basis containers
# ────────────────────────────────────────────────────────────────────────────

"""
    SplineBasisArray

Typed container for an array of [`CubicBSpline.Spline1D`](@ref) objects.
Using a concrete wrapper (rather than `Vector{Any}`) preserves type stability
across all grid operations.

# Fields
- `data::Array{CubicBSpline.Spline1D}`: N-dimensional array of spline objects,
  indexed as `data[var]` (1-D) or `data[indices..., var]` (multi-D).

See also: [`FourierBasisArray`](@ref), [`ChebyshevBasisArray`](@ref),
[`NoBasisArray`](@ref)
"""
struct SplineBasisArray
    data::Array{CubicBSpline.Spline1D}
end

"""
    FourierBasisArray

Typed container for an array of [`Fourier.Fourier1D`](@ref) objects.

# Fields
- `data::Array{Fourier.Fourier1D}`: Array of Fourier ring objects.

See also: [`SplineBasisArray`](@ref)
"""
struct FourierBasisArray
    data::Array{Fourier.Fourier1D}
end

"""
    ChebyshevBasisArray

Typed container for an array of [`Chebyshev.Chebyshev1D`](@ref) objects.

# Fields
- `data::Array{Chebyshev.Chebyshev1D}`: Array of Chebyshev column objects.

See also: [`SplineBasisArray`](@ref)
"""
struct ChebyshevBasisArray
    data::Array{Chebyshev.Chebyshev1D}
end

"""
    NoBasisArray

Sentinel container representing an unused dimension.  Holds no data.
Used as the `jbasis` or `kbasis` field for grid types whose j- or k-dimension
is inactive.

See also: [`SplineBasisArray`](@ref), `num_deriv_slots`(@ref)
"""
struct NoBasisArray end

# ────────────────────────────────────────────────────────────────────────────
# Unified parametric grid struct
# ────────────────────────────────────────────────────────────────────────────

"""
    SpringsteelGrid{G, I, J, K} <: AbstractGrid

Unified parametric grid struct for semi-spectral numerical models.  A single
type replaces the eight legacy grid structs (`R_Grid`, `RL_Grid`, `RZ_Grid`,
`RR_Grid`, `RLZ_Grid`, `RRR_Grid`, `Spline1D_Grid`, `Spline2D_Grid`) and adds
support for spherical geometry (`SL_Grid`, `SLZ_Grid`).

# Type Parameters
- `G <: AbstractGeometry`: Geometry sentinel.  Selects the transform pipeline
  and ring-size formula.
  - `CartesianGeometry`   — tensor-product (all non-periodic dimensions)
  - `CylindricalGeometry` — Fourier ring sizes grow linearly with radius
  - `SphericalGeometry`   — Fourier ring sizes follow sin(θ) distribution
- `I`: i-dimension (radial / primary) basis container.
  Typically `SplineBasisArray`.
- `J`: j-dimension (azimuthal / secondary) basis container.
  `FourierBasisArray`, `SplineBasisArray`, or `NoBasisArray`.
- `K`: k-dimension (vertical / tertiary) basis container.
  `ChebyshevBasisArray`, `SplineBasisArray`, or `NoBasisArray`.

# Fields
- `params::SpringsteelGridParameters`: All grid configuration (domain bounds,
  cell counts, boundary conditions, variable map, tiling indices).
- `ibasis::I`: i-dimension basis objects (e.g. `SplineBasisArray`).
- `jbasis::J`: j-dimension basis objects (or `NoBasisArray` if unused).
- `kbasis::K`: k-dimension basis objects (or `NoBasisArray` if unused).
- `spectral::Array{Float64}`: Spectral coefficient array,
  size `(spectral_total, nvars)`.
- `physical::Array{Float64}`: Physical-space values and derivatives,
  size `(physical_total, nvars, nderiv)` where
  `nderiv = num_deriv_slots(jbasis, kbasis)`.

# Backward-compatible type aliases
```julia
const R_Grid        = SpringsteelGrid{CartesianGeometry,   SplineBasisArray, NoBasisArray,      NoBasisArray}
const Spline1D_Grid = R_Grid
const RZ_Grid       = SpringsteelGrid{CartesianGeometry,   SplineBasisArray, NoBasisArray,      ChebyshevBasisArray}
const RL_Grid       = SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray}
const RR_Grid       = SpringsteelGrid{CartesianGeometry,   SplineBasisArray, SplineBasisArray,  NoBasisArray}
const RLZ_Grid      = SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}
const RRR_Grid      = SpringsteelGrid{CartesianGeometry,   SplineBasisArray, SplineBasisArray,  SplineBasisArray}
const SL_Grid       = SpringsteelGrid{SphericalGeometry,   SplineBasisArray, FourierBasisArray, NoBasisArray}
const SLZ_Grid      = SpringsteelGrid{SphericalGeometry,   SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}
```

# Example
```julia
gp = SpringsteelGridParameters(geometry="RL", num_cells=10,
    iMin=0.0, iMax=100.0,
    vars=Dict("u" => 0), BCL=Dict("u" => CubicBSpline.R0), BCR=Dict("u" => CubicBSpline.R0))
grid = createGrid(gp)   # returns SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray}
spectralTransform!(grid)
gridTransform!(grid)
```

See also: [`createGrid`](@ref), [`SpringsteelGridParameters`](@ref),
[`spectralTransform!`](@ref), [`gridTransform!`](@ref),
[`num_deriv_slots`](@ref)
"""
struct SpringsteelGrid{G <: AbstractGeometry, I, J, K} <: AbstractGrid
    params   :: SpringsteelGridParameters
    ibasis   :: I    # SplineBasisArray | FourierBasisArray | ChebyshevBasisArray | NoBasisArray
    jbasis   :: J
    kbasis   :: K
    spectral :: Array{Float64}
    physical :: Array{Float64}
end

# ────────────────────────────────────────────────────────────────────────────
# num_deriv_slots — count physical-array derivative slots
# ────────────────────────────────────────────────────────────────────────────

"""
    num_deriv_slots(jbasis, kbasis) -> Int

    num_deriv_slots(::NoBasisArray, ::NoBasisArray) -> 3
    num_deriv_slots(::NoBasisArray, _)              -> 5
    num_deriv_slots(_, ::NoBasisArray)              -> 5
    num_deriv_slots(_, _)                           -> 7

Return the number of derivative slots in `physical[:, var, :]` based on which
j- and k-dimension basis containers are active.

| j-basis      | k-basis      | slots | Derivative layout |
|:------------ |:------------ |:-----:|:----------------- |
| `NoBasisArray` | `NoBasisArray` | 3 | `[f, ∂f/∂i, ∂²f/∂i²]` |
| any            | `NoBasisArray` | 5 | `[f, ∂f/∂i, ∂²f/∂i², ∂f/∂j, ∂²f/∂j²]` |
| `NoBasisArray` | any            | 5 | same layout (k mapped to second active dim) |
| any            | any            | 7 | `[f, ∂f/∂i, ∂²f/∂i², ∂f/∂j, ∂²f/∂j², ∂f/∂k, ∂²f/∂k²]` |

# Example
```julia
num_deriv_slots(NoBasisArray(), NoBasisArray())  # → 3  (1D)
num_deriv_slots(FourierBasisArray(...), NoBasisArray())  # → 5  (2D cylindrical)
num_deriv_slots(NoBasisArray(), ChebyshevBasisArray(...))  # → 5  (2D Spline×Cheb)
num_deriv_slots(SplineBasisArray(...), ChebyshevBasisArray(...))  # → 7  (3D)
```

See also: [`SpringsteelGrid`](@ref)
"""
num_deriv_slots(::NoBasisArray, ::NoBasisArray) = 3   # 1D — disambiguator, most specific
num_deriv_slots(::NoBasisArray, _)              = 5   # 2D — j unused, k active (e.g. RZ)
num_deriv_slots(_, ::NoBasisArray)              = 5   # 2D — k unused, j active (e.g. RL, RR)
num_deriv_slots(_, _)                           = 7   # 3D — both active

# ────────────────────────────────────────────────────────────────────────────
# Type aliases — NEW grids (no conflict; activate immediately)
# ────────────────────────────────────────────────────────────────────────────

"""
    SL_Grid

Type alias for a 2-D spherical grid (Spline × Fourier, latitudinally varying
ring sizes).

```julia
const SL_Grid = SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray}
```

See also: [`SLZ_Grid`](@ref), [`RL_Grid`](@ref), [`SpringsteelGrid`](@ref)
"""
const SL_Grid = SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray}

"""
    SLZ_Grid

Type alias for a 3-D spherical grid (Spline × Fourier × Chebyshev).

```julia
const SLZ_Grid = SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}
```

See also: [`SL_Grid`](@ref), [`RLZ_Grid`](@ref), [`SpringsteelGrid`](@ref)
"""
const SLZ_Grid = SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}

# ────────────────────────────────────────────────────────────────────────────
# Type aliases — BACKWARD-COMPATIBLE old grid names (activated in Phase 10)
# Old names → parametric SpringsteelGrid types
# ────────────────────────────────────────────────────────────────────────────

"""
    R_Grid

Type alias for a 1-D Cartesian grid (Spline i-basis only).

```julia
const R_Grid = SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray}
```

See also: [`Spline1D_Grid`](@ref), [`SpringsteelGrid`](@ref)
"""
const R_Grid = SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray}

"""
    Spline1D_Grid

Alias for [`R_Grid`](@ref).  Provided for backward compatibility.

```julia
const Spline1D_Grid = R_Grid
```
"""
const Spline1D_Grid = R_Grid

"""
    RZ_Grid

Type alias for a 2-D Cartesian grid with Spline i-basis and Chebyshev k-basis.

```julia
const RZ_Grid = SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, ChebyshevBasisArray}
```

See also: [`SpringsteelGrid`](@ref)
"""
const RZ_Grid = SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, ChebyshevBasisArray}

"""
    RL_Grid

Type alias for a 2-D cylindrical grid (Spline i-basis, Fourier j-basis).

```julia
const RL_Grid = SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray}
```

See also: [`RLZ_Grid`](@ref), [`SpringsteelGrid`](@ref)
"""
const RL_Grid = SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray}

"""
    RR_Grid

Type alias for a 2-D Cartesian grid with Spline i-basis and Spline j-basis.

```julia
const RR_Grid = SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, NoBasisArray}
```

See also: [`Spline2D_Grid`](@ref), [`SpringsteelGrid`](@ref)
"""
const RR_Grid = SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, NoBasisArray}

"""
    Spline2D_Grid

Alias for [`RR_Grid`](@ref).  Provided for backward compatibility.

```julia
const Spline2D_Grid = RR_Grid
```
"""
const Spline2D_Grid = RR_Grid

"""
    RLZ_Grid

Type alias for a 3-D cylindrical grid (Spline × Fourier × Chebyshev).

```julia
const RLZ_Grid = SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}
```

See also: [`RL_Grid`](@ref), [`SpringsteelGrid`](@ref)
"""
const RLZ_Grid = SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}

"""
    RRR_Grid

Type alias for a 3-D Cartesian grid (Spline × Spline × Spline).

```julia
const RRR_Grid = SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, SplineBasisArray}
```

See also: [`SpringsteelGrid`](@ref)
"""
const RRR_Grid = SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, SplineBasisArray}

module Springsteel

# Functions to define a spectral grid
abstract type AbstractGrid end
abstract type AbstractMultiGrid end

using CSV
using Dates
using DataFrames
using JLD2
using NCDatasets
using SharedArrays
using SparseArrays

#Define some convenient aliases
const real = Float64
const int = Int64
const uint = UInt64

# ── Boundary condition type system ────────────────────────────────────────────
# A single struct encodes every supported BC.  Slots 1–3 constrain u, u', u'';
# slot 4 is a Robin (αu + βu' = γ) condition; the periodic flag is separate.
# `nothing` means "natural / unconstrained" in that slot.

"""
    BoundaryConditions

Basis-agnostic boundary condition specification.  Four semantic slots:

- `u`   : prescribed function value (Dirichlet) — `nothing` if unconstrained
- `du`  : prescribed first derivative (Neumann)  — `nothing` if unconstrained
- `d2u` : prescribed second derivative            — `nothing` if unconstrained
- `robin`: `(α, β, γ)` for the Robin condition `αu + βu' = γ` — `nothing` if unused
- `periodic`: `true` for periodic boundaries

Robin is mutually exclusive with slots 1–3; periodic is mutually exclusive with
all other slots.  Use the convenience constructors [`NaturalBC`](@ref),
[`DirichletBC`](@ref), [`NeumannBC`](@ref), [`SecondDerivativeBC`](@ref),
[`RobinBC`](@ref), [`PeriodicBC`](@ref), [`CauchyBC`](@ref),
[`ExponentialBC`](@ref), [`SymmetricBC`](@ref), [`AntisymmetricBC`](@ref).

The rank (number of constrained derivative orders) is computed automatically by
[`bc_rank`](@ref).
"""
struct BoundaryConditions
    u::Union{Nothing, Float64}
    du::Union{Nothing, Float64}
    d2u::Union{Nothing, Float64}
    robin::Union{Nothing, Tuple{Float64, Float64, Float64}}
    periodic::Bool

    function BoundaryConditions(u, du, d2u, robin, periodic::Bool=false)
        if robin !== nothing && any(!isnothing, (u, du, d2u))
            throw(ArgumentError(
                "Robin BC is mutually exclusive with u/du/d2u constraint slots"))
        end
        if periodic && any(!isnothing, (u, du, d2u, robin))
            throw(ArgumentError(
                "PeriodicBC must have all other slots as nothing"))
        end
        new(u, du, d2u, robin, periodic)
    end
end

# ── Convenience constructors ─────────────────────────────────────────────────

"""No boundary constraint (free). Equivalent to the legacy `R0`."""
NaturalBC() = BoundaryConditions(nothing, nothing, nothing, nothing)

"""Dirichlet BC: prescribe `u(x₀) = v`.  Default `v = 0`."""
DirichletBC(v::Real=0.0) = BoundaryConditions(Float64(v), nothing, nothing, nothing)

"""Neumann BC: prescribe `u'(x₀) = v`.  Default `v = 0`."""
NeumannBC(v::Real=0.0) = BoundaryConditions(nothing, Float64(v), nothing, nothing)

"""Prescribe the second derivative `u''(x₀) = v`.  Default `v = 0`."""
SecondDerivativeBC(v::Real=0.0) = BoundaryConditions(nothing, nothing, Float64(v), nothing)

"""Robin BC: `αu + βu' = γ`.  Default `γ = 0`."""
RobinBC(α::Real, β::Real, γ::Real=0.0) =
    BoundaryConditions(nothing, nothing, nothing,
                       (Float64(α), Float64(β), Float64(γ)))

"""Periodic boundary condition."""
PeriodicBC() = BoundaryConditions(nothing, nothing, nothing, nothing, true)

"""
    CauchyBC(u, du)

Cauchy (compound) BC constraining both value and first derivative.
Equivalent to the legacy `R2T10` when `u = du = 0`.
"""
CauchyBC(u::Real, du::Real) =
    BoundaryConditions(Float64(u), Float64(du), nothing, nothing)

"""
    ExponentialBC(λ)

Ooyama (2002) R1T10(λ): outward exponential decay `u = +λ u'`.
Implemented as `RobinBC(1, -λ, 0)`.
"""
ExponentialBC(λ::Real) = RobinBC(1.0, -Float64(λ), 0.0)

"""Symmetric (reflecting) BC: `u'(x₀) = 0`.  Alias for `NeumannBC(0)`."""
SymmetricBC() = NeumannBC(0.0)

"""
    AntisymmetricBC()

Antisymmetric BC: `u(x₀) = 0` and `u''(x₀) = 0`.
Equivalent to the legacy `R2T20`.
"""
AntisymmetricBC() = BoundaryConditions(0.0, nothing, 0.0, nothing)

"""
    ZerosBC()

Homogeneous rank-3 BC: `u(x₀) = u'(x₀) = u''(x₀) = 0`.
Equivalent to the legacy `R3`.  Permanently eliminates the 3 border degrees
of freedom — no `ahat` mechanism is used.
"""
ZerosBC() = BoundaryConditions(0.0, 0.0, 0.0, nothing)

"""
    FixedBC()
    FixedBC(u::Real, du::Real=0.0, d2u::Real=0.0)

Rank-3 BC constraining value, first derivative, and second derivative.

- `FixedBC()` — interface BC with values determined at runtime by
  `update_interface!`.  Uses NaN sentinels to ensure the R3X (inhomogeneous)
  code path is activated, so the `ahat` mechanism is available.
- `FixedBC(u, du, d2u)` — prescribe specific boundary values.
"""
FixedBC() = BoundaryConditions(NaN, NaN, NaN, nothing)
FixedBC(u::Real, du::Real=0.0, d2u::Real=0.0) =
    BoundaryConditions(Float64(u), Float64(du), Float64(d2u), nothing)

# ── Utility functions ────────────────────────────────────────────────────────

"""
    bc_rank(bc::BoundaryConditions) -> Int

Number of derivative orders constrained.  Robin counts as rank 1.
Periodic returns 0 (rank is handled internally by each basis).
"""
function bc_rank(bc::BoundaryConditions)
    bc.periodic && return 0
    bc.robin !== nothing && return 1
    return count(!isnothing, (bc.u, bc.du, bc.d2u))
end

"""True if the BC is periodic."""
is_periodic(bc::BoundaryConditions) = bc.periodic

"""True if any constrained value is nonzero (inhomogeneous BC)."""
function is_inhomogeneous(bc::BoundaryConditions)
    bc.robin !== nothing && return bc.robin[3] != 0.0
    any(x -> x !== nothing && x != 0.0, (bc.u, bc.du, bc.d2u))
end

# ── End boundary condition type system ───────────────────────────────────────

# These are declared as submodules to avoid namespace clashes with each other and other packages
include("CubicBSpline.jl")
include("Fourier.jl")
include("Chebyshev.jl")
using .CubicBSpline, .Fourier, .Chebyshev

include("basis_cache.jl")
export clear_basis_caches!, basis_cache_sizes

# Define unified generic transform functions at the Springsteel module level.
# Each submodule defines its own version (e.g., CubicBSpline.Btransform!,
# Chebyshev.Btransform!) but those are separate function objects. These
# wrappers create a single dispatch point so downstream code can call
# Btransform!(col) regardless of basis type.
Btransform(obj::CubicBSpline.Spline1D, u::Vector{Float64})  = CubicBSpline.Btransform(obj, u)
Btransform(obj::Chebyshev.Chebyshev1D, u::Vector{Float64})  = Chebyshev.Btransform(obj, u)
Btransform!(obj::CubicBSpline.Spline1D)  = CubicBSpline.Btransform!(obj)
Btransform!(obj::Chebyshev.Chebyshev1D)  = Chebyshev.Btransform!(obj)

Atransform(obj::CubicBSpline.Spline1D, b::AbstractVector)  = CubicBSpline.Atransform(obj, b)
Atransform(obj::Chebyshev.Chebyshev1D, b::AbstractVector)  = Chebyshev.Atransform(obj, b)
Atransform!(obj::CubicBSpline.Spline1D)  = CubicBSpline.Atransform!(obj)
Atransform!(obj::Chebyshev.Chebyshev1D)  = Chebyshev.Atransform!(obj)

Itransform(obj::CubicBSpline.Spline1D, u::AbstractVector)  = CubicBSpline.Itransform(obj, u)
Itransform(obj::Chebyshev.Chebyshev1D, u::AbstractVector)  = Chebyshev.Itransform(obj, u)
Itransform!(obj::CubicBSpline.Spline1D)  = CubicBSpline.Itransform!(obj)
Itransform!(obj::Chebyshev.Chebyshev1D)  = Chebyshev.Itransform!(obj)

Ixtransform(obj::CubicBSpline.Spline1D)  = CubicBSpline.Ixtransform(obj)
Ixtransform(obj::Chebyshev.Chebyshev1D)  = Chebyshev.Ixtransform(obj)

Ixxtransform(obj::CubicBSpline.Spline1D)  = CubicBSpline.Ixxtransform(obj)
Ixxtransform(obj::Chebyshev.Chebyshev1D)  = Chebyshev.Ixxtransform(obj)

IInttransform(obj::CubicBSpline.Spline1D, u::Vector{Float64}, C0::Float64=0.0) = CubicBSpline.IInttransform(obj, u, C0)
IInttransform(obj::Chebyshev.Chebyshev1D, C0::Float64=0.0) = Chebyshev.IInttransform(obj, C0)

export AbstractGrid
export SpringsteelGrid, SpringsteelGridParameters
export CubicBSpline, SplineParameters, Spline1D
export SBtransform, SBtransform!, SAtransform!, SItransform!
export SAtransform, SBxtransform, SItransform, SIxtransform, SIxxtransform
export setMishValues

export FourierParameters, Fourier1D, Fourier
export FBtransform, FBtransform!, FAtransform!, FItransform!
export FBxtransform, FIxtransform, FIxxtransform, FIInttransform

export Chebyshev, ChebyshevParameters, Chebyshev1D
export CBtransform, CBtransform!, CAtransform!, CItransform!
export CBxtransform, CIxtransform, CIxxtransform, CIInttransform

# Generic (basis-agnostic) transform wrappers — dispatch on 1D basis object type
export Btransform, Btransform!, Atransform, Atransform!
export Itransform, Itransform!, Ixtransform, Ixxtransform, IInttransform

export SplineParameters, Spline1D
export createGrid, getGridpoints, calcTileSizes
export read_physical_grid, write_grid, check_grid_dims
export save_grid, load_grid, write_netcdf, read_netcdf
export calcPatchMap, calcHaloMap, allocateSplineBuffer, num_columns
export calcPatchMap_multidim, calcHaloMap_multidim
export sumSpectralTile!, setSpectralTile!, getBorderSpectral, sumSharedSpectral
export spectralTransform!, splineTransform!, tileTransform!, gridTransform!
export regularGridTransform, getRegularGridpoints
export applyFilter!

# Filter type exports
export AbstractFilter, SpectralFilter, GaussianFilter

# Multi-patch exports
export PatchInterface, MultiPatchGrid
export update_interface!, multiGridTransform!
export PatchChain, PatchEmbedded
export AbstractMultiGrid, SpringsteelMultiGrid, createMultiGrid

# Boundary condition type system
export BoundaryConditions, bc_rank, is_periodic, is_inhomogeneous
export NaturalBC, DirichletBC, NeumannBC, SecondDerivativeBC
export RobinBC, PeriodicBC, CauchyBC, ExponentialBC
export SymmetricBC, AntisymmetricBC, ZerosBC, FixedBC

# Unified type system exports
export AbstractGeometry, CartesianGeometry, CylindricalGeometry, SphericalGeometry
export AbstractBasisType, SplineBasisType, FourierBasisType, ChebyshevBasisType, NoBasisType
export SplineBasisArray, FourierBasisArray, ChebyshevBasisArray, NoBasisArray
export SL_Grid, SLZ_Grid
export R_Grid, RL_Grid, RZ_Grid, RR_Grid, RLZ_Grid, RRR_Grid, Spline1D_Grid, Spline2D_Grid
# Geometry-name aliases for spline grids
export Polar_Grid, Cylindrical_Grid, Spline3D_Grid, Samurai_Grid, SphericalShell_Grid, Sphere_Grid
# Fourier-based grids: canonical L / LL / LLZ, descriptive Ring1D / Ring2D / DoublyPeriodic
export L_Grid, LL_Grid, LLZ_Grid
export Ring1D_Grid, Ring2D_Grid, DoublyPeriodic_Grid
# Chebyshev-based grids: canonical Z / ZZ / ZZZ, descriptive Column1D / Column2D / Column3D
export Z_Grid, ZZ_Grid, ZZZ_Grid
export Column1D_Grid, Column2D_Grid, Column3D_Grid

# Factory exports
export parse_geometry, compute_derived_params, convert_to_springsteel_params
export num_deriv_slots

# Basis interface exports
export gridpoints, spectral_dim, physical_dim

# Solver framework exports
export OperatorTerm, AbstractSolverBackend, LocalLinearBackend, OptimizationBackend
export SpringsteelProblem, SpringsteelSolution
export operator_matrix, assemble_operator, assemble_from_equation
export solve, solver_gridpoints

# Interpolation framework exports
export grid_from_regular_data, grid_from_netcdf
export interpolate_to_grid, interpolate_to_grid!
export evaluate_unstructured
export set_boundary_values!
export cartesian_to_cylindrical, cylindrical_to_cartesian
export cartesian_to_cylindrical_3d, cylindrical_to_cartesian_3d
export cartesian_to_spherical, spherical_to_cartesian
export cylindrical_to_spherical, spherical_to_cylindrical
export latlon_to_spherical, spherical_to_latlon

"""
    SpringsteelGridParameters

Configuration structure for spectral grid construction using mixed basis functions.
Uses dimension-agnostic field names (`iMin`/`iMax`, `jMin`/`jMax`, `kMin`/`kMax`)
that map to the i, j, k dimensions of the grid regardless of physical interpretation.

Construct with `@kwdef` keyword syntax.  Many fields are auto-calculated from
primary parameters (marked *auto* below).

# Fields

## Geometry
- `geometry::String = "1D"`: Grid geometry type. Options: `"R"`, `"RZ"`, `"RL"`,
  `"RR"`, `"RLZ"`, `"RRR"`, `"SL"`, `"SLZ"`, `"Z"`, `"ZZ"`, `"ZZZ"`, `"L"`,
  `"LL"`, `"LLZ"`

## i-Dimension (CubicBSpline, Chebyshev, or Fourier depending on geometry)
- `iMin::Float64 = 0.0`: Minimum i-coordinate
- `iMax::Float64 = 0.0`: Maximum i-coordinate
- `num_cells::Int64 = 0`: Number of cubic B-spline cells (spline geometries)
- `mubar::Int64 = 3`: Quadrature points per cell (1–5 for Gauss; any ≥1 for regular)
- `quadrature::Symbol = :gauss`: Quadrature type (`:gauss` or `:regular`)
- `iDim::Int64`: Physical gridpoints (*auto*: `num_cells * mubar`)
- `b_iDim::Int64`: Spectral coefficients (*auto*: `num_cells + 3`)
- `l_q::Dict = Dict("default" => 2.0)`: Spline filter length (per variable)
- `BCL::Dict = CubicBSpline.R0`: Left boundary condition (per variable)
- `BCR::Dict = CubicBSpline.R0`: Right boundary condition (per variable)

## j-Dimension (Fourier or CubicBSpline depending on geometry)
- `jMin::Float64 = 0.0`: Minimum j-coordinate
- `jMax::Float64 = 2π`: Maximum j-coordinate
- `max_wavenumber::Dict = Dict("default" => -1)`: Max Fourier wavenumber (-1 = ring-specific)
- `jDim::Int64 = 0`: Physical gridpoints
- `b_jDim::Int64 = 0`: Spectral modes/coefficients
- `BCU::Dict = Fourier.PERIODIC`: j-left/upper boundary condition (per variable)
- `BCD::Dict = Fourier.PERIODIC`: j-right/lower boundary condition (per variable)

## k-Dimension (Chebyshev or CubicBSpline depending on geometry)
- `kMin::Float64 = 0.0`: Minimum k-coordinate
- `kMax::Float64 = 0.0`: Maximum k-coordinate
- `kDim::Int64 = 0`: Physical gridpoints
- `b_kDim::Int64`: Spectral coefficients (*auto*: anti-aliased from `kDim`)
- `BCB::Dict = Chebyshev.R0`: k-bottom boundary condition (per variable)
- `BCT::Dict = Chebyshev.R0`: k-top boundary condition (per variable)

## Variables
- `vars::Dict = Dict("u" => 1)`: Variable name → index mapping

## Spectral Filters
- `fourier_filter::Dict = Dict()`: Fourier filter config (per variable, `"default"` fallback)
- `chebyshev_filter::Dict = Dict()`: Chebyshev filter config (per variable, `"default"` fallback)

## Tiling (distributed computing)
- `spectralIndexL::Int64 = 1`: Left spectral index for tile
- `spectralIndexR::Int64`: Right spectral index (*auto*)
- `patchOffsetL::Int64`: Left patch offset in gridpoints (*auto*)
- `patchOffsetR::Int64`: Right patch offset (*auto*)
- `tile_num::Int64 = 0`: Tile number identifier

## Regular Output Grid
- `i_regular_out::Int64`: i-points for regular output (*auto*: `num_cells + 1`)
- `j_regular_out::Int64`: j-points for regular output (*auto*: `iDim*2 + 1`)
- `k_regular_out::Int64`: k-points for regular output (*auto*: `kDim + 1`)

# Boundary Condition Options

Boundary conditions can be specified using the basis-agnostic [`BoundaryConditions`](@ref)
type or the legacy module-qualified Dict constants.  Both forms are accepted in the
per-variable Dict values.

**Generic constructors** (recommended):
- `NaturalBC()` — no constraint (free)
- `DirichletBC(v=0)` — fix value
- `NeumannBC(v=0)` — fix first derivative
- `SecondDerivativeBC(v=0)` — fix second derivative
- `RobinBC(α, β, γ=0)` — linear combination `αu + βu' = γ` (CubicBSpline only)
- `ExponentialBC(λ)` — outward decay `u = +λu'` (CubicBSpline only)
- `PeriodicBC()` — periodic
- `CauchyBC(u, du)` — compound: fix value and first derivative (CubicBSpline only)
- `SymmetricBC()` — alias for `NeumannBC(0)`
- `AntisymmetricBC()` — fix value and second derivative to zero (CubicBSpline only)

**Legacy Dict constants** (still supported):
- **CubicBSpline**: `R0`, `R1T0`, `R1T1`, `R1T2`, `R2T10`, `R2T20`, `R3`, `R3X`, `PERIODIC`
- **Fourier**: `PERIODIC`
- **Chebyshev**: `R0`, `R1T0`, `R1T1`

# Example: 1D Spline Grid (generic BCs)
```julia
gp = SpringsteelGridParameters(
    geometry = "R",
    iMin = 0.0, iMax = 10.0,
    num_cells = 20,
    BCL = Dict("u" => DirichletBC()),
    BCR = Dict("u" => DirichletBC()),
    vars = Dict("u" => 1))
grid = createGrid(gp)
```

# Example: 1D Spline Grid (legacy BCs)
```julia
gp = SpringsteelGridParameters(
    geometry = "R",
    iMin = 0.0, iMax = 10.0,
    num_cells = 20,
    BCL = Dict("u" => CubicBSpline.R1T0),
    BCR = Dict("u" => CubicBSpline.R1T0),
    vars = Dict("u" => 1))
grid = createGrid(gp)
```

# Example: 2D Spline × Chebyshev (RZ)
```julia
gp = SpringsteelGridParameters(
    geometry = "RZ",
    iMin = 0.0, iMax = 1.0, num_cells = 15,
    kMin = 0.0, kMax = 1.0, kDim = 20,
    BCL = Dict("u" => CubicBSpline.R1T0),
    BCR = Dict("u" => CubicBSpline.R1T0),
    BCB = Dict("u" => Chebyshev.R1T0),
    BCT = Dict("u" => Chebyshev.R1T0),
    vars = Dict("u" => 1))
```

# Example: 2D Spline × Fourier (RL, cylindrical)
```julia
gp = SpringsteelGridParameters(
    geometry = "RL",
    iMin = 1.0, iMax = 5.0, num_cells = 30,
    max_wavenumber = Dict("v" => 100),
    BCL = Dict("v" => CubicBSpline.R0),
    BCR = Dict("v" => CubicBSpline.R0),
    vars = Dict("v" => 1))
```

See also: [`createGrid`](@ref), [`R_Grid`](@ref)
"""
Base.@kwdef struct SpringsteelGridParameters
    geometry::String = "1D"
    iMin::real = 0.0
    iMax::real = 0.0
    num_cells::int = 0
    mubar::int = 3
    quadrature::Symbol = :gauss
    iDim::int = num_cells * mubar
    b_iDim::int = num_cells + 3
    l_q::Dict = Dict("default" => 2.0)
    BCL::Dict = Dict("default" => CubicBSpline.R0)
    BCR::Dict = Dict("default" => CubicBSpline.R0)
    jMin::real = 0.0
    jMax::real = 2 * π
    max_wavenumber::Dict = Dict("default" => -1) # Default is -1 to indicate ring specific
    jDim::int = 0
    b_jDim::int = 0
    BCU::Dict = Dict("default" => Fourier.PERIODIC)
    BCD::Dict = Dict("default" => Fourier.PERIODIC)
    kMin::real = 0.0
    kMax::real = 0.0
    kDim::int = 0
    b_kDim::int = min(kDim, floor(((2 * kDim) - 1) / 3) + 1)
    BCB::Dict = Dict("default" => Chebyshev.R0)
    BCT::Dict = Dict("default" => Chebyshev.R0)
    vars::Dict = Dict("u" => 1)
    # Spectral filters (per-variable, keyed by variable name or "default")
    fourier_filter::Dict = Dict()
    chebyshev_filter::Dict = Dict()
    # Patch indices
    spectralIndexL::int = 1
    spectralIndexR::int = spectralIndexL + b_iDim - 1
    patchOffsetL::int = (spectralIndexL - 1) * mubar
    patchOffsetR::int = patchOffsetL + iDim
    tile_num::int = 0
    # The default i increment is the number of spline cells
    i_regular_out::int = num_cells + 1
    # The default j_increment is the maximum number of wavenumbers on the outermost ring
    j_regular_out::int = (iDim*2) + 1
    k_regular_out::int = kDim + 1
end

# ── Unified type system ────────────────────────────────────────────────────────────
# Must be included after SpringsteelGridParameters (used as field type in SpringsteelGrid)
include("types.jl")
include("basis_interface.jl")

# ── Grid factory ──────────────────────────────────────────────────────────────────
# Must be included after types.jl (uses geometry/basis sentinel types)
include("factory.jl")

# ── Per-wavenumber ahat registry (shared by transforms and multipatch) ───────
# Must be included after types.jl (uses SpringsteelGrid), before transforms
include("multipatch_registry.jl")

# ── Cartesian and Cylindrical/Spherical transforms ───────────────────────────
# Must be included after factory.jl (uses _RLGrid and related aliases)
include("transforms_cartesian.jl")


# Must be included after factory.jl (uses _RLGrid alias → SpringsteelGrid{...})
include("transforms_cylindrical.jl")

# Must be included after transforms_cylindrical.jl (shares spectral layout conventions)
include("transforms_spherical.jl")

# ── Per-grid scratch buffer registry for transform inner loops ───────────────
# Must be included after the transforms_*.jl files (uses _RLGrid, _SLZGrid, etc.)
include("transform_scratch.jl")

# ── 1D Tiling ───────────────────────────────────────────────────────────────────────
# Must be included after transforms_*.jl (uses num_columns from Cylindrical/Spherical files)
include("tiling.jl")

# ── Solver framework ──────────────────────────────────────────────────────────────────
# Must be included after basis modules and factory.jl
include("solver.jl")

# ── Operator algebra (S1 of the solver refactor) ────────────────────────────────────
# High-level AST for building operators via ∂ᵢ/∂_r/... overloads. Lowers to the
# Vector{OperatorTerm} format consumed by assemble_operator. Additive front-end —
# the low-level solver.jl path is unchanged.
# Must be included after solver.jl (uses OperatorTerm) and types.jl (uses
# SpringsteelGrid and the *Geometry sentinel types).
include("operator_algebra.jl")

export DerivMono, ScaledMono, OperatorExpr
export ∂ᵢ, ∂ⱼ, ∂ₖ, d_i, d_j, d_k
export ∂_x, ∂_y, ∂_z, ∂_r, ∂_θ, ∂_λ
export d_x, d_y, d_z, d_r, d_theta, d_lambda

# ── Stateful linear problem (S2 of the solver refactor) ────────────────────────────
# Adds SpringsteelField, Field alias, Pair-based SpringsteelProblem constructor,
# and solve! that reuses a cached workspace (factorisation, M_eval, BC rows) to
# eliminate the per-call rebuild cost of the legacy solve path.
include("solver_problem.jl")
export SpringsteelField, Field, TypedOperator, solve!
export SparseLinearBackend, KrylovLinearBackend

# ── Interpolation framework ──────────────────────────────────────────────────────────
# Must be included after transforms_*.jl (uses _cheb_eval_pts!) and factory.jl
include("interpolation.jl")

# ── Grid relocation ────────────────────────────────────────────────────────────────
# Must be included after interpolation.jl (uses evaluate_unstructured)
include("relocation.jl")
export relocate_grid, relocate_grid!

# ── Filtering framework ─────────────────────────────────────────────────────────────
# Must be included after transforms_*.jl (uses grid type aliases)
include("filtering.jl")

# ── Multi-patch grid connections ───────────────────────────────────────────────────
# Must be included after transforms_*.jl and types.jl (uses grid types and gridTransform!)
include("multipatch.jl")

# ── Multigrid relocation (must be after both relocation.jl and multipatch.jl) ────
include("relocation_multigrid.jl")


# I/O routines
include("io.jl")

# Module end
end

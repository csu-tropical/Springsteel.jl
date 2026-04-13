```@meta
CurrentModule = Springsteel
```

# Boundary Conditions

Springsteel separates *what* a boundary condition means (prescribe a value,
a derivative, periodicity, a Robin combination) from *how* a particular
basis enforces it (spline `Γ_BC` projection, Chebyshev row replacement,
Fourier periodicity). User code writes a single basis-agnostic
[`BoundaryConditions`](@ref) spec; the grid factory translates it to the
basis-specific internal form at construction time.

## The `BoundaryConditions` type

```@docs
BoundaryConditions
```

`BoundaryConditions` has four constraint slots plus a periodic flag:

| Slot       | Constraint        | Meaning |
|:-----------|:------------------|:--------|
| `u`        | `u(x₀) = v`       | Dirichlet value |
| `du`       | `u'(x₀) = v`      | Neumann (first derivative) |
| `d2u`      | `u''(x₀) = v`     | Second derivative |
| `robin`    | `αu + βu' = γ`    | Three-coefficient linear combination |
| `periodic` | —                 | Periodic boundary (Fourier-only) |

`nothing` in a constraint slot means "unconstrained" (natural boundary).
Robin is mutually exclusive with the first three slots; `periodic` is
mutually exclusive with everything else. These rules are enforced by the
inner constructor.

## Convenience constructors

Use the named constructors rather than calling `BoundaryConditions(...)`
directly — they cover every supported case and fail fast if you try to
combine constraints that don't make sense together.

### No constraint

```@docs
NaturalBC
```

```julia
# A free-boundary spline — no value, derivative, or Robin constraint
BCL = Dict("u" => NaturalBC())
```

### Dirichlet — value

```@docs
DirichletBC
```

```julia
# u(x_left) = 0 (default)
BCL = Dict("u" => DirichletBC())
# u(x_right) = 273.15 (inhomogeneous)
BCR = Dict("u" => DirichletBC(273.15))
```

### Neumann — first derivative

```@docs
NeumannBC
```

```julia
# Zero-flux (symmetric) wall
BCL = Dict("u" => NeumannBC())
# Prescribed gradient
BCR = Dict("u" => NeumannBC(-1.5))
```

### Second derivative

```@docs
SecondDerivativeBC
```

### Robin — linear combination

```@docs
RobinBC
```

```julia
# Newton cooling: αu + βu' = γ
BCR = Dict("T" => RobinBC(1.0, 0.2, 293.15))
```

### Periodic

```@docs
PeriodicBC
```

Periodic is the only legal BC for Fourier dimensions and is automatic on
those dimensions — you rarely need to write it explicitly. The factory
will reject a Fourier dimension paired with a non-periodic
`BoundaryConditions`.

### Compound / aliased forms

```@docs
CauchyBC
ExponentialBC
SymmetricBC
AntisymmetricBC
ZerosBC
FixedBC
```

`ExponentialBC(λ)` implements the Ooyama (2002) outward-decay condition
`u = λ u'` as a Robin constraint. `SymmetricBC` is a zero-Neumann alias;
`AntisymmetricBC` enforces `u = u'' = 0` (rank 2). `ZerosBC` is a rank-3
homogeneous clamp used at grid interiors that should never carry energy.
`FixedBC` is the R3X interface BC — it uses NaN sentinels to activate the
inhomogeneous `ahat` code path so that [`update_interface!`](@ref) can
fill in runtime values from a neighbour patch.

## Specifying BCs on a grid

Per-variable dictionaries are the spec format. Each direction has its
own dict keyed by variable name, with an optional `"default"` fallback:

```julia
using Springsteel

gp = SpringsteelGridParameters(
    geometry  = "RZ",
    iMin      = 0.0, iMax = 1.0, num_cells = 10,
    kMin      = 0.0, kMax = 1.0, kDim = 16,
    vars      = Dict("u" => 1, "v" => 2),
    BCL       = Dict("u" => DirichletBC(),     "v" => NaturalBC()),
    BCR       = Dict("u" => DirichletBC(),     "v" => NaturalBC()),
    BCB       = Dict("u" => NeumannBC(),       "v" => DirichletBC()),
    BCT       = Dict("u" => NeumannBC(),       "v" => DirichletBC(1.0)),
)
grid = createGrid(gp)
```

Direction keys:

| Key   | Side              | Active for |
|:------|:------------------|:-----------|
| `BCL` | i-min (left)      | spline i-axis |
| `BCR` | i-max (right)     | spline i-axis |
| `BCD` | j-min (down)      | spline/Fourier j-axis |
| `BCU` | j-max (up)        | spline/Fourier j-axis |
| `BCB` | k-min (bottom)    | Chebyshev/spline k-axis |
| `BCT` | k-max (top)       | Chebyshev/spline k-axis |

If every variable shares the same BC, use `"default"` instead of listing
each name:

```julia
BCL = Dict("default" => DirichletBC())
```

Variables without an explicit entry fall back to `"default"`; variables
with an entry override the default.

## Basis-specific behaviour

The factory converts each `BoundaryConditions` to the internal form
required by the basis on that dimension:

- **CubicBSpline** — BCs become a `Γ_BC` projection matrix (Ooyama 2002)
  that folds the boundary constraints directly into the operator.
  Rank *k* BCs reduce the system size by *k* spectral coefficients per
  boundary. See the [CubicBSpline](cubicbspline.md) page for the
  underlying representation.
- **Chebyshev** — BCs are applied as **coefficient corrections** during
  the forward transform via a per-column `gammaBC` stripe (vector for
  Dirichlet, full $N \times N$ matrix for Neumann using the Wang et al.
  (1993) global coefficient method). When the solver framework
  assembles an explicit operator, it additionally replaces the boundary
  rows of $\mathbf{L}$ with evaluation or first-derivative constraint
  rows at the boundary CGL nodes. Currently only homogeneous Dirichlet
  (`R1T0` / `DirichletBC()`) and homogeneous Neumann (`R1T1` /
  `NeumannBC()`) are supported on Chebyshev dimensions; higher-rank
  and inhomogeneous BCs throw at grid construction. See the
  [Chebyshev](chebyshev.md) page for the exact `gammaBC` formulation.
- **Fourier** — `PeriodicBC` is automatic; any non-periodic
  `BoundaryConditions` on a Fourier dimension is a configuration error.

The conversion is per-dimension and happens once at [`createGrid`](@ref)
time, so there is no per-transform cost for BC handling.

## Legacy `Dict` boundary specs

Pre-v0.3 code specified BCs as raw Dicts from the submodules:

```julia
BCL = Dict("u" => CubicBSpline.R0)     # legacy natural BC
BCR = Dict("u" => CubicBSpline.R1T0)   # legacy Dirichlet
BCB = Dict("u" => Chebyshev.R1T0)      # legacy Chebyshev Dirichlet
```

These still work in v1.0 — the factory passes raw Dicts through unchanged
and the downstream transforms read them directly. You can mix legacy and
new forms in the same grid spec, or use whichever you prefer. New code
should use the `BoundaryConditions` constructors: they validate eagerly,
make the physical meaning explicit, and don't require knowing the
per-basis Dict keys.

## Inspecting a BC

```@docs
bc_rank
is_periodic
is_inhomogeneous
```

```julia
bc = RobinBC(1.0, 0.2, 293.15)
bc_rank(bc)            # 1
is_periodic(bc)        # false
is_inhomogeneous(bc)   # true (γ ≠ 0)

bc2 = DirichletBC()
bc_rank(bc2)           # 1
is_inhomogeneous(bc2)  # false (v = 0)
```

`bc_rank` returns the number of independent constraint equations the BC
imposes on the spectral coefficients — it's what the spline `Γ_BC`
projection eliminates from the reduced spectral space. Robin and
`FixedBC()` (R3X interface) count as rank 1 and rank 3 respectively.

## See also

- [`SpringsteelGridParameters`](@ref) — where per-variable BC dicts live
- [`createGrid`](@ref) — applies BCs during grid construction
- [Solver Framework](solver.md) — how BCs flow through `solve` / `solve!`

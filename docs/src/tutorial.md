```@meta
CurrentModule = Springsteel
```

# Tutorial

This tutorial demonstrates Springsteel's core workflow through four examples of
increasing complexity. Each example creates a grid, fills it with a Gaussian
function, performs a forward (physical → spectral) and inverse (spectral → physical)
round-trip transform, and examines the results.

A companion Jupyter notebook with visualizations is available at
`notebooks/Springsteel_tutorial.ipynb`.

---

## Example 1 — 1D Spline Grid

The simplest Springsteel grid uses **cubic B-splines** in a single dimension.

### Creating the grid

Use [`SpringsteelGridParameters`](@ref) to configure the grid, then [`createGrid`](@ref) to
build it:

```julia
using Springsteel

gp = SpringsteelGridParameters(
    geometry  = "R",            # 1D radial/Cartesian
    iMin      = -50.0,          # domain left bound
    iMax      = 50.0,           # domain right bound
    num_cells = 30,             # B-spline cells (iDim = 30 × 3 = 90 gridpoints)
    BCL       = Dict("gauss" => CubicBSpline.R0),   # left BC: free boundary (rank-0, no constraint)
    BCR       = Dict("gauss" => CubicBSpline.R0),   # right BC: free boundary (rank-0, no constraint)
    vars      = Dict("gauss" => 1)                   # one variable
)

grid = createGrid(gp)
```

The result is a `SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray}`,
which can also be referred to by the type alias `R_Grid`.

**Physical array dimensions**: `(iDim, n_vars, 3)` — the third dimension holds
`[f, ∂f/∂x, ∂²f/∂x²]`.

### Filling the grid

Use [`getGridpoints`](@ref) to retrieve the physical gridpoint coordinates:

```julia
pts = getGridpoints(grid)   # Vector{Float64} of length iDim

σ = 10.0
for i in eachindex(pts)
    grid.physical[i, 1, 1] = exp(-(pts[i] / σ)^2)
end
```

### Round-trip transform

```julia
spectralTransform!(grid)   # physical → spectral coefficients
gridTransform!(grid)        # spectral → physical + derivatives

# Check accuracy
original = exp.(-(pts ./ σ).^2)
max_error = maximum(abs.(grid.physical[:, 1, 1] .- original))
# max_error ≈ 1e-6 for 30 cells
```

After [`gridTransform!`](@ref), the derivatives are available:
- `grid.physical[:, 1, 2]` — first derivative ∂f/∂x
- `grid.physical[:, 1, 3]` — second derivative ∂²f/∂x²

### Boundary conditions

BC constants follow the **Ooyama (2002)** rank–type naming convention, defined in the
[`CubicBSpline`](@ref CubicBSpline_module) module. The *rank* is the number of
constraints removed at the boundary; the *type* identifies which derivative is
constrained (T0 = value, T1 = first derivative, T2 = second derivative):

| Constant | Ooyama Eq. | Mathematical condition |
|:---|:---|:---|
| `CubicBSpline.R0` | 3.2a | No constraint (free boundary) |
| `CubicBSpline.R1T0` | 3.2b | ``u(x_0) = 0`` (Dirichlet) |
| `CubicBSpline.R1T1` | 3.2c | ``u'(x_0) = 0`` (Neumann) |
| `CubicBSpline.R1T2` | 3.2d | ``u''(x_0) = 0`` |
| `CubicBSpline.R2T10` | 3.2f | ``u(x_0) = u'(x_0) = 0`` (symmetric reflection) |
| `CubicBSpline.R2T20` | 3.2g | ``u(x_0) = u''(x_0) = 0`` (antisymmetric reflection) |
| `CubicBSpline.R3` | 3.2h | ``u = u' = u'' = 0`` at boundary |
| `CubicBSpline.PERIODIC` | §3e | Periodic (cyclically continuous) domain |

---

## Example 2 — 2D Polar Grid (Spline × Fourier)

A **cylindrical** (`"RL"`) grid uses B-splines in the radial direction and Fourier
series in azimuth.

### Key features
- Each radial ring has a **different number of azimuthal gridpoints**: `lpoints = 4 + 4rᵢ`,
  where `rᵢ = r + patchOffsetL`. This avoids over-resolving near the center.
- Fourier BCs are always periodic (set automatically).
- The physical array has **5 derivative slots**: `[f, ∂f/∂r, ∂²f/∂r², ∂f/∂λ, ∂²f/∂λ²]`.

### Grid creation

```julia
gp_rl = SpringsteelGridParameters(
    geometry  = "RL",
    iMin      = 0.0,
    iMax      = 100.0,
    num_cells = 10,
    vars      = Dict("gauss" => 1),
    BCL       = Dict("gauss" => CubicBSpline.R0),
    BCR       = Dict("gauss" => CubicBSpline.R0)
)

grid_rl = createGrid(gp_rl)  # RL_Grid
```

### Filling the grid

For cylindrical grids, gridpoints are packed by ring. Loop over radial
indices, compute ring sizes, and fill sequentially:

```julia
σ = 30.0
iDim = grid_rl.params.iDim
g = 1
for r in 1:iDim
    ri      = r + grid_rl.params.patchOffsetL
    lpoints = 4 + 4 * ri
    r_val   = grid_rl.ibasis.data[1, 1].mishPoints[r]
    val     = exp(-(r_val / σ)^2)
    for l in 1:lpoints
        grid_rl.physical[g, 1, 1] = val
        g += 1
    end
end
```

### Round-trip and gridpoints

```julia
original = copy(grid_rl.physical[:, 1, 1])
spectralTransform!(grid_rl)
gridTransform!(grid_rl)
max_error = maximum(abs.(grid_rl.physical[:, 1, 1] .- original))

# getGridpoints returns an (N, 2) matrix: columns are [r, λ]
pts_rl = getGridpoints(grid_rl)
```

### Evaluating on a regular grid

The cylindrical mish-point layout has irregular spacing (different `lpoints`
per ring). Use [`regularGridTransform`](@ref) to evaluate the spectral
representation on a uniform (r, λ) grid — ideal for plotting or
intercomparison:

```julia
# Default regular grid (uses mish-point radii, uniform λ per ring)
reg_pts = getRegularGridpoints(grid_rl)   # (N, 2) → columns [r, λ]

# Evaluate all variables and derivatives on that grid
reg_vals = regularGridTransform(grid_rl)  # (N, num_vars, 5)

# Or specify custom output points:
r_out = collect(range(0.0, 100.0, length=50))
λ_out = collect(range(0.0, 2π, length=72))
custom_vals = regularGridTransform(grid_rl, r_out, λ_out)
```

The returned array has the same derivative slots as `physical`:
`[f, ∂f/∂r, ∂²f/∂r², ∂f/∂λ, ∂²f/∂λ²]`.

---

## Example 3 — 3D Cylindrical Grid (Spline × Fourier × Chebyshev)

The **RLZ** grid combines all three basis types.

### Key features
- **Chebyshev** parameters: `kMin`, `kMax` set the vertical domain; `kDim` is the
  number of Chebyshev gridpoints.
- The physical array has **7 derivative slots**:
  `[f, ∂f/∂r, ∂²f/∂r², ∂f/∂λ, ∂²f/∂λ², ∂f/∂z, ∂²f/∂z²]`.
- Physical indexing: for each radius `r`, `lpoints` azimuthal points, each with
  `kDim` vertical points packed sequentially.

### Grid creation

```julia
gp_rlz = SpringsteelGridParameters(
    geometry  = "RLZ",
    iMin      = 0.0,
    iMax      = 80.0,
    num_cells = 6,
    kMin      = 0.0,
    kMax      = 20.0,
    kDim      = 10,
    vars      = Dict("gauss" => 1),
    BCL       = Dict("gauss" => CubicBSpline.R0),
    BCR       = Dict("gauss" => CubicBSpline.R0),
    BCB       = Dict("gauss" => Chebyshev.R0),
    BCT       = Dict("gauss" => Chebyshev.R0)
)

grid_rlz = createGrid(gp_rlz)  # RLZ_Grid
```

### Filling the grid

```julia
σ_r = 25.0; σ_z = 5.0
z₀  = (gp_rlz.kMin + gp_rlz.kMax) / 2

iDim = grid_rlz.params.iDim
kDim = grid_rlz.params.kDim
idx  = 1
for r in 1:iDim
    ri      = r + grid_rlz.params.patchOffsetL
    lpoints = 4 + 4 * ri
    r_val   = grid_rlz.ibasis.data[1, 1].mishPoints[r]
    for l in 1:lpoints
        for z in 1:kDim
            z_val = grid_rlz.kbasis.data[1].mishPoints[z]
            grid_rlz.physical[idx, 1, 1] = exp(-(r_val/σ_r)^2 - ((z_val - z₀)/σ_z)^2)
            idx += 1
        end
    end
end
```

### Round-trip

```julia
original = copy(grid_rlz.physical[:, 1, 1])
spectralTransform!(grid_rlz)
gridTransform!(grid_rlz)
max_error = maximum(abs.(grid_rlz.physical[:, 1, 1] .- original))
```

### Evaluating on a regular grid

```julia
reg_pts = getRegularGridpoints(grid_rlz)   # (N, 3) → columns [r, λ, z]
reg_vals = regularGridTransform(grid_rlz)  # (N, num_vars, 7)

# Custom output grid
r_out = collect(range(0.0, 80.0, length=40))
λ_out = collect(range(0.0, 2π, length=36))
z_out = collect(range(0.0, 20.0, length=20))
custom_vals = regularGridTransform(grid_rlz, r_out, λ_out, z_out)
```

The returned array has 7 derivative slots:
`[f, ∂f/∂r, ∂²f/∂r², ∂f/∂λ, ∂²f/∂λ², ∂f/∂z, ∂²f/∂z²]`.

---

## Example 4 — 3D Cartesian Grid (Spline × Spline × Spline) with Two Variables

The **RRR** grid uses cubic B-splines in all three dimensions.

### Key features
- BCs are needed for all three directions: `BCL`/`BCR` (i), `BCU`/`BCD` (j),
  `BCB`/`BCT` (k).
- Physical indexing: `flat = (i-1)*jDim*kDim + (j-1)*kDim + k`.
- **Multiple variables**: assign each a unique integer index in `vars`.

### Grid creation with two variables

```julia
gp_rrr = SpringsteelGridParameters(
    geometry  = "RRR",
    iMin      = -30.0,  iMax = 30.0,
    jMin      = -30.0,  jMax = 30.0,
    kMin      = -30.0,  kMax = 30.0,
    num_cells = 6,
    vars      = Dict("u" => 1, "v" => 2),
    BCL = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
    BCR = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
    BCU = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
    BCD = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
    BCB = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
    BCT = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0)
)

grid_rrr = createGrid(gp_rrr)  # RRR_Grid — 2 variables, 7 derivative slots each
```

### Filling multiple variables

[`getGridpoints`](@ref) returns an `(N, 3)` matrix for 3D Cartesian grids:

```julia
pts = getGridpoints(grid_rrr)   # (N, 3) → columns [x, y, z]

σ_iso = 15.0;  σ_x = 20.0;  σ_y = 10.0

for i in 1:size(pts, 1)
    x, y, z = pts[i, 1], pts[i, 2], pts[i, 3]
    grid_rrr.physical[i, 1, 1] = exp(-(x^2 + y^2 + z^2) / σ_iso^2)
    grid_rrr.physical[i, 2, 1] = exp(-x^2/σ_x^2 - y^2/σ_y^2) * (z / 30.0)
end
```

### Round-trip

```julia
orig_u = copy(grid_rrr.physical[:, 1, 1])
orig_v = copy(grid_rrr.physical[:, 2, 1])

spectralTransform!(grid_rrr)
gridTransform!(grid_rrr)

err_u = maximum(abs.(grid_rrr.physical[:, 1, 1] .- orig_u))
err_v = maximum(abs.(grid_rrr.physical[:, 2, 1] .- orig_v))
```

Both variables are transformed simultaneously — the spectral and inverse transforms
operate on all variables in a single call.

### Evaluating on a regular grid

```julia
reg_pts = getRegularGridpoints(grid_rrr)   # (N, 3) → columns [x, y, z]
reg_vals = regularGridTransform(grid_rrr)  # (N, num_vars, 7)

# Custom output points
x_out = collect(range(-30.0, 30.0, length=50))
y_out = collect(range(-30.0, 30.0, length=50))
z_out = collect(range(-30.0, 30.0, length=50))
custom_vals = regularGridTransform(grid_rrr, x_out, y_out, z_out)
```

All variables are evaluated simultaneously — both `u` and `v` appear in the output.

---

## Summary: Core Workflow

```julia
# 1. Configure
gp = SpringsteelGridParameters(geometry = "...", ...)

# 2. Build
grid = createGrid(gp)

# 3. Fill physical data
grid.physical[:, var_index, 1] .= your_data

# 4. Forward transform
spectralTransform!(grid)

# 5. Inverse transform (fills values + derivatives)
gridTransform!(grid)

# 6. (Optional) Evaluate on a regular output grid
reg_vals = regularGridTransform(grid)
```

All legacy type names (`R_Grid`, `RL_Grid`, `RZ_Grid`, `RR_Grid`, `RLZ_Grid`, `RRR_Grid`)
remain available as type aliases for full backward compatibility.

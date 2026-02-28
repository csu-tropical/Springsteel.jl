# transforms_spherical.jl — Spectral ↔ physical transforms for Spherical SpringsteelGrids
#
# Phase 7: Spherical geometry transforms
#   • SL_Grid  = SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray}
#   • SLZ_Grid = SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}
#
# Provides:
#   • getGridpoints    — (θ, λ) or (θ, λ, z) physical gridpoint coordinates
#   • num_columns      — number of j×k columns (0 for SL, jDim for SLZ)
#   • spectralTransform! / spectralTransform   — physical → spectral
#   • gridTransform!   / gridTransform         — spectral → physical + derivatives
#
# ── Design note ──────────────────────────────────────────────────────────────
# Spherical transforms are structurally identical to their cylindrical counterparts
# (RL/RLZ) with two substitutions:
#   1. Ring size:  `lpoints = ring.params.yDim`  (sin(θ)-based, baked into Fourier1D)
#      vs.         `lpoints = 4 + 4*ri`          (cylindrical)
#   2. Per-ring kmax: `ring.params.kmax`
#      vs.            `r + patchOffsetL`          (cylindrical)
# The spectral array layout (wavenumber-interleaved) and Fourier/Spline operations are
# identical.  The RL/SL convention uses  p = k*2  for wavenumber offsets.
# The RLZ/SLZ convention uses p = (k-1)*2  (TRAP-1 — see REFACTORING_PLAN.md §13.2).
#
# Must be included AFTER types.jl, basis_interface.jl, and factory.jl.

# ── Type aliases for brevity ─────────────────────────────────────────────────
const _SLGrid  = SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray}
const _SLZGrid = SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}

# ═══════════════════════════════════════════════════════════════════════════
# 2D Spherical  (SL_Grid = Spline×Fourier, sin(θ) ring sizes)
# ═══════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────
# getGridpoints — 2D Spherical
# ────────────────────────────────────────────────────────────────────────────

"""
    getGridpoints(grid::SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray}) -> Matrix{Float64}

Return the `(θ, λ)` physical gridpoint pairs for the 2-D spherical (SL) grid.

Each colatitude ring has a different number of azimuthal grid points determined
by `lpoints = 4 * kmax_ring` where `kmax_ring ≈ sin(θ) * max_ri`.

# Returns
A `Matrix{Float64}` of size `(jDim, 2)` where column 1 is the colatitude θ and
column 2 is the longitude λ.  Rows are ordered by colatitude first, then
azimuthal index within each ring.

See also: [`spectralTransform!`](@ref), [`gridTransform!`](@ref)
"""
function getGridpoints(grid::_SLGrid)
    iDim      = grid.params.iDim
    jDim      = grid.params.jDim
    gridpoints = zeros(Float64, jDim, 2)
    g = 1
    for r in 1:iDim
        theta_r = grid.ibasis.data[1, 1].mishPoints[r]
        lpoints = grid.jbasis.data[r, 1].params.yDim
        for l in 1:lpoints
            gridpoints[g, 1] = theta_r
            gridpoints[g, 2] = grid.jbasis.data[r, 1].mishPoints[l]
            g += 1
        end
    end
    return gridpoints
end

# ────────────────────────────────────────────────────────────────────────────
# num_columns — 2D Spherical
# ────────────────────────────────────────────────────────────────────────────

"""
    num_columns(grid::SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray}) -> Int

Return `0` for 2-D spherical grids (no j×k column tiling for the azimuthal dimension).
"""
function num_columns(grid::_SLGrid)
    return 0
end

# ────────────────────────────────────────────────────────────────────────────
# spectralTransform / spectralTransform! — 2D Spherical (SL)
# ────────────────────────────────────────────────────────────────────────────

"""
    spectralTransform!(grid::SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray})
    spectralTransform(grid, physical, spectral)

Forward (physical → spectral) transform for the 2-D spherical (SL) grid.

The transform is structurally identical to the cylindrical RL transform with
sin(θ)-dependent ring sizes:
1. **Fourier stage**: `FBtransform!` on each colatitudinal ring →  Fourier
   B-coefficients per ring.
2. **Spline stage**: `SBtransform!` on the wavenumber-0 spline, then for each
   wavenumber `k = 1..kDim` (where `kDim = iDim + patchOffsetL`), `SBtransform!`
   on the real/imaginary splines.

**Ring sizes** follow `lpoints = 4 * kmax_ring` where
`kmax_ring ≈ round(sin(θ) * max_ri)`.  Near the poles `lpoints` is small (≥ 4);
at the equator `lpoints` reaches its maximum `4 * max_ri`.

**Spectral layout**: wavenumber-interleaved, same as RL:
```
spectral[1 : b_iDim, v]                   — wavenumber 0
spectral[(2k-1)*b_iDim+1 : 2k*b_iDim, v]  — wavenumber k, real
spectral[2k*b_iDim+1 : (2k+1)*b_iDim, v]  — wavenumber k, imaginary
```
Using offset formula `p = k*2` (SL/RL convention — distinct from `(k-1)*2` used in SLZ/RLZ).

See also: [`gridTransform!`](@ref)
"""
function spectralTransform!(grid::_SLGrid)
    spectralTransform(grid, grid.physical, grid.spectral)
    return grid.spectral
end

function spectralTransform(grid::_SLGrid, physical::Array{real}, spectral::Array{real})

    iDim   = grid.params.iDim
    b_iDim = grid.params.b_iDim
    kDim   = iDim + grid.params.patchOffsetL   # global max wavenumber

    for v in values(grid.params.vars)

        # ── Fourier stage: transform each colatitudinal ring ─────────────────
        i = 1
        for r in 1:iDim
            lpoints = grid.jbasis.data[r, v].params.yDim   # sin(θ)-based ring size
            for l in 1:lpoints
                grid.jbasis.data[r, v].uMish[l] = physical[i, v, 1]
                i += 1
            end
            FBtransform!(grid.jbasis.data[r, v])
        end

        # ── Spline stage — wavenumber 0 ──────────────────────────────────────
        grid.ibasis.data[1, v].uMish .= 0.0
        for r in 1:iDim
            grid.ibasis.data[1, v].uMish[r] = grid.jbasis.data[r, v].b[1]
        end
        SBtransform!(grid.ibasis.data[1, v])
        spectral[1:b_iDim, v] .= grid.ibasis.data[1, v].b

        # ── Spline stage — wavenumbers 1..kDim ───────────────────────────────
        for k in 1:kDim
            grid.ibasis.data[2, v].uMish .= 0.0
            grid.ibasis.data[3, v].uMish .= 0.0
            for r in 1:iDim
                # Only rings that can represent wavenumber k contribute
                if k <= grid.jbasis.data[r, v].params.kmax
                    rk = k + 1
                    ik = grid.jbasis.data[r, v].params.bDim - k + 1
                    grid.ibasis.data[2, v].uMish[r] = grid.jbasis.data[r, v].b[rk]
                    grid.ibasis.data[3, v].uMish[r] = grid.jbasis.data[r, v].b[ik]
                end
            end
            SBtransform!(grid.ibasis.data[2, v])
            SBtransform!(grid.ibasis.data[3, v])

            # SL/RL convention: p = k*2  (NOT (k-1)*2 — see TRAP-1 note)
            p  = k * 2
            p1 = ((p - 1) * b_iDim) + 1
            p2 = p * b_iDim
            spectral[p1:p2, v] .= grid.ibasis.data[2, v].b

            p1 = (p * b_iDim) + 1
            p2 = (p + 1) * b_iDim
            spectral[p1:p2, v] .= grid.ibasis.data[3, v].b
        end
    end

    return spectral
end

# ────────────────────────────────────────────────────────────────────────────
# gridTransform / gridTransform! — 2D Spherical (SL)
# ────────────────────────────────────────────────────────────────────────────

"""
    gridTransform!(grid::SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray})
    gridTransform(grid, physical, spectral)

Inverse (spectral → physical) transform for the 2-D spherical (SL) grid,
including colatitudinal and azimuthal derivatives.

Identical in structure to the cylindrical RL inverse with sin(θ)-dependent ring sizes.

**Physical derivative layout** (5 slots):
| Slot | Contents |
|:----:|:-------- |
| 1    | Field values `f(θ, λ)` |
| 2    | `∂f/∂θ` (first colatitudinal derivative) |
| 3    | `∂²f/∂θ²` (second colatitudinal derivative) |
| 4    | `∂f/∂λ` (first azimuthal derivative) |
| 5    | `∂²f/∂λ²` (second azimuthal derivative) |

Near-pole accuracy: BCs enforce `f = 0` at θ = iMin and θ = iMax (R0 BCs).
Functions that vanish at the poles (e.g. `sin(θ)·cos(λ)`) roundtrip to machine precision.

See also: [`spectralTransform!`](@ref)
"""
function gridTransform!(grid::_SLGrid)
    gridTransform(grid, grid.physical, grid.spectral)
    return grid.physical
end

function gridTransform(grid::_SLGrid, physical::Array{real}, spectral::Array{real})

    iDim   = grid.params.iDim
    b_iDim = grid.params.b_iDim
    kDim   = iDim + grid.params.patchOffsetL

    # Buffers for spline-derivative evaluations at each radial gridpoint
    spline_r  = zeros(Float64, iDim, kDim * 2 + 1)
    spline_rr = zeros(Float64, iDim, kDim * 2 + 1)

    for v in values(grid.params.vars)

        # ── Wavenumber 0 ─────────────────────────────────────────────────────
        grid.ibasis.data[1, v].b .= spectral[1:b_iDim, v]
        SAtransform!(grid.ibasis.data[1, v])
        SItransform!(grid.ibasis.data[1, v])
        spline_r[:, 1]  .= SIxtransform(grid.ibasis.data[1, v])
        spline_rr[:, 1] .= SIxxtransform(grid.ibasis.data[1, v])
        for r in 1:iDim
            grid.jbasis.data[r, v].b[1] = grid.ibasis.data[1, v].uMish[r]
        end

        # ── Higher wavenumbers k = 1..kDim ────────────────────────────────────
        for k in 1:kDim
            p  = k * 2   # SL/RL convention: p = k*2

            p1 = ((p - 1) * b_iDim) + 1
            p2 = p * b_iDim
            grid.ibasis.data[2, v].b .= spectral[p1:p2, v]
            SAtransform!(grid.ibasis.data[2, v])
            SItransform!(grid.ibasis.data[2, v])
            spline_r[:, p]  .= SIxtransform(grid.ibasis.data[2, v])
            spline_rr[:, p] .= SIxxtransform(grid.ibasis.data[2, v])

            p1 = (p * b_iDim) + 1
            p2 = (p + 1) * b_iDim
            grid.ibasis.data[3, v].b .= spectral[p1:p2, v]
            SAtransform!(grid.ibasis.data[3, v])
            SItransform!(grid.ibasis.data[3, v])
            spline_r[:, p + 1]  .= SIxtransform(grid.ibasis.data[3, v])
            spline_rr[:, p + 1] .= SIxxtransform(grid.ibasis.data[3, v])

            for r in 1:iDim
                if k <= grid.jbasis.data[r, v].params.kmax   # spherical: ring's kmax
                    rk = k + 1
                    ik = grid.jbasis.data[r, v].params.bDim - k + 1
                    grid.jbasis.data[r, v].b[rk] = grid.ibasis.data[2, v].uMish[r]
                    grid.jbasis.data[r, v].b[ik] = grid.ibasis.data[3, v].uMish[r]
                end
            end
        end

        # ── Field values and azimuthal derivatives ────────────────────────────
        l1 = 0;  l2 = 0
        for r in 1:iDim
            FAtransform!(grid.jbasis.data[r, v])
            FItransform!(grid.jbasis.data[r, v])
            lpoints = grid.jbasis.data[r, v].params.yDim   # spherical ring size
            l1 = l2 + 1
            l2 = l1 + lpoints - 1
            physical[l1:l2, v, 1] .= grid.jbasis.data[r, v].uMish
            physical[l1:l2, v, 4] .= FIxtransform(grid.jbasis.data[r, v])
            physical[l1:l2, v, 5] .= FIxxtransform(grid.jbasis.data[r, v])
        end

        # ── First colatitudinal derivative ∂f/∂θ ─────────────────────────────
        for r in 1:iDim
            grid.jbasis.data[r, v].b[1] = spline_r[r, 1]
        end
        for k in 1:kDim
            p = k * 2
            for r in 1:iDim
                if k <= grid.jbasis.data[r, v].params.kmax
                    rk = k + 1
                    ik = grid.jbasis.data[r, v].params.bDim - k + 1
                    grid.jbasis.data[r, v].b[rk] = spline_r[r, p]
                    grid.jbasis.data[r, v].b[ik] = spline_r[r, p + 1]
                end
            end
        end
        l1 = 0;  l2 = 0
        for r in 1:iDim
            FAtransform!(grid.jbasis.data[r, v])
            FItransform!(grid.jbasis.data[r, v])
            lpoints = grid.jbasis.data[r, v].params.yDim
            l1 = l2 + 1
            l2 = l1 + lpoints - 1
            physical[l1:l2, v, 2] .= grid.jbasis.data[r, v].uMish
        end

        # ── Second colatitudinal derivative ∂²f/∂θ² ──────────────────────────
        for r in 1:iDim
            grid.jbasis.data[r, v].b[1] = spline_rr[r, 1]
        end
        for k in 1:kDim
            p = k * 2
            for r in 1:iDim
                if k <= grid.jbasis.data[r, v].params.kmax
                    rk = k + 1
                    ik = grid.jbasis.data[r, v].params.bDim - k + 1
                    grid.jbasis.data[r, v].b[rk] = spline_rr[r, p]
                    grid.jbasis.data[r, v].b[ik] = spline_rr[r, p + 1]
                end
            end
        end
        l1 = 0;  l2 = 0
        for r in 1:iDim
            FAtransform!(grid.jbasis.data[r, v])
            FItransform!(grid.jbasis.data[r, v])
            lpoints = grid.jbasis.data[r, v].params.yDim
            l1 = l2 + 1
            l2 = l1 + lpoints - 1
            physical[l1:l2, v, 3] .= grid.jbasis.data[r, v].uMish
        end

    end  # for v

    return physical
end

# ═══════════════════════════════════════════════════════════════════════════
# 3D Spherical  (SLZ_Grid = Spline×Fourier×Chebyshev, sin(θ) ring sizes)
# ═══════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────
# getGridpoints — 3D Spherical
# ────────────────────────────────────────────────────────────────────────────

"""
    getGridpoints(grid::SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}) -> Matrix{Float64}

Return the `(θ, λ, z)` physical gridpoint triples for the 3-D spherical (SLZ) grid.

Points are ordered z-fastest: for each ring (θ, λ) the `kDim` vertical levels
are contiguous, analogous to the RLZ physical layout.

# Returns
A `Matrix{Float64}` of size `(kDim*jDim, 3)` where columns are `(θ, λ, z)`.

See also: [`spectralTransform!`](@ref), [`gridTransform!`](@ref)
"""
function getGridpoints(grid::_SLZGrid)
    iDim    = grid.params.iDim
    kDim    = grid.params.kDim
    jDim    = grid.params.jDim
    gridpts = zeros(Float64, kDim * jDim, 3)
    g = 1
    for r in 1:iDim
        theta_r = grid.ibasis.data[1, 1].mishPoints[r]
        lpoints = grid.jbasis.data[r, 1].params.yDim
        for l in 1:lpoints
            l_m = grid.jbasis.data[r, 1].mishPoints[l]
            for z in 1:kDim
                z_m = grid.kbasis.data[1].mishPoints[z]
                gridpts[g, 1] = theta_r
                gridpts[g, 2] = l_m
                gridpts[g, 3] = z_m
                g += 1
            end
        end
    end
    return gridpts
end

# ────────────────────────────────────────────────────────────────────────────
# num_columns — 3D Spherical
# ────────────────────────────────────────────────────────────────────────────

"""
    num_columns(grid::SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}) -> Int

Return the total number of `(λ̃, z̃)` column pairs summed over all colatitudinal rings.
For tiling purposes this equals `jDim` (total azimuthal physical points across all rings).
"""
function num_columns(grid::_SLZGrid)
    return grid.params.jDim
end

# ────────────────────────────────────────────────────────────────────────────
# spectralTransform / spectralTransform! — 3D Spherical (SLZ)
# ────────────────────────────────────────────────────────────────────────────

"""
    spectralTransform!(grid::SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray})
    spectralTransform(grid, physical, spectral)

Forward (physical → spectral) transform for the 3-D spherical (SLZ) grid.

Identical in structure to the cylindrical RLZ transform with sin(θ)-dependent ring sizes:
1. **Chebyshev stage**: `CBtransform!` on each `(θ, λ)` physical column.
2. **Fourier stage**: `FBtransform!` on each `(θ, z_b)` ring per Chebyshev mode.
3. **Spline stage**: Wavenumber-interleaved `SBtransform!` per Chebyshev mode.

**Spectral layout** (z-major, wavenumber-interleaved per z-level):
Uses `p = (k-1)*2` for k ≥ 1 (SLZ/RLZ convention — see TRAP-1 in REFACTORING_PLAN.md §13.2).

**Ring sizes** follow sin(θ): `lpoints = ring.params.yDim`, `kmax_ring = ring.params.kmax`.

See also: [`gridTransform!`](@ref)
"""
function spectralTransform!(grid::_SLZGrid)
    spectralTransform(grid, grid.physical, grid.spectral)
    return grid.spectral
end

function spectralTransform(grid::_SLZGrid, physical::Array{real}, spectral::Array{real})

    kDim_wn = grid.params.iDim + grid.params.patchOffsetL   # global max wavenumber
    kDim    = grid.params.kDim
    b_kDim  = grid.params.b_kDim
    iDim    = grid.params.iDim
    b_iDim  = grid.params.b_iDim

    # Chebyshev coefficient buffer — sized to the maximum ring size
    max_lpoints = 0
    for r in 1:iDim
        max_lpoints = max(max_lpoints, grid.jbasis.data[r, 1].params.yDim)
    end
    tempcb = zeros(Float64, b_kDim, max_lpoints)

    for v in values(grid.params.vars)

        # ── Chebyshev + Fourier stage ────────────────────────────────────────
        i = 1
        for r in 1:iDim
            lpoints = grid.jbasis.data[r, 1].params.yDim   # spherical ring size
            for l in 1:lpoints
                for z in 1:kDim
                    grid.kbasis.data[v].uMish[z] = physical[i, v, 1]
                    i += 1
                end
                tempcb[:, l] .= CBtransform!(grid.kbasis.data[v])
            end
            # Load Chebyshev coefficients into each ring at this radius, then FBtransform!
            for z_b in 1:b_kDim
                for l in 1:lpoints
                    grid.jbasis.data[r, z_b].uMish[l] = tempcb[z_b, l]
                end
                FBtransform!(grid.jbasis.data[r, z_b])
            end
        end

        # ── Spline stage (per z_b Chebyshev coefficient) ─────────────────────
        for z_b in 1:b_kDim
            # Wavenumber 0
            grid.ibasis.data[1, v].uMish .= 0.0
            for r in 1:iDim
                grid.ibasis.data[1, v].uMish[r] = grid.jbasis.data[r, z_b].b[1]
            end
            SBtransform!(grid.ibasis.data[1, v])

            r1 = (z_b - 1) * b_iDim * (1 + kDim_wn * 2) + 1
            r2 = r1 + b_iDim - 1
            spectral[r1:r2, v] .= grid.ibasis.data[1, v].b

            # Wavenumbers 1..kDim_wn
            for k in 1:kDim_wn
                grid.ibasis.data[2, v].uMish .= 0.0
                grid.ibasis.data[3, v].uMish .= 0.0
                for r in 1:iDim
                    # Only rings that can represent wavenumber k contribute
                    if k <= grid.jbasis.data[r, z_b].params.kmax
                        rk = k + 1
                        ik = grid.jbasis.data[r, z_b].params.bDim - k + 1
                        grid.ibasis.data[2, v].uMish[r] = grid.jbasis.data[r, z_b].b[rk]
                        grid.ibasis.data[3, v].uMish[r] = grid.jbasis.data[r, z_b].b[ik]
                    end
                end
                SBtransform!(grid.ibasis.data[2, v])
                SBtransform!(grid.ibasis.data[3, v])

                # SLZ/RLZ convention: p = (k-1)*2  (NOT k*2 — see TRAP-1)
                p  = (k - 1) * 2
                p1 = r2 + 1 + (p * b_iDim)
                p2 = p1 + b_iDim - 1
                spectral[p1:p2, v] .= grid.ibasis.data[2, v].b

                p1 = p2 + 1
                p2 = p1 + b_iDim - 1
                spectral[p1:p2, v] .= grid.ibasis.data[3, v].b
            end
        end
    end

    return spectral
end

# ────────────────────────────────────────────────────────────────────────────
# gridTransform / gridTransform! — 3D Spherical (SLZ)
# ────────────────────────────────────────────────────────────────────────────

"""
    gridTransform!(grid::SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray})
    gridTransform(grid, physical, spectral)

Inverse (spectral → physical) transform for the 3-D spherical (SLZ) grid,
including colatitudinal, azimuthal, and vertical derivatives.

Identical in structure to the cylindrical RLZ inverse with sin(θ)-dependent ring sizes.

**Physical derivative layout** (7 slots):
| Slot | Contents |
|:----:|:-------- |
| 1    | Field values `f(θ, λ, z)` |
| 2    | `∂f/∂θ` (first colatitudinal derivative) |
| 3    | `∂²f/∂θ²` (second colatitudinal derivative) |
| 4    | `∂f/∂λ` (first azimuthal derivative) |
| 5    | `∂²f/∂λ²` (second azimuthal derivative) |
| 6    | `∂f/∂z` (first vertical derivative) |
| 7    | `∂²f/∂z²` (second vertical derivative) |

**SLZ/RLZ wavenumber offset**: `p = (k-1)*2` (TRAP-1 — different from SL/RL's `k*2`).
`kDim_wn = iDim + patchOffsetL` (maximum representable wavenumber).

See also: [`spectralTransform!`](@ref)
"""
function gridTransform!(grid::_SLZGrid)
    gridTransform(grid, grid.physical, grid.spectral)
    return grid.physical
end

function gridTransform(grid::_SLZGrid, physical::Array{real}, spectral::Array{real})

    kDim_wn = grid.params.iDim + grid.params.patchOffsetL
    kDim    = grid.params.kDim
    b_kDim  = grid.params.b_kDim
    iDim    = grid.params.iDim
    b_iDim  = grid.params.b_iDim

    # Spline evaluation buffer [iDim, 3]: slot 1=k0, 2=k_real, 3=k_imag
    splineBuffer = zeros(Float64, iDim, 3)

    for v in values(grid.params.vars)
        for dr in 0:2

            # ── Spline + FAtransform stage ────────────────────────────────────
            for z_b in 1:b_kDim
                r1 = (z_b - 1) * b_iDim * (1 + kDim_wn * 2) + 1
                r2 = r1 + b_iDim - 1

                # Wavenumber 0
                grid.ibasis.data[1, v].b .= spectral[r1:r2, v]
                SAtransform!(grid.ibasis.data[1, v])
                if dr == 0
                    splineBuffer[:, 1] .= SItransform!(grid.ibasis.data[1, v])
                elseif dr == 1
                    splineBuffer[:, 1] .= SIxtransform(grid.ibasis.data[1, v])
                else
                    splineBuffer[:, 1] .= SIxxtransform(grid.ibasis.data[1, v])
                end
                for r in 1:iDim
                    grid.jbasis.data[r, z_b].b[1] = splineBuffer[r, 1]
                end

                # Higher wavenumbers k = 1..kDim_wn
                for k in 1:kDim_wn
                    # SLZ/RLZ convention: p = (k-1)*2
                    p  = (k - 1) * 2
                    p1 = r2 + 1 + (p * b_iDim)
                    p2 = p1 + b_iDim - 1

                    grid.ibasis.data[2, v].b .= spectral[p1:p2, v]
                    SAtransform!(grid.ibasis.data[2, v])
                    if dr == 0
                        splineBuffer[:, 2] .= SItransform!(grid.ibasis.data[2, v])
                    elseif dr == 1
                        splineBuffer[:, 2] .= SIxtransform(grid.ibasis.data[2, v])
                    else
                        splineBuffer[:, 2] .= SIxxtransform(grid.ibasis.data[2, v])
                    end

                    p1 = p2 + 1
                    p2 = p1 + b_iDim - 1
                    grid.ibasis.data[3, v].b .= spectral[p1:p2, v]
                    SAtransform!(grid.ibasis.data[3, v])
                    if dr == 0
                        splineBuffer[:, 3] .= SItransform!(grid.ibasis.data[3, v])
                    elseif dr == 1
                        splineBuffer[:, 3] .= SIxtransform(grid.ibasis.data[3, v])
                    else
                        splineBuffer[:, 3] .= SIxxtransform(grid.ibasis.data[3, v])
                    end

                    for r in 1:iDim
                        if k <= grid.jbasis.data[r, z_b].params.kmax   # spherical
                            rk = k + 1
                            ik = grid.jbasis.data[r, z_b].params.bDim - k + 1
                            grid.jbasis.data[r, z_b].b[rk] = splineBuffer[r, 2]
                            grid.jbasis.data[r, z_b].b[ik] = splineBuffer[r, 3]
                        end
                    end
                end

                # Apply FA transform to all rings at this z-level
                for r in 1:iDim
                    FAtransform!(grid.jbasis.data[r, z_b])
                end
            end  # for z_b

            # ── Fourier + Chebyshev inverse stage ─────────────────────────────
            zi = 1
            for r in 1:iDim
                lpoints    = grid.jbasis.data[r, 1].params.yDim   # spherical ring size
                ringBuffer = zeros(Float64, lpoints, b_kDim)

                for dl in 0:2
                    if dr > 0 && dl > 0
                        continue   # no mixed colatitudinal/azimuthal cross-derivatives
                    end

                    for z_b in 1:b_kDim
                        if dr == 0
                            if dl == 0
                                ringBuffer[:, z_b] .= FItransform!(grid.jbasis.data[r, z_b])
                            elseif dl == 1
                                ringBuffer[:, z_b] .= FIxtransform(grid.jbasis.data[r, z_b])
                            else
                                ringBuffer[:, z_b] .= FIxxtransform(grid.jbasis.data[r, z_b])
                            end
                        else
                            ringBuffer[:, z_b] .= FItransform!(grid.jbasis.data[r, z_b])
                        end
                    end

                    # Chebyshev inverse transform per (θ, λ) column
                    for l in 1:lpoints
                        for z_b in 1:b_kDim
                            grid.kbasis.data[v].b[z_b] = ringBuffer[l, z_b]
                        end
                        CAtransform!(grid.kbasis.data[v])
                        CItransform!(grid.kbasis.data[v])

                        z1 = zi + (l - 1) * kDim
                        z2 = z1 + kDim - 1
                        if dr == 0 && dl == 0
                            physical[z1:z2, v, 1] .= grid.kbasis.data[v].uMish
                            physical[z1:z2, v, 6] .= CIxtransform(grid.kbasis.data[v])
                            physical[z1:z2, v, 7] .= CIxxtransform(grid.kbasis.data[v])
                        elseif dr == 0 && dl == 1
                            physical[z1:z2, v, 4] .= grid.kbasis.data[v].uMish
                        elseif dr == 0 && dl == 2
                            physical[z1:z2, v, 5] .= grid.kbasis.data[v].uMish
                        elseif dr == 1
                            physical[z1:z2, v, 2] .= grid.kbasis.data[v].uMish
                        elseif dr == 2
                            physical[z1:z2, v, 3] .= grid.kbasis.data[v].uMish
                        end
                    end
                end  # for dl

                zi += lpoints * kDim
            end  # for r
        end  # for dr
    end  # for v

    return physical
end

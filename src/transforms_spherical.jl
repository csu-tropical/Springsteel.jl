# transforms_spherical.jl — Spectral ↔ physical transforms for Spherical SpringsteelGrids
#
# Spherical geometry transforms
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
# The RLZ/SLZ convention uses p = (k-1)*2  (see Developer Notes §TRAP-1 for why this differs from RL/SL).
#
# Must be included AFTER types.jl, basis_interface.jl, and factory.jl.

# ── Type aliases for brevity ─────────────────────────────────────────────────
const _SLGrid  = SpringsteelGrid{SphericalGeometry, SplineBasisArray{2}, FourierBasisArray{2}, NoBasisArray}
const _SLZGrid = SpringsteelGrid{SphericalGeometry, SplineBasisArray{2}, FourierBasisArray{2}, ChebyshevBasisArray{1}}

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
    applyFilter!(grid)
    return grid.spectral
end

function spectralTransform(grid::_SLGrid, physical::Array{real}, spectral::Array{real})

    iDim   = grid.params.iDim
    b_iDim = grid.params.b_iDim
    kDim   = iDim + grid.params.patchOffsetL   # global max wavenumber

    for v in 1:size(spectral, 2)

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
    # Vector scratch for SI*xtransform writes — passing a Vector is 0-alloc, while
    # passing a column view of spline_r/spline_rr leaks ~64 B/call from method
    # specialisation against SubArray. Same trick as the RL/RLZ paths.
    spline_scratch = zeros(Float64, iDim)

    has_wn_ahat = _has_wavenumber_ahat(grid)

    for v in 1:size(spectral, 2)

        # ── Wavenumber 0 ─────────────────────────────────────────────────────
        isp0 = grid.ibasis.data[1, v]
        copyto!(isp0.b, view(spectral, 1:b_iDim, v))
        if has_wn_ahat
            isp0.ahat .= _get_wavenumber_ahat(grid, v, 0)
        end
        SAtransform!(isp0)
        SItransform!(isp0)
        SIxtransform(isp0, spline_scratch)
        @inbounds for r in 1:iDim
            spline_r[r, 1] = spline_scratch[r]
        end
        SIxxtransform(isp0, spline_scratch)
        @inbounds for r in 1:iDim
            spline_rr[r, 1] = spline_scratch[r]
        end
        @inbounds for r in 1:iDim
            grid.jbasis.data[r, v].b[1] = isp0.uMish[r]
        end

        # ── Higher wavenumbers k = 1..kDim ────────────────────────────────────
        for k in 1:kDim
            p  = k * 2   # SL/RL convention: p = k*2

            p1 = ((p - 1) * b_iDim) + 1
            p2 = p * b_iDim
            ispR = grid.ibasis.data[2, v]
            copyto!(ispR.b, view(spectral, p1:p2, v))
            if has_wn_ahat
                ispR.ahat .= _get_wavenumber_ahat(grid, v, p)
            end
            SAtransform!(ispR)
            SItransform!(ispR)
            SIxtransform(ispR, spline_scratch)
            @inbounds for r in 1:iDim
                spline_r[r, p] = spline_scratch[r]
            end
            SIxxtransform(ispR, spline_scratch)
            @inbounds for r in 1:iDim
                spline_rr[r, p] = spline_scratch[r]
            end

            p1 = (p * b_iDim) + 1
            p2 = (p + 1) * b_iDim
            ispI = grid.ibasis.data[3, v]
            copyto!(ispI.b, view(spectral, p1:p2, v))
            if has_wn_ahat
                ispI.ahat .= _get_wavenumber_ahat(grid, v, p + 1)
            end
            SAtransform!(ispI)
            SItransform!(ispI)
            SIxtransform(ispI, spline_scratch)
            @inbounds for r in 1:iDim
                spline_r[r, p + 1] = spline_scratch[r]
            end
            SIxxtransform(ispI, spline_scratch)
            @inbounds for r in 1:iDim
                spline_rr[r, p + 1] = spline_scratch[r]
            end

            @inbounds for r in 1:iDim
                if k <= grid.jbasis.data[r, v].params.kmax   # spherical: ring's kmax
                    rk = k + 1
                    ik = grid.jbasis.data[r, v].params.bDim - k + 1
                    grid.jbasis.data[r, v].b[rk] = ispR.uMish[r]
                    grid.jbasis.data[r, v].b[ik] = ispI.uMish[r]
                end
            end
        end

        # ── Field values and azimuthal derivatives ────────────────────────────
        l1 = 0;  l2 = 0
        for r in 1:iDim
            jring = grid.jbasis.data[r, v]
            FAtransform!(jring)
            FItransform!(jring)
            lpoints = jring.params.yDim   # spherical ring size
            l1 = l2 + 1
            l2 = l1 + lpoints - 1
            copyto!(view(physical, l1:l2, v, 1), jring.uMish)
            # Reuse jring.uMish as scratch — its prior contents were just copied above.
            FIxtransform(jring, jring.uMish)
            copyto!(view(physical, l1:l2, v, 4), jring.uMish)
            FIxxtransform(jring, jring.uMish)
            copyto!(view(physical, l1:l2, v, 5), jring.uMish)
        end

        # ── First colatitudinal derivative ∂f/∂θ ─────────────────────────────
        @inbounds for r in 1:iDim
            grid.jbasis.data[r, v].b[1] = spline_r[r, 1]
        end
        for k in 1:kDim
            p = k * 2
            @inbounds for r in 1:iDim
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
            jring = grid.jbasis.data[r, v]
            FAtransform!(jring)
            FItransform!(jring)
            lpoints = jring.params.yDim
            l1 = l2 + 1
            l2 = l1 + lpoints - 1
            copyto!(view(physical, l1:l2, v, 2), jring.uMish)
        end

        # ── Second colatitudinal derivative ∂²f/∂θ² ──────────────────────────
        @inbounds for r in 1:iDim
            grid.jbasis.data[r, v].b[1] = spline_rr[r, 1]
        end
        for k in 1:kDim
            p = k * 2
            @inbounds for r in 1:iDim
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
            jring = grid.jbasis.data[r, v]
            FAtransform!(jring)
            FItransform!(jring)
            lpoints = jring.params.yDim
            l1 = l2 + 1
            l2 = l1 + lpoints - 1
            copyto!(view(physical, l1:l2, v, 3), jring.uMish)
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
Uses `p = (k-1)*2` for k ≥ 1 (SLZ/RLZ convention — see Developer Notes §TRAP-1).

**Ring sizes** follow sin(θ): `lpoints = ring.params.yDim`, `kmax_ring = ring.params.kmax`.

See also: [`gridTransform!`](@ref)
"""
function spectralTransform!(grid::_SLZGrid)
    spectralTransform(grid, grid.physical, grid.spectral)
    applyFilter!(grid)
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

    for v in 1:size(spectral, 2)

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
    # Per-r ringBuffer reused across (dr,dl) iterations. Sized to the largest
    # spherical ring; only the first `lpoints` rows of each column are touched.
    max_lpoints = 0
    for r in 1:iDim
        lp = grid.jbasis.data[r, 1].params.yDim
        if lp > max_lpoints
            max_lpoints = lp
        end
    end
    ringBuffer = zeros(Float64, max_lpoints, b_kDim)
    # Vector scratch for SI*xtransform writes — passing a column view of splineBuffer
    # leaks ~64 B/call from method specialisation. Same trick as RLZ.
    spline_scratch = zeros(Float64, iDim)

    has_wn_ahat = _has_wavenumber_ahat(grid)
    slots_per_z = 1 + 2 * kDim_wn

    for v in 1:size(spectral, 2)
        kcol = grid.kbasis.data[v]
        for dr in 0:2

            # ── Spline + FAtransform stage ────────────────────────────────────
            for z_b in 1:b_kDim
                r1 = (z_b - 1) * b_iDim * (1 + kDim_wn * 2) + 1
                r2 = r1 + b_iDim - 1
                z_slot_base = (z_b - 1) * slots_per_z

                # Wavenumber 0
                isp0 = grid.ibasis.data[1, v]
                copyto!(isp0.b, view(spectral, r1:r2, v))
                if has_wn_ahat
                    isp0.ahat .= _get_wavenumber_ahat(grid, v, z_slot_base + 0)
                end
                SAtransform!(isp0)
                if dr == 0
                    SItransform!(isp0)
                    @inbounds for r in 1:iDim
                        splineBuffer[r, 1] = isp0.uMish[r]
                    end
                elseif dr == 1
                    SIxtransform(isp0, spline_scratch)
                    @inbounds for r in 1:iDim
                        splineBuffer[r, 1] = spline_scratch[r]
                    end
                else
                    SIxxtransform(isp0, spline_scratch)
                    @inbounds for r in 1:iDim
                        splineBuffer[r, 1] = spline_scratch[r]
                    end
                end
                @inbounds for r in 1:iDim
                    grid.jbasis.data[r, z_b].b[1] = splineBuffer[r, 1]
                end

                # Higher wavenumbers k = 1..kDim_wn
                for k in 1:kDim_wn
                    # SLZ/RLZ convention: p = (k-1)*2
                    p  = (k - 1) * 2
                    p1 = r2 + 1 + (p * b_iDim)
                    p2 = p1 + b_iDim - 1

                    ispR = grid.ibasis.data[2, v]
                    copyto!(ispR.b, view(spectral, p1:p2, v))
                    if has_wn_ahat
                        ispR.ahat .= _get_wavenumber_ahat(grid, v, z_slot_base + 1 + p)
                    end
                    SAtransform!(ispR)
                    if dr == 0
                        SItransform!(ispR)
                        @inbounds for r in 1:iDim
                            splineBuffer[r, 2] = ispR.uMish[r]
                        end
                    elseif dr == 1
                        SIxtransform(ispR, spline_scratch)
                        @inbounds for r in 1:iDim
                            splineBuffer[r, 2] = spline_scratch[r]
                        end
                    else
                        SIxxtransform(ispR, spline_scratch)
                        @inbounds for r in 1:iDim
                            splineBuffer[r, 2] = spline_scratch[r]
                        end
                    end

                    p1 = p2 + 1
                    p2 = p1 + b_iDim - 1
                    ispI = grid.ibasis.data[3, v]
                    copyto!(ispI.b, view(spectral, p1:p2, v))
                    if has_wn_ahat
                        ispI.ahat .= _get_wavenumber_ahat(grid, v, z_slot_base + 1 + p + 1)
                    end
                    SAtransform!(ispI)
                    if dr == 0
                        SItransform!(ispI)
                        @inbounds for r in 1:iDim
                            splineBuffer[r, 3] = ispI.uMish[r]
                        end
                    elseif dr == 1
                        SIxtransform(ispI, spline_scratch)
                        @inbounds for r in 1:iDim
                            splineBuffer[r, 3] = spline_scratch[r]
                        end
                    else
                        SIxxtransform(ispI, spline_scratch)
                        @inbounds for r in 1:iDim
                            splineBuffer[r, 3] = spline_scratch[r]
                        end
                    end

                    @inbounds for r in 1:iDim
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
                lpoints = grid.jbasis.data[r, 1].params.yDim   # spherical ring size

                for dl in 0:2
                    if dr > 0 && dl > 0
                        continue   # no mixed colatitudinal/azimuthal cross-derivatives
                    end

                    # Reuse jring.uMish as the FFTW destination for the derivative
                    # ops; its prior contents are not needed past this point.
                    for z_b in 1:b_kDim
                        jring = grid.jbasis.data[r, z_b]
                        if dr == 0
                            if dl == 0
                                FItransform!(jring)
                            elseif dl == 1
                                FIxtransform(jring, jring.uMish)
                            else
                                FIxxtransform(jring, jring.uMish)
                            end
                        else
                            FItransform!(jring)
                        end
                        @inbounds for l in 1:lpoints
                            ringBuffer[l, z_b] = jring.uMish[l]
                        end
                    end

                    # Chebyshev inverse transform per (θ, λ) column
                    for l in 1:lpoints
                        @inbounds for z_b in 1:b_kDim
                            kcol.b[z_b] = ringBuffer[l, z_b]
                        end
                        CAtransform!(kcol)
                        CItransform!(kcol)

                        z1 = zi + (l - 1) * kDim
                        z2 = z1 + kDim - 1
                        if dr == 0 && dl == 0
                            copyto!(view(physical, z1:z2, v, 1), kcol.uMish)
                            # Reuse kcol.uMish as scratch — its prior content was just copied above.
                            CIxtransform(kcol, kcol.uMish)
                            copyto!(view(physical, z1:z2, v, 6), kcol.uMish)
                            CIxxtransform(kcol, kcol.uMish)
                            copyto!(view(physical, z1:z2, v, 7), kcol.uMish)
                        elseif dr == 0 && dl == 1
                            copyto!(view(physical, z1:z2, v, 4), kcol.uMish)
                        elseif dr == 0 && dl == 2
                            copyto!(view(physical, z1:z2, v, 5), kcol.uMish)
                        elseif dr == 1
                            copyto!(view(physical, z1:z2, v, 2), kcol.uMish)
                        elseif dr == 2
                            copyto!(view(physical, z1:z2, v, 3), kcol.uMish)
                        end
                    end
                end  # for dl

                zi += lpoints * kDim
            end  # for r
        end  # for dr
    end  # for v

    return physical
end

# ═══════════════════════════════════════════════════════════════════════════
# Regular-grid output — 2D Spherical (SL)
# ═══════════════════════════════════════════════════════════════════════════

"""
    getRegularGridpoints(grid::_SLGrid) -> Matrix{Float64}

Return an `(n_θ × n_λ, 2)` matrix of uniformly-spaced `(θ, λ)` coordinates
for a 2-D spherical (SL) grid.

Unlike [`getGridpoints`](@ref), which returns the unevenly-spaced spherical mish
points (with sin(θ)-dependent ring sizes), this function returns a regular
tensor-product grid suitable for visualisation and file I/O.

The output dimensions are controlled by [`SpringsteelGridParameters`](@ref) fields:
- `i_regular_out` — colatitude points spanning `[iMin, iMax]`   (default `num_cells + 1`)
- `j_regular_out` — longitude points spanning `[0, 2π)`          (default `iDim * 2 + 1`)

Points are ordered θ-outer, λ-inner (λ varies fastest), matching the layout produced
by [`regularGridTransform`](@ref).

See also: [`regularGridTransform`](@ref), [`getGridpoints`](@ref)
"""
function getRegularGridpoints(grid::_SLGrid)
    n_θ   = grid.params.i_regular_out
    n_λ   = grid.params.j_regular_out
    θ_pts = collect(LinRange(grid.params.iMin, grid.params.iMax, n_θ))
    λ_pts = [2π * (j - 1) / n_λ for j in 1:n_λ]
    pts   = zeros(Float64, n_θ * n_λ, 2)
    idx   = 1
    for i in 1:n_θ
        for j in 1:n_λ
            pts[idx, 1] = θ_pts[i]
            pts[idx, 2] = λ_pts[j]
            idx += 1
        end
    end
    return pts
end

"""
    regularGridTransform(grid::_SLGrid, θ_pts, λ_pts) -> Array{Float64}
    regularGridTransform(grid::_SLGrid, gridpoints)   -> Array{Float64}

Evaluate the SL spectral representation on a regular tensor-product `θ × λ` grid,
returning field values and all five derivatives.

Structurally identical to the cylindrical RL regular-grid transform with the
SL/RL wavenumber convention `p = k*2`.

`grid.spectral` must be populated (call [`spectralTransform!`](@ref) first).

# Returns
`Array{Float64}` of shape `(n_θ × n_λ, nvars, 5)` — λ varies fastest.  Derivative slots:
- `[:,:,1]` — `f(θ, λ)`
- `[:,:,2]` — `∂f/∂θ`
- `[:,:,3]` — `∂²f/∂θ²`
- `[:,:,4]` — `∂f/∂λ`
- `[:,:,5]` — `∂²f/∂λ²`

# Example
```julia
spectralTransform!(grid_sl)
reg_pts  = getRegularGridpoints(grid_sl)
reg_phys = regularGridTransform(grid_sl, reg_pts)
```

See also: [`getRegularGridpoints`](@ref), [`gridTransform!`](@ref)
"""
function regularGridTransform(grid::_SLGrid, θ_pts::AbstractVector{Float64}, λ_pts::AbstractVector{Float64})
    gp     = grid.params
    kDim   = gp.iDim + gp.patchOffsetL
    b_iDim = gp.b_iDim
    nvars  = length(gp.vars)
    n_θ    = length(θ_pts)
    n_λ    = length(λ_pts)
    θ_vec  = collect(Float64, θ_pts)
    λ_vec  = collect(Float64, λ_pts)

    physical = zeros(Float64, n_θ * n_λ, nvars, 5)

    for v in 1:length(gp.vars)
        # Per-wavenumber colatitudinal spline evaluation at θ_pts.
        # Layout: ak[θ_out, k_slot] where k_slot=1 → k=0,
        #   k_slot=2k → cos wavenumber k,  k_slot=2k+1 → sin wavenumber k.
        ak    = zeros(Float64, n_θ, 2 * kDim + 1)
        ak_r  = zeros(Float64, n_θ, 2 * kDim + 1)
        ak_rr = zeros(Float64, n_θ, 2 * kDim + 1)

        # Wavenumber 0 — SL spectral layout: b[1:b_iDim]
        sp0 = grid.ibasis.data[1, v]
        sp0.b .= view(grid.spectral, 1:b_iDim, v)
        SAtransform!(sp0)
        SItransform(sp0,   θ_vec, view(ak,    :, 1))
        SIxtransform(sp0,  θ_vec, view(ak_r,  :, 1))
        SIxxtransform(sp0, θ_vec, view(ak_rr, :, 1))

        # Wavenumbers 1..kDim — SL/RL layout uses p = k*2 (see TRAP-1 note)
        for k in 1:kDim
            p   = k * 2
            p1c = (p - 1) * b_iDim + 1;  p2c = p       * b_iDim   # cosine block
            p1s =  p      * b_iDim + 1;  p2s = (p + 1) * b_iDim   # sine block

            spc = grid.ibasis.data[2, v]
            spc.b .= view(grid.spectral, p1c:p2c, v)
            SAtransform!(spc)
            SItransform(spc,   θ_vec, view(ak,    :, 2k))
            SIxtransform(spc,  θ_vec, view(ak_r,  :, 2k))
            SIxxtransform(spc, θ_vec, view(ak_rr, :, 2k))

            sps = grid.ibasis.data[3, v]
            sps.b .= view(grid.spectral, p1s:p2s, v)
            SAtransform!(sps)
            SItransform(sps,   θ_vec, view(ak,    :, 2k + 1))
            SIxtransform(sps,  θ_vec, view(ak_r,  :, 2k + 1))
            SIxxtransform(sps, θ_vec, view(ak_rr, :, 2k + 1))
        end

        # Reconstruct on regular (θ, λ) grid — λ varies fastest
        # NOTE: Fourier B-coefficients from FBtransform use FFTW R2HC convention
        # where k≥1 cosine/sine amplitudes are stored at half their physical-space
        # amplitude (the inverse FFT in FItransform handles this automatically, but
        # this analytical reconstruction must multiply k≥1 terms by 2).
        idx = 1
        for i in 1:n_θ
            for j in 1:n_λ
                λ   = λ_vec[j]
                f   = ak[i, 1]
                fr  = ak_r[i, 1]
                frr = ak_rr[i, 1]
                fλ  = 0.0
                fλλ = 0.0
                for k in 1:kDim
                    ck   = cos(k * λ);  sk = sin(k * λ)
                    rc   = ak[i, 2k];   rs = ak[i, 2k + 1]
                    f   += 2.0 * (rc * ck + rs * sk)
                    fr  += 2.0 * (ak_r[i, 2k] * ck + ak_r[i, 2k + 1] * sk)
                    frr += 2.0 * (ak_rr[i, 2k] * ck + ak_rr[i, 2k + 1] * sk)
                    fλ  += 2.0 * k * (-rc * sk + rs * ck)
                    fλλ -= 2.0 * k^2 * (rc * ck + rs * sk)
                end
                physical[idx, v, 1] = f
                physical[idx, v, 2] = fr
                physical[idx, v, 3] = frr
                physical[idx, v, 4] = fλ
                physical[idx, v, 5] = fλλ
                idx += 1
            end
        end
    end

    return physical
end

function regularGridTransform(grid::_SLGrid, gridpoints::AbstractMatrix{Float64})
    θ_pts = sort(unique(gridpoints[:, 1]))
    n_θ   = length(θ_pts)
    n_λ   = div(size(gridpoints, 1), n_θ)
    λ_pts = gridpoints[1:n_λ, 2]
    return regularGridTransform(grid, θ_pts, λ_pts)
end

# ═══════════════════════════════════════════════════════════════════════════
# Regular-grid output — 3D Spherical (SLZ)
# ═══════════════════════════════════════════════════════════════════════════

"""
    getRegularGridpoints(grid::_SLZGrid) -> Matrix{Float64}

Return an `(n_θ × n_λ × n_z, 3)` matrix of uniformly-spaced `(θ, λ, z)` coordinates
for a 3-D spherical (SLZ) grid.

The output dimensions are controlled by [`SpringsteelGridParameters`](@ref) fields:
- `i_regular_out` — colatitude points in `[iMin, iMax]`   (default `num_cells + 1`)
- `j_regular_out` — longitude points in `[0, 2π)`          (default `iDim * 2 + 1`)
- `k_regular_out` — vertical points in `[kMin, kMax]`      (default `kDim + 1`)

Points are ordered θ-outer, λ-middle, z-inner (z varies fastest), matching the layout
of [`regularGridTransform`](@ref).

See also: [`regularGridTransform`](@ref), [`getGridpoints`](@ref)
"""
function getRegularGridpoints(grid::_SLZGrid)
    n_θ   = grid.params.i_regular_out
    n_λ   = grid.params.j_regular_out
    n_z   = grid.params.k_regular_out
    θ_pts = collect(LinRange(grid.params.iMin, grid.params.iMax, n_θ))
    λ_pts = [2π * (j - 1) / n_λ for j in 1:n_λ]
    z_pts = collect(LinRange(grid.params.kMin, grid.params.kMax, n_z))
    pts   = zeros(Float64, n_θ * n_λ * n_z, 3)
    idx   = 1
    for i in 1:n_θ
        for j in 1:n_λ
            for k in 1:n_z
                pts[idx, 1] = θ_pts[i]
                pts[idx, 2] = λ_pts[j]
                pts[idx, 3] = z_pts[k]
                idx += 1
            end
        end
    end
    return pts
end

"""
    regularGridTransform(grid::_SLZGrid, θ_pts, λ_pts, z_pts) -> Array{Float64}
    regularGridTransform(grid::_SLZGrid, gridpoints)           -> Array{Float64}

Evaluate the SLZ spectral representation on a regular tensor-product `θ × λ × z` grid.

Structurally identical to the cylindrical RLZ regular-grid transform with the
SLZ/RLZ wavenumber convention `p = (k-1)*2`.

`grid.spectral` must be populated (call [`spectralTransform!`](@ref) first).

# Returns
`Array{Float64}` of shape `(n_θ × n_λ × n_z, nvars, 7)` — z varies fastest.  Slots:
- `[:,:,1]` — `f`, `[:,:,2]` — `∂f/∂θ`, `[:,:,3]` — `∂²f/∂θ²`
- `[:,:,4]` — `∂f/∂λ`, `[:,:,5]` — `∂²f/∂λ²`
- `[:,:,6]` — `∂f/∂z`, `[:,:,7]` — `∂²f/∂z²`

# Example
```julia
spectralTransform!(grid_slz)
reg_pts  = getRegularGridpoints(grid_slz)
reg_phys = regularGridTransform(grid_slz, reg_pts)
```

See also: [`getRegularGridpoints`](@ref), [`gridTransform!`](@ref)
"""
function regularGridTransform(grid::_SLZGrid, θ_pts::AbstractVector{Float64},
                               λ_pts::AbstractVector{Float64}, z_pts::AbstractVector{Float64})
    gp       = grid.params
    kDim_wn  = gp.iDim + gp.patchOffsetL
    b_iDim   = gp.b_iDim
    b_kDim   = gp.b_kDim
    nvars    = length(gp.vars)
    n_θ      = length(θ_pts)
    n_λ      = length(λ_pts)
    n_z      = length(z_pts)
    θ_vec    = collect(Float64, θ_pts)
    λ_vec    = collect(Float64, λ_pts)
    z_vec    = collect(Float64, z_pts)
    n_kslots = 1 + 2 * kDim_wn   # k=0 + (cos + sin) for k = 1..kDim_wn

    physical = zeros(Float64, n_θ * n_λ * n_z, nvars, 7)

    for v in 1:length(gp.vars)
        for dr in 0:2
            # ── Step 1: radial spline evaluation at θ_pts ────────────────────
            # spline_vals[θ_out, z_b, k_slot]: k_slot=1→k=0, 2k→cos k, 2k+1→sin k
            spline_vals = zeros(Float64, n_θ, b_kDim, n_kslots)

            for z_b in 1:b_kDim
                r1 = (z_b - 1) * b_iDim * (1 + kDim_wn * 2) + 1
                r2 = r1 + b_iDim - 1

                # Use ibasis.data[1,v], [2,v], [3,v] as scratch splines.
                # SLZ ibasis has shape [b_kDim, nvars]; indices 1..3 are always valid.
                sp0 = grid.ibasis.data[1, v]
                sp0.b .= view(grid.spectral, r1:r2, v)
                SAtransform!(sp0)
                if dr == 0; SItransform(sp0,   θ_vec, view(spline_vals, :, z_b, 1))
                elseif dr == 1; SIxtransform(sp0,  θ_vec, view(spline_vals, :, z_b, 1))
                else;           SIxxtransform(sp0, θ_vec, view(spline_vals, :, z_b, 1)); end

                for k in 1:kDim_wn
                    p  = (k - 1) * 2            # SLZ/RLZ convention: p = (k-1)*2
                    p1 = r2 + 1 + p * b_iDim;  p2 = p1 + b_iDim - 1
                    spc = grid.ibasis.data[2, v]
                    spc.b .= view(grid.spectral, p1:p2, v)
                    SAtransform!(spc)
                    if dr == 0; SItransform(spc,   θ_vec, view(spline_vals, :, z_b, 2k))
                    elseif dr == 1; SIxtransform(spc,  θ_vec, view(spline_vals, :, z_b, 2k))
                    else;           SIxxtransform(spc, θ_vec, view(spline_vals, :, z_b, 2k)); end

                    p1 = p2 + 1;  p2 = p1 + b_iDim - 1
                    sps = grid.ibasis.data[3, v]
                    sps.b .= view(grid.spectral, p1:p2, v)
                    SAtransform!(sps)
                    if dr == 0; SItransform(sps,   θ_vec, view(spline_vals, :, z_b, 2k + 1))
                    elseif dr == 1; SIxtransform(sps,  θ_vec, view(spline_vals, :, z_b, 2k + 1))
                    else;           SIxxtransform(sps, θ_vec, view(spline_vals, :, z_b, 2k + 1)); end
                end
            end

            # ── Steps 2 & 3: Fourier sum + Chebyshev evaluation ──────────────
            for dl in 0:2
                if dr > 0 && dl > 0; continue; end   # no mixed θ-λ cross-derivatives

                # NOTE: Fourier B-coefficients use FFTW R2HC convention where k≥1
                # amplitudes are half the physical-space amplitude.
                fourier_b = zeros(Float64, n_θ, n_λ, b_kDim)
                for ti in 1:n_θ
                    for j in 1:n_λ
                        λ = λ_vec[j]
                        for z_b in 1:b_kDim
                            val = (dl == 0) ? spline_vals[ti, z_b, 1] : 0.0
                            for k in 1:kDim_wn
                                rc = spline_vals[ti, z_b, 2k]
                                rs = spline_vals[ti, z_b, 2k + 1]
                                ck = cos(k * λ);  sk = sin(k * λ)
                                if dl == 0
                                    val += 2.0 * (rc * ck + rs * sk)
                                elseif dl == 1
                                    val += 2.0 * k * (-rc * sk + rs * ck)
                                else
                                    val -= 2.0 * k^2 * (rc * ck + rs * sk)
                                end
                            end
                            fourier_b[ti, j, z_b] = val
                        end
                    end
                end

                cheb_col = grid.kbasis.data[v]
                for ti in 1:n_θ
                    for j in 1:n_λ
                        for z_b in 1:b_kDim
                            cheb_col.b[z_b] = fourier_b[ti, j, z_b]
                        end
                        CAtransform!(cheb_col)
                        flat = (ti - 1) * n_λ * n_z + (j - 1) * n_z + 1
                        out  = view(physical, flat:flat + n_z - 1, v, :)
                        if dr == 0 && dl == 0
                            _cheb_eval_pts!(cheb_col, z_vec, view(out, :, 1))
                            _cheb_dz_pts!(cheb_col,   z_vec, view(out, :, 6))
                            _cheb_dzz_pts!(cheb_col,  z_vec, view(out, :, 7))
                        elseif dr == 0 && dl == 1
                            _cheb_eval_pts!(cheb_col, z_vec, view(out, :, 4))
                        elseif dr == 0 && dl == 2
                            _cheb_eval_pts!(cheb_col, z_vec, view(out, :, 5))
                        elseif dr == 1
                            _cheb_eval_pts!(cheb_col, z_vec, view(out, :, 2))
                        else
                            _cheb_eval_pts!(cheb_col, z_vec, view(out, :, 3))
                        end
                    end
                end
            end   # dl
        end   # dr
    end   # v

    return physical
end

function regularGridTransform(grid::_SLZGrid, gridpoints::AbstractMatrix{Float64})
    θ_pts = sort(unique(gridpoints[:, 1]))
    λ_pts = sort(unique(gridpoints[:, 2]))
    z_pts = sort(unique(gridpoints[:, 3]))
    return regularGridTransform(grid, θ_pts, λ_pts, z_pts)
end

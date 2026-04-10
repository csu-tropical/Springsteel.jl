# transforms_cylindrical.jl — Spectral ↔ physical transforms for Cylindrical SpringsteelGrids
#
# Covers:
#   • 2D Cylindrical (Spline×Fourier = RL_Grid)
#   • 3D Cylindrical (Spline×Fourier×Chebyshev = RLZ_Grid)
#
# Provides:
#   • getGridpoints          — (r, λ) physical gridpoint pairs
#   • num_columns            — number of j×k columns (0 for RL)
#   • spectralTransform!     — physical → spectral (in-place, grid's own arrays)
#   • spectralTransform      — physical → spectral (explicit-array variant)
#   • gridTransform!         — spectral → physical + derivatives (in-place)
#   • gridTransform          — spectral → physical + derivatives (explicit-array variant)
#
# Transform order (forward / RL):
#   1. FBtransform! on each Fourier ring (one per radius) → Fourier coefficients per r
#   2. SBtransform! on wavenumber-0 spline
#   3. For each wavenumber k = 1..kDim: SBtransform! on real and imaginary splines
#   4. Interleave into spectral array: [k0 coeffs | k1-real | k1-imag | k2-real | ...]
#
# Transform order (inverse / RL):
#   1. SAtransform! + SItransform! on wavenumber-0 and all higher splines → ring values
#   2. Reconstruct b-coefficients per ring from spline evaluation
#   3. FAtransform! + FItransform! per ring → physical values and azimuthal derivatives
#   4. Repeat with spline-derivative coefficients for radial (∂/∂r, ∂²/∂r²) derivatives
#
# Physical array derivative layout (5 slots):
#   physical[:, v, 1] = field values
#   physical[:, v, 2] = ∂f/∂r  (first radial derivative)
#   physical[:, v, 3] = ∂²f/∂r² (second radial derivative)
#   physical[:, v, 4] = ∂f/∂λ  (first azimuthal derivative)
#   physical[:, v, 5] = ∂²f/∂λ² (second azimuthal derivative)
#
# ⚠️ ACCURACY NOTE — wavenumber offset formula:
#   RL uses  p = k*2  for k ≥ 1 (NOT (k-1)*2 — that is RLZ's convention).
#   Do NOT unify these: the spectral array layouts differ between RL and RLZ.
#
# Must be included AFTER types.jl, basis_interface.jl, and factory.jl.

# ── Type alias for brevity ────────────────────────────────────────────────────
const _RLGrid = SpringsteelGrid{CylindricalGeometry, SplineBasisArray{2}, FourierBasisArray{2}, NoBasisArray}

# ────────────────────────────────────────────────────────────────────────────
# getGridpoints
# ────────────────────────────────────────────────────────────────────────────

"""
    getGridpoints(grid::SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray}) -> Matrix{Float64}

Return the `(r, λ)` physical gridpoint pairs for the 2-D cylindrical grid.

Each radius has a different number of azimuthal grid points (ring size
`4 + 4rᵢ` where `rᵢ = r + patchOffsetL`), so the result is a flat matrix of
all `(r, λ)` pairs in physical-array order.

# Returns
A `Matrix{Float64}` of size `(jDim, 2)` where column 1 is the radial coordinate
and column 2 is the azimuthal (λ) coordinate.  Rows are ordered by radius first,
then azimuthal index within each ring.

# Example
```julia
gp   = SpringsteelGridParameters(geometry="RL", num_cells=10,
                                  iMin=0.0, iMax=100.0,
                                  vars=Dict("u"=>1),
                                  BCL=Dict("u"=>CubicBSpline.R0),
                                  BCR=Dict("u"=>CubicBSpline.R0))
grid = createGrid(gp)
pts  = getGridpoints(grid)
size(pts) == (grid.params.jDim, 2)   # true
```

See also: [`spectralTransform!`](@ref), [`gridTransform!`](@ref)
"""
function getGridpoints(grid::_RLGrid)
    gridpoints = zeros(Float64, grid.params.jDim, 2)
    g = 1
    for r in 1:grid.params.iDim
        r_m = grid.ibasis.data[1, 1].mishPoints[r]
        ri  = r + grid.params.patchOffsetL
        lpoints = 4 + 4*ri
        for l in 1:lpoints
            l_m = grid.jbasis.data[r, 1].mishPoints[l]
            gridpoints[g, 1] = r_m
            gridpoints[g, 2] = l_m
            g += 1
        end
    end
    return gridpoints
end

# ────────────────────────────────────────────────────────────────────────────
# num_columns
# ────────────────────────────────────────────────────────────────────────────

"""
    num_columns(grid::SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray}) -> Int

Return the number of j×k columns for tiling purposes.  For 2-D cylindrical
grids (RL) this is always `0` because the azimuthal dimension is not tiled.

See also: [`RL_Grid`](@ref)
"""
function num_columns(grid::_RLGrid)
    return 0
end

# ────────────────────────────────────────────────────────────────────────────
# spectralTransform (explicit-array helper)
# ────────────────────────────────────────────────────────────────────────────

"""
    spectralTransform!(grid::SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray})
    spectralTransform(grid, physical, spectral)

Forward (physical → spectral) transform for the 2-D cylindrical (RL) grid.

The transform proceeds in two stages per variable:
1. **Fourier stage**: `FBtransform!` on each radial ring, producing Fourier
   B-coefficients (real and imaginary parts) per ring.
2. **Spline stage**: `SBtransform!` on the wavenumber-0 spline from the
   ring-averaged values, then `SBtransform!` on the real/imaginary splines for
   each wavenumber `k = 1..kDim`.

The resulting spline B-coefficients are stored in the spectral array using the
wavenumber-interleaved layout:
```
spectral[1 : b_iDim, v]                  — wavenumber 0
spectral[b_iDim+1 : 2*b_iDim, v]         — wavenumber 1, real part
spectral[2*b_iDim+1 : 3*b_iDim, v]       — wavenumber 1, imaginary part
spectral[(2k-1)*b_iDim+1 : 2k*b_iDim, v] — wavenumber k, real part
spectral[2k*b_iDim+1 : (2k+1)*b_iDim, v] — wavenumber k, imaginary part
```

`kDim = iDim + patchOffsetL` is the largest wavenumber that has at least one
ring that can represent it.

⚠️ The offset formula is `p = k*2` for RL (NOT `(k-1)*2`).  The 2-D (RL) and
3-D (RLZ) layouts use different offset formulas because they were developed
independently: RL counts from the absolute array start, while RLZ counts from
the end of the k=0 block within each z-level.  See Developer Notes §TRAP-1.

# Arguments
- `grid`: A cylindrical [`SpringsteelGrid`](@ref) with `CylindricalGeometry`.
- `physical` (explicit variant): Physical-space array, size `(jDim, nvars, nderiv)`.
- `spectral` (explicit variant): Spectral array to fill, size `(b_jDim, nvars)`.

# Returns
`spectral` array (in-place variant modifies `grid.spectral`).

See also: [`gridTransform!`](@ref)
"""
function spectralTransform!(grid::_RLGrid)
    spectralTransform(grid, grid.physical, grid.spectral)
    applyFilter!(grid)
    return grid.spectral
end

function spectralTransform(grid::_RLGrid, physical::Array{real}, spectral::Array{real})

    # Need to include patchOffset to get all available wavenumbers
    kDim = grid.params.iDim + grid.params.patchOffsetL

    for v in 1:size(spectral, 2)
        # ── Fourier stage: transform each ring ──────────────────────────────
        i = 1
        for r in 1:grid.params.iDim
            ri      = r + grid.params.patchOffsetL
            lpoints = 4 + 4*ri
            for l in 1:lpoints
                grid.jbasis.data[r, v].uMish[l] = physical[i, v, 1]
                i += 1
            end
            FBtransform!(grid.jbasis.data[r, v])
        end

        b_iDim = grid.params.b_iDim

        # ── Spline stage — wavenumber 0 ──────────────────────────────────────
        isp0 = grid.ibasis.data[1, v]
        isp0.uMish .= 0.0
        @inbounds for r in 1:grid.params.iDim
            isp0.uMish[r] = grid.jbasis.data[r, v].b[1]
        end
        SBtransform!(isp0)
        @inbounds for k in 0:(b_iDim - 1)
            spectral[1 + k, v] = isp0.b[k + 1]
        end

        # ── Spline stage — wavenumbers 1..kDim ───────────────────────────────
        ispR = grid.ibasis.data[2, v]
        ispI = grid.ibasis.data[3, v]
        for k in 1:kDim
            ispR.uMish .= 0.0
            ispI.uMish .= 0.0
            @inbounds for r in 1:grid.params.iDim
                if k <= r + grid.params.patchOffsetL
                    rk = k + 1                                     # real part index in b
                    ik = grid.jbasis.data[r, v].params.bDim - k + 1  # imag part index
                    ispR.uMish[r] = grid.jbasis.data[r, v].b[rk]
                    ispI.uMish[r] = grid.jbasis.data[r, v].b[ik]
                end
            end
            SBtransform!(ispR)
            SBtransform!(ispI)

            # Interleaved layout: p = k*2  (RL convention — see TRAP-1 note)
            p  = k * 2
            p1 = ((p - 1) * b_iDim) + 1
            @inbounds for k2 in 0:(b_iDim - 1)
                spectral[p1 + k2, v] = ispR.b[k2 + 1]
            end

            p1 = (p * b_iDim) + 1
            @inbounds for k2 in 0:(b_iDim - 1)
                spectral[p1 + k2, v] = ispI.b[k2 + 1]
            end
        end
    end

    return spectral
end

# ────────────────────────────────────────────────────────────────────────────
# gridTransform (explicit-array helper)
# ────────────────────────────────────────────────────────────────────────────

"""
    gridTransform!(grid::SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray})
    gridTransform(grid, physical, spectral)

Inverse (spectral → physical) transform for the 2-D cylindrical (RL) grid,
including radial and azimuthal derivatives.

The inverse transform:
1. Reads wavenumber-interleaved spline coefficients from `spectral`.
2. `SAtransform!` + `SItransform!` on each wavenumber spline → ring b-coefficients.
3. `FAtransform!` + `FItransform!` on each ring → physical values (slot 1),
   `FIxtransform` → ∂f/∂λ (slot 4), `FIxxtransform` → ∂²f/∂λ² (slot 5).
4. Repeat steps 2–3 using `SIxtransform` / `SIxxtransform` results for
   ∂f/∂r (slot 2) and ∂²f/∂r² (slot 3).

# Physical array derivative slots
| Slot | Contents |
|:----:|:-------- |
| 1    | Field values `f(r, λ)` |
| 2    | `∂f/∂r`  (first radial derivative) |
| 3    | `∂²f/∂r²` (second radial derivative) |
| 4    | `∂f/∂λ`  (first azimuthal derivative) |
| 5    | `∂²f/∂λ²` (second azimuthal derivative) |

See also: [`spectralTransform!`](@ref)
"""
function gridTransform!(grid::_RLGrid)
    gridTransform(grid, grid.physical, grid.spectral)
    return grid.physical
end

function gridTransform(grid::_RLGrid, physical::Array{real}, spectral::Array{real})

    # Need to include patchOffset to get all available wavenumbers
    kDim = grid.params.iDim + grid.params.patchOffsetL
    iDim = grid.params.iDim
    b_iDim = grid.params.b_iDim

    # Per-grid scratch (cached): spline_r, spline_rr, spline_scratch.
    # Fourier-derivative writes reuse jring.uMish as scratch (it has been copied
    # out by the time we need it as a derivative target).
    s = _scratch(grid)
    spline_r       = s.spline_r
    spline_rr      = s.spline_rr
    spline_scratch = s.spline_scratch

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

        # ── Higher wavenumbers ────────────────────────────────────────────────
        for k in 1:kDim
            p  = k * 2   # RL convention: p = k*2 (not (k-1)*2 — see TRAP-1)

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
                if k <= r + grid.params.patchOffsetL
                    rk = k + 1
                    ik = grid.jbasis.data[r, v].params.bDim - k + 1
                    grid.jbasis.data[r, v].b[rk] = ispR.uMish[r]
                    grid.jbasis.data[r, v].b[ik] = ispI.uMish[r]
                end
            end
        end

        # ── Field values and azimuthal derivatives ────────────────────────────
        l1 = 0
        l2 = 0
        for r in 1:iDim
            jring = grid.jbasis.data[r, v]
            FAtransform!(jring)
            FItransform!(jring)

            ri      = r + grid.params.patchOffsetL
            l1 = l2 + 1
            l2 = l1 + 3 + (4 * ri)   # l2 - l1 + 1 = lpoints = 4 + 4*ri
            copyto!(view(physical, l1:l2, v, 1), jring.uMish)
            # Reuse jring.uMish as scratch for the derivative writes — its prior
            # contents have just been copied out above.
            FIxtransform(jring, jring.uMish)
            copyto!(view(physical, l1:l2, v, 4), jring.uMish)
            FIxxtransform(jring, jring.uMish)
            copyto!(view(physical, l1:l2, v, 5), jring.uMish)
        end

        # ── First radial derivative ∂f/∂r ─────────────────────────────────────
        # Reconstruct ring b-coefficients using the spline-derivative values
        @inbounds for r in 1:iDim
            grid.jbasis.data[r, v].b[1] = spline_r[r, 1]
        end
        for k in 1:kDim
            p = k * 2
            @inbounds for r in 1:iDim
                if k <= r + grid.params.patchOffsetL
                    rk = k + 1
                    ik = grid.jbasis.data[r, v].params.bDim - k + 1
                    grid.jbasis.data[r, v].b[rk] = spline_r[r, p]
                    grid.jbasis.data[r, v].b[ik] = spline_r[r, p + 1]
                end
            end
        end
        l1 = 0
        l2 = 0
        for r in 1:iDim
            jring = grid.jbasis.data[r, v]
            FAtransform!(jring)
            FItransform!(jring)

            ri = r + grid.params.patchOffsetL
            l1 = l2 + 1
            l2 = l1 + 3 + (4 * ri)
            copyto!(view(physical, l1:l2, v, 2), jring.uMish)
        end

        # ── Second radial derivative ∂²f/∂r² ──────────────────────────────────
        @inbounds for r in 1:iDim
            grid.jbasis.data[r, v].b[1] = spline_rr[r, 1]
        end
        for k in 1:kDim
            p = k * 2
            @inbounds for r in 1:iDim
                if k <= r + grid.params.patchOffsetL
                    rk = k + 1
                    ik = grid.jbasis.data[r, v].params.bDim - k + 1
                    grid.jbasis.data[r, v].b[rk] = spline_rr[r, p]
                    grid.jbasis.data[r, v].b[ik] = spline_rr[r, p + 1]
                end
            end
        end
        l1 = 0
        l2 = 0
        for r in 1:iDim
            jring = grid.jbasis.data[r, v]
            FAtransform!(jring)
            FItransform!(jring)

            ri = r + grid.params.patchOffsetL
            l1 = l2 + 1
            l2 = l1 + 3 + (4 * ri)
            copyto!(view(physical, l1:l2, v, 3), jring.uMish)
        end

    end  # for v

    return physical
end

# ═══════════════════════════════════════════════════════════════════════════
# 3D Cylindrical Transforms  (Spline×Fourier×Chebyshev = RLZ)
# ═══════════════════════════════════════════════════════════════════════════

# ── Type alias for brevity ────────────────────────────────────────────────────
const _RLZGrid = SpringsteelGrid{CylindricalGeometry, SplineBasisArray{2}, FourierBasisArray{2}, ChebyshevBasisArray{1}}

# ────────────────────────────────────────────────────────────────────────────
# getGridpoints
# ────────────────────────────────────────────────────────────────────────────

"""
    getGridpoints(grid::SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}) -> Matrix{Float64}

Return the `(r, λ, z)` physical gridpoint triples for the 3-D cylindrical grid.

Points are ordered with z varying fastest, then λ within a ring, then r:
flat index `zi + (l-1)*kDim + z - 1` for ring starting offset `zi`.

# Returns
A `Matrix{Float64}` of size `(kDim*jDim, 3)` where columns are `(r, λ, z)`.

See also: [`spectralTransform!`](@ref), [`gridTransform!`](@ref)
"""
function getGridpoints(grid::_RLZGrid)
    iDim = grid.params.iDim
    kDim = grid.params.kDim
    jDim = grid.params.jDim
    gridpts = zeros(Float64, kDim * jDim, 3)
    g  = 1
    for r in 1:iDim
        ri      = r + grid.params.patchOffsetL
        lpoints = 4 + 4*ri
        r_m     = grid.ibasis.data[1, 1].mishPoints[r]
        for l in 1:lpoints
            l_m = grid.jbasis.data[r, 1].mishPoints[l]
            for z in 1:kDim
                z_m = grid.kbasis.data[1].mishPoints[z]
                gridpts[g, 1] = r_m
                gridpts[g, 2] = l_m
                gridpts[g, 3] = z_m
                g += 1
            end
        end
    end
    return gridpts
end

# ────────────────────────────────────────────────────────────────────────────
# num_columns
# ────────────────────────────────────────────────────────────────────────────

"""
    num_columns(grid::SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}) -> Int

Return the number of j×k columns for tiling purposes.  For 3-D cylindrical grids (RLZ)
this is the total number of azimuthal physical points summed over all rings (`jDim`).

See also: [`RLZ_Grid`](@ref)
"""
function num_columns(grid::_RLZGrid)
    return grid.params.jDim
end

# ────────────────────────────────────────────────────────────────────────────
# spectralTransform / spectralTransform! — 3D Cylindrical (RLZ)
# ────────────────────────────────────────────────────────────────────────────

"""
    spectralTransform!(grid::SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray})
    spectralTransform(grid, physical, spectral)

Forward (physical → spectral) transform for the 3-D cylindrical (RLZ) grid.

**Transform order** (physical → spectral):
1. **Chebyshev stage**: `CBtransform!` on each `(r, λ)` physical column → `[b_kDim, lpoints]`
   Chebyshev coefficients per ring.
2. **Fourier stage**: Load Chebyshev coefficients into each ring's `uMish`, `FBtransform!` on
   each `(r, z_coeff)` ring.
3. **Spline stage** (per z-coefficient): `SBtransform!` on the wavenumber-0 spline, then
   `SBtransform!` on real/imaginary splines for wavenumbers `k = 1..kDim_wn`.

**Spectral layout** (z-major, wavenumber-interleaved per z-level):
```
spectral[(z-1)*b_iDim*(1+kDim_wn*2) + 1 : +b_iDim]          — z-level z, k=0
spectral[(z-1)*b_iDim*(1+kDim_wn*2) + b_iDim   + (k-1)*2*b_iDim + 1]  — k≥1 real
spectral[(z-1)*b_iDim*(1+kDim_wn*2) + b_iDim + (k-1)*2*b_iDim + b_iDim + 1]  — k≥1 imag
```
where `kDim_wn = iDim + patchOffsetL`.

⚠️ RLZ uses `p = (k-1)*2` for k ≥ 1 (unlike RL which uses `k*2`).  See Developer Notes §TRAP-1.

See also: [`gridTransform!`](@ref)
"""
function spectralTransform!(grid::_RLZGrid)
    spectralTransform(grid, grid.physical, grid.spectral)
    applyFilter!(grid)
    return grid.spectral
end

function spectralTransform(grid::_RLZGrid, physical::Array{real}, spectral::Array{real})

    # FIX BUG-1: include patchOffsetL in kDim for correct wavenumber count
    kDim_wn = grid.params.iDim + grid.params.patchOffsetL
    kDim    = grid.params.kDim
    b_kDim  = grid.params.b_kDim
    iDim    = grid.params.iDim
    b_iDim  = grid.params.b_iDim

    tempcb  = _scratch(grid).tempcb   # cached: [b_kDim, 4 + 4*kDim_wn]

    for v in 1:size(spectral, 2)
        # ── Chebyshev + Fourier stage ────────────────────────────────────────
        i = 1
        kcol = grid.kbasis.data[v]
        for r in 1:iDim
            ri      = r + grid.params.patchOffsetL
            lpoints = 4 + 4*ri
            for l in 1:lpoints
                @inbounds for z in 1:kDim
                    kcol.uMish[z] = physical[i, v, 1]
                    i += 1
                end
                CBtransform!(kcol)
                @inbounds for k in 1:b_kDim
                    tempcb[k, l] = kcol.b[k]
                end
            end
            # For each Chebyshev mode z_b, load the ring's uMish and Fourier-transform
            for z_b in 1:b_kDim
                jring = grid.jbasis.data[r, z_b]
                @inbounds for l in 1:lpoints
                    jring.uMish[l] = tempcb[z_b, l]
                end
                FBtransform!(jring)
            end
        end

        # ── Spline stage (per z_b Chebyshev coefficient) ─────────────────────
        isp0 = grid.ibasis.data[1, v]
        ispR = grid.ibasis.data[2, v]
        ispI = grid.ibasis.data[3, v]
        for z_b in 1:b_kDim
            # k=0: wavenumber-0 spline
            isp0.uMish .= 0.0
            @inbounds for r in 1:iDim
                isp0.uMish[r] = grid.jbasis.data[r, z_b].b[1]
            end
            SBtransform!(isp0)

            # Spectral index for k=0 at this z_b level
            r1 = (z_b - 1) * b_iDim * (1 + kDim_wn * 2) + 1
            r2 = r1 + b_iDim - 1
            @inbounds for k in 0:(b_iDim - 1)
                spectral[r1 + k, v] = isp0.b[k + 1]
            end

            # k ≥ 1: real and imaginary parts
            for k in 1:kDim_wn
                ispR.uMish .= 0.0
                ispI.uMish .= 0.0
                @inbounds for r in 1:iDim
                    if k <= r + grid.params.patchOffsetL
                        rk = k + 1
                        ik = grid.jbasis.data[r, z_b].params.bDim - k + 1
                        ispR.uMish[r] = grid.jbasis.data[r, z_b].b[rk]
                        ispI.uMish[r] = grid.jbasis.data[r, z_b].b[ik]
                    end
                end
                SBtransform!(ispR)
                SBtransform!(ispI)

                # RLZ convention: p = (k-1)*2  (NOT k*2 — see Developer Notes §TRAP-1)
                p  = (k - 1) * 2
                p1 = r2 + 1 + (p * b_iDim)
                @inbounds for k2 in 0:(b_iDim - 1)
                    spectral[p1 + k2, v] = ispR.b[k2 + 1]
                end

                p1 = p1 + b_iDim
                @inbounds for k2 in 0:(b_iDim - 1)
                    spectral[p1 + k2, v] = ispI.b[k2 + 1]
                end
            end
        end
    end

    return spectral
end

# ────────────────────────────────────────────────────────────────────────────
# gridTransform / gridTransform! — 3D Cylindrical (RLZ)
# ────────────────────────────────────────────────────────────────────────────

"""
    gridTransform!(grid::SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray})
    gridTransform(grid, physical, spectral)

Inverse (spectral → physical) transform for the 3-D cylindrical (RLZ) grid,
including radial, azimuthal, and vertical derivatives.

**Transform order** (spectral → physical):
1. Spline inverse: `SAtransform!` + `SItransform!` / `SIxtransform` / `SIxxtransform`
   per wavenumber per z-coefficient → ring b-coefficients.
2. `FAtransform!` + `FItransform!` (or `FIxtransform` / `FIxxtransform`) per ring.
3. Chebyshev inverse: `CAtransform!` + `CItransform!` (or `CIxtransform` / `CIxxtransform`)
   per `(r, λ)` column → physical values.

**Physical derivative layout** (7 slots):
| Slot | Contents |
|:----:|:-------- |
| 1    | Field values `f(r, λ, z)` |
| 2    | `∂f/∂r` (first radial derivative) |
| 3    | `∂²f/∂r²` (second radial derivative) |
| 4    | `∂f/∂λ` (first azimuthal derivative) |
| 5    | `∂²f/∂λ²` (second azimuthal derivative) |
| 6    | `∂f/∂z` (first vertical derivative) |
| 7    | `∂²f/∂z²` (second vertical derivative) |

**BUG-1 fix**: `kDim_wn = iDim + patchOffsetL` (old code incorrectly used `iDim` only,
silently truncating higher wavenumbers in the inverse transform when `patchOffsetL > 0`).

See also: [`spectralTransform!`](@ref)
"""
function gridTransform!(grid::_RLZGrid)
    gridTransform(grid, grid.physical, grid.spectral)
    return grid.physical
end

function gridTransform(grid::_RLZGrid, physical::Array{real}, spectral::Array{real})

    # FIX BUG-1: include patchOffsetL — old code only used rDim (wrong)
    kDim_wn = grid.params.iDim + grid.params.patchOffsetL
    kDim    = grid.params.kDim
    b_kDim  = grid.params.b_kDim
    iDim    = grid.params.iDim
    b_iDim  = grid.params.b_iDim

    # Per-grid scratch (cached): splineBuffer, ringBuffer, spline_scratch
    sc = _scratch(grid)
    splineBuffer   = sc.splineBuffer
    ringBuffer     = sc.ringBuffer
    spline_scratch = sc.spline_scratch

    has_wn_ahat = _has_wavenumber_ahat(grid)
    slots_per_z = 1 + 2 * kDim_wn

    for v in 1:size(spectral, 2)
        kcol = grid.kbasis.data[v]
        for dr in 0:2
            # ── Spline + FAtransform stage ────────────────────────────────────
            for z_b in 1:b_kDim
                # Spectral base offset for this z-coefficient level
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
                    # RLZ convention: p = (k-1)*2
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
                        if k <= r + grid.params.patchOffsetL
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
                ri      = r + grid.params.patchOffsetL
                lpoints = 4 + 4*ri

                for dl in 0:2
                    # No mixed r/λ derivatives
                    if dr > 0 && dl > 0
                        continue
                    end

                    # Fill ringBuffer from Fourier rings at each z-level. We reuse
                    # jring.uMish as the FFTW destination — its previous contents
                    # are not needed past this point in the iteration.
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

                    # Chebyshev inverse transform per (r, l) column
                    for l in 1:lpoints
                        @inbounds for z_b in 1:b_kDim
                            kcol.b[z_b] = ringBuffer[l, z_b]
                        end
                        CAtransform!(kcol)
                        CItransform!(kcol)

                        z1 = zi + (l-1)*kDim
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
# Regular-grid output — 2D Cylindrical (RL)
# ═══════════════════════════════════════════════════════════════════════════

"""
    getRegularGridpoints(grid::_RLGrid) -> Matrix{Float64}

Return an `(n_r × n_λ, 2)` matrix of uniformly-spaced `(r, λ)` coordinates
for a 2-D cylindrical (RL) grid.

Unlike [`getGridpoints`](@ref), which returns the unevenly-spaced Gaussian mish
points (with more azimuthal gridpoints at larger radii), this function returns a
regular tensor-product grid suitable for visualisation and file I/O.

The output dimensions are controlled by [`SpringsteelGridParameters`](@ref) fields:
- `i_regular_out` — radial points spanning `[iMin, iMax]`   (default `num_cells + 1`)
- `j_regular_out` — azimuthal points spanning `[0, 2π)`     (default `iDim * 2 + 1`)

Points are ordered r-outer, λ-inner (λ varies fastest), matching the layout produced
by [`regularGridTransform`](@ref).

See also: [`regularGridTransform`](@ref), [`getGridpoints`](@ref)
"""
function getRegularGridpoints(grid::_RLGrid)
    n_r   = grid.params.i_regular_out
    n_λ   = grid.params.j_regular_out
    r_pts = collect(LinRange(grid.params.iMin, grid.params.iMax, n_r))
    λ_pts = [2π * (j - 1) / n_λ for j in 1:n_λ]
    pts   = zeros(Float64, n_r * n_λ, 2)
    idx   = 1
    for i in 1:n_r
        for j in 1:n_λ
            pts[idx, 1] = r_pts[i]
            pts[idx, 2] = λ_pts[j]
            idx += 1
        end
    end
    return pts
end

"""
    regularGridTransform(grid::_RLGrid, r_pts, λ_pts) -> Array{Float64}
    regularGridTransform(grid::_RLGrid, gridpoints)   -> Array{Float64}

Evaluate the RL spectral representation on a regular tensor-product `r × λ` grid,
returning field values and all five derivatives.

`grid.spectral` must be populated (call [`spectralTransform!`](@ref) first).

**Algorithm**: Evaluates each Fourier-wavenumber radial spline at `r_pts` once (the
``a_k(r)`` amplitudes), then reconstructs the field via the Fourier series at every
`(r, λ)` pair.  The FFTW R2HC convention stores k≥1 cosine/sine amplitudes at half
their physical-space value, so the reconstruction multiplies k≥1 terms by 2:
``f(r,λ) = a_0(r) + 2\\sum_k [a^c_k(r)\\cos(kλ) + a^s_k(r)\\sin(kλ)]``.
Cost: `O(kDim × n_r)` spline evaluations plus `O(n_r × n_λ × kDim)`
floating-point operations.

# Returns
`Array{Float64}` of shape `(n_r × n_λ, nvars, 5)` — λ varies fastest.  Derivative slots:
- `[:,:,1]` — `f(r, λ)`
- `[:,:,2]` — `∂f/∂r`
- `[:,:,3]` — `∂²f/∂r²`
- `[:,:,4]` — `∂f/∂λ`
- `[:,:,5]` — `∂²f/∂λ²`

# Example
```julia
spectralTransform!(grid_rl)
reg_pts  = getRegularGridpoints(grid_rl)
reg_phys = regularGridTransform(grid_rl, reg_pts)
```

See also: [`getRegularGridpoints`](@ref), [`gridTransform!`](@ref)
"""
function regularGridTransform(grid::_RLGrid, r_pts::AbstractVector{Float64}, λ_pts::AbstractVector{Float64})
    gp     = grid.params
    kDim   = gp.iDim + gp.patchOffsetL
    b_iDim = gp.b_iDim
    nvars  = length(gp.vars)
    n_r    = length(r_pts)
    n_λ    = length(λ_pts)
    r_vec  = collect(Float64, r_pts)
    λ_vec  = collect(Float64, λ_pts)

    physical = zeros(Float64, n_r * n_λ, nvars, 5)

    for v in 1:length(gp.vars)
        # Per-wavenumber radial spline evaluation at r_pts.
        # Layout: ak[r_out, k_slot] where k_slot=1 → k=0,
        #   k_slot=2k → cos wavenumber k,  k_slot=2k+1 → sin wavenumber k.
        ak    = zeros(Float64, n_r, 2 * kDim + 1)
        ak_r  = zeros(Float64, n_r, 2 * kDim + 1)
        ak_rr = zeros(Float64, n_r, 2 * kDim + 1)

        # Wavenumber 0 — RL spectral layout: b[1:b_iDim]
        sp0 = grid.ibasis.data[1, v]
        sp0.b .= view(grid.spectral, 1:b_iDim, v)
        SAtransform!(sp0)
        SItransform(sp0,   r_vec, view(ak,    :, 1))
        SIxtransform(sp0,  r_vec, view(ak_r,  :, 1))
        SIxxtransform(sp0, r_vec, view(ak_rr, :, 1))

        # Wavenumbers 1..kDim — RL layout uses p = k*2 (see TRAP-1)
        for k in 1:kDim
            p   = k * 2
            p1c = (p - 1) * b_iDim + 1;  p2c = p       * b_iDim   # cosine block
            p1s =  p      * b_iDim + 1;  p2s = (p + 1) * b_iDim   # sine block

            spc = grid.ibasis.data[2, v]
            spc.b .= view(grid.spectral, p1c:p2c, v)
            SAtransform!(spc)
            SItransform(spc,   r_vec, view(ak,    :, 2k))
            SIxtransform(spc,  r_vec, view(ak_r,  :, 2k))
            SIxxtransform(spc, r_vec, view(ak_rr, :, 2k))

            sps = grid.ibasis.data[3, v]
            sps.b .= view(grid.spectral, p1s:p2s, v)
            SAtransform!(sps)
            SItransform(sps,   r_vec, view(ak,    :, 2k + 1))
            SIxtransform(sps,  r_vec, view(ak_r,  :, 2k + 1))
            SIxxtransform(sps, r_vec, view(ak_rr, :, 2k + 1))
        end

        # Reconstruct on regular (r, λ) grid — λ varies fastest
        # NOTE: Fourier B-coefficients from FBtransform use FFTW R2HC convention
        # where k≥1 cosine/sine amplitudes are stored at half their physical-space
        # amplitude (the inverse FFT in FItransform handles this automatically, but
        # this analytical reconstruction must multiply k≥1 terms by 2).
        idx = 1
        for i in 1:n_r
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

function regularGridTransform(grid::_RLGrid, gridpoints::AbstractMatrix{Float64})
    r_pts = sort(unique(gridpoints[:, 1]))
    n_r   = length(r_pts)
    n_λ   = div(size(gridpoints, 1), n_r)
    λ_pts = gridpoints[1:n_λ, 2]
    return regularGridTransform(grid, r_pts, λ_pts)
end

# ═══════════════════════════════════════════════════════════════════════════
# Chebyshev arbitrary-point evaluation helpers (private)
# ═══════════════════════════════════════════════════════════════════════════

# Evaluate Chebyshev A-coefficient expansion at arbitrary z_pts.
# column.a must be populated via CAtransform!.
# Uses DCT-I convention: f = a[1] + 2*Σ_{k=2}^{N-1} a[k]*cos((k-1)t) + a[N]*cos((N-1)t)
# where t = acos((z - offset)/scale), scale = -0.5*(zmax-zmin), offset = 0.5*(zmin+zmax).
function _cheb_eval_pts!(column::Chebyshev1D, z_pts::AbstractVector{Float64},
                         fz::AbstractVector{Float64})
    cp     = column.params
    N      = cp.zDim
    a      = column.a
    scale  = -0.5 * (cp.zmax - cp.zmin)
    offset =  0.5 * (cp.zmin + cp.zmax)
    for i in eachindex(z_pts)
        ξ   = clamp((z_pts[i] - offset) / scale, -1.0 + 1e-14, 1.0 - 1e-14)
        t   = acos(ξ)
        val = a[1]
        for k in 2:(N - 1)
            val += 2.0 * a[k] * cos((k - 1) * t)
        end
        fz[i] = val + a[N] * cos((N - 1) * t)
    end
    nothing
end

# First z-derivative: df/dz = (df/dt) / (-scale * sin(t))
function _cheb_dz_pts!(column::Chebyshev1D, z_pts::AbstractVector{Float64},
                       dfz::AbstractVector{Float64})
    cp     = column.params
    N      = cp.zDim
    a      = column.a
    scale  = -0.5 * (cp.zmax - cp.zmin)
    offset =  0.5 * (cp.zmin + cp.zmax)
    for i in eachindex(z_pts)
        ξ    = clamp((z_pts[i] - offset) / scale, -1.0 + 1e-10, 1.0 - 1e-10)
        t    = acos(ξ)
        st   = sin(t)
        dfdt = 0.0
        for k in 2:(N - 1)
            dfdt -= 2.0 * (k - 1) * a[k] * sin((k - 1) * t)
        end
        dfdt -= (N - 1) * a[N] * sin((N - 1) * t)
        dfz[i] = dfdt / (-scale * st)
    end
    nothing
end

# Second z-derivative: d²f/dz² = (d²f/dt² * sin(t) - df/dt * cos(t)) / (scale² * sin³(t))
function _cheb_dzz_pts!(column::Chebyshev1D, z_pts::AbstractVector{Float64},
                        d2fz::AbstractVector{Float64})
    cp     = column.params
    N      = cp.zDim
    a      = column.a
    scale  = -0.5 * (cp.zmax - cp.zmin)
    offset =  0.5 * (cp.zmin + cp.zmax)
    for i in eachindex(z_pts)
        ξ      = clamp((z_pts[i] - offset) / scale, -1.0 + 1e-10, 1.0 - 1e-10)
        t      = acos(ξ)
        st     = sin(t);  ct = cos(t)
        dfdt   = 0.0;  d2fdt2 = 0.0
        for k in 2:(N - 1)
            m       = k - 1
            dfdt   -= 2.0 * m * a[k] * sin(m * t)
            d2fdt2 -= 2.0 * m^2 * a[k] * cos(m * t)
        end
        dfdt   -= (N - 1) * a[N] * sin((N - 1) * t)
        d2fdt2 -= (N - 1)^2 * a[N] * cos((N - 1) * t)
        d2fz[i] = (d2fdt2 * st - dfdt * ct) / (scale^2 * st^3)
    end
    nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Regular-grid output — 3D Cylindrical (RLZ)
# ═══════════════════════════════════════════════════════════════════════════

"""
    getRegularGridpoints(grid::_RLZGrid) -> Matrix{Float64}

Return an `(n_r × n_λ × n_z, 3)` matrix of uniformly-spaced `(r, λ, z)` coordinates
for a 3-D cylindrical (RLZ) grid.

The output dimensions are controlled by [`SpringsteelGridParameters`](@ref) fields:
- `i_regular_out` — radial points in `[iMin, iMax]`     (default `num_cells + 1`)
- `j_regular_out` — azimuthal points in `[0, 2π)`       (default `iDim * 2 + 1`)
- `k_regular_out` — vertical points in `[kMin, kMax]`   (default `kDim + 1`)

Points are ordered r-outer, λ-middle, z-inner (z varies fastest), matching the layout
of [`regularGridTransform`](@ref).

See also: [`regularGridTransform`](@ref), [`getGridpoints`](@ref)
"""
function getRegularGridpoints(grid::_RLZGrid)
    n_r   = grid.params.i_regular_out
    n_λ   = grid.params.j_regular_out
    n_z   = grid.params.k_regular_out
    r_pts = collect(LinRange(grid.params.iMin, grid.params.iMax, n_r))
    λ_pts = [2π * (j - 1) / n_λ for j in 1:n_λ]
    z_pts = collect(LinRange(grid.params.kMin, grid.params.kMax, n_z))
    pts   = zeros(Float64, n_r * n_λ * n_z, 3)
    idx   = 1
    for i in 1:n_r
        for j in 1:n_λ
            for k in 1:n_z
                pts[idx, 1] = r_pts[i]
                pts[idx, 2] = λ_pts[j]
                pts[idx, 3] = z_pts[k]
                idx += 1
            end
        end
    end
    return pts
end

"""
    regularGridTransform(grid::_RLZGrid, r_pts, λ_pts, z_pts) -> Array{Float64}
    regularGridTransform(grid::_RLZGrid, gridpoints)           -> Array{Float64}

Evaluate the RLZ spectral representation on a regular tensor-product `r × λ × z` grid.

`grid.spectral` must be populated (call [`spectralTransform!`](@ref) first).

**Algorithm**:
1. For each Chebyshev level `z_b` and Fourier wavenumber `k`, evaluate the radial
   B-spline at `r_pts`.
2. Sum the Fourier series at each `λ` for every `(r, z_b)` pair, multiplying k≥1
   terms by 2 to account for the FFTW R2HC half-amplitude convention.
3. Evaluate the Chebyshev series at `z_pts` using arbitrary-point DCT-I summation.

# Returns
`Array{Float64}` of shape `(n_r × n_λ × n_z, nvars, 7)` — z varies fastest.  Slots:
- `[:,:,1]` — `f`, `[:,:,2]` — `∂f/∂r`, `[:,:,3]` — `∂²f/∂r²`
- `[:,:,4]` — `∂f/∂λ`, `[:,:,5]` — `∂²f/∂λ²`
- `[:,:,6]` — `∂f/∂z`, `[:,:,7]` — `∂²f/∂z²`

# Example
```julia
spectralTransform!(grid_rlz)
reg_pts  = getRegularGridpoints(grid_rlz)
reg_phys = regularGridTransform(grid_rlz, reg_pts)
```

See also: [`getRegularGridpoints`](@ref), [`gridTransform!`](@ref)
"""
function regularGridTransform(grid::_RLZGrid, r_pts::AbstractVector{Float64},
                               λ_pts::AbstractVector{Float64}, z_pts::AbstractVector{Float64})
    gp       = grid.params
    kDim_wn  = gp.iDim + gp.patchOffsetL
    b_iDim   = gp.b_iDim
    b_kDim   = gp.b_kDim
    nvars    = length(gp.vars)
    n_r      = length(r_pts)
    n_λ      = length(λ_pts)
    n_z      = length(z_pts)
    r_vec    = collect(Float64, r_pts)
    λ_vec    = collect(Float64, λ_pts)
    z_vec    = collect(Float64, z_pts)
    n_kslots = 1 + 2 * kDim_wn   # k=0 + (cos + sin) for k = 1..kDim_wn

    physical = zeros(Float64, n_r * n_λ * n_z, nvars, 7)

    for v in 1:length(gp.vars)
        for dr in 0:2
            # ── Step 1: radial spline evaluation at r_pts ────────────────────
            # spline_vals[r_out, z_b, k_slot]: k_slot=1→k=0, 2k→cos k, 2k+1→sin k
            spline_vals = zeros(Float64, n_r, b_kDim, n_kslots)

            for z_b in 1:b_kDim
                r1 = (z_b - 1) * b_iDim * (1 + kDim_wn * 2) + 1
                r2 = r1 + b_iDim - 1

                sp0 = grid.ibasis.data[1, v]
                sp0.b .= view(grid.spectral, r1:r2, v)
                SAtransform!(sp0)
                if dr == 0; SItransform(sp0,   r_vec, view(spline_vals, :, z_b, 1))
                elseif dr == 1; SIxtransform(sp0,  r_vec, view(spline_vals, :, z_b, 1))
                else;           SIxxtransform(sp0, r_vec, view(spline_vals, :, z_b, 1)); end

                for k in 1:kDim_wn
                    p  = (k - 1) * 2            # RLZ convention: p = (k-1)*2
                    p1 = r2 + 1 + p * b_iDim;  p2 = p1 + b_iDim - 1
                    spc = grid.ibasis.data[2, v]
                    spc.b .= view(grid.spectral, p1:p2, v)
                    SAtransform!(spc)
                    if dr == 0; SItransform(spc,   r_vec, view(spline_vals, :, z_b, 2k))
                    elseif dr == 1; SIxtransform(spc,  r_vec, view(spline_vals, :, z_b, 2k))
                    else;           SIxxtransform(spc, r_vec, view(spline_vals, :, z_b, 2k)); end

                    p1 = p2 + 1;  p2 = p1 + b_iDim - 1
                    sps = grid.ibasis.data[3, v]
                    sps.b .= view(grid.spectral, p1:p2, v)
                    SAtransform!(sps)
                    if dr == 0; SItransform(sps,   r_vec, view(spline_vals, :, z_b, 2k + 1))
                    elseif dr == 1; SIxtransform(sps,  r_vec, view(spline_vals, :, z_b, 2k + 1))
                    else;           SIxxtransform(sps, r_vec, view(spline_vals, :, z_b, 2k + 1)); end
                end
            end

            # ── Steps 2 & 3: Fourier sum + Chebyshev evaluation ──────────────
            for dl in 0:2
                if dr > 0 && dl > 0; continue; end   # no mixed r-λ cross-derivatives

                # NOTE: Fourier B-coefficients use FFTW R2HC convention where k≥1
                # amplitudes are half the physical-space amplitude (see RL fix above).
                fourier_b = zeros(Float64, n_r, n_λ, b_kDim)
                for ri in 1:n_r
                    for j in 1:n_λ
                        λ = λ_vec[j]
                        for z_b in 1:b_kDim
                            val = (dl == 0) ? spline_vals[ri, z_b, 1] : 0.0
                            for k in 1:kDim_wn
                                rc = spline_vals[ri, z_b, 2k]
                                rs = spline_vals[ri, z_b, 2k + 1]
                                ck = cos(k * λ);  sk = sin(k * λ)
                                if dl == 0
                                    val += 2.0 * (rc * ck + rs * sk)
                                elseif dl == 1
                                    val += 2.0 * k * (-rc * sk + rs * ck)
                                else
                                    val -= 2.0 * k^2 * (rc * ck + rs * sk)
                                end
                            end
                            fourier_b[ri, j, z_b] = val
                        end
                    end
                end

                cheb_col = grid.kbasis.data[v]
                for ri in 1:n_r
                    for j in 1:n_λ
                        for z_b in 1:b_kDim
                            cheb_col.b[z_b] = fourier_b[ri, j, z_b]
                        end
                        CAtransform!(cheb_col)
                        flat = (ri - 1) * n_λ * n_z + (j - 1) * n_z + 1
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

function regularGridTransform(grid::_RLZGrid, gridpoints::AbstractMatrix{Float64})
    r_pts = sort(unique(gridpoints[:, 1]))
    λ_pts = sort(unique(gridpoints[:, 2]))
    z_pts = sort(unique(gridpoints[:, 3]))
    return regularGridTransform(grid, r_pts, λ_pts, z_pts)
end

# ────────────────────────────────────────────────────────────────────────────
# Grid relocation for cylindrical (RL, RLZ) grids
# ────────────────────────────────────────────────────────────────────────────
#
# Reprojects spectral data to a new coordinate center. For RL/RLZ grids,
# this means shifting the axis of the polar/cylindrical coordinate system
# and re-evaluating all fields on the new grid.
#
# Must be included AFTER interpolation.jl (uses evaluate_unstructured).

const _RL_Grid  = SpringsteelGrid{CylindricalGeometry, <:SplineBasisArray, <:FourierBasisArray, <:NoBasisArray}
const _RLZ_Grid_Reloc = SpringsteelGrid{CylindricalGeometry, <:SplineBasisArray, <:FourierBasisArray, <:ChebyshevBasisArray}

"""
    relocate_grid(grid, new_center; boundary=:azimuthal_mean, out_of_bounds=:nan)

Return a new grid with spectral/physical data reprojected to `new_center`.

`new_center` is `(Δx, Δy)` — the displacement of the new center relative to
the current center in Cartesian coordinates. The new grid's coordinate system
has its origin at the shifted center.

# Boundary strategies for OOB points (where mapped radius exceeds `iMax`):
- `:nan` — fill with NaN
- `:nearest` — clamp source radius to `iMax`
- `:azimuthal_mean` — use only the `k=0` (azimuthal mean) coefficient
- `:bc_respecting` — extrapolate using the grid's radial BCs

# Returns
A new `SpringsteelGrid` with relocated spectral and physical data.

See also: [`relocate_grid!`](@ref)
"""
function relocate_grid(grid::Union{_RL_Grid, _RLZ_Grid_Reloc},
                       new_center::NTuple{2, Float64};
                       boundary::Symbol = :azimuthal_mean,
                       out_of_bounds = :nan)
    new_grid = createGrid(grid.params)
    _relocate_core!(new_grid, grid, new_center; boundary=boundary, out_of_bounds=out_of_bounds)
    return new_grid
end

"""
    relocate_grid!(grid, new_center; boundary=:azimuthal_mean, out_of_bounds=:nan)

Relocate `grid` in place by reprojecting its spectral/physical data to `new_center`.

The grid's `.spectral` and `.physical` arrays are overwritten. The source data
is snapshot before evaluation, so in-place operation is safe.

Returns the modified `grid`.

See also: [`relocate_grid`](@ref)
"""
function relocate_grid!(grid::Union{_RL_Grid, _RLZ_Grid_Reloc},
                        new_center::NTuple{2, Float64};
                        boundary::Symbol = :azimuthal_mean,
                        out_of_bounds = :nan)
    _relocate_core!(grid, grid, new_center; boundary=boundary, out_of_bounds=out_of_bounds)
    return grid
end

function relocate_grid(grid::SpringsteelGrid, new_center::NTuple{2, Float64}; kwargs...)
    throw(ArgumentError("relocate_grid is only implemented for RL and RLZ grids, got $(typeof(grid))"))
end

function relocate_grid!(grid::SpringsteelGrid, new_center::NTuple{2, Float64}; kwargs...)
    throw(ArgumentError("relocate_grid! is only implemented for RL and RLZ grids, got $(typeof(grid))"))
end

# ── Core relocation ────────────────────────────────────────────────────────

function _relocate_core!(target::SpringsteelGrid, source::SpringsteelGrid,
                         new_center::NTuple{2, Float64};
                         boundary::Symbol, out_of_bounds)
    if boundary ∉ (:nan, :nearest, :azimuthal_mean, :bc_respecting)
        throw(ArgumentError("Unknown boundary strategy: $boundary. " *
            "Must be :nan, :nearest, :azimuthal_mean, or :bc_respecting."))
    end

    Δx, Δy = new_center
    gp = source.params

    spectral_copy = copy(source.spectral)

    pts = getGridpoints(target)
    nphys = size(pts, 1)
    nvars = length(gp.vars)
    is_3d = !(target.kbasis isa NoBasisArray)

    ndims_grid = is_3d ? 3 : 2
    src_pts = zeros(Float64, nphys, ndims_grid)

    oob_mask = falses(nphys)

    for i in 1:nphys
        r_new = pts[i, 1]
        λ_new = pts[i, 2]

        x_old = r_new * cos(λ_new) + Δx
        y_old = r_new * sin(λ_new) + Δy
        r_old = sqrt(x_old^2 + y_old^2)
        λ_old = atan(y_old, x_old)

        if r_old > gp.iMax || r_old < gp.iMin
            oob_mask[i] = true
            if boundary == :nearest
                r_old = clamp(r_old, gp.iMin, gp.iMax)
            end
        end

        src_pts[i, 1] = r_old
        src_pts[i, 2] = λ_old
        if is_3d
            src_pts[i, 3] = pts[i, 3]
        end
    end

    source_eval = _make_eval_grid(source, spectral_copy)

    ib_mask = .!oob_mask
    any_ib = any(ib_mask)

    for vname in sort(collect(keys(gp.vars)))
        sv = gp.vars[vname]

        if any_ib
            ib_idx = findall(ib_mask)
            pts_ib = src_pts[ib_idx, :]
            vals_ib = evaluate_unstructured(source_eval, pts_ib;
                                            vars=[vname], out_of_bounds=:nan)
            for (k, idx) in enumerate(ib_idx)
                target.physical[idx, sv, 1] = vals_ib[k, sv]
            end
        end

        if any(oob_mask)
            _fill_oob!(target, source_eval, spectral_copy, sv, gp,
                       pts, src_pts, oob_mask, boundary, out_of_bounds, is_3d)
        end

        for d in 2:size(target.physical, 3)
            target.physical[:, sv, d] .= NaN
        end
    end

    spectralTransform!(target)
    gridTransform!(target)

    return target
end

function _make_eval_grid(source::SpringsteelGrid, spectral_copy::Matrix{Float64})
    eval_grid = createGrid(source.params)
    eval_grid.spectral .= spectral_copy
    return eval_grid
end

function _fill_oob!(target, source_eval, spectral_copy, sv, gp,
                    pts, src_pts, oob_mask, boundary, out_of_bounds, is_3d)
    oob_idx = findall(oob_mask)

    if boundary == :nan
        for idx in oob_idx
            target.physical[idx, sv, 1] = NaN
        end

    elseif boundary == :nearest
        pts_nearest = src_pts[oob_idx, :]
        for (k, idx) in enumerate(oob_idx)
            pts_nearest[k, 1] = clamp(pts_nearest[k, 1], gp.iMin + 1e-10, gp.iMax - 1e-10)
        end
        vname = nothing
        for (name, vidx) in gp.vars
            if vidx == sv; vname = name; break; end
        end
        vals = evaluate_unstructured(source_eval, pts_nearest;
                                     vars=[vname], out_of_bounds=:nan)
        for (k, idx) in enumerate(oob_idx)
            target.physical[idx, sv, 1] = vals[k, sv]
        end

    elseif boundary == :azimuthal_mean
        _fill_oob_azimuthal_mean!(target, source_eval, spectral_copy, sv, gp,
                                  pts, oob_idx, is_3d)

    elseif boundary == :bc_respecting
        _fill_oob_bc_respecting!(target, source_eval, sv, gp, pts, src_pts, oob_idx, is_3d)

    else
        throw(ArgumentError("Unknown boundary strategy: $boundary"))
    end
end

function _fill_oob_azimuthal_mean!(target, source_eval, spectral_copy, sv, gp,
                                    pts, oob_idx, is_3d)
    b_iDim = gp.b_iDim

    sp0 = source_eval.ibasis.data[1, sv]
    sp0.b .= view(spectral_copy, 1:b_iDim, sv)
    SAtransform!(sp0)

    if !is_3d
        for idx in oob_idx
            r_new = pts[idx, 1]
            r_eval = clamp(r_new, gp.iMin + 1e-10, gp.iMax - 1e-10)
            target.physical[idx, sv, 1] = CubicBSpline.SItransform(
                sp0.params, sp0.a, r_eval, 0)
        end
    else
        b_kDim = gp.b_kDim
        kDim_wn = gp.iDim + gp.patchOffsetL
        cheb_col = source_eval.kbasis.data[sv]

        for idx in oob_idx
            r_new = pts[idx, 1]
            z = pts[idx, 3]
            r_eval = clamp(r_new, gp.iMin + 1e-10, gp.iMax - 1e-10)

            for z_b in 1:b_kDim
                r1_base = (z_b - 1) * b_iDim * (1 + kDim_wn * 2) + 1
                r2_base = r1_base + b_iDim - 1
                sp0.b .= view(spectral_copy, r1_base:r2_base, sv)
                SAtransform!(sp0)
                cheb_col.b[z_b] = CubicBSpline.SItransform(
                    sp0.params, sp0.a, r_eval, 0)
            end

            CAtransform!(cheb_col)
            target.physical[idx, sv, 1] = _cheb_eval_single(cheb_col, z)
        end
    end
end

function _fill_oob_bc_respecting!(target, source_eval, sv, gp, pts, src_pts, oob_idx, is_3d)
    vname = nothing
    for (name, vidx) in gp.vars
        if vidx == sv; vname = name; break; end
    end

    bcl_dict = get(gp.BCL, vname, get(gp.BCL, "default", CubicBSpline.R0))
    bcr_dict = get(gp.BCR, vname, get(gp.BCR, "default", CubicBSpline.R0))

    if bcr_dict == CubicBSpline.R0
        throw(ArgumentError(
            "boundary=:bc_respecting requires non-R0 (non-free) outer radial BC. " *
            "Got BCR=$bcr_dict for variable '$vname'. " *
            "Use boundary=:azimuthal_mean or :nearest instead."))
    end

    pts_boundary = copy(src_pts[oob_idx, :])
    for k in 1:length(oob_idx)
        pts_boundary[k, 1] = clamp(pts_boundary[k, 1], gp.iMin + 1e-10, gp.iMax - 1e-10)
    end
    vals = evaluate_unstructured(source_eval, pts_boundary;
                                 vars=[vname], out_of_bounds=:nan)
    for (k, idx) in enumerate(oob_idx)
        target.physical[idx, sv, 1] = vals[k, sv]
    end
end

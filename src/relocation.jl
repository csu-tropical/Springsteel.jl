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

const _GRID_CENTER_REGISTRY = Dict{UInt64, Tuple{Float64, Float64}}()

"""
    grid_center(grid) -> Tuple{Float64, Float64}

Return the current center `(x, y)` of a cylindrical grid in Cartesian
coordinates, relative to its construction origin.

Defaults to `(0.0, 0.0)` for grids that have never been relocated.
Updated automatically by `relocate_grid!`.
"""
function grid_center(grid::Union{_RL_Grid, _RLZ_Grid_Reloc})
    return get(_GRID_CENTER_REGISTRY, objectid(grid), (0.0, 0.0))
end

"""
    grid_center(mg::SpringsteelMultiGrid) -> Tuple{Float64, Float64}

Return the cumulative center shift of the multigrid's first patch.
"""
function grid_center end

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
function relocate_grid(grid::_RL_Grid,
                       new_center::NTuple{2, Float64};
                       boundary::Symbol = :azimuthal_mean,
                       out_of_bounds = :nan,
                       taper_width::Int = 0)
    if boundary == :bc_respecting
        new_grid = createGrid(grid.params)
        _relocate_core!(new_grid, grid, new_center; boundary=boundary,
                        out_of_bounds=out_of_bounds, taper_width=taper_width)
        return new_grid
    end
    new_grid = createGrid(grid.params)
    _relocate_rl_fast!(new_grid, grid, new_center; boundary=boundary, taper_width=taper_width)
    return new_grid
end

function relocate_grid(grid::_RLZ_Grid_Reloc,
                       new_center::NTuple{2, Float64};
                       boundary::Symbol = :azimuthal_mean,
                       out_of_bounds = :nan,
                       taper_width::Int = 0)
    new_grid = createGrid(grid.params)
    if boundary == :bc_respecting
        _relocate_core!(new_grid, grid, new_center; boundary=boundary,
                        out_of_bounds=out_of_bounds, taper_width=taper_width)
    else
        _relocate_rlz_fast!(new_grid, grid, new_center; boundary=boundary, taper_width=taper_width)
    end
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
function relocate_grid!(grid::_RL_Grid,
                        new_center::NTuple{2, Float64};
                        boundary::Symbol = :azimuthal_mean,
                        out_of_bounds = :nan,
                        taper_width::Int = 0)
    if boundary == :bc_respecting
        _relocate_core!(grid, grid, new_center; boundary=boundary,
                        out_of_bounds=out_of_bounds, taper_width=taper_width)
    else
        _relocate_rl_fast!(grid, grid, new_center; boundary=boundary, taper_width=taper_width)
    end
    _update_grid_center!(grid, new_center)
    return grid
end

function relocate_grid!(grid::_RLZ_Grid_Reloc,
                        new_center::NTuple{2, Float64};
                        boundary::Symbol = :azimuthal_mean,
                        out_of_bounds = :nan,
                        taper_width::Int = 0)
    if boundary == :bc_respecting
        _relocate_core!(grid, grid, new_center; boundary=boundary,
                        out_of_bounds=out_of_bounds, taper_width=taper_width)
    else
        _relocate_rlz_fast!(grid, grid, new_center; boundary=boundary, taper_width=taper_width)
    end
    _update_grid_center!(grid, new_center)
    return grid
end

function _update_grid_center!(grid::SpringsteelGrid, shift::NTuple{2, Float64})
    old = get(_GRID_CENTER_REGISTRY, objectid(grid), (0.0, 0.0))
    _GRID_CENTER_REGISTRY[objectid(grid)] = (old[1] + shift[1], old[2] + shift[2])
end

function relocate_grid(grid::SpringsteelGrid, new_center::NTuple{2, Float64}; kwargs...)
    throw(ArgumentError("relocate_grid is only implemented for RL and RLZ grids, got $(typeof(grid))"))
end

function relocate_grid!(grid::SpringsteelGrid, new_center::NTuple{2, Float64}; kwargs...)
    throw(ArgumentError("relocate_grid! is only implemented for RL and RLZ grids, got $(typeof(grid))"))
end

# ── Fast per-radius relocation for RL grids ────────────────────────────────

function _relocate_rl_fast!(target::_RL_Grid, source::_RL_Grid,
                            new_center::NTuple{2, Float64};
                            boundary::Symbol, taper_width::Int)
    if boundary ∉ (:nan, :nearest, :azimuthal_mean)
        throw(ArgumentError("Unknown boundary strategy: $boundary. " *
            "Fast path supports :nan, :nearest, :azimuthal_mean."))
    end
    Δx, Δy = new_center
    gp = source.params
    b_iDim = gp.b_iDim
    iDim = gp.iDim
    kDim = iDim + gp.patchOffsetL

    spectral_copy = copy(source.spectral)
    eval_grid = _make_eval_grid(source, spectral_copy)

    n_kslots = 1 + 2 * kDim
    dr = (gp.iMax - gp.iMin) / gp.num_cells
    taper_r_start = taper_width > 0 ? gp.iMax - taper_width * dr : gp.iMax

    max_lpoints = 4 + 4 * (iDim + gp.patchOffsetL)
    r_src = Vector{Float64}(undef, max_lpoints)
    λ_src = Vector{Float64}(undef, max_lpoints)
    oob   = falses(max_lpoints)
    ak    = zeros(Float64, max_lpoints, n_kslots)

    for vname in sort(collect(keys(gp.vars)))
        sv = gp.vars[vname]
        a_cache = _get_ahat_cache_rl(eval_grid, sv)
        sp0 = eval_grid.ibasis.data[1, sv]

        flat = 0
        for r in 1:iDim
            ri = r + gp.patchOffsetL
            lpoints = 4 + 4 * ri
            r_mish = target.ibasis.data[1, 1].mishPoints[r]

            fill!(oob, false)

            for l in 1:lpoints
                λ_new = target.jbasis.data[r, 1].mishPoints[l]
                x_old = r_mish * cos(λ_new) + Δx
                y_old = r_mish * sin(λ_new) + Δy
                r_old = sqrt(x_old^2 + y_old^2)
                λ_old = atan(y_old, x_old)

                if r_old > gp.iMax || r_old < gp.iMin
                    oob[l] = true
                    r_src[l] = clamp(r_old, gp.iMin + 1e-10, gp.iMax - 1e-10)
                else
                    r_src[l] = r_old
                end
                λ_src[l] = λ_old
            end

            for col in 1:n_kslots
                for row in 1:lpoints
                    ak[row, col] = 0.0
                end
            end

            n_ib = 0
            for l in 1:lpoints
                if !oob[l]; n_ib += 1; end
            end

            if n_ib == lpoints
                r_view = view(r_src, 1:lpoints)
                for slot in 1:n_kslots
                    CubicBSpline.SItransform(sp0.params, view(a_cache, :, slot),
                                             r_view, view(ak, 1:lpoints, slot))
                end
            elseif n_ib > 0
                for slot in 1:n_kslots
                    for l in 1:lpoints
                        if !oob[l]
                            ak[l, slot] = CubicBSpline.SItransform(
                                sp0.params, view(a_cache, :, slot), r_src[l], 0)
                        end
                    end
                end
            end

            for l in 1:lpoints
                idx = flat + l
                if oob[l]
                    if boundary == :nan
                        target.physical[idx, sv, 1] = NaN
                    elseif boundary == :azimuthal_mean
                        r_eval = r_src[l]
                        target.physical[idx, sv, 1] = CubicBSpline.SItransform(
                            sp0.params, view(a_cache, :, 1), r_eval, 0)
                    elseif boundary == :nearest
                        for slot in 1:n_kslots
                            ak[l, slot] = CubicBSpline.SItransform(
                                sp0.params, view(a_cache, :, slot), r_src[l], 0)
                        end
                        f = ak[l, 1]
                        λ = λ_src[l]
                        for k in 1:kDim
                            f += 2.0 * (ak[l, 2k] * cos(k * λ) + ak[l, 2k + 1] * sin(k * λ))
                        end
                        target.physical[idx, sv, 1] = f
                    end
                else
                    f = ak[l, 1]
                    λ = λ_src[l]
                    for k in 1:kDim
                        f += 2.0 * (ak[l, 2k] * cos(k * λ) + ak[l, 2k + 1] * sin(k * λ))
                    end
                    target.physical[idx, sv, 1] = f

                    if taper_width > 0 && r_src[l] > taper_r_start
                        w = 0.5 * (1.0 + cos(π * (r_src[l] - taper_r_start) / (gp.iMax - taper_r_start)))
                        azm_val = CubicBSpline.SItransform(
                            sp0.params, view(a_cache, :, 1), r_src[l], 0)
                        target.physical[idx, sv, 1] = w * f + (1.0 - w) * azm_val
                    end
                end
            end

            flat += lpoints
        end

        for d in 2:size(target.physical, 3)
            target.physical[:, sv, d] .= NaN
        end
    end

    spectralTransform!(target)
    gridTransform!(target)
    return target
end

# ── Fast per-radius relocation for RLZ grids ──────────────────────────────

function _relocate_rlz_fast!(target::_RLZ_Grid_Reloc, source::_RLZ_Grid_Reloc,
                              new_center::NTuple{2, Float64};
                              boundary::Symbol, taper_width::Int)
    if boundary ∉ (:nan, :nearest, :azimuthal_mean)
        throw(ArgumentError("Unknown boundary strategy: $boundary. " *
            "Fast path supports :nan, :nearest, :azimuthal_mean."))
    end
    Δx, Δy = new_center
    gp = source.params
    b_iDim = gp.b_iDim
    iDim = gp.iDim
    kDim_wn = iDim + gp.patchOffsetL
    b_kDim = gp.b_kDim
    kDim_z = gp.kDim

    spectral_copy = copy(source.spectral)
    eval_grid = _make_eval_grid(source, spectral_copy)

    n_kslots = 1 + 2 * kDim_wn
    dr = (gp.iMax - gp.iMin) / gp.num_cells
    taper_r_start = taper_width > 0 ? gp.iMax - taper_width * dr : gp.iMax

    max_lpoints = 4 + 4 * (iDim + gp.patchOffsetL)
    r_src = Vector{Float64}(undef, max_lpoints)
    λ_src = Vector{Float64}(undef, max_lpoints)
    oob   = falses(max_lpoints)
    ak    = zeros(Float64, max_lpoints, b_kDim, n_kslots)
    fourier_vals = zeros(Float64, max_lpoints, b_kDim)
    cheb_out = zeros(Float64, kDim_z)

    z_pts = target.kbasis.data[1].mishPoints
    cheb_col = eval_grid.kbasis.data[1]

    for vname in sort(collect(keys(gp.vars)))
        sv = gp.vars[vname]
        a_cache = _get_ahat_cache_rlz(eval_grid, sv)
        sp0 = eval_grid.ibasis.data[1, sv]
        cheb_col_v = eval_grid.kbasis.data[sv]

        flat = 0
        for r in 1:iDim
            ri = r + gp.patchOffsetL
            lpoints = 4 + 4 * ri
            r_mish = target.ibasis.data[1, 1].mishPoints[r]

            fill!(oob, false)
            for l in 1:lpoints
                λ_new = target.jbasis.data[r, 1].mishPoints[l]
                x_old = r_mish * cos(λ_new) + Δx
                y_old = r_mish * sin(λ_new) + Δy
                r_old = sqrt(x_old^2 + y_old^2)
                r_src[l] = r_old
                λ_src[l] = atan(y_old, x_old)
                if r_old > gp.iMax || r_old < gp.iMin
                    oob[l] = true
                    r_src[l] = clamp(r_old, gp.iMin + 1e-10, gp.iMax - 1e-10)
                end
            end

            for z_b in 1:b_kDim
                for slot in 1:n_kslots
                    for l in 1:lpoints; ak[l, z_b, slot] = 0.0; end
                end
            end

            n_ib = count(l -> !oob[l], 1:lpoints)
            if n_ib == lpoints
                r_view = view(r_src, 1:lpoints)
                for z_b in 1:b_kDim
                    stripe_offset = (z_b - 1) * n_kslots
                    for slot in 1:n_kslots
                        CubicBSpline.SItransform(sp0.params,
                            view(a_cache, :, stripe_offset + slot),
                            r_view, view(ak, 1:lpoints, z_b, slot))
                    end
                end
            elseif n_ib > 0
                for z_b in 1:b_kDim
                    stripe_offset = (z_b - 1) * n_kslots
                    for slot in 1:n_kslots
                        for l in 1:lpoints
                            if !oob[l]
                                ak[l, z_b, slot] = CubicBSpline.SItransform(
                                    sp0.params, view(a_cache, :, stripe_offset + slot),
                                    r_src[l], 0)
                            end
                        end
                    end
                end
            end

            for l in 1:lpoints
                for z_b in 1:b_kDim
                    fourier_vals[l, z_b] = 0.0
                end
            end

            for l in 1:lpoints
                if oob[l] && boundary == :nan
                    for z in 1:kDim_z
                        target.physical[flat + (l-1)*kDim_z + z, sv, 1] = NaN
                    end
                    continue
                end

                λ = λ_src[l]
                for z_b in 1:b_kDim
                    if oob[l] && boundary == :azimuthal_mean
                        fourier_vals[l, z_b] = ak[l, z_b, 1]
                    else
                        val = ak[l, z_b, 1]
                        for k in 1:kDim_wn
                            val += 2.0 * (ak[l, z_b, 2k] * cos(k * λ) +
                                          ak[l, z_b, 2k + 1] * sin(k * λ))
                        end
                        fourier_vals[l, z_b] = val
                    end
                end

                for z_b in 1:b_kDim
                    cheb_col_v.b[z_b] = fourier_vals[l, z_b]
                end
                CAtransform!(cheb_col_v)

                for z in 1:kDim_z
                    target.physical[flat + (l-1)*kDim_z + z, sv, 1] =
                        _cheb_eval_single(cheb_col_v, z_pts[z])
                end

                if taper_width > 0 && !oob[l] && r_src[l] > taper_r_start
                    w = 0.5 * (1.0 + cos(π * (r_src[l] - taper_r_start) / (gp.iMax - taper_r_start)))
                    for z_b in 1:b_kDim
                        cheb_col_v.b[z_b] = ak[l, z_b, 1]
                    end
                    CAtransform!(cheb_col_v)
                    for z in 1:kDim_z
                        azm_val = _cheb_eval_single(cheb_col_v, z_pts[z])
                        idx = flat + (l-1)*kDim_z + z
                        target.physical[idx, sv, 1] =
                            w * target.physical[idx, sv, 1] + (1.0 - w) * azm_val
                    end
                end
            end

            flat += lpoints * kDim_z
        end

        for d in 2:size(target.physical, 3)
            target.physical[:, sv, d] .= NaN
        end
    end

    spectralTransform!(target)
    gridTransform!(target)
    return target
end

# ── Core relocation (naive fallback) ──────────────────────────────────────

function _relocate_core!(target::SpringsteelGrid, source::SpringsteelGrid,
                         new_center::NTuple{2, Float64};
                         boundary::Symbol, out_of_bounds, taper_width::Int = 0)
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
    r_old_arr = zeros(Float64, nphys)

    for i in 1:nphys
        r_new = pts[i, 1]
        λ_new = pts[i, 2]

        x_old = r_new * cos(λ_new) + Δx
        y_old = r_new * sin(λ_new) + Δy
        r_old = sqrt(x_old^2 + y_old^2)
        λ_old = atan(y_old, x_old)

        r_old_arr[i] = r_old

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

    taper_r_start = gp.iMax
    if taper_width > 0
        dr = (gp.iMax - gp.iMin) / gp.num_cells
        taper_r_start = gp.iMax - taper_width * dr
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

        if taper_width > 0 && boundary ∈ (:azimuthal_mean, :nearest)
            _apply_taper!(target, source_eval, spectral_copy, sv, gp,
                          pts, r_old_arr, taper_r_start, is_3d)
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

function _apply_taper!(target, source_eval, spectral_copy, sv, gp,
                       pts, r_old_arr, taper_r_start, is_3d)
    b_iDim = gp.b_iDim
    taper_r_end = gp.iMax

    sp0 = source_eval.ibasis.data[1, sv]

    for i in 1:size(pts, 1)
        r_old = r_old_arr[i]
        if r_old < taper_r_start || r_old > taper_r_end
            continue
        end

        w = 0.5 * (1.0 + cos(π * (r_old - taper_r_start) / (taper_r_end - taper_r_start)))

        r_eval = clamp(r_old, gp.iMin + 1e-10, gp.iMax - 1e-10)

        if !is_3d
            sp0.b .= view(spectral_copy, 1:b_iDim, sv)
            SAtransform!(sp0)
            azm_val = CubicBSpline.SItransform(sp0.params, sp0.a, r_eval, 0)
        else
            b_kDim = gp.b_kDim
            kDim_wn = gp.iDim + gp.patchOffsetL
            cheb_col = source_eval.kbasis.data[sv]
            for z_b in 1:b_kDim
                r1_base = (z_b - 1) * b_iDim * (1 + kDim_wn * 2) + 1
                r2_base = r1_base + b_iDim - 1
                sp0.b .= view(spectral_copy, r1_base:r2_base, sv)
                SAtransform!(sp0)
                cheb_col.b[z_b] = CubicBSpline.SItransform(sp0.params, sp0.a, r_eval, 0)
            end
            CAtransform!(cheb_col)
            azm_val = _cheb_eval_single(cheb_col, pts[i, 3])
        end

        full_val = target.physical[i, sv, 1]
        target.physical[i, sv, 1] = w * full_val + (1.0 - w) * azm_val
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

# ────────────────────────────────────────────────────────────────────────────
# Grid relocation for SpringsteelMultiGrid (patchChain + embedding)
# ────────────────────────────────────────────────────────────────────────────
#
# Must be included AFTER both relocation.jl and multipatch.jl.

"""
    relocate_grid!(mg::SpringsteelMultiGrid, new_center; snap=:auto,
                   boundary=:azimuthal_mean, out_of_bounds=:nan, taper_width=0)

Relocate a multigrid (patchChain) to a new center.

For each patch, reprojects spectral/physical data via the same coordinate
transform as solo `relocate_grid!`. On the outermost patch, OOB points
are filled from the embedded outer environment if one was declared via the
`:embedded_in` config key at construction.

# Snap modes
- `:auto` — if `embedded_in` is set, quantize the shift to the snap quantum
  (outer grid node spacing). Otherwise, use the shift as-is.
- `:outer_nodes` — force quantization; errors if no outer environment.
- `false` — use the shift as-is, no quantization.
"""
function relocate_grid!(mg::SpringsteelMultiGrid,
                        new_center::NTuple{2, Float64};
                        snap::Union{Symbol, Bool} = :auto,
                        boundary::Symbol = :azimuthal_mean,
                        out_of_bounds = :nan,
                        taper_width::Int = 0)

    outer_grid = get(mg.config, :embedded_in, nothing)
    snap_q = get(mg.config, :snap_quantum, nothing)

    if snap == :outer_nodes && outer_grid === nothing
        throw(ArgumentError("snap=:outer_nodes requires an embedded_in outer grid"))
    end

    Δx, Δy = new_center
    if snap_q !== nothing && snap != false
        dx_q, dy_q = snap_q
        Δx = round(Δx / dx_q) * dx_q
        Δy = round(Δy / dy_q) * dy_q
    end

    snapped_center = (Δx, Δy)

    patches = mg.mpg.patches
    n_patches = length(patches)

    for (pi, patch) in enumerate(patches)
        is_outermost = (pi == n_patches)

        if is_outermost && outer_grid !== nothing
            _relocate_patch_with_outer!(patch, snapped_center, outer_grid;
                                        boundary=boundary, out_of_bounds=out_of_bounds,
                                        taper_width=taper_width)
        else
            relocate_grid!(patch, snapped_center;
                           boundary=boundary, out_of_bounds=out_of_bounds,
                           taper_width=taper_width)
        end
    end

    return mg
end

function relocate_grid(mg::SpringsteelMultiGrid,
                       new_center::NTuple{2, Float64}; kwargs...)
    throw(ArgumentError(
        "relocate_grid (non-mutating) is not supported for SpringsteelMultiGrid. " *
        "Use relocate_grid! instead."))
end

function _relocate_patch_with_outer!(patch::SpringsteelGrid,
                                     new_center::NTuple{2, Float64},
                                     outer_grid::SpringsteelGrid;
                                     boundary::Symbol, out_of_bounds,
                                     taper_width::Int)
    Δx, Δy = new_center
    gp = patch.params

    spectral_copy = copy(patch.spectral)
    source_eval = _make_eval_grid(patch, spectral_copy)

    pts = getGridpoints(patch)
    nphys = size(pts, 1)
    is_3d = !(patch.kbasis isa NoBasisArray)
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
        end

        src_pts[i, 1] = clamp(r_old, gp.iMin, gp.iMax)
        src_pts[i, 2] = λ_old
        if is_3d
            src_pts[i, 3] = pts[i, 3]
        end
    end

    ib_mask = .!oob_mask

    for vname in sort(collect(keys(gp.vars)))
        sv = gp.vars[vname]

        if any(ib_mask)
            ib_idx = findall(ib_mask)
            pts_ib = src_pts[ib_idx, :]
            vals_ib = evaluate_unstructured(source_eval, pts_ib;
                                            vars=[vname], out_of_bounds=:nan)
            for (k, idx) in enumerate(ib_idx)
                patch.physical[idx, sv, 1] = vals_ib[k, sv]
            end
        end

        if any(oob_mask)
            _fill_oob_from_outer!(patch, outer_grid, sv, gp,
                                  pts, oob_mask, new_center, is_3d)
        end

        if taper_width > 0 && boundary ∈ (:azimuthal_mean, :nearest)
            taper_r_start = gp.iMax - taper_width * (gp.iMax - gp.iMin) / gp.num_cells
            _apply_taper!(patch, source_eval, spectral_copy, sv, gp,
                          pts, r_old_arr, taper_r_start, is_3d)
        end

        for d in 2:size(patch.physical, 3)
            patch.physical[:, sv, d] .= NaN
        end
    end

    spectralTransform!(patch)
    gridTransform!(patch)
end

function _fill_oob_from_outer!(patch, outer_grid, sv, gp,
                               pts, oob_mask, new_center, is_3d)
    Δx, Δy = new_center
    oob_idx = findall(oob_mask)

    outer_ndims = 0
    outer_ndims += !(outer_grid.ibasis isa NoBasisArray) ? 1 : 0
    outer_ndims += !(outer_grid.jbasis isa NoBasisArray) ? 1 : 0
    outer_ndims += !(outer_grid.kbasis isa NoBasisArray) ? 1 : 0

    outer_pts = zeros(Float64, length(oob_idx), outer_ndims)

    same_geometry = typeof(outer_grid).parameters[1] == typeof(patch).parameters[1]

    vname = nothing
    for (name, vidx) in gp.vars
        if vidx == sv; vname = name; break; end
    end

    outer_vars = outer_grid.params.vars
    if !haskey(outer_vars, vname)
        for idx in oob_idx
            patch.physical[idx, sv, 1] = NaN
        end
        return
    end

    for (k, idx) in enumerate(oob_idx)
        r_new = pts[idx, 1]
        λ_new = pts[idx, 2]
        x = r_new * cos(λ_new) + Δx
        y = r_new * sin(λ_new) + Δy

        if same_geometry
            outer_pts[k, 1] = sqrt(x^2 + y^2)
            outer_pts[k, 2] = atan(y, x)
        else
            outer_pts[k, 1] = x
            outer_pts[k, 2] = y
        end

        if is_3d && outer_ndims >= 3
            outer_pts[k, 3] = pts[idx, 3]
        end
    end

    outer_sv = outer_vars[vname]
    vals = evaluate_unstructured(outer_grid, outer_pts;
                                 vars=[vname], out_of_bounds=:nan)
    for (k, idx) in enumerate(oob_idx)
        v = vals[k, outer_sv]
        patch.physical[idx, sv, 1] = isnan(v) ? 0.0 : v
    end
end

"""
    _compute_snap_quantum(outer_grid) -> Tuple{Float64, Float64}

Compute the snap quantum (Δx, Δy) from the outer grid's node spacing.
For Cartesian outer grids (RR, RRR), this is the regular spacing.
For cylindrical/spherical outer grids, uses the minimum spacing.
"""
function _compute_snap_quantum(outer_grid::SpringsteelGrid)
    ogp = outer_grid.params
    if outer_grid.ibasis isa SplineBasisArray && outer_grid.jbasis isa SplineBasisArray
        dx = (ogp.iMax - ogp.iMin) / ogp.num_cells
        dy = (ogp.jMax - ogp.jMin) / (ogp.jDim > 0 ? Int(ogp.jDim / ogp.mubar) : ogp.num_cells)
        return (dx, dy)
    else
        dr = (ogp.iMax - ogp.iMin) / ogp.num_cells
        return (dr, dr)
    end
end

function _setup_embedding!(mg::SpringsteelMultiGrid, outer_grid::SpringsteelGrid)
    mg.config[:embedded_in] = outer_grid
    mg.config[:snap_quantum] = _compute_snap_quantum(outer_grid)
    return nothing
end

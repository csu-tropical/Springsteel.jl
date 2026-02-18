#Functions for RRR Grid

struct RRR_Grid <: AbstractGrid
    params::GridParameters
    splines::Array{Spline1D}
    rings::Array{Spline1D}
    columns::Array{Spline1D}
    spectral::Array{Float64}
    physical::Array{Float64}
end

function create_RRR_Grid(gp::GridParameters)

    # RRR is 3-D grid with splines in R, L, and Z dimensions
    # Calculate the number of points in the L grid direction
    if gp.lDim == 0
        # Calculate from domain size - assume same resolution as R direction
        ly = gp.ymax - gp.ymin
        lx = gp.xmax - gp.xmin
        num_cells_l = Int64(ceil(gp.num_cells * (ly / lx)))
    else
        num_cells_l = Int64(gp.lDim / CubicBSpline.mubar)
    end
    lDim = num_cells_l * CubicBSpline.mubar
    b_lDim = num_cells_l + 3
    
    # Calculate Z dimension if not provided
    if gp.zDim == 0
        # Calculate from domain size
        lz = gp.zmax - gp.zmin
        lx = gp.xmax - gp.xmin
        num_cells_z = Int64(ceil(gp.num_cells * (lz / lx)))
        zDim = num_cells_z * CubicBSpline.mubar
        b_zDim = num_cells_z + 3
    else
        zDim = gp.zDim
        b_zDim = gp.b_zDim
    end
    
    # Have to create a new immutable structure for the parameters
    gp2 = GridParameters(
        geometry = gp.geometry,
        xmin = gp.xmin,
        xmax = gp.xmax,
        num_cells = gp.num_cells,
        rDim = gp.rDim,
        b_rDim = gp.b_rDim,
        l_q = gp.l_q,
        BCL = gp.BCL,
        BCR = gp.BCR,
        ymin = gp.ymin,
        ymax = gp.ymax,
        lDim = lDim,
        b_lDim = b_lDim,
        BCU = gp.BCU,
        BCD = gp.BCD,
        zmin = gp.zmin,
        zmax = gp.zmax,
        zDim = zDim,
        b_zDim = b_zDim,
        BCB = gp.BCB,
        BCT = gp.BCT,
        vars = gp.vars,
        spectralIndexL = gp.spectralIndexL,
        spectralIndexR = gp.spectralIndexR,
        patchOffsetL = gp.patchOffsetL,
        tile_num = gp.tile_num)

    # Splines in R direction (one for each L and Z spectral coefficient)
    splines = Array{Spline1D}(undef, gp2.b_lDim, gp2.b_zDim, length(values(gp2.vars)))
    # Splines in L direction (one for each R gridpoint and Z spectral coefficient)
    rings = Array{Spline1D}(undef, gp2.rDim, gp2.b_zDim, length(values(gp2.vars)))
    # Splines in Z direction (one for each R and L gridpoint)
    columns = Array{Spline1D}(undef, gp2.rDim, gp2.lDim, length(values(gp2.vars)))
    
    spectralDim = gp2.b_rDim * gp2.b_lDim * gp2.b_zDim
    spectral = zeros(Float64, spectralDim, length(values(gp2.vars)))
    
    physical = zeros(Float64, gp2.rDim * gp2.lDim * gp2.zDim, length(values(gp2.vars)), 7)
    
    grid = RRR_Grid(gp2, splines, rings, columns, spectral, physical)
    
    for key in keys(gp2.vars)
        # Allow for spline filter length to be variable specific in R direction
        var_l_q_r = 2.0
        if haskey(gp2.l_q, key)
            var_l_q_r = gp2.l_q[key]
        end
        
        # Create splines for R dimension (one for each L and Z spectral coefficient)
        for l = 1:gp2.b_lDim
            for z = 1:gp2.b_zDim
                grid.splines[l, z, gp2.vars[key]] = Spline1D(SplineParameters(
                    xmin = gp2.xmin,
                    xmax = gp2.xmax,
                    num_cells = gp2.num_cells,
                    l_q = var_l_q_r,
                    BCL = gp2.BCL[key],
                    BCR = gp2.BCR[key]))
            end
        end

        # Create splines for L dimension (one for each R gridpoint and Z spectral coefficient)
        var_l_q_l = 2.0
        if haskey(gp2.l_q, string(key, "_l"))
            var_l_q_l = gp2.l_q[string(key, "_l")]
        end
        
        for r = 1:gp2.rDim
            for z = 1:gp2.b_zDim
                grid.rings[r, z, gp2.vars[key]] = Spline1D(SplineParameters(
                    xmin = gp2.ymin,
                    xmax = gp2.ymax,
                    num_cells = num_cells_l,
                    l_q = var_l_q_l,
                    BCL = gp2.BCU[key],
                    BCR = gp2.BCD[key]))
            end
        end
        
        # Create splines for Z dimension (one for each R and L gridpoint)
        var_l_q_z = 2.0
        if haskey(gp2.l_q, string(key, "_z"))
            var_l_q_z = gp2.l_q[string(key, "_z")]
        end
        
        num_cells_z = Int64(gp2.zDim / CubicBSpline.mubar)
        for r = 1:gp2.rDim
            for l = 1:gp2.lDim
                grid.columns[r, l, gp2.vars[key]] = Spline1D(SplineParameters(
                    xmin = gp2.zmin,
                    xmax = gp2.zmax,
                    num_cells = num_cells_z,
                    l_q = var_l_q_z,
                    BCL = gp2.BCB[key],
                    BCR = gp2.BCT[key]))
            end
        end
    end
    
    return grid
end

function calcTileSizes(patch::RRR_Grid, num_tiles::int)

    # Calculate the appropriate tile size for the given patch
    num_gridpoints = patch.params.rDim
    if patch.params.num_cells / num_tiles < 3.0
        throw(DomainError(num_tiles, "Too many tiles for this grid (need at least 3 cells in R direction)"))
    end

    q,r = divrem(num_gridpoints, num_tiles)
    tile_sizes = [i <= r ? q+1 : q for i = 1:num_tiles]

    # Calculate the dimensions and set the parameters
    DX = (patch.params.xmax - patch.params.xmin) / patch.params.num_cells

    xmins = zeros(Float64,num_tiles)
    xmaxs = zeros(Float64,num_tiles)
    num_cells = zeros(Int64,num_tiles)
    spectralIndicesL = ones(Int64,num_tiles)

    # Check for the special case of only 1 tile
    if num_tiles == 1
        xmins[1] = patch.params.xmin
        xmaxs[1] = patch.params.xmax
        num_cells[1] = patch.params.num_cells
        spectralIndicesL[1] = 1
        tile_sizes[1] = patch.params.rDim
        tile_params = vcat(xmins', xmaxs', num_cells', spectralIndicesL', tile_sizes')
        return tile_params
    end

    # First tile starts on the patch boundary
    xmins[1] = patch.params.xmin
    num_cells[1] = Int64(ceil(tile_sizes[1] / 3))
    xmaxs[1] = (num_cells[1] * DX) + xmins[1]
    # Implicit spectralIndicesL = 1

    for i = 2:num_tiles-1
        xmins[i] = xmaxs[i-1]
        num_cells[i] = Int64(ceil(tile_sizes[i] / 3))
        xmaxs[i] = (num_cells[i] * DX) + xmins[i]
        spectralIndicesL[i] = num_cells[i-1] + spectralIndicesL[i-1]
    end

    # Last tile ends on the patch boundary
    if num_tiles > 1
        xmins[num_tiles] = xmaxs[num_tiles-1]
        xmaxs[num_tiles] = patch.params.xmax
        spectralIndicesL[num_tiles] = num_cells[num_tiles-1] + spectralIndicesL[num_tiles-1]
        num_cells[num_tiles] = patch.params.num_cells - spectralIndicesL[num_tiles] + 1
    end

    tile_params = vcat(xmins', xmaxs', num_cells', spectralIndicesL', tile_sizes')

    if any(x->x<3, num_cells)
        for w in 1:num_tiles
            println("Tile $w: $(tile_params[5,w]) gridpoints in $(tile_params[3,w]) cells from $(tile_params[1,w]) to $(tile_params[2,w]) starting at index $(tile_params[4,w])")
        end
        throw(DomainError(num_tiles, "Too many tiles for this grid (need at least 3 cells in R direction)"))
    end

    return tile_params
end

function getGridpoints(grid::RRR_Grid)

    # Return an array of the gridpoint locations
    gridpoints = zeros(Float64, grid.params.rDim * grid.params.lDim * grid.params.zDim, 3)
    g = 1
    for r = 1:grid.params.rDim
        for l = 1:grid.params.lDim
            for z = 1:grid.params.zDim
                r_m = grid.splines[1, 1, 1].mishPoints[r]
                l_m = grid.rings[r, 1, 1].mishPoints[l]
                z_m = grid.columns[r, l, 1].mishPoints[z]
                gridpoints[g, 1] = r_m
                gridpoints[g, 2] = l_m
                gridpoints[g, 3] = z_m
                g += 1
            end
        end
    end
    return gridpoints
end

function num_columns(grid::RRR_Grid)
    return grid.params.rDim * grid.params.lDim
end

function spectralTransform!(grid::RRR_Grid)
    
    # Transform from the RRR grid to spectral space
    # For RRR grid, varying dimensions are R, L, Z, and variable
    spectral = spectralTransform(grid, grid.physical, grid.spectral)
    return spectral
end

function spectralTransform(grid::RRR_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the RRR grid to spectral space
    # For RRR grid, varying dimensions are R, L, Z, and variable
    # Transform order: Z -> L -> R
    
    tempsb_z = zeros(Float64, grid.params.b_zDim, grid.params.rDim, grid.params.lDim)
    tempsb_l = zeros(Float64, grid.params.b_lDim, grid.params.b_zDim, grid.params.rDim)

    for v in values(grid.params.vars)
        # First, transform in Z direction
        i = 1
        for r = 1:grid.params.rDim
            for l = 1:grid.params.lDim
                for z = 1:grid.params.zDim
                    grid.columns[r, l, v].uMish[z] = physical[i, v, 1]
                    i += 1
                end
                tempsb_z[:, r, l] .= SBtransform!(grid.columns[r, l, v])
            end
        end

        # Second, transform in L direction for each Z coefficient
        for z = 1:grid.params.b_zDim
            for r = 1:grid.params.rDim
                for l = 1:grid.params.lDim
                    grid.rings[r, z, v].uMish[l] = tempsb_z[z, r, l]
                end
                tempsb_l[:, z, r] .= SBtransform!(grid.rings[r, z, v])
            end
        end

        # Finally, transform in R direction for each L and Z coefficient
        for z = 1:grid.params.b_zDim
            for l = 1:grid.params.b_lDim
                # Clear the spline
                grid.splines[l, z, v].uMish .= 0.0
                for r = 1:grid.params.rDim
                    grid.splines[l, z, v].uMish[r] = tempsb_l[l, z, r]
                end
                SBtransform!(grid.splines[l, z, v])

                # Assign the spectral array
                # Index: z * b_lDim * b_rDim + l * b_rDim + r
                idx = ((z-1) * grid.params.b_lDim * grid.params.b_rDim + 
                       (l-1) * grid.params.b_rDim + 1)
                spectral[idx:idx+grid.params.b_rDim-1, v] .= grid.splines[l, z, v].b
            end
        end
    end

    return spectral
end

function gridTransform!(grid::RRR_Grid)
    
    # Transform from the spectral to grid space
    physical = gridTransform(grid, grid.physical, grid.spectral)
    return physical
end

function gridTransform(grid::RRR_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the spectral to grid space
    # Transform order: R -> L -> Z (reverse of spectral transform)
    
    splineBuffer_r = zeros(Float64, grid.params.rDim, grid.params.b_lDim, grid.params.b_zDim)
    splineBuffer_l = zeros(Float64, grid.params.lDim, grid.params.b_zDim)

    for v in values(grid.params.vars)
        for dr in 0:2
            # Transform in R direction for each L and Z coefficient
            for z = 1:grid.params.b_zDim
                for l = 1:grid.params.b_lDim
                    idx = ((z-1) * grid.params.b_lDim * grid.params.b_rDim + 
                           (l-1) * grid.params.b_rDim + 1)
                    grid.splines[l, z, v].b .= spectral[idx:idx+grid.params.b_rDim-1, v]
                    SAtransform!(grid.splines[l, z, v])
                    if (dr == 0)
                        splineBuffer_r[:, l, z] .= SItransform!(grid.splines[l, z, v])
                    elseif (dr == 1)
                        splineBuffer_r[:, l, z] .= SIxtransform(grid.splines[l, z, v])
                    else
                        splineBuffer_r[:, l, z] .= SIxxtransform(grid.splines[l, z, v])
                    end
                end
            end

            # Transform in L direction for each R gridpoint and Z coefficient
            for z = 1:grid.params.b_zDim
                for r = 1:grid.params.rDim
                    for l = 1:grid.params.b_lDim
                        grid.rings[r, z, v].b[l] = splineBuffer_r[r, l, z]
                    end
                    SAtransform!(grid.rings[r, z, v])
                    if (dr == 0)
                        splineBuffer_l[:, z] .= SItransform!(grid.rings[r, z, v])
                    else
                        splineBuffer_l[:, z] .= grid.rings[r, z, v].uMish
                    end
                end

                # Transform in Z direction for each R and L gridpoint
                for r = 1:grid.params.rDim
                    for l = 1:grid.params.lDim
                        for zb = 1:grid.params.b_zDim
                            grid.columns[r, l, v].b[zb] = splineBuffer_l[l, zb]
                        end
                        SAtransform!(grid.columns[r, l, v])
                        SItransform!(grid.columns[r, l, v])

                        # Assign the grid array
                        i = ((r-1) * grid.params.lDim * grid.params.zDim +
                             (l-1) * grid.params.zDim + 1)
                        if (dr == 0)
                            physical[i:i+grid.params.zDim-1, v, 1] .= grid.columns[r, l, v].uMish
                            physical[i:i+grid.params.zDim-1, v, 4] .= SIxtransform(grid.rings[r, z, v])
                            physical[i:i+grid.params.zDim-1, v, 6] .= SIxtransform(grid.columns[r, l, v])
                            physical[i:i+grid.params.zDim-1, v, 7] .= SIxxtransform(grid.columns[r, l, v])
                        elseif (dr == 1)
                            physical[i:i+grid.params.zDim-1, v, 2] .= grid.columns[r, l, v].uMish
                        elseif (dr == 2)
                            physical[i:i+grid.params.zDim-1, v, 3] .= grid.columns[r, l, v].uMish
                        end
                    end
                end
            end
            
            # Also compute L derivatives in physical space index 5
            if dr == 0
                for r = 1:grid.params.rDim
                    for z = 1:grid.params.b_zDim
                        for l = 1:grid.params.lDim
                            splineBuffer_l[l, z] = SIxtransform(grid.rings[r, z, v])[l]
                        end
                    end
                    for l = 1:grid.params.lDim
                        for zb = 1:grid.params.b_zDim
                            grid.columns[r, l, v].b[zb] = splineBuffer_l[l, zb]
                        end
                        SAtransform!(grid.columns[r, l, v])
                        SItransform!(grid.columns[r, l, v])
                        
                        i = ((r-1) * grid.params.lDim * grid.params.zDim +
                             (l-1) * grid.params.zDim + 1)
                        physical[i:i+grid.params.zDim-1, v, 5] .= grid.columns[r, l, v].uMish
                    end
                end
            end
        end
    end

    return physical
end

function gridTransform!(patch::RRR_Grid, tile::RRR_Grid)

    splineBuffer_r = zeros(Float64, patch.params.rDim, patch.params.b_lDim, patch.params.b_zDim)
    splineBuffer_l = zeros(Float64, tile.params.lDim, patch.params.b_zDim)
    physical = gridTransform(patch.splines, patch.spectral, patch.params, tile, splineBuffer_r, splineBuffer_l)
    return physical
end

function gridTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::GridParameters, 
                        tile::RRR_Grid, splineBuffer_r::Array{Float64}, splineBuffer_l::Array{Float64})

    # Transform from the spectral to grid space for tiles
    for v in values(tile.params.vars)
        for dr in 0:2
            for z = 1:pp.b_zDim
                for l = 1:pp.b_lDim
                    idx = ((z-1) * pp.b_lDim * pp.b_rDim + 
                           (l-1) * pp.b_rDim + 1)
                    patchSplines[l, z, v].b .= patchSpectral[idx:idx+pp.b_rDim-1, v]
                    SAtransform!(patchSplines[l, z, v])
                    if (dr == 0)
                        splineBuffer_r[:, l, z] .= SItransform!(patchSplines[l, z, v])
                    elseif (dr == 1)
                        splineBuffer_r[:, l, z] .= SIxtransform(patchSplines[l, z, v])
                    else
                        splineBuffer_r[:, l, z] .= SIxxtransform(patchSplines[l, z, v])
                    end
                end
            end

            for z = 1:pp.b_zDim
                for r = 1:tile.params.rDim
                    ri = r + tile.params.patchOffsetL
                    for l = 1:pp.b_lDim
                        tile.rings[r, z, v].b[l] = splineBuffer_r[ri, l, z]
                    end
                    SAtransform!(tile.rings[r, z, v])
                    if (dr == 0)
                        splineBuffer_l[:, z] .= SItransform!(tile.rings[r, z, v])
                    else
                        splineBuffer_l[:, z] .= tile.rings[r, z, v].uMish
                    end
                end

                for r = 1:tile.params.rDim
                    for l = 1:tile.params.lDim
                        for zb = 1:pp.b_zDim
                            tile.columns[r, l, v].b[zb] = splineBuffer_l[l, zb]
                        end
                        SAtransform!(tile.columns[r, l, v])
                        SItransform!(tile.columns[r, l, v])

                        i = ((r-1) * tile.params.lDim * tile.params.zDim +
                             (l-1) * tile.params.zDim + 1)
                        if (dr == 0)
                            tile.physical[i:i+tile.params.zDim-1, v, 1] .= tile.columns[r, l, v].uMish
                            tile.physical[i:i+tile.params.zDim-1, v, 4] .= SIxtransform(tile.rings[r, z, v])
                            tile.physical[i:i+tile.params.zDim-1, v, 6] .= SIxtransform(tile.columns[r, l, v])
                            tile.physical[i:i+tile.params.zDim-1, v, 7] .= SIxxtransform(tile.columns[r, l, v])
                        elseif (dr == 1)
                            tile.physical[i:i+tile.params.zDim-1, v, 2] .= tile.columns[r, l, v].uMish
                        elseif (dr == 2)
                            tile.physical[i:i+tile.params.zDim-1, v, 3] .= tile.columns[r, l, v].uMish
                        end
                    end
                end
            end
        end
    end

    return tile.physical
end

function splineTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::GridParameters, 
                          sharedSpectral::SharedArray{Float64}, tile::RRR_Grid)

    # Do a partial transform from B to A for splines only
    for v in values(pp.vars)
        for z = 1:pp.b_zDim
            for l = 1:pp.b_lDim
                idx = ((z-1) * pp.b_lDim * pp.b_rDim + 
                       (l-1) * pp.b_rDim + 1)
                patchSpectral[idx:idx+pp.b_rDim-1, v] .= SAtransform(patchSplines[l, z, v], 
                    view(sharedSpectral, idx:idx+pp.b_rDim-1, v))
            end
        end
    end
end

function tileTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::GridParameters, 
                        tile::RRR_Grid, splineBuffer_r::Array{Float64}, splineBuffer_l::Array{Float64})

    # Transform from the spectral to grid space
    for v in values(pp.vars)
        for dr in 0:2
            for z = 1:pp.b_zDim
                for l = 1:pp.b_lDim
                    idx = ((z-1) * pp.b_lDim * pp.b_rDim + 
                           (l-1) * pp.b_rDim + 1)
                    patchSplines[l, z, v].a .= view(patchSpectral, idx:idx+pp.b_rDim-1, v)
                    if (dr == 0)
                        SItransform(patchSplines[l, z, v], tile.splines[1, 1, 1].mishPoints, 
                            view(splineBuffer_r, :, l, z))
                    elseif (dr == 1)
                        SIxtransform(patchSplines[l, z, v], tile.splines[1, 1, 1].mishPoints, 
                            view(splineBuffer_r, :, l, z))
                    else
                        SIxxtransform(patchSplines[l, z, v], tile.splines[1, 1, 1].mishPoints, 
                            view(splineBuffer_r, :, l, z))
                    end
                end
            end

            for z = 1:pp.b_zDim
                for r = 1:tile.params.rDim
                    for l = 1:pp.b_lDim
                        tile.rings[r, z, v].b[l] = splineBuffer_r[r, l, z]
                    end
                    SAtransform!(tile.rings[r, z, v])
                    SItransform!(tile.rings[r, z, v])
                    splineBuffer_l[:, z] .= tile.rings[r, z, v].uMish
                end

                for r = 1:tile.params.rDim
                    for l = 1:tile.params.lDim
                        for zb = 1:pp.b_zDim
                            tile.columns[r, l, v].b[zb] = splineBuffer_l[l, zb]
                        end
                        SAtransform!(tile.columns[r, l, v])
                        SItransform!(tile.columns[r, l, v])

                        i = ((r-1) * tile.params.lDim * tile.params.zDim +
                             (l-1) * tile.params.zDim + 1)
                        if (dr == 0)
                            tile.physical[i:i+tile.params.zDim-1, v, 1] .= tile.columns[r, l, v].uMish
                            tile.physical[i:i+tile.params.zDim-1, v, 4] .= SIxtransform(tile.rings[r, z, v])
                            tile.physical[i:i+tile.params.zDim-1, v, 6] .= SIxtransform(tile.columns[r, l, v])
                            tile.physical[i:i+tile.params.zDim-1, v, 7] .= SIxxtransform(tile.columns[r, l, v])
                        elseif (dr == 1)
                            tile.physical[i:i+tile.params.zDim-1, v, 2] .= tile.columns[r, l, v].uMish
                        elseif (dr == 2)
                            tile.physical[i:i+tile.params.zDim-1, v, 3] .= tile.columns[r, l, v].uMish
                        end
                    end
                end
            end
        end
    end

    return tile.physical
end

function spectralxTransform(grid::RRR_Grid, physical::Array{real}, spectral::Array{real})
    # Not yet implemented
end

function calcPatchMap(patch::RRR_Grid, tile::RRR_Grid)

    patchMap = falses(size(patch.spectral))
    tileView = falses(size(tile.spectral))

    # Indices of sharedArray that won't be touched by other workers
    spectralIndexL = tile.params.spectralIndexL
    patchRstride = patch.params.b_rDim
    tileRstride = tile.params.b_rDim
    tileShare = tileRstride - 4

    for z = 1:tile.params.b_zDim
        for l = 1:tile.params.b_lDim
            idx_p = ((z-1) * patch.params.b_lDim * patch.params.b_rDim +
                     (l-1) * patchRstride + spectralIndexL)
            p1 = idx_p
            p2 = p1 + tileShare
            patchMap[p1:p2, :] .= true

            idx_t = ((z-1) * tile.params.b_lDim * tile.params.b_rDim +
                     (l-1) * tileRstride + 1)
            t1 = idx_t
            t2 = t1 + tileShare
            tileView[t1:t2, :] .= true
        end
    end

    return patchMap, view(tile.spectral, tileView)
end

function calcHaloMap(patch::RRR_Grid, tile::RRR_Grid)

    patchMap = falses(size(patch.spectral))
    tileView = falses(size(tile.spectral))

    # Indices of sharedArray that will be written by other workers (halo region)
    spectralIndexL = tile.params.spectralIndexL
    patchRstride = patch.params.b_rDim
    tileRstride = tile.params.b_rDim
    # Index is 1 more than shared map
    tileShare = tileRstride - 3

    for z = 1:tile.params.b_zDim
        for l = 1:tile.params.b_lDim
            idx_p = ((z-1) * patch.params.b_lDim * patch.params.b_rDim +
                     (l-1) * patchRstride + spectralIndexL + tileShare)
            p1 = idx_p
            p2 = p1 + 2
            patchMap[p1:p2, :] .= true

            idx_t = ((z-1) * tile.params.b_lDim * tile.params.b_rDim +
                     (l-1) * tileRstride + 1 + tileShare)
            t1 = idx_t
            t2 = t1 + 2
            tileView[t1:t2, :] .= true
        end
    end

    return patchMap, view(tile.spectral, tileView)
end

function regularGridTransform(grid::RRR_Grid)
    
    # Output on regular grid
    num_r_gridpoints = grid.params.r_regular_out
    num_l_gridpoints = grid.params.l_regular_out
    num_z_gridpoints = grid.params.z_regular_out
    r_incr_out = (grid.params.xmax - grid.params.xmin) / (grid.params.r_regular_out - 1)
    l_incr_out = (grid.params.ymax - grid.params.ymin) / (grid.params.l_regular_out - 1)
    z_incr_out = (grid.params.zmax - grid.params.zmin) / (grid.params.z_regular_out - 1)

    spline = zeros(Float64, num_r_gridpoints, num_l_gridpoints, num_z_gridpoints)
    spline_r = zeros(Float64, num_r_gridpoints, num_l_gridpoints, num_z_gridpoints)
    spline_rr = zeros(Float64, num_r_gridpoints, num_l_gridpoints, num_z_gridpoints)

    physical = zeros(Float64, num_r_gridpoints, num_l_gridpoints, num_z_gridpoints,
        length(values(grid.params.vars)), 7)

    # Generic splines for output
    num_cells_l = Int64((num_l_gridpoints - 1) / CubicBSpline.mubar)
    num_cells_z = Int64((num_z_gridpoints - 1) / CubicBSpline.mubar)
    
    # Output on the nodes
    rpoints = zeros(Float64, num_r_gridpoints)
    for r = 1:num_r_gridpoints
        rpoints[r] = grid.params.xmin + (r-1)*r_incr_out
    end

    for (key, v) in grid.params.vars
        # Transform in R direction for each L and Z coefficient
        ring = Spline1D(SplineParameters(
            xmin = grid.params.ymin,
            xmax = grid.params.ymax,
            num_cells = num_cells_l,
            l_q = grid.params.l_q,
            BCL = grid.params.BCU[key],
            BCR = grid.params.BCD[key]))
            
        column = Spline1D(SplineParameters(
            xmin = grid.params.zmin,
            xmax = grid.params.zmax,
            num_cells = num_cells_z,
            l_q = grid.params.l_q,
            BCL = grid.params.BCB[key],
            BCR = grid.params.BCT[key]))

        for z = 1:grid.params.b_zDim
            for l = 1:grid.params.b_lDim
                idx = ((z-1) * grid.params.b_lDim * grid.params.b_rDim + 
                       (l-1) * grid.params.b_rDim + 1)
                grid.splines[l, z, v].a .= view(grid.spectral, idx:idx+grid.params.b_rDim-1, v)
                SItransform(grid.splines[l, z, v], rpoints, view(spline, :, l, z))
                SIxtransform(grid.splines[l, z, v], rpoints, view(spline_r, :, l, z))
                SIxxtransform(grid.splines[l, z, v], rpoints, view(spline_rr, :, l, z))
            end
        end

        for r = 1:num_r_gridpoints
            for z = 1:grid.params.b_zDim
                # Value
                ring.b .= spline[r, :, z]
                SAtransform!(ring)
                physical[r, :, z, v, 1] .= SItransform!(ring)
                physical[r, :, z, v, 4] .= SIxtransform(ring)
                
                # dr
                ring.b .= spline_r[r, :, z]
                SAtransform!(ring)
                physical[r, :, z, v, 2] .= SItransform!(ring)
                
                # drr
                ring.b .= spline_rr[r, :, z]
                SAtransform!(ring)
                physical[r, :, z, v, 3] .= SItransform!(ring)
            end
            
            for l = 1:num_l_gridpoints
                # Transform in Z direction
                column.b .= spline[r, l, :]
                SAtransform!(column)
                physical[r, l, :, v, 6] .= SIxtransform(column)
                physical[r, l, :, v, 7] .= SIxxtransform(column)
            end
        end
    end

    return physical
end

function getRegularGridpoints(grid::RRR_Grid)

    # Return an array of regular gridpoint locations
    num_r_gridpoints = grid.params.r_regular_out
    num_l_gridpoints = grid.params.l_regular_out
    num_z_gridpoints = grid.params.z_regular_out
    r_incr_out = (grid.params.xmax - grid.params.xmin) / (grid.params.r_regular_out - 1)
    l_incr_out = (grid.params.ymax - grid.params.ymin) / (grid.params.l_regular_out - 1)
    z_incr_out = (grid.params.zmax - grid.params.zmin) / (grid.params.z_regular_out - 1)

    gridpoints = zeros(Float64, num_r_gridpoints, num_l_gridpoints, num_z_gridpoints, 3)
    for r = 1:num_r_gridpoints
        r_m = grid.params.xmin + (r-1)*r_incr_out
        for l = 1:num_l_gridpoints
            l_m = grid.params.ymin + (l-1)*l_incr_out
            for z = 1:num_z_gridpoints
                z_m = grid.params.zmin + (z-1)*z_incr_out
                gridpoints[r, l, z, 1] = r_m
                gridpoints[r, l, z, 2] = l_m
                gridpoints[r, l, z, 3] = z_m
            end
        end
    end
    return gridpoints
end

function allocateSplineBuffer(patch::RRR_Grid, tile::RRR_Grid)
    splineBuffer_r = zeros(Float64, tile.params.rDim, tile.params.b_lDim, tile.params.b_zDim)
    splineBuffer_l = zeros(Float64, tile.params.lDim, tile.params.b_zDim)
    return splineBuffer_r, splineBuffer_l
end

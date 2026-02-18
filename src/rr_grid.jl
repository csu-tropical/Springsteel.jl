#Functions for RR Grid

struct RR_Grid <: AbstractGrid
    params::GridParameters
    splines::Array{Spline1D}
    rings::Array{Spline1D}
    spectral::Array{Float64}
    physical::Array{Float64}
end

function create_RR_Grid(gp::GridParameters)

    # RR is 2-D grid with splines in both R and L dimensions
    # Calculate the number of points in the L grid direction
    # Determine number of cells in L from ymin/ymax if not provided
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
        vars = gp.vars,
        spectralIndexL = gp.spectralIndexL,
        spectralIndexR = gp.spectralIndexR,
        patchOffsetL = gp.patchOffsetL,
        tile_num = gp.tile_num)

    splines = Array{Spline1D}(undef,gp2.b_lDim,length(values(gp2.vars)))
    rings = Array{Spline1D}(undef,gp2.rDim,length(values(gp2.vars)))
    
    spectralDim = gp2.b_lDim * gp2.b_rDim
    spectral = zeros(Float64, spectralDim, length(values(gp2.vars)))
    physical = zeros(Float64, gp2.rDim * gp2.lDim, length(values(gp2.vars)), 5)
    
    grid = RR_Grid(gp2, splines, rings, spectral, physical)
    
    for key in keys(gp2.vars)
        # Allow for spline filter length to be variable specific in R direction
        var_l_q_r = 2.0
        if haskey(gp2.l_q,key)
            var_l_q_r = gp2.l_q[key]
        end
        
        # Create splines for R dimension (one for each L spectral coefficient)
        for l = 1:gp2.b_lDim
            grid.splines[l,gp2.vars[key]] = Spline1D(SplineParameters(
                xmin = gp2.xmin,
                xmax = gp2.xmax,
                num_cells = gp2.num_cells,
                l_q = var_l_q_r,
                BCL = gp2.BCL[key],
                BCR = gp2.BCR[key]))
        end

        # Create splines for L dimension (one for each R gridpoint)
        # Allow for variable-specific BC in L direction
        var_l_q_l = 2.0
        if haskey(gp2.l_q, string(key, "_l"))
            var_l_q_l = gp2.l_q[string(key, "_l")]
        end
        
        num_cells_l = Int64((gp2.lDim / CubicBSpline.mubar))
        for r = 1:gp2.rDim
            grid.rings[r,gp2.vars[key]] = Spline1D(SplineParameters(
                xmin = gp2.ymin,
                xmax = gp2.ymax,
                num_cells = num_cells_l,
                l_q = var_l_q_l,
                BCL = gp2.BCU[key],
                BCR = gp2.BCD[key]))
        end
    end
    return grid
end

function calcTileSizes(patch::RR_Grid, num_tiles::int)

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

function getGridpoints(grid::RR_Grid)

    # Return an array of the gridpoint locations
    gridpoints = zeros(Float64, grid.params.rDim * grid.params.lDim, 2)
    g = 1
    for r = 1:grid.params.rDim
        for l = 1:grid.params.lDim
            r_m = grid.splines[1,1].mishPoints[r]
            l_m = grid.rings[r,1].mishPoints[l]
            gridpoints[g,1] = r_m
            gridpoints[g,2] = l_m
            g += 1
        end
    end
    return gridpoints
end

function num_columns(grid::RR_Grid)
    return grid.params.rDim
end

function spectralTransform!(grid::RR_Grid)
    
    # Transform from the RR grid to spectral space
    # For RR grid, varying dimensions are R, L, and variable
    spectral = spectralTransform(grid, grid.physical, grid.spectral)
    return spectral
end

function spectralTransform(grid::RR_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the RR grid to spectral space
    # For RR grid, varying dimensions are R, L, and variable
    # First transform in L direction, then in R direction
    
    tempsb = zeros(Float64, grid.params.b_lDim, grid.params.rDim)

    for v in values(grid.params.vars)
        i = 1
        for r = 1:grid.params.rDim
            for l = 1:grid.params.lDim
                grid.rings[r,v].uMish[l] = physical[i,v,1]
                i += 1
            end
            tempsb[:,r] .= SBtransform!(grid.rings[r,v])
        end

        # Now transform in R direction for each L coefficient
        for l = 1:grid.params.b_lDim
            # Clear the spline
            grid.splines[l,v].uMish .= 0.0
            for r = 1:grid.params.rDim
                grid.splines[l,v].uMish[r] = tempsb[l,r]
            end
            SBtransform!(grid.splines[l,v])

            # Assign the spectral array
            r1 = (l-1) * grid.params.b_rDim + 1
            r2 = r1 + grid.params.b_rDim - 1
            spectral[r1:r2,v] .= grid.splines[l,v].b
        end
    end

    return spectral
end

function gridTransform!(grid::RR_Grid)
    
    # Transform from the spectral to grid space
    physical = gridTransform(grid, grid.physical, grid.spectral)
    return physical
end

function gridTransform(grid::RR_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the spectral to grid space
    # Transform in R direction first, then L direction
    splineBuffer = zeros(Float64, grid.params.rDim, grid.params.b_lDim)

    for v in values(grid.params.vars)
        for dr in 0:2
            # Transform in R direction for each L coefficient
            for l = 1:grid.params.b_lDim
                r1 = (l-1) * grid.params.b_rDim + 1
                r2 = r1 + grid.params.b_rDim - 1
                grid.splines[l,v].b .= spectral[r1:r2,v]
                SAtransform!(grid.splines[l,v])
                if (dr == 0)
                    splineBuffer[:,l] .= SItransform!(grid.splines[l,v])
                elseif (dr == 1)
                    splineBuffer[:,l] .= SIxtransform(grid.splines[l,v])
                else
                    splineBuffer[:,l] .= SIxxtransform(grid.splines[l,v])
                end
            end

            # Now transform in L direction for each R gridpoint
            for r = 1:grid.params.rDim
                for l = 1:grid.params.b_lDim
                    grid.rings[r,v].b[l] = splineBuffer[r,l]
                end
                SAtransform!(grid.rings[r,v])
                SItransform!(grid.rings[r,v])

                # Assign the grid array
                l1 = (r-1)*grid.params.lDim + 1
                l2 = l1 + grid.params.lDim - 1
                if (dr == 0)
                    physical[l1:l2,v,1] .= grid.rings[r,v].uMish
                    physical[l1:l2,v,4] .= SIxtransform(grid.rings[r,v])
                    physical[l1:l2,v,5] .= SIxxtransform(grid.rings[r,v])
                elseif (dr == 1)
                    physical[l1:l2,v,2] .= grid.rings[r,v].uMish
                elseif (dr == 2)
                    physical[l1:l2,v,3] .= grid.rings[r,v].uMish
                end
            end
        end
    end

    return physical
end

function gridTransform!(patch::RR_Grid, tile::RR_Grid)

    splineBuffer = zeros(Float64, patch.params.rDim, patch.params.b_lDim)
    physical = gridTransform(patch.splines, patch.spectral, patch.params, tile, splineBuffer)
    return physical
end

function gridTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::GridParameters, tile::RR_Grid, splineBuffer::Array{Float64})

    # Transform from the spectral to grid space
    for v in values(tile.params.vars)
        for dr in 0:2
            for l = 1:pp.b_lDim
                r1 = (l-1) * pp.b_rDim + 1
                r2 = r1 + pp.b_rDim - 1
                patchSplines[l,v].b .= patchSpectral[r1:r2,v]
                SAtransform!(patchSplines[l,v])
                if (dr == 0)
                    splineBuffer[:,l] .= SItransform!(patchSplines[l,v])
                elseif (dr == 1)
                    splineBuffer[:,l] .= SIxtransform(patchSplines[l,v])
                else
                    splineBuffer[:,l] .= SIxxtransform(patchSplines[l,v])
                end
            end

            for r = 1:tile.params.rDim
                ri = r + tile.params.patchOffsetL
                for l = 1:pp.b_lDim
                    tile.rings[r,v].b[l] = splineBuffer[ri,l]
                end
                SAtransform!(tile.rings[r,v])
                SItransform!(tile.rings[r,v])

                # Assign the grid array
                l1 = (r-1)*tile.params.lDim + 1
                l2 = l1 + tile.params.lDim - 1
                if (dr == 0)
                    tile.physical[l1:l2,v,1] .= tile.rings[r,v].uMish
                    tile.physical[l1:l2,v,4] .= SIxtransform(tile.rings[r,v])
                    tile.physical[l1:l2,v,5] .= SIxxtransform(tile.rings[r,v])
                elseif (dr == 1)
                    tile.physical[l1:l2,v,2] .= tile.rings[r,v].uMish
                elseif (dr == 2)
                    tile.physical[l1:l2,v,3] .= tile.rings[r,v].uMish
                end
            end
        end
    end

    return tile.physical
end

function splineTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::GridParameters, sharedSpectral::SharedArray{Float64}, tile::RR_Grid)

    # Do a partial transform from B to A for splines only
    for v in values(pp.vars)
        r1 = 1
        for l in 1:pp.b_lDim
            r2 = r1 + pp.b_rDim - 1
            patchSpectral[r1:r2,v] .= SAtransform(patchSplines[l,v], view(sharedSpectral,r1:r2,v))
            r1 = r2 + 1
        end
    end
end

function tileTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::GridParameters, tile::RR_Grid, splineBuffer::Array{Float64})

    # Transform from the spectral to grid space
    for v in values(pp.vars)
        for dr in 0:2
            for l = 1:pp.b_lDim
                r1 = (l-1) * pp.b_rDim + 1
                r2 = r1 + pp.b_rDim - 1
                patchSplines[l,v].a .= view(patchSpectral,r1:r2,v)
                if (dr == 0)
                    SItransform(patchSplines[l,v],tile.splines[1,1].mishPoints,view(splineBuffer,:,l))
                elseif (dr == 1)
                    SIxtransform(patchSplines[l,v],tile.splines[1,1].mishPoints,view(splineBuffer,:,l))
                else
                    SIxxtransform(patchSplines[l,v],tile.splines[1,1].mishPoints,view(splineBuffer,:,l))
                end
            end

            for r = 1:tile.params.rDim
                for l = 1:pp.b_lDim
                    tile.rings[r,v].b[l] = splineBuffer[r,l]
                end
                SAtransform!(tile.rings[r,v])
                SItransform!(tile.rings[r,v])

                # Assign the grid array
                l1 = (r-1)*tile.params.lDim + 1
                l2 = l1 + tile.params.lDim - 1
                if (dr == 0)
                    tile.physical[l1:l2,v,1] .= tile.rings[r,v].uMish
                    tile.physical[l1:l2,v,4] .= SIxtransform(tile.rings[r,v])
                    tile.physical[l1:l2,v,5] .= SIxxtransform(tile.rings[r,v])
                elseif (dr == 1)
                    tile.physical[l1:l2,v,2] .= tile.rings[r,v].uMish
                elseif (dr == 2)
                    tile.physical[l1:l2,v,3] .= tile.rings[r,v].uMish
                end
            end
        end
    end

    return tile.physical
end

function spectralxTransform(grid::RR_Grid, physical::Array{real}, spectral::Array{real})
    # Not yet implemented
end

function calcPatchMap(patch::RR_Grid, tile::RR_Grid)

    patchMap = falses(size(patch.spectral))
    tileView = falses(size(tile.spectral))

    # Indices of sharedArray that won't be touched by other workers
    spectralIndexL = tile.params.spectralIndexL
    patchRstride = patch.params.b_rDim
    tileRstride = tile.params.b_rDim
    tileShare = tileRstride - 4

    for l = 1:tile.params.b_lDim
        p0 = spectralIndexL + (l-1)*patchRstride
        p1 = p0
        p2 = p1 + tileShare
        patchMap[p1:p2,:] .= true

        t0 = 1 + (l-1)*tileRstride
        t1 = t0
        t2 = t1 + tileShare
        tileView[t1:t2, :] .= true
    end

    return patchMap, view(tile.spectral, tileView)
end

function calcHaloMap(patch::RR_Grid, tile::RR_Grid)

    patchMap = falses(size(patch.spectral))
    tileView = falses(size(tile.spectral))

    # Indices of sharedArray that will be written by other workers (halo region)
    spectralIndexL = tile.params.spectralIndexL
    patchRstride = patch.params.b_rDim
    tileRstride = tile.params.b_rDim
    # Index is 1 more than shared map
    tileShare = tileRstride - 3

    for l = 1:tile.params.b_lDim
        p0 = spectralIndexL + (l-1)*patchRstride
        p1 = p0 + tileShare
        p2 = p1 + 2
        patchMap[p1:p2,:] .= true

        t0 = 1 + (l-1)*tileRstride
        t1 = t0 + tileShare
        t2 = t1 + 2
        tileView[t1:t2, :] .= true
    end

    return patchMap, view(tile.spectral, tileView)
end

function regularGridTransform(grid::RR_Grid)
    
    # Output on regular grid
    num_r_gridpoints = grid.params.r_regular_out
    num_l_gridpoints = grid.params.l_regular_out
    r_incr_out = (grid.params.xmax - grid.params.xmin) / (grid.params.r_regular_out - 1)
    l_incr_out = (grid.params.ymax - grid.params.ymin) / (grid.params.l_regular_out - 1)

    spline = zeros(Float64, num_r_gridpoints, num_l_gridpoints)
    spline_r = zeros(Float64, num_r_gridpoints, num_l_gridpoints)
    spline_rr = zeros(Float64, num_r_gridpoints, num_l_gridpoints)

    physical = zeros(Float64, num_r_gridpoints, num_l_gridpoints,
        length(values(grid.params.vars)),5)

    # Generic ring for output
    num_cells_l = Int64((num_l_gridpoints - 1) / CubicBSpline.mubar)

    # Output on the nodes
    rpoints = zeros(Float64, num_r_gridpoints)
    for r = 1:num_r_gridpoints
        rpoints[r] = grid.params.xmin + (r-1)*r_incr_out
    end

    for (key, v) in grid.params.vars
        # Transform in R direction for each L coefficient
        # Create ring spline with appropriate BCs
        ring = Spline1D(SplineParameters(
            xmin = grid.params.ymin,
            xmax = grid.params.ymax,
            num_cells = num_cells_l,
            l_q = grid.params.l_q,
            BCL = grid.params.BCU[key],
            BCR = grid.params.BCD[key]))

        for l = 1:grid.params.b_lDim
            r1 = (l-1) * grid.params.b_rDim + 1
            r2 = r1 + grid.params.b_rDim - 1
            grid.splines[l,v].a .= view(grid.spectral,r1:r2,v)
            SItransform(grid.splines[l,v], rpoints, view(spline,:,l))
            SIxtransform(grid.splines[l,v], rpoints, view(spline_r,:,l))
            SIxxtransform(grid.splines[l,v], rpoints, view(spline_rr,:,l))
        end

        for r = 1:num_r_gridpoints
            # Value
            ring.b .= spline[r,:]
            SAtransform!(ring)
            l1 = 1
            l2 = num_l_gridpoints
            physical[r,l1:l2,v,1] .= SItransform!(ring)
            physical[r,l1:l2,v,4] .= SIxtransform(ring)
            physical[r,l1:l2,v,5] .= SIxxtransform(ring)

            # dr
            ring.b .= spline_r[r,:]
            SAtransform!(ring)
            l1 = 1
            l2 = num_l_gridpoints
            physical[r,l1:l2,v,2] .= SItransform!(ring)

            # drr
            ring.b .= spline_rr[r,:]
            SAtransform!(ring)
            l1 = 1
            l2 = num_l_gridpoints
            physical[r,l1:l2,v,3] .= SItransform!(ring)
        end
    end

    return physical
end

function getRegularGridpoints(grid::RR_Grid)

    # Return an array of regular gridpoint locations
    num_r_gridpoints = grid.params.r_regular_out
    num_l_gridpoints = grid.params.l_regular_out
    r_incr_out = (grid.params.xmax - grid.params.xmin) / (grid.params.r_regular_out - 1)
    l_incr_out = (grid.params.ymax - grid.params.ymin) / (grid.params.l_regular_out - 1)

    gridpoints = zeros(Float64, num_r_gridpoints, num_l_gridpoints, 2)
    for r = 1:num_r_gridpoints
        r_m = grid.params.xmin + (r-1)*r_incr_out
        for l = 1:num_l_gridpoints
            l_m = grid.params.ymin + (l-1)*l_incr_out
            gridpoints[r,l,1] = r_m
            gridpoints[r,l,2] = l_m
        end
    end
    return gridpoints
end

function allocateSplineBuffer(patch::RR_Grid, tile::RR_Grid)
    return zeros(Float64, tile.params.rDim, tile.params.b_lDim)
end

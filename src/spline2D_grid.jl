#Functions for Spline2D Grid

struct Spline2D_Grid <: SpringsteelGrid
    params::SpringsteelGridParameters
    splines::Array{Spline1D}
    rings::Array{Spline1D}
    spectral::Array{Float64}
    physical::Array{Float64}
end

function create_Spline2D_Grid(gp::SpringsteelGridParameters)

    # Spline2D is 2-D grid with splines in both i and j dimensions
    # Calculate the number of points in the j grid direction
    # Determine number of cells in Y from jMin/jMax if not provided
    if gp.jDim == 0
        # Calculate from domain size - assume same resolution as i direction
        dy = gp.jMax - gp.jMin
        dx = gp.iMax - gp.iMin
        num_cells_j = Int64(ceil(gp.num_cells * (dy / dx)))
    else
        num_cells_j = Int64(gp.jDim / CubicBSpline.mubar)
    end
    jDim = num_cells_j * CubicBSpline.mubar
    b_jDim = num_cells_j + 3
    
    # Have to create a new immutable structure for the parameters
    gp2 = SpringsteelGridParameters(
        geometry = gp.geometry,
        iMin = gp.iMin,
        iMax = gp.iMax,
        num_cells = gp.num_cells,
        iDim = gp.iDim,
        b_iDim = gp.b_iDim,
        l_q = gp.l_q,
        BCL = gp.BCL,
        BCR = gp.BCR,
        jMin = gp.jMin,
        jMax = gp.jMax,
        jDim = jDim,
        b_jDim = b_jDim,
        BCU = gp.BCU,
        BCD = gp.BCD,
        vars = gp.vars,
        spectralIndexL = gp.spectralIndexL,
        spectralIndexR = gp.spectralIndexR,
        patchOffsetL = gp.patchOffsetL,
        tile_num = gp.tile_num)

    splines = Array{Spline1D}(undef,gp2.b_jDim,length(values(gp2.vars)))
    rings = Array{Spline1D}(undef,gp2.iDim,length(values(gp2.vars)))
    
    spectralDim = gp2.b_jDim * gp2.b_iDim
    spectral = zeros(Float64, spectralDim, length(values(gp2.vars)))
    physical = zeros(Float64, gp2.iDim * gp2.jDim, length(values(gp2.vars)), 5)
    
    grid = Spline2D_Grid(gp2, splines, rings, spectral, physical)
    
    for key in keys(gp2.vars)
        # Allow for spline filter length to be variable specific in i direction
        var_l_q_r = 2.0
        if haskey(gp2.l_q,key)
            var_l_q_r = gp2.l_q[key]
        end
        
        # Create splines for i dimension (one for each j spectral coefficient)
        for j = 1:gp2.b_jDim
            grid.splines[j,gp2.vars[key]] = Spline1D(SplineParameters(
                xmin = gp2.iMin,
                xmax = gp2.iMax,
                num_cells = gp2.num_cells,
                l_q = var_l_q_r,
                BCL = gp2.BCL[key],
                BCR = gp2.BCR[key]))
        end

        # Create splines for j dimension (one for each i gridpoint)
        # Allow for variable-specific BC in j direction
        var_l_q_l = 2.0
        if haskey(gp2.l_q, string(key, "_j"))
            var_l_q_l = gp2.l_q[string(key, "_j")]
        end
        
        num_cells_j = Int64((gp2.jDim / CubicBSpline.mubar))
        for i = 1:gp2.iDim
            grid.rings[i,gp2.vars[key]] = Spline1D(SplineParameters(
                xmin = gp2.jMin,
                xmax = gp2.jMax,
                num_cells = num_cells_j,
                l_q = var_l_q_l,
                BCL = gp2.BCU[key],
                BCR = gp2.BCD[key]))
        end
    end
    return grid
end

function calcTileSizes(patch::Spline2D_Grid, num_tiles::int)

    # Calculate the appropriate tile size for the given patch
    num_gridpoints = patch.params.iDim
    if patch.params.num_cells / num_tiles < 3.0
        throw(DomainError(num_tiles, "Too many tiles for this grid (need at least 3 cells in i direction)"))
    end

    q,r = divrem(num_gridpoints, num_tiles)
    tile_sizes = [i <= r ? q+1 : q for i = 1:num_tiles]

    # Calculate the dimensions and set the parameters
    DX = (patch.params.iMax - patch.params.iMin) / patch.params.num_cells

    xmins = zeros(Float64,num_tiles)
    xmaxs = zeros(Float64,num_tiles)
    num_cells = zeros(Int64,num_tiles)
    spectralIndicesL = ones(Int64,num_tiles)

    # Check for the special case of only 1 tile
    if num_tiles == 1
        xmins[1] = patch.params.iMin
        xmaxs[1] = patch.params.iMax
        num_cells[1] = patch.params.num_cells
        spectralIndicesL[1] = 1
        tile_sizes[1] = patch.params.iDim
        tile_params = vcat(xmins', xmaxs', num_cells', spectralIndicesL', tile_sizes')
        return tile_params
    end

    # First tile starts on the patch boundary
    xmins[1] = patch.params.iMin
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
        xmaxs[num_tiles] = patch.params.iMax
        spectralIndicesL[num_tiles] = num_cells[num_tiles-1] + spectralIndicesL[num_tiles-1]
        num_cells[num_tiles] = patch.params.num_cells - spectralIndicesL[num_tiles] + 1
    end

    tile_params = vcat(xmins', xmaxs', num_cells', spectralIndicesL', tile_sizes')

    if any(x->x<3, num_cells)
        for w in 1:num_tiles
            println("Tile $w: $(tile_params[5,w]) gridpoints in $(tile_params[3,w]) cells from $(tile_params[1,w]) to $(tile_params[2,w]) starting at index $(tile_params[4,w])")
        end
        throw(DomainError(num_tiles, "Too many tiles for this grid (need at least 3 cells in i direction)"))
    end

    return tile_params
end

function getGridpoints(grid::Spline2D_Grid)

    # Return an array of the gridpoint locations
    gridpoints = zeros(Float64, grid.params.iDim * grid.params.jDim, 2)
    g = 1
    for i = 1:grid.params.iDim
        for j = 1:grid.params.jDim
            x = grid.splines[1,1].mishPoints[i]
            y = grid.rings[1,1].mishPoints[j]
            gridpoints[g,1] = x
            gridpoints[g,2] = y
            g += 1
        end
    end
    return gridpoints
end

function num_columns(grid::Spline2D_Grid)
    return grid.params.iDim
end

function spectralTransform!(grid::Spline2D_Grid)
    
    # Transform from the RR grid to spectral space
    # For RR grid, varying dimensions are R, L, and variable
    spectral = spectralTransform(grid, grid.physical, grid.spectral)
    return spectral
end

function spectralTransform(grid::Spline2D_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the RR grid to spectral space
    # For RR grid, varying dimensions are R, L, and variable
    # First transform in j direction, then in i direction
    
    tempsb = zeros(Float64, grid.params.b_jDim, grid.params.iDim)

    for v in values(grid.params.vars)
        idx = 1
        for i = 1:grid.params.iDim
            for j = 1:grid.params.jDim
                grid.rings[i,v].uMish[j] = physical[idx,v,1]
                idx += 1
            end
            tempsb[:,i] .= SBtransform!(grid.rings[i,v])
        end

        # Now transform in i direction for each L coefficient
        for j = 1:grid.params.b_jDim
            # Clear the spline
            grid.splines[j,v].uMish .= 0.0
            for i = 1:grid.params.iDim
                grid.splines[j,v].uMish[i] = tempsb[j,i]
            end
            SBtransform!(grid.splines[j,v])

            # Assign the spectral array
            i1 = (j-1) * grid.params.b_iDim + 1
            i2 = i1 + grid.params.b_iDim - 1
            spectral[i1:i2,v] .= grid.splines[j,v].b
        end
    end

    return spectral
end

function gridTransform!(grid::Spline2D_Grid)
    
    # Transform from the spectral to grid space
    physical = gridTransform(grid, grid.physical, grid.spectral)
    return physical
end

function gridTransform(grid::Spline2D_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the spectral to grid space
    # Transform in i direction first, then j direction
    splineBuffer = zeros(Float64, grid.params.iDim, grid.params.b_jDim)

    for v in values(grid.params.vars)
        for di in 0:2
            # Transform in i direction for each coefficient
            for j = 1:grid.params.b_jDim
                i1 = (j-1) * grid.params.b_iDim + 1
                i2 = i1 + grid.params.b_iDim - 1
                grid.splines[j,v].b .= spectral[i1:i2,v]
                SAtransform!(grid.splines[j,v])
                if (di == 0)
                    splineBuffer[:,j] .= SItransform!(grid.splines[j,v])
                elseif (di == 1)
                    splineBuffer[:,j] .= SIxtransform(grid.splines[j,v])
                else
                    splineBuffer[:,j] .= SIxxtransform(grid.splines[j,v])
                end
            end

            # Now transform in j direction for each i gridpoint
            for i = 1:grid.params.iDim
                for j = 1:grid.params.b_jDim
                    grid.rings[i,v].b[j] = splineBuffer[i,j]
                end
                SAtransform!(grid.rings[i,v])
                SItransform!(grid.rings[i,v])

                # Assign the grid array
                j1 = (i-1)*grid.params.jDim + 1
                j2 = j1 + grid.params.jDim - 1
                if (di == 0)
                    physical[j1:j2,v,1] .= grid.rings[i,v].uMish
                    physical[j1:j2,v,4] .= SIxtransform(grid.rings[i,v])
                    physical[j1:j2,v,5] .= SIxxtransform(grid.rings[i,v])
                elseif (di == 1)
                    physical[j1:j2,v,2] .= grid.rings[i,v].uMish
                elseif (di == 2)
                    physical[j1:j2,v,3] .= grid.rings[i,v].uMish
                end
            end
        end
    end

    return physical
end

function gridTransform!(patch::Spline2D_Grid, tile::Spline2D_Grid)

    splineBuffer = zeros(Float64, patch.params.iDim, patch.params.b_jDim)
    physical = gridTransform(patch.splines, patch.spectral, patch.params, tile, splineBuffer)
    return physical
end

function gridTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::SpringsteelGridParameters, tile::Spline2D_Grid, splineBuffer::Array{Float64})

    # Transform from the spectral to grid space
    for v in values(tile.params.vars)
        for di in 0:2
            for j = 1:pp.b_jDim
                i1 = (j-1) * pp.b_iDim + 1
                i2 = i1 + pp.b_iDim - 1
                patchSplines[j,v].b .= patchSpectral[i1:i2,v]
                SAtransform!(patchSplines[j,v])
                if (di == 0)
                    splineBuffer[:,j] .= SItransform!(patchSplines[j,v])
                elseif (di == 1)
                    splineBuffer[:,j] .= SIxtransform(patchSplines[j,v])
                else
                    splineBuffer[:,j] .= SIxxtransform(patchSplines[j,v])
                end
            end

            for i = 1:tile.params.iDim
                ii = i + tile.params.patchOffsetL
                for j = 1:pp.b_jDim
                    tile.rings[i,v].b[l] = splineBuffer[ii,j]
                end
                SAtransform!(tile.rings[i,v])
                SItransform!(tile.rings[i,v])

                # Assign the grid array
                j1 = (i-1)*tile.params.jDim + 1
                j2 = j1 + tile.params.jDim - 1
                if (di == 0)
                    tile.physical[j1:j2,v,1] .= tile.rings[i,v].uMish
                    tile.physical[j1:j2,v,4] .= SIxtransform(tile.rings[i,v])
                    tile.physical[j1:j2,v,5] .= SIxxtransform(tile.rings[i,v])
                elseif (di == 1)
                    tile.physical[j1:j2,v,2] .= tile.rings[i,v].uMish
                elseif (di == 2)
                    tile.physical[j1:j2,v,3] .= tile.rings[i,v].uMish
                end
            end
        end
    end

    return tile.physical
end

function splineTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::SpringsteelGridParameters, sharedSpectral::SharedArray{Float64}, tile::Spline2D_Grid)

    # Do a partial transform from B to A for splines only
    for v in values(pp.vars)
        i1 = 1
        for j in 1:pp.b_jDim
            i2 = i1 + pp.b_iDim - 1
            patchSpectral[i1:i2,v] .= SAtransform(patchSplines[j,v], view(sharedSpectral,i1:i2,v))
            i1 = i2 + 1
        end
    end
end

function tileTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::SpringsteelGridParameters, tile::Spline2D_Grid, splineBuffer::Array{Float64})

    # Transform from the spectral to grid space
    for v in values(pp.vars)
        for di in 0:2
            for j = 1:pp.b_jDim
                i1 = (j-1) * pp.b_iDim + 1
                i2 = i1 + pp.b_iDim - 1
                patchSplines[j,v].a .= view(patchSpectral,i1:i2,v)
                if (di == 0)
                    SItransform(patchSplines[j,v],tile.splines[1,1].mishPoints,view(splineBuffer,:,j))
                elseif (di == 1)
                    SIxtransform(patchSplines[j,v],tile.splines[1,1].mishPoints,view(splineBuffer,:,j))
                else
                    SIxxtransform(patchSplines[j,v],tile.splines[1,1].mishPoints,view(splineBuffer,:,j))
                end
            end

            for i = 1:tile.params.iDim
                for j = 1:pp.b_jDim
                    tile.rings[i,v].b[l] = splineBuffer[i,j]
                end
                SAtransform!(tile.rings[i,v])
                SItransform!(tile.rings[i,v])

                # Assign the grid array
                j1 = (j-1)*tile.params.jDim + 1
                j2 = j1 + tile.params.jDim - 1
                if (di == 0)
                    tile.physical[j1:j2,v,1] .= tile.rings[i,v].uMish
                    tile.physical[j1:j2,v,4] .= SIxtransform(tile.rings[i,v])
                    tile.physical[j1:j2,v,5] .= SIxxtransform(tile.rings[i,v])
                elseif (di == 1)
                    tile.physical[j1:j2,v,2] .= tile.rings[i,v].uMish
                elseif (di == 2)
                    tile.physical[j1:j2,v,3] .= tile.rings[i,v].uMish
                end
            end
        end
    end

    return tile.physical
end

function spectralxTransform(grid::Spline2D_Grid, physical::Array{real}, spectral::Array{real})
    # Not yet implemented
end

function calcPatchMap(patch::Spline2D_Grid, tile::Spline2D_Grid)

    patchMap = falses(size(patch.spectral))
    tileView = falses(size(tile.spectral))

    # Indices of sharedArray that won't be touched by other workers
    spectralIndexL = tile.params.spectralIndexL
    patchRstride = patch.params.b_iDim
    tileRstride = tile.params.b_iDim
    tileShare = tileRstride - 4

    for j = 1:tile.params.b_jDim
        p0 = spectralIndexL + (j-1)*patchRstride
        p1 = p0
        p2 = p1 + tileShare
        patchMap[p1:p2,:] .= true

        t0 = 1 + (j-1)*tileRstride
        t1 = t0
        t2 = t1 + tileShare
        tileView[t1:t2, :] .= true
    end

    return patchMap, view(tile.spectral, tileView)
end

function calcHaloMap(patch::Spline2D_Grid, tile::Spline2D_Grid)

    patchMap = falses(size(patch.spectral))
    tileView = falses(size(tile.spectral))

    # Indices of sharedArray that will be written by other workers (halo region)
    spectralIndexL = tile.params.spectralIndexL
    patchRstride = patch.params.b_iDim
    tileRstride = tile.params.b_iDim
    # Index is 1 more than shared map
    tileShare = tileRstride - 3

    for j = 1:tile.params.b_jDim
        p0 = spectralIndexL + (j-1)*patchRstride
        p1 = p0 + tileShare
        p2 = p1 + 2
        patchMap[p1:p2,:] .= true

        t0 = 1 + (j-1)*tileRstride
        t1 = t0 + tileShare
        t2 = t1 + 2
        tileView[t1:t2, :] .= true
    end

    return patchMap, view(tile.spectral, tileView)
end

function regularGridTransform(grid::Spline2D_Grid)
    
    # Output on regular grid
    num_i_gridpoints = grid.params.i_regular_out
    num_j_gridpoints = grid.params.j_regular_out
    i_incr_out = (grid.params.iMax - grid.params.iMin) / (grid.params.i_regular_out - 1)
    j_incr_out = (grid.params.jMax - grid.params.jMin) / (grid.params.j_regular_out - 1)

    spline = zeros(Float64, num_i_gridpoints, num_j_gridpoints)
    spline_x = zeros(Float64, num_i_gridpoints, num_j_gridpoints)
    spline_xx = zeros(Float64, num_i_gridpoints, num_j_gridpoints)

    physical = zeros(Float64, num_i_gridpoints, num_j_gridpoints,
        length(values(grid.params.vars)),5)

    # Generic ring for output
    num_cells_j = Int64((num_j_gridpoints - 1) / CubicBSpline.mubar)

    # Output on the nodes
    ipoints = zeros(Float64, num_i_gridpoints)
    for i = 1:num_i_gridpoints
        ipoints[i] = grid.params.iMin + (i-1)*i_incr_out
    end

    for (key, v) in grid.params.vars
        # Transform in i direction for each L coefficient
        # Create ring spline with appropriate BCs
        ring = Spline1D(SplineParameters(
            xmin = grid.params.jMin,
            xmax = grid.params.jMax,
            num_cells = num_cells_j,
            l_q = grid.params.l_q,
            BCL = grid.params.BCU[key],
            BCR = grid.params.BCD[key]))

        for j = 1:grid.params.b_jDim
            i1 = (j-1) * grid.params.b_iDim + 1
            i2 = j1 + grid.params.b_iDim - 1
            grid.splines[j,v].a .= view(grid.spectral,i1:i2,v)
            SItransform(grid.splines[j,v], ipoints, view(spline,:,j))
            SIxtransform(grid.splines[j,v], ipoints, view(spline_x,:,j))
            SIxxtransform(grid.splines[j,v], ipoints, view(spline_xx,:,j))
        end

        for i = 1:num_i_gridpoints
            j1 = 1
            j2 = num_j_gridpoints

            # Value
            ring.b .= spline[i,:]
            SAtransform!(ring)
            physical[i,j1:j2,v,1] .= SItransform!(ring)
            physical[i,j1:j2,v,4] .= SIxtransform(ring)
            physical[i,j1:j2,v,5] .= SIxxtransform(ring)

            # dx
            ring.b .= spline_x[i,:]
            SAtransform!(ring)
            physical[i,j1:j2,v,2] .= SItransform!(ring)

            # dxx
            ring.b .= spline_xx[i,:]
            SAtransform!(ring)
            physical[i,j1:j2,v,3] .= SItransform!(ring)
        end
    end

    return physical
end

function getRegularGridpoints(grid::Spline2D_Grid)

    # Return an array of regular gridpoint locations
    num_i_gridpoints = grid.params.r_regular_out
    num_j_gridpoints = grid.params.l_regular_out
    i_incr_out = (grid.params.iMax - grid.params.iMin) / (grid.params.i_regular_out - 1)
    j_incr_out = (grid.params.jMax - grid.params.jMin) / (grid.params.j_regular_out - 1)

    gridpoints = zeros(Float64, num_i_gridpoints, num_j_gridpoints, 2)
    for i = 1:num_i_gridpoints
        x = grid.params.iMin + (i-1)*i_incr_out
        for j = 1:num_j_gridpoints
            y = grid.params.jMin + (j-1)*j_incr_out
            gridpoints[i,j,1] = x
            gridpoints[i,j,2] = y
        end
    end
    return gridpoints
end

function allocateSplineBuffer(patch::Spline2D_Grid, tile::Spline2D_Grid)
    return zeros(Float64, tile.params.iDim, tile.params.b_jDim)
end

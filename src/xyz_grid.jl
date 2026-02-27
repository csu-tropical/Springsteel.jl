#Functions for XYZ Grid

struct XYZ_Grid <: AbstractGrid
    params::GridParameters
    splines::Array{Spline1D}
    columns::Array{Chebyshev1D}
    spectral::Array{Float64}
    physical::Array{Float64}
end

function create_XYZ_Grid(gp::GridParameters)

    # XYZ is 3-D grid with splines in the horizontal and Chebyshev basis in the vertical
    
    # Need to use the info from X to set the number of cells in Y to keep horizontal resolution the same
    DX = (gp.xmax - gp.xmin) / gp.num_cells
    num_y_cells = (gp.ymax - gp.ymin) / DX
    # If num_y_cells is not an integer then we have a problem
    if !isinteger(num_y_cells)
        throw(DomainError(num_y_cells, "The Y grid does not have an integer number of cells, please fix and re-run"))
    end
    lDim::int = num_y_cells * CubicBSpline.mubar
    b_lDim::int = num_y_cells + 3

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
        BCN = gp.BCN,
        BCS = gp.BCS,
        zmin = gp.zmin,
        zmax = gp.zmax,
        zDim = gp.zDim,
        b_zDim = gp.b_zDim,
        BCB = gp.BCB,
        BCT = gp.BCT,
        vars = gp.vars,
        spectralIndexL = gp.spectralIndexL,
        spectralIndexR = gp.spectralIndexR,
        patchOffsetL = gp.patchOffsetL,
        tile_num = gp.tile_num)

    splines = Array{Spline1D}(undef,2,length(values(gp.vars)))
    columns = Array{Chebyshev1D}(undef,length(values(gp.vars)))

    # Use b_lDim or gp2 since it is updated, not gp
    spectralDim = gp.b_rDim * b_lDim * gp.b_zDim
    spectral = zeros(Float64, spectralDim, length(values(gp.vars)))
    physical = zeros(Float64, gp.rDim * lDim * gp.zDim, length(values(gp.vars)), 7)
    
    # Use gp2 here since it has updated lDim and b_lDim
    grid = XYZ_Grid(gp2, splines, columns, spectral, physical)

    for key in keys(gp.vars)

        # Allocate the X-direction first
        grid.splines[1,gp.vars[key]] = Spline1D(SplineParameters(
                xmin = gp.xmin,
                xmax = gp.xmax,
                num_cells = gp.num_cells,
                BCL = gp.BCL[key], 
                BCR = gp.BCR[key]))

        # Y-direction
        grid.splines[2,gp.vars[key]] = Spline1D(SplineParameters(
                xmin = gp.ymin,
                xmax = gp.ymax,
                num_cells = num_y_cells,
                BCL = gp.BCN[key], 
                BCR = gp.BCS[key]))

        # Z-direction
        grid.columns[gp.vars[key]] = Chebyshev1D(ChebyshevParameters(
            zmin = gp.zmin,
            zmax = gp.zmax,
            zDim = gp.zDim,
            bDim = gp.b_zDim,
            BCB = gp.BCB[key],
            BCT = gp.BCT[key]))
    end

    return grid
end

function calcTileSizes(patch::XYZ_Grid, num_tiles::int)

    # Calculate the appropriate tile size for the given patch
    num_gridpoints = patch.params.rDim * patch.params.lDim
    if patch.params.num_cells / num_tiles < 3.0
        throw(DomainError(num_tiles, "Too many tiles for this grid (need at least 3 cells in X direction)"))
    end

    # Target an even number of gridpoints per tile
    q,r = divrem(num_gridpoints, num_tiles)
    tile_targets = [i <= r ? q+1 : q for i = 1:num_tiles]
    tile_min = zeros(Int64,num_tiles)

    # Calculate the dimensions and set the parameters
    DX = (patch.params.xmax - patch.params.xmin) / patch.params.num_cells

    xmins = zeros(Float64,num_tiles)
    xmaxs = zeros(Float64,num_tiles)
    num_cells = zeros(Int64,num_tiles)
    spectralIndicesL = ones(Int64,num_tiles)
    patchOffsetsL = zeros(Int64,num_tiles)
    tile_sizes = zeros(Int64,num_tiles)

    # Check for the special case of only 1 tile
    if num_tiles == 1
        xmins[1] = patch.params.xmin
        xmaxs[1] = patch.params.xmax
        num_cells[1] = patch.params.num_cells
        spectralIndicesL[1] = 1
        tile_sizes[1] = num_gridpoints
        tile_params = vcat(xmins', xmaxs', num_cells', spectralIndicesL', tile_sizes')
        return tile_params
    end

    # Easier to balance in the Cartesian case, just divide by the number of cells
    # If domain doesn't divide evenly then last tile gets an extra cell
    num_cells[:] .=  Int64(floor(patch.params.num_cells / num_tiles))

    # First tile starts on the patch boundary
    # Make sure each tile has at least 3 cells and 50% of the target gridpoints
    xmins[1] = patch.params.xmin
    xmaxs[1] = (num_cells[1] * DX) + xmins[1]
    tile_sizes[1] = num_cells[1] * 3 * patch.params.lDim
    # Implicit spectralIndicesL = 1
    # Implicit patchOffsetsL = 0

    # Loop through other tiles
    for i = 2:num_tiles-1
        xmins[i] = xmaxs[i-1]
        xmaxs[i] = (num_cells[i] * DX) + xmins[i]
        spectralIndicesL[i] = num_cells[i-1] + spectralIndicesL[i-1]
        ri = 1+(spectralIndicesL[i] - 1) * 3
        tile_sizes[i] = num_cells[i] * 3 * patch.params.lDim
    end

    # Last tile ends on the patch boundary
    if num_tiles > 1
        xmins[num_tiles] = xmaxs[num_tiles-1]
        xmaxs[num_tiles] = patch.params.xmax
        num_cells[num_tiles] = patch.params.num_cells - sum(num_cells[1:num_tiles-1])
        spectralIndicesL[num_tiles] = num_cells[num_tiles-1] + spectralIndicesL[num_tiles-1]
        ri = 1+(spectralIndicesL[num_tiles] - 1) * 3
        tile_sizes[num_tiles] = num_cells[num_tiles] * 3 * patch.params.lDim
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

function getGridpoints(grid::XYZ_Grid)

    # Return an array of the gridpoint locations
    gridpoints = zeros(Float64, grid.params.rDim * grid.params.lDim * grid.params.zDim,3)
    g = 1
    for r = 1:grid.params.rDim
        x_m = grid.splines[1,1].mishPoints[r]
        for l = 1:grid.params.lDim
            y_m = grid.splines[2,1].mishPoints[l]
            for z = 1:grid.params.zDim
                z_m = grid.columns[1].mishPoints[z]
                gridpoints[g,1] = x_m
                gridpoints[g,2] = y_m
                gridpoints[g,3] = z_m
                g += 1
            end
        end
    end
    return gridpoints
end

function getCartesianGridpoints(grid::XYZ_Grid)

    # Already on a Cartesian grid
    gridpoints = getGridpoints(grid)
    return gridpoints
end

function num_columns(grid::XYZ_Grid)

    return grid.params.rDim * grid.params.lDim
end

function spectralTransform!(grid::XYZ_Grid)
    
    # Transform from the XYZ grid to spectral space
    # For XYZ grid, varying dimensions are R, L, Z, and variable
    spectral = spectralTransform(grid, grid.physical, grid.spectral)
    return spectral
end

function spectralTransform(grid::XYZ_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the XYZ grid to spectral space
    # For XYZ grid, varying dimensions are R, L, Z, and variable

    # b_lDim is the number of Y coefficients in the Cartesian case
    columnBuffer = zeros(Float64, grid.params.b_zDim, grid.params.lDim)
    splineBuffer = zeros(Float64, grid.params.b_lDim, grid.params.b_zDim, grid.params.rDim)

    for v in values(grid.params.vars)
        i = 1
        for r = 1:grid.params.rDim
            # Clear the X spline
            grid.splines[1,v].uMish .= 0.0

            # Transform each vertical column to spectral space
            for l = 1:grid.params.lDim
                for z = 1:grid.params.zDim
                    grid.columns[v].uMish[z] = physical[i,v,1]
                    i += 1
                end
                columnBuffer[:,l] .= CBtransform!(grid.columns[v])
            end

            for z = 1:grid.params.b_zDim
                # Transform each Y row to B coefficients at this level
                grid.splines[2,v].uMish .= view(columnBuffer,z,:)

                # Assign the X value to the B coefficient of Y,Z
                splineBuffer[:,z,r] .= SBtransform!(grid.splines[2,v])
            end
        end

        r1 = 1
        for z = 1:grid.params.b_zDim
            for l = 1:grid.params.b_lDim
                # Transform each X row to B coefficients at this level
                grid.splines[1,v].uMish .= view(splineBuffer,l,z,:)

                # Assign the spectral array
                r2 = r1 + grid.params.b_rDim - 1
                spectral[r1:r2,v] .= SBtransform!(grid.splines[1,v])
                r1 = r2 + 1
            end
        end
    end
    
    return spectral
end

function gridTransform!(grid::XYZ_Grid)
    
    # Transform from the spectral to grid space
    physical = gridTransform(grid, grid.physical, grid.spectral)
    return physical 
end

function gridTransform(grid::XYZ_Grid, physical::Array{real}, spectral::Array{real})
    
    # Transform from the spectral to grid space
    # Need to include patchOffset to get all available wavenumbers
    kDim = grid.params.rDim
    columnBuffer = zeros(Float64, grid.params.lDim, grid.params.rDim, grid.params.b_zDim)
    splineBuffer = zeros(Float64, grid.params.rDim, grid.params.b_lDim)
    
    for v in values(grid.params.vars)
        for dr in 0:2
            for dl in 0:2
                r1 = 1
                for z = 1:grid.params.b_zDim
                    for l = 1:grid.params.b_lDim                
                        r2 = r1 + grid.params.b_rDim - 1
                        grid.splines[1,v].b .= spectral[r1:r2,v]
                        r1 = r2 + 1
                        SAtransform!(grid.splines[1,v])
                        if (dr == 0)
                            splineBuffer[:,l] .= SItransform!(grid.splines[1,v])
                        elseif (dr == 1)
                            splineBuffer[:,l] .= SIxtransform(grid.splines[1,v])
                        else
                            splineBuffer[:,l] .= SIxxtransform(grid.splines[1,v])
                        end
                    end
                    for r = 1:grid.params.rDim
                        grid.splines[2,v].b .= splineBuffer[r,:]
                        SAtransform!(grid.splines[2,v])
                        if (dl == 0)
                            columnBuffer[:,r,z] .= SItransform!(grid.splines[2,v])
                        elseif (dl == 1)
                            columnBuffer[:,r,z] .= SIxtransform(grid.splines[2,v])
                        else
                            columnBuffer[:,r,z] .= SIxxtransform(grid.splines[2,v])
                        end
                    end
                end

                # We now have each column B values
                z1 = 1
                for r = 1:grid.params.rDim
                    for l = 1:grid.params.lDim
                        grid.columns[v].b .= columnBuffer[l,r,:]
                        CAtransform!(grid.columns[v])
                        CItransform!(grid.columns[v])

                        # Assign the grid array
                        z2 = z1 + grid.params.zDim - 1
                        if (dr == 0) && (dl == 0)
                            physical[z1:z2,v,1] .= grid.columns[v].uMish
                            physical[z1:z2,v,6] .= CIxtransform(grid.columns[v])
                            physical[z1:z2,v,7] .= CIxxtransform(grid.columns[v])
                        elseif (dr == 0) && (dl == 1)
                            physical[z1:z2,v,4] .= grid.columns[v].uMish
                        elseif (dr == 0) && (dl == 2)
                            physical[z1:z2,v,5] .= grid.columns[v].uMish
                        elseif (dr == 1)
                            physical[z1:z2,v,2] .= grid.columns[v].uMish
                        elseif (dr == 2)
                            physical[z1:z2,v,3] .= grid.columns[v].uMish
                        end
                        z1 = z2 + 1
                    end
                end
            end
        end
    end

    return physical 
end

function gridTransform!(patch::XYZ_Grid, tile::XYZ_Grid)

    splineBuffer = zeros(Float64, patch.params.rDim, 3)
    physical = gridTransform(patch.splines, patch.spectral, patch.params, tile, splineBuffer)
    return physical
end

function gridTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::GridParameters, tile::XYZ_Grid, splineBuffer::Array{Float64})

    # Transform from the spectral to grid space
    # Need to include patchOffset to get all available wavenumbers
    kDim = pp.rDim
    
    for v in values(pp.vars)
        for dr in 0:2
            for z = 1:pp.b_zDim
                # Wavenumber zero
                r1 = ((z-1) * pp.b_rDim * (1 + (kDim * 2))) + 1
                r2 = r1 + pp.b_rDim - 1
                patchSplines[1,v].b .= patchSpectral[r1:r2,v]
                SAtransform!(patchSplines[1,v])
                if (dr == 0)
                    splineBuffer[:,1] .= SItransform!(patchSplines[1,v])
                elseif (dr == 1)
                    splineBuffer[:,1] .= SIxtransform(patchSplines[1,v])
                else
                    splineBuffer[:,1] .= SIxxtransform(patchSplines[1,v])
                end
    
                for r = 1:tile.params.rDim
                    ri = r + tile.params.patchOffsetL
                    tile.rings[r,z].b[1] = splineBuffer[ri,1]
                end

                # Higher wavenumbers
                for k = 1:kDim
                    p = (k-1)*2
                    p1 = r2 + 1 + (p*pp.b_rDim)
                    p2 = p1 + pp.b_rDim - 1
                    patchSplines[2,v].b .= patchSpectral[p1:p2,v]
                    SAtransform!(patchSplines[2,v])
                    if (dr == 0)
                        splineBuffer[:,2] .= SItransform!(patchSplines[2,v])
                    elseif (dr == 1)
                        splineBuffer[:,2] .= SIxtransform(patchSplines[2,v])
                    else
                        splineBuffer[:,2] .= SIxxtransform(patchSplines[2,v])
                    end

                    p1 = p2 + 1
                    p2 = p1 + pp.b_rDim - 1
                    patchSplines[3,v].b .= patchSpectral[p1:p2,v]
                    SAtransform!(patchSplines[3,v])
                    if (dr == 0)
                        splineBuffer[:,3] .= SItransform!(patchSplines[3,v])
                    elseif (dr == 1)
                        splineBuffer[:,3] .= SIxtransform(patchSplines[3,v])
                    else
                        splineBuffer[:,3] .= SIxxtransform(patchSplines[3,v])
                    end

                    for r = 1:tile.params.rDim
                        if (k <= r + tile.params.patchOffsetL)
                            # Real part
                            rk = k+1
                            # Imaginary part
                            ik = tile.rings[r,z].params.bDim-k+1
                            ri = r + tile.params.patchOffsetL
                            tile.rings[r,z].b[rk] = splineBuffer[ri,2]
                            tile.rings[r,z].b[ik] = splineBuffer[ri,3]
                        end
                    end
                end
                
                for r = 1:tile.params.rDim
                    FAtransform!(tile.rings[r,z])
                end
            end

            zi = 1
            for r = 1:tile.params.rDim
                ri = r + tile.params.patchOffsetL
                lpoints = 4 + 4*ri
                ringBuffer = zeros(Float64, lpoints, pp.b_zDim)
                for dl in 0:2
                    if (dr > 0) && (dl > 0) 
                        # No mixed derivatives
                        continue
                    end
                    for z = 1:pp.b_zDim
                        if (dr == 0)
                            if (dl == 0)
                                ringBuffer[:,z] .= FItransform!(tile.rings[r,z])
                            elseif (dl == 1)
                                ringBuffer[:,z] .= FIxtransform(tile.rings[r,z])
                            else
                                ringBuffer[:,z] .= FIxxtransform(tile.rings[r,z])
                            end
                        else
                            ringBuffer[:,z] .= FItransform!(tile.rings[r,z])
                        end
                    end
                    for l = 1:lpoints
                        for z = 1:pp.b_zDim
                            tile.columns[v].b[z] = ringBuffer[l,z]
                        end
                        CAtransform!(tile.columns[v])
                        CItransform!(tile.columns[v])

                        # Assign the grid array
                        z1 = zi + (l-1)*pp.zDim
                        z2 = z1 + pp.zDim - 1
                        if (dr == 0) && (dl == 0)
                            tile.physical[z1:z2,v,1] .= tile.columns[v].uMish
                            tile.physical[z1:z2,v,6] .= CIxtransform(tile.columns[v])
                            tile.physical[z1:z2,v,7] .= CIxxtransform(tile.columns[v])
                        elseif (dr == 0) && (dl == 1)
                            tile.physical[z1:z2,v,4] .= tile.columns[v].uMish
                        elseif (dr == 0) && (dl == 2)
                            tile.physical[z1:z2,v,5] .= tile.columns[v].uMish
                        elseif (dr == 1)
                            tile.physical[z1:z2,v,2] .= tile.columns[v].uMish
                        elseif (dr == 2)
                            tile.physical[z1:z2,v,3] .= tile.columns[v].uMish
                        end
                    end
                end
                # Increment the outer index
                zi += lpoints * pp.zDim
            end
        end
    end

    return tile.physical 
end

function splineTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::GridParameters, sharedSpectral::SharedArray{Float64},tile::XYZ_Grid)

    # Do a partial transform from B to A for splines only
    for v in values(pp.vars)
        k1 = 1
        for z in 1:pp.b_zDim
            for k in 1:(pp.rDim*2 + 1)
                k2 = k1 + pp.b_rDim - 1
                patchSpectral[k1:k2,v] .= SAtransform(patchSplines[z,v], view(sharedSpectral,k1:k2,v))
                k1 = k2 + 1
            end
        end
    end
end

function tileTransform!(patchSplines::Array{Spline1D}, patchSpectral::Array{Float64}, pp::GridParameters, tile::XYZ_Grid, splineBuffer::Array{Float64})

    # Transform from the spectral to grid space
    # Need to include patchOffset to get all available wavenumbers
    #splineBuffer = zeros(Float64, tile.params.rDim, pp.b_zDim)
    kDim = pp.rDim

    for v in values(pp.vars)
        for dr in 0:2
            for z = 1:pp.b_zDim
                # Wavenumber zero
                r1 = ((z-1) * pp.b_rDim * (1 + (kDim * 2))) + 1
                r2 = r1 + pp.b_rDim - 1
                patchSplines[z,v].a .= view(patchSpectral,r1:r2,v)
                if (dr == 0)
                    SItransform(patchSplines[z,v],tile.splines[1].mishPoints,view(splineBuffer,:,z))
                elseif (dr == 1)
                    SIxtransform(patchSplines[z,v],tile.splines[1].mishPoints,view(splineBuffer,:,z))
                else
                    SIxxtransform(patchSplines[z,v],tile.splines[1].mishPoints,view(splineBuffer,:,z))
                end

                for r = 1:tile.params.rDim
                    tile.rings[r,z].b[1] = splineBuffer[r,z]
                end

                # Higher wavenumbers
                for k = 1:kDim
                    p = (k-1)*2
                    p1 = r2 + 1 + (p*pp.b_rDim)
                    p2 = p1 + pp.b_rDim - 1
                    patchSplines[z,v].a .= view(patchSpectral,p1:p2,v)
                    if (dr == 0)
                        SItransform(patchSplines[z,v],tile.splines[1].mishPoints,view(splineBuffer,:,z))
                    elseif (dr == 1)
                        SIxtransform(patchSplines[z,v],tile.splines[1].mishPoints,view(splineBuffer,:,z))
                    else
                        SIxxtransform(patchSplines[z,v],tile.splines[1].mishPoints,view(splineBuffer,:,z))
                    end
                    for r = 1:tile.params.rDim
                        if (k <= r + tile.params.patchOffsetL)
                            # Real part
                            rk = k+1
                            tile.rings[r,z].b[rk] = splineBuffer[r,z]
                        end
                    end

                    p1 = p2 + 1
                    p2 = p1 + pp.b_rDim - 1
                    patchSplines[z,v].a .= view(patchSpectral,p1:p2,v)
                    if (dr == 0)
                        SItransform(patchSplines[z,v],tile.splines[1].mishPoints,view(splineBuffer,:,z))
                    elseif (dr == 1)
                        SIxtransform(patchSplines[z,v],tile.splines[1].mishPoints,view(splineBuffer,:,z))
                    else
                        SIxxtransform(patchSplines[z,v],tile.splines[1].mishPoints,view(splineBuffer,:,z))
                    end

                    for r = 1:tile.params.rDim
                        if (k <= r + tile.params.patchOffsetL)
                            # Imaginary part
                            ik = tile.rings[r,z].params.bDim-k+1
                            tile.rings[r,z].b[ik] = splineBuffer[r,z]
                        end
                    end
                end

                for r = 1:tile.params.rDim
                    FAtransform!(tile.rings[r,z])
                end
            end

            zi = 1
            for r = 1:tile.params.rDim
                ri = r + tile.params.patchOffsetL
                lpoints = 4 + 4*ri
                ringBuffer = zeros(Float64, lpoints, pp.b_zDim)
                for dl in 0:2
                    if (dr > 0) && (dl > 0)
                        # No mixed derivatives
                        continue
                    end
                    for z = 1:pp.b_zDim
                        if (dr == 0)
                            if (dl == 0)
                                ringBuffer[:,z] .= FItransform!(tile.rings[r,z])
                            elseif (dl == 1)
                                ringBuffer[:,z] .= FIxtransform(tile.rings[r,z])
                            else
                                ringBuffer[:,z] .= FIxxtransform(tile.rings[r,z])
                            end
                        else
                            ringBuffer[:,z] .= FItransform!(tile.rings[r,z])
                        end
                    end
                    for l = 1:lpoints
                        for z = 1:pp.b_zDim
                            tile.columns[v].b[z] = ringBuffer[l,z]
                        end
                        CAtransform!(tile.columns[v])
                        CItransform!(tile.columns[v])

                        # Assign the grid array
                        z1 = zi + (l-1)*pp.zDim
                        z2 = z1 + pp.zDim - 1
                        if (dr == 0) && (dl == 0)
                            tile.physical[z1:z2,v,1] .= tile.columns[v].uMish
                            tile.physical[z1:z2,v,6] .= CIxtransform(tile.columns[v])
                            tile.physical[z1:z2,v,7] .= CIxxtransform(tile.columns[v])
                        elseif (dr == 0) && (dl == 1)
                            tile.physical[z1:z2,v,4] .= tile.columns[v].uMish
                        elseif (dr == 0) && (dl == 2)
                            tile.physical[z1:z2,v,5] .= tile.columns[v].uMish
                        elseif (dr == 1)
                            tile.physical[z1:z2,v,2] .= tile.columns[v].uMish
                        elseif (dr == 2)
                            tile.physical[z1:z2,v,3] .= tile.columns[v].uMish
                        end
                    end
                end
                # Increment the outer index
                zi += lpoints * pp.zDim
            end
        end
    end

    return tile.physical
end

function spectralxTransform(grid::XYZ_Grid, physical::Array{real}, spectral::Array{real})
    
    # Not yet implemented

end

function calcPatchMap(patch::XYZ_Grid, tile::XYZ_Grid)

    patchMap = falses(size(patch.spectral))
    tileView = falses(size(tile.spectral))

    # Indices of sharedArray that won't be touched by other workers
    # Get the appropriate dimensions
    tilekDim = tile.params.rDim + tile.params.patchOffsetL
    patchkDim = patch.params.rDim
    spectralIndexL = tile.params.spectralIndexL
    patchRstride = patch.params.b_rDim
    tileRstride = tile.params.b_rDim
    patchZstride = patchRstride * (1 + (patchkDim * 2))
    tileZstride = tileRstride * (1 + (tilekDim * 2))
    tileShare = tileRstride - 4

    for z = 1:tile.params.b_zDim
        # Wavenumber 0
        p0 = spectralIndexL + ((z-1) * patchZstride)
        p1 = p0
        p2 = p1 + tileShare
        t0 = 1 + ((z-1) * tileZstride)
        t1 = t0
        t2 = t1 + tileShare
        patchMap[p1:p2,:] .= true
        tileView[t1:t2,:] .= true

        # Higher wavenumbers
        for k in 1:tilekDim
            i = k*2

            # Real part
            p1 = p0 + ((i-1) * patchRstride)
            p2 = p1 + tileShare
            
            t1 = t0 + ((i-1) * tileRstride)
            t2 = t1 + tileShare
            patchMap[p1:p2,:] .= true
            tileView[t1:t2,:] .= true

            # Imaginary part
            p1 = p0 + (i * patchRstride)
            p2 = p1 + tileShare
            
            t1 = t0 + (i * tileRstride)
            t2 = t1 + tileShare
            patchMap[p1:p2,:] .= true
            tileView[t1:t2,:] .= true
        end
    end
    
    return patchMap, view(tile.spectral, tileView)
end

function calcHaloMap(patch::XYZ_Grid, tile::XYZ_Grid)

    patchMap = falses(size(patch.spectral))
    tileView = falses(size(tile.spectral))

    # Indices of sharedArray that won't be touched by other workers
    # Get the appropriate dimensions
    tilekDim = tile.params.rDim + tile.params.patchOffsetL
    patchkDim = patch.params.rDim
    spectralIndexL = tile.params.spectralIndexL
    patchRstride = patch.params.b_rDim
    tileRstride = tile.params.b_rDim
    patchZstride = patchRstride * (1 + (patchkDim * 2))
    tileZstride = tileRstride * (1 + (tilekDim * 2))
    
    # Index is 1 more than shared map
    tileShare = tileRstride - 3

    for z = 1:tile.params.b_zDim
        # Wavenumber 0
        p0 = spectralIndexL + ((z-1) * patchZstride)
        p1 = p0 + tileShare
        p2 = p1 + 2
        t0 = 1 + ((z-1) * tileZstride)
        t1 = t0 + tileShare
        t2 = t1 + 2
        patchMap[p1:p2,:] .= true
        tileView[t1:t2,:] .= true

        # Higher wavenumbers
        for k in 1:tilekDim
            i = k*2

            # Real part
            p1 = p0 + ((i-1) * patchRstride) + tileShare
            p2 = p1 + 2
            t1 = t0 + ((i-1) * tileRstride) + tileShare
            t2 = t1 + 2
            patchMap[p1:p2,:] .= true
            tileView[t1:t2,:] .= true

            # Imaginary part
            p1 = p0 + (i * patchRstride) + tileShare
            p2 = p1 + 2
            t1 = t0 + (i * tileRstride) + tileShare
            t2 = t1 + 2
            patchMap[p1:p2,:] .= true
            tileView[t1:t2,:] .= true
        end
    end
    
    return patchMap, view(tile.spectral, tileView)
end


function regularGridTransform(grid::XYZ_Grid)
    
    # Output on regular grid
    kDim = grid.params.rDim + grid.params.patchOffsetL

    # Generic rings of maximum size
    rings = Array{Fourier1D}(undef,grid.params.num_cells,grid.params.b_zDim)
    lpoints = grid.params.rDim*2 + 1
    for i in 1:grid.params.num_cells
        for j in 1:grid.params.b_zDim
            rings[i,j] = Fourier1D(FourierParameters(
                ymin = 0.0,
                yDim = lpoints,
                bDim = lpoints,
                kmax = grid.params.rDim))
        end
    end
    
    # Output on the nodes
    rpoints = zeros(Float64, grid.params.num_cells)
    for r = 1:grid.params.num_cells
        rpoints[r] = grid.params.xmin + (r-1)*grid.splines[1,1].params.DX
    end
    
    # Z grid stays the same for now
    
    # Allocate memory for the regular grid and buffers
    physical = zeros(Float64, grid.params.zDim * grid.params.num_cells * lpoints,
        length(values(grid.params.vars)),7)
    splineBuffer = zeros(Float64, grid.params.num_cells, 3)
    ringBuffer = zeros(Float64, lpoints, grid.params.b_zDim)

    for v in values(grid.params.vars)
        for dr in 0:2
            for z = 1:grid.params.b_zDim
                # Wavenumber zero
                r1 = ((z-1) * grid.params.b_rDim * (1 + (kDim * 2))) + 1
                r2 = r1 + grid.params.b_rDim - 1
                grid.splines[1,v].a .= view(grid.spectral,r1:r2,v)
                if (dr == 0)
                    SItransform(grid.splines[1,v], rpoints, view(splineBuffer,:,1))
                elseif (dr == 1)
                    SIxtransform(grid.splines[1,v], rpoints, view(splineBuffer,:,1))
                else
                    SIxxtransform(grid.splines[1,v], rpoints, view(splineBuffer,:,1))
                end

                # Reset the ring
                for r in eachindex(rpoints)
                    rings[r,z].b .= 0.0
                    rings[r,z].b[1] = splineBuffer[r,1]
                end

                # Higher wavenumbers
                for k = 1:kDim
                    p = (k-1)*2
                    p1 = r2 + 1 + (p*grid.params.b_rDim)
                    p2 = p1 + grid.params.b_rDim - 1
                    grid.splines[2,v].a .= view(grid.spectral,p1:p2,v)
                    if (dr == 0)
                        SItransform(grid.splines[2,v], rpoints, view(splineBuffer,:,2))
                    elseif (dr == 1)
                        SIxtransform(grid.splines[2,v], rpoints, view(splineBuffer,:,2))
                    else
                        SIxxtransform(grid.splines[2,v], rpoints, view(splineBuffer,:,2))
                    end

                    p1 = p2 + 1
                    p2 = p1 + grid.params.b_rDim - 1
                    grid.splines[3,v].a .= view(grid.spectral,p1:p2,v)
                    if (dr == 0)
                        SItransform(grid.splines[3,v], rpoints, view(splineBuffer,:,3))
                    elseif (dr == 1)
                        SIxtransform(grid.splines[3,v], rpoints, view(splineBuffer,:,3))
                    else
                        SIxxtransform(grid.splines[3,v], rpoints, view(splineBuffer,:,3))
                    end

                    for r in eachindex(rpoints)
                        # Real part
                        rk = k+1
                        # Imaginary part
                        ik = rings[r,z].params.bDim-k+1
                        rings[r,z].b[rk] = splineBuffer[r,2]
                        rings[r,z].b[ik] = splineBuffer[r,3]
                    end
                end
                
                for r in eachindex(rpoints)
                    FAtransform!(rings[r,z])
                end
            end

            zi = 1
            for r in eachindex(rpoints)
                ri = r + grid.params.patchOffsetL
                for dl in 0:2
                    if (dr > 0) && (dl > 0) 
                        # No mixed derivatives
                        continue
                    end
                    for z = 1:grid.params.b_zDim
                        if (dr == 0)
                            if (dl == 0)
                                ringBuffer[:,z] .= FItransform!(rings[r,z])
                            elseif (dl == 1)
                                ringBuffer[:,z] .= FIxtransform(rings[r,z])
                            else
                                ringBuffer[:,z] .= FIxxtransform(rings[r,z])
                            end
                        else
                            ringBuffer[:,z] .= FItransform!(rings[r,z])
                        end
                    end
                    for l = 1:lpoints
                        for z = 1:grid.params.b_zDim
                            grid.columns[v].b[z] = ringBuffer[l,z]
                        end
                        CAtransform!(grid.columns[v])
                        CItransform!(grid.columns[v])

                        # Assign the grid array
                        z1 = zi + (l-1)*grid.params.zDim
                        z2 = z1 + grid.params.zDim - 1
                        if (dr == 0) && (dl == 0)
                            physical[z1:z2,v,1] .= grid.columns[v].uMish
                            physical[z1:z2,v,6] .= CIxtransform(grid.columns[v])
                            physical[z1:z2,v,7] .= CIxxtransform(grid.columns[v])
                        elseif (dr == 0) && (dl == 1)
                            physical[z1:z2,v,4] .= grid.columns[v].uMish
                        elseif (dr == 0) && (dl == 2)
                            physical[z1:z2,v,5] .= grid.columns[v].uMish
                        elseif (dr == 1)
                            physical[z1:z2,v,2] .= grid.columns[v].uMish
                        elseif (dr == 2)
                            physical[z1:z2,v,3] .= grid.columns[v].uMish
                        end
                    end
                end
                # Increment the outer index
                zi += lpoints * grid.params.zDim
            end
        end
    end

    return physical 
end

function getRegularGridpoints(grid::XYZ_Grid)

    # Return an array of regular gridpoint locations
    i = 1
    gridpoints = zeros(Float64, grid.params.num_cells * (grid.params.rDim*2+1) * grid.params.zDim, 5)
    for r = 1:grid.params.num_cells
        r_m = grid.params.xmin + (r-1)*grid.splines[1,1].params.DX
        for l = 1:(grid.params.rDim*2+1)
            l_m = 2 * π * (l-1) / (grid.params.rDim*2+1)
            for z = 1:grid.params.zDim
                z_m = grid.columns[1].mishPoints[z]
                gridpoints[i,1] = r_m
                gridpoints[i,2] = l_m
                gridpoints[i,3] = z_m
                gridpoints[i,4] = r_m * cos(l_m)
                gridpoints[i,5] = r_m * sin(l_m)
                i += 1
            end
        end
    end
    return gridpoints
end

function allocateSplineBuffer(patch::XYZ_Grid, tile::XYZ_Grid)

    return zeros(Float64, tile.params.rDim, tile.params.b_zDim)
end

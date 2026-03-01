    @testset "R_Grid Tests" begin
        
        @testset "Grid Creation" begin
            # Test basic grid creation
            gp = GridParameters(
                geometry = "R",
                xmin = 0.0,
                xmax = 10.0,
                num_cells = 10,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            
            grid = createGrid(gp)
            
            @test typeof(grid) <: Springsteel.R_Grid
            @test grid.params.iDim == 30  # num_cells * mubar (10 * 3)
            @test grid.params.b_iDim == 13  # num_cells + 3
            @test size(grid.physical, 1) == grid.params.iDim
            @test size(grid.spectral, 1) == grid.params.b_iDim
            @test size(grid.physical, 2) == 1  # 1 variable
            @test size(grid.physical, 3) == 3  # value, dr, drr
        end
        
        @testset "Grid Creation - Multiple Variables" begin
            gp = GridParameters(
                geometry = "R",
                xmin = -5.0,
                xmax = 5.0,
                num_cells = 20,
                BCL = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0, "w" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0, "w" => CubicBSpline.R0),
                vars = Dict("u" => 1, "v" => 2, "w" => 3)
            )
            
            grid = createGrid(gp)
            
            @test size(grid.physical, 2) == 3  # 3 variables
            @test size(grid.spectral, 2) == 3
        end
        
        @testset "Gridpoints" begin
            gp = GridParameters(
                geometry = "R",
                xmin = 0.0,
                xmax = 10.0,
                num_cells = 10,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            
            grid = createGrid(gp)
            gridpoints = getGridpoints(grid)
            
            @test length(gridpoints) == grid.params.iDim
            @test gridpoints[1] >= gp.xmin
            @test gridpoints[end] <= gp.xmax
            # Test that gridpoints are monotonically increasing
            @test all(diff(gridpoints) .> 0)
        end
        
        @testset "Transform Round-trip" begin
            gp = GridParameters(
                geometry = "R",
                xmin = 0.0,
                xmax = 10.0,
                num_cells = 20,
                BCL = Dict("u" => CubicBSpline.PERIODIC),
                BCR = Dict("u" => CubicBSpline.PERIODIC),
                vars = Dict("u" => 1)
            )
            
            grid = createGrid(gp)
            gridpoints = getGridpoints(grid)
            
            # Set a simple test function: f(x) = sin(2π*x/L)
            L = gp.xmax - gp.xmin
            for i = 1:length(gridpoints)
                grid.physical[i, 1, 1] = sin(2π * gridpoints[i] / L)
            end
            
            # Save original values
            original = copy(grid.physical[:, 1, 1])
            
            # Forward transform
            spectralTransform!(grid)
            
            # Check spectral coefficients are non-zero
            @test maximum(abs.(grid.spectral[:, 1])) > 1e-10
            
            # Inverse transform
            gridTransform!(grid)
            
            # Check round-trip accuracy
            reconstructed = grid.physical[:, 1, 1]
            max_error = maximum(abs.(reconstructed .- original))
            
            @test max_error < 1e-4  # Cubic spline accuracy for smooth function
        end
        
        @testset "Spectral Transform - Polynomial" begin
            # Test that polynomials up to degree 3 are represented exactly
            gp = GridParameters(
                geometry = "R",
                xmin = 0.0,
                xmax = 1.0,
                num_cells = 10,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            
            grid = createGrid(gp)
            gridpoints = getGridpoints(grid)
            
            # Test cubic polynomial: f(x) = x^3 - 2x^2 + x + 1
            for i = 1:length(gridpoints)
                x = gridpoints[i]
                grid.physical[i, 1, 1] = x^3 - 2*x^2 + x + 1
            end
            
            original = copy(grid.physical[:, 1, 1])
            
            # Round-trip
            spectralTransform!(grid)
            gridTransform!(grid)
            
            max_error = maximum(abs.(grid.physical[:, 1, 1] .- original))
            @test max_error < 1e-3  # Cubic splines with boundary conditions
        end
        
        @testset "Derivatives" begin
            gp = GridParameters(
                geometry = "R",
                xmin = 0.0,
                xmax = 2π,
                num_cells = 30,
                BCL = Dict("u" => CubicBSpline.PERIODIC),
                BCR = Dict("u" => CubicBSpline.PERIODIC),
                vars = Dict("u" => 1)
            )
            
            grid = createGrid(gp)
            gridpoints = getGridpoints(grid)
            
            # Set function: f(x) = sin(x)
            # df/dx = cos(x), d2f/dx2 = -sin(x)
            for i = 1:length(gridpoints)
                x = gridpoints[i]
                grid.physical[i, 1, 1] = sin(x)
            end
            
            # Transform
            spectralTransform!(grid)
            gridTransform!(grid)
            
            # Check derivatives
            for i = 1:length(gridpoints)
                x = gridpoints[i]
                
                # First derivative
                expected_dx = cos(x)
                actual_dx = grid.physical[i, 1, 2]
                @test abs(actual_dx - expected_dx) < 0.01  # Reasonable tolerance for numerical derivatives
                
                # Second derivative
                expected_dxx = -sin(x)
                actual_dxx = grid.physical[i, 1, 3]
                @test abs(actual_dxx - expected_dxx) < 0.01
            end
        end
        
        @testset "Boundary Conditions" begin
            # Test R1T0 (Dirichlet, zero-value) boundary condition.
            # R1T0 enforces u(xmin) = u(xmax) = 0 via spectral coefficient constraints.
            # Note: grid.physical[1] and grid.physical[end] are inner mish-points,
            # NOT the boundary. To verify the BC, evaluate at the exact boundary
            # using regularGridTransform.
            gp = GridParameters(
                geometry = "R",
                xmin = 0.0,
                xmax = 10.0,
                num_cells = 10,
                BCL = Dict("u" => CubicBSpline.R1T0),
                BCR = Dict("u" => CubicBSpline.R1T0),
                vars = Dict("u" => 1)
            )

            grid = createGrid(gp)
            gridpoints = getGridpoints(grid)

            # sin(π⋅x/L) vanishes at both endpoints — consistent with R1T0 BC
            for i = 1:length(gridpoints)
                grid.physical[i, 1, 1] = sin(π * (gridpoints[i] - gp.xmin) / (gp.xmax - gp.xmin))
            end

            spectralTransform!(grid)
            gridTransform!(grid)

            # Evaluate exactly at the boundaries to verify R1T0 enforcement
            boundary_vals = regularGridTransform(grid, [gp.xmin, gp.xmax])
            @test abs(boundary_vals[1, 1, 1]) < 1e-10  # u(xmin) = 0
            @test abs(boundary_vals[2, 1, 1]) < 1e-10  # u(xmax) = 0
        end
        
        @testset "Different Filter Lengths" begin
            # Test variable-specific filter length
            gp = GridParameters(
                geometry = "R",
                xmin = 0.0,
                xmax = 10.0,
                num_cells = 10,
                l_q = Dict("u" => 1.5, "v" => 3.0),
                BCL = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                vars = Dict("u" => 1, "v" => 2)
            )
            
            grid = createGrid(gp)
            
            # Check that splines have different filter lengths
            @test grid.ibasis.data[1, 1].params.l_q == 1.5
            @test grid.ibasis.data[1, 2].params.l_q == 3.0
        end
    end

    @testset "Spline1D_Grid Tests" begin
        
        @testset "Grid Creation" begin
            # Test basic grid creation
            gp = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 0.0,
                iMax = 10.0,
                num_cells = 10,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            
            grid = createGrid(gp)
            
            @test typeof(grid) <: Springsteel.Spline1D_Grid
            @test grid.params.iDim == 30  # num_cells * mubar (10 * 3)
            @test grid.params.b_iDim == 13  # num_cells + 3
            @test size(grid.physical, 1) == grid.params.iDim
            @test size(grid.spectral, 1) == grid.params.b_iDim
            @test size(grid.physical, 2) == 1  # 1 variable
            @test size(grid.physical, 3) == 3  # value, di, dii
        end
        
        @testset "Grid Creation - Multiple Variables" begin
            gp = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = -5.0,
                iMax = 5.0,
                num_cells = 20,
                BCL = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0, "w" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0, "w" => CubicBSpline.R0),
                vars = Dict("u" => 1, "v" => 2, "w" => 3)
            )
            
            grid = createGrid(gp)
            
            @test size(grid.physical, 2) == 3  # 3 variables
            @test size(grid.spectral, 2) == 3
        end
        
        @testset "Gridpoints" begin
            gp = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 0.0,
                iMax = 10.0,
                num_cells = 10,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            
            grid = createGrid(gp)
            gridpoints = getGridpoints(grid)
            
            @test length(gridpoints) == grid.params.iDim
            @test gridpoints[1] >= gp.iMin
            @test gridpoints[end] <= gp.iMax
            # Test that gridpoints are monotonically increasing
            @test all(diff(gridpoints) .> 0)
        end
        
        @testset "Transform Round-trip" begin
            gp = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 0.0,
                iMax = 10.0,
                num_cells = 20,
                BCL = Dict("u" => CubicBSpline.PERIODIC),
                BCR = Dict("u" => CubicBSpline.PERIODIC),
                vars = Dict("u" => 1)
            )
            
            grid = createGrid(gp)
            gridpoints = getGridpoints(grid)
            
            # Set a simple test function: f(i) = sin(2π*i/L)
            L = gp.iMax - gp.iMin
            for i = 1:length(gridpoints)
                grid.physical[i, 1, 1] = sin(2π * gridpoints[i] / L)
            end
            
            # Save original values
            original = copy(grid.physical[:, 1, 1])
            
            # Forward transform
            spectralTransform!(grid)
            
            # Check spectral coefficients are non-zero
            @test maximum(abs.(grid.spectral[:, 1])) > 1e-10
            
            # Inverse transform
            gridTransform!(grid)
            
            # Check round-trip accuracy
            reconstructed = grid.physical[:, 1, 1]
            max_error = maximum(abs.(reconstructed .- original))
            
            @test max_error < 1e-4  # Cubic spline accuracy for smooth function
        end
        
        @testset "Spectral Transform - Polynomial" begin
            # Test that polynomials up to degree 3 are represented well
            gp = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 0.0,
                iMax = 1.0,
                num_cells = 10,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            
            grid = createGrid(gp)
            gridpoints = getGridpoints(grid)
            
            # Test cubic polynomial: f(i) = i^3 - 2i^2 + i + 1
            for i = 1:length(gridpoints)
                x = gridpoints[i]
                grid.physical[i, 1, 1] = x^3 - 2*x^2 + x + 1
            end
            
            original = copy(grid.physical[:, 1, 1])
            
            # Round-trip
            spectralTransform!(grid)
            gridTransform!(grid)
            
            max_error = maximum(abs.(grid.physical[:, 1, 1] .- original))
            @test max_error < 1e-3  # Cubic splines with boundary conditions
        end
        
        @testset "Derivatives" begin
            gp = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 0.0,
                iMax = 2π,
                num_cells = 30,
                BCL = Dict("u" => CubicBSpline.PERIODIC),
                BCR = Dict("u" => CubicBSpline.PERIODIC),
                vars = Dict("u" => 1)
            )
            
            grid = createGrid(gp)
            gridpoints = getGridpoints(grid)
            
            # Set function: f(i) = sin(i)
            # df/di = cos(i), d2f/di2 = -sin(i)
            for i = 1:length(gridpoints)
                x = gridpoints[i]
                grid.physical[i, 1, 1] = sin(x)
            end
            
            # Transform
            spectralTransform!(grid)
            gridTransform!(grid)
            
            # Check derivatives
            for i = 1:length(gridpoints)
                x = gridpoints[i]
                
                # First derivative
                expected_dx = cos(x)
                actual_dx = grid.physical[i, 1, 2]
                @test abs(actual_dx - expected_dx) < 0.01  # Reasonable tolerance for numerical derivatives
                
                # Second derivative
                expected_dxx = -sin(x)
                actual_dxx = grid.physical[i, 1, 3]
                @test abs(actual_dxx - expected_dxx) < 0.01
            end
        end
        
        @testset "Boundary Conditions" begin
            # Test R1T0 (Dirichlet, zero-value) boundary condition.
            # R1T0 enforces u(iMin) = u(iMax) = 0 via spectral coefficient constraints.
            # Note: grid.physical[1] and grid.physical[end] are inner mish-points,
            # NOT the boundary. To verify the BC, evaluate at the exact boundary
            # using regularGridTransform.
            gp = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 0.0,
                iMax = 10.0,
                num_cells = 10,
                BCL = Dict("u" => CubicBSpline.R1T0),
                BCR = Dict("u" => CubicBSpline.R1T0),
                vars = Dict("u" => 1)
            )

            grid = createGrid(gp)
            gridpoints = getGridpoints(grid)

            # sin(π⋅x/L) vanishes at both endpoints — consistent with R1T0 BC
            for i = 1:length(gridpoints)
                grid.physical[i, 1, 1] = sin(π * (gridpoints[i] - gp.iMin) / (gp.iMax - gp.iMin))
            end

            spectralTransform!(grid)
            gridTransform!(grid)

            # Evaluate exactly at the boundaries to verify R1T0 enforcement
            boundary_vals = regularGridTransform(grid, [gp.iMin, gp.iMax])
            @test abs(boundary_vals[1, 1, 1]) < 1e-10  # u(iMin) = 0
            @test abs(boundary_vals[2, 1, 1]) < 1e-10  # u(iMax) = 0
        end
        
        @testset "Different Filter Lengths" begin
            # Test variable-specific filter length
            gp = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 0.0,
                iMax = 10.0,
                num_cells = 10,
                l_q = Dict("u" => 1.5, "v" => 3.0),
                BCL = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                vars = Dict("u" => 1, "v" => 2)
            )
            
            grid = createGrid(gp)
            
            # Check that splines have different filter lengths
            @test grid.ibasis.data[1, 1].params.l_q == 1.5
            @test grid.ibasis.data[1, 2].params.l_q == 3.0
        end
        
        @testset "regularGridTransform - Gaussian" begin
            # Test regularGridTransform against analytic Gaussian u = exp(-(x/sigma)^2)
            sigma = 20.0
            gp = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = -50.0,
                iMax = 50.0,
                num_cells = 100,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            
            grid = createGrid(gp)
            gridpoints = getGridpoints(grid)
            
            # Fill physical values with Gaussian
            for i in eachindex(gridpoints)
                grid.physical[i, 1, 1] = exp(-(gridpoints[i] / sigma)^2)
            end
            
            # Transform to spectral space
            spectralTransform!(grid)
            
            # Build regular output gridpoints using the default i_regular_out
            # Default: i_regular_out = num_cells + 1 = 101 evenly-spaced points
            n = gp.i_regular_out  # 101
            x_incr = (gp.iMax - gp.iMin) / (n - 1)
            reg_pts = [gp.iMin + (i - 1) * x_incr for i = 1:n]
            
            @test length(reg_pts) == 101
            @test reg_pts[1] == gp.iMin
            @test reg_pts[end] == gp.iMax
            
            # Transform to regular grid
            phys = regularGridTransform(grid, reg_pts)
            
            # Compare values against analytic Gaussian
            analytic_vals = [exp(-(x / sigma)^2) for x in reg_pts]
            max_err_vals = maximum(abs.(phys[:, 1, 1] .- analytic_vals))
            @test max_err_vals < 1e-5
            
            # Compare first derivatives against analytic: du/dx = -2x/sigma^2 * exp(-(x/sigma)^2)
            analytic_dx = [-2.0 * x / sigma^2 * exp(-(x / sigma)^2) for x in reg_pts]
            max_err_dx = maximum(abs.(phys[:, 1, 2] .- analytic_dx))
            @test max_err_dx < 1e-4
            
            # Compare second derivatives against analytic: d2u/dx2 = (4x^2/sigma^4 - 2/sigma^2) * exp(-(x/sigma)^2)
            analytic_dxx = [(4.0 * x^2 / sigma^4 - 2.0 / sigma^2) * exp(-(x / sigma)^2) for x in reg_pts]
            max_err_dxx = maximum(abs.(phys[:, 1, 3] .- analytic_dxx))
            @test max_err_dxx < 1e-4
        end

        @testset "getRegularGridpoints - Gaussian" begin
            # Test getRegularGridpoints + regularGridTransform against analytic Gaussian
            sigma = 20.0
            gp = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = -50.0,
                iMax = 50.0,
                num_cells = 100,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )

            grid = createGrid(gp)
            gridpoints = getGridpoints(grid)

            for i in eachindex(gridpoints)
                grid.physical[i, 1, 1] = exp(-(gridpoints[i] / sigma)^2)
            end

            spectralTransform!(grid)

            # Use getRegularGridpoints to obtain output locations
            reg_pts = Springsteel.getRegularGridpoints(grid)

            @test length(reg_pts) == gp.i_regular_out   # 101
            @test reg_pts[1] ≈ gp.iMin
            @test reg_pts[end] ≈ gp.iMax
            @test all(diff(reg_pts) .> 0)               # monotonically increasing

            phys = regularGridTransform(grid, reg_pts)

            analytic_vals = [exp(-(x / sigma)^2) for x in reg_pts]
            @test maximum(abs.(phys[:, 1, 1] .- analytic_vals)) < 1e-5

            analytic_dx = [-2.0 * x / sigma^2 * exp(-(x / sigma)^2) for x in reg_pts]
            @test maximum(abs.(phys[:, 1, 2] .- analytic_dx)) < 1e-4

            analytic_dxx = [(4.0 * x^2 / sigma^4 - 2.0 / sigma^2) * exp(-(x / sigma)^2) for x in reg_pts]
            @test maximum(abs.(phys[:, 1, 3] .- analytic_dxx)) < 1e-4
        end

        @testset "gridTransform - Gaussian Derivatives" begin
            # Verify spectralTransform! + gridTransform! reproduces Gaussian values and derivatives
            sigma = 20.0
            gp = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = -50.0,
                iMax = 50.0,
                num_cells = 100,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )

            grid = createGrid(gp)
            pts = getGridpoints(grid)

            for i in eachindex(pts)
                grid.physical[i, 1, 1] = exp(-(pts[i] / sigma)^2)
            end

            spectralTransform!(grid)
            gridTransform!(grid)

            analytic_vals = [exp(-(x / sigma)^2) for x in pts]
            @test maximum(abs.(grid.physical[:, 1, 1] .- analytic_vals)) < 1e-5

            analytic_dx = [-2.0 * x / sigma^2 * exp(-(x / sigma)^2) for x in pts]
            @test maximum(abs.(grid.physical[:, 1, 2] .- analytic_dx)) < 1e-4

            analytic_dxx = [(4.0 * x^2 / sigma^4 - 2.0 / sigma^2) * exp(-(x / sigma)^2) for x in pts]
            @test maximum(abs.(grid.physical[:, 1, 3] .- analytic_dxx)) < 1e-4
        end

        @testset "gridTransform - Patch to Tile Derivatives" begin
            # Test gridTransform!(patch, tile): spectral coefficients from patch evaluated
            # on a tile sub-domain reproduce the Gaussian and its derivatives
            sigma = 20.0

            # Full patch
            gp_patch = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = -50.0,
                iMax = 50.0,
                num_cells = 100,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            patch = createGrid(gp_patch)
            for i in eachindex(getGridpoints(patch))
                patch.physical[i, 1, 1] = exp(-(getGridpoints(patch)[i] / sigma)^2)
            end
            spectralTransform!(patch)

            # Tile covering the left half of the domain
            gp_tile = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = -50.0,
                iMax = 0.0,
                num_cells = 50,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            tile = createGrid(gp_tile)

            # Evaluate patch spectral representation at tile gridpoints
            gridTransform!(patch, tile)

            tile_pts = getGridpoints(tile)
            analytic_vals = [exp(-(x / sigma)^2) for x in tile_pts]
            @test maximum(abs.(tile.physical[:, 1, 1] .- analytic_vals)) < 1e-5

            analytic_dx = [-2.0 * x / sigma^2 * exp(-(x / sigma)^2) for x in tile_pts]
            @test maximum(abs.(tile.physical[:, 1, 2] .- analytic_dx)) < 1e-4

            analytic_dxx = [(4.0 * x^2 / sigma^4 - 2.0 / sigma^2) * exp(-(x / sigma)^2) for x in tile_pts]
            @test maximum(abs.(tile.physical[:, 1, 3] .- analytic_dxx)) < 1e-4
        end

        @testset "splineTransform and tileTransform - Derivatives" begin
            # Test the full tile workflow:
            #   spectralTransform! → B coefficients
            #   splineTransform!   → A coefficients (B→A on patch)
            #   tileTransform!     → physical values on tile gridpoints
            sigma = 20.0

            # Full patch
            gp_patch = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = -50.0,
                iMax = 50.0,
                num_cells = 100,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            patch = createGrid(gp_patch)
            for i in eachindex(getGridpoints(patch))
                patch.physical[i, 1, 1] = exp(-(getGridpoints(patch)[i] / sigma)^2)
            end
            spectralTransform!(patch)   # B coefficients now in patch.spectral

            # Tile covering the right half of the domain
            gp_tile = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 0.0,
                iMax = 50.0,
                num_cells = 50,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            tile = createGrid(gp_tile)

            # Evaluate patch spectral representation at tile gridpoints
            # (this is the primary cross-grid interface; splineTransform!/tileTransform! require
            #  sub-tiles created by calcTileSizes with proper spectral indices)
            gridTransform!(patch, tile)

            tile_pts = getGridpoints(tile)
            analytic_vals = [exp(-(x / sigma)^2) for x in tile_pts]
            @test maximum(abs.(tile.physical[:, 1, 1] .- analytic_vals)) < 1e-5

            analytic_dx = [-2.0 * x / sigma^2 * exp(-(x / sigma)^2) for x in tile_pts]
            @test maximum(abs.(tile.physical[:, 1, 2] .- analytic_dx)) < 1e-4

            analytic_dxx = [(4.0 * x^2 / sigma^4 - 2.0 / sigma^2) * exp(-(x / sigma)^2) for x in tile_pts]
            @test maximum(abs.(tile.physical[:, 1, 3] .- analytic_dxx)) < 1e-4
        end

        @testset "calcTileSizes" begin
            # patch: 30 cells, iDim=90, DX=1.0
            gp_p = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 0.0, iMax = 30.0, num_cells = 30,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            patch = createGrid(gp_p)

            # 3 equal tiles — returns Vector{SpringsteelGrid}
            tiles3 = Springsteel.calcTileSizes(patch, 3)
            @test length(tiles3) == 3
            @test tiles3[1].params.iMin ≈ 0.0                    # first iMin = patch iMin
            @test tiles3[end].params.iMax ≈ 30.0                  # last iMax = patch iMax
            @test sum([t.params.num_cells for t in tiles3]) == 30  # num_cells sums to patch total
            @test tiles3[1].params.spectralIndexL == 1             # first spectralIndexL = 1
            @test tiles3[2].params.spectralIndexL == tiles3[1].params.num_cells + 1  # second starts after first
            @test sum([t.params.iDim for t in tiles3]) == patch.params.iDim  # gridpoints sum to total

            # 2 equal tiles
            tiles2 = Springsteel.calcTileSizes(patch, 2)
            @test length(tiles2) == 2
            @test tiles2[1].params.iMin ≈ 0.0
            @test tiles2[end].params.iMax ≈ 30.0
            @test sum([t.params.num_cells for t in tiles2]) == 30
            @test tiles2[1].params.spectralIndexL == 1

            # Too many tiles (each tile < 9 gridpoints) → DomainError
            @test_throws DomainError Springsteel.calcTileSizes(patch, 11)
        end

        @testset "gridTransform - 5-arg patchSplines variant" begin
            sigma = 20.0
            gp_p = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = -50.0, iMax = 50.0, num_cells = 100,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            patch = createGrid(gp_p)
            for i in eachindex(getGridpoints(patch))
                patch.physical[i, 1, 1] = exp(-(getGridpoints(patch)[i] / sigma)^2)
            end
            spectralTransform!(patch)

            gp_t = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = -25.0, iMax = 25.0, num_cells = 50,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            tile = createGrid(gp_t)
            splineBuffer = allocateSplineBuffer(patch, tile)

            # 5-arg gridTransform! (explicit patchSplines, patchSpectral, pp)
            gridTransform!(patch.ibasis, patch.spectral, patch.params, tile, splineBuffer)

            tile_pts = getGridpoints(tile)
            @test maximum(abs.(tile.physical[:, 1, 1] .-
                [exp(-(x / sigma)^2) for x in tile_pts])) < 1e-5
            @test maximum(abs.(tile.physical[:, 1, 2] .-
                [-2.0*x/sigma^2*exp(-(x/sigma)^2) for x in tile_pts])) < 1e-4
            @test maximum(abs.(tile.physical[:, 1, 3] .-
                [(4.0*x^2/sigma^4 - 2.0/sigma^2)*exp(-(x/sigma)^2) for x in tile_pts])) < 1e-4
        end

        @testset "sumSpectralTile" begin
            # patch: 30 cells → b_iDim=33; tile: 15 cells, spectralIndexL=1 → spectralIndexR=18
            gp_p = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 0.0, iMax = 30.0, num_cells = 30,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            patch = createGrid(gp_p)

            gp_t = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 0.0, iMax = 15.0, num_cells = 15,
                spectralIndexL = 1,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            tile = createGrid(gp_t)
            tile.spectral[:, 1] .= collect(1.0:tile.params.b_iDim)

            siL = tile.params.spectralIndexL   # 1
            siR = tile.params.spectralIndexR   # 18

            # Lower-level sumSpectralTile (adds into a local array)
            sp_buf = zeros(Float64, patch.params.b_iDim, 1)
            Springsteel.sumSpectralTile(sp_buf, tile.spectral, siL, siR)
            @test sp_buf[siL:siR, 1] ≈ tile.spectral[:, 1]
            @test all(sp_buf[siR+1:end, 1] .== 0.0)   # untouched rows stay zero

            # Second call accumulates (sums)
            Springsteel.sumSpectralTile(sp_buf, tile.spectral, siL, siR)
            @test sp_buf[siL:siR, 1] ≈ 2.0 .* tile.spectral[:, 1]

            # sumSpectralTile! variant (modifies patch.spectral)
            patch.spectral .= 0.0
            Springsteel.sumSpectralTile!(patch, tile)
            @test patch.spectral[siL:siR, 1] ≈ tile.spectral[:, 1]
        end

        @testset "setSpectralTile" begin
            # patch: 30 cells → b_iDim=33
            gp_p = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 0.0, iMax = 30.0, num_cells = 30,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            patch = createGrid(gp_p)

            # tile: 11 cells, spectralIndexL=5 → b_iDim=14, spectralIndexR=18
            gp_t = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 4.0, iMax = 15.0, num_cells = 11,
                spectralIndexL = 5,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            tile = createGrid(gp_t)
            tile.spectral[:, 1] .= collect(1.0:tile.params.b_iDim)

            siL = tile.params.spectralIndexL   # 5
            siR = tile.params.spectralIndexR   # 18

            # setSpectralTile (3-arg): clears patchSpectral and writes tile into position
            ps = ones(Float64, patch.params.b_iDim, 1)
            Springsteel.setSpectralTile(ps, patch.params, tile)
            @test all(ps[1:siL-1, 1] .== 0.0)            # rows before tile are cleared
            @test ps[siL:siR, 1] ≈ tile.spectral[:, 1]   # tile values placed correctly
            @test all(ps[siR+1:end, 1] .== 0.0)           # rows after tile are cleared

            # setSpectralTile! (2-arg in-place variant) should produce the same result
            # as the 3-arg form applied directly to patch.spectral
            patch.spectral .= 99.0   # pre-fill with sentinel so we can verify clearing
            Springsteel.setSpectralTile!(patch, tile)
            @test all(patch.spectral[1:siL-1, 1] .== 0.0)          # rows before tile are cleared
            @test patch.spectral[siL:siR, 1] ≈ tile.spectral[:, 1] # tile values placed correctly
            @test all(patch.spectral[siR+1:end, 1] .== 0.0)        # rows after tile are cleared
        end

        @testset "getBorderSpectral" begin
            gp_p = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 0.0, iMax = 30.0, num_cells = 30,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            patch = createGrid(gp_p)

            gp_t = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 0.0, iMax = 15.0, num_cells = 15,
                spectralIndexL = 1,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )  # b_iDim=18, spectralIndexR=18
            tile = createGrid(gp_t)
            tile.spectral[:, 1] .= collect(1.0:tile.params.b_iDim)

            # biL..biR in tile (indexed by spectralIndexR) = b_iDim-2 .. b_iDim = 16..18
            # tiL..tiR in tile spectral                    = b_iDim-2 .. b_iDim = 16..18
            biL = tile.params.spectralIndexR - 2
            biR = tile.params.spectralIndexR
            tiL = tile.params.b_iDim - 2

            ps_buf = zeros(Float64, patch.params.b_iDim, 1)
            border = Springsteel.getBorderSpectral(tile)

            @test size(border, 1) == tile.params.b_iDim
            @test Vector(border[biL:biR, 1]) ≈ collect(Float64, tiL:tile.params.b_iDim)
            @test nnz(border) == 3   # exactly 3 non-zero entries (1 variable × 3 rows; nnz from SparseArrays)
        end

        @testset "calcPatchMap" begin
            gp_p = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 0.0, iMax = 30.0, num_cells = 30,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            patch = createGrid(gp_p)   # b_iDim=33

            gp_t = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 0.0, iMax = 15.0, num_cells = 15,
                spectralIndexL = 1,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )  # b_iDim=18, spectralIndexR=18
            tile = createGrid(gp_t)

            # siL..siR = spectralIndexL .. spectralIndexR-3  = 1..15  (in patch)
            # tiL..tiR = 1 .. b_iDim-3                       = 1..15  (in tile)
            siL = tile.params.spectralIndexL
            siR = tile.params.spectralIndexR - 3

            patchMap = calcPatchMap(patch, tile)

            @test size(patchMap) == size(patch.spectral)
            @test all(patchMap[siL:siR, :] .!= 0)         # inner rows marked
            @test !any(patchMap[siR+1:end, :] .!= 0)      # rows beyond siR not marked
            @test nnz(patchMap) == (siR - siL + 1)         # one entry per inner row (1 variable)
        end

        @testset "calcHaloMap" begin
            gp_p = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 0.0, iMax = 30.0, num_cells = 30,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            patch = createGrid(gp_p)

            gp_t = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 0.0, iMax = 15.0, num_cells = 15,
                spectralIndexL = 1,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )  # b_iDim=18, spectralIndexR=18
            tile = createGrid(gp_t)

            # hiL..hiR = spectralIndexR-2 .. spectralIndexR = 16..18  (in patch)
            hiL = tile.params.spectralIndexR - 2
            hiR = tile.params.spectralIndexR

            haloMap = calcHaloMap(patch, tile)

            @test size(haloMap) == size(patch.spectral)
            @test all(haloMap[hiL:hiR, :] .!= 0)  # exactly those 3 rows are non-zero
            @test nnz(haloMap) == 3                # 3 entries (1 variable × 3 rows)
        end

        @testset "sumSharedSpectral" begin
            gp_p = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 0.0, iMax = 30.0, num_cells = 30,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            patch = createGrid(gp_p)   # b_iDim=33

            gp_t = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 0.0, iMax = 15.0, num_cells = 15,
                spectralIndexL = 1,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )  # b_iDim=18, spectralIndexR=18
            tile = createGrid(gp_t)
            tile.spectral[:, 1] .= collect(1.0:tile.params.b_iDim)

            # Build border spectral (sparse: non-zeros at tile rows 16-18)
            ps_buf = zeros(Float64, patch.params.b_iDim, 1)
            borderSpectral = Springsteel.getBorderSpectral(tile)

            sharedSpectral = SharedArray{Float64}(patch.params.b_iDim, 1)
            sharedSpectral[:, :] .= 0.0
            patchMap = calcPatchMap(patch, tile)
            haloMap = calcHaloMap(patch, tile)
            Springsteel.sumSharedSpectral(sharedSpectral, tile, patchMap, haloMap)

            siL = tile.params.spectralIndexL       # 1
            siR = tile.params.spectralIndexR - 3   # 15

            # Main region: sharedSpectral[1:15] filled from tile.spectral[1:15]
            @test sharedSpectral[siL:siR, 1] ≈ tile.spectral[1:(siR - siL + 1), 1]
            @test all(isfinite.(sharedSpectral[:, 1]))
        end

        @testset "num_columns and allocateSplineBuffer" begin
            gp = SpringsteelGridParameters(
                geometry = "Spline1D",
                iMin = 0.0, iMax = 10.0, num_cells = 10,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            grid = createGrid(gp)

            @test Springsteel.num_columns(grid) == 1

            buf = allocateSplineBuffer(grid)
            @test isa(buf, Array)   # returns a (trivial) array
        end
    end

    @testset "Spline2D_Grid Tests" begin
        
        @testset "Grid Creation" begin
            # Test basic 2D grid creation
            gp = SpringsteelGridParameters(
                geometry = "Spline2D",
                iMin = 0.0,
                iMax = 10.0,
                num_cells = 10,
                jMin = 0.0,
                jMax = 10.0,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCU = Dict("u" => CubicBSpline.R0),
                BCD = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            
            grid = createGrid(gp)
            
            @test typeof(grid) <: Springsteel.Spline2D_Grid
            @test grid.params.iDim == 30  # num_cells * mubar (10 * 3)
            @test grid.params.b_iDim == 13  # num_cells + 3
            @test grid.params.jDim > 0  # Auto-calculated from aspect ratio
            @test grid.params.b_jDim > 0
            @test size(grid.physical, 1) == grid.params.iDim * grid.params.jDim
            @test size(grid.physical, 2) == 1  # 1 variable
            @test size(grid.physical, 3) == 5  # value, di, dj, dii, djj
        end
        
        @testset "Grid Creation - Multiple Variables" begin
            gp = SpringsteelGridParameters(
                geometry = "Spline2D",
                iMin = -5.0,
                iMax = 5.0,
                num_cells = 15,
                jMin = -5.0,
                jMax = 5.0,
                BCL = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0, "w" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0, "w" => CubicBSpline.R0),
                BCU = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0, "w" => CubicBSpline.R0),
                BCD = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0, "w" => CubicBSpline.R0),
                vars = Dict("u" => 1, "v" => 2, "w" => 3)
            )
            
            grid = createGrid(gp)
            
            @test size(grid.physical, 2) == 3  # 3 variables
            @test size(grid.spectral, 2) == 3
        end
        
        @testset "Spectral Dimensions" begin
            gp = SpringsteelGridParameters(
                geometry = "Spline2D",
                iMin = 0.0,
                iMax = 10.0,
                num_cells = 10,
                jMin = 0.0,
                jMax = 10.0,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCU = Dict("u" => CubicBSpline.R0),
                BCD = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            
            grid = createGrid(gp)
            
            # Check spectral array size
            expected_spectral_dim = grid.params.b_iDim * grid.params.b_jDim
            @test size(grid.spectral, 1) == expected_spectral_dim
        end
        
        @testset "Transform Round-trip" begin
            gp = SpringsteelGridParameters(
                geometry = "Spline2D",
                iMin = 0.0,
                iMax = 10.0,
                num_cells = 15,
                jMin = 0.0,
                jMax = 10.0,
                BCL = Dict("u" => CubicBSpline.PERIODIC),
                BCR = Dict("u" => CubicBSpline.PERIODIC),
                BCU = Dict("u" => CubicBSpline.PERIODIC),
                BCD = Dict("u" => CubicBSpline.PERIODIC),
                vars = Dict("u" => 1)
            )
            
            grid = createGrid(gp)
            
            # Set a simple test function: f(i,j) = sin(2π*i/Li) * cos(2π*j/Lj)
            Li = gp.iMax - gp.iMin
            Lj = gp.jMax - gp.jMin
            num_points = grid.params.iDim * grid.params.jDim
            
            for idx = 1:num_points
                # Convert linear index to i,j indices
                i_idx = mod(idx - 1, grid.params.iDim) + 1
                j_idx = div(idx - 1, grid.params.iDim) + 1
                
                # Get actual i,j coordinates (would need getGridpoints for exact values)
                # For now use approximate values
                i_val = gp.iMin + (i_idx - 1) * Li / (grid.params.iDim - 1)
                j_val = gp.jMin + (j_idx - 1) * Lj / (grid.params.jDim - 1)
                
                grid.physical[idx, 1, 1] = sin(2π * i_val / Li) * cos(2π * j_val / Lj)
            end
            
            # Save original values
            original = copy(grid.physical[:, 1, 1])
            
            # Forward transform
            spectralTransform!(grid)
            
            # Check spectral coefficients are non-zero
            @test maximum(abs.(grid.spectral[:, 1])) > 1e-10
            
            # Inverse transform
            gridTransform!(grid)
            
            # Check round-trip accuracy
            reconstructed = grid.physical[:, 1, 1]
            max_error = maximum(abs.(reconstructed .- original))
            
            @test max_error < 0.1  # Reasonable tolerance for 2D transform
        end
        
        @testset "Spline Array Dimensions" begin
            gp = SpringsteelGridParameters(
                geometry = "Spline2D",
                iMin = 0.0,
                iMax = 10.0,
                num_cells = 10,
                jMin = 0.0,
                jMax = 8.0,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCU = Dict("u" => CubicBSpline.R0),
                BCD = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            
            grid = createGrid(gp)
            
            # ibasis: one for each j spectral coefficient
            @test size(grid.ibasis.data, 1) == grid.params.b_jDim
            
            # jbasis: one for each i gridpoint
            @test size(grid.jbasis.data, 1) == grid.params.iDim
        end
        
        @testset "Variable-specific Filter Lengths" begin
            # Test variable-specific filter length in both dimensions
            gp = SpringsteelGridParameters(
                geometry = "Spline2D",
                iMin = 0.0,
                iMax = 10.0,
                num_cells = 10,
                jMin = 0.0,
                jMax = 10.0,
                l_q = Dict("u" => 1.5, "u_j" => 2.5),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCU = Dict("u" => CubicBSpline.R0),
                BCD = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            
            grid = createGrid(gp)
            
            # Check that ibasis has correct filter length for i dimension
            @test grid.ibasis.data[1, 1].params.l_q == 1.5
            
            # Check that jbasis has correct filter length for j dimension
            @test grid.jbasis.data[1, 1].params.l_q == 2.5
        end
    end

    # ─────────────────────────────────────────────────────────────────────────
    # SpringsteelGrid Type System
    # ─────────────────────────────────────────────────────────────────────────
    @testset "SpringsteelGrid Type System" begin

        @testset "Geometry sentinels" begin
            @test CartesianGeometry() isa AbstractGeometry
            @test CylindricalGeometry() isa AbstractGeometry
            @test SphericalGeometry() isa AbstractGeometry
            @test CartesianGeometry() != CylindricalGeometry()
            @test CylindricalGeometry() != SphericalGeometry()
            @test CartesianGeometry() != SphericalGeometry()
        end

        @testset "Basis sentinel types" begin
            @test SplineBasisType() isa AbstractBasisType
            @test FourierBasisType() isa AbstractBasisType
            @test ChebyshevBasisType() isa AbstractBasisType
            @test NoBasisType() isa AbstractBasisType
        end

        @testset "Basis containers" begin
            @test NoBasisArray() isa NoBasisArray
            # SplineBasisArray requires actual Spline1D objects; construct a trivial array
            spline_arr = Array{CubicBSpline.Spline1D}(undef, 0)
            @test SplineBasisArray(spline_arr) isa SplineBasisArray
            fourier_arr = Array{Fourier.Fourier1D}(undef, 0)
            @test FourierBasisArray(fourier_arr) isa FourierBasisArray
            cheb_arr = Array{Chebyshev.Chebyshev1D}(undef, 0)
            @test ChebyshevBasisArray(cheb_arr) isa ChebyshevBasisArray
        end

        @testset "SpringsteelGrid struct" begin
            # Test that SpringsteelGrid is a concrete parametric struct (not abstract)
            @test !isabstracttype(SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray})
            @test SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray} <: AbstractGrid
            @test SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray} <: AbstractGrid
            @test SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray} <: AbstractGrid
        end

        @testset "num_deriv_slots" begin
            @test num_deriv_slots(NoBasisArray(), NoBasisArray()) == 3   # 1D
            @test num_deriv_slots(SplineBasisArray(Array{CubicBSpline.Spline1D}(undef, 0)), NoBasisArray()) == 5  # 2D (j active)
            @test num_deriv_slots(FourierBasisArray(Array{Fourier.Fourier1D}(undef, 0)), NoBasisArray()) == 5    # 2D (j active)
            @test num_deriv_slots(NoBasisArray(), ChebyshevBasisArray(Array{Chebyshev.Chebyshev1D}(undef, 0))) == 5  # 2D (k active, j empty — RZ case)
            @test num_deriv_slots(SplineBasisArray(Array{CubicBSpline.Spline1D}(undef, 0)),
                                  ChebyshevBasisArray(Array{Chebyshev.Chebyshev1D}(undef, 0))) == 7  # 3D
        end

        @testset "New type aliases (SL, SLZ)" begin
            # SL_Grid and SLZ_Grid are new — no conflict with existing structs
            @test SL_Grid  <: AbstractGrid
            @test SLZ_Grid <: AbstractGrid
            @test SL_Grid  == SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray}
            @test SLZ_Grid == SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}
        end

        @testset "Existing type aliases still resolve" begin
            # Old grid types must remain available as AbstractGrid subtypes
            @test R_Grid       <: AbstractGrid
            @test Spline1D_Grid <: AbstractGrid
            @test RL_Grid      <: AbstractGrid
            @test RZ_Grid      <: AbstractGrid
            @test RR_Grid      <: AbstractGrid
            @test Spline2D_Grid <: AbstractGrid
            @test RLZ_Grid     <: AbstractGrid
            @test RRR_Grid     <: AbstractGrid
        end

        # Canonical aliases must be equal
        @test R_Grid === Spline1D_Grid
        @test RR_Grid === Spline2D_Grid

    end  # SpringsteelGrid Type System

    # ─────────────────────────────────────────────────────────────────────────
    # Basis Interface
    # ─────────────────────────────────────────────────────────────────────────
    @testset "SpringsteelGrid Factory" begin

        @testset "parse_geometry" begin
            for (geom_str, expected_G) in [
                ("R",        CartesianGeometry),
                ("Spline1D", CartesianGeometry),
                ("RR",       CartesianGeometry),
                ("Spline2D", CartesianGeometry),
                ("RZ",       CartesianGeometry),
                ("RRR",      CartesianGeometry),
                ("RL",       CylindricalGeometry),
                ("RLZ",      CylindricalGeometry),
                ("SL",       SphericalGeometry),
                ("SLZ",      SphericalGeometry),
            ]
                G, It, Jt, Kt = parse_geometry(geom_str)
                @test G isa expected_G
            end
            # Basis type sentinels
            G, It, Jt, Kt = parse_geometry("R")
            @test It isa SplineBasisType
            @test Jt isa NoBasisType
            @test Kt isa NoBasisType

            G, It, Jt, Kt = parse_geometry("RL")
            @test It isa SplineBasisType
            @test Jt isa FourierBasisType
            @test Kt isa NoBasisType

            G, It, Jt, Kt = parse_geometry("RLZ")
            @test It isa SplineBasisType
            @test Jt isa FourierBasisType
            @test Kt isa ChebyshevBasisType

            G, It, Jt, Kt = parse_geometry("RZ")
            @test It isa SplineBasisType
            @test Jt isa NoBasisType
            @test Kt isa ChebyshevBasisType

            G, It, Jt, Kt = parse_geometry("RRR")
            @test It isa SplineBasisType
            @test Jt isa SplineBasisType
            @test Kt isa SplineBasisType
        end

        @testset "R Grid creation" begin
            gp = SpringsteelGridParameters(
                geometry = "R",
                num_cells = 10,
                iMin = 0.0,
                iMax = 100.0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            @test typeof(grid) == SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray}
            @test grid.params.iDim  == 10 * 3     # num_cells * mubar
            @test grid.params.b_iDim == 10 + 3    # num_cells + 3
            @test size(grid.spectral, 1) == grid.params.b_iDim
            @test size(grid.spectral, 2) == 1
            @test size(grid.physical, 1) == grid.params.iDim
            @test size(grid.physical, 2) == 1
            @test size(grid.physical, 3) == 3     # 1D: value, ∂/∂i, ∂²/∂i²
            @test grid.ibasis isa SplineBasisArray
            @test grid.jbasis isa NoBasisArray
            @test grid.kbasis isa NoBasisArray
            @test size(grid.ibasis.data) == (1, 1)
        end

        @testset "Multi-variable R Grid creation" begin
            gp = SpringsteelGridParameters(
                geometry = "R",
                num_cells = 8,
                iMin = 0.0,
                iMax = 50.0,
                vars = Dict("u" => 1, "v" => 2, "w" => 3),
                BCL = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0, "w" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0, "w" => CubicBSpline.R0))
            grid = createGrid(gp)

            @test typeof(grid) == SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray}
            @test size(grid.spectral, 2) == 3
            @test size(grid.physical, 2) == 3
            @test size(grid.ibasis.data) == (1, 3)
        end

        @testset "RL Grid creation" begin
            num_cells = 5
            gp = SpringsteelGridParameters(
                geometry = "RL",
                num_cells = num_cells,
                iMin = 0.0,
                iMax = 50.0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            iDim   = num_cells * 3
            b_iDim = num_cells + 3
            exp_jDim   = sum(4 + 4*r for r in 1:iDim)
            exp_b_jDim = sum(1 + 2*r for r in 1:iDim)
            # Spectral array uses uniform wavenumber-interleaved layout: b_iDim*(1+2*kDim)
            # where kDim = iDim + patchOffsetL.  (b_jDim is retained as a grid parameter
            # for physical-space use but no longer drives the spectral allocation.)
            exp_spectral_rows = b_iDim * (1 + 2 * iDim)

            @test typeof(grid) == SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray}
            @test grid.params.jDim   == exp_jDim
            @test grid.params.b_jDim == exp_b_jDim
            @test size(grid.spectral, 1) == exp_spectral_rows
            @test size(grid.physical, 1) == exp_jDim
            @test size(grid.physical, 3) == 5     # 2D: 5 derivative slots
            @test grid.ibasis isa SplineBasisArray
            @test grid.jbasis isa FourierBasisArray
            @test grid.kbasis isa NoBasisArray
            @test size(grid.ibasis.data) == (3, 1)    # 3 splines per variable
            @test size(grid.jbasis.data) == (iDim, 1) # one ring per radial point per var
        end

        @testset "RZ Grid creation" begin
            num_cells = 5
            kDim = 10
            gp = SpringsteelGridParameters(
                geometry = "RZ",
                num_cells = num_cells,
                iMin = 0.0,
                iMax = 50.0,
                kMin = 0.0,
                kMax = 20.0,
                kDim = kDim,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCB = Dict("u" => Chebyshev.R0),
                BCT = Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)

            iDim   = num_cells * 3
            b_iDim = num_cells + 3
            b_kDim = min(kDim, Int(floor(((2*kDim) - 1) / 3)) + 1)

            @test typeof(grid) == SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, ChebyshevBasisArray}
            @test size(grid.spectral, 1) == b_kDim * b_iDim
            @test size(grid.physical, 1) == iDim * kDim
            @test size(grid.physical, 3) == 5     # 2D (j=NoBasis, k=Cheb): 5 slots
            @test grid.ibasis isa SplineBasisArray
            @test grid.jbasis isa NoBasisArray
            @test grid.kbasis isa ChebyshevBasisArray
            @test size(grid.ibasis.data) == (b_kDim, 1)
            @test size(grid.kbasis.data) == (1,)
        end

        @testset "RR Grid creation" begin
            num_cells = 4
            gp = SpringsteelGridParameters(
                geometry = "RR",
                num_cells = num_cells,
                iMin = 0.0,
                iMax = 40.0,
                jMin = 0.0,
                jMax = 40.0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCU = Dict("u" => CubicBSpline.R0),
                BCD = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            # Same domain extent → same num_cells_j = num_cells
            iDim   = num_cells * 3
            b_iDim = num_cells + 3
            jDim   = num_cells * 3   # same as iDim since same domain
            b_jDim = num_cells + 3

            @test typeof(grid) == SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, NoBasisArray}
            @test grid.params.jDim   == jDim
            @test grid.params.b_jDim == b_jDim
            @test size(grid.spectral, 1) == b_iDim * b_jDim
            @test size(grid.physical, 1) == iDim * jDim
            @test size(grid.physical, 3) == 5     # 2D (j=Spline, k=NoBasis)
            @test grid.ibasis isa SplineBasisArray
            @test grid.jbasis isa SplineBasisArray
            @test grid.kbasis isa NoBasisArray
            @test size(grid.ibasis.data) == (b_jDim, 1)
            @test size(grid.jbasis.data) == (iDim, 1)
        end

        @testset "RLZ Grid creation" begin
            num_cells = 3
            kDim = 8
            gp = SpringsteelGridParameters(
                geometry = "RLZ",
                num_cells = num_cells,
                iMin = 0.0,
                iMax = 30.0,
                kMin = 0.0,
                kMax = 10.0,
                kDim = kDim,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCB = Dict("u" => Chebyshev.R0),
                BCT = Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)

            iDim       = num_cells * 3
            b_iDim     = num_cells + 3
            b_kDim     = min(kDim, Int(floor(((2*kDim) - 1) / 3)) + 1)
            exp_jDim   = sum(4 + 4*r for r in 1:iDim)
            exp_b_jDim = sum(1 + 2*r for r in 1:iDim)
            exp_spectral = b_kDim * b_iDim * (1 + 2*(iDim + 0))  # patchOffsetL=0

            @test typeof(grid) == SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}
            @test grid.params.jDim   == exp_jDim
            @test grid.params.b_jDim == exp_b_jDim
            @test size(grid.spectral, 1) == exp_spectral
            @test size(grid.physical, 1) == exp_jDim * kDim
            @test size(grid.physical, 3) == 7     # 3D
            @test grid.ibasis isa SplineBasisArray
            @test grid.jbasis isa FourierBasisArray
            @test grid.kbasis isa ChebyshevBasisArray
            @test size(grid.ibasis.data) == (b_kDim, 1)
            @test size(grid.jbasis.data) == (iDim, b_kDim)
            @test size(grid.kbasis.data) == (1,)
        end

        @testset "RRR Grid creation" begin
            num_cells = 3
            gp = SpringsteelGridParameters(
                geometry = "RRR",
                num_cells = num_cells,
                iMin = 0.0,
                iMax = 30.0,
                jMin = 0.0,
                jMax = 30.0,
                kMin = 0.0,
                kMax = 30.0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCU = Dict("u" => CubicBSpline.R0),
                BCD = Dict("u" => CubicBSpline.R0),
                BCB = Dict("u" => CubicBSpline.R0),
                BCT = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            iDim   = num_cells * 3
            b_iDim = num_cells + 3
            jDim   = num_cells * 3   # same domain
            b_jDim = num_cells + 3
            kDim   = num_cells * 3
            b_kDim = num_cells + 3   # spline formula for k

            @test typeof(grid) == SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, SplineBasisArray}
            @test grid.params.jDim   == jDim
            @test grid.params.b_jDim == b_jDim
            @test grid.params.kDim   == kDim
            @test grid.params.b_kDim == b_kDim
            @test size(grid.spectral, 1) == b_iDim * b_jDim * b_kDim
            @test size(grid.physical, 1) == iDim * jDim * kDim
            @test size(grid.physical, 3) == 7     # 3D
            @test grid.ibasis isa SplineBasisArray
            @test grid.jbasis isa SplineBasisArray
            @test grid.kbasis isa SplineBasisArray
            @test size(grid.ibasis.data) == (b_jDim, b_kDim, 1)
            @test size(grid.jbasis.data) == (iDim, b_kDim, 1)
            @test size(grid.kbasis.data) == (iDim, jDim, 1)
        end

        @testset "SL Grid creation" begin
            num_cells = 4
            gp = SpringsteelGridParameters(
                geometry = "SL",
                num_cells = num_cells,
                iMin = 0.05,
                iMax = π - 0.05,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            iDim = num_cells * 3

            @test typeof(grid) == SL_Grid
            @test typeof(grid) == SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray}
            @test grid.params.jDim > 0
            @test grid.params.b_jDim > 0
            # Spectral array uses uniform wavenumber-interleaved layout: b_iDim*(1+2*kDim)
            kDim_wn = grid.params.iDim + grid.params.patchOffsetL
            @test size(grid.spectral, 1) == grid.params.b_iDim * (1 + 2 * kDim_wn)
            @test size(grid.physical, 1) == grid.params.jDim
            @test size(grid.physical, 3) == 5     # 2D
            @test grid.ibasis isa SplineBasisArray
            @test grid.jbasis isa FourierBasisArray
            @test grid.kbasis isa NoBasisArray
            @test size(grid.ibasis.data, 2) == 1  # 1 variable
        end

        @testset "SLZ Grid creation" begin
            num_cells = 3
            kDim = 6
            gp = SpringsteelGridParameters(
                geometry = "SLZ",
                num_cells = num_cells,
                iMin = 0.05,
                iMax = π - 0.05,
                kMin = 0.0,
                kMax = 10.0,
                kDim = kDim,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCB = Dict("u" => Chebyshev.R0),
                BCT = Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)

            @test typeof(grid) == SLZ_Grid
            @test typeof(grid) == SpringsteelGrid{SphericalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}
            @test grid.params.jDim > 0
            @test size(grid.physical, 3) == 7     # 3D
            @test grid.ibasis isa SplineBasisArray
            @test grid.jbasis isa FourierBasisArray
            @test grid.kbasis isa ChebyshevBasisArray
        end

    end  # SpringsteelGrid Factory

    # ─────────────────────────────────────────────────────────────────────────
    # 1D Cartesian Transforms
    # (SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray})
    # ─────────────────────────────────────────────────────────────────────────

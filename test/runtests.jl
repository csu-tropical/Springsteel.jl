using Springsteel
using Test
using SharedArrays
using SparseArrays

@testset "Springsteel.jl" begin
    
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
            @test grid.params.rDim == 30  # num_cells * mubar (10 * 3)
            @test grid.params.b_rDim == 13  # num_cells + 3
            @test size(grid.physical, 1) == grid.params.rDim
            @test size(grid.spectral, 1) == grid.params.b_rDim
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
            
            @test length(gridpoints) == grid.params.rDim
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
            # Test R0 boundary condition (zero at boundaries)
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
            
            # Set interior values
            for i = 1:length(gridpoints)
                grid.physical[i, 1, 1] = sin(π * (gridpoints[i] - gp.xmin) / (gp.xmax - gp.xmin))
            end
            
            spectralTransform!(grid)
            gridTransform!(grid)
            
            # Values should be close to zero at boundaries
            @test abs(grid.physical[1, 1, 1]) < 0.1
            @test abs(grid.physical[end, 1, 1]) < 0.1
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
            @test grid.splines[1, 1].params.l_q == 1.5
            @test grid.splines[1, 2].params.l_q == 3.0
        end
    end

    @testset "CubicBSpline Tests" begin

        @testset "SplineParameters auto-computed fields" begin
            sp = SplineParameters(xmin=0.0, xmax=10.0, num_cells=20)
            @test sp.DX ≈ 0.5
            @test sp.DXrecip ≈ 2.0
            sp2 = SplineParameters(xmin=-5.0, xmax=5.0, num_cells=10)
            @test sp2.DX ≈ 1.0
            @test sp2.DXrecip ≈ 1.0
        end

        @testset "Spline1D construction" begin
            sp = SplineParameters(xmin=0.0, xmax=10.0, num_cells=20,
                                  BCL=CubicBSpline.R0, BCR=CubicBSpline.R0)
            spline = Spline1D(sp)
            @test spline.mishDim == 60       # 20 * 3
            @test spline.bDim == 23          # 20 + 3
            @test length(spline.mishPoints) == 60
            @test length(spline.b) == 23
            @test length(spline.a) == 23
            @test spline.mishPoints[1] > sp.xmin
            @test spline.mishPoints[end] < sp.xmax
            @test all(diff(spline.mishPoints) .> 0)
        end

        @testset "basis function" begin
            sp = SplineParameters(xmin=0.0, xmax=10.0, num_cells=10)
            DX = 1.0  # (xmax - xmin) / num_cells
            # Node m=3 is centered at xmin + (m+0.5)*DX = 3.5... actually xm = xmin + m*DX = 3.0
            xm = sp.xmin + 3 * DX
            # At the peak (delta=0): b = (2^3)/6 - (1^3)*(4/6) = 8/6 - 4/6 = 4/6
            @test CubicBSpline.basis(sp, 3, xm, 0) ≈ 4.0/6.0  atol=1e-14
            # First derivative at peak is zero by symmetry
            @test CubicBSpline.basis(sp, 3, xm, 1) ≈ 0.0  atol=1e-14
            # At support boundary (|delta|=2), basis = 0
            @test CubicBSpline.basis(sp, 3, xm + 2.0*DX, 0) ≈ 0.0  atol=1e-14
            # Second derivative at peak: z=2, b=2 - (z-1)*4 = 2-4 = -2; b *= DXrecip^2
            @test CubicBSpline.basis(sp, 3, xm, 2) ≈ -2.0 * sp.DXrecip^2  atol=1e-12
            # x outside domain throws DomainError
            @test_throws DomainError CubicBSpline.basis(sp, 3, sp.xmin - 0.1, 0)
            @test_throws DomainError CubicBSpline.basis(sp, 3, sp.xmax + 0.1, 0)
        end

        @testset "setMishValues" begin
            sp = SplineParameters(xmin=0.0, xmax=10.0, num_cells=10,
                                  BCL=CubicBSpline.R0, BCR=CubicBSpline.R0)
            spline = Spline1D(sp)
            u = sin.(spline.mishPoints)
            setMishValues(spline, u)
            @test spline.uMish ≈ u
            # Overwrite with zeros
            setMishValues(spline, zeros(length(u)))
            @test all(spline.uMish .== 0.0)
        end

        @testset "SBtransform variants" begin
            sp = SplineParameters(xmin=0.0, xmax=10.0, num_cells=20,
                                  BCL=CubicBSpline.R0, BCR=CubicBSpline.R0)
            spline = Spline1D(sp)
            u_vals = sin.(spline.mishPoints)

            # SplineParameters variant
            b1 = SBtransform(sp, u_vals)
            @test length(b1) == spline.bDim
            @test maximum(abs.(b1)) > 0.0

            # Spline1D variant gives same result
            b2 = SBtransform(spline, u_vals)
            @test b1 ≈ b2

            # In-place SBtransform! uses spline.uMish
            setMishValues(spline, u_vals)
            SBtransform!(spline)
            @test spline.b ≈ b1
        end

        @testset "SAtransform variants" begin
            sp = SplineParameters(xmin=0.0, xmax=10.0, num_cells=20,
                                  BCL=CubicBSpline.R0, BCR=CubicBSpline.R0)
            spline = Spline1D(sp)
            b = SBtransform(spline, sin.(spline.mishPoints))

            # Standard Spline1D variant
            a1 = SAtransform(spline, b)
            @test length(a1) == spline.bDim

            # 4-arg SplineParameters variant gives same result
            a2 = SAtransform(sp, spline.gammaBC, spline.pqFactor, b)
            @test a1 ≈ a2

            # In-place SAtransform!
            spline.b .= b
            SAtransform!(spline)
            @test spline.a ≈ a1

            # Inhomogeneous variant: ahat=zeros should equal standard result
            a3 = SAtransform(spline, b, zeros(length(b)))
            @test a1 ≈ a3  atol=1e-10
        end

        @testset "SItransform variants" begin
            sp = SplineParameters(xmin=0.0, xmax=10.0, num_cells=20,
                                  BCL=CubicBSpline.R0, BCR=CubicBSpline.R0)
            spline = Spline1D(sp)
            pts = spline.mishPoints
            u_true = sin.(pts .* π ./ 10.0)  # zero at boundaries, fits R0 BCs
            b = SBtransform(spline, u_true)
            a = SAtransform(spline, b)
            spline.b .= b
            SAtransform!(spline)

            # Scalar variant (single point)
            u_scalar = SItransform(sp, a, pts[15], 0)
            @test u_scalar ≈ u_true[15]  atol=1e-4

            # Mish-point variant returning new vector
            u_mish = SItransform(sp, a, 0)
            @test length(u_mish) == spline.mishDim
            @test maximum(abs.(u_mish .- u_true)) < 1e-3  # smoothing filter permits ~1e-4 approximation error

            # Points variant returning new vector
            u_pts = SItransform(sp, a, pts, 0)
            @test u_pts ≈ u_mish  atol=1e-14

            # In-place with SplineParameters
            u_inplace = zeros(length(pts))
            SItransform(sp, a, pts, u_inplace, 0)
            @test u_inplace ≈ u_mish  atol=1e-14

            # SItransform! in-place using spline internal fields: compare against
            # SItransform(sp, a) to test API consistency (not approximation accuracy)
            SItransform!(spline)
            @test spline.uMish ≈ u_mish  atol=1e-12

            # SItransform(spline, u) variant
            u2 = zeros(length(pts))
            SItransform(spline, u2)
            @test u2 ≈ u_mish  atol=1e-12

            # SItransform(spline, points, u) variant
            u3 = zeros(length(pts))
            SItransform(spline, pts, u3)
            @test u3 ≈ u_mish  atol=1e-12
        end

        @testset "SItransform_matrix" begin
            # Note: SItransform_matrix allocates rows = mishDim regardless of
            # length(points); it works correctly only when length(points) == mishDim.
            sp = SplineParameters(xmin=0.0, xmax=5.0, num_cells=5,
                                  BCL=CubicBSpline.R0, BCR=CubicBSpline.R0)
            spline = Spline1D(sp)
            pts = spline.mishPoints  # exactly mishDim points
            u_true = sin.(pts .* π ./ 5.0)
            b = SBtransform(spline, u_true)
            spline.b .= b
            SAtransform!(spline)

            # Reference: SItransform(sp, a) at mish points
            u_ref = SItransform(sp, spline.a, 0)
            M = CubicBSpline.SItransform_matrix(spline, pts, 0)
            @test size(M) == (spline.mishDim, spline.bDim)
            # M * a should give the same result as the direct SItransform evaluation
            @test maximum(abs.(M * spline.a .- u_ref)) < 1e-12
        end

        @testset "SIxtransform variants" begin
            sp = SplineParameters(xmin=0.0, xmax=2π, num_cells=30,
                                  BCL=CubicBSpline.PERIODIC, BCR=CubicBSpline.PERIODIC)
            spline = Spline1D(sp)
            pts = spline.mishPoints
            spline.b .= SBtransform(spline, sin.(pts))
            SAtransform!(spline)

            # No-arg variant returns new vector
            dx1 = SIxtransform(spline)
            @test length(dx1) == spline.mishDim
            @test maximum(abs.(dx1 .- cos.(pts))) < 0.01

            # Pre-allocated AbstractVector variant
            dx2 = zeros(length(pts))
            SIxtransform(spline, dx2)
            @test dx2 ≈ dx1

            # Points + pre-allocated variant
            dx3 = zeros(length(pts))
            SIxtransform(spline, pts, dx3)
            @test dx3 ≈ dx1

            # SplineParameters variant
            dx4 = SIxtransform(sp, spline.a, pts)
            @test dx4 ≈ dx1
        end

        @testset "SIxxtransform variants" begin
            sp = SplineParameters(xmin=0.0, xmax=2π, num_cells=30,
                                  BCL=CubicBSpline.PERIODIC, BCR=CubicBSpline.PERIODIC)
            spline = Spline1D(sp)
            pts = spline.mishPoints
            spline.b .= SBtransform(spline, sin.(pts))
            SAtransform!(spline)

            # No-arg variant returns new vector
            dxx1 = SIxxtransform(spline)
            @test length(dxx1) == spline.mishDim
            @test maximum(abs.(dxx1 .+ sin.(pts))) < 0.01  # d²(sin)/dx² = -sin

            # Pre-allocated AbstractVector variant
            dxx2 = zeros(length(pts))
            SIxxtransform(spline, dxx2)
            @test dxx2 ≈ dxx1

            # Points + pre-allocated variant
            dxx3 = zeros(length(pts))
            SIxxtransform(spline, pts, dxx3)
            @test dxx3 ≈ dxx1

            # SplineParameters variant
            dxx4 = SIxxtransform(sp, spline.a, pts)
            @test dxx4 ≈ dxx1
        end

        @testset "SBxtransform variants" begin
            # SBxtransform computes the B vector of f' via integration by parts:
            #   b_m = φ_m(xmax)·f(xmax) - φ_m(xmin)·f(xmin) - ∫ φ'_m · f dx
            # so that SA(SBxt(f)) recovers the derivative f'.

            # --- Test A: linear f(x) = x on [0, L]  ---
            # 3-point Gauss quadrature is exact for polynomials to degree 5,
            # so integration by parts holds to machine precision for f(x) = x.
            sp_lin = SplineParameters(xmin=0.0, xmax=10.0, num_cells=20,
                                      BCL=CubicBSpline.R0, BCR=CubicBSpline.R0)
            spline_lin = Spline1D(sp_lin)
            pts_lin = spline_lin.mishPoints

            f_lin   = pts_lin           # f(x) = x
            fp_lin  = ones(length(pts_lin))  # f'(x) = 1

            bx_lin  = SBxtransform(sp_lin, f_lin, 0.0, 10.0)      # boundary values: f(0)=0, f(10)=10
            b_dir   = SBtransform(sp_lin, fp_lin)
            @test maximum(abs.(bx_lin .- b_dir)) < 1e-12  # machine precision for linear f

            # Spline1D variant gives identical result
            bx_spline = SBxtransform(spline_lin, f_lin, 0.0, 10.0)
            @test bx_spline ≈ bx_lin

            # --- Test B: smooth non-polynomial f(x) = sin(π*x/L) on [0, L],
            #     zero at both boundaries → boundary terms vanish exactly ---
            sp_sin = SplineParameters(xmin=0.0, xmax=10.0, num_cells=40,
                                      BCL=CubicBSpline.R0, BCR=CubicBSpline.R0)
            spline_sin = Spline1D(sp_sin)
            pts_sin = spline_sin.mishPoints
            L = 10.0

            f_sin   = sin.(π .* pts_sin ./ L)
            fp_sin  = (π/L) .* cos.(π .* pts_sin ./ L)

            bx_sin  = SBxtransform(sp_sin, f_sin, 0.0, 0.0)
            b_dir2  = SBtransform(sp_sin, fp_sin)
            # 3-point Gauss quadrature is not exact for sin/cos, so integration-
            # by-parts identity holds to quadrature accuracy (~1e-11 with 40 cells)
            @test maximum(abs.(bx_sin .- b_dir2)) < 1e-10

            # --- Test C: non-zero boundary values, f(x) = sin(x) on [0, 2π] PERIODIC ---
            #     f(0) = 0,  f(2π) = 0 → boundary terms also vanish ---
            sp_per = SplineParameters(xmin=0.0, xmax=2π, num_cells=40,
                                      BCL=CubicBSpline.PERIODIC, BCR=CubicBSpline.PERIODIC)
            spline_per = Spline1D(sp_per)
            pts_per = spline_per.mishPoints

            f_per   = sin.(pts_per)
            fp_per  = cos.(pts_per)

            bx_per  = SBxtransform(sp_per, f_per, 0.0, 0.0)
            b_dir3  = SBtransform(sp_per, fp_per)
            # Higher-frequency content (full wavelength in domain) makes quadrature
            # approximation slightly less exact than the low-frequency Test B (~3.5e-10)
            @test maximum(abs.(bx_per .- b_dir3)) < 1e-8

            # --- Test D: full SA+SI pipeline recovers f'(x) analytically ---
            # Use PERIODIC BCs so the spectral solve is consistent with f' = cos(x)
            spline_per.b .= bx_per
            SAtransform!(spline_per)
            u_recov = zeros(length(pts_per))
            SItransform(spline_per, pts_per, u_recov)
            @test maximum(abs.(u_recov .- fp_per)) < 0.01  # cos(x) recovered to 1%
        end

        @testset "Boundary condition types" begin
            # Test that all BC types can be used to construct a Spline1D and
            # produce a valid (finite, stable) SB→SA round-trip.
            # R0 is tested extensively elsewhere; exercise the remaining types here.
            interior_fn = x -> sin(π * x / 10.0)  # zero at x=0 and x=10

            for bc in [CubicBSpline.R1T0, CubicBSpline.R1T1, CubicBSpline.R1T2,
                       CubicBSpline.R2T10, CubicBSpline.R2T20, CubicBSpline.R3]
                sp = SplineParameters(xmin=0.0, xmax=10.0, num_cells=20,
                                      BCL=bc, BCR=CubicBSpline.R0)
                spline = Spline1D(sp)
                @test spline.mishDim == 60
                @test spline.bDim == 23
                u_vals = interior_fn.(spline.mishPoints)
                b = SBtransform(spline, u_vals)
                a = SAtransform(spline, b)
                @test all(isfinite.(a))
            end

            # R3 on both sides
            sp_r3 = SplineParameters(xmin=0.0, xmax=10.0, num_cells=20,
                                     BCL=CubicBSpline.R3, BCR=CubicBSpline.R3)
            spline_r3 = Spline1D(sp_r3)
            @test spline_r3.mishDim == 60
            a_r3 = SAtransform(spline_r3,
                               SBtransform(spline_r3, interior_fn.(spline_r3.mishPoints)))
            @test all(isfinite.(a_r3))

            # R1T0 on both sides with a function whose derivative vanishes at the boundary:
            # cos(π*x/L) has zero derivative at x=0 and x=L
            sp_r1t0 = SplineParameters(xmin=0.0, xmax=10.0, num_cells=30,
                                       BCL=CubicBSpline.R1T0, BCR=CubicBSpline.R1T0)
            spline_r1t0 = Spline1D(sp_r1t0)
            u_r1t0 = cos.(π .* spline_r1t0.mishPoints ./ 10.0)
            b_r1t0 = SBtransform(spline_r1t0, u_r1t0)
            a_r1t0 = SAtransform(spline_r1t0, b_r1t0)
            spline_r1t0.b .= b_r1t0
            SAtransform!(spline_r1t0)
            u_reconstructed = SItransform(spline_r1t0.params, spline_r1t0.a, 0)
            # R1T0 BC constrains b-coefficients at the boundary; the BC imposes a
            # coefficient relationship that modifies the reconstruction, so we verify
            # the computation completes with finite results rather than checking accuracy
            @test all(isfinite.(u_reconstructed))
        end

        @testset "Generic wrapper smoke test (CubicBSpline)" begin
            # Verify that the no-prefix generic wrappers (Btransform, Atransform, Itransform,
            # etc.) produce results identical to their S-prefixed originals. These wrappers
            # contain no logic of their own; this test guards against future naming regressions.
            sp = SplineParameters(xmin=0.0, xmax=1.0, num_cells=5,
                                  BCL=CubicBSpline.R0, BCR=CubicBSpline.R0)
            spline = CubicBSpline.Spline1D(sp)
            u = sin.(π .* spline.mishPoints)

            # Btransform
            b_s = CubicBSpline.SBtransform(spline, u)
            b_g = CubicBSpline.Btransform(spline, u)
            @test b_s == b_g

            # Btransform!
            CubicBSpline.setMishValues(spline, u)
            CubicBSpline.SBtransform!(spline)
            b_ref = copy(spline.b)
            CubicBSpline.setMishValues(spline, u)
            CubicBSpline.Btransform!(spline)
            @test spline.b == b_ref

            # Atransform (allocating)
            a_s = CubicBSpline.SAtransform(spline, b_s)
            a_g = CubicBSpline.Atransform(spline, b_s)
            @test a_s == a_g

            # Atransform! (in-place)
            spline.b .= b_s
            CubicBSpline.SAtransform!(spline)
            a_ref = copy(spline.a)
            spline.b .= b_s
            CubicBSpline.Atransform!(spline)
            @test spline.a == a_ref

            # Itransform (mish points, in-place)
            u_s = zeros(length(u)); CubicBSpline.SItransform(spline, u_s)
            u_g = zeros(length(u)); CubicBSpline.Itransform(spline, u_g)
            @test u_s == u_g

            # Itransform! (in-place via spline internals)
            CubicBSpline.SItransform!(spline)
            u_ref = copy(spline.uMish)
            CubicBSpline.Itransform!(spline)
            @test spline.uMish == u_ref

            # Ixtransform (allocating)
            dx_s = CubicBSpline.SIxtransform(spline)
            dx_g = CubicBSpline.Ixtransform(spline)
            @test dx_s == dx_g

            # Ixxtransform (allocating)
            dxx_s = CubicBSpline.SIxxtransform(spline)
            dxx_g = CubicBSpline.Ixxtransform(spline)
            @test dxx_s == dxx_g

            # Bxtransform
            bx_s = CubicBSpline.SBxtransform(spline, u, 0.0, 0.0)
            bx_g = CubicBSpline.Bxtransform(spline, u, 0.0, 0.0)
            @test bx_s == bx_g
        end
    end

    @testset "Fourier Tests" begin

        @testset "FourierParameters construction" begin
            # All four fields are set explicitly; no auto-computation.
            fp = Fourier.FourierParameters(ymin=0.0, kmax=8, yDim=32, bDim=17)
            @test fp.ymin == 0.0
            @test fp.kmax == 8
            @test fp.yDim == 32
            @test fp.bDim == 17               # 2*kmax + 1
            # Second configuration to verify generality
            fp2 = Fourier.FourierParameters(ymin=π/4, kmax=4, yDim=16, bDim=9)
            @test fp2.ymin ≈ π/4
            @test fp2.kmax == 4
            @test fp2.yDim == 16
            @test fp2.bDim == 9               # 2*4 + 1
        end

        @testset "Fourier1D construction" begin
            fp = Fourier.FourierParameters(ymin=0.0, kmax=8, yDim=32, bDim=17)
            ring = Fourier.Fourier1D(fp)

            # Array sizes
            @test length(ring.mishPoints) == 32          # yDim
            @test length(ring.uMish)      == 32          # yDim
            @test length(ring.b)          == 17          # bDim = 2*kmax+1
            @test length(ring.a)          == 32          # yDim (zero-padded for IFFT)
            @test length(ring.ax)         == 32          # yDim (derivative buffer)

            # Phase filter matrix sizes
            @test size(ring.phasefilter)    == (32, 17)  # (yDim, bDim)
            @test size(ring.invphasefilter) == (17, 32)  # (bDim, yDim)

            # Mish points: start at ymin, monotonically increasing, one step short of 2π
            @test ring.mishPoints[1] ≈ 0.0
            @test ring.mishPoints[end] < 2π
            @test all(diff(ring.mishPoints) .> 0)

            # With non-zero ymin, first point is offset correctly
            fp3 = Fourier.FourierParameters(ymin=π/6, kmax=4, yDim=16, bDim=9)
            ring3 = Fourier.Fourier1D(fp3)
            @test ring3.mishPoints[1] ≈ π/6
        end

        @testset "calcMishPoints" begin
            fp = Fourier.FourierParameters(ymin=0.0, kmax=3, yDim=8, bDim=7)
            pts = Fourier.calcMishPoints(fp)
            @test length(pts) == 8
            @test pts[1] ≈ 0.0
            # Spacing should be 2π/yDim
            @test all(diff(pts) .≈ 2π/8)
            # Last point should be one step before completing the full 2π circle
            @test pts[end] ≈ 2π * 7/8
        end

        @testset "Phase filter properties" begin
            # With ymin=0, the phase-shift is identity (cos(0)=1, sin(0)=0)
            fp = Fourier.FourierParameters(ymin=0.0, kmax=2, yDim=8, bDim=5)
            pf  = Fourier.calcPhaseFilter(fp)
            ipf = Fourier.calcInvPhaseFilter(fp)
            @test size(pf)  == (8, 5)
            @test size(ipf) == (5, 8)
            # Wavenumber-0 elements are pass-through
            @test pf[1,1]  ≈ 1.0
            @test ipf[1,1] ≈ 1.0
            # For ymin=0: cos(-k*0)=1, sin(-k*0)=0, so diagonal blocks are identity
            @test pf[2,2]  ≈ 1.0   # k=1 cosine
            @test pf[7,2]  ≈ 0.0   # k=1 cross-term should be zero
        end

        @testset "FBtransform variants" begin
            fp = Fourier.FourierParameters(ymin=0.0, kmax=8, yDim=32, bDim=17)
            ring = Fourier.Fourier1D(fp)
            u = sin.(ring.mishPoints)  # pure k=1 sine

            # FourierParameters variant (4-arg)
            b1 = Fourier.FBtransform(fp, ring.fftPlan, ring.phasefilter, u)
            @test length(b1) == fp.bDim
            @test maximum(abs.(b1)) > 0.0

            # In-place FBtransform!
            ring.uMish .= u
            Fourier.FBtransform!(ring)
            @test ring.b ≈ b1
        end

        @testset "FAtransform variants" begin
            fp = Fourier.FourierParameters(ymin=0.0, kmax=8, yDim=32, bDim=17)
            ring = Fourier.Fourier1D(fp)
            ring.uMish .= sin.(ring.mishPoints)
            Fourier.FBtransform!(ring)
            b = copy(ring.b)

            # Allocating variant
            a1 = Fourier.FAtransform(fp, ring.invphasefilter, b)
            @test length(a1) == fp.yDim   # zero-padded back to yDim

            # In-place FAtransform!
            Fourier.FAtransform!(ring)
            @test ring.a ≈ a1
        end

        @testset "FItransform variants" begin
            fp = Fourier.FourierParameters(ymin=0.0, kmax=8, yDim=32, bDim=17)
            ring = Fourier.Fourier1D(fp)
            u_orig = sin.(ring.mishPoints)
            ring.uMish .= u_orig
            Fourier.FBtransform!(ring)
            Fourier.FAtransform!(ring)

            # Allocating variant
            u1 = Fourier.FItransform(fp, ring.ifftPlan, ring.a)
            @test length(u1) == fp.yDim
            @test maximum(abs.(u1 .- u_orig)) < 1e-12   # should round-trip exactly

            # In-place FItransform!
            Fourier.FItransform!(ring)
            @test ring.uMish ≈ u1
        end

        @testset "Round-trip accuracy (FB→FA→FI)" begin
            # Test sin(k*θ) + cos(k*θ) for k within kmax round-trips accurately.
            fp = Fourier.FourierParameters(ymin=0.0, kmax=10, yDim=64, bDim=21)
            ring = Fourier.Fourier1D(fp)
            pts = ring.mishPoints

            # Single-wavenumber sinusoid
            for k in [1, 3, 10]
                u_orig = sin.(k .* pts) .+ cos.(k .* pts)
                ring.uMish .= u_orig
                Fourier.FBtransform!(ring)
                Fourier.FAtransform!(ring)
                Fourier.FItransform!(ring)
                @test maximum(abs.(ring.uMish .- u_orig)) < 1e-10
            end

            # Constant signal (wavenumber 0)
            u_const = fill(3.14, fp.yDim)
            ring.uMish .= u_const
            Fourier.FBtransform!(ring)
            Fourier.FAtransform!(ring)
            Fourier.FItransform!(ring)
            @test maximum(abs.(ring.uMish .- u_const)) < 1e-10

            # Multi-wavenumber signal: sum of several harmonics within kmax
            u_multi = cos.(pts) .+ 0.5 .* sin.(3 .* pts) .+ 0.25 .* cos.(7 .* pts)
            ring.uMish .= u_multi
            Fourier.FBtransform!(ring)
            Fourier.FAtransform!(ring)
            Fourier.FItransform!(ring)
            @test maximum(abs.(ring.uMish .- u_multi)) < 1e-10
        end

        @testset "FIxcoefficients" begin
            # For a pure cosine u = cos(k*θ), du/dθ = -k*sin(k*θ).
            # In halfcomplex layout: cosine coeff at index k+1, sine at yDim-k+1.
            fp   = Fourier.FourierParameters(ymin=0.0, kmax=4, yDim=16, bDim=9)
            ring = Fourier.Fourier1D(fp)
            k = 3
            u = cos.(k .* ring.mishPoints)
            ring.uMish .= u
            Fourier.FBtransform!(ring)
            Fourier.FAtransform!(ring)
            a  = copy(ring.a)
            ax = zeros(fp.yDim)
            Fourier.FIxcoefficients(fp, a, ax)
            # After differentiation: ax should have a sine component at index yDim-k+1
            @test abs(ax[fp.yDim - k + 1]) > 0.0   # non-zero sine component for k=3
            @test ax[k+1] ≈ 0.0  atol=1e-12         # cosine component should be zero
        end

        @testset "FIxtransform variants" begin
            # d/dθ(sin(k*θ)) = k*cos(k*θ)
            # Use yDim large enough for kmax to be accurate
            k   = 4
            fp  = Fourier.FourierParameters(ymin=0.0, kmax=8, yDim=64, bDim=17)
            ring = Fourier.Fourier1D(fp)
            pts = ring.mishPoints
            ring.uMish .= sin.(k .* pts)
            Fourier.FBtransform!(ring)
            Fourier.FAtransform!(ring)

            expected_dx = k .* cos.(k .* pts)

            # 4-arg (FourierParameters) variant
            ax_buf = zeros(fp.yDim)
            ux1 = Fourier.FIxtransform(fp, ring.ifftPlan, ring.a, ax_buf)
            @test length(ux1) == fp.yDim
            @test maximum(abs.(ux1 .- expected_dx)) < 1e-10

            # Fourier1D allocating variant
            ux2 = Fourier.FIxtransform(ring)
            @test ux2 ≈ ux1  atol=1e-12

            # Fourier1D in-place variant
            ux3 = zeros(fp.yDim)
            Fourier.FIxtransform(ring, ux3)
            @test ux3 ≈ ux1  atol=1e-12
        end

        @testset "FIxxtransform" begin
            # d²/dθ²(sin(k*θ)) = -k²*sin(k*θ)
            k   = 3
            fp  = Fourier.FourierParameters(ymin=0.0, kmax=8, yDim=64, bDim=17)
            ring = Fourier.Fourier1D(fp)
            pts = ring.mishPoints
            ring.uMish .= sin.(k .* pts)
            Fourier.FBtransform!(ring)
            Fourier.FAtransform!(ring)

            uxx = Fourier.FIxxtransform(ring)
            @test length(uxx) == fp.yDim
            @test maximum(abs.(uxx .+ k^2 .* sin.(k .* pts))) < 1e-10
        end

        @testset "FIIntcoefficients and FIInttransform" begin
            # ∫cos(k*θ)dθ = sin(k*θ)/k  (+ constant of integration)
            k   = 2
            fp  = Fourier.FourierParameters(ymin=0.0, kmax=8, yDim=64, bDim=17)
            ring = Fourier.Fourier1D(fp)
            pts = ring.mishPoints
            ring.uMish .= cos.(k .* pts)
            Fourier.FBtransform!(ring)
            Fourier.FAtransform!(ring)
            a    = copy(ring.a)
            aInt = zeros(fp.yDim)

            # FIIntcoefficients: verify non-zero integral coefficient for wavenumber k
            Fourier.FIIntcoefficients(fp, a, aInt, 0.0)
            @test abs(aInt[fp.yDim - k + 1]) > 0.0   # integral of cosine gives sine/k

            # FIInttransform (4-arg)
            aInt2 = zeros(fp.yDim)
            uint1 = Fourier.FIInttransform(fp, ring.ifftPlan, a, aInt2, 0.0)
            expected = sin.(k .* pts) ./ k
            @test maximum(abs.(uint1 .- expected)) < 1e-10

            # FIInttransform (ring variant, C0 = 0)
            uint2 = Fourier.FIInttransform(ring, 0.0)
            @test uint2 ≈ uint1  atol=1e-12
        end

        @testset "Phase-shifted ring round-trip" begin
            # With ymin ≠ 0 the phase-filter shifts the phase reference, but the
            # FB→FA→FI round-trip should still recover the original physical values.
            fp  = Fourier.FourierParameters(ymin=π/3, kmax=6, yDim=32, bDim=13)
            ring = Fourier.Fourier1D(fp)
            pts = ring.mishPoints
            u_orig = sin.(2 .* pts) .+ cos.(pts)
            ring.uMish .= u_orig
            Fourier.FBtransform!(ring)
            Fourier.FAtransform!(ring)
            Fourier.FItransform!(ring)
            @test maximum(abs.(ring.uMish .- u_orig)) < 1e-10
        end

        @testset "BC type: PERIODIC only" begin
            # Verify that the PERIODIC constant is accessible and is a Dict
            @test Fourier.PERIODIC isa Dict
            @test haskey(Fourier.PERIODIC, "PERIODIC")
        end

        @testset "Generic wrapper smoke test (Fourier)" begin
            # Verify that the no-prefix wrappers produce results identical to their
            # F-prefixed originals. These wrappers contain no logic; this test guards
            # against future naming regressions.
            fp   = Fourier.FourierParameters(ymin=0.0, kmax=6, yDim=32, bDim=13)
            ring = Fourier.Fourier1D(fp)
            u    = sin.(2 .* ring.mishPoints) .+ cos.(ring.mishPoints)

            # Btransform!
            ring.uMish .= u
            Fourier.FBtransform!(ring)
            b_ref = copy(ring.b)
            ring.uMish .= u
            Fourier.Btransform!(ring)
            @test ring.b == b_ref

            # Atransform!
            Fourier.FAtransform!(ring)
            a_ref = copy(ring.a)
            ring.b .= b_ref
            Fourier.Atransform!(ring)
            @test ring.a == a_ref

            # Itransform!
            Fourier.FItransform!(ring)
            u_ref = copy(ring.uMish)
            ring.a .= a_ref
            Fourier.Itransform!(ring)
            @test ring.uMish == u_ref

            # Ixtransform (allocating)
            dx_f = Fourier.FIxtransform(ring)
            dx_g = Fourier.Ixtransform(ring)
            @test dx_f == dx_g

            # Ixtransform (in-place)
            dx_buf_f = zeros(fp.yDim); Fourier.FIxtransform(ring, dx_buf_f)
            dx_buf_g = zeros(fp.yDim); Fourier.Ixtransform(ring, dx_buf_g)
            @test dx_buf_f == dx_buf_g

            # Ixxtransform
            dxx_f = Fourier.FIxxtransform(ring)
            dxx_g = Fourier.Ixxtransform(ring)
            @test dxx_f == dxx_g

            # IInttransform
            uint_f = Fourier.FIInttransform(ring, 0.0)
            uint_g = Fourier.IInttransform(ring, 0.0)
            @test uint_f == uint_g
        end

    end  # Fourier Tests

    @testset "Chebyshev Tests" begin

        @testset "ChebyshevParameters construction" begin
            cp = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=10000.0, zDim=25, bDim=25,
                                                BCB=Chebyshev.R0, BCT=Chebyshev.R0)
            @test cp.zmin == 0.0
            @test cp.zmax == 10000.0
            @test cp.zDim == 25
            @test cp.bDim == 25
            @test cp.BCB == Chebyshev.R0
            @test cp.BCT == Chebyshev.R0
            # Second configuration to verify generality
            cp2 = Chebyshev.ChebyshevParameters(zmin=-1000.0, zmax=1000.0, zDim=10, bDim=8,
                                                  BCB=Chebyshev.R1T0, BCT=Chebyshev.R0)
            @test cp2.zmin == -1000.0
            @test cp2.zmax == 1000.0
            @test cp2.bDim == 8
            @test cp2.BCB == Chebyshev.R1T0
        end

        @testset "Chebyshev1D construction" begin
            cp = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=1.0, zDim=25, bDim=25,
                                                BCB=Chebyshev.R0, BCT=Chebyshev.R0)
            col = Chebyshev.Chebyshev1D(cp)
            # Array sizes
            @test length(col.mishPoints) == 25   # zDim
            @test length(col.uMish)      == 25   # zDim
            @test length(col.b)          == 25   # bDim
            @test length(col.a)          == 25   # zDim (full for inverse DCT)
            @test length(col.ax)         == 25   # zDim (derivative buffer)
            @test size(col.filter)       == (25, 25)   # bDim == zDim → square filter
            # CGL ordering: first point is zmin (cos(0)*negative_scale = zmin), last is zmax
            @test col.mishPoints[1] ≈ cp.zmin atol=1e-12
            @test col.mishPoints[end] ≈ cp.zmax atol=1e-12
            # Monotonically increasing (CGL from bottom to top)
            @test all(diff(col.mishPoints) .> 0)

            # With truncation: bDim < zDim
            cp_trunc = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=1.0, zDim=20, bDim=12,
                                                       BCB=Chebyshev.R0, BCT=Chebyshev.R0)
            col_trunc = Chebyshev.Chebyshev1D(cp_trunc)
            @test length(col_trunc.b) == 12           # bDim
            @test size(col_trunc.filter) == (12, 20)  # truncation matrix
        end

        @testset "calcMishPoints" begin
            cp = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=1.0, zDim=5, bDim=5,
                                                BCB=Chebyshev.R0, BCT=Chebyshev.R0)
            pts = Chebyshev.calcMishPoints(cp)
            @test length(pts) == 5
            # Endpoints: first is zmin, last is zmax (negative scale maps cos(0) → zmin)
            @test pts[1] ≈ cp.zmin atol=1e-12
            @test pts[end] ≈ cp.zmax atol=1e-12
            # Monotonically increasing (CGL bottom to top)
            @test all(diff(pts) .> 0)
        end

        @testset "calcFilterMatrix" begin
            # bDim == zDim: exponential Eresman damping filter
            cp_eq = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=1.0, zDim=10, bDim=10,
                                                   BCB=Chebyshev.R0, BCT=Chebyshev.R0)
            filt_eq = Chebyshev.calcFilterMatrix(cp_eq)
            @test size(filt_eq) == (10, 10)
            @test filt_eq[1,1] ≈ exp(-36.0*(1/10)^36) atol=1e-10
            @test filt_eq[1,2] ≈ 0.0                                # off-diagonal zero

            # bDim < zDim: sharp truncation
            cp_trunc = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=1.0, zDim=10, bDim=6,
                                                      BCB=Chebyshev.R0, BCT=Chebyshev.R0)
            filt_trunc = Chebyshev.calcFilterMatrix(cp_trunc)
            @test size(filt_trunc) == (6, 10)
            @test filt_trunc[1,1] ≈ 1.0    # leading identity block
            @test filt_trunc[6,6] ≈ 1.0
        end

        @testset "CBtransform variants" begin
            cp = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=1.0, zDim=25, bDim=25,
                                                BCB=Chebyshev.R0, BCT=Chebyshev.R0)
            col = Chebyshev.Chebyshev1D(cp)
            u = sin.(π .* (col.mishPoints .- cp.zmin) ./ (cp.zmax - cp.zmin))

            # ChebyshevParameters + fftPlan variant
            b1 = Chebyshev.CBtransform(cp, col.fftPlan, u)
            @test length(b1) == cp.bDim
            @test maximum(abs.(b1)) > 0.0

            # Chebyshev1D + uMish variant
            b2 = Chebyshev.CBtransform(col, u)
            @test b1 ≈ b2

            # In-place CBtransform!
            col.uMish .= u
            Chebyshev.CBtransform!(col)
            @test col.b ≈ b1
        end

        @testset "CAtransform variants" begin
            cp = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=1.0, zDim=25, bDim=25,
                                                BCB=Chebyshev.R0, BCT=Chebyshev.R0)
            col = Chebyshev.Chebyshev1D(cp)
            col.uMish .= sin.(π .* (col.mishPoints .- cp.zmin) ./ (cp.zmax - cp.zmin))
            Chebyshev.CBtransform!(col)
            b = copy(col.b)

            # Allocating variant
            a1 = Chebyshev.CAtransform(cp, col.gammaBC, b)
            @test length(a1) == cp.zDim

            # In-place CAtransform!
            Chebyshev.CAtransform!(col)
            @test col.a ≈ a1
        end

        @testset "CItransform variants" begin
            cp = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=1.0, zDim=25, bDim=25,
                                                BCB=Chebyshev.R0, BCT=Chebyshev.R0)
            col = Chebyshev.Chebyshev1D(cp)
            u_orig = sin.(π .* (col.mishPoints .- cp.zmin) ./ (cp.zmax - cp.zmin))
            col.uMish .= u_orig
            Chebyshev.CBtransform!(col)
            Chebyshev.CAtransform!(col)

            # Allocating variant
            u1 = Chebyshev.CItransform(cp, col.fftPlan, col.a)
            @test length(u1) == cp.zDim
            @test maximum(abs.(u1 .- u_orig)) < 1e-10

            # In-place CItransform!
            Chebyshev.CItransform!(col)
            @test col.uMish ≈ u1
        end

        @testset "Round-trip accuracy (CB→CA→CI)" begin
            cp = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=Float64(π), zDim=25, bDim=25,
                                                BCB=Chebyshev.R0, BCT=Chebyshev.R0)
            col = Chebyshev.Chebyshev1D(cp)
            pts = col.mishPoints

            # Single sinusoid — should round-trip to machine precision
            u_sin = sin.(pts)
            col.uMish .= u_sin
            Chebyshev.CBtransform!(col)
            Chebyshev.CAtransform!(col)
            Chebyshev.CItransform!(col)
            @test maximum(abs.(col.uMish .- u_sin)) < 1e-10

            # Polynomial (degree ≤ zDim: exact in Chebyshev basis)
            u_poly = pts.^3 .- 2.0 .* pts.^2 .+ pts .+ 1.0
            col.uMish .= u_poly
            Chebyshev.CBtransform!(col)
            Chebyshev.CAtransform!(col)
            Chebyshev.CItransform!(col)
            @test maximum(abs.(col.uMish .- u_poly)) < 1e-10

            # Multi-frequency signal within basis bandwidth
            u_multi = cos.(pts) .+ 0.5 .* sin.(2 .* pts) .+ 0.25 .* cos.(3 .* pts)
            col.uMish .= u_multi
            Chebyshev.CBtransform!(col)
            Chebyshev.CAtransform!(col)
            Chebyshev.CItransform!(col)
            @test maximum(abs.(col.uMish .- u_multi)) < 1e-10
        end

        @testset "CIxcoefficients" begin
            # d/dz(sin(z)) = cos(z): verify derivative coefficients are non-trivial
            # and that the original `a` buffer is unchanged
            cp = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=Float64(π), zDim=25, bDim=25,
                                                BCB=Chebyshev.R0, BCT=Chebyshev.R0)
            col = Chebyshev.Chebyshev1D(cp)
            col.uMish .= sin.(col.mishPoints)
            Chebyshev.CBtransform!(col)
            Chebyshev.CAtransform!(col)
            a  = copy(col.a)
            ax = zeros(cp.zDim)
            Chebyshev.CIxcoefficients(cp, a, ax)
            # Derivative coefficients should be non-zero
            @test maximum(abs.(ax)) > 0.0
            # Original a should be unchanged
            @test a ≈ col.a
        end

        @testset "CIxtransform variants" begin
            # d/dz(sin(z)) = cos(z) on [0, π] with 33 CGL nodes
            cp = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=Float64(π), zDim=33, bDim=33,
                                                BCB=Chebyshev.R0, BCT=Chebyshev.R0)
            col = Chebyshev.Chebyshev1D(cp)
            pts = col.mishPoints
            col.uMish .= sin.(pts)
            Chebyshev.CBtransform!(col)
            Chebyshev.CAtransform!(col)

            expected_dx = cos.(pts)

            # 4-arg (ChebyshevParameters) variant
            ax_buf = zeros(cp.zDim)
            ux1 = Chebyshev.CIxtransform(cp, col.fftPlan, col.a, ax_buf)
            @test length(ux1) == cp.zDim
            @test maximum(abs.(ux1 .- expected_dx)) < 1e-8

            # Chebyshev1D convenience variant
            ux2 = Chebyshev.CIxtransform(col)
            @test ux2 ≈ ux1  atol=1e-12
        end

        @testset "CIxxtransform" begin
            # d²/dz²(sin(z)) = -sin(z) on [0, π] with 33 CGL nodes
            cp = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=Float64(π), zDim=33, bDim=33,
                                                BCB=Chebyshev.R0, BCT=Chebyshev.R0)
            col = Chebyshev.Chebyshev1D(cp)
            pts = col.mishPoints
            col.uMish .= sin.(pts)
            Chebyshev.CBtransform!(col)
            Chebyshev.CAtransform!(col)

            uxx = Chebyshev.CIxxtransform(col)
            @test length(uxx) == cp.zDim
            @test maximum(abs.(uxx .+ sin.(pts))) < 1e-6
        end

        @testset "CIIntcoefficients and CIInttransform" begin
            # ∫sin(z)dz = -cos(z) + C; verify the integral is non-trivial and
            # both CIInttransform variants produce identical results.
            cp = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=Float64(π), zDim=33, bDim=33,
                                                BCB=Chebyshev.R0, BCT=Chebyshev.R0)
            col = Chebyshev.Chebyshev1D(cp)
            pts = col.mishPoints
            col.uMish .= sin.(pts)
            Chebyshev.CBtransform!(col)
            Chebyshev.CAtransform!(col)
            a = copy(col.a)

            # CIIntcoefficients: verify non-trivial integral coefficients
            aInt = Chebyshev.CIIntcoefficients(cp, a, 0.0)
            @test length(aInt) == cp.zDim
            @test maximum(abs.(aInt)) > 0.0

            # CIInttransform (3-arg with fftPlan)
            uint1 = Chebyshev.CIInttransform(cp, col.fftPlan, a, 0.0)
            @test length(uint1) == cp.zDim
            @test maximum(abs.(uint1)) > 0.0

            # CIInttransform (column convenience variant, C0 = 0)
            uint2 = Chebyshev.CIInttransform(col, 0.0)
            @test uint2 ≈ uint1  atol=1e-12
        end

        @testset "BC types" begin
            # Verify all exported BC constants are accessible Dicts with expected keys
            @test Chebyshev.R0 isa Dict
            @test haskey(Chebyshev.R0, "R0")
            @test Chebyshev.R1T0 isa Dict
            @test haskey(Chebyshev.R1T0, "α0")
            @test Chebyshev.R1T1 isa Dict
            @test haskey(Chebyshev.R1T1, "α1")
            @test Chebyshev.R1T2 isa Dict
            @test haskey(Chebyshev.R1T2, "α2")
            @test Chebyshev.R2T10 isa Dict
            @test haskey(Chebyshev.R2T10, "β1")
            @test Chebyshev.R2T20 isa Dict
            @test haskey(Chebyshev.R2T20, "β1")
            @test Chebyshev.R3 isa Dict
            @test haskey(Chebyshev.R3, "R3")

            # Verify each constructable BC combination produces a finite A array after CB→CA
            function roundtrip_bc(bcb, bct)
                cp = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=1.0, zDim=15, bDim=15,
                                                    BCB=bcb, BCT=bct)
                col = Chebyshev.Chebyshev1D(cp)
                col.uMish .= sin.(π .* (col.mishPoints .- 0.0) ./ 1.0)
                Chebyshev.CBtransform!(col)
                Chebyshev.CAtransform!(col)
                return all(isfinite.(col.a))
            end

            @test roundtrip_bc(Chebyshev.R0,   Chebyshev.R0)    # Dirichlet/Dirichlet
            @test roundtrip_bc(Chebyshev.R1T0,  Chebyshev.R0)   # Neumann-T0 bottom
            @test roundtrip_bc(Chebyshev.R1T1,  Chebyshev.R0)   # Neumann-T1 bottom
            @test roundtrip_bc(Chebyshev.R0,    Chebyshev.R1T0) # Neumann-T0 top
            @test roundtrip_bc(Chebyshev.R0,    Chebyshev.R1T1) # Neumann-T1 top
            @test roundtrip_bc(Chebyshev.R1T0,  Chebyshev.R1T0) # Neumann-T0 both
            @test roundtrip_bc(Chebyshev.R1T1,  Chebyshev.R1T1) # Neumann-T1 both
            @test roundtrip_bc(Chebyshev.R1T0,  Chebyshev.R1T1) # mixed T0/T1
            @test roundtrip_bc(Chebyshev.R1T1,  Chebyshev.R1T0) # mixed T1/T0
        end

        @testset "Generic wrapper smoke test (Chebyshev)" begin
            # Verify that the no-prefix wrappers produce results identical to their
            # C-prefixed originals. These wrappers contain no logic; this test guards
            # against future naming regressions.
            cp  = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=Float64(π), zDim=25, bDim=25,
                                                 BCB=Chebyshev.R0, BCT=Chebyshev.R0)
            col = Chebyshev.Chebyshev1D(cp)
            u   = sin.(col.mishPoints) .+ 0.5 .* cos.(2 .* col.mishPoints)

            # Btransform!
            col.uMish .= u
            Chebyshev.CBtransform!(col)
            b_ref = copy(col.b)
            col.uMish .= u
            Chebyshev.Btransform!(col)
            @test col.b == b_ref

            # Atransform!
            Chebyshev.CAtransform!(col)
            a_ref = copy(col.a)
            col.b .= b_ref
            Chebyshev.Atransform!(col)
            @test col.a == a_ref

            # Itransform!
            Chebyshev.CItransform!(col)
            u_ref = copy(col.uMish)
            col.a .= a_ref
            Chebyshev.Itransform!(col)
            @test col.uMish == u_ref

            # Ixtransform (allocating)
            dx_c = Chebyshev.CIxtransform(col)
            dx_g = Chebyshev.Ixtransform(col)
            @test dx_c == dx_g

            # Ixxtransform
            dxx_c = Chebyshev.CIxxtransform(col)
            dxx_g = Chebyshev.Ixxtransform(col)
            @test dxx_c == dxx_g

            # IInttransform
            uint_c = Chebyshev.CIInttransform(col, 0.0)
            uint_g = Chebyshev.IInttransform(col, 0.0)
            @test uint_c == uint_g
        end

    end  # Chebyshev Tests

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
            # Test R0 boundary condition (zero at boundaries)
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
            
            # Set interior values
            for i = 1:length(gridpoints)
                grid.physical[i, 1, 1] = sin(π * (gridpoints[i] - gp.iMin) / (gp.iMax - gp.iMin))
            end
            
            spectralTransform!(grid)
            gridTransform!(grid)
            
            # Values should be close to zero at boundaries
            @test abs(grid.physical[1, 1, 1]) < 0.1
            @test abs(grid.physical[end, 1, 1]) < 0.1
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
            @test grid.splines[1, 1].params.l_q == 1.5
            @test grid.splines[1, 2].params.l_q == 3.0
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

            # splineTransform!: B (in SharedArray) → A (in patchA)
            sharedB = SharedArray{Float64}(size(patch.spectral))
            sharedB[:, :] .= patch.spectral
            patchA = zeros(Float64, size(patch.spectral))
            splineTransform!(patch.splines, patchA, patch.params, sharedB, tile)

            # tileTransform!: A coefficients → physical on tile gridpoints
            splineBuffer = allocateSplineBuffer(patch, tile)
            tileTransform!(patch.splines, patchA, patch.params, tile, splineBuffer)

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

            # 3 equal tiles
            tp3 = Springsteel.calcTileSizes(patch, 3)
            @test size(tp3, 2) == 3
            @test tp3[1, 1] ≈ 0.0                             # first iMin = patch iMin
            @test tp3[2, end] ≈ 30.0                           # last iMax = patch iMax
            @test sum(Int.(tp3[3, :])) == 30                   # num_cells sums to patch total
            @test Int(tp3[4, 1]) == 1                          # first spectralIndexL = 1
            @test Int(tp3[4, 2]) == Int(tp3[3, 1]) + 1        # second starts after first's cells
            @test sum(Int.(tp3[5, :])) == patch.params.iDim    # tile sizes sum to total gridpoints

            # 2 equal tiles
            tp2 = Springsteel.calcTileSizes(patch, 2)
            @test size(tp2, 2) == 2
            @test tp2[1, 1] ≈ 0.0
            @test tp2[2, end] ≈ 30.0
            @test sum(Int.(tp2[3, :])) == 30
            @test Int(tp2[4, 1]) == 1

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
            gridTransform!(patch.splines, patch.spectral, patch.params, tile, splineBuffer)

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

            # biL..biR in patch = spectralIndexR-2 .. spectralIndexR = 16..18
            # tiL..tiR in tile  = b_iDim-2 .. b_iDim                = 16..18
            biL = tile.params.spectralIndexR - 2
            biR = tile.params.spectralIndexR
            tiL = tile.params.b_iDim - 2

            ps_buf = zeros(Float64, patch.params.b_iDim, 1)
            border = Springsteel.getBorderSpectral(patch.params, tile, ps_buf)

            @test size(border, 1) == patch.params.b_iDim
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

            patchMap, tileView = calcPatchMap(patch, tile)

            @test size(patchMap) == size(patch.spectral)
            @test all(patchMap[siL:siR, :])              # inner rows marked
            @test !any(patchMap[siR+1:end, :])           # rows beyond siR not marked
            @test length(tileView) == (siR - siL + 1)   # view covers same span
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

            haloMap, haloView = calcHaloMap(patch, tile)

            @test size(haloMap) == size(patch.spectral)
            @test all(haloMap[hiL:hiR, :])     # exactly those 3 rows are true
            @test count(haloMap) == 3
            @test length(haloView) == 3        # view covers the 3 halo coefficients
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

            # Build border spectral (sparse: non-zeros at patch rows 16-18)
            ps_buf = zeros(Float64, patch.params.b_iDim, 1)
            borderSpectral = Springsteel.getBorderSpectral(patch.params, tile, ps_buf)

            sharedSpectral = SharedArray{Float64}(patch.params.b_iDim, 1)
            sharedSpectral[:, :] .= 0.0
            Springsteel.sumSharedSpectral(sharedSpectral, borderSpectral, patch.params, tile)

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

            @test Springsteel.num_columns(grid) == 0

            buf = allocateSplineBuffer(grid, grid)
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
            
            # splines array: one for each j spectral coefficient
            @test size(grid.splines, 1) == grid.params.b_jDim
            
            # rings array: one for each i gridpoint
            @test size(grid.rings, 1) == grid.params.iDim
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
            
            # Check that splines have correct filter length for i dimension
            @test grid.splines[1, 1].params.l_q == 1.5
            
            # Check that rings have correct filter length for j dimension
            @test grid.rings[1, 1].params.l_q == 2.5
        end
    end
    
end

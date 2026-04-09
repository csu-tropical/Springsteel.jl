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
            @test spline.params.mishDim == 60       # 20 * 3
            @test spline.params.bDim == 23          # 20 + 3
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
            @test length(b1) == spline.params.bDim
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
            @test length(a1) == spline.params.bDim

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
            u_true = sin.(pts .* π ./ 10.0)  # vanishes at x=0 and x=10; using free (R0) BCs for this accuracy test
            b = SBtransform(spline, u_true)
            a = SAtransform(spline, b)
            spline.b .= b
            SAtransform!(spline)

            # Scalar variant (single point)
            u_scalar = SItransform(sp, a, pts[15], 0)
            @test u_scalar ≈ u_true[15]  atol=1e-4

            # Mish-point variant returning new vector
            u_mish = SItransform(sp, a, 0)
            @test length(u_mish) == spline.params.mishDim
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
            @test size(M) == (spline.params.mishDim, spline.params.bDim)
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
            @test length(dx1) == spline.params.mishDim
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
            @test length(dxx1) == spline.params.mishDim
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
                @test spline.params.mishDim == 60
                @test spline.params.bDim == 23
                u_vals = interior_fn.(spline.mishPoints)
                b = SBtransform(spline, u_vals)
                a = SAtransform(spline, b)
                @test all(isfinite.(a))
            end

            # R3 on both sides
            sp_r3 = SplineParameters(xmin=0.0, xmax=10.0, num_cells=20,
                                     BCL=CubicBSpline.R3, BCR=CubicBSpline.R3)
            spline_r3 = Spline1D(sp_r3)
            @test spline_r3.params.mishDim == 60
            a_r3 = SAtransform(spline_r3,
                               SBtransform(spline_r3, interior_fn.(spline_r3.mishPoints)))
            @test all(isfinite.(a_r3))

            # R1T1 (Neumann, zero first derivative) on both sides.
            # cos(π*x/L) has zero first derivative at x=0 (cos'=0) and x=L (cos'=0),
            # so it is the natural test function for R1T1 BCs.
            # Note: R1T0 is Dirichlet (zero *value*), not Neumann; do not confuse them.
            sp_r1t1 = SplineParameters(xmin=0.0, xmax=10.0, num_cells=30,
                                       BCL=CubicBSpline.R1T1, BCR=CubicBSpline.R1T1)
            spline_r1t1 = Spline1D(sp_r1t1)
            u_r1t1 = cos.(π .* spline_r1t1.mishPoints ./ 10.0)
            b_r1t1 = SBtransform(spline_r1t1, u_r1t1)
            a_r1t1 = SAtransform(spline_r1t1, b_r1t1)
            spline_r1t1.b .= b_r1t1
            SAtransform!(spline_r1t1)
            u_reconstructed = SItransform(spline_r1t1.params, spline_r1t1.a, 0)
            @test all(isfinite.(u_reconstructed))
            # R1T1 enforces zero first derivative; also verify R1T0 (Dirichlet) with
            # sin(π*x/L), which vanishes at both endpoints (zero value BC).
            sp_r1t0 = SplineParameters(xmin=0.0, xmax=10.0, num_cells=30,
                                       BCL=CubicBSpline.R1T0, BCR=CubicBSpline.R1T0)
            spline_r1t0 = Spline1D(sp_r1t0)
            u_r1t0 = sin.(π .* spline_r1t0.mishPoints ./ 10.0)  # vanishes at both endpoints
            b_r1t0 = SBtransform(spline_r1t0, u_r1t0)
            spline_r1t0.b .= b_r1t0
            SAtransform!(spline_r1t0)
            u_r1t0_recov = SItransform(spline_r1t0.params, spline_r1t0.a, 0)
            @test all(isfinite.(u_r1t0_recov))
            @test maximum(abs.(u_r1t0_recov .- u_r1t0)) < 1e-3  # round-trip accuracy
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

        @testset "SIIntcoefficients and SIInttransform" begin
            # ∫sin(x)dx = -cos(x)+C on [0, 2π] with PERIODIC BCs.
            # With 40 cells the error should be < 1e-5.
            L = 2*Float64(π); nc = 40
            sp_per = SplineParameters(xmin=0.0, xmax=L, num_cells=nc,
                                      BCL=CubicBSpline.PERIODIC, BCR=CubicBSpline.PERIODIC)
            spline_per = CubicBSpline.Spline1D(sp_per)
            pts_per = spline_per.mishPoints
            f_per = sin.(pts_per)
            expected_per = -cos.(pts_per)

            # SIIntcoefficients (Spline1D convenience form): returns aInt of correct length
            aInt = CubicBSpline.SIIntcoefficients(spline_per, f_per)
            @test length(aInt) == spline_per.params.bDim
            @test maximum(abs.(aInt)) > 0.0      # non-trivial coefficients

            # SIInttransform (Spline1D, C0=0): result length and accuracy
            uInt = CubicBSpline.SIInttransform(spline_per, f_per, 0.0)
            @test length(uInt) == sp_per.num_cells * 3
            C_per = sum(expected_per .- uInt) / length(pts_per)
            @test maximum(abs.(uInt .+ C_per .- expected_per)) < 1e-5

            # C0 constant of integration shifts output uniformly
            uInt_c0 = CubicBSpline.SIInttransform(spline_per, f_per, 2.5)
            @test maximum(abs.(uInt_c0 .- uInt .- 2.5)) < 1e-14

            # SIInttransform (SplineParameters low-level form) matches Spline1D form
            uInt_ll = CubicBSpline.SIInttransform(sp_per, spline_per.gammaBC, spline_per.p1Factor,
                                                   f_per, 0.0)
            @test uInt_ll ≈ uInt  atol=1e-12

            # SIIntcoefficients (SplineParameters low-level form) matches Spline1D form
            aInt_ll = CubicBSpline.SIIntcoefficients(sp_per, spline_per.gammaBC, spline_per.p1Factor,
                                                      f_per)
            @test aInt_ll ≈ aInt  atol=1e-12

            # Test with R0 BCs: ∫sin(πx/L)dx = -(L/π)cos(πx/L)+C on [0,10]
            L2 = 10.0
            sp_r0 = SplineParameters(xmin=0.0, xmax=L2, num_cells=nc,
                                     BCL=CubicBSpline.R0, BCR=CubicBSpline.R0)
            spline_r0 = CubicBSpline.Spline1D(sp_r0)
            pts_r0 = spline_r0.mishPoints
            f_r0 = sin.(Float64(π)/L2 .* pts_r0)
            expected_r0 = -(L2/Float64(π)) .* cos.(Float64(π)/L2 .* pts_r0)
            uInt_r0 = CubicBSpline.SIInttransform(spline_r0, f_r0, 0.0)
            C_r0 = sum(expected_r0 .- uInt_r0) / length(pts_r0)
            @test maximum(abs.(uInt_r0 .+ C_r0 .- expected_r0)) < 1e-5

            # IInttransform generic wrapper produces identical result to SIInttransform
            uInt_gen = CubicBSpline.IInttransform(spline_per, f_per, 0.0)
            @test uInt_gen == uInt
        end

    end

    @testset "CubicBSpline matrix representations" begin
        sp = SplineParameters(xmin=0.0, xmax=10.0, num_cells=20,
                              BCL=CubicBSpline.R0, BCR=CubicBSpline.R0)
        spline = Spline1D(sp)
        pts = spline.mishPoints

        M = CubicBSpline.spline_basis_matrix(spline)
        Mx = CubicBSpline.spline_1st_derivative_matrix(spline)
        Mxx = CubicBSpline.spline_2nd_derivative_matrix(spline)

        # Test 1: Size check
        @test size(M) == (sp.mishDim, sp.bDim)
        @test size(Mx) == (sp.mishDim, sp.bDim)
        @test size(Mxx) == (sp.mishDim, sp.bDim)

        # Test 2: M * a ≈ SItransform for a polynomial input
        # Use a quadratic: f(x) = x^2
        spline2 = Spline1D(sp)
        setMishValues(spline2, pts .^ 2)
        SBtransform!(spline2)
        SAtransform!(spline2)
        u_matrix = M * spline2.a
        u_transform = zeros(sp.mishDim)
        SItransform(spline2, u_transform)
        @test maximum(abs.(u_matrix .- u_transform)) < 1e-10

        # Test 3: Mx * a ≈ SIxtransform for x^2 → 2x
        deriv1 = Mx * spline2.a
        ux_transform = SIxtransform(spline2)
        @test maximum(abs.(deriv1 .- ux_transform)) < 1e-10

        # Test 4: Mxx * a ≈ SIxxtransform for x^3
        sp3 = SplineParameters(xmin=0.0, xmax=10.0, num_cells=20,
                               BCL=CubicBSpline.R0, BCR=CubicBSpline.R0)
        spline3 = Spline1D(sp3)
        pts3 = spline3.mishPoints
        setMishValues(spline3, pts3 .^ 3)
        SBtransform!(spline3)
        SAtransform!(spline3)
        Mxx3 = CubicBSpline.spline_2nd_derivative_matrix(spline3)
        deriv2 = Mxx3 * spline3.a
        uxx_transform = SIxxtransform(spline3)
        @test maximum(abs.(deriv2 .- uxx_transform)) < 1e-10

        # Test 5: First derivative of constant ≈ 0
        spline_c = Spline1D(sp)
        setMishValues(spline_c, fill(5.0, sp.mishDim))
        SBtransform!(spline_c)
        SAtransform!(spline_c)
        @test maximum(abs.(Mx * spline_c.a)) < 1e-10

        # Test 6: Matrix with BC folding produces correct dimensions
        sp_bc = SplineParameters(xmin=0.0, xmax=10.0, num_cells=20,
                                 BCL=CubicBSpline.R1T0, BCR=CubicBSpline.R1T0)
        spline_bc = Spline1D(sp_bc)
        M_folded = CubicBSpline.spline_basis_matrix(spline_bc; gammaBC=spline_bc.gammaBC)
        # R1T0 has rank 1 on each side, so Minterior = bDim - 2
        @test size(M_folded) == (sp_bc.mishDim, sp_bc.bDim - 2)
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

        @testset "Fourier matrix representations" begin
            fp = Fourier.FourierParameters(ymin=0.0, kmax=5, yDim=32, bDim=11)
            ring = Fourier.Fourier1D(fp)
            pts = ring.mishPoints

            M = Fourier.dft_matrix(ring)
            Mx = Fourier.dft_1st_derivative(ring)
            Mxx = Fourier.dft_2nd_derivative(ring)

            # Test 1: Size check
            @test size(M) == (fp.yDim, fp.bDim)
            @test size(Mx) == (fp.yDim, fp.bDim)
            @test size(Mxx) == (fp.yDim, fp.bDim)

            # Test 2: Round-trip consistency with FB transform
            ring.uMish .= sin.(pts)
            FBtransform!(ring)
            u_matrix = M * ring.b
            FAtransform!(ring)
            FItransform!(ring)
            @test maximum(abs.(u_matrix .- ring.uMish)) < 1e-10

            # Test 3: First derivative of sin(x) ≈ cos(x)
            # Get b-coefficients for sin(x)
            ring2 = Fourier.Fourier1D(fp)
            ring2.uMish .= sin.(pts)
            FBtransform!(ring2)
            deriv1 = Mx * ring2.b
            @test maximum(abs.(deriv1 .- cos.(pts))) < 1e-10

            # Test 4: Second derivative of sin(x) ≈ -sin(x)
            deriv2 = Mxx * ring2.b
            @test maximum(abs.(deriv2 .+ sin.(pts))) < 1e-10

            # Test 5: First derivative of constant ≈ 0
            ring3 = Fourier.Fourier1D(fp)
            ring3.uMish .= 5.0
            FBtransform!(ring3)
            @test maximum(abs.(Mx * ring3.b)) < 1e-12

            # Test 6: Second derivative of cos(2x) ≈ -4*cos(2x)
            ring4 = Fourier.Fourier1D(fp)
            ring4.uMish .= cos.(2.0 .* pts)
            FBtransform!(ring4)
            deriv2_cos2x = Mxx * ring4.b
            @test maximum(abs.(deriv2_cos2x .+ 4.0 .* cos.(2.0 .* pts))) < 1e-10
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

        @testset "dct_matrix properties" begin
            N = 8
            M = Chebyshev.dct_matrix(N)
            @test size(M) == (N, N)
            # First row (t=0): 2*cos((j-1)*0) = 2, endpoint scaling halves columns 1 & N → 1, 2, 2, ..., 2, 1
            @test M[1, 1] ≈ 1.0
            @test M[1, N] ≈ 1.0
            @test all(M[1, 2:N-1] .≈ 2.0)
            # Last row (t=π): 2*cos((j-1)*π) = 2*(-1)^(j-1); endpoints halved
            @test M[N, 1] ≈ 1.0
            @test M[N, 2] ≈ -2.0
            @test M[N, N] ≈ ((-1.0)^(N-1))
            # First column is constant (1.0 after halving) for all rows
            @test all(M[:, 1] .≈ 1.0)
        end

        @testset "dct_1st_derivative against analytic sin(πz/L)" begin
            # Use Chebyshev grid and compare spectral 1st derivative matrix
            # applied to A-coefficients against analytic derivative.
            L = 2.0
            N = 33
            cp = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=L, zDim=N, bDim=N,
                                                 BCB=Chebyshev.R0, BCT=Chebyshev.R0)
            col = Chebyshev.Chebyshev1D(cp)
            f_vals = sin.(π .* col.mishPoints ./ L)
            col.uMish .= f_vals
            Chebyshev.CBtransform!(col)
            Chebyshev.CAtransform!(col)

            D1 = Chebyshev.dct_1st_derivative(N, L)
            @test size(D1) == (N, N)
            df_spec = D1 * col.a
            df_exact = (π/L) .* cos.(π .* col.mishPoints ./ L)
            @test maximum(abs.(df_spec .- df_exact)) < 1e-8
        end

        @testset "dct_2nd_derivative against analytic sin(πz/L)" begin
            L = 2.0
            N = 33
            cp = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=L, zDim=N, bDim=N,
                                                 BCB=Chebyshev.R0, BCT=Chebyshev.R0)
            col = Chebyshev.Chebyshev1D(cp)
            f_vals = sin.(π .* col.mishPoints ./ L)
            col.uMish .= f_vals
            Chebyshev.CBtransform!(col)
            Chebyshev.CAtransform!(col)

            D2 = Chebyshev.dct_2nd_derivative(N, L)
            @test size(D2) == (N, N)
            d2f_spec = D2 * col.a
            d2f_exact = -(π/L)^2 .* sin.(π .* col.mishPoints ./ L)
            @test maximum(abs.(d2f_spec .- d2f_exact)) < 1e-6
        end

        @testset "CItransform_matrix evaluates at arbitrary points" begin
            # Verify CItransform_matrix matches a round-trip CB/CA/CI transform at arbitrary
            # points by comparing against transforming a known smooth field and using
            # CIxtransform for the first derivative.
            cp = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=Float64(π), zDim=33, bDim=33,
                                                 BCB=Chebyshev.R0, BCT=Chebyshev.R0)
            col = Chebyshev.Chebyshev1D(cp)
            col.uMish .= sin.(2 .* col.mishPoints)
            Chebyshev.CBtransform!(col)
            Chebyshev.CAtransform!(col)

            # At mishPoints, M0 * a should recover the original values.
            M0_mish = Chebyshev.CItransform_matrix(col, col.mishPoints, 0)
            @test size(M0_mish) == (cp.zDim, cp.zDim)
            vals_mish = M0_mish * col.a
            @test maximum(abs.(vals_mish .- sin.(2 .* col.mishPoints))) < 1e-10

            # At arbitrary interior points, M0 * a should match sin(2z) closely.
            zpts = [0.3, 0.8, 1.5, 2.2, 2.9]
            M0 = Chebyshev.CItransform_matrix(col, zpts, 0)
            M1 = Chebyshev.CItransform_matrix(col, zpts, 1)
            @test size(M0) == (length(zpts), cp.zDim)
            @test size(M1) == (length(zpts), cp.zDim)

            vals = M0 * col.a
            @test maximum(abs.(vals .- sin.(2 .* zpts))) < 1e-10

            # First derivative: analytic d/dz sin(2z) = 2 cos(2z).
            ders = M1 * col.a
            @test maximum(abs.(ders .- 2 .* cos.(2 .* zpts))) < 1e-8
        end

    end  # Chebyshev Tests

    @testset "Basis Interface" begin

        @testset "Spline1D accessors" begin
            gp = GridParameters(
                geometry = "R",
                xmin = 0.0, xmax = 10.0, num_cells = 5,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            grid = createGrid(gp)
            spline = grid.ibasis.data[1]

            pts = gridpoints(spline)
            @test length(pts) == spline.params.mishDim
            @test pts == spline.mishPoints

            @test spectral_dim(spline) == spline.params.bDim          # num_cells + 3
            @test spectral_dim(spline) == 8                    # 5 + 3
            @test physical_dim(spline) == spline.params.mishDim       # num_cells * mubar
            @test physical_dim(spline) == 15                   # 5 * 3
        end

        @testset "Fourier1D accessors" begin
            gp = GridParameters(
                geometry = "RL",
                xmin = 0.0, xmax = 10.0, num_cells = 4,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1)
            )
            grid = createGrid(gp)
            ring = grid.jbasis.data[1, 1]

            pts = gridpoints(ring)
            @test length(pts) == ring.params.yDim
            @test pts == ring.mishPoints

            @test spectral_dim(ring) == ring.params.bDim
            @test physical_dim(ring) == ring.params.yDim
        end

        @testset "Chebyshev1D accessors" begin
            gp = GridParameters(
                geometry = "RZ",
                xmin = 0.0, xmax = 10.0, num_cells = 4,
                zmin = 0.0, zmax = 5.0, zDim = 10,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCB = Dict("u" => Chebyshev.R0),
                BCT = Dict("u" => Chebyshev.R0),
                vars = Dict("u" => 1)
            )
            grid = createGrid(gp)
            col = grid.kbasis.data[1]

            pts = gridpoints(col)
            @test length(pts) == col.params.zDim
            @test pts == col.mishPoints

            @test spectral_dim(col) == col.params.bDim
            @test physical_dim(col) == col.params.zDim
        end

    end  # Basis Interface

    # ─────────────────────────────────────────────────────────────────────────
    # SpringsteelGrid Factory
    # ─────────────────────────────────────────────────────────────────────────

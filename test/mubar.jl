    @testset "Configurable mubar & Regular Grid" begin

        # ── T2: SplineParameters Construction ─────────────────────────────────
        @testset "SplineParameters mubar/quadrature" begin
            # Default values preserved
            sp = SplineParameters(xmin=0, xmax=10, num_cells=10)
            @test sp.mubar == 3
            @test sp.quadrature == :gauss
            @test sp.mishDim == 30

            # Custom mubar
            sp2 = SplineParameters(xmin=0, xmax=10, num_cells=10, mubar=2)
            @test sp2.mubar == 2
            @test sp2.mishDim == 20

            sp5 = SplineParameters(xmin=0, xmax=10, num_cells=10, mubar=5)
            @test sp5.mubar == 5
            @test sp5.mishDim == 50

            # Regular quadrature
            sp_reg = SplineParameters(xmin=0, xmax=10, num_cells=10, quadrature=:regular)
            @test sp_reg.quadrature == :regular
            @test sp_reg.mishDim == 30

            # mubar=1 allowed
            sp1 = SplineParameters(xmin=0, xmax=10, num_cells=10, mubar=1)
            @test sp1.mishDim == 10

            # Invalid inputs
            @test_throws ArgumentError Spline1D(SplineParameters(xmin=0, xmax=10, num_cells=10, mubar=0))
            @test_throws ArgumentError Spline1D(SplineParameters(xmin=0, xmax=10, num_cells=10, quadrature=:foo))
            @test_throws ArgumentError Spline1D(SplineParameters(xmin=0, xmax=10, num_cells=10, mubar=6, quadrature=:gauss))

            # mubar=6 regular is fine
            sp6_reg = SplineParameters(xmin=0, xmax=10, num_cells=10, mubar=6, quadrature=:regular)
            spline6 = Spline1D(sp6_reg)
            @test sp6_reg.mishDim == 60
            @test length(spline6.mishPoints) == 60
        end

        # ── T3: Quadrature Rule Accuracy ──────────────────────────────────────
        @testset "Quadrature rule tables" begin
            for mb in 1:5
                qpts, qwts = CubicBSpline._quadrature_rule(mb, :gauss)
                @test length(qpts) == mb
                @test length(qwts) == mb
                @test sum(qwts) ≈ 1.0 atol=1e-14
                max_exact = 2 * mb - 1
                for p in 0:max_exact
                    exact = 1.0 / (p + 1)
                    numerical = sum(qwts[i] * qpts[i]^p for i in 1:mb)
                    @test numerical ≈ exact atol=1e-13
                end
            end

            for mb in 1:5
                qpts, qwts = CubicBSpline._quadrature_rule(mb, :regular)
                @test length(qpts) == mb
                for i in 1:mb
                    @test qpts[i] ≈ (2i - 1) / (2 * mb) atol=1e-14
                end
                @test all(w ≈ 1.0/mb for w in qwts)
                @test sum(qwts) ≈ 1.0 atol=1e-14
            end

            @test_throws ArgumentError CubicBSpline._quadrature_rule(0, :gauss)
            @test_throws ArgumentError CubicBSpline._quadrature_rule(6, :gauss)
            @test_throws ArgumentError CubicBSpline._quadrature_rule(3, :foo)
        end

        # ── T4: calcMishPoints Verification ───────────────────────────────────
        @testset "calcMishPoints" begin
            sp3 = SplineParameters(xmin=0, xmax=10, num_cells=10, mubar=3, quadrature=:gauss)
            pts3 = CubicBSpline.calcMishPoints(sp3)
            @test length(pts3) == 30
            sqrt35 = sqrt(3.0/5.0)
            DX = 1.0
            for mc in 0:9, mu in 1:3
                i = mu + 3*mc
                expected = 0.0 + mc*DX + DX*((mu/2.0 - 1.0)*sqrt35) + DX*0.5
                @test pts3[i] ≈ expected atol=1e-14
            end

            sp2 = SplineParameters(xmin=0, xmax=10, num_cells=10, mubar=2, quadrature=:gauss)
            pts2 = CubicBSpline.calcMishPoints(sp2)
            @test length(pts2) == 20
            @test all(0.0 .< pts2 .< 10.0)

            sp_reg = SplineParameters(xmin=0, xmax=10, num_cells=10, mubar=3, quadrature=:regular)
            pts_reg = CubicBSpline.calcMishPoints(sp_reg)
            @test length(pts_reg) == 30
            diffs = diff(pts_reg)
            @test all(d ≈ diffs[1] for d in diffs)
            @test diffs[1] ≈ 1.0/3.0 atol=1e-14

            sp_reg1 = SplineParameters(xmin=0, xmax=10, num_cells=10, mubar=1, quadrature=:regular)
            pts_reg1 = CubicBSpline.calcMishPoints(sp_reg1)
            @test length(pts_reg1) == 10
            @test pts_reg1[1] ≈ 0.5 atol=1e-14
            @test all(diff(pts_reg1) .≈ 1.0)

            sp5 = SplineParameters(xmin=0, xmax=10, num_cells=10, mubar=5, quadrature=:gauss)
            pts5 = CubicBSpline.calcMishPoints(sp5)
            @test length(pts5) == 50
            @test all(0.0 .< pts5 .< 10.0)
            @test all(diff(pts5) .> 0)
        end

        # ── T5: Round-Trip Transform Accuracy ─────────────────────────────────
        # Use R0 (free) BCs and a smooth function to test the pure transform accuracy.
        # The l_q filter means round-trip is not exact even for Gauss — test that
        # accuracy scales with mubar and quadrature type.
        @testset "Round-trip transform accuracy" begin
            f(x) = sin(2 * pi * x / 10.0)

            configs = [
                (mubar=3, quad=:gauss,   label="gauss-3 (legacy)", tol=5e-3),
                (mubar=2, quad=:gauss,   label="gauss-2",          tol=5e-2),
                (mubar=5, quad=:gauss,   label="gauss-5",          tol=5e-3),
                (mubar=3, quad=:regular, label="regular-3",        tol=5e-2),
                (mubar=2, quad=:regular, label="regular-2",        tol=5e-2),
                (mubar=1, quad=:regular, label="regular-1",        tol=2e-1),
                (mubar=1, quad=:gauss,   label="gauss-1",          tol=2e-1),
            ]

            for cfg in configs
                @testset "$(cfg.label)" begin
                    sp = SplineParameters(
                        xmin=0, xmax=10, num_cells=20,
                        mubar=cfg.mubar, quadrature=cfg.quad,
                        l_q=2.0, BCL=CubicBSpline.R0, BCR=CubicBSpline.R0)
                    spline = Spline1D(sp)
                    pts = spline.mishPoints

                    u_orig = f.(pts)
                    setMishValues(spline, u_orig)

                    SBtransform!(spline)
                    SAtransform!(spline)

                    u_recon = CubicBSpline.SItransform(spline.params, spline.a)

                    max_err = maximum(abs.(u_recon .- u_orig))
                    @test max_err < cfg.tol
                end
            end
        end

        # ── T6: P+Q Matrix Properties ────────────────────────────────────────
        @testset "PQ matrix properties" begin
            for mb in [1, 2, 3, 5]
                for quad in [:gauss, :regular]
                    @testset "mubar=$mb quad=$quad" begin
                        sp = SplineParameters(
                            xmin=0, xmax=10, num_cells=10,
                            mubar=mb, quadrature=quad,
                            l_q=2.0, BCL=CubicBSpline.R1T0, BCR=CubicBSpline.R1T0)
                        gammaBC = CubicBSpline.calcGammaBC(sp)
                        PQ, PQfactor = CubicBSpline.calcPQfactor(sp, gammaBC)
                        @test PQ ≈ PQ' atol=1e-14
                        # Cholesky succeeded — just check the factorization object exists
                        @test PQfactor !== nothing
                    end
                end
            end
        end

        # ── T7: Derivative Accuracy ──────────────────────────────────────────
        @testset "Derivative accuracy" begin
            # Use a function that satisfies R1T0 BCs: sin(pi*x/10) vanishes at 0 and 10
            f(x)  = sin(pi * x / 10.0)
            fp(x) = (pi / 10.0) * cos(pi * x / 10.0)

            for (mb, quad, tol) in [(3, :gauss, 5e-3), (3, :regular, 5e-2), (2, :gauss, 5e-2)]
                @testset "mubar=$mb quad=$quad" begin
                    sp = SplineParameters(
                        xmin=0, xmax=10, num_cells=20,
                        mubar=mb, quadrature=quad,
                        l_q=2.0, BCL=CubicBSpline.R1T0, BCR=CubicBSpline.R1T0)
                    spline = Spline1D(sp)
                    pts = spline.mishPoints

                    u = f.(pts)
                    setMishValues(spline, u)
                    SBtransform!(spline)
                    SAtransform!(spline)

                    u_deriv = CubicBSpline.SItransform(sp, spline.a, pts, zeros(length(pts)), 1)
                    u_exact = fp.(pts)

                    max_err = maximum(abs.(u_deriv .- u_exact))
                    @test max_err < tol
                end
            end
        end

        # ── T8: Grid Construction with Non-Default mubar ─────────────────────
        @testset "Grid construction with mubar" begin
            gp = SpringsteelGridParameters(
                geometry="R", iMin=0, iMax=10, num_cells=10,
                mubar=2, quadrature=:regular,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)
            @test grid.params.iDim == 20
            @test grid.params.mubar == 2
            @test grid.params.quadrature == :regular

            gp_g2 = SpringsteelGridParameters(
                geometry="R", iMin=0, iMax=10, num_cells=10,
                mubar=2, quadrature=:gauss,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0))
            grid_g2 = createGrid(gp_g2)
            @test grid_g2.params.iDim == 20

            # Default mubar unchanged
            gp_default = SpringsteelGridParameters(
                geometry="R", iMin=0, iMax=10, num_cells=10,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0))
            grid_default = createGrid(gp_default)
            @test grid_default.params.iDim == 30
            @test grid_default.params.mubar == 3
            @test grid_default.params.quadrature == :gauss
        end

        # ── T9: Tiling with Non-Default mubar ────────────────────────────────
        @testset "Tiling with mubar" begin
            gp = SpringsteelGridParameters(
                geometry="R", iMin=0, iMax=100, num_cells=30,
                mubar=2, quadrature=:regular,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)
            tiles = calcTileSizes(grid, 3)
            @test length(tiles) == 3
            for tile in tiles
                @test tile.params.mubar == 2
                @test tile.params.quadrature == :regular
            end
            total_cells = sum(t.params.num_cells for t in tiles)
            @test total_cells == 30
        end

        # ── T10: Regular Grid Initialization from Data ───────────────────────
        @testset "Regular-to-Gauss initialization" begin
            f(x) = sin(pi * x / 10.0)

            num_cells = 20
            mb = 3
            sp_reg = SplineParameters(
                xmin=0, xmax=10, num_cells=num_cells,
                mubar=mb, quadrature=:regular,
                l_q=2.0, BCL=CubicBSpline.R1T0, BCR=CubicBSpline.R1T0)
            spline_reg = Spline1D(sp_reg)
            pts_reg = spline_reg.mishPoints
            u_reg = f.(pts_reg)

            setMishValues(spline_reg, u_reg)
            SBtransform!(spline_reg)
            SAtransform!(spline_reg)

            sp_gauss = SplineParameters(
                xmin=0, xmax=10, num_cells=num_cells,
                mubar=mb, quadrature=:gauss,
                l_q=2.0, BCL=CubicBSpline.R1T0, BCR=CubicBSpline.R1T0)
            spline_gauss = Spline1D(sp_gauss)
            pts_gauss = spline_gauss.mishPoints
            u_gauss = CubicBSpline.SItransform(sp_gauss, spline_reg.a, pts_gauss, zeros(length(pts_gauss)))

            u_exact = f.(pts_gauss)
            @test maximum(abs.(u_gauss .- u_exact)) < 5e-3
        end

        # ── T11: Solver with Regular Grid ────────────────────────────────────
        @testset "Solver with regular grid" begin
            gp = SpringsteelGridParameters(
                geometry="R", iMin=0.0, iMax=1.0, num_cells=20,
                mubar=3, quadrature=:regular,
                BCL=Dict("u" => CubicBSpline.R1T0),
                BCR=Dict("u" => CubicBSpline.R1T0),
                vars=Dict("u" => 1))
            grid = createGrid(gp)

            pts = solver_gridpoints(grid, "u")
            L = assemble_from_equation(grid, "u"; d_ii=1.0)
            f = -(pi^2) .* sin.(pi .* pts)

            prob = SpringsteelProblem(grid; operator=L, rhs=f)
            sol = solve(prob)

            u_exact = sin.(pi .* pts)
            max_err = maximum(abs.(sol.physical .- u_exact))
            @test sol.converged == true
            @test max_err < 0.1
        end

        # ── T12: Backward Compatibility of CubicBSpline.mubar ────────────────
        @testset "CubicBSpline.mubar constant" begin
            @test CubicBSpline.mubar == 3
            @test CubicBSpline.gaussweight == [5.0/18.0, 8.0/18.0, 5.0/18.0]
        end

        # ── Additional: Spline1D stores quadpoints/quadweights ────────────────
        @testset "Spline1D quadpoints/quadweights" begin
            for (mb, quad) in [(3, :gauss), (2, :gauss), (3, :regular), (1, :regular)]
                sp = SplineParameters(xmin=0, xmax=10, num_cells=10, mubar=mb, quadrature=quad)
                spline = Spline1D(sp)
                @test length(spline.quadpoints) == mb
                @test length(spline.quadweights) == mb
                @test sum(spline.quadweights) ≈ 1.0 atol=1e-14
                @test all(0.0 .< spline.quadpoints .< 1.0)
            end
        end

    end

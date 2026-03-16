using LinearAlgebra

@testset "R3X Boundary Conditions" begin

    # ── Phase 1: CubicBSpline core ──────────────────────────────────────────

    @testset "R3X constant definition" begin
        @test CubicBSpline.R3X == Dict("R3X" => 0)
        @test haskey(CubicBSpline.R3X, "R3X")
        @test CubicBSpline.R3X != CubicBSpline.R3
    end

    @testset "R3X gammaBC matches R3" begin
        sp_r3 = CubicBSpline.SplineParameters(
            xmin=0.0, xmax=10.0, num_cells=10,
            BCL=CubicBSpline.R3, BCR=CubicBSpline.R3)
        sp_r3x = CubicBSpline.SplineParameters(
            xmin=0.0, xmax=10.0, num_cells=10,
            BCL=CubicBSpline.R3X, BCR=CubicBSpline.R3X)
        g_r3 = CubicBSpline.calcGammaBC(sp_r3)
        g_r3x = CubicBSpline.calcGammaBC(sp_r3x)
        @test g_r3 == g_r3x
    end

    @testset "R3X mixed with other BCs" begin
        # R3X left, R1T0 right
        sp = CubicBSpline.SplineParameters(
            xmin=0.0, xmax=10.0, num_cells=10,
            BCL=CubicBSpline.R3X, BCR=CubicBSpline.R1T0)
        g = CubicBSpline.calcGammaBC(sp)
        # R3X left removes 3 coeffs, R1T0 right removes 1 → interior = 13 - 4 = 9
        @test size(g) == (9, 13)

        # R0 left, R3X right
        sp2 = CubicBSpline.SplineParameters(
            xmin=0.0, xmax=10.0, num_cells=10,
            BCL=CubicBSpline.R0, BCR=CubicBSpline.R3X)
        g2 = CubicBSpline.calcGammaBC(sp2)
        # R0 removes 0, R3X removes 3 → interior = 13 - 3 = 10
        @test size(g2) == (10, 13)
    end

    @testset "Spline1D has ahat field" begin
        sp = CubicBSpline.SplineParameters(
            xmin=0.0, xmax=10.0, num_cells=10,
            BCL=CubicBSpline.R3X, BCR=CubicBSpline.R3X)
        spline = CubicBSpline.Spline1D(sp)
        @test length(spline.ahat) == sp.bDim
        @test all(spline.ahat .== 0.0)
    end

    @testset "_has_r3x helper" begin
        sp_r3x = CubicBSpline.SplineParameters(
            xmin=0.0, xmax=1.0, num_cells=5,
            BCL=CubicBSpline.R3X, BCR=CubicBSpline.R0)
        @test CubicBSpline._has_r3x(sp_r3x) == true

        sp_r3 = CubicBSpline.SplineParameters(
            xmin=0.0, xmax=1.0, num_cells=5,
            BCL=CubicBSpline.R3, BCR=CubicBSpline.R3)
        @test CubicBSpline._has_r3x(sp_r3) == false

        sp_r0 = CubicBSpline.SplineParameters(
            xmin=0.0, xmax=1.0, num_cells=5,
            BCL=CubicBSpline.R0, BCR=CubicBSpline.R0)
        @test CubicBSpline._has_r3x(sp_r0) == false
    end

    @testset "_border_matrix" begin
        sp = CubicBSpline.SplineParameters(
            xmin=0.0, xmax=10.0, num_cells=10,
            BCL=CubicBSpline.R3X, BCR=CubicBSpline.R3X)

        # Left border matrix should be invertible
        M_left = CubicBSpline._border_matrix(sp, :left)
        @test size(M_left) == (3, 3)
        @test abs(det(M_left)) > 1e-10

        # Right border matrix should be invertible
        M_right = CubicBSpline._border_matrix(sp, :right)
        @test size(M_right) == (3, 3)
        @test abs(det(M_right)) > 1e-10

        # Row 1 of M should be basis function values at boundary (sum = 1)
        @test isapprox(sum(M_left[1, :]), 1.0, atol=1e-12)
        @test isapprox(sum(M_right[1, :]), 1.0, atol=1e-12)
    end

    @testset "set_ahat_r3x!" begin
        sp = CubicBSpline.SplineParameters(
            xmin=0.0, xmax=10.0, num_cells=10,
            BCL=CubicBSpline.R3X, BCR=CubicBSpline.R3X)
        spline = CubicBSpline.Spline1D(sp)

        # Set left boundary
        CubicBSpline.set_ahat_r3x!(spline, 1.0, 0.0, 0.0, :left)
        @test !all(spline.ahat .== 0.0)
        @test all(spline.ahat[4:end-3] .== 0.0)  # only border coeffs nonzero

        # Verify the border coefficients reproduce the boundary values
        M = CubicBSpline._border_matrix(sp, :left)
        @test isapprox(M * spline.ahat[1:3], [1.0, 0.0, 0.0], atol=1e-12)

        # Set right boundary too
        CubicBSpline.set_ahat_r3x!(spline, 2.0, 0.5, 0.0, :right)
        M_r = CubicBSpline._border_matrix(sp, :right)
        @test isapprox(M_r * spline.ahat[end-2:end], [2.0, 0.5, 0.0], atol=1e-12)

        # Invalid side
        @test_throws ArgumentError CubicBSpline.set_ahat_r3x!(spline, 0.0, 0.0, 0.0, :top)
    end

    @testset "R3X with zero values matches R3" begin
        nc = 10
        sp_r3 = CubicBSpline.SplineParameters(
            xmin=0.0, xmax=10.0, num_cells=nc,
            BCL=CubicBSpline.R3, BCR=CubicBSpline.R3)
        sp_r3x = CubicBSpline.SplineParameters(
            xmin=0.0, xmax=10.0, num_cells=nc,
            BCL=CubicBSpline.R3X, BCR=CubicBSpline.R3X)

        spline_r3 = CubicBSpline.Spline1D(sp_r3)
        spline_r3x = CubicBSpline.Spline1D(sp_r3x)

        # ahat is zeros → R3X should behave identically to R3
        # Set some test data
        for i in 1:sp_r3.mishDim
            x = spline_r3.mishPoints[i]
            val = sin(2π * x / 10.0)
            spline_r3.uMish[i] = val
            spline_r3x.uMish[i] = val
        end

        CubicBSpline.SBtransform!(spline_r3)
        CubicBSpline.SAtransform!(spline_r3)
        CubicBSpline.SBtransform!(spline_r3x)
        CubicBSpline.SAtransform!(spline_r3x)

        @test isapprox(spline_r3.a, spline_r3x.a, atol=1e-12)
    end

    @testset "R3X round-trip 1D with known boundary values" begin
        nc = 20
        sp = CubicBSpline.SplineParameters(
            xmin=0.0, xmax=10.0, num_cells=nc,
            BCL=CubicBSpline.R3X, BCR=CubicBSpline.R3X)
        spline = CubicBSpline.Spline1D(sp)

        # Analytic function: f(x) = sin(2πx/10) + 1
        f(x) = sin(2π * x / 10.0) + 1.0
        fp(x) = (2π/10.0) * cos(2π * x / 10.0)
        fpp(x) = -(2π/10.0)^2 * sin(2π * x / 10.0)

        # Set boundary values from analytic function
        CubicBSpline.set_ahat_r3x!(spline, f(0.0), fp(0.0), fpp(0.0), :left)
        CubicBSpline.set_ahat_r3x!(spline, f(10.0), fp(10.0), fpp(10.0), :right)

        # Set physical data
        for i in 1:sp.mishDim
            spline.uMish[i] = f(spline.mishPoints[i])
        end

        # Transform
        CubicBSpline.SBtransform!(spline)
        CubicBSpline.SAtransform!(spline)
        CubicBSpline.SItransform!(spline)

        # Check accuracy at mish points
        max_err = 0.0
        for i in 1:sp.mishDim
            err = abs(spline.uMish[i] - f(spline.mishPoints[i]))
            max_err = max(max_err, err)
        end
        @test max_err < 1e-4

        # Check boundary values match
        u_left = CubicBSpline.SItransform(sp, spline.a, sp.xmin, 0)
        u_right = CubicBSpline.SItransform(sp, spline.a, sp.xmax, 0)
        @test isapprox(u_left, f(0.0), atol=1e-4)
        @test isapprox(u_right, f(10.0), atol=1e-4)

        # Check first derivative at boundaries
        up_left = CubicBSpline.SItransform(sp, spline.a, sp.xmin, 1)
        up_right = CubicBSpline.SItransform(sp, spline.a, sp.xmax, 1)
        @test isapprox(up_left, fp(0.0), atol=1e-2)
        @test isapprox(up_right, fp(10.0), atol=1e-2)
    end

    @testset "R3X on one side, other BC on other side" begin
        nc = 20
        sp = CubicBSpline.SplineParameters(
            xmin=0.0, xmax=10.0, num_cells=nc,
            BCL=CubicBSpline.R3X, BCR=CubicBSpline.R1T0)
        spline = CubicBSpline.Spline1D(sp)

        # Function that is zero at right boundary (for R1T0)
        f(x) = sin(π * x / 10.0)
        fp(x) = (π/10.0) * cos(π * x / 10.0)
        fpp(x) = -(π/10.0)^2 * sin(π * x / 10.0)

        # Set R3X on left
        CubicBSpline.set_ahat_r3x!(spline, f(0.0), fp(0.0), fpp(0.0), :left)

        for i in 1:sp.mishDim
            spline.uMish[i] = f(spline.mishPoints[i])
        end

        CubicBSpline.SBtransform!(spline)
        CubicBSpline.SAtransform!(spline)
        CubicBSpline.SItransform!(spline)

        max_err = maximum(abs.(spline.uMish .- f.(spline.mishPoints)))
        @test max_err < 1e-4
    end

    # ── Phase 2: Solver ─────────────────────────────────────────────────────

    @testset "R3X solver 1D Poisson" begin
        # u'' = f with u(0) = 1, u(L) = 2
        # Analytic: u(x) = 1 + x/L (for f = 0)
        L_domain = 10.0
        nc = 20
        gp = SpringsteelGridParameters(
            geometry="R",
            iMin=0.0, iMax=L_domain,
            num_cells=nc,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))
        grid = createGrid(gp)

        # Set boundary values: u(0)=1, u'(0)=1/L, u''(0)=0
        #                      u(L)=2, u'(L)=1/L, u''(L)=0
        set_boundary_values!(grid, :left, "u", 1.0, 1.0/L_domain, 0.0)
        set_boundary_values!(grid, :right, "u", 2.0, 1.0/L_domain, 0.0)

        # Assemble u'' operator
        L_op = assemble_from_equation(grid, "u"; d_ii=1.0)
        # RHS = 0
        f = zeros(size(L_op, 1))

        prob = SpringsteelProblem(grid; operator=L_op, rhs=f,
                                  parameters=Dict{String,Any}("var" => "u"))
        sol = solve(prob)
        @test sol.converged

        # Check against analytic solution u(x) = 1 + x/L
        pts = solver_gridpoints(grid, "u")
        analytic = 1.0 .+ pts ./ L_domain
        @test isapprox(sol.physical, analytic, atol=1e-4)
    end

    @testset "R3X solver 1D with nonzero RHS" begin
        # u'' = -π²sin(πx) with u(0) = 0, u(1) = 0
        # Analytic: u(x) = sin(πx)
        nc = 20
        gp = SpringsteelGridParameters(
            geometry="R",
            iMin=0.0, iMax=1.0,
            num_cells=nc,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))
        grid = createGrid(gp)

        analytic(x) = sin(π * x)
        analytic_d1(x) = π * cos(π * x)
        analytic_d2(x) = -π^2 * sin(π * x)

        set_boundary_values!(grid, :left, "u",
            analytic(0.0), analytic_d1(0.0), analytic_d2(0.0))
        set_boundary_values!(grid, :right, "u",
            analytic(1.0), analytic_d1(1.0), analytic_d2(1.0))

        L_op = assemble_from_equation(grid, "u"; d_ii=1.0)
        pts = solver_gridpoints(grid, "u")
        f = -π^2 .* sin.(π .* pts)

        prob = SpringsteelProblem(grid; operator=L_op, rhs=f,
                                  parameters=Dict{String,Any}("var" => "u"))
        sol = solve(prob)
        @test sol.converged
        @test isapprox(sol.physical, analytic.(pts), atol=1e-3)
    end

    @testset "R3X solver 1D with R3X left, R1T0 right" begin
        # u'' = 0 with u(0)=5, u(L)=0 (Dirichlet)
        # Analytic: u(x) = 5(1 - x/L)
        L_domain = 10.0
        nc = 20
        gp = SpringsteelGridParameters(
            geometry="R",
            iMin=0.0, iMax=L_domain,
            num_cells=nc,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R1T0),
            vars=Dict("u" => 1))
        grid = createGrid(gp)

        set_boundary_values!(grid, :left, "u", 5.0, -5.0/L_domain, 0.0)

        L_op = assemble_from_equation(grid, "u"; d_ii=1.0)
        f = zeros(size(L_op, 1))

        prob = SpringsteelProblem(grid; operator=L_op, rhs=f,
                                  parameters=Dict{String,Any}("var" => "u"))
        sol = solve(prob)
        @test sol.converged

        pts = solver_gridpoints(grid, "u")
        analytic = 5.0 .* (1.0 .- pts ./ L_domain)
        @test isapprox(sol.physical, analytic, atol=1e-3)
    end

    # ── Phase 3: Grid-level interface ────────────────────────────────────────

    @testset "set_boundary_values! 1D R grid" begin
        gp = SpringsteelGridParameters(
            geometry="R",
            iMin=0.0, iMax=10.0,
            num_cells=20,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))
        grid = createGrid(gp)

        set_boundary_values!(grid, :left, "u", 1.0, 0.0, 0.0)
        set_boundary_values!(grid, :right, "u", 2.0, 0.0, 0.0)

        # Verify ahat was set on the spline
        spline = grid.ibasis.data[1, 1]
        @test !all(spline.ahat .== 0.0)
    end

    @testset "set_boundary_values! 2D RR grid" begin
        nc_j = 5
        jDim = nc_j * 3
        gp = SpringsteelGridParameters(
            geometry="RR",
            iMin=0.0, iMax=10.0,
            num_cells=20,
            jMin=0.0, jMax=5.0,
            jDim=jDim,
            b_jDim=nc_j + 3,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R3X),
            BCU=Dict("u" => CubicBSpline.R0),
            BCD=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        grid = createGrid(gp)

        # Uniform BC along j-boundary
        u0 = fill(3.0, jDim)
        u1 = fill(0.0, jDim)
        u2 = fill(0.0, jDim)

        set_boundary_values!(grid, :left, "u", u0, u1, u2)

        # Verify ahat was set on at least the first i-spline
        @test !all(grid.ibasis.data[1, 1].ahat .== 0.0)
    end

    @testset "set_boundary_values! 2D RZ grid" begin
        gp = SpringsteelGridParameters(
            geometry="RZ",
            iMin=0.0, iMax=10.0,
            num_cells=20,
            kMin=0.0, kMax=5.0,
            kDim=10,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R3X),
            BCB=Dict("u" => Chebyshev.R0),
            BCT=Dict("u" => Chebyshev.R0),
            vars=Dict("u" => 1))
        grid = createGrid(gp)

        kDim = gp.kDim
        u0 = fill(1.0, kDim)
        u1 = fill(0.0, kDim)
        u2 = fill(0.0, kDim)

        set_boundary_values!(grid, :left, "u", u0, u1, u2)

        # Verify ahat was set
        @test !all(grid.ibasis.data[1, 1].ahat .== 0.0)
    end

    @testset "Multi-variable R3X" begin
        gp = SpringsteelGridParameters(
            geometry="R",
            iMin=0.0, iMax=10.0,
            num_cells=20,
            BCL=Dict("u" => CubicBSpline.R3X, "v" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R3X, "v" => CubicBSpline.R0),
            vars=Dict("u" => 1, "v" => 2))
        grid = createGrid(gp)

        # Set R3X on u only
        set_boundary_values!(grid, :left, "u", 1.0, 0.0, 0.0)
        set_boundary_values!(grid, :right, "u", 2.0, 0.0, 0.0)

        # u should have ahat set
        @test !all(grid.ibasis.data[1, 1].ahat .== 0.0)
        # v should have ahat = 0 (R0, not R3X)
        @test all(grid.ibasis.data[1, 2].ahat .== 0.0)
    end

end

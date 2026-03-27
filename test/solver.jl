using LinearAlgebra

@testset "Solver Tests" begin

    # ─────────────────────────────────────────────────────────────────────────
    # Operator Matrix Assembly
    # ─────────────────────────────────────────────────────────────────────────

    @testset "Operator Matrix Assembly" begin

        @testset "1D operator_matrix for R_Grid (Spline)" begin
            gp = SpringsteelGridParameters(
                geometry = "R",
                iMin = 0.0, iMax = 1.0, num_cells = 10,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            M0 = operator_matrix(grid, :i, 0, "u")
            M1 = operator_matrix(grid, :i, 1, "u")
            M2 = operator_matrix(grid, :i, 2, "u")
            @test size(M0) == (gp.iDim, gp.b_iDim)
            @test size(M1) == (gp.iDim, gp.b_iDim)
            @test size(M2) == (gp.iDim, gp.b_iDim)
        end

        @testset "1D operator_matrix for Z_Grid (Chebyshev)" begin
            gp = SpringsteelGridParameters(
                geometry = "Z",
                iMin = 0.0, iMax = 1.0, iDim = 20, b_iDim = 20,
                BCL = Dict("u" => Chebyshev.R1T0),
                BCR = Dict("u" => Chebyshev.R1T0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            M0 = operator_matrix(grid, :i, 0, "u")
            M2 = operator_matrix(grid, :i, 2, "u")
            @test size(M0) == (20, 20)
            @test size(M2) == (20, 20)
        end

        @testset "2D operator_matrix for RZ_Grid" begin
            gp = SpringsteelGridParameters(
                geometry = "RZ",
                iMin = 0.0, iMax = 1.0, num_cells = 5,
                kMin = 0.0, kMax = 1.0, kDim = 8, b_kDim = 8,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCB = Dict("u" => Chebyshev.R0),
                BCT = Dict("u" => Chebyshev.R0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            Mi = operator_matrix(grid, :i, 0, "u")
            Mk = operator_matrix(grid, :k, 0, "u")
            @test size(Mi) == (gp.iDim, gp.b_iDim)
            @test size(Mk) == (gp.kDim, gp.b_kDim)
        end

        @testset "assemble_operator identity (all 0th order)" begin
            gp = SpringsteelGridParameters(
                geometry = "R",
                iMin = 0.0, iMax = 1.0, num_cells = 5,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            terms = [OperatorTerm(0, 0, 0, 1.0)]
            L = assemble_operator(grid, terms, "u")
            M0 = operator_matrix(grid, :i, 0, "u")
            @test L ≈ M0
        end

        @testset "assemble_operator 1D Laplacian" begin
            gp = SpringsteelGridParameters(
                geometry = "R",
                iMin = 0.0, iMax = 1.0, num_cells = 5,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            terms = [OperatorTerm(2, 0, 0, 1.0)]
            L = assemble_operator(grid, terms, "u")
            M2 = operator_matrix(grid, :i, 2, "u")
            @test L ≈ M2
        end

        @testset "assemble_operator 2D Laplacian dimensions" begin
            gp = SpringsteelGridParameters(
                geometry = "RZ",
                iMin = 0.0, iMax = 1.0, num_cells = 5,
                kMin = 0.0, kMax = 1.0, kDim = 8, b_kDim = 8,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCB = Dict("u" => Chebyshev.R0),
                BCT = Dict("u" => Chebyshev.R0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            terms = [
                OperatorTerm(2, 0, 0, 1.0),
                OperatorTerm(0, 0, 2, 1.0),
            ]
            L = assemble_operator(grid, terms, "u")
            n_phys = gp.iDim * gp.kDim
            n_spec = gp.b_iDim * gp.b_kDim
            @test size(L) == (n_phys, n_spec)
        end

        @testset "assemble_operator scalar coefficient" begin
            gp = SpringsteelGridParameters(
                geometry = "R",
                iMin = 0.0, iMax = 1.0, num_cells = 5,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            terms_1 = [OperatorTerm(2, 0, 0, 1.0)]
            terms_3 = [OperatorTerm(2, 0, 0, 3.0)]
            L1 = assemble_operator(grid, terms_1, "u")
            L3 = assemble_operator(grid, terms_3, "u")
            @test L3 ≈ 3.0 * L1
        end

        @testset "assemble_operator spatially varying coefficient" begin
            gp = SpringsteelGridParameters(
                geometry = "R",
                iMin = 0.0, iMax = 1.0, num_cells = 5,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            M2 = operator_matrix(grid, :i, 2, "u")
            coeff = ones(gp.iDim) .* 2.0
            terms = [OperatorTerm(2, 0, 0, coeff)]
            L = assemble_operator(grid, terms, "u")
            @test L ≈ Diagonal(coeff) * M2
        end

    end  # Operator Matrix Assembly

    # ─────────────────────────────────────────────────────────────────────────
    # SpringsteelProblem and Linear Solver
    # ─────────────────────────────────────────────────────────────────────────

    @testset "Linear Solver" begin

        @testset "SpringsteelProblem constructor" begin
            gp = SpringsteelGridParameters(
                geometry = "R",
                iMin = 0.0, iMax = 1.0, num_cells = 5,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            M = operator_matrix(grid, :i, 0, "u")
            f = ones(gp.iDim)
            prob = SpringsteelProblem(grid; operator=M, rhs=f)
            @test prob.operator === M
            @test prob.rhs === f
            @test prob.backend isa LocalLinearBackend
            @test prob.cost === nothing
        end

        @testset "solve() 1D Chebyshev Poisson" begin
            # Solve u'' = f where u(x) = sin(πx), f = -π²sin(πx)
            # Domain [0, 1], Dirichlet BCs u(0) = u(1) = 0
            N = 25
            gp = SpringsteelGridParameters(
                geometry = "Z",
                iMin = 0.0, iMax = 1.0, iDim = N, b_iDim = N,
                BCL = Dict("u" => Chebyshev.R1T0),
                BCR = Dict("u" => Chebyshev.R1T0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)
            pts = solver_gridpoints(grid, "u")

            # Assemble Laplacian operator
            L = assemble_from_equation(grid, "u"; d_ii=1.0)
            f = -π^2 .* sin.(π .* pts)

            prob = SpringsteelProblem(grid; operator=L, rhs=f)
            sol = solve(prob)

            u_analytic = sin.(π .* pts)
            @test sol.converged == true
            @test maximum(abs.(sol.physical .- u_analytic)) < 1e-4
        end

        @testset "solve() 1D Chebyshev BVP matches Chebyshev.bvp" begin
            # General 2nd-order ODE: u'' + u' + u = f
            N = 25
            gp = SpringsteelGridParameters(
                geometry = "Z",
                iMin = 0.0, iMax = 1.0, iDim = N, b_iDim = N,
                BCL = Dict("u" => Chebyshev.R1T0),
                BCR = Dict("u" => Chebyshev.R1T0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)
            pts = solver_gridpoints(grid, "u")

            # u(x) = sin(πx), u'(x) = πcos(πx), u''(x) = -π²sin(πx)
            # f = u'' + u' + u = -π²sin(πx) + πcos(πx) + sin(πx)
            f_vals = -π^2 .* sin.(π .* pts) .+ π .* cos.(π .* pts) .+ sin.(π .* pts)

            L = assemble_from_equation(grid, "u"; d_ii=1.0, d_i=1.0, d0=1.0)
            prob = SpringsteelProblem(grid; operator=L, rhs=f_vals)
            sol = solve(prob)

            u_analytic = sin.(π .* pts)
            @test sol.converged == true
            @test maximum(abs.(sol.physical .- u_analytic)) < 1e-3
        end

        @testset "solve() 1D CubicBSpline BVP" begin
            # Solve u'' = -4π²sin(2πx) on [0, 1] with u = sin(2πx)
            # Dirichlet BCs: u(0) = 0, u(1) = 0
            gp = SpringsteelGridParameters(
                geometry = "R",
                iMin = 0.0, iMax = 1.0, num_cells = 30,
                BCL = Dict("u" => CubicBSpline.R1T0),
                BCR = Dict("u" => CubicBSpline.R1T0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)
            pts = solver_gridpoints(grid, "u")

            L = assemble_from_equation(grid, "u"; d_ii=1.0)
            f = -(2π)^2 .* sin.(2π .* pts)

            prob = SpringsteelProblem(grid; operator=L, rhs=f)
            sol = solve(prob)

            u_analytic = sin.(2π .* pts)
            @test sol.converged == true
            @test maximum(abs.(sol.physical .- u_analytic)) < 0.05
        end

        @testset "converged flag is true for well-posed problem" begin
            N = 15
            gp = SpringsteelGridParameters(
                geometry = "Z",
                iMin = 0.0, iMax = 1.0, iDim = N, b_iDim = N,
                BCL = Dict("u" => Chebyshev.R1T0),
                BCR = Dict("u" => Chebyshev.R1T0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)
            pts = solver_gridpoints(grid, "u")

            L = assemble_from_equation(grid, "u"; d_ii=1.0)
            f = ones(N)
            prob = SpringsteelProblem(grid; operator=L, rhs=f)
            sol = solve(prob)
            @test sol.converged == true
        end

        @testset "solve() 1D Chebyshev mixed Dirichlet/Neumann BCs" begin
            # Solve u'' = f on [0,1] with u(0) = 0 (Dirichlet left), u'(1) = 0 (Neumann right)
            # Analytic solution: u(x) = sin(πx/2)
            # f = -(π/2)² sin(πx/2)
            # This test catches BCT/BCB row-swap bugs because the BC types differ.
            N = 25
            gp = SpringsteelGridParameters(
                geometry = "Z",
                iMin = 0.0, iMax = 1.0, iDim = N, b_iDim = N,
                BCL = Dict("u" => Chebyshev.R1T0),
                BCR = Dict("u" => Chebyshev.R1T1),
                vars = Dict("u" => 1))
            grid = createGrid(gp)
            pts = solver_gridpoints(grid, "u")

            L = assemble_from_equation(grid, "u"; d_ii=1.0)
            f = -(π/2)^2 .* sin.(π/2 .* pts)

            prob = SpringsteelProblem(grid; operator=L, rhs=f)
            sol = solve(prob)

            u_analytic = sin.(π/2 .* pts)
            @test sol.converged == true
            @test maximum(abs.(sol.physical .- u_analytic)) < 1e-3
        end

        @testset "repeated solve with different RHS" begin
            N = 20
            gp = SpringsteelGridParameters(
                geometry = "Z",
                iMin = 0.0, iMax = 1.0, iDim = N, b_iDim = N,
                BCL = Dict("u" => Chebyshev.R1T0),
                BCR = Dict("u" => Chebyshev.R1T0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)
            pts = solver_gridpoints(grid, "u")

            L = assemble_from_equation(grid, "u"; d_ii=1.0)

            # Solve with first RHS
            f1 = -π^2 .* sin.(π .* pts)
            params = Dict{String, Any}()
            prob1 = SpringsteelProblem(grid; operator=L, rhs=f1, parameters=params)
            sol1 = solve(prob1)
            @test sol1.converged == true

            # Solve again with different RHS (reuses factorisation)
            f2 = -(2π)^2 .* sin.(2π .* pts)
            prob2 = SpringsteelProblem(grid; operator=L, rhs=f2, parameters=params)
            sol2 = solve(prob2)
            @test sol2.converged == true

            # Verify factorisation was cached
            @test haskey(params, "_factorisation")
        end

    end  # Linear Solver

    # ─────────────────────────────────────────────────────────────────────────
    # Symbolic Operator Interface (assemble_from_equation)
    # ─────────────────────────────────────────────────────────────────────────

    @testset "assemble_from_equation" begin

        @testset "1D Poisson matches manual assemble_operator" begin
            gp = SpringsteelGridParameters(
                geometry = "R",
                iMin = 0.0, iMax = 1.0, num_cells = 5,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            L_manual = assemble_operator(grid, [OperatorTerm(2, 0, 0, 1.0)], "u")
            L_eq = assemble_from_equation(grid, "u"; d_ii=1.0)
            @test L_eq ≈ L_manual
        end

        @testset "general 2nd-order ODE" begin
            gp = SpringsteelGridParameters(
                geometry = "Z",
                iMin = 0.0, iMax = 1.0, iDim = 15, b_iDim = 15,
                BCL = Dict("u" => Chebyshev.R1T0),
                BCR = Dict("u" => Chebyshev.R1T0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            L_eq = assemble_from_equation(grid, "u"; d_ii=2.0, d_i=3.0, d0=1.0)
            M0 = operator_matrix(grid, :i, 0, "u")
            M1 = operator_matrix(grid, :i, 1, "u")
            M2 = operator_matrix(grid, :i, 2, "u")
            L_manual = 2.0 * M2 + 3.0 * M1 + 1.0 * M0
            @test L_eq ≈ L_manual
        end

        @testset "2D Laplacian on RZ grid" begin
            gp = SpringsteelGridParameters(
                geometry = "RZ",
                iMin = 0.0, iMax = 1.0, num_cells = 5,
                kMin = 0.0, kMax = 1.0, kDim = 8, b_kDim = 8,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCB = Dict("u" => Chebyshev.R0),
                BCT = Dict("u" => Chebyshev.R0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            L = assemble_from_equation(grid, "u"; d_ii=1.0, d_kk=1.0)
            n_phys = gp.iDim * gp.kDim
            n_spec = gp.b_iDim * gp.b_kDim
            @test size(L) == (n_phys, n_spec)
        end

        @testset "variable-coefficient operator" begin
            gp = SpringsteelGridParameters(
                geometry = "R",
                iMin = 0.0, iMax = 1.0, num_cells = 5,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            coeff = collect(1.0:gp.iDim) ./ gp.iDim
            L = assemble_from_equation(grid, "u"; d0=coeff)
            M0 = operator_matrix(grid, :i, 0, "u")
            @test L ≈ Diagonal(coeff) * M0
        end

        @testset "error for no keywords" begin
            gp = SpringsteelGridParameters(
                geometry = "R",
                iMin = 0.0, iMax = 1.0, num_cells = 5,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            @test_throws ArgumentError assemble_from_equation(grid, "u")
        end

        @testset "end-to-end: assemble_from_equation → solve" begin
            N = 25
            gp = SpringsteelGridParameters(
                geometry = "Z",
                iMin = 0.0, iMax = 1.0, iDim = N, b_iDim = N,
                BCL = Dict("u" => Chebyshev.R1T0),
                BCR = Dict("u" => Chebyshev.R1T0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)
            pts = solver_gridpoints(grid, "u")

            L = assemble_from_equation(grid, "u"; d_ii=1.0)
            f = -π^2 .* sin.(π .* pts)
            prob = SpringsteelProblem(grid; operator=L, rhs=f)
            sol = solve(prob)

            @test sol.converged == true
            @test maximum(abs.(sol.physical .- sin.(π .* pts))) < 1e-4
        end

        # ─────────────────────────────────────────────────────────────────
        # Mixed-derivative keywords
        # ─────────────────────────────────────────────────────────────────

        @testset "d_ij matches manual OperatorTerm on ZZ grid" begin
            Ni = 12; Nj = 10
            gp = SpringsteelGridParameters(
                geometry = "ZZ",
                iMin = 0.0, iMax = 1.0, iDim = Ni, b_iDim = Ni,
                jMin = 0.0, jMax = 1.0, jDim = Nj, b_jDim = Nj,
                BCL = Dict("u" => Chebyshev.R1T0),
                BCR = Dict("u" => Chebyshev.R1T0),
                BCU = Dict("u" => Chebyshev.R1T0),
                BCD = Dict("u" => Chebyshev.R1T0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            L_eq = assemble_from_equation(grid, "u"; d_ij=1.0)
            L_manual = assemble_operator(grid, [OperatorTerm(1, 1, 0, 1.0)], "u")
            @test L_eq ≈ L_manual
        end

        @testset "d_ik matches manual OperatorTerm on RZ grid" begin
            gp = SpringsteelGridParameters(
                geometry = "RZ",
                iMin = 0.0, iMax = 1.0, num_cells = 5,
                kMin = 0.0, kMax = 1.0, kDim = 8, b_kDim = 8,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCB = Dict("u" => Chebyshev.R0),
                BCT = Dict("u" => Chebyshev.R0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            L_eq = assemble_from_equation(grid, "u"; d_ik=1.0)
            L_manual = assemble_operator(grid, [OperatorTerm(1, 0, 1, 1.0)], "u")
            @test L_eq ≈ L_manual
        end

        @testset "d_jk matches manual OperatorTerm on ZZ grid" begin
            # Use ZZZ to have all 3 dims active, but ZZ only has i and j.
            # For d_jk we need j and k active. Use a 3D grid.
            Ni = 6; Nj = 5; Nk = 4
            gp = SpringsteelGridParameters(
                geometry = "ZZZ",
                iMin = 0.0, iMax = 1.0, iDim = Ni, b_iDim = Ni,
                jMin = 0.0, jMax = 1.0, jDim = Nj, b_jDim = Nj,
                kMin = 0.0, kMax = 1.0, kDim = Nk, b_kDim = Nk,
                BCL = Dict("u" => Chebyshev.R0),
                BCR = Dict("u" => Chebyshev.R0),
                BCU = Dict("u" => Chebyshev.R0),
                BCD = Dict("u" => Chebyshev.R0),
                BCB = Dict("u" => Chebyshev.R0),
                BCT = Dict("u" => Chebyshev.R0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            L_eq = assemble_from_equation(grid, "u"; d_jk=1.0)
            L_manual = assemble_operator(grid, [OperatorTerm(0, 1, 1, 1.0)], "u")
            @test L_eq ≈ L_manual
        end

        @testset "d_ijk matches manual OperatorTerm on ZZZ grid" begin
            Ni = 6; Nj = 5; Nk = 4
            gp = SpringsteelGridParameters(
                geometry = "ZZZ",
                iMin = 0.0, iMax = 1.0, iDim = Ni, b_iDim = Ni,
                jMin = 0.0, jMax = 1.0, jDim = Nj, b_jDim = Nj,
                kMin = 0.0, kMax = 1.0, kDim = Nk, b_kDim = Nk,
                BCL = Dict("u" => Chebyshev.R0),
                BCR = Dict("u" => Chebyshev.R0),
                BCU = Dict("u" => Chebyshev.R0),
                BCD = Dict("u" => Chebyshev.R0),
                BCB = Dict("u" => Chebyshev.R0),
                BCT = Dict("u" => Chebyshev.R0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            L_eq = assemble_from_equation(grid, "u"; d_ijk=1.0)
            L_manual = assemble_operator(grid, [OperatorTerm(1, 1, 1, 1.0)], "u")
            @test L_eq ≈ L_manual
        end

        @testset "combined pure + mixed: d_ii + d_ij" begin
            Ni = 12; Nj = 10
            gp = SpringsteelGridParameters(
                geometry = "ZZ",
                iMin = 0.0, iMax = 1.0, iDim = Ni, b_iDim = Ni,
                jMin = 0.0, jMax = 1.0, jDim = Nj, b_jDim = Nj,
                BCL = Dict("u" => Chebyshev.R1T0),
                BCR = Dict("u" => Chebyshev.R1T0),
                BCU = Dict("u" => Chebyshev.R1T0),
                BCD = Dict("u" => Chebyshev.R1T0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            L_eq = assemble_from_equation(grid, "u"; d_ii=1.0, d_ij=2.0)
            L_manual = assemble_operator(grid,
                [OperatorTerm(2, 0, 0, 1.0), OperatorTerm(1, 1, 0, 2.0)], "u")
            @test L_eq ≈ L_manual
        end

        @testset "variable-coefficient mixed derivative" begin
            Ni = 12; Nj = 10
            gp = SpringsteelGridParameters(
                geometry = "ZZ",
                iMin = 0.0, iMax = 1.0, iDim = Ni, b_iDim = Ni,
                jMin = 0.0, jMax = 1.0, jDim = Nj, b_jDim = Nj,
                BCL = Dict("u" => Chebyshev.R1T0),
                BCR = Dict("u" => Chebyshev.R1T0),
                BCU = Dict("u" => Chebyshev.R1T0),
                BCD = Dict("u" => Chebyshev.R1T0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            coeff = ones(Ni * Nj) .* 3.5
            L_eq = assemble_from_equation(grid, "u"; d_ij=coeff)
            L_manual = assemble_operator(grid, [OperatorTerm(1, 1, 0, coeff)], "u")
            @test L_eq ≈ L_manual
        end

        @testset "solve with mixed derivative: 2D ZZ" begin
            # Solve ∂²u/∂x² + 2∂²u/∂x∂y + ∂²u/∂y² = f on [0,1]×[0,1]
            # This is (∂/∂x + ∂/∂y)² u = f
            # Analytic: u(x,y) = sin(πx)sin(πy)
            # ∂²u/∂x² = -π²sin(πx)sin(πy)
            # ∂²u/∂x∂y = π²cos(πx)cos(πy)
            # ∂²u/∂y² = -π²sin(πx)sin(πy)
            # f = -2π²sin(πx)sin(πy) + 2π²cos(πx)cos(πy)
            Ni = 20; Nj = 20
            gp = SpringsteelGridParameters(
                geometry = "ZZ",
                iMin = 0.0, iMax = 1.0, iDim = Ni, b_iDim = Ni,
                jMin = 0.0, jMax = 1.0, jDim = Nj, b_jDim = Nj,
                BCL = Dict("u" => Chebyshev.R1T0),
                BCR = Dict("u" => Chebyshev.R1T0),
                BCU = Dict("u" => Chebyshev.R1T0),
                BCD = Dict("u" => Chebyshev.R1T0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            pts_i = solver_gridpoints(grid, "u")
            obj_j = grid.jbasis.data[grid.params.vars["u"]]
            pts_j = Springsteel.gridpoints(obj_j)

            f = zeros(Ni * Nj)
            u_analytic = zeros(Ni * Nj)
            for i in 1:Ni
                for j in 1:Nj
                    idx = (i-1)*Nj + j
                    xi = pts_i[i]; yj = pts_j[j]
                    u_analytic[idx] = sin(π*xi) * sin(π*yj)
                    f[idx] = -2π^2*sin(π*xi)*sin(π*yj) + 2π^2*cos(π*xi)*cos(π*yj)
                end
            end

            L = assemble_from_equation(grid, "u"; d_ii=1.0, d_ij=2.0, d_jj=1.0)
            prob = SpringsteelProblem(grid; operator=L, rhs=f)
            sol = solve(prob)

            @test sol.converged == true
            @test maximum(abs.(sol.physical .- u_analytic)) < 0.01
        end

        @testset "all higher-order mixed keywords match OperatorTerm" begin
            # Use a small 3D ZZZ grid so all three dimensions are active
            Ni = 5; Nj = 4; Nk = 3
            gp = SpringsteelGridParameters(
                geometry = "ZZZ",
                iMin = 0.0, iMax = 1.0, iDim = Ni, b_iDim = Ni,
                jMin = 0.0, jMax = 1.0, jDim = Nj, b_jDim = Nj,
                kMin = 0.0, kMax = 1.0, kDim = Nk, b_kDim = Nk,
                BCL = Dict("u" => Chebyshev.R0),
                BCR = Dict("u" => Chebyshev.R0),
                BCU = Dict("u" => Chebyshev.R0),
                BCD = Dict("u" => Chebyshev.R0),
                BCB = Dict("u" => Chebyshev.R0),
                BCT = Dict("u" => Chebyshev.R0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            # Map keyword name → (i_order, j_order, k_order)
            kw_map = [
                # 3rd-order mixed
                (:d_iij,    (2,1,0)),
                (:d_iik,    (2,0,1)),
                (:d_ijj,    (1,2,0)),
                (:d_jjk,    (0,2,1)),
                (:d_ikk,    (1,0,2)),
                (:d_jkk,    (0,1,2)),
                # 4th-order mixed
                (:d_iijj,   (2,2,0)),
                (:d_iikk,   (2,0,2)),
                (:d_jjkk,   (0,2,2)),
                (:d_iijk,   (2,1,1)),
                (:d_ijjk,   (1,2,1)),
                (:d_ijkk,   (1,1,2)),
                # 5th-order mixed
                (:d_iijjk,  (2,2,1)),
                (:d_iijkk,  (2,1,2)),
                (:d_ijjkk,  (1,2,2)),
                # 6th-order mixed
                (:d_iijjkk, (2,2,2)),
            ]

            for (kw, (io, jo, ko)) in kw_map
                L_eq = assemble_from_equation(grid, "u"; Dict(kw => 1.0)...)
                L_manual = assemble_operator(grid, [OperatorTerm(io, jo, ko, 1.0)], "u")
                @test L_eq ≈ L_manual
            end
        end

        @testset "d_ijj combined with d_ii on ZZ grid" begin
            Ni = 10; Nj = 8
            gp = SpringsteelGridParameters(
                geometry = "ZZ",
                iMin = 0.0, iMax = 1.0, iDim = Ni, b_iDim = Ni,
                jMin = 0.0, jMax = 1.0, jDim = Nj, b_jDim = Nj,
                BCL = Dict("u" => Chebyshev.R1T0),
                BCR = Dict("u" => Chebyshev.R1T0),
                BCU = Dict("u" => Chebyshev.R1T0),
                BCD = Dict("u" => Chebyshev.R1T0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            L_eq = assemble_from_equation(grid, "u"; d_ii=1.0, d_ijj=3.0)
            L_manual = assemble_operator(grid,
                [OperatorTerm(2, 0, 0, 1.0), OperatorTerm(1, 2, 0, 3.0)], "u")
            @test L_eq ≈ L_manual
        end

    end  # assemble_from_equation

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-D Linear Solver
    # ─────────────────────────────────────────────────────────────────────────

    @testset "Multi-D Linear Solver" begin

        @testset "2D Poisson on ZZ_Grid (Chebyshev x Chebyshev)" begin
            # Solve ∂²u/∂x² + ∂²u/∂y² = f on [0,1]×[0,1]
            # Analytic solution: u(x,y) = sin(πx)sin(πy)
            # f = -2π²sin(πx)sin(πy)
            Ni = 20; Nj = 20
            gp = SpringsteelGridParameters(
                geometry = "ZZ",
                iMin = 0.0, iMax = 1.0, iDim = Ni, b_iDim = Ni,
                jMin = 0.0, jMax = 1.0, jDim = Nj, b_jDim = Nj,
                BCL = Dict("u" => Chebyshev.R1T0),
                BCR = Dict("u" => Chebyshev.R1T0),
                BCU = Dict("u" => Chebyshev.R1T0),
                BCD = Dict("u" => Chebyshev.R1T0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            # Get gridpoints for each dimension
            pts_i = solver_gridpoints(grid, "u")
            var_idx = grid.params.vars["u"]
            obj_j = grid.jbasis.data[var_idx]
            pts_j = Springsteel.gridpoints(obj_j)

            # Build RHS: f[(i-1)*Nj + j] = -2π²sin(πx_i)sin(πy_j)
            f = zeros(Ni * Nj)
            for i in 1:Ni
                for j in 1:Nj
                    f[(i-1)*Nj + j] = -2π^2 * sin(π * pts_i[i]) * sin(π * pts_j[j])
                end
            end

            L = assemble_from_equation(grid, "u"; d_ii=1.0, d_jj=1.0)
            prob = SpringsteelProblem(grid; operator=L, rhs=f)
            sol = solve(prob)

            # Build analytic solution on the same ordering
            u_analytic = zeros(Ni * Nj)
            for i in 1:Ni
                for j in 1:Nj
                    u_analytic[(i-1)*Nj + j] = sin(π * pts_i[i]) * sin(π * pts_j[j])
                end
            end

            @test sol.converged == true
            @test maximum(abs.(sol.physical .- u_analytic)) < 0.01
        end

        @testset "2D Poisson on RZ_Grid (Spline x Chebyshev)" begin
            # Solve ∂²u/∂r² + ∂²u/∂z² = f on [0,1]×[0,1]
            # Analytic solution: u(r,z) = sin(πr)sin(πz)
            # f = -2π²sin(πr)sin(πz)
            num_cells_i = 15
            Nk = 15
            gp = SpringsteelGridParameters(
                geometry = "RZ",
                iMin = 0.0, iMax = 1.0, num_cells = num_cells_i,
                kMin = 0.0, kMax = 1.0, kDim = Nk, b_kDim = Nk,
                BCL = Dict("u" => CubicBSpline.R1T0),
                BCR = Dict("u" => CubicBSpline.R1T0),
                BCB = Dict("u" => Chebyshev.R1T0),
                BCT = Dict("u" => Chebyshev.R1T0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            # Get gridpoints
            pts_i = solver_gridpoints(grid, "u")
            var_idx = grid.params.vars["u"]
            obj_k = grid.kbasis.data[var_idx]
            pts_k = Springsteel.gridpoints(obj_k)

            iDim = grid.params.iDim
            kDim = grid.params.kDim

            # Build RHS: f[(i-1)*kDim + k]
            f = zeros(iDim * kDim)
            for i in 1:iDim
                for k in 1:kDim
                    f[(i-1)*kDim + k] = -2π^2 * sin(π * pts_i[i]) * sin(π * pts_k[k])
                end
            end

            L = assemble_from_equation(grid, "u"; d_ii=1.0, d_kk=1.0)
            prob = SpringsteelProblem(grid; operator=L, rhs=f)
            sol = solve(prob)

            u_analytic = zeros(iDim * kDim)
            for i in 1:iDim
                for k in 1:kDim
                    u_analytic[(i-1)*kDim + k] = sin(π * pts_i[i]) * sin(π * pts_k[k])
                end
            end

            @test sol.converged == true
            @test maximum(abs.(sol.physical .- u_analytic)) < 0.1
        end

        @testset "2D Helmholtz on ZZ_Grid" begin
            # Solve ∇²u + k²u = f where k²=1
            # Analytic solution: u = sin(πx)sin(πy)
            # f = (1 - 2π²)sin(πx)sin(πy)
            Ni = 20; Nj = 20
            gp = SpringsteelGridParameters(
                geometry = "ZZ",
                iMin = 0.0, iMax = 1.0, iDim = Ni, b_iDim = Ni,
                jMin = 0.0, jMax = 1.0, jDim = Nj, b_jDim = Nj,
                BCL = Dict("u" => Chebyshev.R1T0),
                BCR = Dict("u" => Chebyshev.R1T0),
                BCU = Dict("u" => Chebyshev.R1T0),
                BCD = Dict("u" => Chebyshev.R1T0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            pts_i = solver_gridpoints(grid, "u")
            var_idx = grid.params.vars["u"]
            obj_j = grid.jbasis.data[var_idx]
            pts_j = Springsteel.gridpoints(obj_j)

            f = zeros(Ni * Nj)
            for i in 1:Ni
                for j in 1:Nj
                    f[(i-1)*Nj + j] = (1.0 - 2π^2) * sin(π * pts_i[i]) * sin(π * pts_j[j])
                end
            end

            L = assemble_from_equation(grid, "u"; d_ii=1.0, d_jj=1.0, d0=1.0)
            prob = SpringsteelProblem(grid; operator=L, rhs=f)
            sol = solve(prob)

            u_analytic = zeros(Ni * Nj)
            for i in 1:Ni
                for j in 1:Nj
                    u_analytic[(i-1)*Nj + j] = sin(π * pts_i[i]) * sin(π * pts_j[j])
                end
            end

            @test sol.converged == true
            @test maximum(abs.(sol.physical .- u_analytic)) < 0.01
        end

        @testset "Spectral convergence test (1D Chebyshev)" begin
            # Solve u'' = -π²sin(πx) at two resolutions
            # Confirm error decreases significantly (exponential convergence)
            function chebyshev_poisson_error(N)
                gp = SpringsteelGridParameters(
                    geometry = "Z",
                    iMin = 0.0, iMax = 1.0, iDim = N, b_iDim = N,
                    BCL = Dict("u" => Chebyshev.R1T0),
                    BCR = Dict("u" => Chebyshev.R1T0),
                    vars = Dict("u" => 1))
                grid = createGrid(gp)
                pts = solver_gridpoints(grid, "u")
                L = assemble_from_equation(grid, "u"; d_ii=1.0)
                f = -π^2 .* sin.(π .* pts)
                prob = SpringsteelProblem(grid; operator=L, rhs=f)
                sol = solve(prob)
                u_analytic = sin.(π .* pts)
                return maximum(abs.(sol.physical .- u_analytic))
            end

            error_coarse = chebyshev_poisson_error(10)
            error_fine   = chebyshev_poisson_error(20)

            @test error_coarse / error_fine > 10
        end

        @testset "2D operator assembly dimensions (ZZ_Grid)" begin
            Ni = 12; Nj = 10
            gp = SpringsteelGridParameters(
                geometry = "ZZ",
                iMin = 0.0, iMax = 1.0, iDim = Ni, b_iDim = Ni,
                jMin = 0.0, jMax = 1.0, jDim = Nj, b_jDim = Nj,
                BCL = Dict("u" => Chebyshev.R1T0),
                BCR = Dict("u" => Chebyshev.R1T0),
                BCU = Dict("u" => Chebyshev.R1T0),
                BCD = Dict("u" => Chebyshev.R1T0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            L = assemble_from_equation(grid, "u"; d_ii=1.0, d_jj=1.0)
            @test size(L) == (Ni * Nj, Ni * Nj)
        end

        @testset "1D R_Grid higher resolution Poisson" begin
            # Solve u'' = -4π²sin(2πx) on [0,1] with 50 cells
            gp = SpringsteelGridParameters(
                geometry = "R",
                iMin = 0.0, iMax = 1.0, num_cells = 50,
                BCL = Dict("u" => CubicBSpline.R1T0),
                BCR = Dict("u" => CubicBSpline.R1T0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)
            pts = solver_gridpoints(grid, "u")

            L = assemble_from_equation(grid, "u"; d_ii=1.0)
            f = -(2π)^2 .* sin.(2π .* pts)

            prob = SpringsteelProblem(grid; operator=L, rhs=f)
            sol = solve(prob)

            u_analytic = sin.(2π .* pts)
            @test sol.converged == true
            @test maximum(abs.(sol.physical .- u_analytic)) < 0.01
        end

        # ─────────────────────────────────────────────────────────────────
        # Asymmetric BC tests — catch BCU/BCD swap bugs
        #
        # These tests use DIFFERENT BCs on the j-dimension boundaries
        # (Dirichlet on one side, Neumann on the other). If BCU/BCD are
        # swapped in the factory, the solution will satisfy the wrong BCs
        # and the accuracy test will fail with O(1) error.
        #
        # For RR grids, the mish points are Gaussian quadrature points
        # that do NOT include the boundary endpoints. Boundary values are
        # evaluated directly from the spectral coefficients using
        # CubicBSpline.basis() at the actual boundary point.
        # ─────────────────────────────────────────────────────────────────

        @testset "2D RR asymmetric j-BCs: Dirichlet bottom, Neumann top" begin
            # Solve ∂²u/∂x² + ∂²u/∂y² = f on [0,1]×[0,1]
            # BCs: u=0 at x=0,1 (Dirichlet); u=0 at y=0 (BCD, Dirichlet);
            #      ∂u/∂y=0 at y=1 (BCU, Neumann)
            # Analytic solution: u(x,y) = sin(πx) sin(πy/2)
            # u(x,0) = 0 ✓ (Dirichlet at bottom)
            # ∂u/∂y(x,1) = sin(πx)(π/2)cos(π/2) = 0 ✓ (Neumann at top)
            # f = -(π² + (π/2)²) sin(πx) sin(πy/2)
            num_cells = 20
            gp = SpringsteelGridParameters(
                geometry = "RR",
                num_cells = num_cells,
                iMin = 0.0, iMax = 1.0,
                jMin = 0.0, jMax = 1.0,
                BCL = Dict("u" => CubicBSpline.R1T0),
                BCR = Dict("u" => CubicBSpline.R1T0),
                BCD = Dict("u" => CubicBSpline.R1T0),
                BCU = Dict("u" => CubicBSpline.R1T1),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            iDim = grid.params.iDim
            jDim = grid.params.jDim
            pts_x = [grid.ibasis.data[1, 1].mishPoints[r] for r in 1:iDim]
            pts_y = [grid.jbasis.data[1, 1].mishPoints[l] for l in 1:jDim]

            rhs = zeros(iDim * jDim)
            u_analytic = zeros(iDim * jDim)
            for r in 1:iDim, l in 1:jDim
                idx = (r-1)*jDim + l
                rhs[idx] = -(π^2 + (π/2)^2) * sin(π * pts_x[r]) * sin(π/2 * pts_y[l])
                u_analytic[idx] = sin(π * pts_x[r]) * sin(π/2 * pts_y[l])
            end

            L = assemble_from_equation(grid, "u"; d_ii=1.0, d_jj=1.0)
            prob = SpringsteelProblem(grid; operator=L, rhs=rhs)
            sol = solve(prob)

            @test sol.converged == true
            @test maximum(abs.(sol.physical .- u_analytic)) < 0.05

            # Evaluate solution at actual boundary endpoints using B-spline basis
            sp_j = grid.jbasis.data[1, 1].params
            sp_i = grid.ibasis.data[1, 1].params
            Nspec_j = sp_j.num_cells + 3
            Nspec_i = sp_i.num_cells + 3

            # Evaluate u at jMin (y=0) for each i-basis function
            # by summing a_coeff * basis_i(x_r) * basis_j(y=0)
            for r in 1:iDim
                u_bottom = 0.0
                for si in -1:(sp_i.num_cells+1)
                    bi = CubicBSpline.basis(sp_i, si, pts_x[r], 0)
                    bi == 0.0 && continue
                    for sj in -1:(sp_j.num_cells+1)
                        bj = CubicBSpline.basis(sp_j, sj, gp.jMin, 0)
                        bj == 0.0 && continue
                        coeff_idx = (si+2-1)*Nspec_j + (sj+2)
                        u_bottom += sol.coefficients[coeff_idx] * bi * bj
                    end
                end
                @test abs(u_bottom) < 1e-10  # Dirichlet at jMin: exact zero
            end

            # Evaluate u at jMax (y=1) — should be NONZERO (Neumann BC)
            u_top_max = 0.0
            for r in 1:iDim
                u_top = 0.0
                for si in -1:(sp_i.num_cells+1)
                    bi = CubicBSpline.basis(sp_i, si, pts_x[r], 0)
                    bi == 0.0 && continue
                    for sj in -1:(sp_j.num_cells+1)
                        bj = CubicBSpline.basis(sp_j, sj, gp.jMax, 0)
                        bj == 0.0 && continue
                        coeff_idx = (si+2-1)*Nspec_j + (sj+2)
                        u_top += sol.coefficients[coeff_idx] * bi * bj
                    end
                end
                u_top_max = max(u_top_max, abs(u_top))
            end
            @test u_top_max > 0.1  # Neumann side: nonzero
        end

        @testset "2D RR asymmetric j-BCs: Neumann bottom, Dirichlet top" begin
            # Analytic solution: u(x,y) = sin(πx) cos(πy/2)
            # ∂u/∂y(x,0) = 0 ✓ (Neumann at bottom)
            # u(x,1) = 0 ✓ (Dirichlet at top)
            # f = -(π² + (π/2)²) sin(πx) cos(πy/2)
            num_cells = 20
            gp = SpringsteelGridParameters(
                geometry = "RR",
                num_cells = num_cells,
                iMin = 0.0, iMax = 1.0,
                jMin = 0.0, jMax = 1.0,
                BCL = Dict("u" => CubicBSpline.R1T0),
                BCR = Dict("u" => CubicBSpline.R1T0),
                BCD = Dict("u" => CubicBSpline.R1T1),
                BCU = Dict("u" => CubicBSpline.R1T0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            iDim = grid.params.iDim
            jDim = grid.params.jDim
            pts_x = [grid.ibasis.data[1, 1].mishPoints[r] for r in 1:iDim]
            pts_y = [grid.jbasis.data[1, 1].mishPoints[l] for l in 1:jDim]

            rhs = zeros(iDim * jDim)
            u_analytic = zeros(iDim * jDim)
            for r in 1:iDim, l in 1:jDim
                idx = (r-1)*jDim + l
                rhs[idx] = -(π^2 + (π/2)^2) * sin(π * pts_x[r]) * cos(π/2 * pts_y[l])
                u_analytic[idx] = sin(π * pts_x[r]) * cos(π/2 * pts_y[l])
            end

            L = assemble_from_equation(grid, "u"; d_ii=1.0, d_jj=1.0)
            prob = SpringsteelProblem(grid; operator=L, rhs=rhs)
            sol = solve(prob)

            @test sol.converged == true
            @test maximum(abs.(sol.physical .- u_analytic)) < 0.05

            # Evaluate solution at actual boundary endpoints
            sp_j = grid.jbasis.data[1, 1].params
            sp_i = grid.ibasis.data[1, 1].params
            Nspec_j = sp_j.num_cells + 3

            # Evaluate u at jMax (y=1) — should be zero (Dirichlet)
            for r in 1:iDim
                u_top = 0.0
                for si in -1:(sp_i.num_cells+1)
                    bi = CubicBSpline.basis(sp_i, si, pts_x[r], 0)
                    bi == 0.0 && continue
                    for sj in -1:(sp_j.num_cells+1)
                        bj = CubicBSpline.basis(sp_j, sj, gp.jMax, 0)
                        bj == 0.0 && continue
                        coeff_idx = (si+2-1)*Nspec_j + (sj+2)
                        u_top += sol.coefficients[coeff_idx] * bi * bj
                    end
                end
                @test abs(u_top) < 1e-10  # Dirichlet at jMax: exact zero
            end

            # Evaluate u at jMin (y=0) — should be NONZERO (Neumann)
            u_bot_max = 0.0
            for r in 1:iDim
                u_bot = 0.0
                for si in -1:(sp_i.num_cells+1)
                    bi = CubicBSpline.basis(sp_i, si, pts_x[r], 0)
                    bi == 0.0 && continue
                    for sj in -1:(sp_j.num_cells+1)
                        bj = CubicBSpline.basis(sp_j, sj, gp.jMin, 0)
                        bj == 0.0 && continue
                        coeff_idx = (si+2-1)*Nspec_j + (sj+2)
                        u_bot += sol.coefficients[coeff_idx] * bi * bj
                    end
                end
                u_bot_max = max(u_bot_max, abs(u_bot))
            end
            @test u_bot_max > 0.1  # Neumann side: nonzero
        end

        @testset "2D ZZ asymmetric j-BCs: Dirichlet bottom, Neumann top" begin
            # Same PDE on Chebyshev×Chebyshev grid
            # u(x,y) = sin(πx) sin(πy/2)
            # Chebyshev grids include the boundary endpoints, so direct check works
            Ni = 20; Nj = 20
            gp = SpringsteelGridParameters(
                geometry = "ZZ",
                iMin = 0.0, iMax = 1.0, iDim = Ni, b_iDim = Ni,
                jMin = 0.0, jMax = 1.0, jDim = Nj, b_jDim = Nj,
                BCL = Dict("u" => Chebyshev.R1T0),
                BCR = Dict("u" => Chebyshev.R1T0),
                BCD = Dict("u" => Chebyshev.R1T0),
                BCU = Dict("u" => Chebyshev.R1T1),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            pts_i = solver_gridpoints(grid, "u")
            var_idx = grid.params.vars["u"]
            obj_j = grid.jbasis.data[var_idx]
            pts_j = Springsteel.gridpoints(obj_j)

            rhs = zeros(Ni * Nj)
            u_analytic = zeros(Ni * Nj)
            for i in 1:Ni, j in 1:Nj
                idx = (i-1)*Nj + j
                rhs[idx] = -(π^2 + (π/2)^2) * sin(π * pts_i[i]) * sin(π/2 * pts_j[j])
                u_analytic[idx] = sin(π * pts_i[i]) * sin(π/2 * pts_j[j])
            end

            L = assemble_from_equation(grid, "u"; d_ii=1.0, d_jj=1.0)
            prob = SpringsteelProblem(grid; operator=L, rhs=rhs)
            sol = solve(prob)

            @test sol.converged == true
            @test maximum(abs.(sol.physical .- u_analytic)) < 0.01

            # Chebyshev grids include endpoints: first point is jMin, last is jMax
            # Verify Dirichlet at jMin: u ≈ 0
            bottom_vals = [sol.physical[(i-1)*Nj + 1] for i in 1:Ni]
            @test maximum(abs.(bottom_vals)) < 1e-6

            # Key asymmetry check: u at jMax should be NONZERO
            top_vals = [sol.physical[(i-1)*Nj + Nj] for i in 1:Ni]
            @test maximum(abs.(top_vals)) > 0.1
        end

    end  # Multi-D Linear Solver

    # ─────────────────────────────────────────────────────────────────────────
    # Optimization.jl Backend (conditional)
    # ─────────────────────────────────────────────────────────────────────────

    @testset "OptimizationBackend" begin
        @testset "OptimizationBackend constructor" begin
            ob = OptimizationBackend(:LBFGS)
            @test ob.algorithm == :LBFGS
            @test isempty(ob.options)

            ob2 = OptimizationBackend(:NelderMead, Dict{String,Any}("maxiters" => 500))
            @test ob2.algorithm == :NelderMead
            @test ob2.options["maxiters"] == 500
        end

        @testset "error without Optimization.jl loaded" begin
            gp = SpringsteelGridParameters(
                geometry = "Z",
                iMin = 0.0, iMax = 1.0, iDim = 10, b_iDim = 10,
                BCL = Dict("u" => Chebyshev.R1T0),
                BCR = Dict("u" => Chebyshev.R1T0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)

            cost(phys, params) = sum(phys.^2)
            prob = SpringsteelProblem(grid;
                cost=cost,
                backend=OptimizationBackend(:NelderMead))
            # Without Optimization.jl loaded, solve should throw MethodError
            @test_throws MethodError solve(prob)
        end
    end

end  # Solver Tests

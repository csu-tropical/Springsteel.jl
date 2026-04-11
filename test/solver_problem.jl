# test/solver_problem.jl — S2 of the solver refactor
#
# Round-trip tests for the Pair-based SpringsteelProblem constructor and its
# stateful solve! path. Matches results against known analytical solutions
# and confirms that repeated solves reuse the cached workspace (no rebuild).
#
# Run via: TEST_GROUP=solver_problem julia --project test/runtests.jl

using Test
using Springsteel
using Springsteel.CubicBSpline, Springsteel.Chebyshev

@testset "SpringsteelProblem (stateful) — S2" begin

    @testset "1D Chebyshev Poisson u''=f, Dirichlet" begin
        N = 25
        gp = SpringsteelGridParameters(
            geometry = "Z",
            iMin = 0.0, iMax = 1.0, iDim = N, b_iDim = N,
            BCL = Dict("u" => Chebyshev.R1T0, "f" => Chebyshev.R0),
            BCR = Dict("u" => Chebyshev.R1T0, "f" => Chebyshev.R0),
            vars = Dict("u" => 1, "f" => 2))
        grid = createGrid(gp)
        pts  = solver_gridpoints(grid, "u")

        # Seed the RHS slot with -π² sin(π x).
        grid.physical[:, 2, 1] .= -π^2 .* sin.(π .* pts)

        u = Field(grid, "u")
        prob = SpringsteelProblem(grid, ∂ᵢ^2 * u => :f)
        solve!(prob)

        u_analytic = sin.(π .* pts)
        @test maximum(abs.(grid.physical[:, 1, 1] .- u_analytic)) < 1e-4
    end

    @testset "1D CubicBSpline Poisson, literal vector RHS" begin
        gp = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 1.0, num_cells = 30,
            BCL = Dict("u" => CubicBSpline.R1T0),
            BCR = Dict("u" => CubicBSpline.R1T0),
            vars = Dict("u" => 1))
        grid = createGrid(gp)
        pts  = solver_gridpoints(grid, "u")

        u = Field(grid, "u")
        f = -(2π)^2 .* sin.(2π .* pts)
        prob = SpringsteelProblem(grid, ∂_x^2 * u => f)
        solve!(prob)

        u_analytic = sin.(2π .* pts)
        @test maximum(abs.(grid.physical[:, 1, 1] .- u_analytic)) < 0.05
    end

    @testset "2D RZ Laplacian via ∂_x² + ∂_z²" begin
        gp = SpringsteelGridParameters(
            geometry = "RZ",
            iMin = 0.0, iMax = 1.0, num_cells = 20,
            kMin = 0.0, kMax = 1.0, kDim = 20, b_kDim = 20,
            BCL = Dict("u" => CubicBSpline.R1T0),
            BCR = Dict("u" => CubicBSpline.R1T0),
            BCB = Dict("u" => Chebyshev.R1T0),
            BCT = Dict("u" => Chebyshev.R1T0),
            vars = Dict("u" => 1))
        grid = createGrid(gp)
        pts = getGridpoints(grid)        # iDim*kDim × 2

        # Manufactured: u = sin(π x) sin(π z), Lu = -2π² u
        u_ana = [sin(π*p[1]) * sin(π*p[2]) for p in eachrow(pts)]
        f     = -2π^2 .* u_ana

        u = Field(grid, "u")
        prob = SpringsteelProblem(grid, (∂_x^2 + ∂_z^2) * u => f)
        solve!(prob)
        @test maximum(abs.(grid.physical[:, 1, 1] .- u_ana)) < 1e-2
    end

    @testset "Repeated solve! reuses cached factorisation" begin
        N = 20
        gp = SpringsteelGridParameters(
            geometry = "Z",
            iMin = 0.0, iMax = 1.0, iDim = N, b_iDim = N,
            BCL = Dict("u" => Chebyshev.R1T0, "f" => Chebyshev.R0),
            BCR = Dict("u" => Chebyshev.R1T0, "f" => Chebyshev.R0),
            vars = Dict("u" => 1, "f" => 2))
        grid = createGrid(gp)
        pts  = solver_gridpoints(grid, "u")

        u = Field(grid, "u")
        grid.physical[:, 2, 1] .= -π^2 .* sin.(π .* pts)
        prob = SpringsteelProblem(grid, ∂ᵢ^2 * u => :f)
        solve!(prob)
        F1 = prob.workspace.factorization
        u1 = copy(grid.physical[:, 1, 1])

        # Second RHS — factorisation object must be the same
        grid.physical[:, 2, 1] .= -(2π)^2 .* sin.(2π .* pts)
        solve!(prob)
        F2 = prob.workspace.factorization
        @test F1 === F2

        u2 = copy(grid.physical[:, 1, 1])
        @test maximum(abs.(u1 .- sin.(π .* pts)))      < 1e-4
        @test maximum(abs.(u2 .- sin.(2π .* pts)))     < 1e-3
    end

    @testset "Block system: 2-variable decoupled diagonal" begin
        # Two independent Poisson problems:
        #   u'' = -π²sin(π x),     u(0)=u(1)=0, analytic: u = sin(π x)
        #   v'' = -(2π)²sin(2π x), v(0)=v(1)=0, analytic: v = sin(2π x)
        N = 30
        gp = SpringsteelGridParameters(
            geometry = "Z",
            iMin = 0.0, iMax = 1.0, iDim = N, b_iDim = N,
            BCL = Dict("u" => Chebyshev.R1T0, "v" => Chebyshev.R1T0,
                       "fu" => Chebyshev.R0,  "fv" => Chebyshev.R0),
            BCR = Dict("u" => Chebyshev.R1T0, "v" => Chebyshev.R1T0,
                       "fu" => Chebyshev.R0,  "fv" => Chebyshev.R0),
            vars = Dict("u" => 1, "v" => 2, "fu" => 3, "fv" => 4))
        grid = createGrid(gp)
        pts  = solver_gridpoints(grid, "u")

        grid.physical[:, 3, 1] .= -π^2      .* sin.(π .* pts)
        grid.physical[:, 4, 1] .= -(2π)^2   .* sin.(2π .* pts)

        u = Field(grid, "u")
        v = Field(grid, "v")
        eqs = [
            ∂ᵢ^2 * u => :fu,
            ∂ᵢ^2 * v => :fv,
        ]
        prob = SpringsteelProblem(grid, eqs)
        solve!(prob)

        u_ana = sin.(π .* pts)
        v_ana = sin.(2π .* pts)
        @test maximum(abs.(grid.physical[:, 1, 1] .- u_ana)) < 1e-3
        @test maximum(abs.(grid.physical[:, 2, 1] .- v_ana)) < 1e-3
    end

    @testset "Block system: 2-variable coupled (off-diagonal)" begin
        # Manufactured coupled system on 1D Chebyshev, Dirichlet u=v=0 at ends:
        #   u'' + v = f_u
        #   u + v'' = f_v
        # with u = sin(π x), v = sin(2π x):
        #   f_u = -π² sin(π x)  + sin(2π x)
        #   f_v = sin(π x)       - 4π² sin(2π x)
        N = 40
        gp = SpringsteelGridParameters(
            geometry = "Z",
            iMin = 0.0, iMax = 1.0, iDim = N, b_iDim = N,
            BCL = Dict("u" => Chebyshev.R1T0, "v" => Chebyshev.R1T0,
                       "fu" => Chebyshev.R0,  "fv" => Chebyshev.R0),
            BCR = Dict("u" => Chebyshev.R1T0, "v" => Chebyshev.R1T0,
                       "fu" => Chebyshev.R0,  "fv" => Chebyshev.R0),
            vars = Dict("u" => 1, "v" => 2, "fu" => 3, "fv" => 4))
        grid = createGrid(gp)
        pts  = solver_gridpoints(grid, "u")

        grid.physical[:, 3, 1] .= -π^2 .* sin.(π .* pts) .+ sin.(2π .* pts)
        grid.physical[:, 4, 1] .= sin.(π .* pts) .- (2π)^2 .* sin.(2π .* pts)

        u = Field(grid, "u")
        v = Field(grid, "v")
        eqs = [
            (∂ᵢ^2 * u) + (∂ᵢ^0 * v) => :fu,
            (∂ᵢ^0 * u) + (∂ᵢ^2 * v) => :fv,
        ]
        prob = SpringsteelProblem(grid, eqs)
        solve!(prob)

        @test maximum(abs.(grid.physical[:, 1, 1] .- sin.(π .* pts)))  < 1e-3
        @test maximum(abs.(grid.physical[:, 2, 1] .- sin.(2π .* pts))) < 1e-3
    end

    @testset "Block system error paths" begin
        gp = SpringsteelGridParameters(
            geometry = "Z",
            iMin = 0.0, iMax = 1.0, iDim = 10, b_iDim = 10,
            BCL = Dict("u" => Chebyshev.R1T0, "v" => Chebyshev.R1T0),
            BCR = Dict("u" => Chebyshev.R1T0, "v" => Chebyshev.R1T0),
            vars = Dict("u" => 1, "v" => 2))
        grid = createGrid(gp)
        u = Field(grid, "u")
        v = Field(grid, "v")

        # Wrong equation count
        @test_throws ArgumentError SpringsteelProblem(grid,
            Pair{TypedOperator,Any}[∂ᵢ^2 * u + ∂ᵢ^0 * v => 0])

        # Diagonal missing: first-appearance ordering gives fields = [u, v];
        # eq 2 only has a term in u → L[2,2] block is empty. Should error.
        @test_throws ArgumentError SpringsteelProblem(grid,
            [∂ᵢ^2 * u + ∂ᵢ^2 * v => 0, ∂ᵢ^2 * u => 0])
    end

    @testset "S4a backend parity: dense vs sparse" begin
        # 1D CubicBSpline Poisson — sparse is the default (non-Chebyshev grid)
        gp = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 1.0, num_cells = 30,
            BCL = Dict("u" => CubicBSpline.R1T0),
            BCR = Dict("u" => CubicBSpline.R1T0),
            vars = Dict("u" => 1))
        grid_d = createGrid(gp); pts = solver_gridpoints(grid_d, "u")
        grid_s = createGrid(gp)
        f = -(2π)^2 .* sin.(2π .* pts)

        u_d = Field(grid_d, "u")
        u_s = Field(grid_s, "u")
        prob_d = SpringsteelProblem(grid_d, ∂_x^2 * u_d => f; backend = :dense)
        prob_s = SpringsteelProblem(grid_s, ∂_x^2 * u_s => f; backend = :sparse)
        solve!(prob_d); solve!(prob_s)

        @test prob_d.backend isa LocalLinearBackend
        @test prob_s.backend isa SparseLinearBackend
        @test maximum(abs.(grid_d.physical[:, 1, 1] .- grid_s.physical[:, 1, 1])) < 1e-10
    end

    @testset "S4a auto-pick: pure Chebyshev → dense, RZ → sparse" begin
        gp_cheb = SpringsteelGridParameters(
            geometry = "Z",
            iMin = 0.0, iMax = 1.0, iDim = 12, b_iDim = 12,
            BCL = Dict("u" => Chebyshev.R1T0),
            BCR = Dict("u" => Chebyshev.R1T0),
            vars = Dict("u" => 1))
        grid_cheb = createGrid(gp_cheb)
        u_c = Field(grid_cheb, "u")
        pts_c = solver_gridpoints(grid_cheb, "u")
        f_c = -π^2 .* sin.(π .* pts_c)
        prob_c = SpringsteelProblem(grid_cheb, ∂ᵢ^2 * u_c => f_c)
        @test prob_c.backend isa LocalLinearBackend

        gp_rz = SpringsteelGridParameters(
            geometry = "RZ",
            iMin = 0.0, iMax = 1.0, num_cells = 10,
            kMin = 0.0, kMax = 1.0, kDim = 10, b_kDim = 10,
            BCL = Dict("u" => CubicBSpline.R1T0),
            BCR = Dict("u" => CubicBSpline.R1T0),
            BCB = Dict("u" => Chebyshev.R1T0),
            BCT = Dict("u" => Chebyshev.R1T0),
            vars = Dict("u" => 1))
        grid_rz = createGrid(gp_rz)
        u_rz = Field(grid_rz, "u")
        prob_rz = SpringsteelProblem(grid_rz, (∂_x^2 + ∂_z^2) * u_rz => zeros(gp_rz.iDim * gp_rz.kDim))
        @test prob_rz.backend isa SparseLinearBackend
    end

    @testset "S4a block system with sparse backend" begin
        # Coupled 1D Chebyshev — force sparse backend even though auto-pick
        # would choose dense (to verify sparse LU handles blocks correctly).
        N = 30
        gp = SpringsteelGridParameters(
            geometry = "Z",
            iMin = 0.0, iMax = 1.0, iDim = N, b_iDim = N,
            BCL = Dict("u" => Chebyshev.R1T0, "v" => Chebyshev.R1T0,
                       "fu" => Chebyshev.R0,  "fv" => Chebyshev.R0),
            BCR = Dict("u" => Chebyshev.R1T0, "v" => Chebyshev.R1T0,
                       "fu" => Chebyshev.R0,  "fv" => Chebyshev.R0),
            vars = Dict("u" => 1, "v" => 2, "fu" => 3, "fv" => 4))
        grid = createGrid(gp)
        pts  = solver_gridpoints(grid, "u")

        grid.physical[:, 3, 1] .= -π^2    .* sin.(π .* pts)
        grid.physical[:, 4, 1] .= -(2π)^2 .* sin.(2π .* pts)

        u = Field(grid, "u"); v = Field(grid, "v")
        eqs = [∂ᵢ^2 * u => :fu, ∂ᵢ^2 * v => :fv]
        prob = SpringsteelProblem(grid, eqs; backend = :sparse)
        solve!(prob)
        @test prob.backend isa SparseLinearBackend
        @test maximum(abs.(grid.physical[:, 1, 1] .- sin.(π .* pts)))  < 1e-3
        @test maximum(abs.(grid.physical[:, 2, 1] .- sin.(2π .* pts))) < 1e-3
    end

    @testset "S4a unknown backend symbol errors" begin
        gp = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 1.0, num_cells = 5,
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0),
            vars = Dict("u" => 1))
        grid = createGrid(gp)
        u = Field(grid, "u")
        @test_throws ArgumentError SpringsteelProblem(grid,
            ∂ᵢ^2 * u => zeros(gp.iDim); backend = :nonsense)
    end

    @testset "Error paths" begin
        gp = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 1.0, num_cells = 5,
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0),
            vars = Dict("u" => 1))
        grid = createGrid(gp)

        @test_throws ArgumentError Field(grid, "nonexistent")

        # Legacy (workspace-less) problem → solve! errors
        prob_legacy = SpringsteelProblem(grid; operator = rand(gp.iDim, gp.iDim),
                                          rhs = ones(gp.iDim))
        @test_throws ArgumentError solve!(prob_legacy)
    end
end

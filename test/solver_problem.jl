# test/solver_problem.jl — S2 of the solver refactor
#
# Round-trip tests for the Pair-based SpringsteelProblem constructor and its
# stateful solve! path. Matches results against known analytical solutions
# and confirms that repeated solves reuse the cached workspace (no rebuild).
#
# Run via: TEST_GROUP=solver_problem julia --project test/runtests.jl

using Test
using LinearAlgebra
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

    @testset "S4b Krylov backend parity: square 1D Chebyshev" begin
        N = 25
        gp = SpringsteelGridParameters(
            geometry = "Z",
            iMin = 0.0, iMax = 1.0, iDim = N, b_iDim = N,
            BCL = Dict("u" => Chebyshev.R1T0),
            BCR = Dict("u" => Chebyshev.R1T0),
            vars = Dict("u" => 1))
        grid_d = createGrid(gp); pts = solver_gridpoints(grid_d, "u")
        grid_k = createGrid(gp)
        f = -π^2 .* sin.(π .* pts)

        u_d = Field(grid_d, "u"); u_k = Field(grid_k, "u")
        prob_d = SpringsteelProblem(grid_d, ∂ᵢ^2 * u_d => f; backend = :dense)
        prob_k = SpringsteelProblem(grid_k, ∂ᵢ^2 * u_k => f; backend = :krylov)
        solve!(prob_d); solve!(prob_k)

        @test prob_k.backend isa KrylovLinearBackend
        @test maximum(abs.(grid_d.physical[:, 1, 1] .- grid_k.physical[:, 1, 1])) < 1e-6
        @test maximum(abs.(grid_k.physical[:, 1, 1] .- sin.(π .* pts))) < 1e-4
    end

    @testset "S4b Krylov backend parity: rectangular 1D spline" begin
        gp = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 1.0, num_cells = 30,
            BCL = Dict("u" => CubicBSpline.R1T0),
            BCR = Dict("u" => CubicBSpline.R1T0),
            vars = Dict("u" => 1))
        grid_s = createGrid(gp); pts = solver_gridpoints(grid_s, "u")
        grid_k = createGrid(gp)
        f = -(2π)^2 .* sin.(2π .* pts)

        u_s = Field(grid_s, "u"); u_k = Field(grid_k, "u")
        prob_s = SpringsteelProblem(grid_s, ∂_x^2 * u_s => f; backend = :sparse)
        prob_k = SpringsteelProblem(grid_k, ∂_x^2 * u_k => f; backend = :krylov)
        solve!(prob_s); solve!(prob_k)

        @test prob_k.backend isa KrylovLinearBackend
        @test maximum(abs.(grid_s.physical[:, 1, 1] .- grid_k.physical[:, 1, 1])) < 1e-6
    end

    @testset "S4b Krylov preconditioner plumbing" begin
        # Pass an identity preconditioner through the backend to verify the
        # hook is wired — an identity left-preconditioner must give the same
        # answer as no preconditioner at all. S4c will add a proper default
        # diagonal preconditioner helper.
        N = 25
        gp = SpringsteelGridParameters(
            geometry = "Z",
            iMin = 0.0, iMax = 1.0, iDim = N, b_iDim = N,
            BCL = Dict("u" => Chebyshev.R1T0),
            BCR = Dict("u" => Chebyshev.R1T0),
            vars = Dict("u" => 1))
        grid = createGrid(gp); pts = solver_gridpoints(grid, "u")
        f = -π^2 .* sin.(π .* pts)

        Minv = Diagonal(ones(N))   # identity left preconditioner
        u = Field(grid, "u")
        prob = SpringsteelProblem(grid, ∂ᵢ^2 * u => f;
                                   backend = KrylovLinearBackend(Minv))
        solve!(prob)
        @test prob.backend.preconditioner === Minv
        @test maximum(abs.(grid.physical[:, 1, 1] .- sin.(π .* pts))) < 1e-4
    end

    @testset "S4c preconditioner kwarg on constructor" begin
        N = 25
        gp = SpringsteelGridParameters(
            geometry = "Z",
            iMin = 0.0, iMax = 1.0, iDim = N, b_iDim = N,
            BCL = Dict("u" => Chebyshev.R1T0),
            BCR = Dict("u" => Chebyshev.R1T0),
            vars = Dict("u" => 1))
        grid_a = createGrid(gp); pts = solver_gridpoints(grid_a, "u")
        grid_b = createGrid(gp)
        grid_c = createGrid(gp)
        f = -π^2 .* sin.(π .* pts)

        # Identity matrix as a pass-through preconditioner
        Ident = Diagonal(ones(N))
        u_a = Field(grid_a, "u"); u_b = Field(grid_b, "u"); u_c = Field(grid_c, "u")

        prob_a = SpringsteelProblem(grid_a, ∂ᵢ^2 * u_a => f;
                                     backend = :krylov,
                                     preconditioner = Ident)
        prob_b = SpringsteelProblem(grid_b, ∂ᵢ^2 * u_b => f;
                                     backend = :krylov,
                                     preconditioner = nothing)
        prob_c = SpringsteelProblem(grid_c, ∂ᵢ^2 * u_c => f;
                                     backend = :krylov)      # default (= nothing)
        solve!(prob_a); solve!(prob_b); solve!(prob_c)

        @test prob_a.backend.preconditioner === Ident
        @test prob_b.backend.preconditioner === nothing
        @test prob_c.backend.preconditioner === nothing

        u_ana = sin.(π .* pts)
        @test maximum(abs.(grid_a.physical[:, 1, 1] .- u_ana)) < 1e-4
        @test maximum(abs.(grid_b.physical[:, 1, 1] .- u_ana)) < 1e-4
        @test maximum(abs.(grid_c.physical[:, 1, 1] .- u_ana)) < 1e-4
    end

    @testset "S4c :diag opt-in builds a diagonal preconditioner" begin
        N = 20
        gp = SpringsteelGridParameters(
            geometry = "Z",
            iMin = 0.0, iMax = 1.0, iDim = N, b_iDim = N,
            BCL = Dict("u" => Chebyshev.R0),    # free BCs → no row replacement
            BCR = Dict("u" => Chebyshev.R0),
            vars = Dict("u" => 1))
        grid = createGrid(gp)
        u = Field(grid, "u")
        # Well-conditioned 0th-order system: α·I with α random
        α = 2.0 .+ rand(N)
        f = rand(N)
        prob = SpringsteelProblem(grid, α * ∂ᵢ^0 * u => f;
                                   backend = :krylov,
                                   preconditioner = :diag)
        solve!(prob)
        @test prob.workspace.factorization.preconditioner isa Diagonal
        # A·u = f with A = Diagonal(α), so u = f ./ α
        @test maximum(abs.(grid.physical[:, 1, 1] .- f ./ α)) < 1e-8
    end

    @testset "S4c preconditioner rejected on non-Krylov backends" begin
        gp = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 1.0, num_cells = 5,
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0),
            vars = Dict("u" => 1))
        grid = createGrid(gp)
        u = Field(grid, "u")
        @test_throws ArgumentError SpringsteelProblem(grid,
            ∂ᵢ^2 * u => zeros(gp.iDim);
            backend = :sparse,
            preconditioner = Diagonal(ones(gp.iDim)))

        # :default and nothing are fine on non-Krylov backends
        prob1 = SpringsteelProblem(grid, ∂ᵢ^2 * u => zeros(gp.iDim);
                                    backend = :sparse, preconditioner = :default)
        @test prob1.backend isa SparseLinearBackend
        prob2 = SpringsteelProblem(grid, ∂ᵢ^2 * u => zeros(gp.iDim);
                                    backend = :sparse, preconditioner = nothing)
        @test prob2.backend isa SparseLinearBackend
    end

    @testset "S4c unknown preconditioner symbol errors" begin
        gp = SpringsteelGridParameters(
            geometry = "Z",
            iMin = 0.0, iMax = 1.0, iDim = 10, b_iDim = 10,
            BCL = Dict("u" => Chebyshev.R0),
            BCR = Dict("u" => Chebyshev.R0),
            vars = Dict("u" => 1))
        grid = createGrid(gp)
        u = Field(grid, "u")
        @test_throws ArgumentError SpringsteelProblem(grid,
            ∂ᵢ^0 * u => ones(10);
            backend = :krylov,
            preconditioner = :bogus)
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

    @testset "S5 coefficient resolution: Vector/Function/Symbol parity" begin
        # Solve u''(x) + α(x) u'(x) = f(x) where α(x) = sin(x) + 2.
        # Compare the three forms of α: precomputed Vector, Function, and
        # Symbol (stored on the grid). Must produce identical results.
        gp = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.1, iMax = 1.0, num_cells = 40,
            BCL = Dict("u" => CubicBSpline.R1T0, "α" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R1T0, "α" => CubicBSpline.R0),
            vars = Dict("u" => 1, "α" => 2))

        α_fn(x) = sin(x) + 2.0

        grid_v = createGrid(gp); pts = solver_gridpoints(grid_v, "u")
        grid_f = createGrid(gp)
        grid_s = createGrid(gp)

        α_vec = α_fn.(pts)
        # Seed α on the grid_s variable slot so the Symbol resolves against it
        grid_s.physical[:, 2, 1] .= α_vec

        f = -(2π)^2 .* sin.(2π .* pts)    # any reasonable RHS

        u_v = Field(grid_v, "u")
        u_f = Field(grid_f, "u")
        u_s = Field(grid_s, "u")

        prob_v = SpringsteelProblem(grid_v, (∂_x^2 + α_vec * ∂_x) * u_v => f)
        prob_f = SpringsteelProblem(grid_f, (∂_x^2 + α_fn  * ∂_x) * u_f => f)
        prob_s = SpringsteelProblem(grid_s, (∂_x^2 + :α    * ∂_x) * u_s => f)

        solve!(prob_v); solve!(prob_f); solve!(prob_s)

        uv = grid_v.physical[:, 1, 1]
        uf = grid_f.physical[:, 1, 1]
        us = grid_s.physical[:, 1, 1]

        @test maximum(abs.(uv .- uf)) < 1e-12
        @test maximum(abs.(uv .- us)) < 1e-12
    end

    @testset "S5 Function coefficient on 2D grid (RZ)" begin
        gp = SpringsteelGridParameters(
            geometry = "RZ",
            iMin = 0.0, iMax = 1.0, num_cells = 10,
            kMin = 0.0, kMax = 1.0, kDim = 10, b_kDim = 10,
            BCL = Dict("u" => CubicBSpline.R1T0),
            BCR = Dict("u" => CubicBSpline.R1T0),
            BCB = Dict("u" => Chebyshev.R1T0),
            BCT = Dict("u" => Chebyshev.R1T0),
            vars = Dict("u" => 1))
        grid1 = createGrid(gp); grid2 = createGrid(gp)

        # Use a function that depends on both x and z
        α_fn(x, z) = 1.0 + 0.1 * x * z
        pts_mat = getGridpoints(grid1)
        α_vec = [α_fn(pts_mat[i, 1], pts_mat[i, 2]) for i in 1:size(pts_mat, 1)]

        u1 = Field(grid1, "u"); u2 = Field(grid2, "u")
        rhs = zeros(gp.iDim * gp.kDim)

        prob1 = SpringsteelProblem(grid1, (∂_x^2 + ∂_z^2 + α_vec * ∂_x^0) * u1 => rhs)
        prob2 = SpringsteelProblem(grid2, (∂_x^2 + ∂_z^2 + α_fn  * ∂_x^0) * u2 => rhs)
        solve!(prob1); solve!(prob2)

        @test maximum(abs.(grid1.physical[:, 1, 1] .- grid2.physical[:, 1, 1])) < 1e-12
    end

    @testset "S5 unknown Symbol coefficient errors" begin
        gp = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 1.0, num_cells = 5,
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0),
            vars = Dict("u" => 1))
        grid = createGrid(gp)
        u = Field(grid, "u")
        @test_throws ArgumentError SpringsteelProblem(grid,
            (:nonexistent * ∂_x) * u => zeros(gp.iDim))
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

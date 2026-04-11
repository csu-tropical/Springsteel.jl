# test/operator_algebra.jl — parity tests for S1 of the solver refactor.
#
# Confirms that the DerivMono / OperatorExpr DSL lowers to the same
# Vector{OperatorTerm} that the low-level solver expects. This guards against
# drift during S2+ when the new SpringsteelProblem API starts using _lower to
# feed assemble_operator.
#
# Run via: TEST_GROUP=operator_algebra julia --project test/runtests.jl

using Test
using Springsteel
using Springsteel.CubicBSpline, Springsteel.Chebyshev

# Unexported internals needed for the parity comparison — the test is
# deliberately white-box.
const _lower = Springsteel._lower
const OperatorTerm = Springsteel.OperatorTerm

# Order-insensitive comparison of term vectors. Dict iteration in DerivMono is
# not stable, so term order is not guaranteed.
function _terms_equal(a::Vector{OperatorTerm}, b::Vector{OperatorTerm})
    length(a) == length(b) || return false
    key(t) = (t.i_order, t.j_order, t.k_order,
              t.coefficient isa Vector ? (:vec, length(t.coefficient), sum(t.coefficient)) :
              t.coefficient === nothing ? (:none,) :
              (:scalar, t.coefficient))
    return Set(key.(a)) == Set(key.(b))
end

function _make_grid_R()
    return createGrid(SpringsteelGridParameters(
        geometry="R", iMin=0.0, iMax=10.0, num_cells=8,
        BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0),
        vars=Dict("u"=>1)))
end

function _make_grid_RR()
    return createGrid(SpringsteelGridParameters(
        geometry="RR", iMin=0.0, iMax=10.0, jMin=0.0, jMax=10.0, num_cells=8,
        BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0),
        BCU=Dict("u"=>CubicBSpline.R0), BCD=Dict("u"=>CubicBSpline.R0),
        vars=Dict("u"=>1)))
end

function _make_grid_RZ()
    return createGrid(SpringsteelGridParameters(
        geometry="RZ", iMin=0.0, iMax=10.0, kMin=0.0, kMax=10.0,
        num_cells=8, kDim=16,
        BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0),
        BCB=Dict("u"=>Chebyshev.R0),    BCT=Dict("u"=>Chebyshev.R0),
        vars=Dict("u"=>1)))
end

function _make_grid_RLZ()
    return createGrid(SpringsteelGridParameters(
        geometry="RLZ", iMin=0.0, iMax=10.0, kMin=0.0, kMax=10.0,
        num_cells=8, kDim=16,
        BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0),
        BCB=Dict("u"=>Chebyshev.R0),    BCT=Dict("u"=>Chebyshev.R0),
        vars=Dict("u"=>1)))
end

function _make_grid_SLZ()
    return createGrid(SpringsteelGridParameters(
        geometry="SLZ", iMin=0.05, iMax=π-0.05, kMin=0.0, kMax=10.0,
        num_cells=8, kDim=16,
        BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0),
        BCB=Dict("u"=>Chebyshev.R0),    BCT=Dict("u"=>Chebyshev.R0),
        vars=Dict("u"=>1)))
end

@testset "Operator algebra — S1 parity" begin
    @testset "Builds AST correctly" begin
        @test (∂ᵢ^2).orders == Dict(:i => 2)
        @test (∂ᵢ * ∂ⱼ).orders == Dict(:i => 1, :j => 1)
        @test (∂ᵢ^2 + ∂ⱼ^2) isa Springsteel.OperatorExpr
        @test length((∂ᵢ^2 + ∂ⱼ^2).terms) == 2

        @test d_i === ∂ᵢ
        @test d_r === ∂_r

        e = 2.0 * ∂ᵢ^2
        @test e.terms[1].coeff == 2.0

        v = rand(5)
        e = v * ∂ᵢ
        @test e.terms[1].coeff === v

        e = -∂ᵢ^2
        @test e.terms[1].coeff == -1.0

        g = _make_grid_R()
        @test_throws ArgumentError _lower(∂ᵢ^3, g)
    end

    @testset "R: Laplacian d_ii parity" begin
        grid = _make_grid_R()
        dsl = _lower(∂ᵢ^2, grid)
        ref = [OperatorTerm(2, 0, 0, nothing)]
        @test _terms_equal(dsl, ref)
    end

    @testset "R: d_ii + d_i (advection-diffusion)" begin
        grid = _make_grid_R()
        α = 0.7
        β = -1.5
        dsl = _lower(α*∂ᵢ^2 + β*∂ᵢ, grid)
        ref = [OperatorTerm(2, 0, 0, α), OperatorTerm(1, 0, 0, β)]
        @test _terms_equal(dsl, ref)
    end

    @testset "R (Cartesian): ∂_x is the valid physical alias, ∂_r errors" begin
        grid = _make_grid_R()
        @test_throws ArgumentError _lower(∂_r^2, grid)
        @test _terms_equal(_lower(∂_x^2, grid), [OperatorTerm(2, 0, 0, nothing)])
    end

    @testset "RR: Laplacian ∂_x² + ∂_y² == ∂ᵢ² + ∂ⱼ²" begin
        grid = _make_grid_RR()
        dsl_generic  = _lower(∂ᵢ^2 + ∂ⱼ^2, grid)
        dsl_physical = _lower(∂_x^2 + ∂_y^2, grid)
        @test _terms_equal(dsl_generic, dsl_physical)
        ref = [OperatorTerm(2, 0, 0, nothing), OperatorTerm(0, 2, 0, nothing)]
        @test _terms_equal(dsl_generic, ref)
    end

    @testset "RR: mixed derivative ∂_x ∂_y" begin
        grid = _make_grid_RR()
        dsl = _lower(∂ᵢ * ∂ⱼ, grid)
        ref = [OperatorTerm(1, 1, 0, nothing)]
        @test _terms_equal(dsl, ref)
    end

    @testset "RZ: ∂_x² + ∂_z² Laplacian" begin
        grid = _make_grid_RZ()
        dsl = _lower(∂_x^2 + ∂_z^2, grid)
        ref = [OperatorTerm(2, 0, 0, nothing), OperatorTerm(0, 0, 2, nothing)]
        @test _terms_equal(dsl, ref)
    end

    @testset "RLZ: ∂_r² + ∂_z² (cylindrical physical axes)" begin
        grid = _make_grid_RLZ()
        dsl_physical = _lower(∂_r^2 + ∂_z^2, grid)
        dsl_generic  = _lower(∂ᵢ^2 + ∂ₖ^2, grid)
        @test _terms_equal(dsl_physical, dsl_generic)
    end

    @testset "RLZ: spatially varying coefficient α * ∂_r" begin
        grid = _make_grid_RLZ()
        α = rand(7)
        dsl = _lower(α * ∂_r, grid)
        @test length(dsl) == 1
        @test dsl[1].i_order == 1 && dsl[1].j_order == 0 && dsl[1].k_order == 0
        @test dsl[1].coefficient === α
    end

    @testset "RLZ: θ aliases to λ (azimuth)" begin
        grid = _make_grid_RLZ()
        @test _terms_equal(_lower(∂_θ, grid), _lower(∂_λ, grid))
        @test _terms_equal(_lower(∂_θ, grid), [OperatorTerm(0, 1, 0, nothing)])
    end

    @testset "SLZ: ∂_θ (colatitude) maps to :i" begin
        grid = _make_grid_SLZ()
        dsl = _lower(∂_θ^2, grid)
        @test _terms_equal(dsl, [OperatorTerm(2, 0, 0, nothing)])
    end

    @testset "SLZ: ∂_r errors (not a spherical axis)" begin
        grid = _make_grid_SLZ()
        @test_throws ArgumentError _lower(∂_r^2, grid)
    end

    @testset "Scalar distributes over sum" begin
        grid = _make_grid_RR()
        e1 = _lower(2.0 * (∂ᵢ^2 + ∂ⱼ^2), grid)
        e2 = _lower(2.0 * ∂ᵢ^2 + 2.0 * ∂ⱼ^2, grid)
        @test _terms_equal(e1, e2)
    end

    @testset "Subtraction produces negative coefficients" begin
        grid = _make_grid_RZ()
        dsl = _lower(∂_x^2 - ∂_z^2, grid)
        ref = [OperatorTerm(2, 0, 0, nothing), OperatorTerm(0, 0, 2, -1.0)]
        @test _terms_equal(dsl, ref)
    end

    @testset "Identity term from zero-power" begin
        grid = _make_grid_R()
        dsl = _lower(3.5 * ∂ᵢ^0, grid)
        ref = [OperatorTerm(0, 0, 0, 3.5)]
        @test _terms_equal(dsl, ref)
    end
end

@testset "BandedCholesky3" begin
    using LinearAlgebra
    using Springsteel.CubicBSpline: BandedCholesky3, DenseSplineFactor,
        AbstractSplineFactor, cholesky_banded3, cholesky_dense_factor

    # Helper: build a SPD matrix with half-bandwidth ≤ 3.
    function spd_band3(n::Int; rng_seed::Int = 0)
        # Deterministic SPD construction: M = B*B' + diag where B is sparse
        # with ≤ 4 nonzeros per column placed in the band.
        B = zeros(n, n)
        for j in 1:n
            for i in j:min(n, j + 3)
                B[i, j] = 1.0 / (1.0 + (i - j)) + 0.1 * (i + j)
            end
        end
        A = B * B'
        # Symmetrize numerically and add diagonal weight to ensure SPD.
        A = 0.5 * (A + A')
        for i in 1:n
            A[i, i] += 2.0
        end
        return Symmetric(A)
    end

    @testset "round-trip vs LinearAlgebra.cholesky" begin
        for n in (1, 2, 3, 4, 5, 13, 53, 100, 200)
            A    = spd_band3(n)
            Fban = cholesky_banded3(A)
            Fden = cholesky(A)

            b = collect(Float64, 1:n) .+ 0.5
            x_ref = Fden \ b

            x_ban = similar(b)
            ldiv!(x_ban, Fban, b)
            @test isapprox(x_ban, x_ref; rtol = 1e-12, atol = 1e-12)

            # Backslash convenience method
            x_bk = Fban \ b
            @test isapprox(x_bk, x_ref; rtol = 1e-12, atol = 1e-12)
        end
    end

    @testset "ldiv! is allocation-free" begin
        n   = 100
        A   = spd_band3(n)
        F   = cholesky_banded3(A)
        b   = randn(n)
        out = zeros(n)
        ldiv!(out, F, b)                              # warm up
        allocs = @allocated ldiv!(out, F, b)
        @test allocs == 0
    end

    @testset "DenseSplineFactor (periodic fallback)" begin
        n = 20
        A = spd_band3(n)
        F = cholesky_dense_factor(A)
        @test F isa DenseSplineFactor
        @test F isa AbstractSplineFactor
        b = randn(n)
        out = zeros(n)
        ldiv!(out, F, b)
        x_ref = cholesky(A) \ b
        @test isapprox(out, x_ref; rtol = 1e-12, atol = 1e-12)

        # ldiv! must also be allocation-free for the dense fallback
        allocs = @allocated ldiv!(out, F, b)
        @test allocs == 0
    end

    @testset "bandwidth assertion" begin
        n = 10
        A = Matrix(spd_band3(n).data)
        # Inject an out-of-band entry: A[1, 6] (and its symmetric image)
        A[1, 6] = 1.0
        A[6, 1] = 1.0
        @test_throws ArgumentError cholesky_banded3(Symmetric(A))
    end

    @testset "throws on non-PD matrix" begin
        n = 5
        # Build a banded but indefinite matrix
        A = zeros(n, n)
        for j in 1:n, i in max(1, j - 3):min(n, j + 3)
            A[i, j] = i == j ? -1.0 : 0.1
        end
        @test_throws LinearAlgebra.PosDefException cholesky_banded3(Symmetric(0.5*(A+A')))
    end
end

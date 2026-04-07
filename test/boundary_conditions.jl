@testset "BoundaryConditions type system" begin

    # ── Constructor validation ────────────────────────────────────────────
    @testset "Constructor validation" begin
        # Basic construction
        @test NaturalBC() == BoundaryConditions(nothing, nothing, nothing, nothing, false)
        @test DirichletBC() == BoundaryConditions(0.0, nothing, nothing, nothing, false)
        @test DirichletBC(5.0) == BoundaryConditions(5.0, nothing, nothing, nothing, false)
        @test NeumannBC() == BoundaryConditions(nothing, 0.0, nothing, nothing, false)
        @test NeumannBC(1.5) == BoundaryConditions(nothing, 1.5, nothing, nothing, false)
        @test SecondDerivativeBC() == BoundaryConditions(nothing, nothing, 0.0, nothing, false)
        @test PeriodicBC() == BoundaryConditions(nothing, nothing, nothing, nothing, true)
        @test RobinBC(1.0, -2.0) == BoundaryConditions(nothing, nothing, nothing, (1.0, -2.0, 0.0), false)
        @test RobinBC(1.0, -2.0, 3.0) == BoundaryConditions(nothing, nothing, nothing, (1.0, -2.0, 3.0), false)

        # Aliases
        @test SymmetricBC() == NeumannBC(0.0)
        @test AntisymmetricBC() == BoundaryConditions(0.0, nothing, 0.0, nothing, false)
        @test CauchyBC(0.0, 0.0) == BoundaryConditions(0.0, 0.0, nothing, nothing, false)
        @test ExponentialBC(2.0) == RobinBC(1.0, -2.0, 0.0)
        @test ZerosBC() == BoundaryConditions(0.0, 0.0, 0.0, nothing, false)
        @test FixedBC().u === NaN  # Sentinel for runtime-determined values
        @test bc_rank(FixedBC()) == 3
        @test FixedBC(1.0, 2.0, 3.0) == BoundaryConditions(1.0, 2.0, 3.0, nothing, false)
        @test FixedBC(5.0) == BoundaryConditions(5.0, 0.0, 0.0, nothing, false)
        @test FixedBC(0.0, 0.0, 0.0) == BoundaryConditions(0.0, 0.0, 0.0, nothing, false)

        # Robin exclusivity: cannot combine with u/du/d2u
        @test_throws ArgumentError BoundaryConditions(1.0, nothing, nothing, (1.0, 2.0, 0.0))
        @test_throws ArgumentError BoundaryConditions(nothing, 1.0, nothing, (1.0, 2.0, 0.0))
        @test_throws ArgumentError BoundaryConditions(nothing, nothing, 1.0, (1.0, 2.0, 0.0))

        # Periodic exclusivity: cannot combine with anything
        @test_throws ArgumentError BoundaryConditions(1.0, nothing, nothing, nothing, true)
        @test_throws ArgumentError BoundaryConditions(nothing, 1.0, nothing, nothing, true)
        @test_throws ArgumentError BoundaryConditions(nothing, nothing, nothing, (1.0, 2.0, 0.0), true)
    end

    # ── Rank computation ──────────────────────────────────────────────────
    @testset "bc_rank" begin
        @test bc_rank(NaturalBC()) == 0
        @test bc_rank(DirichletBC()) == 1
        @test bc_rank(NeumannBC()) == 1
        @test bc_rank(SecondDerivativeBC()) == 1
        @test bc_rank(RobinBC(1.0, -1.0)) == 1
        @test bc_rank(CauchyBC(0.0, 0.0)) == 2         # R2T10
        @test bc_rank(AntisymmetricBC()) == 2            # R2T20
        @test bc_rank(BoundaryConditions(0.0, 0.0, 0.0, nothing)) == 3  # R3
        @test bc_rank(ZerosBC()) == 3
        @test bc_rank(FixedBC()) == 3
        @test bc_rank(FixedBC(1.0, 2.0, 3.0)) == 3
        @test bc_rank(PeriodicBC()) == 0
    end

    # ── Utility functions ─────────────────────────────────────────────────
    @testset "is_periodic / is_inhomogeneous" begin
        @test is_periodic(PeriodicBC()) == true
        @test is_periodic(NaturalBC()) == false
        @test is_periodic(DirichletBC()) == false

        @test is_inhomogeneous(NaturalBC()) == false
        @test is_inhomogeneous(DirichletBC(0.0)) == false
        @test is_inhomogeneous(DirichletBC(1.0)) == true
        @test is_inhomogeneous(NeumannBC(0.0)) == false
        @test is_inhomogeneous(NeumannBC(0.5)) == true
        @test is_inhomogeneous(BoundaryConditions(1.0, 2.0, 3.0, nothing)) == true
        @test is_inhomogeneous(BoundaryConditions(0.0, 0.0, 0.0, nothing)) == false
        @test is_inhomogeneous(ZerosBC()) == false
        @test is_inhomogeneous(FixedBC()) == true  # NaN sentinels are inhomogeneous
        @test is_inhomogeneous(FixedBC(0.0, 0.0, 0.0)) == false  # Explicit zeros are homogeneous
        @test is_inhomogeneous(FixedBC(1.0)) == true
        @test is_inhomogeneous(FixedBC(0.0, 0.5)) == true
        @test is_inhomogeneous(RobinBC(1.0, -1.0, 0.0)) == false
        @test is_inhomogeneous(RobinBC(1.0, -1.0, 5.0)) == true
    end

    # ── Spline conversion ─────────────────────────────────────────────────
    @testset "Spline Dict conversion" begin
        # Access the internal conversion function
        to_spline = Springsteel._bc_to_spline_dict

        # Natural → R0
        d = to_spline(NaturalBC())
        @test d == CubicBSpline.R0

        # Dirichlet → R1T0 coefficients
        d = to_spline(DirichletBC())
        @test d["α1"] ≈ -4.0
        @test d["β1"] ≈ -1.0

        # Neumann → R1T1 coefficients
        d = to_spline(NeumannBC())
        @test d["α1"] ≈ 0.0
        @test d["β1"] ≈ 1.0

        # SecondDerivative → R1T2 coefficients
        d = to_spline(SecondDerivativeBC())
        @test d["α1"] ≈ 2.0
        @test d["β1"] ≈ -1.0

        # R2T10 (CauchyBC)
        d = to_spline(CauchyBC(0.0, 0.0))
        @test d["α2"] ≈ 1.0
        @test d["β2"] ≈ -0.5

        # R2T20 (AntisymmetricBC)
        d = to_spline(AntisymmetricBC())
        @test d["α2"] ≈ -1.0
        @test d["β2"] ≈ 0.0

        # R3 (homogeneous rank-3)
        d = to_spline(BoundaryConditions(0.0, 0.0, 0.0, nothing))
        @test d == CubicBSpline.R3

        # R3X (inhomogeneous rank-3)
        d = to_spline(BoundaryConditions(1.0, 2.0, 3.0, nothing))
        @test d == Dict("R3X" => 0)

        # ZerosBC → R3
        @test to_spline(ZerosBC()) == CubicBSpline.R3

        # FixedBC convenience constructors
        @test to_spline(FixedBC()) == Dict("R3X" => 0)  # NaN sentinels → R3X
        @test to_spline(FixedBC(1.0, 0.0, 0.0)) == Dict("R3X" => 0)
        @test to_spline(FixedBC(0.0, 0.0, 0.0)) == CubicBSpline.R3  # Explicit zeros → R3

        # Periodic
        d = to_spline(PeriodicBC())
        @test d == CubicBSpline.PERIODIC

        # Robin: verify Ooyama conversion formula
        # λ = -β/α; α₁ = -4/(3λ+1), β₁ = (3λ-1)/(3λ+1)
        λ = 2.0
        d = to_spline(RobinBC(1.0, -λ, 0.0))
        @test d["α1"] ≈ -4.0 / (3λ + 1)
        @test d["β1"] ≈ (3λ - 1) / (3λ + 1)

        # Robin limiting case: λ → 0 should match Dirichlet
        d = to_spline(RobinBC(1.0, 0.0, 0.0))
        @test d["α1"] ≈ -4.0
        @test d["β1"] ≈ -1.0

        # Robin with ExponentialBC
        λ = 5.0
        d = to_spline(ExponentialBC(λ))
        @test d["α1"] ≈ -4.0 / (3λ + 1)
        @test d["β1"] ≈ (3λ - 1) / (3λ + 1)

        # Unsupported rank-2 combination (du + d2u without u)
        @test_throws ArgumentError to_spline(BoundaryConditions(nothing, 0.0, 0.0, nothing))
    end

    # ── Chebyshev conversion ─────────────────────────────────────────────
    @testset "Chebyshev Dict conversion" begin
        to_cheb = Springsteel._bc_to_chebyshev_dict

        # Natural → R0
        @test to_cheb(NaturalBC()) == Chebyshev.R0

        # Dirichlet → α0 key with value
        d = to_cheb(DirichletBC(0.0))
        @test d["α0"] ≈ 0.0

        d = to_cheb(DirichletBC(5.0))
        @test d["α0"] ≈ 5.0

        # Neumann → α1 key with value
        d = to_cheb(NeumannBC(0.0))
        @test d["α1"] ≈ 0.0

        d = to_cheb(NeumannBC(3.0))
        @test d["α1"] ≈ 3.0

        # SecondDerivative → α2 key
        d = to_cheb(SecondDerivativeBC(0.0))
        @test d["α2"] ≈ 0.0

        # Error cases
        @test_throws ArgumentError to_cheb(PeriodicBC())
        @test_throws ArgumentError to_cheb(RobinBC(1.0, -1.0))
        @test_throws ArgumentError to_cheb(CauchyBC(0.0, 0.0))
    end

    # ── Fourier validation ───────────────────────────────────────────────
    @testset "Fourier validation" begin
        validate = Springsteel._validate_fourier_bc
        @test validate(PeriodicBC()) == Fourier.PERIODIC
        @test_throws ArgumentError validate(NaturalBC())
        @test_throws ArgumentError validate(DirichletBC())
    end

    # ── Round-trip: BoundaryConditions produces identical grids ───────────
    @testset "Round-trip: 1D Spline (R)" begin
        # Grid with legacy Dict BCs
        gp_old = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 10.0,
            num_cells = 10,
            BCL = Dict("u" => CubicBSpline.R1T0, "default" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R1T0, "default" => CubicBSpline.R0),
            vars = Dict("u" => 1))
        grid_old = createGrid(gp_old)

        # Grid with new BoundaryConditions
        gp_new = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 10.0,
            num_cells = 10,
            BCL = Dict("u" => DirichletBC(), "default" => NaturalBC()),
            BCR = Dict("u" => DirichletBC(), "default" => NaturalBC()),
            vars = Dict("u" => 1))
        grid_new = createGrid(gp_new)

        # gammaBC matrices must match
        @test grid_old.ibasis.data[1,1].gammaBC ≈ grid_new.ibasis.data[1,1].gammaBC
    end

    @testset "Round-trip: 1D Spline with Neumann" begin
        gp_old = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 10.0, num_cells = 10,
            BCL = Dict("u" => CubicBSpline.R1T1, "default" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R1T1, "default" => CubicBSpline.R0),
            vars = Dict("u" => 1))
        grid_old = createGrid(gp_old)

        gp_new = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 10.0, num_cells = 10,
            BCL = Dict("u" => NeumannBC(), "default" => NaturalBC()),
            BCR = Dict("u" => NeumannBC(), "default" => NaturalBC()),
            vars = Dict("u" => 1))
        grid_new = createGrid(gp_new)

        @test grid_old.ibasis.data[1,1].gammaBC ≈ grid_new.ibasis.data[1,1].gammaBC
    end

    @testset "Round-trip: 1D Spline with R1T2" begin
        gp_old = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 10.0, num_cells = 10,
            BCL = Dict("u" => CubicBSpline.R1T2, "default" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0, "default" => CubicBSpline.R0),
            vars = Dict("u" => 1))
        grid_old = createGrid(gp_old)

        gp_new = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 10.0, num_cells = 10,
            BCL = Dict("u" => SecondDerivativeBC(), "default" => NaturalBC()),
            BCR = Dict("u" => NaturalBC(), "default" => NaturalBC()),
            vars = Dict("u" => 1))
        grid_new = createGrid(gp_new)

        @test grid_old.ibasis.data[1,1].gammaBC ≈ grid_new.ibasis.data[1,1].gammaBC
    end

    @testset "Round-trip: 1D Spline with R2T10 (Cauchy)" begin
        gp_old = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 10.0, num_cells = 10,
            BCL = Dict("u" => CubicBSpline.R2T10, "default" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0, "default" => CubicBSpline.R0),
            vars = Dict("u" => 1))
        grid_old = createGrid(gp_old)

        gp_new = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 10.0, num_cells = 10,
            BCL = Dict("u" => CauchyBC(0.0, 0.0), "default" => NaturalBC()),
            BCR = Dict("u" => NaturalBC(), "default" => NaturalBC()),
            vars = Dict("u" => 1))
        grid_new = createGrid(gp_new)

        @test grid_old.ibasis.data[1,1].gammaBC ≈ grid_new.ibasis.data[1,1].gammaBC
    end

    @testset "Round-trip: 1D Spline with R2T20 (Antisymmetric)" begin
        gp_old = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 10.0, num_cells = 10,
            BCL = Dict("u" => CubicBSpline.R2T20, "default" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0, "default" => CubicBSpline.R0),
            vars = Dict("u" => 1))
        grid_old = createGrid(gp_old)

        gp_new = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 10.0, num_cells = 10,
            BCL = Dict("u" => AntisymmetricBC(), "default" => NaturalBC()),
            BCR = Dict("u" => NaturalBC(), "default" => NaturalBC()),
            vars = Dict("u" => 1))
        grid_new = createGrid(gp_new)

        @test grid_old.ibasis.data[1,1].gammaBC ≈ grid_new.ibasis.data[1,1].gammaBC
    end

    @testset "Round-trip: 1D Spline with R3" begin
        gp_old = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 10.0, num_cells = 10,
            BCL = Dict("u" => CubicBSpline.R3, "default" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0, "default" => CubicBSpline.R0),
            vars = Dict("u" => 1))
        grid_old = createGrid(gp_old)

        gp_new = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 10.0, num_cells = 10,
            BCL = Dict("u" => BoundaryConditions(0.0, 0.0, 0.0, nothing), "default" => NaturalBC()),
            BCR = Dict("u" => NaturalBC(), "default" => NaturalBC()),
            vars = Dict("u" => 1))
        grid_new = createGrid(gp_new)

        @test grid_old.ibasis.data[1,1].gammaBC ≈ grid_new.ibasis.data[1,1].gammaBC
    end

    @testset "Round-trip: 1D Spline with PERIODIC" begin
        gp_old = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 10.0, num_cells = 10,
            BCL = Dict("u" => CubicBSpline.PERIODIC, "default" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.PERIODIC, "default" => CubicBSpline.R0),
            vars = Dict("u" => 1))
        grid_old = createGrid(gp_old)

        gp_new = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 10.0, num_cells = 10,
            BCL = Dict("u" => PeriodicBC(), "default" => NaturalBC()),
            BCR = Dict("u" => PeriodicBC(), "default" => NaturalBC()),
            vars = Dict("u" => 1))
        grid_new = createGrid(gp_new)

        @test grid_old.ibasis.data[1,1].gammaBC ≈ grid_new.ibasis.data[1,1].gammaBC
    end

    @testset "Round-trip: 2D RZ (Spline × Chebyshev)" begin
        gp_old = SpringsteelGridParameters(
            geometry = "RZ",
            iMin = 0.0, iMax = 1.0, num_cells = 8,
            kMin = 0.0, kMax = 1.0, kDim = 10,
            BCL = Dict("u" => CubicBSpline.R1T0, "default" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R1T0, "default" => CubicBSpline.R0),
            BCB = Dict("u" => Chebyshev.R1T0, "default" => Chebyshev.R0),
            BCT = Dict("u" => Chebyshev.R1T0, "default" => Chebyshev.R0),
            vars = Dict("u" => 1))
        grid_old = createGrid(gp_old)

        gp_new = SpringsteelGridParameters(
            geometry = "RZ",
            iMin = 0.0, iMax = 1.0, num_cells = 8,
            kMin = 0.0, kMax = 1.0, kDim = 10,
            BCL = Dict("u" => DirichletBC(), "default" => NaturalBC()),
            BCR = Dict("u" => DirichletBC(), "default" => NaturalBC()),
            BCB = Dict("u" => DirichletBC(), "default" => NaturalBC()),
            BCT = Dict("u" => DirichletBC(), "default" => NaturalBC()),
            vars = Dict("u" => 1))
        grid_new = createGrid(gp_new)

        # Compare spline gammaBC
        @test grid_old.ibasis.data[1,1].gammaBC ≈ grid_new.ibasis.data[1,1].gammaBC
        # Compare Chebyshev BC params
        @test grid_old.kbasis.data[1].params.BCB == grid_new.kbasis.data[1].params.BCB
        @test grid_old.kbasis.data[1].params.BCT == grid_new.kbasis.data[1].params.BCT
    end

    @testset "Round-trip: 2D RR (Spline × Spline)" begin
        gp_old = SpringsteelGridParameters(
            geometry = "RR",
            iMin = 0.0, iMax = 1.0, num_cells = 8,
            jMin = 0.0, jMax = 1.0, jDim = 24, b_jDim = 11,
            BCL = Dict("u" => CubicBSpline.R1T0, "default" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R1T0, "default" => CubicBSpline.R0),
            BCU = Dict("u" => CubicBSpline.R1T1, "default" => CubicBSpline.R0),
            BCD = Dict("u" => CubicBSpline.R1T1, "default" => CubicBSpline.R0),
            vars = Dict("u" => 1))
        grid_old = createGrid(gp_old)

        gp_new = SpringsteelGridParameters(
            geometry = "RR",
            iMin = 0.0, iMax = 1.0, num_cells = 8,
            jMin = 0.0, jMax = 1.0, jDim = 24, b_jDim = 11,
            BCL = Dict("u" => DirichletBC(), "default" => NaturalBC()),
            BCR = Dict("u" => DirichletBC(), "default" => NaturalBC()),
            BCU = Dict("u" => NeumannBC(), "default" => NaturalBC()),
            BCD = Dict("u" => NeumannBC(), "default" => NaturalBC()),
            vars = Dict("u" => 1))
        grid_new = createGrid(gp_new)

        @test grid_old.ibasis.data[1,1].gammaBC ≈ grid_new.ibasis.data[1,1].gammaBC
        @test grid_old.jbasis.data[1,1].gammaBC ≈ grid_new.jbasis.data[1,1].gammaBC
    end

    @testset "Round-trip: 1D Chebyshev (Z)" begin
        gp_old = SpringsteelGridParameters(
            geometry = "Z",
            iMin = 0.0, iMax = 1.0, iDim = 10, b_iDim = 10,
            BCL = Dict("u" => Chebyshev.R1T0, "default" => Chebyshev.R0),
            BCR = Dict("u" => Chebyshev.R1T0, "default" => Chebyshev.R0),
            vars = Dict("u" => 1))
        grid_old = createGrid(gp_old)

        gp_new = SpringsteelGridParameters(
            geometry = "Z",
            iMin = 0.0, iMax = 1.0, iDim = 10, b_iDim = 10,
            BCL = Dict("u" => DirichletBC(), "default" => NaturalBC()),
            BCR = Dict("u" => DirichletBC(), "default" => NaturalBC()),
            vars = Dict("u" => 1))
        grid_new = createGrid(gp_new)

        @test grid_old.ibasis.data[1].params.BCB == grid_new.ibasis.data[1].params.BCB
        @test grid_old.ibasis.data[1].params.BCT == grid_new.ibasis.data[1].params.BCT
    end

    # ── Per-variable mixing ──────────────────────────────────────────────
    @testset "Per-variable BC mixing" begin
        gp = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 10.0, num_cells = 10,
            BCL = Dict("u" => DirichletBC(), "v" => NeumannBC(), "default" => NaturalBC()),
            BCR = Dict("u" => DirichletBC(), "v" => NeumannBC(), "default" => NaturalBC()),
            vars = Dict("u" => 1, "v" => 2))
        grid = createGrid(gp)

        # "u" should have Dirichlet gammaBC
        gp_u_ref = SpringsteelGridParameters(
            geometry = "R", iMin = 0.0, iMax = 10.0, num_cells = 10,
            BCL = Dict("default" => CubicBSpline.R1T0),
            BCR = Dict("default" => CubicBSpline.R1T0),
            vars = Dict("u" => 1))
        grid_u_ref = createGrid(gp_u_ref)
        @test grid.ibasis.data[1,1].gammaBC ≈ grid_u_ref.ibasis.data[1,1].gammaBC

        # "v" should have Neumann gammaBC
        gp_v_ref = SpringsteelGridParameters(
            geometry = "R", iMin = 0.0, iMax = 10.0, num_cells = 10,
            BCL = Dict("default" => CubicBSpline.R1T1),
            BCR = Dict("default" => CubicBSpline.R1T1),
            vars = Dict("v" => 1))
        grid_v_ref = createGrid(gp_v_ref)
        @test grid.ibasis.data[1,2].gammaBC ≈ grid_v_ref.ibasis.data[1,1].gammaBC
    end

    # ── Mixed old/new BCs in same Dict ─────────────────────────────────
    @testset "Mixed Dict/BoundaryConditions in per-variable Dict" begin
        gp = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 10.0, num_cells = 10,
            BCL = Dict("u" => DirichletBC(), "v" => CubicBSpline.R1T1, "default" => NaturalBC()),
            BCR = Dict("u" => DirichletBC(), "v" => CubicBSpline.R1T1, "default" => NaturalBC()),
            vars = Dict("u" => 1, "v" => 2))
        grid = createGrid(gp)

        # Both should produce valid grids with correct gammaBC
        @test size(grid.ibasis.data[1,1].gammaBC, 1) > 0
        @test size(grid.ibasis.data[1,2].gammaBC, 1) > 0
    end

    # ── Backward compatibility: Dict BCs unchanged ───────────────────────
    @testset "Backward compatibility" begin
        # Old-style Dict BCs must still work identically
        gp = SpringsteelGridParameters(
            geometry = "R",
            iMin = 0.0, iMax = 10.0, num_cells = 10,
            BCL = Dict("u" => CubicBSpline.R1T0, "default" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0, "default" => CubicBSpline.R0),
            vars = Dict("u" => 1))
        grid = createGrid(gp)
        @test size(grid.ibasis.data[1,1].gammaBC, 1) > 0
    end

end

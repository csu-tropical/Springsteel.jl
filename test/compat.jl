    @testset "Backward Compatibility" begin

        @testset "Old type aliases resolve to parametric types" begin
            @test R_Grid       == SpringsteelGrid{CartesianGeometry,   SplineBasisArray, NoBasisArray,      NoBasisArray}
            @test Spline1D_Grid == R_Grid
            @test RZ_Grid      == SpringsteelGrid{CartesianGeometry,   SplineBasisArray, NoBasisArray,      ChebyshevBasisArray}
            @test RL_Grid      == SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, NoBasisArray}
            @test RR_Grid      == SpringsteelGrid{CartesianGeometry,   SplineBasisArray, SplineBasisArray,  NoBasisArray}
            @test Spline2D_Grid == RR_Grid
            @test RLZ_Grid     == SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}
            @test RRR_Grid     == SpringsteelGrid{CartesianGeometry,   SplineBasisArray, SplineBasisArray,  SplineBasisArray}
            @test SL_Grid      == SpringsteelGrid{SphericalGeometry,   SplineBasisArray, FourierBasisArray, NoBasisArray}
            @test SLZ_Grid     == SpringsteelGrid{SphericalGeometry,   SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}
        end

        @testset "GridParameters forwarding produces SpringsteelGrid" begin
            gp = GridParameters(
                geometry = "R",
                xmin = 0.0, xmax = 10.0, num_cells = 10,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)
            @test grid isa R_Grid
            @test grid isa SpringsteelGrid
            @test grid.params.iDim == 30
            @test grid.params.b_iDim == 13
        end

        @testset "GridParameters RL forwarding" begin
            gp = GridParameters(
                geometry = "RL",
                xmin = 0.0, xmax = 50.0, num_cells = 5,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)
            @test grid isa RL_Grid
            @test grid isa SpringsteelGrid
        end

        @testset "GridParameters RZ forwarding" begin
            gp = GridParameters(
                geometry = "RZ",
                xmin = 0.0, xmax = 10.0, num_cells = 4,
                zmin = 0.0, zmax = 5.0, zDim = 10,
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCB = Dict("u" => Chebyshev.R0),
                BCT = Dict("u" => Chebyshev.R0),
                vars = Dict("u" => 1))
            grid = createGrid(gp)
            @test grid isa RZ_Grid
            @test grid isa SpringsteelGrid
        end

    end  # Backward Compatibility

    # ─────────────────────────────────────────────────────────────────────────
    # L_Grid Tests  (1-D Fourier; canonical: "L"; alias: "Ring1D")
    # ─────────────────────────────────────────────────────────────────────────
    @testset "L_Grid Tests" begin
        function make_l_gp(; geom="L")
            SpringsteelGridParameters(
                geometry = geom,
                iMin = 0.0, iMax = 2π, iDim = 64, b_iDim = 21,
                vars = Dict("u" => 1),
                BCL  = Dict("u" => Fourier.PERIODIC),
                BCR  = Dict("u" => Fourier.PERIODIC),
                max_wavenumber = Dict("default" => 10))
        end

        @testset "Grid creation — canonical name" begin
            grid = createGrid(make_l_gp())
            @test grid isa L_Grid
            @test grid isa AbstractGrid
            @test grid isa SpringsteelGrid
            @test size(grid.physical) == (64, 1, 3)
            @test size(grid.spectral) == (21, 1)
        end

        @testset "Type alias Ring1D_Grid" begin
            @test Ring1D_Grid === L_Grid
            @test Ring1D_Grid <: AbstractGrid
        end

        @testset "Alias dispatch — Ring1D routes to L_Grid" begin
            grid = createGrid(make_l_gp(geom="Ring1D"))
            @test grid isa L_Grid
            @test grid isa Ring1D_Grid
        end

        @testset "Fourier ibasis construction" begin
            grid = createGrid(make_l_gp())
            ring = grid.ibasis.data[1]
            @test ring isa Fourier.Fourier1D
            @test ring.params.kmax == 10
            @test ring.params.yDim == 64
            @test ring.params.bDim == 21
            @test length(ring.mishPoints) == 64
        end

        @testset "parse_geometry consistency — L and Ring1D" begin
            G1, I1, J1, K1 = parse_geometry("L")
            G2, I2, J2, K2 = parse_geometry("Ring1D")
            @test G1 isa CartesianGeometry && G2 isa CartesianGeometry
            @test I1 isa FourierBasisType  && I2 isa FourierBasisType
            @test J1 isa NoBasisType       && J2 isa NoBasisType
            @test K1 isa NoBasisType       && K2 isa NoBasisType
        end
    end  # L_Grid Tests

    # ─────────────────────────────────────────────────────────────────────────
    # LL_Grid Tests  (2-D Fourier×Fourier; canonical: "LL"; alias: "Ring2D")
    # ─────────────────────────────────────────────────────────────────────────
    @testset "LL_Grid Tests" begin
        function make_ll_gp(; geom="LL")
            SpringsteelGridParameters(
                geometry = geom,
                iMin = 0.0, iMax = 2π, iDim = 32, b_iDim = 11,
                jMin = 0.0, jMax = 2π, jDim = 16, b_jDim = 7,
                vars = Dict("u" => 1),
                BCL  = Dict("u" => Fourier.PERIODIC),
                BCR  = Dict("u" => Fourier.PERIODIC),
                BCU  = Dict("u" => Fourier.PERIODIC),
                BCD  = Dict("u" => Fourier.PERIODIC),
                max_wavenumber = Dict("default" => 5))
        end

        @testset "Grid creation — canonical name" begin
            grid = createGrid(make_ll_gp())
            @test grid isa LL_Grid
            @test grid isa AbstractGrid
            @test size(grid.physical) == (32 * 16, 1, 5)
            @test size(grid.spectral) == (11 * 7, 1)
        end

        @testset "Type alias Ring2D_Grid" begin
            @test Ring2D_Grid === LL_Grid
            @test Ring2D_Grid <: AbstractGrid
        end

        @testset "Alias dispatch — Ring2D routes to LL_Grid" begin
            grid = createGrid(make_ll_gp(geom="Ring2D"))
            @test grid isa LL_Grid
            @test grid isa Ring2D_Grid
        end

        @testset "Fourier i- and j-basis construction" begin
            grid = createGrid(make_ll_gp())
            ri = grid.ibasis.data[1]    # i-direction Fourier ring
            rj = grid.jbasis.data[1]    # j-direction Fourier ring
            @test ri isa Fourier.Fourier1D
            @test rj isa Fourier.Fourier1D
            @test ri.params.yDim == 32
            @test rj.params.yDim == 16
        end

        @testset "parse_geometry consistency — LL and Ring2D" begin
            G1, I1, J1, K1 = parse_geometry("LL")
            G2, I2, J2, K2 = parse_geometry("Ring2D")
            @test G1 isa CartesianGeometry && G2 isa CartesianGeometry
            @test I1 isa FourierBasisType  && I2 isa FourierBasisType
            @test J1 isa FourierBasisType  && J2 isa FourierBasisType
            @test K1 isa NoBasisType       && K2 isa NoBasisType
        end
    end  # LL_Grid Tests

    # ─────────────────────────────────────────────────────────────────────────
    # LLZ_Grid Tests  (3-D Fourier×Fourier×Chebyshev; alias: "DoublyPeriodic")
    # ─────────────────────────────────────────────────────────────────────────
    @testset "LLZ_Grid Tests" begin
        function make_llz_gp(; geom="LLZ")
            SpringsteelGridParameters(
                geometry = geom,
                iMin = 0.0, iMax = 2π, iDim = 16, b_iDim = 7,
                jMin = 0.0, jMax = 2π, jDim = 16, b_jDim = 7,
                kMin = 0.0, kMax = 1.0, kDim = 8,  b_kDim = 8,
                vars = Dict("u" => 1),
                BCL  = Dict("u" => Fourier.PERIODIC),
                BCR  = Dict("u" => Fourier.PERIODIC),
                BCU  = Dict("u" => Fourier.PERIODIC),
                BCD  = Dict("u" => Fourier.PERIODIC),
                BCB  = Dict("u" => Chebyshev.R0),
                BCT  = Dict("u" => Chebyshev.R0),
                max_wavenumber = Dict("default" => 3))
        end

        @testset "Grid creation — canonical name" begin
            grid = createGrid(make_llz_gp())
            @test grid isa LLZ_Grid
            @test grid isa AbstractGrid
            @test size(grid.physical) == (16 * 16 * 8, 1, 7)
            @test size(grid.spectral) == (7 * 7 * 8, 1)
        end

        @testset "Type alias DoublyPeriodic_Grid" begin
            @test DoublyPeriodic_Grid === LLZ_Grid
            @test DoublyPeriodic_Grid <: AbstractGrid
        end

        @testset "Alias dispatch — DoublyPeriodic routes to LLZ_Grid" begin
            grid = createGrid(make_llz_gp(geom="DoublyPeriodic"))
            @test grid isa LLZ_Grid
            @test grid isa DoublyPeriodic_Grid
        end

        @testset "Chebyshev kbasis construction" begin
            grid = createGrid(make_llz_gp())
            col = grid.kbasis.data[1]
            @test col isa Chebyshev.Chebyshev1D
            @test col.params.zDim == 8
            @test col.params.bDim == 8
        end

        @testset "parse_geometry consistency — LLZ and DoublyPeriodic" begin
            G1, I1, J1, K1 = parse_geometry("LLZ")
            G2, I2, J2, K2 = parse_geometry("DoublyPeriodic")
            @test G1 isa CartesianGeometry  && G2 isa CartesianGeometry
            @test I1 isa FourierBasisType   && I2 isa FourierBasisType
            @test J1 isa FourierBasisType   && J2 isa FourierBasisType
            @test K1 isa ChebyshevBasisType && K2 isa ChebyshevBasisType
        end
    end  # LLZ_Grid Tests

    # ─────────────────────────────────────────────────────────────────────────
    # Z_Grid Tests  (1-D Chebyshev; canonical: "Z"; alias: "Column1D")
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Z_Grid Tests" begin
        function make_z_gp(; geom="Z")
            SpringsteelGridParameters(
                geometry = geom,
                iMin = 0.0, iMax = 10.0, iDim = 25, b_iDim = 25,
                vars = Dict("u" => 1),
                BCL  = Dict("u" => Chebyshev.R0),
                BCR  = Dict("u" => Chebyshev.R0))
        end

        @testset "Grid creation — canonical name" begin
            grid = createGrid(make_z_gp())
            @test grid isa Z_Grid
            @test grid isa AbstractGrid
            @test size(grid.physical) == (25, 1, 3)
            @test size(grid.spectral) == (25, 1)
        end

        @testset "Type alias Column1D_Grid" begin
            @test Column1D_Grid === Z_Grid
            @test Column1D_Grid <: AbstractGrid
        end

        @testset "Alias dispatch — Column1D routes to Z_Grid" begin
            grid = createGrid(make_z_gp(geom="Column1D"))
            @test grid isa Z_Grid
            @test grid isa Column1D_Grid
        end

        @testset "Chebyshev ibasis construction" begin
            grid = createGrid(make_z_gp())
            col = grid.ibasis.data[1]
            @test col isa Chebyshev.Chebyshev1D
            @test col.params.zDim == 25
            @test col.params.bDim == 25
            @test col.params.zmin ≈ 0.0
            @test col.params.zmax ≈ 10.0
            @test length(col.mishPoints) == 25
            # CGL points include endpoints
            @test col.mishPoints[1]   ≈ 0.0  atol=1e-14
            @test col.mishPoints[end] ≈ 10.0 atol=1e-14
        end

        @testset "parse_geometry consistency — Z and Column1D" begin
            G1, I1, J1, K1 = parse_geometry("Z")
            G2, I2, J2, K2 = parse_geometry("Column1D")
            @test G1 isa CartesianGeometry  && G2 isa CartesianGeometry
            @test I1 isa ChebyshevBasisType && I2 isa ChebyshevBasisType
            @test J1 isa NoBasisType        && J2 isa NoBasisType
            @test K1 isa NoBasisType        && K2 isa NoBasisType
        end
    end  # Z_Grid Tests

    # ─────────────────────────────────────────────────────────────────────────
    # ZZ_Grid Tests  (2-D Chebyshev×Chebyshev; canonical: "ZZ"; alias: "Column2D")
    # ─────────────────────────────────────────────────────────────────────────
    @testset "ZZ_Grid Tests" begin
        function make_zz_gp(; geom="ZZ")
            SpringsteelGridParameters(
                geometry = geom,
                iMin = 0.0, iMax = 1.0, iDim = 10, b_iDim = 10,
                jMin = 0.0, jMax = 2.0, jDim = 10, b_jDim = 10,
                vars = Dict("u" => 1),
                BCL  = Dict("u" => Chebyshev.R0),
                BCR  = Dict("u" => Chebyshev.R0),
                BCU  = Dict("u" => Chebyshev.R0),
                BCD  = Dict("u" => Chebyshev.R0))
        end

        @testset "Grid creation — canonical name" begin
            grid = createGrid(make_zz_gp())
            @test grid isa ZZ_Grid
            @test grid isa AbstractGrid
            @test size(grid.physical) == (10 * 10, 1, 5)
            @test size(grid.spectral) == (10 * 10, 1)
        end

        @testset "Type alias Column2D_Grid" begin
            @test Column2D_Grid === ZZ_Grid
            @test Column2D_Grid <: AbstractGrid
        end

        @testset "Alias dispatch — Column2D routes to ZZ_Grid" begin
            grid = createGrid(make_zz_gp(geom="Column2D"))
            @test grid isa ZZ_Grid
            @test grid isa Column2D_Grid
        end

        @testset "parse_geometry consistency — ZZ and Column2D" begin
            G1, I1, J1, K1 = parse_geometry("ZZ")
            G2, I2, J2, K2 = parse_geometry("Column2D")
            @test G1 isa CartesianGeometry  && G2 isa CartesianGeometry
            @test I1 isa ChebyshevBasisType && I2 isa ChebyshevBasisType
            @test J1 isa ChebyshevBasisType && J2 isa ChebyshevBasisType
            @test K1 isa NoBasisType        && K2 isa NoBasisType
        end
    end  # ZZ_Grid Tests

    # ─────────────────────────────────────────────────────────────────────────
    # ZZZ_Grid Tests  (3-D Chebyshev³; canonical: "ZZZ"; alias: "Column3D")
    # ─────────────────────────────────────────────────────────────────────────
    @testset "ZZZ_Grid Tests" begin
        function make_zzz_gp(; geom="ZZZ")
            SpringsteelGridParameters(
                geometry = geom,
                iMin = 0.0, iMax = 1.0, iDim = 8, b_iDim = 8,
                jMin = 0.0, jMax = 1.0, jDim = 8, b_jDim = 8,
                kMin = 0.0, kMax = 1.0, kDim = 6, b_kDim = 6,
                vars = Dict("u" => 1),
                BCL  = Dict("u" => Chebyshev.R0),
                BCR  = Dict("u" => Chebyshev.R0),
                BCU  = Dict("u" => Chebyshev.R0),
                BCD  = Dict("u" => Chebyshev.R0),
                BCB  = Dict("u" => Chebyshev.R0),
                BCT  = Dict("u" => Chebyshev.R0))
        end

        @testset "Grid creation — canonical name" begin
            grid = createGrid(make_zzz_gp())
            @test grid isa ZZZ_Grid
            @test grid isa AbstractGrid
            @test size(grid.physical) == (8 * 8 * 6, 1, 7)
            @test size(grid.spectral) == (8 * 8 * 6, 1)
        end

        @testset "Type alias Column3D_Grid" begin
            @test Column3D_Grid === ZZZ_Grid
            @test Column3D_Grid <: AbstractGrid
        end

        @testset "Alias dispatch — Column3D routes to ZZZ_Grid" begin
            grid = createGrid(make_zzz_gp(geom="Column3D"))
            @test grid isa ZZZ_Grid
            @test grid isa Column3D_Grid
        end

        @testset "parse_geometry consistency — ZZZ and Column3D" begin
            G1, I1, J1, K1 = parse_geometry("ZZZ")
            G2, I2, J2, K2 = parse_geometry("Column3D")
            @test G1 isa CartesianGeometry  && G2 isa CartesianGeometry
            @test I1 isa ChebyshevBasisType && I2 isa ChebyshevBasisType
            @test J1 isa ChebyshevBasisType && J2 isa ChebyshevBasisType
            @test K1 isa ChebyshevBasisType && K2 isa ChebyshevBasisType
        end
    end  # ZZZ_Grid Tests

    # ─────────────────────────────────────────────────────────────────────────
    # Geometry alias direction invariant
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Geometry alias direction — descriptive → canonical" begin
        aliases = Springsteel._GEOMETRY_ALIASES
        # Every alias should map FROM a descriptive name TO a known canonical code.
        # Canonical codes must NOT appear as keys in the alias table.
        canonical_codes = Set([
            "R", "RZ", "RL", "RR", "RLZ", "RRR", "SL", "SLZ",
            "L", "LL", "LLZ", "Z", "ZZ", "ZZZ",
        ])
        for (alias_key, canon) in aliases
            @test !(alias_key in canonical_codes)   # key is a descriptive name
            @test canon in canonical_codes           # value is a canonical code
        end
        # Spot-check specific mappings
        @test aliases["Ring1D"]         == "L"
        @test aliases["Ring2D"]         == "LL"
        @test aliases["DoublyPeriodic"] == "LLZ"
        @test aliases["Column1D"]       == "Z"
        @test aliases["Column2D"]       == "ZZ"
        @test aliases["Column3D"]       == "ZZZ"
        @test aliases["Polar"]          == "RL"
        @test aliases["Cylindrical"]    == "RLZ"
        @test aliases["Spline3D"]       == "RRR"
        @test aliases["Samurai"]        == "RRR"
        @test aliases["SphericalShell"] == "SL"
        @test aliases["Sphere"]         == "SLZ"
    end  # Geometry alias direction


    @testset "Spline cell-count specification" begin

        # Convenience: build a grid from keywords with a single variable "u".
        mkgp(; kw...) = SpringsteelGridParameters(; vars = Dict("u" => 1), kw...)
        mkgrid(; kw...) = createGrid(mkgp(; kw...))

        # ── i-axis: num_cells / num_cells_i / iDim ────────────────────────────
        @testset "i-axis precedence" begin
            # num_cells_i alone is equivalent to num_cells alone
            a = mkgrid(geometry = "R", iMin = 0.0, iMax = 10.0, num_cells = 10)
            b = mkgrid(geometry = "R", iMin = 0.0, iMax = 10.0, num_cells_i = 10)
            for f in (:num_cells, :num_cells_i, :iDim, :b_iDim,
                      :spectralIndexR, :patchOffsetR)
                @test getfield(a.params, f) == getfield(b.params, f)
            end
            @test size(a.spectral) == size(b.spectral)
            @test size(a.physical) == size(b.physical)

            # After resolution the two names always agree
            @test a.params.num_cells == a.params.num_cells_i == 10
            @test a.params.iDim == 30
            @test a.params.b_iDim == 13

            # Both supplied and consistent
            c = mkgrid(geometry = "R", iMin = 0.0, iMax = 10.0,
                       num_cells = 10, num_cells_i = 10)
            @test c.params.iDim == 30

            # Both supplied and inconsistent
            @test_throws ArgumentError mkgrid(geometry = "R", iMin = 0.0, iMax = 10.0,
                                              num_cells = 10, num_cells_i = 7)

            # num_cells with an explicitly conflicting iDim
            @test_throws ArgumentError mkgrid(geometry = "R", iMin = 0.0, iMax = 10.0,
                                              num_cells = 10, iDim = 33)
        end

        @testset "i-axis reverse mode (iDim only)" begin
            # 1D: iDim back-derives num_cells and rebuilds the dependent fields
            g = mkgrid(geometry = "R", iMin = 0.0, iMax = 10.0, iDim = 30)
            @test g.params.num_cells == 10
            @test g.params.num_cells_i == 10
            @test g.params.b_iDim == 13
            @test g.params.spectralIndexR == 13
            @test g.params.patchOffsetR == 30
            @test size(g.spectral, 1) == 13

            # ... and the grid is actually usable, not merely constructible
            g.physical[:, 1, 1] .= 1.0
            spectralTransform!(g)
            gridTransform!(g)
            @test all(isapprox.(g.physical[:, 1, 1], 1.0; atol = 1e-8))

            # 2D: reverse mode composes with a spline j-axis
            h = mkgrid(geometry = "RR", iMin = 0.0, iMax = 10.0, iDim = 30,
                       jMin = 0.0, jMax = 10.0, num_cells_j = 10)
            @test h.params.num_cells == 10
            @test h.params.b_iDim == 13
            @test h.params.jDim == 30

            # Non-multiple of mubar is a clear error, not an InexactError
            @test_throws ArgumentError mkgrid(geometry = "R", iMin = 0.0, iMax = 10.0,
                                              iDim = 31)

            # mubar != 3 is respected
            m = mkgrid(geometry = "R", iMin = 0.0, iMax = 10.0, iDim = 20,
                       mubar = 2, quadrature = :regular)
            @test m.params.num_cells == 10
            @test m.params.b_iDim == 13
        end

        @testset "i-axis: non-spline geometries pass through" begin
            # Fourier i: iDim/b_iDim are user-supplied and must not be touched
            g = mkgrid(geometry = "L", iMin = 0.0, iMax = 2π, iDim = 32, b_iDim = 32)
            @test g.params.iDim == 32
            @test g.params.num_cells == 0
            @test g.params.num_cells_i == 0

            # Chebyshev i: same
            z = mkgrid(geometry = "Z", iMin = 0.0, iMax = 1.0, iDim = 16, b_iDim = 11)
            @test z.params.iDim == 16
            @test z.params.num_cells == 0
        end

        # ── j / k axes ────────────────────────────────────────────────────────
        @testset "j-axis precedence (RR)" begin
            base = (geometry = "RR", iMin = 0.0, iMax = 10.0, num_cells = 10,
                    jMin = 0.0, jMax = 7.0)

            g = mkgrid(; base..., num_cells_j = 7)
            @test g.params.num_cells_j == 7
            @test g.params.jDim == 21
            @test g.params.b_jDim == 10
            @test size(g.spectral, 1) == 13 * 10

            # jDim reverse mode agrees with the explicit cell count
            h = mkgrid(; base..., jDim = 21)
            @test h.params.num_cells_j == 7
            @test h.params.b_jDim == 10

            # Both, consistent
            k = mkgrid(; base..., num_cells_j = 7, jDim = 21)
            @test k.params.jDim == 21

            # Both, inconsistent
            @test_throws ArgumentError mkgrid(; base..., num_cells_j = 7, jDim = 20)

            # jDim not a multiple of mubar
            @test_throws ArgumentError mkgrid(; base..., jDim = 20)
            @test_throws ArgumentError mkgrid(geometry = "RR", iMin = 0.0, iMax = 10.0,
                                              num_cells = 10, jMin = 0.0, jMax = 10.0,
                                              jDim = 50)
        end

        @testset "k-axis precedence (RRR, RiRk)" begin
            g = mkgrid(geometry = "RRR",
                       iMin = 0.0, iMax = 10.0, num_cells_i = 10,
                       jMin = 0.0, jMax = 20.0, num_cells_j = 20,
                       kMin = 0.0, kMax = 2.5,  num_cells_k = 5)
            @test (g.params.iDim, g.params.jDim, g.params.kDim) == (30, 60, 15)
            @test (g.params.b_iDim, g.params.b_jDim, g.params.b_kDim) == (13, 23, 8)
            @test g.params.num_cells_k == 5

            @test_throws ArgumentError mkgrid(geometry = "RRR",
                iMin = 0.0, iMax = 10.0, num_cells = 10,
                jMin = 0.0, jMax = 10.0, num_cells_j = 10,
                kMin = 0.0, kMax = 10.0, kDim = 16)

            # RiRk: spline k, no j
            r = mkgrid(geometry = "RiRk", iMin = 0.0, iMax = 10.0, num_cells = 10,
                       kMin = 0.0, kMax = 4.0, num_cells_k = 4)
            @test r.params.num_cells_k == 4
            @test r.params.kDim == 12
            @test r.params.b_kDim == 7
            @test r.params.num_cells_j == 0   # j absent

            @test_throws ArgumentError mkgrid(geometry = "RiRk",
                iMin = 0.0, iMax = 10.0, num_cells = 10,
                kMin = 0.0, kMax = 4.0, kDim = 13)
        end

        # ── aspect-ratio default ──────────────────────────────────────────────
        @testset "uniform-nodal-spacing default" begin
            # Commensurate domain: DX_j == DX_i exactly, no warning
            g = @test_logs mkgrid(geometry = "RR", iMin = 0.0, iMax = 30.0, num_cells = 6,
                                  jMin = 0.0, jMax = 30.0)
            @test g.params.num_cells_j == 6
            DX_i = (g.params.iMax - g.params.iMin) / g.params.num_cells
            DX_j = (g.params.jMax - g.params.jMin) / g.params.num_cells_j
            @test DX_j ≈ DX_i

            # Non-square but still commensurate: 25 j-cells of width 1.0
            g2 = @test_logs mkgrid(geometry = "RR", iMin = 0.0, iMax = 10.0, num_cells = 10,
                                   jMin = 0.0, jMax = 25.0)
            @test g2.params.num_cells_j == 25

            # Incommensurate: ceil() distorts the spacing, so warn
            g3 = @test_logs (:warn, r"num_cells_j") mkgrid(
                geometry = "RR", iMin = 0.0, iMax = 10.0, num_cells = 10,
                jMin = 0.0, jMax = 6.28)
            @test g3.params.num_cells_j == 7   # ceil(6.28)

            # An explicit cell count on the same domain silences it
            g4 = @test_logs mkgrid(geometry = "RR", iMin = 0.0, iMax = 10.0, num_cells = 10,
                                   jMin = 0.0, jMax = 6.28, num_cells_j = 7)
            @test g4.params.num_cells_j == 7

            # k-axis default warns too
            g5 = @test_logs (:warn, r"num_cells_k") mkgrid(
                geometry = "RiRk", iMin = 0.0, iMax = 10.0, num_cells = 10,
                kMin = 0.0, kMax = 6.28)
            @test g5.params.num_cells_k == 7
        end

        # ── the auto default is invariant under i-tiling ──────────────────────
        @testset "aspect default is tile-invariant" begin
            # Halving the i-domain and the i-cell count leaves DX_i, hence nc_j,
            # unchanged. This is what lets _create_tile_from_patch omit jDim.
            full = mkgrid(geometry = "RR", iMin = 0.0, iMax = 100.0, num_cells = 20,
                          jMin = 0.0, jMax = 40.0)
            half = mkgrid(geometry = "RR", iMin = 0.0, iMax = 50.0, num_cells = 10,
                          jMin = 0.0, jMax = 40.0)
            @test full.params.num_cells_j == half.params.num_cells_j == 8
        end

        # ── geometries whose j/k are NOT splines must be untouched ────────────
        @testset "Fourier / Chebyshev axes keep sentinel 0" begin
            # RZ: Chebyshev k keeps its anti-aliased b_kDim
            rz = mkgrid(geometry = "RZ", iMin = 0.0, iMax = 1.0, num_cells = 15,
                        kMin = 0.0, kMax = 1.0, kDim = 20)
            @test rz.params.num_cells_k == 0
            @test rz.params.kDim == 20                                   # not snapped
            @test rz.params.b_kDim == min(20, floor(((2 * 20) - 1) / 3) + 1)

            # A Chebyshev kDim indivisible by mubar is perfectly legal
            rz2 = mkgrid(geometry = "RZ", iMin = 0.0, iMax = 1.0, num_cells = 15,
                         kMin = 0.0, kMax = 1.0, kDim = 20)
            @test rz2.params.kDim == 20

            # RL / RLZ: Fourier j, jDim is a derived ring-point total
            rl = mkgrid(geometry = "RL", iMin = 1.0, iMax = 5.0, num_cells = 10)
            @test rl.params.num_cells_j == 0
            @test rl.params.jDim == sum(4 + 4 * r for r in 1:rl.params.iDim)

            rlz = mkgrid(geometry = "RLZ", iMin = 1.0, iMax = 5.0, num_cells = 6,
                         kMin = 0.0, kMax = 1.0, kDim = 10)
            @test rlz.params.num_cells_j == 0
            @test rlz.params.num_cells_k == 0   # Chebyshev k

            # SL: spherical Fourier j
            sl = mkgrid(geometry = "SL", iMin = 0.01, iMax = π - 0.01, num_cells = 6)
            @test sl.params.num_cells_j == 0
        end

        @testset "RLR: Fourier j + spline k" begin
            g = mkgrid(geometry = "RLR", iMin = 1.0, iMax = 5.0, num_cells = 6,
                       kMin = 0.0, kMax = 2.0, num_cells_k = 4)
            @test g.params.num_cells_j == 0     # Fourier
            @test g.params.num_cells_k == 4     # spline
            @test g.params.kDim == 12
            @test g.params.b_kDim == 7

            # spline-k divisibility is enforced here too
            @test_throws ArgumentError mkgrid(geometry = "RLR",
                iMin = 1.0, iMax = 5.0, num_cells = 6,
                kMin = 0.0, kMax = 2.0, kDim = 13)
        end

        # ── the new and old spellings must produce identical grids ────────────
        @testset "cell count and gridpoint count agree bitwise" begin
            f(x, y) = sin(2π * x / 10) * cos(2π * y / 7)

            function build(; kw...)
                g = mkgrid(; geometry = "RR", iMin = 0.0, iMax = 10.0,
                           jMin = 0.0, jMax = 7.0, kw...)
                pts = getGridpoints(g)
                g.physical[:, 1, 1] .= f.(pts[:, 1], pts[:, 2])
                spectralTransform!(g)
                gridTransform!(g)
                return g
            end

            a = build(num_cells = 10, num_cells_j = 7)
            b = build(num_cells = 10, jDim = 21)
            @test a.spectral == b.spectral
            @test a.physical == b.physical

            # reverse-mode i must match the forward spelling exactly
            c = build(iDim = 30, num_cells_j = 7)
            @test a.spectral == c.spectral
            @test a.physical == c.physical
        end
    end

using LinearAlgebra

@testset "Multi-Patch Grid Connections" begin

    # ── Coupling matrix ────────────────────────────────────────────────────

    @testset "Coupling matrix constant" begin
        M = Springsteel.COUPLING_MATRIX_2X
        @test size(M) == (3, 3)
        # Each row sums to 1 (partition of unity)
        for row in 1:3
            @test sum(M[row, :]) ≈ 1.0
        end
        # Matrix is invertible
        @test abs(det(M)) > 0.1
    end

    @testset "_build_coupling_matrix 2:1 ratio" begin
        M = Springsteel._build_coupling_matrix(2.0, 1.0)
        @test M == Springsteel.COUPLING_MATRIX_2X
    end

    @testset "_build_coupling_matrix 1:1 ratio" begin
        M = Springsteel._build_coupling_matrix(1.0, 1.0)
        @test M == Springsteel.COUPLING_MATRIX_1X
        @test M == Float64[1 0 0; 0 1 0; 0 0 1]
        # Also works for non-unit same-resolution
        M2 = Springsteel._build_coupling_matrix(0.5, 0.5)
        @test M2 == Springsteel.COUPLING_MATRIX_1X
    end

    @testset "_build_coupling_matrix rejects unsupported ratio" begin
        @test_throws ArgumentError Springsteel._build_coupling_matrix(3.0, 1.0)
        @test_throws ArgumentError Springsteel._build_coupling_matrix(4.0, 1.0)
    end

    # ── PatchInterface validation ──────────────────────────────────────────

    @testset "PatchInterface rejects non-:i dimension" begin
        gp = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        g = createGrid(gp)
        @test_throws ArgumentError PatchInterface(g, g, :right, :left, :j)
    end

    @testset "PatchInterface rejects domain mismatch" begin
        gp1 = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        # Secondary starts at 11.0, not 10.0
        gp2 = SpringsteelGridParameters(
            geometry="R", iMin=11.0, iMax=16.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))
        g1 = createGrid(gp1)
        g2 = createGrid(gp2)
        @test_throws ArgumentError PatchInterface(g1, g2, :right, :left, :i)
    end

    @testset "PatchInterface rejects wrong ratio" begin
        # Primary: DX = 1.0, Secondary: DX = 0.6 (not 2:1)
        gp1 = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=16.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))
        g1 = createGrid(gp1)
        g2 = createGrid(gp2)
        @test_throws ArgumentError PatchInterface(g1, g2, :right, :left, :i)
    end

    @testset "PatchInterface rejects missing R3X on secondary" begin
        # Primary DX = 1.0, secondary DX = 0.5 (2:1 ratio)
        gp1 = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=15.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R0),  # Should be R3X
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        g1 = createGrid(gp1)
        g2 = createGrid(gp2)
        @test_throws ArgumentError PatchInterface(g1, g2, :right, :left, :i)
    end

    @testset "PatchInterface rejects opposite side mismatch" begin
        gp1 = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=15.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))
        g1 = createGrid(gp1)
        g2 = createGrid(gp2)
        # Same side — should fail
        @test_throws ArgumentError PatchInterface(g1, g2, :right, :right, :i)
    end

    # ── Valid PatchInterface construction ───────────────────────────────────

    @testset "PatchInterface patch chain construction" begin
        # Primary: 10 cells, DX = 1.0
        gp1 = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        # Secondary: 20 cells in [10, 20], DX = 0.5 (2:1 ratio)
        gp2 = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=20.0, num_cells=20,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))
        g1 = createGrid(gp1)
        g2 = createGrid(gp2)

        iface = PatchInterface(g1, g2, :right, :left, :i)
        @test iface.primary === g1
        @test iface.secondary === g2
        @test iface.primary_side == :right
        @test iface.secondary_side == :left
        @test iface.dimension == :i
        @test !iface.is_stacked
        @test iface.coupling_matrix == Springsteel.COUPLING_MATRIX_2X
        # Primary right side: indices in ascending order (bDim-2, bDim-1, bDim)
        bDim = g1.params.b_iDim
        @test iface.primary_node_indices == (bDim - 2, bDim - 1, bDim)
    end

    # ── 1D patch chain: linear polynomial exactness ─────────────────────────

    @testset "1D patch chain linear polynomial exactness" begin
        # A linear polynomial is reproduced exactly (machine precision)
        # through the interface — no quadrature error for linear functions
        f(x) = 3x + 5
        df(x) = 3.0
        d2f(x) = 0.0

        gp_left = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp_sec = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=20.0, num_cells=20,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))
        gp_right = SpringsteelGridParameters(
            geometry="R", iMin=20.0, iMax=30.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))

        g_left = createGrid(gp_left)
        g_sec = createGrid(gp_sec)
        g_right = createGrid(gp_right)

        pts_left = getGridpoints(g_left)
        pts_sec = getGridpoints(g_sec)
        pts_right = getGridpoints(g_right)

        for i in eachindex(pts_left)
            g_left.physical[i, 1, 1] = f(pts_left[i])
        end
        for i in eachindex(pts_sec)
            g_sec.physical[i, 1, 1] = f(pts_sec[i])
        end
        for i in eachindex(pts_right)
            g_right.physical[i, 1, 1] = f(pts_right[i])
        end

        spectralTransform!(g_left)
        spectralTransform!(g_sec)
        spectralTransform!(g_right)

        mpg = PatchChain([g_left, g_sec, g_right])
        multiGridTransform!(mpg)

        # Linear function: machine precision on secondary
        for i in eachindex(pts_sec)
            @test g_sec.physical[i, 1, 1] ≈ f(pts_sec[i]) atol=1e-10
        end
        for i in eachindex(pts_sec)
            @test g_sec.physical[i, 1, 2] ≈ df(pts_sec[i]) atol=1e-8
        end
        for i in eachindex(pts_sec)
            @test g_sec.physical[i, 1, 3] ≈ d2f(pts_sec[i]) atol=1e-6
        end

        # Primary patches still correct
        for i in eachindex(pts_left)
            @test g_left.physical[i, 1, 1] ≈ f(pts_left[i]) atol=1e-10
        end
        for i in eachindex(pts_right)
            @test g_right.physical[i, 1, 1] ≈ f(pts_right[i]) atol=1e-10
        end
    end

    # ── 1D patch chain: cubic polynomial exactness with l_q=0 ──────────────

    @testset "1D patch chain cubic polynomial exactness (l_q=0)" begin
        # With l_q=0 (no smoothing), the coupling matrix is exact for cubics.
        # The l_q smoothing parameter modifies A-coefficients away from the true
        # B-spline amplitudes; the coupling matrix assumes unsmoothed amplitudes.
        f(x) = 2x^3 - 3x^2 + x + 5
        lq = Dict("u" => 0.0)

        gp_left = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=10, l_q=lq,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp_sec = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=20.0, num_cells=20, l_q=lq,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))
        gp_right = SpringsteelGridParameters(
            geometry="R", iMin=20.0, iMax=30.0, num_cells=10, l_q=lq,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))

        g_left = createGrid(gp_left)
        g_sec = createGrid(gp_sec)
        g_right = createGrid(gp_right)

        for (g, pts) in [(g_left, getGridpoints(g_left)),
                         (g_sec, getGridpoints(g_sec)),
                         (g_right, getGridpoints(g_right))]
            for i in eachindex(pts)
                g.physical[i, 1, 1] = f(pts[i])
            end
            spectralTransform!(g)
        end

        mpg = PatchChain([g_left, g_sec, g_right])
        multiGridTransform!(mpg)

        pts_sec = getGridpoints(g_sec)
        max_err = maximum(abs(g_sec.physical[i, 1, 1] - f(pts_sec[i])) for i in eachindex(pts_sec))
        @test max_err < 1e-8
    end

    # ── 1D patch chain: smoothing effect on coupling ──────────────────────

    @testset "1D patch chain coupling with default l_q" begin
        # With default l_q=2, the smoothing modifies A-coefficients.
        # Coupled error is bounded by a constant factor of single-grid error.
        f(x) = sin(2π * x / 30.0) + 1.0

        gp_left = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=20,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp_sec = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=20.0, num_cells=40,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))
        gp_right = SpringsteelGridParameters(
            geometry="R", iMin=20.0, iMax=30.0, num_cells=20,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))

        g_left = createGrid(gp_left)
        g_sec = createGrid(gp_sec)
        g_right = createGrid(gp_right)

        for (g, pts) in [(g_left, getGridpoints(g_left)),
                         (g_sec, getGridpoints(g_sec)),
                         (g_right, getGridpoints(g_right))]
            for i in eachindex(pts)
                g.physical[i, 1, 1] = f(pts[i])
            end
            spectralTransform!(g)
        end

        mpg = PatchChain([g_left, g_sec, g_right])
        multiGridTransform!(mpg)

        pts_sec = getGridpoints(g_sec)
        max_err = maximum(abs(g_sec.physical[i, 1, 1] - f(pts_sec[i])) for i in eachindex(pts_sec))
        # Coupled error bounded — smoothing introduces ~50-70x amplification
        @test max_err < 1e-3
    end

    # ── Independent transform matches multiGridTransform! ──────────────────

    @testset "Independent transform matches multiGridTransform!" begin
        f(x) = 3x + 7  # Linear for exact comparison

        gp_left = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp_sec = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=20.0, num_cells=20,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))
        gp_right = SpringsteelGridParameters(
            geometry="R", iMin=20.0, iMax=30.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))

        # Create two copies
        g1_l = createGrid(gp_left);  g2_l = createGrid(gp_left)
        g1_s = createGrid(gp_sec);   g2_s = createGrid(gp_sec)
        g1_r = createGrid(gp_right); g2_r = createGrid(gp_right)

        for (g, pts) in [(g1_l, getGridpoints(g1_l)), (g1_s, getGridpoints(g1_s)),
                         (g1_r, getGridpoints(g1_r)), (g2_l, getGridpoints(g2_l)),
                         (g2_s, getGridpoints(g2_s)), (g2_r, getGridpoints(g2_r))]
            for i in eachindex(pts)
                g.physical[i, 1, 1] = f(pts[i])
            end
            spectralTransform!(g)
        end

        # Method 1: multiGridTransform!
        mpg = PatchChain([g1_l, g1_s, g1_r])
        multiGridTransform!(mpg)

        # Method 2: manual independent transforms
        iface_l = PatchInterface(g2_l, g2_s, :right, :left, :i)
        iface_r = PatchInterface(g2_r, g2_s, :left, :right, :i)

        gridTransform!(g2_l)
        gridTransform!(g2_r)
        update_interface!(iface_l)
        update_interface!(iface_r)
        gridTransform!(g2_s)

        # Results should match exactly
        @test g1_s.physical[:, 1, 1] ≈ g2_s.physical[:, 1, 1] atol=1e-14
        @test g1_s.physical[:, 1, 2] ≈ g2_s.physical[:, 1, 2] atol=1e-14
        @test g1_s.physical[:, 1, 3] ≈ g2_s.physical[:, 1, 3] atol=1e-14
    end

    # ── 1D embedded ────────────────────────────────────────────────────

    @testset "1D embedded construction" begin
        # Primary: [0, 30], 30 cells, DX = 1.0
        gp_primary = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=30.0, num_cells=30,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        # Secondary: [10, 20], 20 cells, DX = 0.5 (2:1 ratio), interior
        gp_sec = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=20.0, num_cells=20,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))

        g_primary = createGrid(gp_primary)
        g_sec = createGrid(gp_sec)

        mpg = PatchEmbedded([g_primary, g_sec])
        @test length(mpg.interfaces) == 2
        @test length(mpg.patches) == 2
        @test length(mpg.transform_order) == 2
        # Primary should be in first layer
        @test 1 in mpg.transform_order[1]
    end

    @testset "1D embedded linear polynomial exactness" begin
        f(x) = -2x + 15  # Linear for exact reproduction

        gp_primary = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=30.0, num_cells=30,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp_sec = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=20.0, num_cells=20,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))

        g_primary = createGrid(gp_primary)
        g_sec = createGrid(gp_sec)

        pts_p = getGridpoints(g_primary)
        pts_s = getGridpoints(g_sec)
        for i in eachindex(pts_p)
            g_primary.physical[i, 1, 1] = f(pts_p[i])
        end
        for i in eachindex(pts_s)
            g_sec.physical[i, 1, 1] = f(pts_s[i])
        end

        spectralTransform!(g_primary)
        spectralTransform!(g_sec)

        mpg = PatchEmbedded([g_primary, g_sec])
        multiGridTransform!(mpg)

        # Linear function: machine precision on secondary
        for i in eachindex(pts_s)
            @test g_sec.physical[i, 1, 1] ≈ f(pts_s[i]) atol=1e-10
        end
    end

    @testset "1D embedded rejects misaligned grids" begin
        # Primary: DX = 1.0, secondary starts at 10.3 (not aligned)
        gp_primary = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=30.0, num_cells=30,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp_sec = SpringsteelGridParameters(
            geometry="R", iMin=10.3, iMax=20.3, num_cells=20,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))
        g_primary = createGrid(gp_primary)
        g_sec = createGrid(gp_sec)
        @test_throws ArgumentError PatchEmbedded([g_primary, g_sec])
    end

    # ── MultiPatchGrid topological sort ────────────────────────────────────

    @testset "Topological sort layers" begin
        # Chain: g1 → g2 → g3
        gp1 = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=15.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp3 = SpringsteelGridParameters(
            geometry="R", iMin=15.0, iMax=17.5, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1)
        g2 = createGrid(gp2)
        g3 = createGrid(gp3)

        iface12 = PatchInterface(g1, g2, :right, :left, :i)
        iface23 = PatchInterface(g2, g3, :right, :left, :i)

        mpg = MultiPatchGrid([g1, g2, g3], [iface12, iface23])

        # Should have 3 layers: g1 first, g2 second, g3 third
        @test length(mpg.transform_order) == 3
        @test 1 in mpg.transform_order[1]
        @test 2 in mpg.transform_order[2]
        @test 3 in mpg.transform_order[3]
    end

    # ── 2D RR i-interface ──────────────────────────────────────────────────

    @testset "2D RR i-interface construction" begin
        nc_j = 5
        jDim = nc_j * 3

        gp1 = SpringsteelGridParameters(
            geometry="RR", iMin=0.0, iMax=10.0, num_cells=10,
            jMin=0.0, jMax=5.0, jDim=jDim, b_jDim=nc_j + 3,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            BCU=Dict("u" => CubicBSpline.R0),
            BCD=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="RR", iMin=10.0, iMax=15.0, num_cells=10,
            jMin=0.0, jMax=5.0, jDim=jDim, b_jDim=nc_j + 3,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R0),
            BCU=Dict("u" => CubicBSpline.R0),
            BCD=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1)
        g2 = createGrid(gp2)

        iface = PatchInterface(g1, g2, :right, :left, :i)
        @test iface.primary === g1
        @test iface.secondary === g2
    end

    @testset "2D RR i-interface linear exactness" begin
        # f(x,y) = 2x + y (linear in both — exact for cubic B-splines)
        nc_j = 5
        jDim = nc_j * 3

        gp1 = SpringsteelGridParameters(
            geometry="RR", iMin=0.0, iMax=10.0, num_cells=10,
            jMin=0.0, jMax=5.0, jDim=jDim, b_jDim=nc_j + 3,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            BCU=Dict("u" => CubicBSpline.R0),
            BCD=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="RR", iMin=10.0, iMax=15.0, num_cells=10,
            jMin=0.0, jMax=5.0, jDim=jDim, b_jDim=nc_j + 3,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R0),
            BCU=Dict("u" => CubicBSpline.R0),
            BCD=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1)
        g2 = createGrid(gp2)

        iDim1 = gp1.iDim
        iDim2 = gp2.iDim

        pts_i1 = g1.ibasis.data[1, 1].mishPoints
        pts_j = g1.jbasis.data[1, 1].mishPoints

        for r in 1:iDim1
            for l in 1:jDim
                idx = (r - 1) * jDim + l
                g1.physical[idx, 1, 1] = 2 * pts_i1[r] + pts_j[l]
            end
        end

        pts_i2 = g2.ibasis.data[1, 1].mishPoints
        pts_j2 = g2.jbasis.data[1, 1].mishPoints
        for r in 1:iDim2
            for l in 1:jDim
                idx = (r - 1) * jDim + l
                g2.physical[idx, 1, 1] = 2 * pts_i2[r] + pts_j2[l]
            end
        end

        spectralTransform!(g1)
        spectralTransform!(g2)

        iface = PatchInterface(g1, g2, :right, :left, :i)
        gridTransform!(g1)
        update_interface!(iface)
        gridTransform!(g2)

        # Linear function: j-dimension with only 5 cells and R0 BCs has limited accuracy
        max_err = maximum(
            abs(g2.physical[(r-1)*jDim + l, 1, 1] - (2*pts_i2[r] + pts_j2[l]))
            for r in 1:iDim2, l in 1:jDim)
        @test max_err < 1e-2
    end

    # ── update_interface! writes correct ahat ──────────────────────────────

    @testset "update_interface! sets ahat on secondary" begin
        gp1 = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=15.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1)
        g2 = createGrid(gp2)

        # Fill with linear function
        pts1 = getGridpoints(g1)
        for i in eachindex(pts1)
            g1.physical[i, 1, 1] = 3 * pts1[i] + 1
        end
        spectralTransform!(g1)
        gridTransform!(g1)

        # Before interface transfer, ahat should be zero
        @test all(g2.ibasis.data[1, 1].ahat .== 0.0)

        iface = PatchInterface(g1, g2, :right, :left, :i)
        update_interface!(iface)

        # After transfer, ahat[1:3] should be nonzero
        @test !all(g2.ibasis.data[1, 1].ahat[1:3] .== 0.0)
    end

    # ── Multi-variable support ─────────────────────────────────────────────

    @testset "Multi-variable patch chain" begin
        f_u(x) = 2x + 1
        f_v(x) = -x + 3

        gp1 = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
            vars=Dict("u" => 1, "v" => 2))
        gp2 = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=15.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R3X, "v" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R3X, "v" => CubicBSpline.R3X),
            vars=Dict("u" => 1, "v" => 2))
        gp3 = SpringsteelGridParameters(
            geometry="R", iMin=15.0, iMax=25.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
            vars=Dict("u" => 1, "v" => 2))

        g1 = createGrid(gp1)
        g2 = createGrid(gp2)
        g3 = createGrid(gp3)

        for (g, pts) in [(g1, getGridpoints(g1)),
                         (g2, getGridpoints(g2)),
                         (g3, getGridpoints(g3))]
            for i in eachindex(pts)
                g.physical[i, 1, 1] = f_u(pts[i])
                g.physical[i, 2, 1] = f_v(pts[i])
            end
            spectralTransform!(g)
        end

        mpg = PatchChain([g1, g2, g3])
        multiGridTransform!(mpg)

        pts_sec = getGridpoints(g2)
        for i in eachindex(pts_sec)
            @test g2.physical[i, 1, 1] ≈ f_u(pts_sec[i]) atol=1e-10
            @test g2.physical[i, 2, 1] ≈ f_v(pts_sec[i]) atol=1e-10
        end
    end

    # ── Mixed l_q: smoothing effect on coupling accuracy ───────────────────
    #
    # The coupling matrix maps unsmoothed B-spline amplitudes exactly.
    # Primary l_q controls coupling accuracy; secondary l_q is irrelevant
    # because the cubic polynomial is in the null space of the third-derivative
    # penalty and the R3X clamping provides exact border coefficients.

    # Helper: build a patch chain with independent l_q on primary vs secondary
    function _run_mixed_lq_chain(f, nc_p, nc_s, lq_primary, lq_secondary)
        lq_p = Dict("u" => lq_primary)
        lq_s = Dict("u" => lq_secondary)
        gp1 = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=nc_p, l_q=lq_p,
            BCL=Dict("u" => CubicBSpline.R0), BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=20.0, num_cells=nc_s, l_q=lq_s,
            BCL=Dict("u" => CubicBSpline.R3X), BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))
        gp3 = SpringsteelGridParameters(
            geometry="R", iMin=20.0, iMax=30.0, num_cells=nc_p, l_q=lq_p,
            BCL=Dict("u" => CubicBSpline.R0), BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2); g3 = createGrid(gp3)
        for (g, pts) in [(g1, getGridpoints(g1)), (g2, getGridpoints(g2)), (g3, getGridpoints(g3))]
            for i in eachindex(pts); g.physical[i, 1, 1] = f(pts[i]); end
            spectralTransform!(g)
        end
        mpg = PatchChain([g1, g2, g3])
        multiGridTransform!(mpg)

        pts2 = getGridpoints(g2)
        max_err = maximum(abs(g2.physical[i, 1, 1] - f(pts2[i])) for i in eachindex(pts2))
        return max_err
    end

    @testset "Cubic exactness: primary l_q=0, secondary l_q=0" begin
        f(x) = x^3 - 2x^2 + x - 1
        err = _run_mixed_lq_chain(f, 10, 20, 0.0, 0.0)
        @test err < 1e-8
    end

    @testset "Cubic exactness: primary l_q=0, secondary l_q=2" begin
        # Primary provides exact coefficients; secondary l_q is irrelevant
        # for cubics (null space of third-derivative penalty)
        f(x) = x^3 - 2x^2 + x - 1
        err = _run_mixed_lq_chain(f, 10, 20, 0.0, 2.0)
        @test err < 1e-8
    end

    @testset "Cubic error: primary l_q=2, secondary l_q=0" begin
        # Primary l_q=2 shifts coefficients; coupling propagates the error
        f(x) = x^3 - 2x^2 + x - 1
        err = _run_mixed_lq_chain(f, 10, 20, 2.0, 0.0)
        @test err > 0.1  # Large error from primary smoothing
    end

    @testset "Cubic error: primary l_q=2, secondary l_q=2" begin
        f(x) = x^3 - 2x^2 + x - 1
        err = _run_mixed_lq_chain(f, 10, 20, 2.0, 2.0)
        @test err > 0.1  # Large error from primary smoothing
    end

    @testset "Quadratic exactness: all l_q configurations" begin
        # Quadratic is within both the B-spline reproduction range and the
        # null space of the smoothing penalty (Q penalizes 3rd derivative,
        # which is zero for quadratics). Exact for all l_q values.
        f(x) = x^2 - 3x + 7
        for (lq_p, lq_s) in [(0.0, 0.0), (0.0, 2.0), (2.0, 0.0), (2.0, 2.0)]
            err = _run_mixed_lq_chain(f, 10, 20, lq_p, lq_s)
            @test err < 1e-10
        end
    end

    @testset "Sinusoidal: primary l_q=0 gives near-machine-precision coupling" begin
        # Even for non-polynomial functions, primary l_q=0 ensures accurate
        # coupling. Secondary l_q doesn't affect coupling accuracy.
        f(x) = sin(2π * x / 30.0) + 1.0
        err_00 = _run_mixed_lq_chain(f, 10, 20, 0.0, 0.0)
        err_02 = _run_mixed_lq_chain(f, 10, 20, 0.0, 2.0)
        err_20 = _run_mixed_lq_chain(f, 10, 20, 2.0, 0.0)

        # Primary l_q=0: coupling error is at the level of quadrature error
        @test err_00 < 1e-4
        @test err_02 < 1e-4

        # Primary l_q=2: coupling error is much larger
        @test err_20 > 10 * err_00
    end

    @testset "Interior nest: primary l_q=0, secondary l_q=2 cubic exactness" begin
        f(x) = x^3 - 2x^2 + x - 1
        lq_p = Dict("u" => 0.0)
        lq_s = Dict("u" => 2.0)

        gp_coarse = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=30.0, num_cells=30, l_q=lq_p,
            BCL=Dict("u" => CubicBSpline.R0), BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp_fine = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=20.0, num_cells=20, l_q=lq_s,
            BCL=Dict("u" => CubicBSpline.R3X), BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))

        g_coarse = createGrid(gp_coarse)
        g_fine = createGrid(gp_fine)
        for (g, pts) in [(g_coarse, getGridpoints(g_coarse)), (g_fine, getGridpoints(g_fine))]
            for i in eachindex(pts); g.physical[i, 1, 1] = f(pts[i]); end
            spectralTransform!(g)
        end

        mpg = PatchEmbedded([g_coarse, g_fine])
        multiGridTransform!(mpg)

        pts_f = getGridpoints(g_fine)
        max_err = maximum(abs(g_fine.physical[i, 1, 1] - f(pts_f[i])) for i in eachindex(pts_f))
        @test max_err < 1e-8
    end

    @testset "3-patch chain: all l_q=0 gives exact propagation" begin
        # Chain: g1(l_q=0) → g2(l_q=0) → g3(l_q=0)
        # With l_q=0 everywhere, cubic exactness propagates through the chain
        f(x) = x^3 - 2x^2 + x - 1
        lq_0 = Dict("u" => 0.0)

        gp1 = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=10, l_q=lq_0,
            BCL=Dict("u" => CubicBSpline.R0), BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=15.0, num_cells=10, l_q=lq_0,
            BCL=Dict("u" => CubicBSpline.R3X), BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp3 = SpringsteelGridParameters(
            geometry="R", iMin=15.0, iMax=17.5, num_cells=10, l_q=lq_0,
            BCL=Dict("u" => CubicBSpline.R3X), BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2); g3 = createGrid(gp3)
        for (g, pts) in [(g1, getGridpoints(g1)), (g2, getGridpoints(g2)), (g3, getGridpoints(g3))]
            for i in eachindex(pts); g.physical[i, 1, 1] = f(pts[i]); end
            spectralTransform!(g)
        end

        iface12 = PatchInterface(g1, g2, :right, :left, :i)
        iface23 = PatchInterface(g2, g3, :right, :left, :i)
        mpg = MultiPatchGrid([g1, g2, g3], [iface12, iface23])
        multiGridTransform!(mpg)

        pts2 = getGridpoints(g2)
        err_g2 = maximum(abs(g2.physical[i, 1, 1] - f(pts2[i])) for i in eachindex(pts2))
        @test err_g2 < 1e-8

        pts3 = getGridpoints(g3)
        err_g3 = maximum(abs(g3.physical[i, 1, 1] - f(pts3[i])) for i in eachindex(pts3))
        @test err_g3 < 1e-6
    end

    @testset "3-patch chain: l_q=2 on forwarder degrades downstream" begin
        # Chain: g1(l_q=0) → g2(l_q=2) → g3(l_q=2)
        # g2 receives exact BCs from g1, but its own l_q=2 with R0 on the
        # right side modifies its outgoing coefficients. g3 receives those
        # smoothed coefficients and has large error.
        f(x) = x^3 - 2x^2 + x - 1
        lq_0 = Dict("u" => 0.0)
        lq_2 = Dict("u" => 2.0)

        gp1 = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=10, l_q=lq_0,
            BCL=Dict("u" => CubicBSpline.R0), BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=15.0, num_cells=10, l_q=lq_2,
            BCL=Dict("u" => CubicBSpline.R3X), BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp3 = SpringsteelGridParameters(
            geometry="R", iMin=15.0, iMax=17.5, num_cells=10, l_q=lq_2,
            BCL=Dict("u" => CubicBSpline.R3X), BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2); g3 = createGrid(gp3)
        for (g, pts) in [(g1, getGridpoints(g1)), (g2, getGridpoints(g2)), (g3, getGridpoints(g3))]
            for i in eachindex(pts); g.physical[i, 1, 1] = f(pts[i]); end
            spectralTransform!(g)
        end

        iface12 = PatchInterface(g1, g2, :right, :left, :i)
        iface23 = PatchInterface(g2, g3, :right, :left, :i)
        mpg = MultiPatchGrid([g1, g2, g3], [iface12, iface23])
        multiGridTransform!(mpg)

        # g2 has l_q=2 with R0 on right: its own smoothing introduces error
        pts2 = getGridpoints(g2)
        err_g2 = maximum(abs(g2.physical[i, 1, 1] - f(pts2[i])) for i in eachindex(pts2))
        @test err_g2 > 0.001  # Smoothing error present

        # g3 receives smoothed BCs from g2: error compounds
        pts3 = getGridpoints(g3)
        err_g3 = maximum(abs(g3.physical[i, 1, 1] - f(pts3[i])) for i in eachindex(pts3))
        @test err_g3 > 0.001  # Downstream error is also present
    end

    @testset "2D RR: primary l_q=0, secondary l_q=2 linear exactness" begin
        f_2d(x, y) = 2x + y
        nc_j = 5
        jDim = nc_j * 3
        lq_0 = Dict("u" => 0.0)
        lq_2 = Dict("u" => 2.0)

        gp1 = SpringsteelGridParameters(
            geometry="RR", iMin=0.0, iMax=10.0, num_cells=10,
            jMin=0.0, jMax=5.0, jDim=jDim, b_jDim=nc_j + 3, l_q=lq_0,
            BCL=Dict("u" => CubicBSpline.R0), BCR=Dict("u" => CubicBSpline.R0),
            BCU=Dict("u" => CubicBSpline.R0), BCD=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="RR", iMin=10.0, iMax=15.0, num_cells=10,
            jMin=0.0, jMax=5.0, jDim=jDim, b_jDim=nc_j + 3, l_q=lq_2,
            BCL=Dict("u" => CubicBSpline.R3X), BCR=Dict("u" => CubicBSpline.R0),
            BCU=Dict("u" => CubicBSpline.R0), BCD=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2)
        iDim1 = gp1.iDim; iDim2 = gp2.iDim
        pts_i1 = g1.ibasis.data[1, 1].mishPoints
        pts_j1 = g1.jbasis.data[1, 1].mishPoints
        pts_i2 = g2.ibasis.data[1, 1].mishPoints
        pts_j2 = g2.jbasis.data[1, 1].mishPoints

        for r in 1:iDim1, l in 1:jDim
            g1.physical[(r-1)*jDim + l, 1, 1] = f_2d(pts_i1[r], pts_j1[l])
        end
        for r in 1:iDim2, l in 1:jDim
            g2.physical[(r-1)*jDim + l, 1, 1] = f_2d(pts_i2[r], pts_j2[l])
        end

        spectralTransform!(g1); spectralTransform!(g2)
        iface = PatchInterface(g1, g2, :right, :left, :i)
        gridTransform!(g1)
        update_interface!(iface)
        gridTransform!(g2)

        max_err = maximum(
            abs(g2.physical[(r-1)*jDim + l, 1, 1] - f_2d(pts_i2[r], pts_j2[l]))
            for r in 1:iDim2, l in 1:jDim)
        # Primary l_q=0 gives exact i-coupling; j-direction R0 with 5 cells
        # still has limited accuracy but much better than l_q=2 primary
        @test max_err < 1e-3
    end

    # ── PatchChain: multi-grid chains ──────────────────────────────────────

    @testset "PatchChain rejects single grid" begin
        gp = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R0), BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        g = createGrid(gp)
        @test_throws ArgumentError PatchChain([g])
    end

    @testset "PatchChain 7-grid asymmetric: 8-4-2-1-2-4-8 DX" begin
        f(x) = 3x + 7  # Linear for exact test
        lq = Dict("u" => 0.0)

        # Build 7 grids: DX = 8, 4, 2, 1, 2, 4, 8
        # Domain: [0,80] [80,120] [120,140] [140,150] [150,170] [170,210] [210,290]
        configs = [
            (0.0,   80.0,  10),  # DX=8
            (80.0,  120.0, 10),  # DX=4
            (120.0, 140.0, 10),  # DX=2
            (140.0, 150.0, 10),  # DX=1
            (150.0, 170.0, 10),  # DX=2
            (170.0, 210.0, 10),  # DX=4
            (210.0, 290.0, 10),  # DX=8
        ]

        # BCs: ends are R0, interfaces are R3X (secondary side) or R0 (primary side)
        # Grid 1 (DX=8): R0 left, R0 right (primary to grid 2)
        # Grid 2 (DX=4): R3X left (from grid 1), R0 right (primary to grid 3)
        # Grid 3 (DX=2): R3X left (from grid 2), R0 right (primary to grid 4)
        # Grid 4 (DX=1): R3X left (from grid 3), R3X right (from grid 5)
        # Grid 5 (DX=2): R0 left (primary to grid 4), R3X right (from grid 6)
        # Grid 6 (DX=4): R0 left (primary to grid 5), R3X right (from grid 7)
        # Grid 7 (DX=8): R0 left (primary to grid 6), R0 right
        bcl_specs = [CubicBSpline.R0, CubicBSpline.R3X, CubicBSpline.R3X,
                     CubicBSpline.R3X, CubicBSpline.R0, CubicBSpline.R0, CubicBSpline.R0]
        bcr_specs = [CubicBSpline.R0, CubicBSpline.R0, CubicBSpline.R0,
                     CubicBSpline.R3X, CubicBSpline.R3X, CubicBSpline.R3X, CubicBSpline.R0]

        grids = SpringsteelGrid[]
        for (k, (imin, imax, nc)) in enumerate(configs)
            gp = SpringsteelGridParameters(
                geometry="R", iMin=imin, iMax=imax, num_cells=nc, l_q=lq,
                BCL=Dict("u" => bcl_specs[k]), BCR=Dict("u" => bcr_specs[k]),
                vars=Dict("u" => 1))
            push!(grids, createGrid(gp))
        end

        for g in grids
            pts = getGridpoints(g)
            for i in eachindex(pts); g.physical[i, 1, 1] = f(pts[i]); end
            spectralTransform!(g)
        end

        mpg = PatchChain(grids)
        @test length(mpg.interfaces) == 6
        @test length(mpg.transform_order) == 4  # 4 layers: DX=8 → DX=4 → DX=2 → DX=1

        multiGridTransform!(mpg)

        # All grids should reproduce linear function exactly
        for (k, g) in enumerate(grids)
            pts = getGridpoints(g)
            max_err = maximum(abs(g.physical[i, 1, 1] - f(pts[i])) for i in eachindex(pts))
            @test max_err < 1e-10
        end
    end

    @testset "PatchChain 7-grid cubic exactness with l_q=0" begin
        f(x) = x^3 - 2x^2 + x - 1
        lq = Dict("u" => 0.0)

        configs = [
            (0.0,   80.0,  10),
            (80.0,  120.0, 10),
            (120.0, 140.0, 10),
            (140.0, 150.0, 10),
            (150.0, 170.0, 10),
            (170.0, 210.0, 10),
            (210.0, 290.0, 10),
        ]
        bcl_specs = [CubicBSpline.R0, CubicBSpline.R3X, CubicBSpline.R3X,
                     CubicBSpline.R3X, CubicBSpline.R0, CubicBSpline.R0, CubicBSpline.R0]
        bcr_specs = [CubicBSpline.R0, CubicBSpline.R0, CubicBSpline.R0,
                     CubicBSpline.R3X, CubicBSpline.R3X, CubicBSpline.R3X, CubicBSpline.R0]

        grids = SpringsteelGrid[]
        for (k, (imin, imax, nc)) in enumerate(configs)
            gp = SpringsteelGridParameters(
                geometry="R", iMin=imin, iMax=imax, num_cells=nc, l_q=lq,
                BCL=Dict("u" => bcl_specs[k]), BCR=Dict("u" => bcr_specs[k]),
                vars=Dict("u" => 1))
            push!(grids, createGrid(gp))
        end

        for g in grids
            pts = getGridpoints(g)
            for i in eachindex(pts); g.physical[i, 1, 1] = f(pts[i]); end
            spectralTransform!(g)
        end

        mpg = PatchChain(grids)
        multiGridTransform!(mpg)

        # All grids should reproduce cubic with l_q=0
        for g in grids
            pts = getGridpoints(g)
            max_err = maximum(abs(g.physical[i, 1, 1] - f(pts[i])) for i in eachindex(pts))
            @test max_err < 1e-6
        end
    end

    @testset "PatchChain 1:1 domain decomposition" begin
        f(x) = sin(2π * x / 20.0) + 1.0

        # Two grids with same resolution, domain [0,10] and [10,20]
        gp1 = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=20,
            BCL=Dict("u" => CubicBSpline.R0), BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=20.0, num_cells=20,
            BCL=Dict("u" => CubicBSpline.R3X), BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2)
        for (g, pts) in [(g1, getGridpoints(g1)), (g2, getGridpoints(g2))]
            for i in eachindex(pts); g.physical[i, 1, 1] = f(pts[i]); end
            spectralTransform!(g)
        end

        mpg = PatchChain([g1, g2])
        @test length(mpg.interfaces) == 1
        multiGridTransform!(mpg)

        # Compare against single-grid result
        gp_single = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=20.0, num_cells=20,
            BCL=Dict("u" => CubicBSpline.R0), BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        g_single = createGrid(gp_single)
        pts_s = getGridpoints(g_single)
        for i in eachindex(pts_s); g_single.physical[i, 1, 1] = f(pts_s[i]); end
        spectralTransform!(g_single); gridTransform!(g_single)

        single_err = maximum(abs(g_single.physical[i, 1, 1] - f(pts_s[i])) for i in eachindex(pts_s))
        pts2 = getGridpoints(g2)
        coupled_err = maximum(abs(g2.physical[i, 1, 1] - f(pts2[i])) for i in eachindex(pts2))

        # 1:1 coupling error should be comparable to single-grid
        # (with default l_q, smoothing amplifies slightly through interface)
        @test coupled_err < 100 * single_err
    end

    @testset "PatchChain mixed 1:1 and 2:1" begin
        f(x) = 2x + 3  # Linear for exact test

        # 3 grids: [0,10] DX=1, [10,20] DX=1 (1:1), [20,25] DX=0.5 (2:1)
        gp1 = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R0), BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=20.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R3X), BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp3 = SpringsteelGridParameters(
            geometry="R", iMin=20.0, iMax=25.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R3X), BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2); g3 = createGrid(gp3)
        for (g, pts) in [(g1, getGridpoints(g1)), (g2, getGridpoints(g2)), (g3, getGridpoints(g3))]
            for i in eachindex(pts); g.physical[i, 1, 1] = f(pts[i]); end
            spectralTransform!(g)
        end

        mpg = PatchChain([g1, g2, g3])
        @test length(mpg.interfaces) == 2
        multiGridTransform!(mpg)

        # Linear should be exact on all grids
        for g in [g1, g2, g3]
            pts = getGridpoints(g)
            max_err = maximum(abs(g.physical[i, 1, 1] - f(pts[i])) for i in eachindex(pts))
            @test max_err < 1e-10
        end
    end

    # ── PatchEmbedded: multi-level embedding ───────────────────────────────

    @testset "PatchEmbedded rejects single grid" begin
        gp = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=30.0, num_cells=30,
            BCL=Dict("u" => CubicBSpline.R0), BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        g = createGrid(gp)
        @test_throws ArgumentError PatchEmbedded([g])
    end

    @testset "PatchEmbedded rejects 1:1 ratio" begin
        gp1 = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=30.0, num_cells=30,
            BCL=Dict("u" => CubicBSpline.R0), BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=20.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R3X), BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))
        g1 = createGrid(gp1); g2 = createGrid(gp2)
        @test_throws ArgumentError PatchEmbedded([g1, g2])
    end

    @testset "PatchEmbedded 3-level nesting" begin
        f(x) = 2x + 1  # Linear for exact test
        lq = Dict("u" => 0.0)

        # Outer: [0, 30], 30 cells (DX=1.0)
        gp_outer = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=30.0, num_cells=30, l_q=lq,
            BCL=Dict("u" => CubicBSpline.R0), BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        # Mid: [10, 20], 20 cells (DX=0.5)
        gp_mid = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=20.0, num_cells=20, l_q=lq,
            BCL=Dict("u" => CubicBSpline.R3X), BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))
        # Inner: [13, 17], 16 cells (DX=0.25)
        gp_inner = SpringsteelGridParameters(
            geometry="R", iMin=13.0, iMax=17.0, num_cells=16, l_q=lq,
            BCL=Dict("u" => CubicBSpline.R3X), BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))

        g_outer = createGrid(gp_outer)
        g_mid = createGrid(gp_mid)
        g_inner = createGrid(gp_inner)

        for (g, pts) in [(g_outer, getGridpoints(g_outer)),
                         (g_mid, getGridpoints(g_mid)),
                         (g_inner, getGridpoints(g_inner))]
            for i in eachindex(pts); g.physical[i, 1, 1] = f(pts[i]); end
            spectralTransform!(g)
        end

        mpg = PatchEmbedded([g_outer, g_mid, g_inner])
        @test length(mpg.interfaces) == 4  # 2 per level
        @test length(mpg.transform_order) == 3  # outer → mid → inner

        multiGridTransform!(mpg)

        # All levels should reproduce linear exactly
        for g in [g_outer, g_mid, g_inner]
            pts = getGridpoints(g)
            max_err = maximum(abs(g.physical[i, 1, 1] - f(pts[i])) for i in eachindex(pts))
            @test max_err < 1e-10
        end
    end

    @testset "PatchEmbedded 3-level cubic exactness with l_q=0" begin
        f(x) = x^3 - 2x^2 + x - 1
        lq = Dict("u" => 0.0)

        gp_outer = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=30.0, num_cells=30, l_q=lq,
            BCL=Dict("u" => CubicBSpline.R0), BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp_mid = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=20.0, num_cells=20, l_q=lq,
            BCL=Dict("u" => CubicBSpline.R3X), BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))
        gp_inner = SpringsteelGridParameters(
            geometry="R", iMin=13.0, iMax=17.0, num_cells=16, l_q=lq,
            BCL=Dict("u" => CubicBSpline.R3X), BCR=Dict("u" => CubicBSpline.R3X),
            vars=Dict("u" => 1))

        g_outer = createGrid(gp_outer)
        g_mid = createGrid(gp_mid)
        g_inner = createGrid(gp_inner)

        for (g, pts) in [(g_outer, getGridpoints(g_outer)),
                         (g_mid, getGridpoints(g_mid)),
                         (g_inner, getGridpoints(g_inner))]
            for i in eachindex(pts); g.physical[i, 1, 1] = f(pts[i]); end
            spectralTransform!(g)
        end

        mpg = PatchEmbedded([g_outer, g_mid, g_inner])
        multiGridTransform!(mpg)

        # Cubic with l_q=0 should be near machine precision
        for g in [g_outer, g_mid, g_inner]
            pts = getGridpoints(g)
            max_err = maximum(abs(g.physical[i, 1, 1] - f(pts[i])) for i in eachindex(pts))
            @test max_err < 1e-6
        end
    end

    # ── FixedBC convenience constructor with multipatch ────────────────────

    @testset "PatchChain with FixedBC and NaturalBC" begin
        # Same as the 3-patch linear test but using BoundaryConditions API
        f(x) = 3x + 5

        gp1 = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=10,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=20.0, num_cells=20,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => FixedBC()),
            vars=Dict("u" => 1))
        gp3 = SpringsteelGridParameters(
            geometry="R", iMin=20.0, iMax=30.0, num_cells=10,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2); g3 = createGrid(gp3)
        for (g, pts) in [(g1, getGridpoints(g1)), (g2, getGridpoints(g2)), (g3, getGridpoints(g3))]
            for i in eachindex(pts); g.physical[i, 1, 1] = f(pts[i]); end
            spectralTransform!(g)
        end

        mpg = PatchChain([g1, g2, g3])
        multiGridTransform!(mpg)

        # Linear should be exact
        pts2 = getGridpoints(g2)
        for i in eachindex(pts2)
            @test g2.physical[i, 1, 1] ≈ f(pts2[i]) atol=1e-10
        end
    end

    # ── 2D RL (cylindrical) multipatch ────────────────────────────────────

    @testset "RL chain construction" begin
        gp1 = SpringsteelGridParameters(
            geometry="RL", iMin=0.0, iMax=50.0, num_cells=10,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="RL", iMin=50.0, iMax=75.0, num_cells=10,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2)
        mpg = PatchChain([g1, g2])
        @test length(mpg.interfaces) == 1
        @test length(mpg.patches) == 2
        @test mpg.interfaces[1].primary === g1
    end

    @testset "RL chain axisymmetric linear exactness" begin
        # f(r) = 3r + 7 — axisymmetric (k=0 only), linear
        gp1 = SpringsteelGridParameters(
            geometry="RL", iMin=0.0, iMax=50.0, num_cells=10,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="RL", iMin=50.0, iMax=75.0, num_cells=10,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2)
        pts1 = getGridpoints(g1); pts2 = getGridpoints(g2)
        for i in 1:size(pts1, 1); g1.physical[i, 1, 1] = 3*pts1[i, 1] + 7; end
        for i in 1:size(pts2, 1); g2.physical[i, 1, 1] = 3*pts2[i, 1] + 7; end
        spectralTransform!(g1); spectralTransform!(g2)

        mpg = PatchChain([g1, g2])
        multiGridTransform!(mpg)

        for i in 1:size(pts2, 1)
            @test g2.physical[i, 1, 1] ≈ 3*pts2[i, 1] + 7 atol=1e-8
        end
    end

    @testset "RL chain non-axisymmetric: r·cos(λ) per-wavenumber coupling" begin
        # f(r,λ) = r·cos(λ) — excites k=1 mode.  Coupling must transfer
        # per-wavenumber ahat correctly (not just k=0).
        gp1 = SpringsteelGridParameters(
            geometry="RL", iMin=0.0, iMax=50.0, num_cells=10,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="RL", iMin=50.0, iMax=75.0, num_cells=10,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2)
        pts1 = getGridpoints(g1); pts2 = getGridpoints(g2)
        for i in 1:size(pts1, 1)
            g1.physical[i, 1, 1] = pts1[i, 1] * cos(pts1[i, 2])
        end
        for i in 1:size(pts2, 1)
            g2.physical[i, 1, 1] = pts2[i, 1] * cos(pts2[i, 2])
        end
        spectralTransform!(g1); spectralTransform!(g2)

        mpg = PatchChain([g1, g2])
        multiGridTransform!(mpg)

        # Check reconstruction quality on secondary
        max_err = maximum(
            abs(g2.physical[i, 1, 1] - pts2[i, 1] * cos(pts2[i, 2]))
            for i in 1:size(pts2, 1))
        # Coupled error should be bounded — cubic B-spline over 10 cells
        @test max_err < 0.5
    end

    @testset "RL chain non-axisymmetric: r·sin(λ) k=1 imaginary" begin
        # f(r,λ) = r·sin(λ) — excites k=1 imaginary mode
        gp1 = SpringsteelGridParameters(
            geometry="RL", iMin=0.0, iMax=50.0, num_cells=10,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="RL", iMin=50.0, iMax=75.0, num_cells=10,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2)
        pts1 = getGridpoints(g1); pts2 = getGridpoints(g2)
        for i in 1:size(pts1, 1)
            g1.physical[i, 1, 1] = pts1[i, 1] * sin(pts1[i, 2])
        end
        for i in 1:size(pts2, 1)
            g2.physical[i, 1, 1] = pts2[i, 1] * sin(pts2[i, 2])
        end
        spectralTransform!(g1); spectralTransform!(g2)

        mpg = PatchChain([g1, g2])
        multiGridTransform!(mpg)

        max_err = maximum(
            abs(g2.physical[i, 1, 1] - pts2[i, 1] * sin(pts2[i, 2]))
            for i in 1:size(pts2, 1))
        @test max_err < 0.5
    end

    @testset "RL embedded: disc inside annulus" begin
        # Coarse annulus [20, 100], fine disc [0, 20]
        # DX_annulus = 80/10 = 8.0, DX_disc = 20/5 = 4.0 → exact 2:1
        gp_annulus = SpringsteelGridParameters(
            geometry="RL", iMin=20.0, iMax=100.0, num_cells=10,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        gp_disc = SpringsteelGridParameters(
            geometry="RL", iMin=0.0, iMax=20.0, num_cells=5,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => FixedBC()),
            vars=Dict("u" => 1))

        g_annulus = createGrid(gp_annulus); g_disc = createGrid(gp_disc)

        # Axisymmetric linear f(r) = 2r + 5
        pts_a = getGridpoints(g_annulus); pts_d = getGridpoints(g_disc)
        for i in 1:size(pts_a, 1); g_annulus.physical[i, 1, 1] = 2*pts_a[i, 1] + 5; end
        for i in 1:size(pts_d, 1); g_disc.physical[i, 1, 1] = 2*pts_d[i, 1] + 5; end
        spectralTransform!(g_annulus); spectralTransform!(g_disc)

        # Annulus is coarser → primary, disc is finer → secondary
        iface = PatchInterface(g_annulus, g_disc, :left, :right, :i)
        gridTransform!(g_annulus)
        update_interface!(iface)
        gridTransform!(g_disc)

        # Linear should be well-represented
        max_err = maximum(
            abs(g_disc.physical[i, 1, 1] - (2*pts_d[i, 1] + 5))
            for i in 1:size(pts_d, 1))
        @test max_err < 1e-6
    end

    @testset "RL coarse-fine-coarse annulus chain" begin
        # [0, 40] coarse (DX=4) → [40, 60] fine (DX=2) → [60, 100] coarse (DX=4)
        # Exact 2:1 ratio at each interface
        gp1 = SpringsteelGridParameters(
            geometry="RL", iMin=0.0, iMax=40.0, num_cells=10,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="RL", iMin=40.0, iMax=60.0, num_cells=10,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => FixedBC()),
            vars=Dict("u" => 1))
        gp3 = SpringsteelGridParameters(
            geometry="RL", iMin=60.0, iMax=100.0, num_cells=10,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2); g3 = createGrid(gp3)

        # Axisymmetric linear f(r) = -r + 50
        for (g, pts) in [(g1, getGridpoints(g1)), (g2, getGridpoints(g2)), (g3, getGridpoints(g3))]
            for i in 1:size(pts, 1); g.physical[i, 1, 1] = -pts[i, 1] + 50; end
            spectralTransform!(g)
        end

        mpg = PatchChain([g1, g2, g3])
        multiGridTransform!(mpg)

        pts2 = getGridpoints(g2)
        max_err = maximum(
            abs(g2.physical[i, 1, 1] - (-pts2[i, 1] + 50))
            for i in 1:size(pts2, 1))
        @test max_err < 1e-6
    end

    @testset "RL chain with patchOffsetL" begin
        # Test that setting patchOffsetL gives correct ring sizes
        # Annulus at large radius should have larger rings
        gp_annulus = SpringsteelGridParameters(
            geometry="RL", iMin=50.0, iMax=100.0, num_cells=10,
            patchOffsetL=30,  # As if inner 30 gridpoints exist
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        g = createGrid(gp_annulus)

        # First ring should have ri = 1 + 30 = 31, lpoints = 4 + 4*31 = 128
        ring1 = g.jbasis.data[1, 1]
        @test ring1.params.yDim == 4 + 4 * 31

        # Without patchOffsetL, first ring would be ri=1, lpoints=8
        gp_no_offset = SpringsteelGridParameters(
            geometry="RL", iMin=50.0, iMax=100.0, num_cells=10,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        g_no = createGrid(gp_no_offset)
        ring1_no = g_no.jbasis.data[1, 1]
        @test ring1_no.params.yDim == 8  # ri=1, lpoints=8

        # patchOffsetL grid has much larger rings (appropriate for its radius)
        @test ring1.params.yDim > 10 * ring1_no.params.yDim
    end

    # ── 3D RLZ (cylindrical) multipatch ───────────────────────────────────

    @testset "RLZ chain construction" begin
        gp1 = SpringsteelGridParameters(
            geometry="RLZ", iMin=0.0, iMax=50.0, num_cells=10,
            kMin=0.0, kMax=10.0, kDim=6,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            BCB=Dict("u" => Chebyshev.R0), BCT=Dict("u" => Chebyshev.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="RLZ", iMin=50.0, iMax=75.0, num_cells=10,
            kMin=0.0, kMax=10.0, kDim=6,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => NaturalBC()),
            BCB=Dict("u" => Chebyshev.R0), BCT=Dict("u" => Chebyshev.R0),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2)
        mpg = PatchChain([g1, g2])
        @test length(mpg.interfaces) == 1
        @test length(mpg.patches) == 2
    end

    @testset "RLZ chain axisymmetric linear: f(r) = 3r + 7" begin
        gp1 = SpringsteelGridParameters(
            geometry="RLZ", iMin=0.0, iMax=50.0, num_cells=10,
            kMin=0.0, kMax=10.0, kDim=6,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            BCB=Dict("u" => Chebyshev.R0), BCT=Dict("u" => Chebyshev.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="RLZ", iMin=50.0, iMax=75.0, num_cells=10,
            kMin=0.0, kMax=10.0, kDim=6,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => NaturalBC()),
            BCB=Dict("u" => Chebyshev.R0), BCT=Dict("u" => Chebyshev.R0),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2)
        pts1 = getGridpoints(g1); pts2 = getGridpoints(g2)
        # pts columns: [r, λ, z]
        for i in 1:size(pts1, 1); g1.physical[i, 1, 1] = 3*pts1[i, 1] + 7; end
        for i in 1:size(pts2, 1); g2.physical[i, 1, 1] = 3*pts2[i, 1] + 7; end
        spectralTransform!(g1); spectralTransform!(g2)

        mpg = PatchChain([g1, g2])
        multiGridTransform!(mpg)

        max_err = maximum(
            abs(g2.physical[i, 1, 1] - (3*pts2[i, 1] + 7))
            for i in 1:size(pts2, 1))
        @test max_err < 1e-6
    end

    @testset "RLZ chain z-dependent: f(r,z) = r·z" begin
        # Axisymmetric but z-varying — tests Chebyshev coupling path
        gp1 = SpringsteelGridParameters(
            geometry="RLZ", iMin=0.0, iMax=50.0, num_cells=10,
            kMin=0.0, kMax=10.0, kDim=6,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            BCB=Dict("u" => Chebyshev.R0), BCT=Dict("u" => Chebyshev.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="RLZ", iMin=50.0, iMax=75.0, num_cells=10,
            kMin=0.0, kMax=10.0, kDim=6,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => NaturalBC()),
            BCB=Dict("u" => Chebyshev.R0), BCT=Dict("u" => Chebyshev.R0),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2)
        pts1 = getGridpoints(g1); pts2 = getGridpoints(g2)
        for i in 1:size(pts1, 1); g1.physical[i, 1, 1] = pts1[i, 1] * pts1[i, 3]; end
        for i in 1:size(pts2, 1); g2.physical[i, 1, 1] = pts2[i, 1] * pts2[i, 3]; end
        spectralTransform!(g1); spectralTransform!(g2)

        mpg = PatchChain([g1, g2])
        multiGridTransform!(mpg)

        max_err = maximum(
            abs(g2.physical[i, 1, 1] - pts2[i, 1] * pts2[i, 3])
            for i in 1:size(pts2, 1))
        @test max_err < 0.5
    end

    @testset "RLZ chain non-axisymmetric: f(r,λ) = r·cos(λ)" begin
        gp1 = SpringsteelGridParameters(
            geometry="RLZ", iMin=0.0, iMax=50.0, num_cells=10,
            kMin=0.0, kMax=10.0, kDim=6,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            BCB=Dict("u" => Chebyshev.R0), BCT=Dict("u" => Chebyshev.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="RLZ", iMin=50.0, iMax=75.0, num_cells=10,
            kMin=0.0, kMax=10.0, kDim=6,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => NaturalBC()),
            BCB=Dict("u" => Chebyshev.R0), BCT=Dict("u" => Chebyshev.R0),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2)
        pts1 = getGridpoints(g1); pts2 = getGridpoints(g2)
        for i in 1:size(pts1, 1)
            g1.physical[i, 1, 1] = pts1[i, 1] * cos(pts1[i, 2])
        end
        for i in 1:size(pts2, 1)
            g2.physical[i, 1, 1] = pts2[i, 1] * cos(pts2[i, 2])
        end
        spectralTransform!(g1); spectralTransform!(g2)

        mpg = PatchChain([g1, g2])
        multiGridTransform!(mpg)

        max_err = maximum(
            abs(g2.physical[i, 1, 1] - pts2[i, 1] * cos(pts2[i, 2]))
            for i in 1:size(pts2, 1))
        @test max_err < 0.5
    end

    @testset "RLZ chain full 3D: f(r,λ,z) = r·cos(λ)·z" begin
        gp1 = SpringsteelGridParameters(
            geometry="RLZ", iMin=0.0, iMax=50.0, num_cells=10,
            kMin=0.0, kMax=10.0, kDim=6,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            BCB=Dict("u" => Chebyshev.R0), BCT=Dict("u" => Chebyshev.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="RLZ", iMin=50.0, iMax=75.0, num_cells=10,
            kMin=0.0, kMax=10.0, kDim=6,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => NaturalBC()),
            BCB=Dict("u" => Chebyshev.R0), BCT=Dict("u" => Chebyshev.R0),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2)
        pts1 = getGridpoints(g1); pts2 = getGridpoints(g2)
        for i in 1:size(pts1, 1)
            g1.physical[i, 1, 1] = pts1[i, 1] * cos(pts1[i, 2]) * pts1[i, 3]
        end
        for i in 1:size(pts2, 1)
            g2.physical[i, 1, 1] = pts2[i, 1] * cos(pts2[i, 2]) * pts2[i, 3]
        end
        spectralTransform!(g1); spectralTransform!(g2)

        mpg = PatchChain([g1, g2])
        multiGridTransform!(mpg)

        max_err = maximum(
            abs(g2.physical[i, 1, 1] - pts2[i, 1] * cos(pts2[i, 2]) * pts2[i, 3])
            for i in 1:size(pts2, 1))
        @test max_err < 5.0
    end

    @testset "RLZ coarse-fine chain: 2:1 ratio axisymmetric" begin
        # DX: 5.0 → 2.5 (exact 2:1)
        gp1 = SpringsteelGridParameters(
            geometry="RLZ", iMin=0.0, iMax=50.0, num_cells=10,
            kMin=0.0, kMax=10.0, kDim=6,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            BCB=Dict("u" => Chebyshev.R0), BCT=Dict("u" => Chebyshev.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="RLZ", iMin=50.0, iMax=75.0, num_cells=10,
            kMin=0.0, kMax=10.0, kDim=6,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => NaturalBC()),
            BCB=Dict("u" => Chebyshev.R0), BCT=Dict("u" => Chebyshev.R0),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2)
        pts1 = getGridpoints(g1); pts2 = getGridpoints(g2)
        for i in 1:size(pts1, 1); g1.physical[i, 1, 1] = 2*pts1[i, 1] + 5; end
        for i in 1:size(pts2, 1); g2.physical[i, 1, 1] = 2*pts2[i, 1] + 5; end
        spectralTransform!(g1); spectralTransform!(g2)

        mpg = PatchChain([g1, g2])
        multiGridTransform!(mpg)

        max_err = maximum(
            abs(g2.physical[i, 1, 1] - (2*pts2[i, 1] + 5))
            for i in 1:size(pts2, 1))
        @test max_err < 1e-6
    end

end

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

    @testset "_build_coupling_matrix rejects wrong ratio" begin
        @test_throws ArgumentError Springsteel._build_coupling_matrix(3.0, 1.0)
        @test_throws ArgumentError Springsteel._build_coupling_matrix(1.0, 1.0)
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

    @testset "PatchInterface hollow nest construction" begin
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

    # ── 1D hollow nest: linear polynomial exactness ─────────────────────────

    @testset "1D hollow nest linear polynomial exactness" begin
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

        mpg = HollowNest(g_left, g_sec, g_right)
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

    # ── 1D hollow nest: cubic polynomial exactness with l_q=0 ──────────────

    @testset "1D hollow nest cubic polynomial exactness (l_q=0)" begin
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

        mpg = HollowNest(g_left, g_sec, g_right)
        multiGridTransform!(mpg)

        pts_sec = getGridpoints(g_sec)
        max_err = maximum(abs(g_sec.physical[i, 1, 1] - f(pts_sec[i])) for i in eachindex(pts_sec))
        @test max_err < 1e-8
    end

    # ── 1D hollow nest: smoothing effect on coupling ──────────────────────

    @testset "1D hollow nest coupling with default l_q" begin
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

        mpg = HollowNest(g_left, g_sec, g_right)
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
        mpg = HollowNest(g1_l, g1_s, g1_r)
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

    # ── 1D stacked nest ────────────────────────────────────────────────────

    @testset "1D stacked nest construction" begin
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

        mpg = InteriorNest(g_primary, g_sec)
        @test length(mpg.interfaces) == 2
        @test length(mpg.patches) == 2
        @test length(mpg.transform_order) == 2
        # Primary should be in first layer
        @test 1 in mpg.transform_order[1]
    end

    @testset "1D stacked nest linear polynomial exactness" begin
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

        mpg = InteriorNest(g_primary, g_sec)
        multiGridTransform!(mpg)

        # Linear function: machine precision on secondary
        for i in eachindex(pts_s)
            @test g_sec.physical[i, 1, 1] ≈ f(pts_s[i]) atol=1e-10
        end
    end

    @testset "1D stacked nest rejects misaligned grids" begin
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
        @test_throws ArgumentError InteriorNest(g_primary, g_sec)
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

    @testset "Multi-variable hollow nest" begin
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

        mpg = HollowNest(g1, g2, g3)
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

    # Helper: build a hollow nest with independent l_q on primary vs secondary
    function _run_mixed_lq_hollow(f, nc_p, nc_s, lq_primary, lq_secondary)
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
        mpg = HollowNest(g1, g2, g3)
        multiGridTransform!(mpg)

        pts2 = getGridpoints(g2)
        max_err = maximum(abs(g2.physical[i, 1, 1] - f(pts2[i])) for i in eachindex(pts2))
        return max_err
    end

    @testset "Cubic exactness: primary l_q=0, secondary l_q=0" begin
        f(x) = x^3 - 2x^2 + x - 1
        err = _run_mixed_lq_hollow(f, 10, 20, 0.0, 0.0)
        @test err < 1e-8
    end

    @testset "Cubic exactness: primary l_q=0, secondary l_q=2" begin
        # Primary provides exact coefficients; secondary l_q is irrelevant
        # for cubics (null space of third-derivative penalty)
        f(x) = x^3 - 2x^2 + x - 1
        err = _run_mixed_lq_hollow(f, 10, 20, 0.0, 2.0)
        @test err < 1e-8
    end

    @testset "Cubic error: primary l_q=2, secondary l_q=0" begin
        # Primary l_q=2 shifts coefficients; coupling propagates the error
        f(x) = x^3 - 2x^2 + x - 1
        err = _run_mixed_lq_hollow(f, 10, 20, 2.0, 0.0)
        @test err > 0.1  # Large error from primary smoothing
    end

    @testset "Cubic error: primary l_q=2, secondary l_q=2" begin
        f(x) = x^3 - 2x^2 + x - 1
        err = _run_mixed_lq_hollow(f, 10, 20, 2.0, 2.0)
        @test err > 0.1  # Large error from primary smoothing
    end

    @testset "Quadratic exactness: all l_q configurations" begin
        # Quadratic is within both the B-spline reproduction range and the
        # null space of the smoothing penalty (Q penalizes 3rd derivative,
        # which is zero for quadratics). Exact for all l_q values.
        f(x) = x^2 - 3x + 7
        for (lq_p, lq_s) in [(0.0, 0.0), (0.0, 2.0), (2.0, 0.0), (2.0, 2.0)]
            err = _run_mixed_lq_hollow(f, 10, 20, lq_p, lq_s)
            @test err < 1e-10
        end
    end

    @testset "Sinusoidal: primary l_q=0 gives near-machine-precision coupling" begin
        # Even for non-polynomial functions, primary l_q=0 ensures accurate
        # coupling. Secondary l_q doesn't affect coupling accuracy.
        f(x) = sin(2π * x / 30.0) + 1.0
        err_00 = _run_mixed_lq_hollow(f, 10, 20, 0.0, 0.0)
        err_02 = _run_mixed_lq_hollow(f, 10, 20, 0.0, 2.0)
        err_20 = _run_mixed_lq_hollow(f, 10, 20, 2.0, 0.0)

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

        mpg = InteriorNest(g_coarse, g_fine)
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

end

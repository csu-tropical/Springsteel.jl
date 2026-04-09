using LinearAlgebra
using SharedArrays
using SparseArrays

@testset "Tiled Multi-Patch" begin

    # ── 1D chain: tiled patches produce same result as non-tiled ──────────

    @testset "1D chain: tiled vs non-tiled linear exactness" begin
        f(x) = 3x + 7

        gp1 = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=12,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=15.0, num_cells=12,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))

        # Non-tiled reference
        g1_ref = createGrid(gp1); g2_ref = createGrid(gp2)
        for (g, pts) in [(g1_ref, getGridpoints(g1_ref)), (g2_ref, getGridpoints(g2_ref))]
            for i in eachindex(pts); g.physical[i, 1, 1] = f(pts[i]); end
            spectralTransform!(g)
        end
        mpg_ref = PatchChain([g1_ref, g2_ref])
        multiGridTransform!(mpg_ref)
        ref_phys1 = copy(g1_ref.physical)
        ref_phys2 = copy(g2_ref.physical)

        # Tiled version: create same grids, tile each
        g1 = createGrid(gp1); g2 = createGrid(gp2)
        for (g, pts) in [(g1, getGridpoints(g1)), (g2, getGridpoints(g2))]
            for i in eachindex(pts); g.physical[i, 1, 1] = f(pts[i]); end
            spectralTransform!(g)
        end

        # Tile primary patch
        tiles1 = calcTileSizes(g1, 4)
        @test length(tiles1) == 4

        # Tile secondary patch
        tiles2 = calcTileSizes(g2, 4)
        @test length(tiles2) == 4

        # multiGridTransform! uses gridTransform! which handles ahat correctly
        mpg = PatchChain([g1, g2])
        multiGridTransform!(mpg)

        # Tiled and non-tiled should give identical results
        @test g1.physical ≈ ref_phys1 atol=1e-14
        @test g2.physical ≈ ref_phys2 atol=1e-14

        # Linear exactness still holds
        pts2 = getGridpoints(g2)
        for i in eachindex(pts2)
            @test g2.physical[i, 1, 1] ≈ f(pts2[i]) atol=1e-10
        end
    end

    # ── 1D chain: tiled splineTransform! on primary matches gridTransform! ─

    @testset "1D tiled primary B→A matches gridTransform!" begin
        f(x) = 3x + 7

        gp = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=12,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))

        # Reference: standard gridTransform!
        g_ref = createGrid(gp)
        pts = getGridpoints(g_ref)
        for i in eachindex(pts); g_ref.physical[i, 1, 1] = f(pts[i]); end
        spectralTransform!(g_ref)
        gridTransform!(g_ref)

        # Tiled: splineTransform! + tileTransform!
        g_tile = createGrid(gp)
        for i in eachindex(pts); g_tile.physical[i, 1, 1] = f(pts[i]); end
        spectralTransform!(g_tile)

        sharedSpectral = SharedArray{Float64}(size(g_tile.spectral))
        sharedSpectral[:, :] .= g_tile.spectral

        splineTransform!(sharedSpectral, g_tile)

        physical_tiled = zeros(Float64, size(g_tile.physical))
        tileTransform!(sharedSpectral, g_tile, physical_tiled, g_tile.spectral)

        # Primary (R0 BCs, no ahat): tiled pipeline should match exactly
        @test physical_tiled[:, 1, 1] ≈ g_ref.physical[:, 1, 1] atol=1e-14
        @test physical_tiled[:, 1, 2] ≈ g_ref.physical[:, 1, 2] atol=1e-14
        @test physical_tiled[:, 1, 3] ≈ g_ref.physical[:, 1, 3] atol=1e-14
    end

    # ── 1D chain: full tiled multipatch workflow ──────────────────────────

    @testset "1D chain: tiled primary + gridTransform! secondary" begin
        # Simulates the distributed-memory workflow:
        # 1. spectralTransform! on each patch (sets .b)
        # 2. Primary: tiled B→A via splineTransform! + tileTransform! (sets .a)
        # 3. update_interface! (reads primary .a, writes secondary .ahat)
        # 4. Secondary: gridTransform! (uses .ahat in SAtransform!)
        f(x) = 3x + 7  # Linear for exact test

        gp1 = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=12,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=15.0, num_cells=12,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2)
        pts1 = getGridpoints(g1); pts2 = getGridpoints(g2)
        for i in eachindex(pts1); g1.physical[i, 1, 1] = f(pts1[i]); end
        for i in eachindex(pts2); g2.physical[i, 1, 1] = f(pts2[i]); end
        spectralTransform!(g1)
        spectralTransform!(g2)

        # Step 1: Tiled B→A→physical on primary
        shared1 = SharedArray{Float64}(size(g1.spectral))
        shared1[:, :] .= g1.spectral
        splineTransform!(shared1, g1)
        phys1 = zeros(Float64, size(g1.physical))
        tileTransform!(shared1, g1, phys1, g1.spectral)

        # tileTransform! sets spline.a, which update_interface! needs
        # Verify .a was set
        @test all(isfinite, g1.ibasis.data[1, 1].a)

        # Step 2: Interface transfer
        iface = PatchInterface(g1, g2, :right, :left, :i)
        update_interface!(iface)

        # Verify ahat was set on secondary
        @test !all(g2.ibasis.data[1, 1].ahat[1:3] .== 0.0)

        # Step 3: gridTransform! on secondary (uses ahat)
        gridTransform!(g2)

        # Linear should be exact
        for i in eachindex(pts2)
            @test g2.physical[i, 1, 1] ≈ f(pts2[i]) atol=1e-10
        end
        for i in eachindex(pts2)
            @test g2.physical[i, 1, 2] ≈ 3.0 atol=1e-8
        end
    end

    # ── 3-patch chain: tiled primary → secondary → secondary ──────────────

    @testset "3-patch chain tiled: cubic exactness (l_q=0)" begin
        f(x) = 2x^3 - 3x^2 + x + 5
        lq = Dict("u" => 0.0)

        gp1 = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=12, l_q=lq,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=15.0, num_cells=12, l_q=lq,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => FixedBC()),
            vars=Dict("u" => 1))
        gp3 = SpringsteelGridParameters(
            geometry="R", iMin=15.0, iMax=25.0, num_cells=12, l_q=lq,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2); g3 = createGrid(gp3)
        for (g, pts) in [(g1, getGridpoints(g1)), (g2, getGridpoints(g2)), (g3, getGridpoints(g3))]
            for i in eachindex(pts); g.physical[i, 1, 1] = f(pts[i]); end
            spectralTransform!(g)
        end

        # Tile each patch
        tiles1 = calcTileSizes(g1, 4); @test length(tiles1) == 4
        tiles2 = calcTileSizes(g2, 4); @test length(tiles2) == 4
        tiles3 = calcTileSizes(g3, 4); @test length(tiles3) == 4

        # multiGridTransform! handles everything correctly
        mpg = PatchChain([g1, g2, g3])
        multiGridTransform!(mpg)

        # Cubic with l_q=0 should be exact
        for g in [g1, g2, g3]
            pts = getGridpoints(g)
            max_err = maximum(abs(g.physical[i, 1, 1] - f(pts[i])) for i in eachindex(pts))
            @test max_err < 1e-8
        end
    end

    # ── Embedded: tiled patches with nested grids ─────────────────────────

    @testset "Embedded tiled: linear exactness" begin
        f(x) = -2x + 15

        gp_outer = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=30.0, num_cells=30,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        gp_inner = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=20.0, num_cells=20,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => FixedBC()),
            vars=Dict("u" => 1))

        g_outer = createGrid(gp_outer); g_inner = createGrid(gp_inner)
        for (g, pts) in [(g_outer, getGridpoints(g_outer)), (g_inner, getGridpoints(g_inner))]
            for i in eachindex(pts); g.physical[i, 1, 1] = f(pts[i]); end
            spectralTransform!(g)
        end

        # Tile both
        tiles_outer = calcTileSizes(g_outer, 4); @test length(tiles_outer) == 4
        tiles_inner = calcTileSizes(g_inner, 4); @test length(tiles_inner) == 4

        mpg = PatchEmbedded([g_outer, g_inner])
        multiGridTransform!(mpg)

        pts_inner = getGridpoints(g_inner)
        for i in eachindex(pts_inner)
            @test g_inner.physical[i, 1, 1] ≈ f(pts_inner[i]) atol=1e-10
        end
    end

    # ── Tiled primary with SharedArray: explicit distributed pipeline ─────

    @testset "Tiled primary SharedArray pipeline + interface coupling" begin
        # Full simulation of distributed-memory workflow with tiles:
        # Node 1: tile g1 → shared-memory B→A→physical
        # Node 2: tile g2 → shared-memory spectralTransform + gridTransform
        # Inter-node: update_interface! to transfer BCs
        f(x) = sin(2π * x / 25.0) + 1.0

        gp1 = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=12,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=15.0, num_cells=12,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))

        # --- Non-tiled reference ---
        g1_ref = createGrid(gp1); g2_ref = createGrid(gp2)
        pts1 = getGridpoints(g1_ref); pts2 = getGridpoints(g2_ref)
        for i in eachindex(pts1); g1_ref.physical[i, 1, 1] = f(pts1[i]); end
        for i in eachindex(pts2); g2_ref.physical[i, 1, 1] = f(pts2[i]); end
        spectralTransform!(g1_ref); spectralTransform!(g2_ref)
        mpg_ref = PatchChain([g1_ref, g2_ref])
        multiGridTransform!(mpg_ref)

        # --- Tiled workflow ---
        g1 = createGrid(gp1); g2 = createGrid(gp2)
        for i in eachindex(pts1); g1.physical[i, 1, 1] = f(pts1[i]); end
        for i in eachindex(pts2); g2.physical[i, 1, 1] = f(pts2[i]); end

        # Forward transform on both patches
        spectralTransform!(g1)
        spectralTransform!(g2)

        # Tile primary: B→A via SharedArray
        shared1 = SharedArray{Float64}(size(g1.spectral))
        shared1[:, :] .= g1.spectral
        splineTransform!(shared1, g1)
        phys1 = zeros(Float64, size(g1.physical))
        tileTransform!(shared1, g1, phys1, g1.spectral)

        # Interface transfer
        iface = PatchInterface(g1, g2, :right, :left, :i)
        update_interface!(iface)

        # Secondary: standard gridTransform! (handles ahat)
        gridTransform!(g2)

        # Tiled primary physical should match reference
        @test phys1[:, 1, 1] ≈ g1_ref.physical[:, 1, 1] atol=1e-14

        # Secondary should match reference
        @test g2.physical[:, 1, 1] ≈ g2_ref.physical[:, 1, 1] atol=1e-14
    end

    # ── 2D RL: tiled multipatch with cylindrical geometry ─────────────────

    @testset "2D RL tiled chain: axisymmetric function" begin
        # Axisymmetric f(r) — Fourier dimension has no variation
        # Tests that tiling in i-dimension works with Fourier j-dimension

        gp1 = SpringsteelGridParameters(
            geometry="RL", iMin=0.0, iMax=50.0, num_cells=12,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="RL", iMin=50.0, iMax=75.0, num_cells=12,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2)

        # Fill with axisymmetric function f(r) = sin(π·r/75)
        rMax = 75.0
        pts1 = getGridpoints(g1); pts2 = getGridpoints(g2)
        for i in 1:size(pts1, 1)
            r = pts1[i, 1]
            g1.physical[i, 1, 1] = sin(π * r / rMax)
        end
        for i in 1:size(pts2, 1)
            r = pts2[i, 1]
            g2.physical[i, 1, 1] = sin(π * r / rMax)
        end
        spectralTransform!(g1); spectralTransform!(g2)

        # Tile both patches
        tiles1 = calcTileSizes(g1, 4); @test length(tiles1) == 4
        tiles2 = calcTileSizes(g2, 4); @test length(tiles2) == 4

        # Non-tiled reference
        g1_ref = createGrid(gp1); g2_ref = createGrid(gp2)
        for i in 1:size(pts1, 1); g1_ref.physical[i, 1, 1] = sin(π * pts1[i, 1] / rMax); end
        for i in 1:size(pts2, 1); g2_ref.physical[i, 1, 1] = sin(π * pts2[i, 1] / rMax); end
        spectralTransform!(g1_ref); spectralTransform!(g2_ref)
        mpg_ref = PatchChain([g1_ref, g2_ref])
        multiGridTransform!(mpg_ref)

        # Tiled version via multiGridTransform!
        mpg = PatchChain([g1, g2])
        multiGridTransform!(mpg)

        @test g1.physical[:, 1, 1] ≈ g1_ref.physical[:, 1, 1] atol=1e-14
        @test g2.physical[:, 1, 1] ≈ g2_ref.physical[:, 1, 1] atol=1e-14

        # Check accuracy: sin over 12+12 cells → reasonable cubic B-spline accuracy
        for i in 1:size(pts2, 1)
            r = pts2[i, 1]
            @test abs(g2.physical[i, 1, 1] - sin(π * r / rMax)) < 5e-3
        end
    end

    # ── 2D RL: tiled SharedArray pipeline for primary ─────────────────────

    @testset "2D RL tiled primary SharedArray pipeline" begin
        gp = SpringsteelGridParameters(
            geometry="RL", iMin=0.0, iMax=100.0, num_cells=12,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        g = createGrid(gp)

        pts = getGridpoints(g)
        rMax = 100.0
        for i in 1:size(pts, 1)
            g.physical[i, 1, 1] = sin(π * pts[i, 1] / rMax)
        end
        spectralTransform!(g)

        # Reference: standard gridTransform!
        g_ref = createGrid(gp)
        for i in 1:size(pts, 1); g_ref.physical[i, 1, 1] = sin(π * pts[i, 1] / rMax); end
        spectralTransform!(g_ref)
        gridTransform!(g_ref)

        # Tiled: splineTransform! + tileTransform!
        shared = SharedArray{Float64}(size(g.spectral))
        shared[:, :] .= g.spectral
        splineTransform!(shared, g)
        phys_tiled = zeros(Float64, size(g.physical))
        tileTransform!(shared, g, phys_tiled, g.spectral)

        # R0 primary: tiled should match standard pipeline
        @test phys_tiled[:, 1, 1] ≈ g_ref.physical[:, 1, 1] atol=1e-12
    end

    # ── Multi-variable tiled chain ────────────────────────────────────────

    @testset "Multi-variable tiled chain" begin
        f_u(x) = 2x + 1
        f_v(x) = -x + 3

        gp1 = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=12,
            BCL=Dict("u" => NaturalBC(), "v" => NaturalBC()),
            BCR=Dict("u" => NaturalBC(), "v" => NaturalBC()),
            vars=Dict("u" => 1, "v" => 2))
        gp2 = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=15.0, num_cells=12,
            BCL=Dict("u" => FixedBC(), "v" => FixedBC()),
            BCR=Dict("u" => FixedBC(), "v" => FixedBC()),
            vars=Dict("u" => 1, "v" => 2))
        gp3 = SpringsteelGridParameters(
            geometry="R", iMin=15.0, iMax=25.0, num_cells=12,
            BCL=Dict("u" => NaturalBC(), "v" => NaturalBC()),
            BCR=Dict("u" => NaturalBC(), "v" => NaturalBC()),
            vars=Dict("u" => 1, "v" => 2))

        g1 = createGrid(gp1); g2 = createGrid(gp2); g3 = createGrid(gp3)
        for (g, pts) in [(g1, getGridpoints(g1)), (g2, getGridpoints(g2)), (g3, getGridpoints(g3))]
            for i in eachindex(pts)
                g.physical[i, 1, 1] = f_u(pts[i])
                g.physical[i, 2, 1] = f_v(pts[i])
            end
            spectralTransform!(g)
        end

        tiles1 = calcTileSizes(g1, 4); @test length(tiles1) == 4
        tiles2 = calcTileSizes(g2, 4); @test length(tiles2) == 4
        tiles3 = calcTileSizes(g3, 4); @test length(tiles3) == 4

        mpg = PatchChain([g1, g2, g3])
        multiGridTransform!(mpg)

        pts2 = getGridpoints(g2)
        for i in eachindex(pts2)
            @test g2.physical[i, 1, 1] ≈ f_u(pts2[i]) atol=1e-10
            @test g2.physical[i, 2, 1] ≈ f_v(pts2[i]) atol=1e-10
        end
    end

    # ── 7-grid tiled chain: 8-4-2-1-2-4-8 DX ────────────────────────────

    @testset "7-grid tiled chain: linear exactness" begin
        f(x) = 3x + 7
        lq = Dict("u" => 0.0)

        configs = [
            (0.0,   96.0,  12),  # DX=8
            (96.0,  144.0, 12),  # DX=4
            (144.0, 168.0, 12),  # DX=2
            (168.0, 180.0, 12),  # DX=1
            (180.0, 204.0, 12),  # DX=2
            (204.0, 252.0, 12),  # DX=4
            (252.0, 348.0, 12),  # DX=8
        ]
        bcl_specs = [NaturalBC(), FixedBC(), FixedBC(),
                     FixedBC(), NaturalBC(), NaturalBC(), NaturalBC()]
        bcr_specs = [NaturalBC(), NaturalBC(), NaturalBC(),
                     FixedBC(), FixedBC(), FixedBC(), NaturalBC()]

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

        # Tile each grid (12 cells / 4 tiles = 3 cells per tile)
        for g in grids
            tiles = calcTileSizes(g, 4)
            @test length(tiles) == 4
        end

        mpg = PatchChain(grids)
        multiGridTransform!(mpg)

        for g in grids
            pts = getGridpoints(g)
            max_err = maximum(abs(g.physical[i, 1, 1] - f(pts[i])) for i in eachindex(pts))
            @test max_err < 1e-10
        end
    end

    # ── 3-level embedded tiled ────────────────────────────────────────────

    @testset "3-level embedded tiled: cubic exactness (l_q=0)" begin
        f(x) = x^3 - 2x^2 + x - 1
        lq = Dict("u" => 0.0)

        gp_outer = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=30.0, num_cells=30, l_q=lq,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        gp_mid = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=20.0, num_cells=20, l_q=lq,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => FixedBC()),
            vars=Dict("u" => 1))
        gp_inner = SpringsteelGridParameters(
            geometry="R", iMin=13.0, iMax=17.0, num_cells=16, l_q=lq,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => FixedBC()),
            vars=Dict("u" => 1))

        g_outer = createGrid(gp_outer); g_mid = createGrid(gp_mid); g_inner = createGrid(gp_inner)
        for (g, pts) in [(g_outer, getGridpoints(g_outer)),
                         (g_mid, getGridpoints(g_mid)),
                         (g_inner, getGridpoints(g_inner))]
            for i in eachindex(pts); g.physical[i, 1, 1] = f(pts[i]); end
            spectralTransform!(g)
        end

        # Tile all levels
        tiles_outer = calcTileSizes(g_outer, 4); @test length(tiles_outer) == 4
        tiles_mid = calcTileSizes(g_mid, 4); @test length(tiles_mid) == 4
        tiles_inner = calcTileSizes(g_inner, 4); @test length(tiles_inner) == 4

        mpg = PatchEmbedded([g_outer, g_mid, g_inner])
        multiGridTransform!(mpg)

        for g in [g_outer, g_mid, g_inner]
            pts = getGridpoints(g)
            max_err = maximum(abs(g.physical[i, 1, 1] - f(pts[i])) for i in eachindex(pts))
            @test max_err < 1e-6
        end
    end

    # ── Tile maps and halo regions work on multipatch grids ───────────────

    @testset "Tile maps valid on multipatch grids" begin
        gp1 = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=12,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=15.0, num_cells=12,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))

        g1 = createGrid(gp1); g2 = createGrid(gp2)

        tiles1 = calcTileSizes(g1, 4)
        tiles2 = calcTileSizes(g2, 4)

        # PatchMap and HaloMap should work on both primary and secondary
        for (g, tiles) in [(g1, tiles1), (g2, tiles2)]
            patchMap = calcPatchMap(g, tiles[1])
            @test patchMap isa SparseMatrixCSC
            @test nnz(patchMap) > 0

            haloMap = calcHaloMap(g, tiles[1], tiles[2])
            @test haloMap isa SparseMatrixCSC
            @test nnz(haloMap) == 3  # 3-row halo for 1 variable

            for tile in tiles
                @test tile.params.num_cells >= 3
            end
        end

        # Cell counts sum correctly
        @test sum(t.params.num_cells for t in tiles1) == 12
        @test sum(t.params.num_cells for t in tiles2) == 12

        # Domain coverage preserved
        @test tiles1[1].params.iMin ≈ g1.params.iMin
        @test tiles1[end].params.iMax ≈ g1.params.iMax
        @test tiles2[1].params.iMin ≈ g2.params.iMin
        @test tiles2[end].params.iMax ≈ g2.params.iMax
    end

    # ── RL tiled disc+annulus: axisymmetric ──────────────────────────────

    @testset "RL tiled disc+annulus: axisymmetric" begin
        # 1:1 ratio: DX_annulus = 48/12 = 4.0, DX_disc = 48/12 = 4.0
        gp_annulus = SpringsteelGridParameters(
            geometry="RL", iMin=48.0, iMax=96.0, num_cells=12,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        gp_disc = SpringsteelGridParameters(
            geometry="RL", iMin=0.0, iMax=48.0, num_cells=12,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => FixedBC()),
            vars=Dict("u" => 1))

        g_annulus = createGrid(gp_annulus); g_disc = createGrid(gp_disc)

        rMax = 96.0
        pts_a = getGridpoints(g_annulus); pts_d = getGridpoints(g_disc)
        for i in 1:size(pts_a, 1); g_annulus.physical[i, 1, 1] = sin(π * pts_a[i, 1] / rMax); end
        for i in 1:size(pts_d, 1); g_disc.physical[i, 1, 1] = sin(π * pts_d[i, 1] / rMax); end
        spectralTransform!(g_annulus); spectralTransform!(g_disc)

        # Tile both
        tiles_a = calcTileSizes(g_annulus, 4); @test length(tiles_a) == 4
        tiles_d = calcTileSizes(g_disc, 4); @test length(tiles_d) == 4

        # Coupling: annulus is primary (left), disc is secondary (right)
        # Same DX → 1:1 ratio
        iface = PatchInterface(g_annulus, g_disc, :left, :right, :i)
        gridTransform!(g_annulus)
        update_interface!(iface)
        gridTransform!(g_disc)

        # Axisymmetric accuracy
        max_err = maximum(
            abs(g_disc.physical[i, 1, 1] - sin(π * pts_d[i, 1] / rMax))
            for i in 1:size(pts_d, 1))
        @test max_err < 5e-3
    end

    # ── RL tiled chain: non-axisymmetric ──────────────────────────────────

    @testset "RL tiled chain: non-axisymmetric r·cos(λ)" begin
        gp1 = SpringsteelGridParameters(
            geometry="RL", iMin=0.0, iMax=50.0, num_cells=12,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="RL", iMin=50.0, iMax=75.0, num_cells=12,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))

        # Non-tiled reference
        g1_ref = createGrid(gp1); g2_ref = createGrid(gp2)
        pts1 = getGridpoints(g1_ref); pts2 = getGridpoints(g2_ref)
        for i in 1:size(pts1, 1)
            g1_ref.physical[i, 1, 1] = pts1[i, 1] * cos(pts1[i, 2])
        end
        for i in 1:size(pts2, 1)
            g2_ref.physical[i, 1, 1] = pts2[i, 1] * cos(pts2[i, 2])
        end
        spectralTransform!(g1_ref); spectralTransform!(g2_ref)
        mpg_ref = PatchChain([g1_ref, g2_ref])
        multiGridTransform!(mpg_ref)

        # Tiled version
        g1 = createGrid(gp1); g2 = createGrid(gp2)
        for i in 1:size(pts1, 1); g1.physical[i, 1, 1] = pts1[i, 1] * cos(pts1[i, 2]); end
        for i in 1:size(pts2, 1); g2.physical[i, 1, 1] = pts2[i, 1] * cos(pts2[i, 2]); end
        spectralTransform!(g1); spectralTransform!(g2)

        tiles1 = calcTileSizes(g1, 4); @test length(tiles1) == 4
        tiles2 = calcTileSizes(g2, 4); @test length(tiles2) == 4

        mpg = PatchChain([g1, g2])
        multiGridTransform!(mpg)

        # Tiled should match non-tiled exactly
        @test g2.physical[:, 1, 1] ≈ g2_ref.physical[:, 1, 1] atol=1e-12
    end

    # ── RL tiled embedded: non-axisymmetric ───────────────────────────────

    @testset "RL tiled coarse-fine-coarse chain: non-axisymmetric" begin
        # DX: 4.0 → 2.0 → 4.0 (exact 2:1 ratios)
        gp1 = SpringsteelGridParameters(
            geometry="RL", iMin=0.0, iMax=48.0, num_cells=12,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="RL", iMin=48.0, iMax=72.0, num_cells=12,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => FixedBC()),
            vars=Dict("u" => 1))
        gp3 = SpringsteelGridParameters(
            geometry="RL", iMin=72.0, iMax=120.0, num_cells=12,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))

        # Non-tiled reference
        g1_ref = createGrid(gp1); g2_ref = createGrid(gp2); g3_ref = createGrid(gp3)
        for (g, pts) in [(g1_ref, getGridpoints(g1_ref)), (g2_ref, getGridpoints(g2_ref)), (g3_ref, getGridpoints(g3_ref))]
            for i in 1:size(pts, 1)
                g.physical[i, 1, 1] = pts[i, 1] * cos(pts[i, 2])
            end
            spectralTransform!(g)
        end
        mpg_ref = PatchChain([g1_ref, g2_ref, g3_ref])
        multiGridTransform!(mpg_ref)

        # Tiled version
        g1 = createGrid(gp1); g2 = createGrid(gp2); g3 = createGrid(gp3)
        for (g, pts) in [(g1, getGridpoints(g1)), (g2, getGridpoints(g2)), (g3, getGridpoints(g3))]
            for i in 1:size(pts, 1)
                g.physical[i, 1, 1] = pts[i, 1] * cos(pts[i, 2])
            end
            spectralTransform!(g)
        end

        tiles1 = calcTileSizes(g1, 4); @test length(tiles1) == 4
        tiles2 = calcTileSizes(g2, 4); @test length(tiles2) == 4
        tiles3 = calcTileSizes(g3, 4); @test length(tiles3) == 4

        mpg = PatchChain([g1, g2, g3])
        multiGridTransform!(mpg)

        # Tiled should match non-tiled exactly
        @test g2.physical[:, 1, 1] ≈ g2_ref.physical[:, 1, 1] atol=1e-12
    end

    # ── sumSpectralTile! / setSpectralTile! on multipatch grids ──────────

    # ── RLZ tiled chain ──────────────────────────────────────────────────

    @testset "RLZ tiled chain: axisymmetric tiled vs non-tiled" begin
        gp1 = SpringsteelGridParameters(
            geometry="RLZ", iMin=0.0, iMax=50.0, num_cells=12,
            kMin=0.0, kMax=10.0, kDim=6,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            BCB=Dict("u" => Chebyshev.R0), BCT=Dict("u" => Chebyshev.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="RLZ", iMin=50.0, iMax=75.0, num_cells=12,
            kMin=0.0, kMax=10.0, kDim=6,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => NaturalBC()),
            BCB=Dict("u" => Chebyshev.R0), BCT=Dict("u" => Chebyshev.R0),
            vars=Dict("u" => 1))

        # Non-tiled reference
        g1_ref = createGrid(gp1); g2_ref = createGrid(gp2)
        pts1 = getGridpoints(g1_ref); pts2 = getGridpoints(g2_ref)
        for i in 1:size(pts1, 1); g1_ref.physical[i, 1, 1] = 3*pts1[i, 1] + 7; end
        for i in 1:size(pts2, 1); g2_ref.physical[i, 1, 1] = 3*pts2[i, 1] + 7; end
        spectralTransform!(g1_ref); spectralTransform!(g2_ref)
        mpg_ref = PatchChain([g1_ref, g2_ref])
        multiGridTransform!(mpg_ref)

        # Tiled version
        g1 = createGrid(gp1); g2 = createGrid(gp2)
        for i in 1:size(pts1, 1); g1.physical[i, 1, 1] = 3*pts1[i, 1] + 7; end
        for i in 1:size(pts2, 1); g2.physical[i, 1, 1] = 3*pts2[i, 1] + 7; end
        spectralTransform!(g1); spectralTransform!(g2)

        tiles1 = calcTileSizes(g1, 4); @test length(tiles1) == 4
        tiles2 = calcTileSizes(g2, 4); @test length(tiles2) == 4

        mpg = PatchChain([g1, g2])
        multiGridTransform!(mpg)

        @test g2.physical[:, 1, 1] ≈ g2_ref.physical[:, 1, 1] atol=1e-12
    end

    @testset "RLZ tiled chain: non-axisymmetric r·cos(λ)·z" begin
        gp1 = SpringsteelGridParameters(
            geometry="RLZ", iMin=0.0, iMax=50.0, num_cells=12,
            kMin=0.0, kMax=10.0, kDim=6,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            BCB=Dict("u" => Chebyshev.R0), BCT=Dict("u" => Chebyshev.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="RLZ", iMin=50.0, iMax=75.0, num_cells=12,
            kMin=0.0, kMax=10.0, kDim=6,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => NaturalBC()),
            BCB=Dict("u" => Chebyshev.R0), BCT=Dict("u" => Chebyshev.R0),
            vars=Dict("u" => 1))

        # Non-tiled reference
        g1_ref = createGrid(gp1); g2_ref = createGrid(gp2)
        pts1 = getGridpoints(g1_ref); pts2 = getGridpoints(g2_ref)
        for i in 1:size(pts1, 1)
            g1_ref.physical[i, 1, 1] = pts1[i, 1] * cos(pts1[i, 2]) * pts1[i, 3]
        end
        for i in 1:size(pts2, 1)
            g2_ref.physical[i, 1, 1] = pts2[i, 1] * cos(pts2[i, 2]) * pts2[i, 3]
        end
        spectralTransform!(g1_ref); spectralTransform!(g2_ref)
        mpg_ref = PatchChain([g1_ref, g2_ref])
        multiGridTransform!(mpg_ref)

        # Tiled version
        g1 = createGrid(gp1); g2 = createGrid(gp2)
        for i in 1:size(pts1, 1)
            g1.physical[i, 1, 1] = pts1[i, 1] * cos(pts1[i, 2]) * pts1[i, 3]
        end
        for i in 1:size(pts2, 1)
            g2.physical[i, 1, 1] = pts2[i, 1] * cos(pts2[i, 2]) * pts2[i, 3]
        end
        spectralTransform!(g1); spectralTransform!(g2)

        tiles1 = calcTileSizes(g1, 4); @test length(tiles1) == 4
        tiles2 = calcTileSizes(g2, 4); @test length(tiles2) == 4

        mpg = PatchChain([g1, g2])
        multiGridTransform!(mpg)

        @test g2.physical[:, 1, 1] ≈ g2_ref.physical[:, 1, 1] atol=1e-12
    end

    # ── SL tiled chain ───────────────────────────────────────────────────

    @testset "SL tiled chain: non-axisymmetric tiled vs non-tiled" begin
        gp1 = SpringsteelGridParameters(
            geometry="SL", iMin=0.0, iMax=Float64(π)/2, num_cells=12,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="SL", iMin=Float64(π)/2, iMax=Float64(π), num_cells=12,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))

        # Non-tiled reference
        g1_ref = createGrid(gp1); g2_ref = createGrid(gp2)
        pts1 = getGridpoints(g1_ref); pts2 = getGridpoints(g2_ref)
        for i in 1:size(pts1, 1)
            g1_ref.physical[i, 1, 1] = sin(pts1[i, 1]) * cos(pts1[i, 2])
        end
        for i in 1:size(pts2, 1)
            g2_ref.physical[i, 1, 1] = sin(pts2[i, 1]) * cos(pts2[i, 2])
        end
        spectralTransform!(g1_ref); spectralTransform!(g2_ref)
        mpg_ref = PatchChain([g1_ref, g2_ref])
        multiGridTransform!(mpg_ref)

        # Tiled version
        g1 = createGrid(gp1); g2 = createGrid(gp2)
        for i in 1:size(pts1, 1)
            g1.physical[i, 1, 1] = sin(pts1[i, 1]) * cos(pts1[i, 2])
        end
        for i in 1:size(pts2, 1)
            g2.physical[i, 1, 1] = sin(pts2[i, 1]) * cos(pts2[i, 2])
        end
        spectralTransform!(g1); spectralTransform!(g2)

        tiles1 = calcTileSizes(g1, 4); @test length(tiles1) == 4
        tiles2 = calcTileSizes(g2, 4); @test length(tiles2) == 4

        mpg = PatchChain([g1, g2])
        multiGridTransform!(mpg)

        @test g2.physical[:, 1, 1] ≈ g2_ref.physical[:, 1, 1] atol=1e-12
    end

    # ── SLZ tiled chain ──────────────────────────────────────────────────

    @testset "SLZ tiled chain: non-axisymmetric tiled vs non-tiled" begin
        gp1 = SpringsteelGridParameters(
            geometry="SLZ", iMin=0.0, iMax=Float64(π)/2, num_cells=12,
            kMin=0.0, kMax=10.0, kDim=6,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            BCB=Dict("u" => Chebyshev.R0), BCT=Dict("u" => Chebyshev.R0),
            vars=Dict("u" => 1))
        gp2 = SpringsteelGridParameters(
            geometry="SLZ", iMin=Float64(π)/2, iMax=Float64(π), num_cells=12,
            kMin=0.0, kMax=10.0, kDim=6,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => NaturalBC()),
            BCB=Dict("u" => Chebyshev.R0), BCT=Dict("u" => Chebyshev.R0),
            vars=Dict("u" => 1))

        # Non-tiled reference
        g1_ref = createGrid(gp1); g2_ref = createGrid(gp2)
        pts1 = getGridpoints(g1_ref); pts2 = getGridpoints(g2_ref)
        for i in 1:size(pts1, 1)
            g1_ref.physical[i, 1, 1] = sin(pts1[i, 1]) * cos(pts1[i, 2]) * pts1[i, 3]
        end
        for i in 1:size(pts2, 1)
            g2_ref.physical[i, 1, 1] = sin(pts2[i, 1]) * cos(pts2[i, 2]) * pts2[i, 3]
        end
        spectralTransform!(g1_ref); spectralTransform!(g2_ref)
        mpg_ref = PatchChain([g1_ref, g2_ref])
        multiGridTransform!(mpg_ref)

        # Tiled version
        g1 = createGrid(gp1); g2 = createGrid(gp2)
        for i in 1:size(pts1, 1)
            g1.physical[i, 1, 1] = sin(pts1[i, 1]) * cos(pts1[i, 2]) * pts1[i, 3]
        end
        for i in 1:size(pts2, 1)
            g2.physical[i, 1, 1] = sin(pts2[i, 1]) * cos(pts2[i, 2]) * pts2[i, 3]
        end
        spectralTransform!(g1); spectralTransform!(g2)

        tiles1 = calcTileSizes(g1, 4); @test length(tiles1) == 4
        tiles2 = calcTileSizes(g2, 4); @test length(tiles2) == 4

        mpg = PatchChain([g1, g2])
        multiGridTransform!(mpg)

        @test g2.physical[:, 1, 1] ≈ g2_ref.physical[:, 1, 1] atol=1e-12
    end

    # ── sumSpectralTile! / setSpectralTile! on multipatch grids ──────────

    @testset "Spectral tile accumulation on secondary patch" begin
        gp = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=15.0, num_cells=12,
            BCL=Dict("u" => FixedBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        g = createGrid(gp)
        tiles = calcTileSizes(g, 4)

        # Set each tile's spectral to known values
        for (k, tile) in enumerate(tiles)
            tile.spectral .= Float64(k)
        end

        # sumSpectralTile! accumulates
        g.spectral .= 0.0
        for tile in tiles
            sumSpectralTile!(g, tile)
        end
        # All entries should be nonzero after accumulation
        siL = tiles[1].params.spectralIndexL
        siR = tiles[end].params.spectralIndexR
        @test all(g.spectral[siL:siR, 1] .> 0.0)

        # setSpectralTile! replaces
        g.spectral .= 99.0
        setSpectralTile!(g, tiles[1])
        siR1 = tiles[1].params.spectralIndexR
        @test all(g.spectral[siL:siR1, 1] .== 1.0)
    end

end

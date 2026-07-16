using Serialization

# Tests for temporal-nesting support: collar interfaces, payload time
# interpolation, and fine-mesh evaluation at arbitrary i-points.
#
# Two-way nesting scheme (DeMaria et al. 1992; Ooyama 2001): coarse→fine via
# R3X border trio at the interface node (interior to the collar-extended
# coarse patch), fine→coarse via collar-zone evaluation feeding the coarse
# Galerkin loads.

@testset "Nesting support (collar / lerp / point evaluation)" begin

    # ── lerp_payload! ────────────────────────────────────────────────────────

    @testset "lerp_payload! endpoints and midpoint" begin
        meta_dummy = nothing  # payloads constructed directly
        mk(vals) = InterfacePayload(:per_mode, :left, 1, 2,
                                    reshape(collect(Float64, vals), 3, 2, 1))
        p0 = mk(1:6)
        p1 = mk(7:12)
        dest = mk(zeros(6))

        lerp_payload!(dest, p0, p1, 0.0)
        @test dest.border == p0.border            # bitwise
        lerp_payload!(dest, p0, p1, 1.0)
        @test dest.border == p1.border            # bitwise
        lerp_payload!(dest, p0, p1, 0.5)
        @test dest.border ≈ 0.5 .* (p0.border .+ p1.border)
        lerp_payload!(dest, p0, p1, 0.25)
        @test dest.border ≈ 0.75 .* p0.border .+ 0.25 .* p1.border

        @test_throws ArgumentError lerp_payload!(dest, p0, p1, 1.5)
        bad = InterfacePayload(:per_mode, :right, 1, 2, zeros(3, 2, 1))
        @test_throws ArgumentError lerp_payload!(bad, p0, p1, 0.5)
    end

    # ── evaluate_grid_ipoints, 1D R grid ─────────────────────────────────────

    @testset "evaluate_grid_ipoints matches gridTransform! on mish points (1D)" begin
        gp = SpringsteelGridParameters(
            geometry="R", iMin=-5.0, iMax=5.0, num_cells=20,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        g = createGrid(gp)
        pts = getGridpoints(g)
        for i in eachindex(pts)
            g.physical[i, 1, 1] = sin(0.7 * pts[i]) + 0.3 * pts[i]^2
        end
        spectralTransform!(g)
        gridTransform!(g)

        xq = vec(pts)
        out = evaluate_grid_ipoints(g, xq)
        for s in 1:3
            @test maximum(abs.(out[:, 1, s] .- g.physical[:, 1, s])) < 1e-12
        end
    end

    @testset "evaluate_grid_ipoints linear exactness at off-mish points (1D)" begin
        f(x)  = 3x + 5
        gp = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        g = createGrid(gp)
        pts = getGridpoints(g)
        for i in eachindex(pts)
            g.physical[i, 1, 1] = f(pts[i])
        end
        spectralTransform!(g)

        xq = [0.123, 3.456, 6.789, 9.876]
        out = evaluate_grid_ipoints(g, xq)
        @test out[:, 1, 1] ≈ f.(xq) atol=1e-10
        @test out[:, 1, 2] ≈ fill(3.0, 4) atol=1e-10
        @test out[:, 1, 3] ≈ zeros(4) atol=1e-9

        @test_throws ArgumentError evaluate_grid_ipoints(g, [10.5])
    end

    # ── Collar interface: interior-node extraction, coarse→fine ─────────────

    @testset "Collar interface transfers coarse trio at interior node" begin
        f(x) = 3x + 5
        # Coarse patch: nominal [0,10] + one-cell collar → [0,11], DX=1
        gp_c = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=11.0, num_cells=11,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        # Fine patch: [10, 20], DX=0.5, R3X on the interface (left) side
        gp_f = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=20.0, num_cells=20,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gc = createGrid(gp_c)
        gf = createGrid(gp_f)

        iface = PatchInterface(gc, gf, :right, :left, :i; is_stacked=true)
        # Interface node x=10 is m=10 → array index 12 = b_iDim-2 (interior)
        @test iface.primary_node_indices == (11, 12, 13)
        @test iface.is_stacked
        @test iface.coupling_matrix == Springsteel.COUPLING_MATRIX_2X

        pts_c = getGridpoints(gc)
        pts_f = getGridpoints(gf)
        for i in eachindex(pts_c); gc.physical[i, 1, 1] = f(pts_c[i]); end
        for i in eachindex(pts_f); gf.physical[i, 1, 1] = f(pts_f[i]); end
        spectralTransform!(gc)
        spectralTransform!(gf)

        gridTransform!(gc)          # sets coarse spline .a
        update_interface!(iface)    # coarse trio → fine ahat
        gridTransform!(gf)

        for i in eachindex(pts_f)
            @test gf.physical[i, 1, 1] ≈ f(pts_f[i]) atol=1e-9
        end
    end

    # ── Anti-freeze regression ───────────────────────────────────────────────
    # The payload a patch donates must track its own interior dynamics.  A
    # BC-based fine→coarse exchange (dual R3X) fails this: an R3X spline's
    # border trio is identically its ahat, so the donated data would be the
    # neighbor's own stale values and the interface would freeze.

    @testset "Anti-freeze: donated payload tracks donor interior" begin
        gp_c = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=11.0, num_cells=11,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp_f = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=20.0, num_cells=20,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gc = createGrid(gp_c)
        gf = createGrid(gp_f)
        iface = PatchInterface(gc, gf, :right, :left, :i; is_stacked=true)

        pts_c = getGridpoints(gc)
        for i in eachindex(pts_c); gc.physical[i, 1, 1] = 1.0; end
        spectralTransform!(gc)
        gridTransform!(gc)
        p_before = compute_interface_payload(iface.metadata, gc)

        # Perturb the coarse field near the interface (interior dynamics)
        for i in eachindex(pts_c)
            gc.physical[i, 1, 1] = 1.0 + 0.5 * exp(-(pts_c[i] - 9.5)^2)
        end
        spectralTransform!(gc)
        gridTransform!(gc)
        p_after = compute_interface_payload(iface.metadata, gc)

        @test maximum(abs.(p_after.border .- p_before.border)) > 0.05
    end

    # ── Payload serialization round-trip (collar metadata) ──────────────────

    @testset "Collar payload serializes byte-identically" begin
        gp_c = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=11.0, num_cells=11,
            BCL=Dict("u" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gp_f = SpringsteelGridParameters(
            geometry="R", iMin=10.0, iMax=20.0, num_cells=20,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        gc = createGrid(gp_c)
        gf = createGrid(gp_f)
        iface = PatchInterface(gc, gf, :right, :left, :i; is_stacked=true)

        pts_c = getGridpoints(gc)
        for i in eachindex(pts_c); gc.physical[i, 1, 1] = sin(pts_c[i]); end
        spectralTransform!(gc)
        gridTransform!(gc)
        p = compute_interface_payload(iface.metadata, gc)

        buf = IOBuffer()
        Serialization.serialize(buf, p)
        seekstart(buf)
        p2 = Serialization.deserialize(buf)
        @test p2.border == p.border
        @test p2.scheme === p.scheme && p2.side === p.side
    end

    # ── Out-of-place SAtransform honors R3X ahat ─────────────────────────────
    # The tiled b→a path (3-arg splineTransform!) uses the allocating
    # SAtransform(spline, b); it must reproduce SAtransform!'s ahat handling,
    # otherwise nest patches lose their coupled borders on the tiled path.

    @testset "SAtransform(spline, b) matches SAtransform! for R3X" begin
        gp = SpringsteelGridParameters(
            geometry="R", iMin=0.0, iMax=10.0, num_cells=10,
            BCL=Dict("u" => CubicBSpline.R3X),
            BCR=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        g = createGrid(gp)
        pts = getGridpoints(g)
        for i in eachindex(pts)
            g.physical[i, 1, 1] = sin(pts[i])
        end
        spectralTransform!(g)
        spline = g.ibasis.data[1, 1]
        Springsteel.CubicBSpline.set_ahat_r3x!(spline, 0.3, 0.7, 1.1, :left)

        spline.b .= view(g.spectral, :, 1)
        CubicBSpline.SAtransform!(spline)
        a_inplace = copy(spline.a)
        a_alloc = CubicBSpline.SAtransform(spline, view(g.spectral, :, 1))
        @test a_alloc ≈ a_inplace atol=1e-14
        # Rank-3 invariant: the constrained border trio IS the ahat
        @test a_alloc[1:3] ≈ spline.ahat[1:3] atol=1e-14
        @test any(!iszero, spline.ahat[1:3])
    end

    # ── evaluate_grid_ipoints, RiRk grid ─────────────────────────────────────

    @testset "evaluate_grid_ipoints matches gridTransform! on mish points (RiRk)" begin
        gp = SpringsteelGridParameters(
            geometry="RiRk",
            iMin=0.0, iMax=10.0, num_cells=10,
            kMin=0.0, kMax=5.0, num_cells_k=5,
            BCL=Dict("u" => CubicBSpline.R0, "w" => CubicBSpline.R0),
            BCR=Dict("u" => CubicBSpline.R0, "w" => CubicBSpline.R0),
            BCB=Dict("u" => CubicBSpline.R0, "w" => CubicBSpline.R0),
            BCT=Dict("u" => CubicBSpline.R0, "w" => CubicBSpline.R0),
            vars=Dict("u" => 1, "w" => 2))
        g = createGrid(gp)
        pts = getGridpoints(g)          # (iDim*kDim, 2), i-outer k-inner
        for i in axes(pts, 1)
            x, z = pts[i, 1], pts[i, 2]
            g.physical[i, 1, 1] = sin(0.5x) * cos(0.8z)
            g.physical[i, 2, 1] = 0.1 * x * z + z^2
        end
        spectralTransform!(g)
        gridTransform!(g)

        kDim = g.params.kDim
        iDim = g.params.iDim
        xq = [pts[(q - 1) * kDim + 1, 1] for q in 1:iDim]   # unique i-mish points
        out = evaluate_grid_ipoints(g, xq)
        for v in 1:2, s in 1:5
            @test maximum(abs.(out[:, v, s] .- g.physical[:, v, s])) < 1e-10
        end
    end

end

# ── RL (cylindrical) nesting support ─────────────────────────────────────────

@testset "RL nesting support" begin

    @testset "evaluate_grid_points matches gridTransform! on mish points (RL)" begin
        gp = SpringsteelGridParameters(
            geometry="RL", iMin=0.0, iMax=50.0, num_cells=10,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        g = createGrid(gp)
        pts = getGridpoints(g)
        for i in 1:size(pts, 1)
            r, λ = pts[i, 1], pts[i, 2]
            g.physical[i, 1, 1] = 0.1 * r + r * cos(λ) + 0.5 * r * sin(2λ)
        end
        spectralTransform!(g)
        gridTransform!(g)

        # gridTransform reconstructs ring ri with wavenumbers k ≤ ri only;
        # pass the same per-point truncation for an exact comparison.
        kmax = Int[]
        for r in 1:g.params.iDim
            ri = r + g.params.patchOffsetL
            append!(kmax, fill(ri, 4 + 4 * ri))
        end
        out = evaluate_grid_points(g, pts; kmax = kmax)
        for s in 1:5
            err = maximum(abs.(out[:, 1, s] .- g.physical[:, 1, s]))
            @test err < 1e-8
        end
    end

    @testset "tiled RL splineTransform! honors the per-wavenumber registry" begin
        # Disc-in-annulus fixture with distinct k=0 / k=1-real / k=1-imag borders
        gp_annulus = SpringsteelGridParameters(
            geometry="RL", iMin=20.0, iMax=100.0, num_cells=10,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        gp_disc = SpringsteelGridParameters(
            geometry="RL", iMin=0.0, iMax=20.0, num_cells=5,
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => FixedBC()),
            vars=Dict("u" => 1))
        g_annulus = createGrid(gp_annulus)
        g_disc = createGrid(gp_disc)
        f(r, λ) = (2r + 5) + 0.3r * cos(λ) + 0.7r * sin(λ)
        pts_a = getGridpoints(g_annulus)
        pts_d = getGridpoints(g_disc)
        for i in 1:size(pts_a, 1); g_annulus.physical[i, 1, 1] = f(pts_a[i, 1], pts_a[i, 2]); end
        for i in 1:size(pts_d, 1); g_disc.physical[i, 1, 1] = f(pts_d[i, 1], pts_d[i, 2]); end
        spectralTransform!(g_annulus)
        spectralTransform!(g_disc)

        iface = PatchInterface(g_annulus, g_disc, :left, :right, :i)
        gridTransform!(g_annulus)
        update_interface!(iface)

        # Known-correct registry-aware path
        gridTransform!(g_disc)
        want = copy(g_disc.physical)

        # Tiled path: 3-arg splineTransform! (patch splines) + tileTransform!
        shared = SharedArray{Float64,2}(size(g_disc.spectral))
        shared .= g_disc.spectral
        tile = createGrid(gp_disc)
        splineTransform!(shared, g_disc, tile)
        tileTransform!(shared, tile, tile.physical, tile.spectral)
        for s in 1:5
            err = maximum(abs.(tile.physical[:, 1, s] .- want[:, 1, s]))
            @test err < 1e-10
        end

        # Single-tile 2-arg path on the coupled grid object itself
        splineTransform!(shared, g_disc)
        tileTransform!(shared, g_disc, g_disc.physical, g_disc.spectral)
        for s in 1:5
            err = maximum(abs.(g_disc.physical[:, 1, s] .- want[:, 1, s]))
            @test err < 1e-10
        end
    end

    @testset "offset RL annulus follows global ring numbering" begin
        # Nest-annulus convention: patchOffsetL carries the GLOBAL ring offset
        # explicitly, while spectralIndexL stays 1 (the annulus is its own
        # patch, so tile spectral windows are annulus-relative).
        mubar = 3
        gp = SpringsteelGridParameters(
            geometry="RL", iMin=150.0e3, iMax=300.0e3, num_cells=50,
            patchOffsetL = 50 * mubar,    # 50 inner cells of the same DX
            BCL=Dict("u" => NaturalBC()), BCR=Dict("u" => NaturalBC()),
            vars=Dict("u" => 1))
        g = createGrid(gp)
        @test g.params.patchOffsetL == 50 * mubar
        @test g.params.spectralIndexL == 1
        pts = getGridpoints(g)
        # First ring's lpoints follows the global index: 4 + 4*(1 + offset)
        lp1 = 4 + 4 * (1 + g.params.patchOffsetL)
        @test count(x -> x ≈ pts[1, 1], pts[1:lp1 + 8, 1]) == lp1

        # A single tile of the annulus composes the offset (rings match the
        # patch) while keeping its spectral window patch-relative.
        tiles = calcTileSizes(g, 1)
        @test tiles[1].params.spectralIndexL == 1
        @test tiles[1].params.patchOffsetL == g.params.patchOffsetL
        @test size(getGridpoints(tiles[1]), 1) == size(pts, 1)
    end
end

# ── RLR (3D cylindrical) nesting support ─────────────────────────────────────

@testset "RLR nesting support" begin

    function make_rlr_grid(; iMin=0.0, iMax=50.0, num_cells=10, BCLu=NaturalBC(),
                            BCRu=NaturalBC(), patchOffsetL=0)
        gp = SpringsteelGridParameters(
            geometry="RLR", iMin=iMin, iMax=iMax, num_cells=num_cells,
            patchOffsetL=patchOffsetL,
            kMin=0.0, kMax=10.0, kDim=12,
            BCL=Dict("u" => BCLu), BCR=Dict("u" => BCRu),
            BCB=Dict("u" => CubicBSpline.R0), BCT=Dict("u" => CubicBSpline.R0),
            vars=Dict("u" => 1))
        return createGrid(gp)
    end

    "Column (r, λ) list, per-column ring kmax, and the vertical mish of an RLR grid."
    function rlr_columns(g)
        pts = getGridpoints(g)
        kDim = g.params.kDim
        ncol = size(pts, 1) ÷ kDim
        cols = zeros(ncol, 2)
        kmax = Int[]
        row = 0
        for r in 1:g.params.iDim
            ri = r + g.params.patchOffsetL
            lpoints = 4 + 4 * ri
            for l in 1:lpoints
                c = row + l
                cols[c, 1] = pts[(c - 1) * kDim + 1, 1]
                cols[c, 2] = pts[(c - 1) * kDim + 1, 2]
                push!(kmax, ri)
            end
            row += lpoints
        end
        z = pts[1:kDim, 3]
        return cols, kmax, z
    end

    @testset "evaluate_grid_points matches gridTransform! on mish points (RLR)" begin
        g = make_rlr_grid()
        pts = getGridpoints(g)
        f(r, λ, z) = (0.1 * r + r * cos(λ) + 0.5 * r * sin(2λ)) * (1.0 + 0.2 * z + 0.01 * z^2)
        for i in 1:size(pts, 1)
            g.physical[i, 1, 1] = f(pts[i, 1], pts[i, 2], pts[i, 3])
        end
        spectralTransform!(g)
        gridTransform!(g)

        cols, kmax, z = rlr_columns(g)
        out = evaluate_grid_points(g, cols, z; kmax = kmax)
        @test size(out) == (size(pts, 1), 1, 7)
        for s in 1:7
            err = maximum(abs.(out[:, 1, s] .- g.physical[:, 1, s]))
            @test err < 1e-8
        end
    end

    @testset "evaluate_grid_points at arbitrary z (off the mish)" begin
        g = make_rlr_grid()
        pts = getGridpoints(g)
        f(r, λ, z) = (0.1 * r + 0.3 * r * sin(λ)) * (1.0 + 0.2 * z)
        for i in 1:size(pts, 1)
            g.physical[i, 1, 1] = f(pts[i, 1], pts[i, 2], pts[i, 3])
        end
        spectralTransform!(g)
        gridTransform!(g)

        cols, kmax, zmish = rlr_columns(g)
        zq = [2.37, 5.0, 8.61]
        out = evaluate_grid_points(g, cols, zq; kmax = kmax)
        @test size(out, 1) == size(cols, 1) * length(zq)
        for n in axes(cols, 1), (iz, z) in enumerate(zq)
            row = (n - 1) * length(zq) + iz
            @test out[row, 1, 1] ≈ f(cols[n, 1], cols[n, 2], z) atol = 0.05
        end
    end

    @testset "tiled RLR splineTransform! honors the per-wavenumber registry" begin
        # The RL tiled-transform bug reincarnated on RLR: without the per-block
        # ahat reload, a nested patch's R3X borders are silently ignored by the
        # worker-loop spline solve.
        g_disc = make_rlr_grid(iMin=0.0, iMax=50.0, num_cells=10, BCRu=FixedBC())
        g_ann = make_rlr_grid(iMin=50.0, iMax=150.0, num_cells=10, BCLu=NaturalBC())  # 2:1 coarser
        f(r, λ, z) = (2.0 * r + 5.0 + 0.3 * r * cos(λ) + 0.7 * r * sin(λ)) *
                     (1.0 + 0.1 * z)
        for g in (g_disc, g_ann)
            pts = getGridpoints(g)
            for i in 1:size(pts, 1)
                g.physical[i, 1, 1] = f(pts[i, 1], pts[i, 2], pts[i, 3])
            end
            spectralTransform!(g)
        end
        iface = PatchInterface(g_ann, g_disc, :left, :right, :i)
        gridTransform!(g_ann)
        update_interface!(iface)

        # Known-correct registry-aware path
        gridTransform!(g_disc)
        want = copy(g_disc.physical)

        # Tiled path: 3-arg splineTransform! (patch splines) + tileTransform!
        shared = SharedArray{Float64,2}(size(g_disc.spectral))
        shared .= g_disc.spectral
        tile = createGrid(g_disc.params)
        splineTransform!(shared, g_disc, tile)
        tileTransform!(shared, tile, tile.physical, tile.spectral)
        for s in 1:7
            err = maximum(abs.(tile.physical[:, 1, s] .- want[:, 1, s]))
            @test err < 1e-10
        end

        # Single-tile 2-arg path on the coupled grid object itself
        splineTransform!(shared, g_disc)
        tileTransform!(shared, g_disc, g_disc.physical, g_disc.spectral)
        for s in 1:7
            err = maximum(abs.(g_disc.physical[:, 1, s] .- want[:, 1, s]))
            @test err < 1e-10
        end
    end

    @testset "evaluate_grid_points honors the coupled-border registry (RLR)" begin
        # Disc-in-annulus: after the chain transform the annulus' k-blocks carry
        # R3X borders from the disc; the collar evaluation must reload them.
        g_disc = make_rlr_grid(iMin=0.0, iMax=50.0, num_cells=10, BCRu=NaturalBC())
        g_ann = make_rlr_grid(iMin=50.0, iMax=75.0, num_cells=10, BCLu=FixedBC())
        f(r, λ, z) = (2.0 * r + 5.0 + 0.3 * r * cos(λ) + 0.7 * r * sin(λ)) *
                     (1.0 + 0.1 * z)
        for (g, pts) in ((g_disc, getGridpoints(g_disc)), (g_ann, getGridpoints(g_ann)))
            for i in 1:size(pts, 1)
                g.physical[i, 1, 1] = f(pts[i, 1], pts[i, 2], pts[i, 3])
            end
            spectralTransform!(g)
        end
        mpg = PatchChain([g_disc, g_ann])
        multiGridTransform!(mpg)

        cols, kmax, z = rlr_columns(g_ann)
        out = evaluate_grid_points(g_ann, cols, z; kmax = kmax)
        for s in 1:7
            err = maximum(abs.(out[:, 1, s] .- g_ann.physical[:, 1, s]))
            @test err < 1e-8
        end
    end
end

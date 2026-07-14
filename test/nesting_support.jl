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

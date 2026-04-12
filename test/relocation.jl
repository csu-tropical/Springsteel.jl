@testset "Grid Relocation" begin

    # ── Test helpers ──────────────────────────────────────────────────────

    function make_rl_vortex(nc=10)
        gp = SpringsteelGridParameters(geometry="RL", iMin=0.0, iMax=10.0,
            num_cells=nc, vars=Dict("u"=>1),
            BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0),
            max_wavenumber=Dict("default"=>-1))
        g = createGrid(gp)
        pts = getGridpoints(g)
        for i in 1:size(pts, 1)
            r = pts[i, 1]
            g.physical[i, 1, 1] = exp(-r^2 / 4.0)
        end
        spectralTransform!(g)
        gridTransform!(g)
        return g
    end

    function make_rlz_vortex(nc=10, kDim=8)
        gp = SpringsteelGridParameters(geometry="RLZ", iMin=0.0, iMax=10.0,
            kMin=0.0, kMax=5.0, num_cells=nc, kDim=kDim,
            vars=Dict("u"=>1),
            BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0),
            BCB=Dict("u"=>Chebyshev.R0), BCT=Dict("u"=>Chebyshev.R0),
            max_wavenumber=Dict("default"=>-1))
        g = createGrid(gp)
        pts = getGridpoints(g)
        for i in 1:size(pts, 1)
            r = pts[i, 1]
            z = pts[i, 3]
            g.physical[i, 1, 1] = exp(-r^2 / 4.0) * cos(π * z / 5.0)
        end
        spectralTransform!(g)
        gridTransform!(g)
        return g
    end

    # ── RL basic relocation ───────────────────────────────────────────────

    @testset "RL relocate_grid basic" begin
        g = make_rl_vortex()
        g2 = relocate_grid(g, (0.5, 0.0); boundary=:azimuthal_mean)

        @test size(g2.physical) == size(g.physical)
        @test size(g2.spectral) == size(g.spectral)
        @test maximum(g2.physical[:, 1, 1]) > 0.0
        @test !any(isnan, g2.physical[:, 1, 1])
    end

    @testset "RL relocate_grid! in-place" begin
        g = make_rl_vortex()
        g_copy = createGrid(g.params)
        g_copy.physical .= g.physical
        g_copy.spectral .= g.spectral

        g2 = relocate_grid(g, (0.5, 0.0))
        ret = relocate_grid!(g_copy, (0.5, 0.0))

        @test ret === g_copy
        @test g2.physical[:, 1, 1] ≈ g_copy.physical[:, 1, 1] atol=1e-12
    end

    @testset "RL round-trip" begin
        g = make_rl_vortex(15)
        orig_phys = copy(g.physical[:, 1, 1])

        g2 = relocate_grid(g, (0.3, 0.2); boundary=:azimuthal_mean)
        g3 = relocate_grid(g2, (-0.3, -0.2); boundary=:azimuthal_mean)

        inner_mask = getGridpoints(g3)[:, 1] .< 8.0
        err = maximum(abs.(g3.physical[inner_mask, 1, 1] .- orig_phys[inner_mask]))
        @test err < 0.2
    end

    @testset "RL azimuthal mean preservation" begin
        g = make_rl_vortex(15)
        g2 = relocate_grid(g, (0.1, 0.0); boundary=:azimuthal_mean)

        pts = getGridpoints(g)
        pts2 = getGridpoints(g2)

        safe_r = 0.5 * g.params.iMax
        inner_mask = pts[:, 1] .< safe_r
        inner_mask2 = pts2[:, 1] .< safe_r

        mean_orig = sum(g.physical[inner_mask, 1, 1]) / count(inner_mask)
        mean_reloc = sum(g2.physical[inner_mask2, 1, 1]) / count(inner_mask2)

        @test abs(mean_orig - mean_reloc) / abs(mean_orig) < 0.1
    end

    # ── Boundary strategies ───────────────────────────────────────────────

    @testset "RL boundary :nan" begin
        g = make_rl_vortex()
        g2 = relocate_grid(g, (8.0, 0.0); boundary=:nan)
        @test any(isnan, g2.physical[:, 1, 1])
    end

    @testset "RL boundary :nearest" begin
        g = make_rl_vortex()
        g2 = relocate_grid(g, (8.0, 0.0); boundary=:nearest)
        @test !any(isnan, g2.physical[:, 1, 1])
    end

    @testset "RL boundary :azimuthal_mean" begin
        g = make_rl_vortex()
        g2 = relocate_grid(g, (8.0, 0.0); boundary=:azimuthal_mean)
        @test !any(isnan, g2.physical[:, 1, 1])
    end

    @testset "RL boundary :bc_respecting error on R0" begin
        g = make_rl_vortex()
        @test_throws ArgumentError relocate_grid(g, (8.0, 0.0); boundary=:bc_respecting)
    end

    @testset "RL boundary :bc_respecting with Dirichlet" begin
        gp = SpringsteelGridParameters(geometry="RL", iMin=0.0, iMax=10.0,
            num_cells=10, vars=Dict("u"=>1),
            BCL=Dict("u"=>CubicBSpline.R0),
            BCR=Dict("u"=>CubicBSpline.R1T0),
            max_wavenumber=Dict("default"=>-1))
        g = createGrid(gp)
        pts = getGridpoints(g)
        for i in 1:size(pts, 1)
            g.physical[i, 1, 1] = exp(-pts[i, 1]^2 / 4.0)
        end
        spectralTransform!(g)
        gridTransform!(g)
        g2 = relocate_grid(g, (8.0, 0.0); boundary=:bc_respecting)
        @test !any(isnan, g2.physical[:, 1, 1])
    end

    # ── Error paths ───────────────────────────────────────────────────────

    @testset "relocate_grid on unsupported geometry" begin
        gp = SpringsteelGridParameters(geometry="R", iMin=0.0, iMax=1.0,
            num_cells=10, vars=Dict("u"=>1),
            BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0))
        g = createGrid(gp)
        @test_throws ArgumentError relocate_grid(g, (0.1, 0.0))
        @test_throws ArgumentError relocate_grid!(g, (0.1, 0.0))
    end

    @testset "Unknown boundary strategy" begin
        g = make_rl_vortex()
        @test_throws ArgumentError relocate_grid(g, (0.1, 0.0); boundary=:invalid)
    end

    # ── RLZ relocation ────────────────────────────────────────────────────

    @testset "RLZ relocate_grid basic" begin
        g = make_rlz_vortex()
        g2 = relocate_grid(g, (0.5, 0.0); boundary=:azimuthal_mean)

        @test size(g2.physical) == size(g.physical)
        @test size(g2.spectral) == size(g.spectral)
        @test maximum(g2.physical[:, 1, 1]) > 0.0
        @test !any(isnan, g2.physical[:, 1, 1])
    end

    @testset "RLZ relocate_grid! in-place" begin
        g = make_rlz_vortex()
        g_copy = createGrid(g.params)
        g_copy.physical .= g.physical
        g_copy.spectral .= g.spectral

        g2 = relocate_grid(g, (0.5, 0.0))
        relocate_grid!(g_copy, (0.5, 0.0))

        @test g2.physical[:, 1, 1] ≈ g_copy.physical[:, 1, 1] atol=1e-12
    end

    @testset "RLZ round-trip" begin
        g = make_rlz_vortex(10, 8)
        orig_phys = copy(g.physical[:, 1, 1])

        g2 = relocate_grid(g, (0.3, 0.2); boundary=:azimuthal_mean)
        g3 = relocate_grid(g2, (-0.3, -0.2); boundary=:azimuthal_mean)

        pts = getGridpoints(g3)
        inner_mask = pts[:, 1] .< 8.0
        err = maximum(abs.(g3.physical[inner_mask, 1, 1] .- orig_phys[inner_mask]))
        @test err < 0.2
    end

    # ── R2: Conservation and taper ────────────────────────────────────────

    @testset "RL k=0 conservation (axisymmetric)" begin
        gp = SpringsteelGridParameters(geometry="RL", iMin=0.0, iMax=10.0,
            num_cells=15, vars=Dict("u"=>1),
            BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0),
            max_wavenumber=Dict("default"=>-1))
        g = createGrid(gp)
        pts = getGridpoints(g)
        for i in 1:size(pts, 1)
            g.physical[i, 1, 1] = exp(-pts[i, 1]^2 / 4.0)
        end
        spectralTransform!(g)
        gridTransform!(g)
        spec_orig = copy(g.spectral[:, 1])

        g2 = relocate_grid(g, (0.1, 0.0); boundary=:azimuthal_mean)
        b_iDim = gp.b_iDim
        k0_orig = spec_orig[1:b_iDim]
        k0_reloc = g2.spectral[1:b_iDim, 1]

        @test maximum(abs.(k0_orig .- k0_reloc)) < 0.01
    end

    @testset "RL taper smoothness" begin
        g = make_rl_vortex(15)
        g_no_taper = relocate_grid(g, (5.0, 0.0); boundary=:azimuthal_mean, taper_width=0)
        g_taper    = relocate_grid(g, (5.0, 0.0); boundary=:azimuthal_mean, taper_width=3)

        @test !any(isnan, g_taper.physical[:, 1, 1])
        @test size(g_taper.physical) == size(g_no_taper.physical)
    end

    @testset "RLZ taper" begin
        g = make_rlz_vortex()
        g_taper = relocate_grid(g, (5.0, 0.0); boundary=:azimuthal_mean, taper_width=2)
        @test !any(isnan, g_taper.physical[:, 1, 1])
    end

    @testset "RLZ boundary strategies" begin
        g = make_rlz_vortex()
        g_nan = relocate_grid(g, (8.0, 0.0); boundary=:nan)
        @test any(isnan, g_nan.physical[:, 1, 1])

        g_near = relocate_grid(g, (8.0, 0.0); boundary=:nearest)
        @test !any(isnan, g_near.physical[:, 1, 1])

        g_azm = relocate_grid(g, (8.0, 0.0); boundary=:azimuthal_mean)
        @test !any(isnan, g_azm.physical[:, 1, 1])
    end
end

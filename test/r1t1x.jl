@testset "R1T1X inhomogeneous Neumann" begin

    f(x)  = exp(-2x) + 0.3x^2
    fp(x) = -2exp(-2x) + 0.6x

    mkspline(bcl, bcr) = CubicBSpline.Spline1D(CubicBSpline.SplineParameters(
        xmin = 0.0, xmax = 2.0, num_cells = 20, BCL = bcl, BCR = bcr))

    @testset "constant definition and rank" begin
        @test CubicBSpline.R1T1X["α1"] == CubicBSpline.R1T1["α1"]
        @test CubicBSpline.R1T1X["β1"] == CubicBSpline.R1T1["β1"]
        @test haskey(CubicBSpline.R1T1X, "X1")
        # Same gammaBC as homogeneous R1T1 => same admissible subspace, hence
        # the same solver stability. This is the whole point of R1T1X.
        sp_x = CubicBSpline.SplineParameters(xmin = 0.0, xmax = 2.0, num_cells = 20,
            BCL = CubicBSpline.R1T1X, BCR = CubicBSpline.R1T1X)
        sp_h = CubicBSpline.SplineParameters(xmin = 0.0, xmax = 2.0, num_cells = 20,
            BCL = CubicBSpline.R1T1, BCR = CubicBSpline.R1T1)
        @test CubicBSpline.calcGammaBC(sp_x) == CubicBSpline.calcGammaBC(sp_h)
        @test CubicBSpline._has_ahat(sp_x) == true
        @test CubicBSpline._has_ahat(sp_h) == false
    end

    @testset "prescribed wall derivative is attained" begin
        s = mkspline(CubicBSpline.R1T1X, CubicBSpline.R1T1X)
        s.uMish .= f.(s.mishPoints)
        SBtransform!(s)
        CubicBSpline.set_ahat_neumann!(s, fp(0.0), :left)
        CubicBSpline.set_ahat_neumann!(s, fp(2.0), :right)
        SAtransform!(s)
        dl = CubicBSpline.SItransform(s.params, s.a, [0.0], 1)[1]
        dr = CubicBSpline.SItransform(s.params, s.a, [2.0], 1)[1]
        @test dl ≈ fp(0.0) atol = 1e-12
        @test dr ≈ fp(2.0) atol = 1e-12
        # and the interior fit is still a fit
        SItransform!(s)
        @test maximum(abs, s.uMish .- f.(s.mishPoints)) < 1e-3
    end

    @testset "du = 0 reproduces homogeneous R1T1 bitwise" begin
        h = mkspline(CubicBSpline.R1T1, CubicBSpline.R1T1)
        h.uMish .= f.(h.mishPoints); SBtransform!(h); SAtransform!(h)
        z = mkspline(CubicBSpline.R1T1X, CubicBSpline.R1T1X)
        z.uMish .= f.(z.mishPoints); SBtransform!(z)
        CubicBSpline.set_ahat_neumann!(z, 0.0, :left)
        CubicBSpline.set_ahat_neumann!(z, 0.0, :right)
        SAtransform!(z)
        @test z.a == h.a
    end

    @testset "one-sided: R1T1X left, plain R1T1 right" begin
        s = mkspline(CubicBSpline.R1T1X, CubicBSpline.R1T1)
        s.uMish .= f.(s.mishPoints)
        SBtransform!(s)
        CubicBSpline.set_ahat_neumann!(s, fp(0.0), :left)
        SAtransform!(s)
        @test CubicBSpline.SItransform(s.params, s.a, [0.0], 1)[1] ≈ fp(0.0) atol = 1e-12
        @test CubicBSpline.SItransform(s.params, s.a, [2.0], 1)[1] ≈ 0.0 atol = 1e-12
    end

    @testset "ahat is affine: response is linear in du" begin
        # Needed for the claim that a state-dependent du cannot change the
        # stability of a solver built on this basis.
        base = mkspline(CubicBSpline.R1T1X, CubicBSpline.R1T1)
        base.uMish .= f.(base.mishPoints); SBtransform!(base)
        CubicBSpline.set_ahat_neumann!(base, 0.0, :left); SAtransform!(base)
        a0 = copy(base.a)

        a = Vector{Vector{Float64}}()
        for du in (1.0, 2.0)
            s = mkspline(CubicBSpline.R1T1X, CubicBSpline.R1T1)
            s.uMish .= f.(s.mishPoints); SBtransform!(s)
            CubicBSpline.set_ahat_neumann!(s, du, :left); SAtransform!(s)
            push!(a, s.a .- a0)
        end
        @test a[2] ≈ 2 .* a[1] rtol = 1e-12
    end

    @testset "reusable: changing du between fits leaves no residue" begin
        s = mkspline(CubicBSpline.R1T1X, CubicBSpline.R1T1)
        for du in (3.0, -1.5, 0.0)
            s.uMish .= f.(s.mishPoints); SBtransform!(s)
            CubicBSpline.set_ahat_neumann!(s, du, :left); SAtransform!(s)
            @test CubicBSpline.SItransform(s.params, s.a, [0.0], 1)[1] ≈ du atol = 1e-12
        end
    end

    @testset "grid-level: wall derivative attained, i-derivative uncorrupted" begin
        # Regression for a bug that cost a 2500x jump in a Scythe gradient-wind
        # residual: the 2-D transform takes the i-derivative BEFORE fitting in k,
        # so the dr = 1/2 passes need d(wall)/di, not the wall value. Feeding them
        # the value asserts dg/di = g and wrecks the i-derivative slots at the
        # boundary cell, while the z-derivative slot still looks perfect.
        xbc = Dict("p" => CubicBSpline.R1T1X, "q" => NeumannBC())
        nbc = Dict("p" => NeumannBC(), "q" => NeumannBC())
        gp = GridParameters(geometry = "RiRk", iMin = 0.0, iMax = 1.0, num_cells_i = 8,
            kMin = 0.0, kMax = 2.0, num_cells_k = 16,
            BCL = nbc, BCR = nbc, BCB = xbc, BCT = xbc, vars = Dict("p" => 1, "q" => 2))
        g = createGrid(gp)
        pts = getGridpoints(g); kDim = g.params.kDim; iDim = g.params.iDim
        F(x, z)  = (1 + x) * (exp(-2z) + 0.3z^2)          # separable
        Fz(x, z) = (1 + x) * (-2exp(-2z) + 0.6z)
        Fx(x, z) = exp(-2z) + 0.3z^2
        for n in 1:size(pts, 1), v in 1:2
            g.physical[n, v, 1] = F(pts[n, 1], pts[n, 2])
        end
        xr = [pts[(r-1)*kDim + 1, 1] for r in 1:iDim]
        set_wall_derivatives!(g, :bottom, "p", [Fz(x, 0.0) for x in xr])
        set_wall_derivatives!(g, :top,    "p", [Fz(x, 2.0) for x in xr])
        spectralTransform!(g); gridTransform!(g)

        z0 = pts[1, 2]
        # slot 4 = d/dz: R1T1X must beat homogeneous Neumann by orders of magnitude
        errp = maximum(abs(g.physical[(r-1)*kDim+1, 1, 4] - Fz(xr[r], z0)) for r in 1:iDim)
        errq = maximum(abs(g.physical[(r-1)*kDim+1, 2, 4] - Fz(xr[r], z0)) for r in 1:iDim)
        @test errp < 0.05
        @test errq / errp > 50

        # slot 2 = d/di: must be UNHARMED by the wall condition. Compared against
        # the Neumann variable, which cannot be affected by it at all.
        dip = maximum(abs(g.physical[(r-1)*kDim+1, 1, 2] - Fx(xr[r], z0)) for r in 1:iDim)
        diq = maximum(abs(g.physical[(r-1)*kDim+1, 2, 2] - Fx(xr[r], z0)) for r in 1:iDim)
        @test dip < 10 * max(diq, 1e-12)
    end

    @testset "set_ahat_neumann! rejects a bad side" begin
        s = mkspline(CubicBSpline.R1T1X, CubicBSpline.R1T1X)
        @test_throws ArgumentError CubicBSpline.set_ahat_neumann!(s, 1.0, :middle)
    end
end

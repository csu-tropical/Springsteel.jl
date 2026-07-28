@testset "Positivity-constrained SA solve" begin

    # A sharp positive-definite spike: narrow enough that the unconstrained cubic
    # least-squares fit undershoots on its flanks, which is the whole problem this
    # constraint exists to remove.
    NC = 100
    _sp(bc) = SplineParameters(xmin = 0.0, xmax = 25.0e3, num_cells = NC, mubar = 3,
                               l_q = 2.0, BCL = bc, BCR = bc)
    _spike(pts) = [exp(-((z - 6000.0) / 400.0)^2)^4 for z in pts]

    # The model's own mass metric: Gauss weights on the mish (exact for a cubic).
    _, _qw = CubicBSpline._quadrature_rule(3, :gauss)
    WMISH = repeat(_qw .* (25.0e3 / NC), outer = NC)
    _mass(u) = sum(WMISH .* u)

    @testset "basis_integrals are exact and positive" begin
        s = Spline1D(_sp(CubicBSpline.R0))
        w = basis_integrals(s)
        @test all(w .> 0.0)
        # Partition of unity: Σᵢ ∫Bᵢ = ∫1 = domain length.
        @test sum(w) ≈ 25.0e3 rtol = 1e-12
        # Interior basis functions each integrate to exactly one cell width.
        @test w[NC ÷ 2] ≈ 25.0e3 / NC rtol = 1e-12
    end

    @testset "unconstrained fit undershoots, constrained does not" for bc in
            (CubicBSpline.R0, CubicBSpline.R1T1)
        s_unc = Spline1D(_sp(bc))
        s_con = Spline1D(_sp(bc); lower = zeros(NC + 3))
        u = _spike(s_unc.mishPoints)

        for s in (s_unc, s_con)
            s.uMish .= u
            SBtransform!(s)
        end
        SAtransform!(s_unc);         SItransform!(s_unc)
        SAtransform_bounded!(s_con); SItransform!(s_con)

        # The premise: the plain fit really does go negative on a positive-definite input.
        @test minimum(u) ≥ 0.0
        @test minimum(s_unc.a) < -1e-3
        @test minimum(s_unc.uMish) < -1e-3

        # The constraint: non-negative coefficients, hence a non-negative reconstruction
        # everywhere (convex-hull property), not merely at the mish points.
        @test minimum(s_con.a) ≥ 0.0
        @test minimum(s_con.uMish) ≥ 0.0

        # Conservation, in both the coefficient metric and the model's Gauss metric.
        w = basis_integrals(s_con)
        @test sum(w .* s_con.a) ≈ sum(w .* s_unc.a) rtol = 1e-12
        @test _mass(s_con.uMish) ≈ _mass(u) rtol = 1e-12
        @test bound_shortfall(s_con) == 0.0

        # The limiter pays for the clipped deficit by shaving the peak, not by inventing mass.
        @test maximum(s_con.uMish) < maximum(s_unc.uMish)
    end

    @testset "boundary condition survives clipping" begin
        s = Spline1D(_sp(CubicBSpline.R1T1); lower = zeros(NC + 3))
        s.uMish .= _spike(s.mishPoints)
        SBtransform!(s)
        SAtransform_bounded!(s)
        # R1T1 mirrors the ghost coefficients onto their interior partners; clipping in
        # free space and re-forming a = Γᵀy must preserve that exactly.
        @test s.a[1] == s.a[3]
        @test s.a[end] == s.a[end - 2]
    end

    @testset "fast path is bit-identical when no bound is active" begin
        for bc in (CubicBSpline.R0, CubicBSpline.R1T1)
            s_unc = Spline1D(_sp(bc))
            s_con = Spline1D(_sp(bc); lower = zeros(NC + 3))
            # A smooth strictly positive field whose fit needs no clipping at all.
            u = [1.0 + 0.5 * sin(2π * z / 25.0e3) for z in s_unc.mishPoints]
            for s in (s_unc, s_con)
                s.uMish .= u
                SBtransform!(s)
            end
            SAtransform!(s_unc)
            SAtransform_bounded!(s_con)
            @test s_unc.a == s_con.a          # bitwise, not approximate
        end
    end

    @testset "no bound installed leaves the transform untouched" begin
        s_a = Spline1D(_sp(CubicBSpline.R0))
        s_b = Spline1D(_sp(CubicBSpline.R0))
        u = _spike(s_a.mishPoints)
        for s in (s_a, s_b)
            s.uMish .= u
            SBtransform!(s)
        end
        SAtransform!(s_a)
        SAtransform_bounded!(s_b)           # unbounded spline: must be the same transform
        @test s_a.a == s_b.a
    end

    @testset "nonzero and per-coefficient bounds" begin
        # A perturbation carried against a constant reference of 0.5 is bounded below by
        # -0.5, and the reconstruction must respect that rather than zero.
        s = Spline1D(_sp(CubicBSpline.R0); lower = fill(-0.5, NC + 3))
        s.uMish .= _spike(s.mishPoints) .- 0.5
        SBtransform!(s)
        SAtransform_bounded!(s)
        SItransform!(s)
        @test minimum(s.a) ≥ -0.5 - 1e-14
        @test minimum(s.uMish) ≥ -0.5 - 1e-14
    end

    @testset "support-minimum rule from a reference profile" begin
        s = Spline1D(_sp(CubicBSpline.R0))
        ref = [1.0 + z / 25.0e3 for z in s.mishPoints]     # strictly positive, varying
        set_lower_bound_from_profile!(s, ref)
        # Sufficient: every coefficient bound is at most the negated local reference.
        @test all(s.lower .<= 0.0)
        @test length(s.lower) == NC + 3
        # A perturbation that would drive the total negative gets caught.
        s.uMish .= -2.0 .* ref
        SBtransform!(s)
        SAtransform_bounded!(s)
        SItransform!(s)
        @test all(s.uMish .+ ref .>= -1e-9)
    end

    @testset "unsupported boundary conditions are rejected" begin
        # Dirichlet forms a[1] = -4a[2] - a[3]: a box constraint on the free coefficients
        # says nothing about the slaved one, so this must throw rather than mis-clip.
        @test_throws ErrorException Spline1D(_sp(CubicBSpline.R1T0); lower = zeros(NC + 3))
        @test_throws ErrorException Spline1D(_sp(CubicBSpline.R3); lower = zeros(NC + 3))
        # Wrong length is a DimensionMismatch, not a silent truncation.
        s = Spline1D(_sp(CubicBSpline.R0))
        @test_throws DimensionMismatch set_lower_bound!(s, zeros(NC))
    end

    @testset "clear_lower_bound! restores the unconstrained solve" begin
        s_ref = Spline1D(_sp(CubicBSpline.R0))
        s = Spline1D(_sp(CubicBSpline.R0); lower = zeros(NC + 3))
        clear_lower_bound!(s)
        u = _spike(s.mishPoints)
        for t in (s_ref, s)
            t.uMish .= u
            SBtransform!(t)
        end
        SAtransform!(s_ref)
        SAtransform_bounded!(s)
        @test s_ref.a == s.a
        @test minimum(s.a) < 0.0            # and it really is the unconstrained answer
    end

    @testset "feasibility theorem: b >= 0 implies the next leg has non-negative mass" begin
        # The load-bearing identity behind bounding an earlier leg (RULE 2 in the
        # MULTI-DIMENSIONAL DESIGN note): Σₘ bₘ = ∫u by partition of unity, and
        # Σₘ aₘ wₘ = ∫u because the fit preserves the integral — the l_q penalty contributes
        # nothing, since Σₘ φₘ ≡ 1 has zero third derivative. Hence a componentwise
        # non-negative b guarantees the SA solve's column mass is non-negative, so the next
        # leg can never hit the infeasible branch.
        for l_q in (0.0, 2.0, 8.0)
            s = Spline1D(SplineParameters(xmin = 0.0, xmax = 25.0e3, num_cells = NC,
                                          mubar = 3, l_q = l_q,
                                          BCL = CubicBSpline.R0, BCR = CubicBSpline.R0))
            s.uMish .= _spike(s.mishPoints)
            SBtransform!(s)
            SAtransform!(s)
            w = basis_integrals(s)
            @test sum(s.b) ≈ sum(w .* s.a) rtol = 1e-11     # holds for every l_q
            @test sum(s.b) ≈ _mass(s.uMish) rtol = 1e-11
            @test all(s.b .>= 0.0)                          # non-negative input ⟹ b ≥ 0
            @test sum(w .* s.a) > 0.0                       # ⟹ the column is feasible
        end
    end

    @testset "grid-level opt-in via GridParameters.positivity" begin
        vars = Dict("q" => 1, "u" => 2)
        gp = SpringsteelGridParameters(
            geometry = "RiRk",
            iMin = 0.0, iMax = 10.0e3, num_cells = 20, mubar = 3,
            kMin = 0.0, kMax = 10.0e3, num_cells_k = 20, kDim = 60,
            BCL = Dict("default" => CubicBSpline.R0),
            BCR = Dict("default" => CubicBSpline.R0),
            BCB = Dict("default" => CubicBSpline.R0),
            BCT = Dict("default" => CubicBSpline.R0),
            vars = vars,
            positivity = Dict("q" => Dict(:k => 0.0)))
        grid = createGrid(gp)
        # Only the named variable's k-column is bounded; everything else is untouched.
        @test !isempty(grid.kbasis.data[vars["q"]].lower)
        @test all(grid.kbasis.data[vars["q"]].lower .== 0.0)
        @test isempty(grid.kbasis.data[vars["u"]].lower)
        @test isempty(grid.ibasis.data[1, vars["q"]].lower)

        # End to end through the grid transform: a spike in q comes back non-negative,
        # while the same field in the unbounded slot u does not.
        kDim = gp.kDim
        for i in 1:gp.iDim, k in 1:kDim
            z = grid.kbasis.data[1].mishPoints[k]
            x = grid.ibasis.data[1, 1].mishPoints[i]
            val = exp(-((z - 5000.0) / 300.0)^2)^4 * exp(-((x - 5000.0) / 1500.0)^2)
            grid.physical[(i - 1) * kDim + k, vars["q"], 1] = val
            grid.physical[(i - 1) * kDim + k, vars["u"], 1] = val
        end
        spectralTransform!(grid)
        gridTransform!(grid)
        @test minimum(grid.physical[:, vars["q"], 1]) ≥ 0.0
        @test minimum(grid.physical[:, vars["u"], 1]) < 0.0
    end

    @testset "both legs bounded: positivity with no shortfall" begin
        # RULE 1 gives positivity from the last leg alone; RULE 2 says the EARLIER leg is
        # what keeps the last one feasible. This checks both, on a field deliberately
        # shaped like the O01 rain shaft: a narrow column of rain with empty air either
        # side, so the k-only case has columns made entirely of ringing.
        function _build(pos)
            vars = Dict("q" => 1)
            gp = SpringsteelGridParameters(
                geometry = "RiRk",
                iMin = 0.0, iMax = 30.0e3, num_cells = 30, mubar = 3,
                kMin = 0.0, kMax = 10.0e3, num_cells_k = 20, kDim = 60,
                BCL = Dict("default" => CubicBSpline.R0),
                BCR = Dict("default" => CubicBSpline.R0),
                BCB = Dict("default" => CubicBSpline.R0),
                BCT = Dict("default" => CubicBSpline.R0),
                vars = vars, positivity = pos)
            g = createGrid(gp)
            kD = gp.kDim
            for i in 1:gp.iDim, k in 1:kD
                z = g.kbasis.data[1].mishPoints[k]
                x = g.ibasis.data[1, 1].mishPoints[i]
                g.physical[(i - 1) * kD + k, 1, 1] =
                    exp(-((z - 5000.0) / 400.0)^2)^4 * exp(-((x - 15000.0) / 800.0)^2)^4
            end
            spectralTransform!(g)
            gridTransform!(g)
            return g
        end

        g_none = _build(Dict{String,Dict{Symbol,Float64}}())
        g_k    = _build(Dict("q" => Dict(:k => 0.0)))
        g_ik   = _build(Dict("q" => Dict(:i => 0.0, :k => 0.0)))

        @test minimum(g_none.physical[:, 1, 1]) < 0.0     # the problem exists
        @test minimum(g_k.physical[:, 1, 1])   ≥ 0.0      # RULE 1: last leg suffices
        @test minimum(g_ik.physical[:, 1, 1])  ≥ 0.0      # ... and still does with both

        # RULE 2: with only the last leg bounded, columns of pure ringing are infeasible and
        # the limiter has to create mass. Bounding the earlier leg removes them at the
        # source, where the donor mass is.
        short_k  = sum(bound_shortfall(g_k.kbasis.data[1]) for _ in 1:1)
        short_ik = bound_shortfall(g_ik.kbasis.data[1])
        @test short_k > 0.0
        @test short_ik < short_k
        @test short_ik ≈ 0.0 atol = 1e-12
    end
end

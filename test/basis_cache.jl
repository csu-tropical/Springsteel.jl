@testset "Basis Template Caching" begin

    # ── F2: Parameter equality and hashing ────────────────────────────────

    @testset "SplineParameters equality/hash" begin
        sp1 = CubicBSpline.SplineParameters(xmin=0.0, xmax=1.0, num_cells=10,
            BCL=Dict("α0" => 0.0), BCR=Dict("α0" => 0.0))
        sp2 = CubicBSpline.SplineParameters(xmin=0.0, xmax=1.0, num_cells=10,
            BCL=Dict("α0" => 0.0), BCR=Dict("α0" => 0.0))
        sp3 = CubicBSpline.SplineParameters(xmin=0.0, xmax=2.0, num_cells=10,
            BCL=Dict("α0" => 0.0), BCR=Dict("α0" => 0.0))

        @test sp1 == sp2
        @test hash(sp1) == hash(sp2)
        @test sp1 != sp3
        @test hash(sp1) != hash(sp3)
    end

    @testset "FourierParameters equality/hash" begin
        fp1 = Fourier.FourierParameters(ymin=0.0, kmax=5, yDim=32, bDim=11)
        fp2 = Fourier.FourierParameters(ymin=0.0, kmax=5, yDim=32, bDim=11)
        fp3 = Fourier.FourierParameters(ymin=0.1, kmax=5, yDim=32, bDim=11)

        @test fp1 == fp2
        @test hash(fp1) == hash(fp2)
        @test fp1 != fp3
        @test hash(fp1) != hash(fp3)
    end

    @testset "ChebyshevParameters equality/hash" begin
        cp1 = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=1.0, zDim=16, bDim=16,
            BCB=Dict("α0" => 0.0), BCT=Dict())
        cp2 = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=1.0, zDim=16, bDim=16,
            BCB=Dict("α0" => 0.0), BCT=Dict())
        cp3 = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=1.0, zDim=32, bDim=32,
            BCB=Dict("α0" => 0.0), BCT=Dict())

        @test cp1 == cp2
        @test hash(cp1) == hash(cp2)
        @test cp1 != cp3
        @test hash(cp1) != hash(cp3)
    end

    # ── F3: Spline1D template sharing ─────────────────────────────────────

    @testset "Spline1D template sharing" begin
        clear_basis_caches!()
        @test basis_cache_sizes().spline == 0

        sp = CubicBSpline.SplineParameters(xmin=0.0, xmax=1.0, num_cells=20,
            BCL=CubicBSpline.R0, BCR=CubicBSpline.R0)

        s1 = CubicBSpline.Spline1D(sp)
        @test basis_cache_sizes().spline == 1

        s2 = CubicBSpline.Spline1D(sp)
        @test basis_cache_sizes().spline == 1

        # Template fields shared by reference
        @test s1.pqFactor === s2.pqFactor
        @test s1.gammaBC === s2.gammaBC
        @test s1.mishPoints === s2.mishPoints
        @test s1._sb_matrix === s2._sb_matrix
        @test s1.pq === s2.pq
        @test s1.p1 === s2.p1
        @test s1.p1Factor === s2.p1Factor
        @test s1.quadpoints === s2.quadpoints
        @test s1.quadweights === s2.quadweights

        # Mutable buffers are distinct
        @test s1.b !== s2.b
        @test s1.a !== s2.a
        @test s1.ahat !== s2.ahat
        @test s1.uMish !== s2.uMish
        @test s1._scratch_btilde !== s2._scratch_btilde
        @test s1._scratch_Min !== s2._scratch_Min
        @test s1._scratch_Mout !== s2._scratch_Mout
        @test s1._scratch_bx !== s2._scratch_bx
    end

    @testset "Spline1D isolation" begin
        clear_basis_caches!()
        sp = CubicBSpline.SplineParameters(xmin=0.0, xmax=1.0, num_cells=10,
            BCL=CubicBSpline.R0, BCR=CubicBSpline.R0)
        s1 = CubicBSpline.Spline1D(sp)
        s2 = CubicBSpline.Spline1D(sp)

        # Mutate s1 buffers
        s1.b .= 1.0
        s1.a .= 2.0
        s1.uMish .= 3.0

        # s2 must be unaffected
        @test all(s2.b .== 0.0)
        @test all(s2.a .== 0.0)
        @test all(s2.uMish .== 0.0)

        # Run a forward transform on s1, verify s2 is unchanged
        s1.uMish .= sin.(π .* s1.mishPoints)
        CubicBSpline.SBtransform!(s1)
        CubicBSpline.SAtransform!(s1)
        @test all(s2.a .== 0.0)
        @test all(s2.b .== 0.0)
    end

    @testset "Spline1D cold/warm speedup" begin
        clear_basis_caches!()
        sp = CubicBSpline.SplineParameters(xmin=0.0, xmax=10.0, num_cells=50,
            BCL=CubicBSpline.R0, BCR=CubicBSpline.R0)

        t_cold = @elapsed CubicBSpline.Spline1D(sp)
        t_warm = @elapsed CubicBSpline.Spline1D(sp)
        @test t_warm < t_cold * 0.1
    end

    # ── F4: Fourier1D template sharing ────────────────────────────────────

    @testset "Fourier1D template sharing" begin
        clear_basis_caches!()
        fp = Fourier.FourierParameters(ymin=0.0, kmax=5, yDim=32, bDim=11)

        f1 = Fourier.Fourier1D(fp)
        @test basis_cache_sizes().fourier == 1

        f2 = Fourier.Fourier1D(fp)
        @test basis_cache_sizes().fourier == 1

        # Template fields shared
        @test f1.fftPlan === f2.fftPlan
        @test f1.ifftPlan === f2.ifftPlan
        @test f1.phasefilter === f2.phasefilter
        @test f1.mishPoints === f2.mishPoints

        # Mutable buffers distinct
        @test f1.b !== f2.b
        @test f1.a !== f2.a
        @test f1.uMish !== f2.uMish
        @test f1.ax !== f2.ax
        @test f1._scratch_fft !== f2._scratch_fft
        @test f1._scratch_ax !== f2._scratch_ax
    end

    @testset "Fourier1D isolation" begin
        clear_basis_caches!()
        fp = Fourier.FourierParameters(ymin=0.0, kmax=5, yDim=32, bDim=11)
        f1 = Fourier.Fourier1D(fp)
        f2 = Fourier.Fourier1D(fp)

        f1.b .= 1.0
        f1.uMish .= sin.(f1.mishPoints)
        @test all(f2.b .== 0.0)
        @test all(f2.uMish .== 0.0)
    end

    @testset "Fourier1D cold/warm speedup" begin
        clear_basis_caches!()
        fp = Fourier.FourierParameters(ymin=0.0, kmax=10, yDim=64, bDim=21)
        t_cold = @elapsed Fourier.Fourier1D(fp)
        t_warm = @elapsed Fourier.Fourier1D(fp)
        @test t_warm < t_cold * 0.5
    end

    # ── F5: Chebyshev1D template sharing ──────────────────────────────────

    @testset "Chebyshev1D template sharing" begin
        clear_basis_caches!()
        cp = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=1.0, zDim=16, bDim=16,
            BCB=Chebyshev.R0, BCT=Chebyshev.R0)

        c1 = Chebyshev.Chebyshev1D(cp)
        @test basis_cache_sizes().chebyshev == 1

        c2 = Chebyshev.Chebyshev1D(cp)
        @test basis_cache_sizes().chebyshev == 1

        # Template fields shared
        @test c1.fftPlan === c2.fftPlan
        @test c1.gammaBC === c2.gammaBC
        @test c1.filter === c2.filter
        @test c1.mishPoints === c2.mishPoints

        # Mutable buffers distinct
        @test c1.b !== c2.b
        @test c1.a !== c2.a
        @test c1.uMish !== c2.uMish
        @test c1.ax !== c2.ax
        @test c1._scratch_dct !== c2._scratch_dct
        @test c1._scratch_bfill !== c2._scratch_bfill
        @test c1._scratch_ax !== c2._scratch_ax
    end

    @testset "Chebyshev1D isolation" begin
        clear_basis_caches!()
        cp = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=1.0, zDim=16, bDim=16,
            BCB=Chebyshev.R0, BCT=Chebyshev.R0)
        c1 = Chebyshev.Chebyshev1D(cp)
        c2 = Chebyshev.Chebyshev1D(cp)

        c1.b .= 1.0
        c1.uMish .= sin.(c1.mishPoints)
        @test all(c2.b .== 0.0)
        @test all(c2.uMish .== 0.0)
    end

    @testset "Chebyshev1D cold/warm speedup" begin
        clear_basis_caches!()
        cp = Chebyshev.ChebyshevParameters(zmin=0.0, zmax=1.0, zDim=32, bDim=32,
            BCB=Chebyshev.R0, BCT=Chebyshev.R0)
        t_cold = @elapsed Chebyshev.Chebyshev1D(cp)
        t_warm = @elapsed Chebyshev.Chebyshev1D(cp)
        @test t_warm < t_cold * 0.5
    end

    # ── Cache introspection ───────────────────────────────────────────────

    @testset "clear_basis_caches!" begin
        sp = CubicBSpline.SplineParameters(xmin=0.0, xmax=1.0, num_cells=5)
        CubicBSpline.Spline1D(sp)
        @test basis_cache_sizes().spline >= 1

        clear_basis_caches!()
        @test basis_cache_sizes() == (spline=0, fourier=0, chebyshev=0)
    end

    # ── F6: createGrid cold/warm regression ───────────────────────────────

    @testset "createGrid cold/warm R" begin
        clear_basis_caches!()
        gp = SpringsteelGridParameters(geometry="R", iMin=0.0, iMax=1.0,
            num_cells=20, vars=Dict("u"=>1),
            BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0))
        t1 = @elapsed createGrid(gp)
        t2 = @elapsed createGrid(gp)
        @test t2 < t1 * 0.5
    end

    @testset "createGrid cache hit RR" begin
        clear_basis_caches!()
        gp = SpringsteelGridParameters(geometry="RR", iMin=0.0, iMax=1.0,
            jMin=0.0, jMax=1.0, num_cells=20, vars=Dict("u"=>1),
            BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0),
            BCD=Dict("u"=>CubicBSpline.R0), BCU=Dict("u"=>CubicBSpline.R0),
            max_wavenumber=Dict("default"=>0))
        g1 = createGrid(gp)
        n1 = basis_cache_sizes().spline
        @test n1 >= 1
        g2 = createGrid(gp)
        n2 = basis_cache_sizes().spline
        @test n2 == n1  # no new entries — cache was hit
        # Shared templates, distinct buffers
        @test g1.ibasis.data[1,1].pqFactor === g2.ibasis.data[1,1].pqFactor
        @test g1.ibasis.data[1,1].b !== g2.ibasis.data[1,1].b
    end

    @testset "createGrid cache hit RZ" begin
        clear_basis_caches!()
        gp = SpringsteelGridParameters(geometry="RZ", iMin=0.0, iMax=1.0,
            kMin=0.0, kMax=1.0, num_cells=10, kDim=16, vars=Dict("u"=>1),
            BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0),
            BCB=Dict("u"=>Chebyshev.R0), BCT=Dict("u"=>Chebyshev.R0),
            max_wavenumber=Dict("default"=>0))
        g1 = createGrid(gp)
        n1_s = basis_cache_sizes().spline
        n1_c = basis_cache_sizes().chebyshev
        @test n1_s >= 1
        @test n1_c >= 1
        g2 = createGrid(gp)
        @test basis_cache_sizes().spline == n1_s
        @test basis_cache_sizes().chebyshev == n1_c
        @test g1.ibasis.data[1,1].pqFactor === g2.ibasis.data[1,1].pqFactor
        @test g1.kbasis.data[1].fftPlan === g2.kbasis.data[1].fftPlan
    end

    @testset "createGrid cache hit RL" begin
        clear_basis_caches!()
        gp = SpringsteelGridParameters(geometry="RL", iMin=0.0, iMax=10.0,
            num_cells=5, vars=Dict("u"=>1),
            BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0),
            max_wavenumber=Dict("default"=>-1))
        g1 = createGrid(gp)
        n1_s = basis_cache_sizes().spline
        n1_f = basis_cache_sizes().fourier
        @test n1_s >= 1
        @test n1_f >= 1
        g2 = createGrid(gp)
        @test basis_cache_sizes().spline == n1_s
        @test basis_cache_sizes().fourier == n1_f
        @test g1.ibasis.data[1,1].pqFactor === g2.ibasis.data[1,1].pqFactor
        @test g1.jbasis.data[1,1].fftPlan === g2.jbasis.data[1,1].fftPlan
    end

    @testset "createGrid bitwise parity R" begin
        clear_basis_caches!()
        gp = SpringsteelGridParameters(geometry="R", iMin=0.0, iMax=1.0,
            num_cells=10, vars=Dict("u"=>1),
            BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0))
        g1 = createGrid(gp)
        g1.physical[:, 1, 1] .= sin.(π .* getGridpoints(g1))
        spectralTransform!(g1)
        gridTransform!(g1)
        vals1 = copy(g1.physical[:, 1, 1])

        g2 = createGrid(gp)
        g2.physical[:, 1, 1] .= sin.(π .* getGridpoints(g2))
        spectralTransform!(g2)
        gridTransform!(g2)
        vals2 = copy(g2.physical[:, 1, 1])

        @test vals1 == vals2
    end

    @testset "createGrid bitwise parity RR" begin
        clear_basis_caches!()
        gp = SpringsteelGridParameters(geometry="RR", iMin=0.0, iMax=1.0,
            jMin=0.0, jMax=1.0, num_cells=10, vars=Dict("u"=>1),
            BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0),
            BCD=Dict("u"=>CubicBSpline.R0), BCU=Dict("u"=>CubicBSpline.R0),
            max_wavenumber=Dict("default"=>0))
        pts = getGridpoints(createGrid(gp))

        g1 = createGrid(gp)
        g1.physical[:, 1, 1] .= sin.(π .* pts[:, 1]) .* sin.(π .* pts[:, 2])
        spectralTransform!(g1)
        gridTransform!(g1)
        spec1 = copy(g1.spectral[:, 1])

        g2 = createGrid(gp)
        g2.physical[:, 1, 1] .= sin.(π .* pts[:, 1]) .* sin.(π .* pts[:, 2])
        spectralTransform!(g2)
        gridTransform!(g2)
        spec2 = copy(g2.spectral[:, 1])

        @test spec1 == spec2
    end
end

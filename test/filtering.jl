using LinearAlgebra: norm

@testset "Filtering" begin

    # ════════════════════════════════════════════════════════════════════════
    # Filter types and weight functions
    # ════════════════════════════════════════════════════════════════════════

    @testset "SpectralFilter weights" begin
        # Boxcar low-pass
        f = SpectralFilter(low_pass=10)
        @test Springsteel._filter_weight(f, 0) ≈ 1.0
        @test Springsteel._filter_weight(f, 5) ≈ 1.0
        @test Springsteel._filter_weight(f, 10) ≈ 1.0
        @test Springsteel._filter_weight(f, 11) ≈ 0.0
        @test Springsteel._filter_weight(f, 100) ≈ 0.0

        # Boxcar high-pass
        f = SpectralFilter(high_pass=5)
        @test Springsteel._filter_weight(f, 0) ≈ 0.0
        @test Springsteel._filter_weight(f, 4) ≈ 0.0
        @test Springsteel._filter_weight(f, 5) ≈ 1.0
        @test Springsteel._filter_weight(f, 10) ≈ 1.0

        # Band-pass
        f = SpectralFilter(low_pass=20, high_pass=5)
        @test Springsteel._filter_weight(f, 0) ≈ 0.0
        @test Springsteel._filter_weight(f, 4) ≈ 0.0
        @test Springsteel._filter_weight(f, 5) ≈ 1.0
        @test Springsteel._filter_weight(f, 15) ≈ 1.0
        @test Springsteel._filter_weight(f, 20) ≈ 1.0
        @test Springsteel._filter_weight(f, 21) ≈ 0.0

        # Notch filter
        f = SpectralFilter(notch=[0, 3, 7])
        @test Springsteel._filter_weight(f, 0) ≈ 0.0
        @test Springsteel._filter_weight(f, 1) ≈ 1.0
        @test Springsteel._filter_weight(f, 3) ≈ 0.0
        @test Springsteel._filter_weight(f, 5) ≈ 1.0
        @test Springsteel._filter_weight(f, 7) ≈ 0.0

        # Low-pass + notch
        f = SpectralFilter(low_pass=10, notch=[1])
        @test Springsteel._filter_weight(f, 0) ≈ 1.0
        @test Springsteel._filter_weight(f, 1) ≈ 0.0
        @test Springsteel._filter_weight(f, 5) ≈ 1.0
        @test Springsteel._filter_weight(f, 11) ≈ 0.0

        # No filter (defaults)
        f = SpectralFilter()
        for k in 0:50
            @test Springsteel._filter_weight(f, k) ≈ 1.0
        end
    end

    @testset "SpectralFilter windowed taper" begin
        # Low-pass with Hann taper
        f = SpectralFilter(low_pass=10, window=:hann, taper_width=5)
        @test Springsteel._filter_weight(f, 8) ≈ 1.0     # inside passband
        @test Springsteel._filter_weight(f, 10) ≈ 1.0    # at cutoff edge
        w11 = Springsteel._filter_weight(f, 11)
        @test 0.0 < w11 < 1.0                             # in taper region
        @test Springsteel._filter_weight(f, 16) ≈ 0.0    # beyond taper

        # Verify taper is monotonically decreasing
        for k in 11:14
            @test Springsteel._filter_weight(f, k) >= Springsteel._filter_weight(f, k+1)
        end

        # Lanczos taper
        f = SpectralFilter(low_pass=10, window=:lanczos, taper_width=5)
        @test Springsteel._filter_weight(f, 10) ≈ 1.0
        w12 = Springsteel._filter_weight(f, 12)
        @test 0.0 < w12 < 1.0

        # Exponential taper
        f = SpectralFilter(low_pass=10, window=:exponential, taper_width=5)
        @test Springsteel._filter_weight(f, 10) ≈ 1.0
        w13 = Springsteel._filter_weight(f, 13)
        @test 0.0 < w13 < 1.0

        # High-pass with taper
        f = SpectralFilter(high_pass=10, window=:hann, taper_width=3)
        @test Springsteel._filter_weight(f, 7) ≈ 0.0     # well below cutoff
        w8 = Springsteel._filter_weight(f, 8)
        @test 0.0 < w8 < 1.0                              # in taper region
        @test Springsteel._filter_weight(f, 10) ≈ 1.0    # at cutoff edge
        @test Springsteel._filter_weight(f, 15) ≈ 1.0    # well above cutoff

        # Boxcar with taper_width=0 should be equivalent to boxcar without
        f1 = SpectralFilter(low_pass=10, window=:hann, taper_width=0)
        f2 = SpectralFilter(low_pass=10, window=:boxcar)
        for k in 0:20
            @test Springsteel._filter_weight(f1, k) ≈ Springsteel._filter_weight(f2, k)
        end
    end

    @testset "GaussianFilter weights" begin
        # Standard Gaussian
        f = GaussianFilter(sigma=10.0)
        @test Springsteel._filter_weight(f, 0) ≈ 1.0
        @test Springsteel._filter_weight(f, 10) ≈ exp(-1.0)
        @test Springsteel._filter_weight(f, 20) ≈ exp(-4.0)

        # Higher order (sharper)
        f = GaussianFilter(sigma=10.0, order=2)
        @test Springsteel._filter_weight(f, 0) ≈ 1.0
        @test Springsteel._filter_weight(f, 10) ≈ exp(-1.0)
        @test Springsteel._filter_weight(f, 5) > exp(-1.0)  # less attenuation than k=σ

        # Very wide Gaussian (effectively no filter)
        f = GaussianFilter(sigma=1000.0)
        for k in 0:50
            @test Springsteel._filter_weight(f, k) > 0.99
        end
    end

    @testset "Window functions" begin
        # All windows should return ~1 at t=0 and ~0 at t=1
        for w in (:boxcar, :hann, :lanczos, :exponential)
            @test Springsteel._window_weight(w, 0.0) ≈ 1.0 atol=0.02
        end
        @test Springsteel._window_weight(:hann, 1.0) ≈ 0.0
        @test Springsteel._window_weight(:boxcar, 1.0) ≈ 0.0
        @test Springsteel._window_weight(:exponential, 1.0) < 0.02

        # Invalid window
        @test_throws ArgumentError Springsteel._window_weight(:invalid, 0.5)
    end

    # ════════════════════════════════════════════════════════════════════════
    # Grid-level filtering tests
    # ════════════════════════════════════════════════════════════════════════

    @testset "R grid — no-op" begin
        gp = SpringsteelGridParameters(
            geometry = "R", num_cells = 10,
            iMin = 0.0, iMax = 100.0,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0),
            fourier_filter = Dict("u" => SpectralFilter(low_pass=5)))
        grid = createGrid(gp)
        # Fill spectral with ones
        grid.spectral .= 1.0
        spec_before = copy(grid.spectral)
        applyFilter!(grid)
        @test grid.spectral ≈ spec_before  # no change for pure spline grid
    end

    @testset "RL grid — Fourier filtering" begin
        gp = SpringsteelGridParameters(
            geometry = "RL", num_cells = 10,
            iMin = 0.0, iMax = 100.0,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0),
            fourier_filter = Dict("u" => SpectralFilter(low_pass=3, notch=[1])))
        grid = createGrid(gp)
        b_iDim = gp.b_iDim
        kDim = grid.params.iDim + grid.params.patchOffsetL

        # Fill spectral with ones
        grid.spectral .= 1.0
        applyFilter!(grid)

        # k=0 should be kept
        @test all(grid.spectral[1:b_iDim, 1] .≈ 1.0)

        # k=1 should be zeroed (notch)
        r1_real = (2*1 - 1) * b_iDim + 1
        r2_imag = (2*1 + 1) * b_iDim
        @test all(grid.spectral[r1_real:r2_imag, 1] .≈ 0.0)

        # k=2, k=3 should be kept
        for k in 2:3
            r1 = (2*k - 1) * b_iDim + 1
            r2 = (2*k + 1) * b_iDim
            @test all(grid.spectral[r1:r2, 1] .≈ 1.0)
        end

        # k > 3 should be zeroed (low-pass)
        for k in 4:kDim
            r1 = (2*k - 1) * b_iDim + 1
            r2 = (2*k + 1) * b_iDim
            @test all(grid.spectral[r1:r2, 1] .≈ 0.0)
        end
    end

    @testset "RL grid — round-trip with low-pass" begin
        gp_nofilt = SpringsteelGridParameters(
            geometry = "RL", num_cells = 10,
            iMin = 0.0, iMax = 100.0,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0))
        gp_filt = SpringsteelGridParameters(
            geometry = "RL", num_cells = 10,
            iMin = 0.0, iMax = 100.0,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0),
            fourier_filter = Dict("u" => SpectralFilter(low_pass=3)))

        grid_nf = createGrid(gp_nofilt)
        grid_f  = createGrid(gp_filt)

        # Fill with same random data
        data = randn(size(grid_nf.physical, 1))
        grid_nf.physical[:, 1, 1] .= data
        grid_f.physical[:, 1, 1] .= data

        spectralTransform!(grid_nf)
        spectralTransform!(grid_f)   # auto-applies filter

        gridTransform!(grid_nf)
        gridTransform!(grid_f)

        # Filtered should have smaller amplitude at small scales
        @test norm(grid_f.physical[:, 1, 1]) <= norm(grid_nf.physical[:, 1, 1])
        # But they shouldn't be identical (filter should have changed something)
        @test !(grid_f.physical[:, 1, 1] ≈ grid_nf.physical[:, 1, 1])
    end

    @testset "RZ grid — Chebyshev filtering" begin
        gp = SpringsteelGridParameters(
            geometry = "RZ", num_cells = 10,
            iMin = 0.0, iMax = 100.0,
            kMin = 0.0, kMax = 10.0, kDim = 12,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0),
            BCB = Dict("u" => Chebyshev.R0),
            BCT = Dict("u" => Chebyshev.R0),
            chebyshev_filter = Dict("u" => SpectralFilter(low_pass=3)))
        grid = createGrid(gp)
        b_iDim = grid.params.b_iDim
        b_kDim = grid.params.b_kDim

        # Fill spectral with ones
        grid.spectral .= 1.0
        applyFilter!(grid)

        # Chebyshev modes 0-3 (z_b 1-4) should be kept
        for z in 1:4
            r1 = (z - 1) * b_iDim + 1
            r2 = z * b_iDim
            @test all(grid.spectral[r1:r2, 1] .≈ 1.0)
        end

        # Chebyshev modes > 3 (z_b > 4) should be zeroed
        for z in 5:b_kDim
            r1 = (z - 1) * b_iDim + 1
            r2 = z * b_iDim
            @test all(grid.spectral[r1:r2, 1] .≈ 0.0)
        end
    end

    @testset "RLZ grid — Fourier + Chebyshev filtering" begin
        gp = SpringsteelGridParameters(
            geometry = "RLZ", num_cells = 5,
            iMin = 0.0, iMax = 50.0,
            kMin = 0.0, kMax = 10.0, kDim = 6,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0),
            BCB = Dict("u" => Chebyshev.R0),
            BCT = Dict("u" => Chebyshev.R0),
            fourier_filter = Dict("u" => SpectralFilter(low_pass=2)),
            chebyshev_filter = Dict("u" => SpectralFilter(low_pass=2)))
        grid = createGrid(gp)
        b_iDim = grid.params.b_iDim
        b_kDim = grid.params.b_kDim
        kDim_wn = grid.params.iDim + grid.params.patchOffsetL
        block_size = b_iDim * (1 + kDim_wn * 2)

        # Fill spectral with ones
        grid.spectral .= 1.0
        applyFilter!(grid)

        # Check z_b=1 (Chebyshev mode 0), k=0 (Fourier mode 0) — should be kept
        @test all(grid.spectral[1:b_iDim, 1] .≈ 1.0)

        # Check z_b=1, k=3 (Fourier mode 3) — should be zeroed (low_pass=2)
        r1_k3_real = b_iDim + (3 - 1) * 2 * b_iDim + 1
        r2_k3_real = r1_k3_real + b_iDim - 1
        @test all(grid.spectral[r1_k3_real:r2_k3_real, 1] .≈ 0.0)

        # Check z_b=4 (Chebyshev mode 3), k=0 — should be zeroed (cheb low_pass=2)
        block_start_z4 = 3 * block_size
        @test all(grid.spectral[block_start_z4+1:block_start_z4+b_iDim, 1] .≈ 0.0)

        # Check z_b=1 (Chebyshev mode 0), k=1 — should be kept (both filters pass)
        r1_k1_real = b_iDim + 1
        r2_k1_real = r1_k1_real + b_iDim - 1
        @test all(grid.spectral[r1_k1_real:r2_k1_real, 1] .≈ 1.0)
    end

    @testset "GaussianFilter on RL grid" begin
        gp = SpringsteelGridParameters(
            geometry = "RL", num_cells = 10,
            iMin = 0.0, iMax = 100.0,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0),
            fourier_filter = Dict("u" => GaussianFilter(sigma=5.0)))
        grid = createGrid(gp)
        b_iDim = gp.b_iDim

        # Fill spectral with ones
        grid.spectral .= 1.0
        applyFilter!(grid)

        # k=0 should be exactly 1 (Gaussian at k=0 is exp(0) = 1)
        @test all(grid.spectral[1:b_iDim, 1] .≈ 1.0)

        # k=5 should be exp(-1) ≈ 0.368
        w5 = exp(-1.0)
        r1 = (2*5 - 1) * b_iDim + 1
        r2 = (2*5 + 1) * b_iDim
        @test all(grid.spectral[r1:r2, 1] .≈ w5)

        # k=10 should be exp(-4) ≈ 0.018
        w10 = exp(-4.0)
        r1 = (2*10 - 1) * b_iDim + 1
        r2 = (2*10 + 1) * b_iDim
        @test all(grid.spectral[r1:r2, 1] .≈ w10)
    end

    @testset "SL grid — Fourier filtering" begin
        gp = SpringsteelGridParameters(
            geometry = "SL", num_cells = 10,
            iMin = 0.0, iMax = Float64(π),
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0),
            fourier_filter = Dict("u" => SpectralFilter(low_pass=3)))
        grid = createGrid(gp)
        b_iDim = gp.b_iDim
        kDim = grid.params.iDim + grid.params.patchOffsetL

        grid.spectral .= 1.0
        applyFilter!(grid)

        # k=0 kept
        @test all(grid.spectral[1:b_iDim, 1] .≈ 1.0)

        # k > 3 zeroed
        for k in 4:kDim
            r1 = (2*k - 1) * b_iDim + 1
            r2 = (2*k + 1) * b_iDim
            @test all(grid.spectral[r1:r2, 1] .≈ 0.0)
        end
    end

    @testset "Multi-variable filtering" begin
        gp = SpringsteelGridParameters(
            geometry = "RL", num_cells = 10,
            iMin = 0.0, iMax = 100.0,
            vars = Dict("u" => 1, "v" => 2),
            BCL = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
            fourier_filter = Dict("u" => SpectralFilter(low_pass=3)))
        grid = createGrid(gp)
        b_iDim = gp.b_iDim
        kDim = grid.params.iDim + grid.params.patchOffsetL

        grid.spectral .= 1.0
        applyFilter!(grid)

        # "u" (var 1) should be filtered — k=5 zeroed
        k = 5
        r1 = (2*k - 1) * b_iDim + 1
        r2 = (2*k + 1) * b_iDim
        @test all(grid.spectral[r1:r2, 1] .≈ 0.0)

        # "v" (var 2) should NOT be filtered — k=5 still 1.0
        @test all(grid.spectral[r1:r2, 2] .≈ 1.0)
    end

    @testset "Default filter key" begin
        gp = SpringsteelGridParameters(
            geometry = "RL", num_cells = 10,
            iMin = 0.0, iMax = 100.0,
            vars = Dict("u" => 1, "v" => 2),
            BCL = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
            fourier_filter = Dict("default" => SpectralFilter(low_pass=3)))
        grid = createGrid(gp)
        b_iDim = gp.b_iDim
        kDim = grid.params.iDim + grid.params.patchOffsetL

        grid.spectral .= 1.0
        applyFilter!(grid)

        # Both variables should be filtered
        k = 5
        r1 = (2*k - 1) * b_iDim + 1
        r2 = (2*k + 1) * b_iDim
        @test all(grid.spectral[r1:r2, 1] .≈ 0.0)
        @test all(grid.spectral[r1:r2, 2] .≈ 0.0)
    end

    @testset "Empty filter — no change" begin
        gp = SpringsteelGridParameters(
            geometry = "RL", num_cells = 10,
            iMin = 0.0, iMax = 100.0,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0))
        grid = createGrid(gp)
        grid.spectral .= 1.0
        spec_before = copy(grid.spectral)
        applyFilter!(grid)
        @test grid.spectral ≈ spec_before
    end

    @testset "Filter integrated into spectralTransform!" begin
        # Verify that spectralTransform! auto-applies the filter
        gp = SpringsteelGridParameters(
            geometry = "RL", num_cells = 10,
            iMin = 0.0, iMax = 100.0,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0),
            fourier_filter = Dict("u" => SpectralFilter(low_pass=3)))
        grid = createGrid(gp)

        # Set physical data
        grid.physical[:, 1, 1] .= randn(size(grid.physical, 1))
        spectralTransform!(grid)

        # After spectralTransform!, high wavenumbers should be zero
        b_iDim = gp.b_iDim
        kDim = grid.params.iDim + grid.params.patchOffsetL
        for k in 4:kDim
            r1 = (2*k - 1) * b_iDim + 1
            r2 = (2*k + 1) * b_iDim
            @test all(grid.spectral[r1:r2, 1] .≈ 0.0)
        end
    end

    @testset "RRR grid — no-op" begin
        gp = SpringsteelGridParameters(
            geometry = "RRR", num_cells = 5,
            iMin = 0.0, iMax = 50.0,
            jMin = 0.0, jMax = 50.0,
            kMin = 0.0, kMax = 50.0,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0),
            BCU = Dict("u" => CubicBSpline.R0),
            BCD = Dict("u" => CubicBSpline.R0),
            BCB = Dict("u" => CubicBSpline.R0),
            BCT = Dict("u" => CubicBSpline.R0),
            fourier_filter = Dict("u" => SpectralFilter(low_pass=3)))
        grid = createGrid(gp)
        grid.spectral .= 1.0
        spec_before = copy(grid.spectral)
        applyFilter!(grid)
        @test grid.spectral ≈ spec_before  # no Fourier/Chebyshev dimensions
    end

end

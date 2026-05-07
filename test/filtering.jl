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

    # ════════════════════════════════════════════════════════════════════════
    # Spline-direction physical-space filtering
    # ════════════════════════════════════════════════════════════════════════

    @testset "Gaussian kernel correctness" begin
        # Symmetry and peak value
        K = Springsteel._gaussian_kernel(2.0, 1.0)   # σ_cells=2, dx=1 ⇒ σ=2
        σ = 2.0
        @test K(0.0) ≈ 1.0 / (σ * sqrt(2π))
        @test K(1.0) ≈ K(-1.0)
        @test K(3.0) > 0.0
        # Tail check at 4σ ≈ 1.3e-4 of peak
        @test K(4 * σ) / K(0.0) < 1e-3
    end

    @testset "Lanczos kernel correctness" begin
        a = 3
        dx = 1.0
        K = Springsteel._lanczos_kernel(a, dx)
        @test K(0.0) ≈ 1.0
        @test K(a * dx) ≈ 0.0 atol=1e-12
        @test K(-a * dx) ≈ 0.0 atol=1e-12
        # Outside the support strictly zero
        @test K(a * dx + 0.5) == 0.0
        @test K(-a * dx - 0.5) == 0.0
        # Symmetry
        for x in (0.3, 0.7, 1.5, 2.4)
            @test K(x) ≈ K(-x)
        end
        # Sinc zero crossings at non-zero integer multiples of dx
        @test abs(K(1.0)) < 1e-12
        @test abs(K(2.0)) < 1e-12
    end

    @testset "Convolution preserves uniform field" begin
        # Zero-extend + renormalise should preserve a constant input exactly
        coords = collect(range(0.0, 10.0, length=21))
        src = fill(2.5, length(coords))
        dst = similar(src)
        K   = Springsteel._gaussian_kernel(1.5, 0.5)
        Springsteel._convolve_axis!(dst, src, coords, K, 4 * 1.5 * 0.5)
        @test dst ≈ src atol=1e-12
    end

    @testset "R grid spline_filter — BC preservation" begin
        gp = SpringsteelGridParameters(
            geometry = "R", num_cells = 16,
            iMin = 0.0, iMax = 1.0,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R1T0),
            BCR = Dict("u" => CubicBSpline.R1T0),
            spline_filter = Dict("u" => Dict(:i => GaussianFilter(sigma=1.0))))
        grid = createGrid(gp)

        pts = getGridpoints(grid)
        for i in eachindex(pts)
            grid.physical[i, 1, 1] = sin(2π * pts[i]) + 0.3 * cos(6π * pts[i])
        end

        spectralTransform!(grid)
        # Reconstruct on a regular grid that includes both boundary points;
        # γ-folding + ahat should pin u(xmin)=u(xmax)=0 regardless of filtering.
        reg_pts = getRegularGridpoints(grid)
        reg_phys = regularGridTransform(grid, reg_pts)
        @test abs(reg_phys[1, 1, 1])   < 1e-10
        @test abs(reg_phys[end, 1, 1]) < 1e-10
    end

    @testset "RZ grid spline_filter — BC preservation (i only)" begin
        gp = SpringsteelGridParameters(
            geometry = "RZ", num_cells = 12,
            iMin = 0.0, iMax = 1.0,
            kMin = 0.0, kMax = 1.0, kDim = 12,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R1T0),
            BCR = Dict("u" => CubicBSpline.R1T0),
            BCB = Dict("u" => Chebyshev.R0),
            BCT = Dict("u" => Chebyshev.R0),
            spline_filter = Dict("u" => Dict(:i => GaussianFilter(sigma=1.0))))
        grid = createGrid(gp)

        pts = getGridpoints(grid)
        @inbounds for i in 1:size(pts, 1)
            x, z = pts[i, 1], pts[i, 2]
            grid.physical[i, 1, 1] = sin(2π*x) * cos(π*z) + 0.05*sin(11π*x)
        end

        spectralTransform!(grid)

        # Evaluate spline a-coefficients at xmin/xmax for each Chebyshev mode;
        # should be 0 to within 1e-10 because R1T0 zeroes those basis fns.
        b_iDim = grid.params.b_iDim
        b_kDim = grid.params.b_kDim
        max_bdry = 0.0
        for z in 1:b_kDim
            isp = grid.ibasis.data[z, 1]
            r1 = (z - 1) * b_iDim + 1
            r2 = z * b_iDim
            isp.b .= view(grid.spectral, r1:r2, 1)
            SAtransform!(isp)
            out = zeros(2)
            SItransform(isp, [gp.iMin, gp.iMax], out)
            max_bdry = max(max_bdry, maximum(abs, out))
        end
        @test max_bdry < 1e-9
    end

    @testset "RL grid spline_filter — BC preservation (i only)" begin
        gp = SpringsteelGridParameters(
            geometry = "RL", num_cells = 8,
            iMin = 0.0, iMax = 100.0,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R1T0),
            spline_filter = Dict("u" => Dict(:i => GaussianFilter(sigma=1.0))))
        grid = createGrid(gp)

        pts = getGridpoints(grid)
        @inbounds for i in 1:size(pts, 1)
            r, λ = pts[i, 1], pts[i, 2]
            grid.physical[i, 1, 1] = (r / gp.iMax) * cos(λ) + 0.05 * sin(7λ)
        end

        spectralTransform!(grid)

        # Evaluate the radial spline for k=0 at iMax; with R1T0 (Dirichlet)
        # at the outer edge the value must be 0 regardless of filtering.
        b_iDim = grid.params.b_iDim
        isp0 = grid.ibasis.data[1, 1]
        isp0.b .= view(grid.spectral, 1:b_iDim, 1)
        SAtransform!(isp0)
        out = zeros(1)
        SItransform(isp0, [gp.iMax], out)
        @test abs(out[1]) < 1e-9
    end

    @testset "Spline filter actually smooths" begin
        # White-noise + smooth signal; filtered version should have smaller
        # high-wavenumber energy and similar low-wavenumber energy.
        gp_nf = SpringsteelGridParameters(
            geometry = "R", num_cells = 32,
            iMin = 0.0, iMax = 1.0,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0))
        gp_f  = SpringsteelGridParameters(
            geometry = "R", num_cells = 32,
            iMin = 0.0, iMax = 1.0,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0),
            spline_filter = Dict("u" => Dict(:i => GaussianFilter(sigma=2.0))))

        grid_nf = createGrid(gp_nf)
        grid_f  = createGrid(gp_f)

        pts = getGridpoints(grid_nf)
        rng_data = sin.(4π .* pts) .+ 0.5 .* sin.(40π .* pts)
        grid_nf.physical[:, 1, 1] .= rng_data
        grid_f.physical[:, 1, 1]  .= rng_data

        spectralTransform!(grid_nf)
        spectralTransform!(grid_f)

        b_iDim = grid_nf.params.b_iDim
        # Compare last 30% of coefficients — high modes — and first 30% — low modes
        hi_lo = Int(floor(0.7 * b_iDim))
        lo_hi = Int(ceil(0.3 * b_iDim))
        hi_nf = norm(grid_nf.spectral[hi_lo:b_iDim, 1])
        hi_f  = norm(grid_f.spectral[hi_lo:b_iDim, 1])
        lo_nf = norm(grid_nf.spectral[1:lo_hi, 1])
        lo_f  = norm(grid_f.spectral[1:lo_hi, 1])
        @test hi_f < hi_nf                          # high modes attenuated
        @test isapprox(lo_f, lo_nf; rtol=0.25)      # low modes ~unchanged
    end

    @testset "SpectralFilter on spline direction errors" begin
        @test_throws ArgumentError SpringsteelGridParameters(
            geometry = "R", num_cells = 8,
            iMin = 0.0, iMax = 1.0,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0),
            spline_filter = Dict("u" => Dict(:i => SpectralFilter(low_pass=3)))) |>
            createGrid
    end

    @testset "spline_filter validation — bad direction" begin
        # Direction :k on a 1D R grid is not a spline direction
        @test_throws ArgumentError SpringsteelGridParameters(
            geometry = "R", num_cells = 8,
            iMin = 0.0, iMax = 1.0,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0),
            spline_filter = Dict("u" => Dict(:k => GaussianFilter(sigma=1.0)))) |>
            createGrid
    end

    @testset "spline_filter :default direction fallback" begin
        # `:default` applies to every spline direction the geometry has
        gp = SpringsteelGridParameters(
            geometry = "RR", num_cells = 8,
            iMin = 0.0, iMax = 1.0,
            jMin = 0.0, jMax = 1.0,
            vars = Dict("u" => 1, "v" => 2),
            BCL = Dict("default" => CubicBSpline.R0),
            BCR = Dict("default" => CubicBSpline.R0),
            BCU = Dict("default" => CubicBSpline.R0),
            BCD = Dict("default" => CubicBSpline.R0),
            spline_filter = Dict(
                "v" => Dict(:default => GaussianFilter(sigma=1.0))))
        grid = createGrid(gp)
        @test Springsteel._resolve_spline_filter(grid.params.spline_filter, "v", :i) isa GaussianFilter
        @test Springsteel._resolve_spline_filter(grid.params.spline_filter, "v", :j) isa GaussianFilter
        @test Springsteel._resolve_spline_filter(grid.params.spline_filter, "u", :i) === nothing
    end

    @testset "LanczosFilter spectral path matches SpectralFilter" begin
        # On Fourier path, LanczosFilter(a=a, low_pass=k) ≡
        # SpectralFilter(window=:lanczos, low_pass=k, taper_width=a).
        f_lan = LanczosFilter(a=3, low_pass=10)
        f_spc = SpectralFilter(low_pass=10, window=:lanczos, taper_width=3)
        for k in 0:30
            @test Springsteel._filter_weight(f_lan, k) ≈ Springsteel._filter_weight(f_spc, k)
        end
        # low_pass=0 on the spectral path is an error
        @test_throws ArgumentError Springsteel._filter_weight(LanczosFilter(a=3, low_pass=0), 5)
    end

    @testset "Empty spline_filter — no-op" begin
        gp = SpringsteelGridParameters(
            geometry = "R", num_cells = 8,
            iMin = 0.0, iMax = 1.0,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0))
        grid = createGrid(gp)
        pts = getGridpoints(grid)
        grid.physical[:, 1, 1] .= sin.(2π .* pts)
        before = copy(grid.physical)
        Springsteel._filter_mish!(grid)
        @test grid.physical ≈ before
    end

end

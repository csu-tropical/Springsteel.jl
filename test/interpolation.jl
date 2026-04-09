using Test
using Springsteel
using Springsteel.CubicBSpline
using Springsteel.Fourier
using Springsteel.Chebyshev

@testset "Interpolation" begin

    # ════════════════════════════════════════════════════════════════════════
    # Basis evaluation matrices at arbitrary points
    # ════════════════════════════════════════════════════════════════════════

    @testset "SItransform_matrix arbitrary points" begin
        sp = SplineParameters(xmin=0.0, xmax=5.0, num_cells=5,
            mubar=3, quadrature=:gauss, l_q=2.0,
            BCL=CubicBSpline.R0, BCR=CubicBSpline.R0)
        spline = Spline1D(sp)

        # Set some test coefficients
        for i in 1:sp.bDim
            spline.a[i] = sin(i * 0.5)
        end

        # Test with mishDim points (existing usage — backward compatible)
        pts_mish = spline.mishPoints
        M_mish = CubicBSpline.SItransform_matrix(spline, pts_mish, 0)
        @test size(M_mish) == (sp.mishDim, sp.bDim)

        # Test with non-mishDim-length points (this was broken before the fix)
        pts_arb = collect(LinRange(0.5, 4.5, 20))
        M_arb = CubicBSpline.SItransform_matrix(spline, pts_arb, 0)
        @test size(M_arb) == (20, sp.bDim)

        # Verify matrix evaluation matches function evaluation
        u_func = SItransform(sp, spline.a, pts_arb, 0)
        u_mat = M_arb * spline.a
        @test u_func ≈ u_mat atol=1e-12
    end

    @testset "FItransform_matrix" begin
        fp = FourierParameters(ymin=0.0, kmax=5, yDim=20, bDim=11)
        ring = Fourier1D(fp)

        # Set some test coefficients
        for i in 1:fp.bDim
            ring.b[i] = cos(i * 0.3)
        end

        # Evaluate at ring's own mish points → should match dft_matrix
        M_mish = Fourier.FItransform_matrix(ring, ring.mishPoints, 0)
        M_dft = Fourier.dft_matrix(ring)
        @test size(M_mish) == (fp.yDim, fp.bDim)
        @test M_mish ≈ M_dft atol=1e-12

        # Evaluate at arbitrary points
        pts = collect(LinRange(0.0, 2π - 0.01, 30))
        M = Fourier.FItransform_matrix(ring, pts, 0)
        @test size(M) == (30, fp.bDim)

        # Check derivative consistency: M1 * b should be the derivative of M0 * b
        M0 = Fourier.FItransform_matrix(ring, pts, 0)
        M1 = Fourier.FItransform_matrix(ring, pts, 1)
        M1_dft = Fourier.dft_1st_derivative(ring)
        M1_mish = Fourier.FItransform_matrix(ring, ring.mishPoints, 1)
        @test M1_mish ≈ M1_dft atol=1e-12

        # Second derivative
        M2 = Fourier.FItransform_matrix(ring, pts, 2)
        M2_dft = Fourier.dft_2nd_derivative(ring)
        M2_mish = Fourier.FItransform_matrix(ring, ring.mishPoints, 2)
        @test M2_mish ≈ M2_dft atol=1e-12
    end

    @testset "CItransform_matrix" begin
        cp = ChebyshevParameters(zmin=0.0, zmax=10.0, zDim=15, bDim=15,
            BCB=Chebyshev.R0, BCT=Chebyshev.R0)
        col = Chebyshev1D(cp)

        # Set some test coefficients
        for i in 1:cp.zDim
            col.a[i] = exp(-0.1 * i)
        end

        # Evaluate at CGL points → should match dct_matrix
        cgl_pts = col.mishPoints
        M_cgl = Chebyshev.CItransform_matrix(col, cgl_pts, 0)
        M_dct = Chebyshev.dct_matrix(cp.zDim)
        @test size(M_cgl) == (cp.zDim, cp.zDim)
        @test M_cgl ≈ M_dct atol=1e-10

        # Evaluate at arbitrary points within domain
        pts = collect(LinRange(0.5, 9.5, 25))
        M = Chebyshev.CItransform_matrix(col, pts, 0)
        @test size(M) == (25, cp.zDim)

        # Verify 0th derivative matches _cheb_eval_pts!
        col.b .= col.a[1:cp.bDim]  # For R0 BCs, CAtransform is identity-like
        Chebyshev.CAtransform!(col)
        ref = zeros(Float64, length(pts))
        Springsteel._cheb_eval_pts!(col, pts, ref)
        result = M * col.a
        @test result ≈ ref atol=1e-10
    end

    # ════════════════════════════════════════════════════════════════════════
    # Layer 1: grid_from_regular_data
    # ════════════════════════════════════════════════════════════════════════

    @testset "grid_from_regular_data 1D" begin
        # Create regular data: f(x) = sin(2πx/L)
        N = 15
        L = 5.0
        h = L / N
        x = [(i - 0.5) * h for i in 1:N]  # midpoint-rule positions for mubar=1, 15 cells
        f = sin.(2π .* x ./ L)
        data = reshape(f, :, 1)

        grid = grid_from_regular_data(x, data; mubar=1, vars=["u"])
        @test grid isa R_Grid
        @test grid.params.num_cells == 15
        @test grid.params.mubar == 1
        @test grid.params.quadrature == :regular
        @test grid.params.iMin ≈ 0.0 atol=1e-12
        @test grid.params.iMax ≈ L atol=1e-12
        @test grid.physical[:, 1, 1] ≈ f atol=1e-12

        # Derivative slots should be NaN
        @test all(isnan, grid.physical[:, 1, 2])
        @test all(isnan, grid.physical[:, 1, 3])

        # Test with mubar=3
        N3 = 12
        h3 = L / N3
        x3 = [(i - 0.5) * h3 for i in 1:N3]
        f3 = sin.(2π .* x3 ./ L)
        data3 = reshape(f3, :, 1)

        grid3 = grid_from_regular_data(x3, data3; mubar=3, vars=["u"])
        @test grid3.params.num_cells == 4
        @test grid3.params.mubar == 3
        @test grid3.params.iDim == 12

        # Test error for non-divisible mubar
        @test_throws ArgumentError grid_from_regular_data(x, data; mubar=4, vars=["u"])

        # Test error for non-uniform spacing
        x_bad = [1.0, 2.0, 3.5]
        @test_throws ArgumentError grid_from_regular_data(x_bad, ones(3, 1); mubar=1)
    end

    @testset "grid_from_regular_data 2D" begin
        Nx, Ny = 6, 9
        Lx, Ly = 4.0, 6.0
        hx, hy = Lx / Nx, Ly / Ny
        x = [(i - 0.5) * hx for i in 1:Nx]
        y = [(j - 0.5) * hy for j in 1:Ny]

        # f(x,y) = sin(x) * cos(y) — data in i-outer, j-inner order
        data = zeros(Nx * Ny, 1)
        for i in 1:Nx
            for j in 1:Ny
                flat = (i - 1) * Ny + j
                data[flat, 1] = sin(x[i]) * cos(y[j])
            end
        end

        grid = grid_from_regular_data(x, y, data; mubar=3, vars=["u"])
        @test grid isa RR_Grid
        @test grid.params.iDim == 6
        @test grid.params.jDim == 9
        @test grid.params.num_cells == 2  # 6/3
        @test grid.params.iMin ≈ 0.0 atol=1e-12
        @test grid.params.iMax ≈ Lx atol=1e-12
        @test grid.params.jMin ≈ 0.0 atol=1e-12
        @test grid.params.jMax ≈ Ly atol=1e-12
        @test grid.physical[:, 1, 1] ≈ data[:, 1] atol=1e-12
        @test all(isnan, grid.physical[:, 1, 2])
    end

    @testset "grid_from_regular_data 3D" begin
        Nx, Ny, Nz = 3, 3, 3
        Lx, Ly, Lz = 3.0, 3.0, 3.0
        hx, hy, hz = Lx / Nx, Ly / Ny, Lz / Nz
        x = [(i - 0.5) * hx for i in 1:Nx]
        y = [(j - 0.5) * hy for j in 1:Ny]
        z = [(k - 0.5) * hz for k in 1:Nz]

        data = zeros(Nx * Ny * Nz, 1)
        idx = 1
        for i in 1:Nx
            for j in 1:Ny
                for k in 1:Nz
                    data[idx, 1] = x[i] + y[j] + z[k]
                    idx += 1
                end
            end
        end

        grid = grid_from_regular_data(x, y, z, data; mubar=1, vars=["u"])
        @test grid isa RRR_Grid
        @test grid.params.iDim == 3
        @test grid.params.jDim == 3
        @test grid.params.kDim == 3
    end

    @testset "grid_from_regular_data auto var names" begin
        x = collect(LinRange(0.5, 4.5, 5))
        data = hcat(sin.(x), cos.(x))
        grid = grid_from_regular_data(x, data; mubar=1)
        @test haskey(grid.params.vars, "v1")
        @test haskey(grid.params.vars, "v2")
    end

    # ════════════════════════════════════════════════════════════════════════
    # Layer 1: grid_from_netcdf
    # ════════════════════════════════════════════════════════════════════════

    @testset "grid_from_netcdf 1D roundtrip" begin
        # Create a source grid, write to netcdf, read back with grid_from_netcdf
        # Note: write_netcdf outputs on regular grid with i_regular_out points,
        # so we need to make the output grid size compatible with the desired mubar.
        N = 12
        L = 6.0
        h = L / N
        x = [(i - 0.5) * h for i in 1:N]
        f = sin.(2π .* x ./ L)
        data = reshape(f, :, 1)

        source = grid_from_regular_data(x, data; mubar=3, vars=["temperature"])

        # Write a simple netcdf file manually for testing
        tmpfile = tempname() * ".nc"
        try
            NCDatasets.NCDataset(tmpfile, "c") do ds
                NCDatasets.defDim(ds, "x", N)
                ncx = NCDatasets.defVar(ds, "x", Float64, ("x",))
                ncf = NCDatasets.defVar(ds, "temperature", Float64, ("x",))
                ncx[:] = x
                ncf[:] = f
            end

            loaded = grid_from_netcdf(tmpfile; mubar=3)
            @test loaded isa R_Grid
            @test loaded.params.num_cells == 4  # 12/3
            @test loaded.params.iDim == 12
            @test loaded.physical[:, 1, 1] ≈ f atol=1e-12
        finally
            isfile(tmpfile) && rm(tmpfile)
        end
    end

    # ════════════════════════════════════════════════════════════════════════
    # Layer 2: interpolate_to_grid — 1D
    # ════════════════════════════════════════════════════════════════════════

    @testset "interpolate_to_grid 1D R→R" begin
        # Create source: regular grid with a smooth function
        N_src = 30
        L = 5.0
        h = L / N_src
        x = [(i - 0.5) * h for i in 1:N_src]
        f = sin.(2π .* x ./ L)
        data = reshape(f, :, 1)

        source = grid_from_regular_data(x, data; mubar=3, vars=["u"])
        spectralTransform!(source)

        # Create target: Gauss-Legendre grid with different resolution
        tgp = SpringsteelGridParameters(
            geometry  = "R",
            iMin      = 0.0,
            iMax      = L,
            num_cells = 8,
            mubar     = 3,
            quadrature = :gauss,
            BCL       = Dict("u" => CubicBSpline.R0),
            BCR       = Dict("u" => CubicBSpline.R0),
            vars      = Dict("u" => 1),
        )
        target = createGrid(tgp)

        result = interpolate_to_grid(source, target)
        @test size(result) == (target.params.iDim, 1)

        # Check accuracy: the interpolated values should match the analytic function
        t_pts = getGridpoints(target)
        f_exact = sin.(2π .* t_pts ./ L)
        @test result[:, 1] ≈ f_exact atol=0.05  # limited by source resolution

        # Test mutating version
        interpolate_to_grid!(source, target)
        @test target.physical[:, 1, 1] ≈ result[:, 1] atol=1e-12
        @test all(isnan, target.physical[:, 1, 2])
    end

    @testset "interpolate_to_grid 1D variable matching" begin
        N = 12
        L = 4.0
        h = L / N
        x = [(i - 0.5) * h for i in 1:N]
        data_src = hcat(sin.(x), cos.(x))

        source = grid_from_regular_data(x, data_src; mubar=3, vars=["u", "v"])
        spectralTransform!(source)

        # Target has only "u" + extra "w"
        tgp = SpringsteelGridParameters(
            geometry  = "R",
            iMin      = 0.0,
            iMax      = L,
            num_cells = 3,
            mubar     = 3,
            BCL       = Dict("u" => CubicBSpline.R0, "w" => CubicBSpline.R0),
            BCR       = Dict("u" => CubicBSpline.R0, "w" => CubicBSpline.R0),
            vars      = Dict("u" => 1, "w" => 2),
        )
        target = createGrid(tgp)

        result = interpolate_to_grid(source, target)
        @test size(result) == (target.params.iDim, 2)
        # "u" should be interpolated
        @test !any(isnan, result[:, 1])
        # "w" not in source → NaN
        @test all(isnan, result[:, 2])
    end

    @testset "interpolate_to_grid 1D out_of_bounds" begin
        N = 9
        h = 1.0
        x = [(i - 0.5) * h for i in 1:N]
        data = reshape(ones(N), :, 1)

        source = grid_from_regular_data(x, data; mubar=3, vars=["u"])
        spectralTransform!(source)

        # Target extends beyond source domain
        tgp = SpringsteelGridParameters(
            geometry  = "R",
            iMin      = -2.0,
            iMax      = 12.0,
            num_cells = 7,
            mubar     = 3,
            BCL       = Dict("u" => CubicBSpline.R0),
            BCR       = Dict("u" => CubicBSpline.R0),
            vars      = Dict("u" => 1),
        )
        target = createGrid(tgp)

        # Default: NaN for out-of-bounds
        result_nan = interpolate_to_grid(source, target)
        @test any(isnan, result_nan[:, 1])

        # Fill value
        result_fill = interpolate_to_grid(source, target; out_of_bounds=0.0)
        @test !any(isnan, result_fill[:, 1])

        # Error mode
        @test_throws DomainError interpolate_to_grid(source, target; out_of_bounds=:error)
    end

    # ════════════════════════════════════════════════════════════════════════
    # Layer 2: interpolate_to_grid — 2D
    # ════════════════════════════════════════════════════════════════════════

    @testset "interpolate_to_grid 2D RR→RR" begin
        # Source: regular grid with f(x,y) = x + y (linear, exactly representable)
        Nx, Ny = 9, 9
        Lx, Ly = 3.0, 3.0
        hx, hy = Lx / Nx, Ly / Ny
        x = [(i - 0.5) * hx for i in 1:Nx]
        y = [(j - 0.5) * hy for j in 1:Ny]

        data = zeros(Nx * Ny, 1)
        for i in 1:Nx
            for j in 1:Ny
                data[(i-1)*Ny + j, 1] = x[i] + y[j]
            end
        end

        source = grid_from_regular_data(x, y, data; mubar=3, vars=["u"])
        spectralTransform!(source)

        # Target: Gauss-Legendre grid
        tgp = SpringsteelGridParameters(
            geometry  = "RR",
            iMin      = 0.0,
            iMax      = Lx,
            num_cells = 4,
            mubar     = 3,
            jMin      = 0.0,
            jMax      = Ly,
            jDim      = 12,
            BCL       = Dict("u" => CubicBSpline.R0),
            BCR       = Dict("u" => CubicBSpline.R0),
            BCU       = Dict("u" => CubicBSpline.R0),
            BCD       = Dict("u" => CubicBSpline.R0),
            vars      = Dict("u" => 1),
        )
        target = createGrid(tgp)

        result = interpolate_to_grid(source, target)
        @test size(result, 1) == target.params.iDim * target.params.jDim

        # Check accuracy for linear function (should be very accurate)
        t_pts = getGridpoints(target)
        f_exact = t_pts[:, 1] .+ t_pts[:, 2]
        @test result[:, 1] ≈ f_exact atol=0.1

        # Test mutating version
        interpolate_to_grid!(source, target)
        @test target.physical[:, 1, 1] ≈ result[:, 1] atol=1e-12
    end

    # ════════════════════════════════════════════════════════════════════════
    # Layer 2: interpolate_to_grid — roundtrip accuracy
    # ════════════════════════════════════════════════════════════════════════

    @testset "interpolate_to_grid roundtrip 1D" begin
        # Create source grid (Gauss), set a known function, transform to spectral
        gp_src = SpringsteelGridParameters(
            geometry  = "R",
            iMin      = 0.0,
            iMax      = 4.0,
            num_cells = 10,
            mubar     = 3,
            quadrature = :gauss,
            BCL       = Dict("u" => CubicBSpline.R0),
            BCR       = Dict("u" => CubicBSpline.R0),
            vars      = Dict("u" => 1),
        )
        source = createGrid(gp_src)
        pts = getGridpoints(source)
        source.physical[:, 1, 1] .= sin.(2π .* pts ./ 4.0)
        spectralTransform!(source)

        # Target: different Gauss grid
        gp_tgt = SpringsteelGridParameters(
            geometry  = "R",
            iMin      = 0.0,
            iMax      = 4.0,
            num_cells = 12,
            mubar     = 3,
            quadrature = :gauss,
            BCL       = Dict("u" => CubicBSpline.R0),
            BCR       = Dict("u" => CubicBSpline.R0),
            vars      = Dict("u" => 1),
        )
        target = createGrid(gp_tgt)

        result = interpolate_to_grid(source, target)
        t_pts = getGridpoints(target)
        f_exact = sin.(2π .* t_pts ./ 4.0)

        # Accuracy limited by source resolution (10 cells with R0 BCs)
        @test result[:, 1] ≈ f_exact atol=0.01
    end

    # ════════════════════════════════════════════════════════════════════════
    # Layer 3: Coordinate mapping functions
    # ════════════════════════════════════════════════════════════════════════

    @testset "Coordinate mappings" begin

        @testset "Cart <-> Cyl 2D" begin
            # Basic roundtrip: first quadrant
            x, y = 3.0, 4.0
            r, λ = Springsteel.cartesian_to_cylindrical(x, y)
            @test r ≈ 5.0 atol=1e-14
            @test λ ≈ atan(y, x) atol=1e-14
            x2, y2 = Springsteel.cylindrical_to_cartesian(r, λ)
            @test x2 ≈ x atol=1e-14
            @test y2 ≈ y atol=1e-14

            # Negative quadrant: lambda should wrap to [0, 2pi)
            x_neg, y_neg = -1.0, -1.0
            r_neg, λ_neg = Springsteel.cartesian_to_cylindrical(x_neg, y_neg)
            @test r_neg ≈ sqrt(2.0) atol=1e-14
            @test λ_neg >= 0.0
            @test λ_neg < 2π
            @test λ_neg ≈ atan(y_neg, x_neg) + 2π atol=1e-14
            x3, y3 = Springsteel.cylindrical_to_cartesian(r_neg, λ_neg)
            @test x3 ≈ x_neg atol=1e-13
            @test y3 ≈ y_neg atol=1e-13

            # Along negative x-axis
            r_nx, λ_nx = Springsteel.cartesian_to_cylindrical(-5.0, 0.0)
            @test r_nx ≈ 5.0 atol=1e-14
            @test λ_nx ≈ π atol=1e-14

            # Origin (r=0): lambda undefined but should not error
            r0, λ0 = Springsteel.cartesian_to_cylindrical(0.0, 0.0)
            @test r0 ≈ 0.0 atol=1e-14

            # Multiple roundtrips at various angles
            for angle in [0.0, π/6, π/3, π/2, π, 3π/2, 2π - 0.01]
                R = 10.0
                xr = R * cos(angle)
                yr = R * sin(angle)
                rr, λr = Springsteel.cartesian_to_cylindrical(xr, yr)
                xb, yb = Springsteel.cylindrical_to_cartesian(rr, λr)
                @test xb ≈ xr atol=1e-12
                @test yb ≈ yr atol=1e-12
            end
        end

        @testset "Cart <-> Cyl 3D" begin
            x, y, z = 3.0, 4.0, 7.0
            r, λ, z_out = Springsteel.cartesian_to_cylindrical_3d(x, y, z)
            @test r ≈ 5.0 atol=1e-14
            @test z_out ≈ z atol=1e-14
            x2, y2, z2 = Springsteel.cylindrical_to_cartesian_3d(r, λ, z_out)
            @test x2 ≈ x atol=1e-14
            @test y2 ≈ y atol=1e-14
            @test z2 ≈ z atol=1e-14

            # z should pass through unchanged
            for zv in [-100.0, 0.0, 42.0]
                _, _, zr = Springsteel.cartesian_to_cylindrical_3d(1.0, 0.0, zv)
                @test zr ≈ zv atol=1e-14
            end
        end

        @testset "Cart <-> Sph 3D" begin
            # Basic roundtrip
            x, y, z = 1.0, 2.0, 3.0
            rho, θ, φ = Springsteel.cartesian_to_spherical(x, y, z)
            @test rho ≈ sqrt(x^2 + y^2 + z^2) atol=1e-14
            @test θ >= 0.0
            @test θ <= π
            @test φ >= 0.0
            @test φ < 2π
            x2, y2, z2 = Springsteel.spherical_to_cartesian(rho, θ, φ)
            @test x2 ≈ x atol=1e-13
            @test y2 ≈ y atol=1e-13
            @test z2 ≈ z atol=1e-13

            # Pole cases: z-axis → theta=0
            rho_z, θ_z, _ = Springsteel.cartesian_to_spherical(0.0, 0.0, 5.0)
            @test rho_z ≈ 5.0 atol=1e-14
            @test θ_z ≈ 0.0 atol=1e-14

            # Negative z-axis → theta=pi
            rho_nz, θ_nz, _ = Springsteel.cartesian_to_spherical(0.0, 0.0, -5.0)
            @test rho_nz ≈ 5.0 atol=1e-14
            @test θ_nz ≈ π atol=1e-14

            # Origin
            rho0, _, _ = Springsteel.cartesian_to_spherical(0.0, 0.0, 0.0)
            @test rho0 ≈ 0.0 atol=1e-14

            # Equatorial plane roundtrip (theta = pi/2)
            for φ_test in [0.0, π/4, π, 3π/2]
                R = 8.0
                xt = R * sin(π/2) * cos(φ_test)
                yt = R * sin(π/2) * sin(φ_test)
                zt = R * cos(π/2)
                rr, tr, pr = Springsteel.cartesian_to_spherical(xt, yt, zt)
                xb, yb, zb = Springsteel.spherical_to_cartesian(rr, tr, pr)
                @test xb ≈ xt atol=1e-12
                @test yb ≈ yt atol=1e-12
                @test zb ≈ zt atol=1e-12
            end
        end

        @testset "Cyl <-> Sph 3D" begin
            # Cylindrical (r_cyl, lambda, z) -> Spherical (rho, theta, phi)
            r_cyl, λ_cyl, z_cyl = 3.0, π/4, 4.0
            rho, θ, φ = Springsteel.cylindrical_to_spherical(r_cyl, λ_cyl, z_cyl)
            @test rho ≈ sqrt(r_cyl^2 + z_cyl^2) atol=1e-14
            @test φ ≈ λ_cyl atol=1e-14  # azimuth should be preserved
            r2, λ2, z2 = Springsteel.spherical_to_cylindrical(rho, θ, φ)
            @test r2 ≈ r_cyl atol=1e-13
            @test λ2 ≈ λ_cyl atol=1e-13
            @test z2 ≈ z_cyl atol=1e-13

            # On the z-axis: r_cyl=0
            rho_z, θ_z, _ = Springsteel.cylindrical_to_spherical(0.0, 0.0, 10.0)
            @test rho_z ≈ 10.0 atol=1e-14
            @test θ_z ≈ 0.0 atol=1e-14

            # In the equatorial plane: z=0
            rho_eq, θ_eq, φ_eq = Springsteel.cylindrical_to_spherical(5.0, π/3, 0.0)
            @test rho_eq ≈ 5.0 atol=1e-14
            @test θ_eq ≈ π/2 atol=1e-14
            @test φ_eq ≈ π/3 atol=1e-14
        end

        @testset "Lat/lon helpers" begin
            # Equator: lat=0, lon=0 → theta=pi/2, phi=0
            θ_eq, φ_eq = Springsteel.latlon_to_spherical(0.0, 0.0)
            @test θ_eq ≈ π/2 atol=1e-14
            @test φ_eq ≈ 0.0 atol=1e-14

            # North pole: lat=90 → theta=0
            θ_np, _ = Springsteel.latlon_to_spherical(0.0, 90.0)
            @test θ_np ≈ 0.0 atol=1e-14

            # South pole: lat=-90 → theta=pi
            θ_sp, _ = Springsteel.latlon_to_spherical(0.0, -90.0)
            @test θ_sp ≈ π atol=1e-14

            # Date line: lon=180 → phi=pi
            _, φ_dl = Springsteel.latlon_to_spherical(180.0, 0.0)
            @test φ_dl ≈ π atol=1e-14

            # Negative longitude: lon=-90 → phi=3pi/2 (wraps to [0,2pi))
            _, φ_neg = Springsteel.latlon_to_spherical(-90.0, 0.0)
            @test φ_neg ≈ 3π/2 atol=1e-14
            @test φ_neg >= 0.0
            @test φ_neg < 2π

            # Roundtrip lat/lon → spherical → lat/lon
            for (lon, lat) in [(90.0, 45.0), (300.0, -30.0), (179.9, 0.0), (0.0, 89.0)]
                θ_rt, φ_rt = Springsteel.latlon_to_spherical(lon, lat)
                lon2, lat2 = Springsteel.spherical_to_latlon(θ_rt, φ_rt)
                @test lat2 ≈ lat atol=1e-10
                @test lon2 ≈ lon atol=1e-10
            end
        end

    end  # "Coordinate mappings"

    # ════════════════════════════════════════════════════════════════════════
    # Layer 3: Unstructured point evaluation
    # ════════════════════════════════════════════════════════════════════════

    @testset "evaluate_unstructured" begin

        @testset "R grid" begin
            # Create R grid with f(x) = sin(2pi*x/L)
            L = 5.0
            gp = SpringsteelGridParameters(
                geometry  = "R",
                iMin      = 0.0,
                iMax      = L,
                num_cells = 10,
                mubar     = 3,
                quadrature = :gauss,
                BCL       = Dict("u" => CubicBSpline.R0),
                BCR       = Dict("u" => CubicBSpline.R0),
                vars      = Dict("u" => 1),
            )
            grid = createGrid(gp)
            pts = getGridpoints(grid)
            grid.physical[:, 1, 1] .= sin.(2π .* pts ./ L)
            spectralTransform!(grid)

            # Evaluate at regular grid points using both methods
            reg_pts = getRegularGridpoints(grid)
            reg_phys = regularGridTransform(grid, reg_pts)

            # evaluate_unstructured takes an (N, ndim) matrix of points
            pts_mat = reshape(reg_pts, :, 1)
            result = Springsteel.evaluate_unstructured(grid, pts_mat)

            # Compare function values (slot 1)
            @test size(result, 1) == length(reg_pts)
            @test result[:, 1] ≈ reg_phys[:, 1, 1] atol=1e-10
        end

        @testset "RL grid" begin
            # f(r, λ) = r * cos(λ) — this is x in Cartesian coordinates
            gp = SpringsteelGridParameters(
                geometry = "RL",
                iMin = 5.0, iMax = 50.0,
                num_cells = 8,
                patchOffsetL = 0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            # Set physical values: f(r,λ) = r*cos(λ)
            mish_pts = getGridpoints(grid)
            for i in 1:size(mish_pts, 1)
                r_val = mish_pts[i, 1]
                λ_val = mish_pts[i, 2]
                grid.physical[i, 1, 1] = r_val * cos(λ_val)
            end
            spectralTransform!(grid)

            # Get regular grid points and evaluate via regularGridTransform
            reg_pts = getRegularGridpoints(grid)
            reg_phys = regularGridTransform(grid, reg_pts)

            # Now evaluate at the same points via evaluate_unstructured
            result = Springsteel.evaluate_unstructured(grid, reg_pts)

            # Compare function values
            @test size(result, 1) == size(reg_pts, 1)
            @test result[:, 1] ≈ reg_phys[:, 1, 1] atol=1e-8
        end

        @testset "RR grid" begin
            # f(x, y) = sin(pi*x/Lx) * sin(pi*y/Ly)
            Lx, Ly = 4.0, 4.0
            gp = SpringsteelGridParameters(
                geometry  = "RR",
                iMin      = 0.0, iMax = Lx,
                num_cells = 6,
                mubar     = 3,
                jMin      = 0.0, jMax = Ly, jDim = 18,
                vars      = Dict("u" => 1),
                BCL       = Dict("u" => CubicBSpline.R0),
                BCR       = Dict("u" => CubicBSpline.R0),
                BCU       = Dict("u" => CubicBSpline.R0),
                BCD       = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            mish_pts = getGridpoints(grid)
            for i in 1:size(mish_pts, 1)
                x_val = mish_pts[i, 1]
                y_val = mish_pts[i, 2]
                grid.physical[i, 1, 1] = sin(π * x_val / Lx) * sin(π * y_val / Ly)
            end
            spectralTransform!(grid)

            # Get regular grid points
            reg_pts = getRegularGridpoints(grid)
            reg_phys = regularGridTransform(grid, reg_pts)

            # evaluate_unstructured
            result = Springsteel.evaluate_unstructured(grid, reg_pts)

            @test size(result, 1) == size(reg_pts, 1)
            @test result[:, 1] ≈ reg_phys[:, 1, 1] atol=1e-8
        end

        @testset "RZ grid" begin
            # f(r, z) = r * z (bilinear, well-representable)
            gp = SpringsteelGridParameters(
                geometry  = "RZ",
                iMin      = 1.0, iMax = 5.0,
                num_cells = 6,
                mubar     = 3,
                kMin      = 0.0, kMax = 10.0, kDim = 12,
                vars      = Dict("u" => 1),
                BCL       = Dict("u" => CubicBSpline.R0),
                BCR       = Dict("u" => CubicBSpline.R0),
                BCB       = Dict("u" => Chebyshev.R0),
                BCT       = Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)

            mish_pts = getGridpoints(grid)
            for i in 1:size(mish_pts, 1)
                r_val = mish_pts[i, 1]
                z_val = mish_pts[i, 2]
                grid.physical[i, 1, 1] = r_val * z_val
            end
            spectralTransform!(grid)

            # Compare
            reg_pts = getRegularGridpoints(grid)
            reg_phys = regularGridTransform(grid, reg_pts)

            result = Springsteel.evaluate_unstructured(grid, reg_pts)

            @test size(result, 1) == size(reg_pts, 1)
            @test result[:, 1] ≈ reg_phys[:, 1, 1] atol=1e-8
        end

        @testset "OOB handling" begin
            L = 4.0
            gp = SpringsteelGridParameters(
                geometry  = "R",
                iMin      = 0.0,
                iMax      = L,
                num_cells = 8,
                mubar     = 3,
                BCL       = Dict("u" => CubicBSpline.R0),
                BCR       = Dict("u" => CubicBSpline.R0),
                vars      = Dict("u" => 1),
            )
            grid = createGrid(gp)
            pts = getGridpoints(grid)
            grid.physical[:, 1, 1] .= ones(length(pts))
            spectralTransform!(grid)

            # Point outside domain → NaN by default
            oob_pts = reshape([-1.0, 2.0, 5.0], :, 1)
            result = Springsteel.evaluate_unstructured(grid, oob_pts)
            @test isnan(result[1, 1])   # -1.0 is out of bounds
            @test !isnan(result[2, 1])  # 2.0 is in bounds
            @test isnan(result[3, 1])   # 5.0 is out of bounds

            # :error mode → DomainError
            @test_throws DomainError Springsteel.evaluate_unstructured(
                grid, oob_pts; out_of_bounds=:error)

            # Numeric fill value
            result_fill = Springsteel.evaluate_unstructured(
                grid, oob_pts; out_of_bounds=-999.0)
            @test result_fill[1, 1] ≈ -999.0 atol=1e-14
            @test result_fill[3, 1] ≈ -999.0 atol=1e-14
            @test !isnan(result_fill[2, 1])
        end

        @testset "RL axisymmetric" begin
            # Axisymmetric function f(r) = sin(pi*r/rMax) — independent of λ
            rMin, rMax = 0.0, 40.0
            gp = SpringsteelGridParameters(
                geometry = "RL",
                iMin = rMin, iMax = rMax,
                num_cells = 10,
                patchOffsetL = 0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            mish_pts = getGridpoints(grid)
            for i in 1:size(mish_pts, 1)
                r_val = mish_pts[i, 1]
                grid.physical[i, 1, 1] = sin(π * r_val / rMax)
            end
            spectralTransform!(grid)

            # Evaluate at a few known (r, λ) points
            test_pts = Float64[
                10.0  0.0;
                15.0  π/4;
                20.0  π/2;
                25.0  π;
                30.0  3π/2;
                 5.0  0.3;
            ]
            result = Springsteel.evaluate_unstructured(grid, test_pts)
            @test size(result) == (6, 1)
            for i in 1:size(test_pts, 1)
                r_val = test_pts[i, 1]
                expected = sin(π * r_val / rMax)
                @test result[i, 1] ≈ expected atol=0.05
            end
        end

        @testset "RLZ axisymmetric" begin
            # f(r, λ, z) = sin(pi*r/rMax) — independent of λ and z
            rMin, rMax = 0.0, 30.0
            zMin, zMax = 0.0, 10.0
            gp = SpringsteelGridParameters(
                geometry = "RLZ",
                iMin = rMin, iMax = rMax,
                num_cells = 8,
                patchOffsetL = 0,
                kMin = zMin, kMax = zMax, kDim = 10,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCB = Dict("u" => Chebyshev.R0),
                BCT = Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)

            mish_pts = getGridpoints(grid)
            for i in 1:size(mish_pts, 1)
                r_val = mish_pts[i, 1]
                grid.physical[i, 1, 1] = sin(π * r_val / rMax)
            end
            spectralTransform!(grid)

            # Evaluate at a few known (r, λ, z) points
            test_pts = Float64[
                 5.0  0.0   5.0;
                10.0  π/4   2.0;
                15.0  π/2   7.0;
                20.0  π     4.0;
                25.0  3π/2  8.0;
            ]
            result = Springsteel.evaluate_unstructured(grid, test_pts)
            @test size(result) == (5, 1)
            for i in 1:size(test_pts, 1)
                r_val = test_pts[i, 1]
                expected = sin(π * r_val / rMax)
                @test result[i, 1] ≈ expected atol=0.05
            end
        end

    end  # "evaluate_unstructured"

    # ════════════════════════════════════════════════════════════════════════
    # Layer 3: Cross-geometry interpolation
    # ════════════════════════════════════════════════════════════════════════

    @testset "Cross-geometry interpolation" begin

        @testset "RL -> RR" begin
            # f(r, λ) = r * cos(λ) = x in Cartesian
            # On the RL grid, set f = r*cos(λ), then interpolate to RR grid
            # and check that target values ≈ x (the Cartesian x-coordinate)

            gp_src = SpringsteelGridParameters(
                geometry = "RL",
                iMin = 5.0, iMax = 40.0,
                num_cells = 8,
                patchOffsetL = 0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0))
            source = createGrid(gp_src)

            mish_pts = getGridpoints(source)
            for i in 1:size(mish_pts, 1)
                r_val = mish_pts[i, 1]
                λ_val = mish_pts[i, 2]
                source.physical[i, 1, 1] = r_val * cos(λ_val)
            end
            spectralTransform!(source)

            # Target: Cartesian RR grid that fits inside the RL annular domain
            # The RL domain is an annulus r in [5, 40], so the RR domain must
            # be inside the circle of radius 40 and outside radius 5
            gp_tgt = SpringsteelGridParameters(
                geometry  = "RR",
                iMin      = -30.0, iMax = 30.0,
                num_cells = 8,
                mubar     = 3,
                jMin      = -30.0, jMax = 30.0, jDim = 24,
                vars      = Dict("u" => 1),
                BCL       = Dict("u" => CubicBSpline.R0),
                BCR       = Dict("u" => CubicBSpline.R0),
                BCU       = Dict("u" => CubicBSpline.R0),
                BCD       = Dict("u" => CubicBSpline.R0))
            target = createGrid(gp_tgt)

            result = interpolate_to_grid(source, target;
                coordinate_map=Springsteel.cartesian_to_cylindrical)

            # Check in-bounds points: result should equal x-coordinate of target
            t_pts = getGridpoints(target)
            for i in 1:size(t_pts, 1)
                x_val = t_pts[i, 1]
                y_val = t_pts[i, 2]
                r_check = sqrt(x_val^2 + y_val^2)
                if r_check >= 5.0 && r_check <= 40.0
                    # f(r, λ) = r*cos(λ) = x → expect result ≈ x
                    @test result[i, 1] ≈ x_val atol=0.5
                end
            end
        end

        @testset "RR -> RL" begin
            # f(x, y) = x^2 + y^2 = r^2 in cylindrical coordinates
            # On the RR grid, set f = x^2 + y^2, then interpolate to RL grid
            # and check that target values ≈ r^2

            Lx, Ly = 30.0, 30.0
            gp_src = SpringsteelGridParameters(
                geometry  = "RR",
                iMin      = -Lx, iMax = Lx,
                num_cells = 10,
                mubar     = 3,
                jMin      = -Ly, jMax = Ly, jDim = 30,
                vars      = Dict("u" => 1),
                BCL       = Dict("u" => CubicBSpline.R0),
                BCR       = Dict("u" => CubicBSpline.R0),
                BCU       = Dict("u" => CubicBSpline.R0),
                BCD       = Dict("u" => CubicBSpline.R0))
            source = createGrid(gp_src)

            mish_pts = getGridpoints(source)
            for i in 1:size(mish_pts, 1)
                x_val = mish_pts[i, 1]
                y_val = mish_pts[i, 2]
                source.physical[i, 1, 1] = x_val^2 + y_val^2
            end
            spectralTransform!(source)

            # Target: RL grid fully inside the RR domain
            gp_tgt = SpringsteelGridParameters(
                geometry = "RL",
                iMin = 2.0, iMax = 25.0,
                num_cells = 6,
                patchOffsetL = 0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0))
            target = createGrid(gp_tgt)

            result = interpolate_to_grid(source, target;
                coordinate_map=Springsteel.cylindrical_to_cartesian)

            # Check: result should ≈ r^2 at each target grid point
            t_pts = getGridpoints(target)
            for i in 1:size(t_pts, 1)
                r_val = t_pts[i, 1]
                expected = r_val^2
                @test result[i, 1] ≈ expected atol=2.0  # modest tolerance for spectral approx
            end
        end

        @testset "coordinate_map keyword" begin
            # Verify that providing an explicit coordinate_map gives the same
            # result as the default for same-geometry grids that have a known
            # default mapping. Here we test with a simple identity-like wrapper
            # for RR -> RR (which is same-geometry and doesn't need a map).

            Lx = 4.0
            gp = SpringsteelGridParameters(
                geometry  = "RR",
                iMin      = 0.0, iMax = Lx,
                num_cells = 4,
                mubar     = 3,
                jMin      = 0.0, jMax = Lx, jDim = 12,
                vars      = Dict("u" => 1),
                BCL       = Dict("u" => CubicBSpline.R0),
                BCR       = Dict("u" => CubicBSpline.R0),
                BCU       = Dict("u" => CubicBSpline.R0),
                BCD       = Dict("u" => CubicBSpline.R0))
            source = createGrid(gp)

            mish_pts = getGridpoints(source)
            for i in 1:size(mish_pts, 1)
                source.physical[i, 1, 1] = mish_pts[i, 1] + mish_pts[i, 2]
            end
            spectralTransform!(source)

            gp_tgt = SpringsteelGridParameters(
                geometry  = "RR",
                iMin      = 0.5, iMax = Lx - 0.5,
                num_cells = 3,
                mubar     = 3,
                jMin      = 0.5, jMax = Lx - 0.5, jDim = 9,
                vars      = Dict("u" => 1),
                BCL       = Dict("u" => CubicBSpline.R0),
                BCR       = Dict("u" => CubicBSpline.R0),
                BCU       = Dict("u" => CubicBSpline.R0),
                BCD       = Dict("u" => CubicBSpline.R0))
            target = createGrid(gp_tgt)

            # Same-geometry interpolation (no coordinate_map)
            result_default = interpolate_to_grid(source, target)

            # With explicit identity map (x,y) -> (x,y)
            identity_map(x, y) = (x, y)
            result_mapped = interpolate_to_grid(source, target;
                coordinate_map=identity_map)

            @test result_default[:, 1] ≈ result_mapped[:, 1] atol=1e-10
        end

        @testset "missing coordinate_map error" begin
            # Cross-geometry interpolation without coordinate_map should raise
            # an error when the default is not known (e.g. RR -> SL)

            gp_src = SpringsteelGridParameters(
                geometry  = "RR",
                iMin      = -10.0, iMax = 10.0,
                num_cells = 4,
                mubar     = 3,
                jMin      = -10.0, jMax = 10.0, jDim = 12,
                vars      = Dict("u" => 1),
                BCL       = Dict("u" => CubicBSpline.R0),
                BCR       = Dict("u" => CubicBSpline.R0),
                BCU       = Dict("u" => CubicBSpline.R0),
                BCD       = Dict("u" => CubicBSpline.R0))
            source = createGrid(gp_src)

            mish_pts = getGridpoints(source)
            source.physical[:, 1, 1] .= ones(size(mish_pts, 1))
            spectralTransform!(source)

            # SL grid (spherical) — different geometry from RR
            gp_tgt = SpringsteelGridParameters(
                geometry = "SL",
                iMin     = 5.0, iMax = 15.0,
                num_cells = 3,
                patchOffsetL = 0,
                vars     = Dict("u" => 1),
                BCL      = Dict("u" => CubicBSpline.R0),
                BCR      = Dict("u" => CubicBSpline.R0))
            target = createGrid(gp_tgt)

            # Without coordinate_map, cross-geometry should fail
            @test_throws ArgumentError interpolate_to_grid(source, target)
        end

        @testset "RL -> RL identity map" begin
            # Same-geometry RL→RL with explicit identity coordinate_map
            # exercises the unstructured path with cross-geometry routing.
            rMin, rMax = 5.0, 35.0
            gp_src = SpringsteelGridParameters(
                geometry = "RL",
                iMin = rMin, iMax = rMax,
                num_cells = 8,
                patchOffsetL = 0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0))
            source = createGrid(gp_src)

            mish_pts = getGridpoints(source)
            for i in 1:size(mish_pts, 1)
                r_val = mish_pts[i, 1]
                source.physical[i, 1, 1] = sin(π * (r_val - rMin) / (rMax - rMin))
            end
            spectralTransform!(source)

            gp_tgt = SpringsteelGridParameters(
                geometry = "RL",
                iMin = 10.0, iMax = 30.0,
                num_cells = 6,
                patchOffsetL = 0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0))
            target = createGrid(gp_tgt)

            identity_rl(r, λ) = (r, λ)
            result = interpolate_to_grid(source, target; coordinate_map=identity_rl)

            # Check accuracy against the analytic axisymmetric function
            t_pts = getGridpoints(target)
            for i in 1:size(t_pts, 1)
                r_val = t_pts[i, 1]
                expected = sin(π * (r_val - rMin) / (rMax - rMin))
                @test result[i, 1] ≈ expected atol=0.05
            end
        end

    end  # "Cross-geometry interpolation"

end

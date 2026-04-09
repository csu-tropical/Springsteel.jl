    @testset "1D Cartesian roundtrip" begin

        @testset "getGridpoints" begin
            gp = SpringsteelGridParameters(
                geometry = "R",
                num_cells = 10,
                iMin = 0.0,
                iMax = 10.0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)
            pts = getGridpoints(grid)

            @test length(pts) == grid.params.iDim       # 30 = 10*3 mish points
            @test pts[1] >= gp.iMin
            @test pts[end] <= gp.iMax
            @test all(diff(pts) .> 0)                   # monotonically increasing
        end

        @testset "Cubic polynomial roundtrip (exact)" begin
            # Cubic B-splines represent cubic polynomials exactly,
            # so f(x) = x^3 - 2x^2 + x + 1 should roundtrip to machine precision.
            gp = SpringsteelGridParameters(
                geometry = "R",
                num_cells = 10,
                iMin = 0.0,
                iMax = 1.0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)
            pts  = getGridpoints(grid)

            for i in eachindex(pts)
                x = pts[i]
                grid.physical[i, 1, 1] = x^3 - 2*x^2 + x + 1
            end
            original = copy(grid.physical[:, 1, 1])

            spectralTransform!(grid)
            @test maximum(abs.(grid.spectral[:, 1])) > 1e-10  # non-trivial coefficients

            gridTransform!(grid)
            max_err = maximum(abs.(grid.physical[:, 1, 1] .- original))
            @test max_err < 1e-3    # cubic spline accuracy with BCs
        end

        @testset "Sinusoid roundtrip (smooth function)" begin
            # Smooth periodic function: f(x) = sin(2π x / L) on [0, L]
            gp = SpringsteelGridParameters(
                geometry = "R",
                num_cells = 20,
                iMin = 0.0,
                iMax = 10.0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.PERIODIC),
                BCR = Dict("u" => CubicBSpline.PERIODIC))
            grid = createGrid(gp)
            pts  = getGridpoints(grid)
            L    = gp.iMax - gp.iMin

            for i in eachindex(pts)
                grid.physical[i, 1, 1] = sin(2π * pts[i] / L)
            end
            original = copy(grid.physical[:, 1, 1])

            spectralTransform!(grid)
            gridTransform!(grid)

            max_err = maximum(abs.(grid.physical[:, 1, 1] .- original))
            @test max_err < 1e-4    # cubic spline accuracy for smooth function
        end

        @testset "Multi-variable roundtrip" begin
            gp = SpringsteelGridParameters(
                geometry = "R",
                num_cells = 15,
                iMin = 0.0,
                iMax = 2π,
                vars = Dict("u" => 1, "v" => 2),
                BCL = Dict("u" => CubicBSpline.PERIODIC, "v" => CubicBSpline.PERIODIC),
                BCR = Dict("u" => CubicBSpline.PERIODIC, "v" => CubicBSpline.PERIODIC))
            grid = createGrid(gp)
            pts  = getGridpoints(grid)

            for i in eachindex(pts)
                grid.physical[i, 1, 1] = sin(pts[i])
                grid.physical[i, 2, 1] = cos(pts[i])
            end
            orig_u = copy(grid.physical[:, 1, 1])
            orig_v = copy(grid.physical[:, 2, 1])

            spectralTransform!(grid)
            gridTransform!(grid)

            @test maximum(abs.(grid.physical[:, 1, 1] .- orig_u)) < 1e-4
            @test maximum(abs.(grid.physical[:, 2, 1] .- orig_v)) < 1e-4
        end

        @testset "Derivative accuracy: sin(x)" begin
            # f(x) = sin(x) on [0, 2π] periodic
            # df/dx = cos(x),  d²f/dx² = -sin(x)
            gp = SpringsteelGridParameters(
                geometry = "R",
                num_cells = 30,
                iMin = 0.0,
                iMax = 2π,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.PERIODIC),
                BCR = Dict("u" => CubicBSpline.PERIODIC))
            grid = createGrid(gp)
            pts  = getGridpoints(grid)

            for i in eachindex(pts)
                grid.physical[i, 1, 1] = sin(pts[i])
            end

            spectralTransform!(grid)
            gridTransform!(grid)

            # Value
            @test maximum(abs.(grid.physical[:, 1, 1] .- sin.(pts))) < 1e-4
            # First derivative: cos(x)
            @test maximum(abs.(grid.physical[:, 1, 2] .- cos.(pts))) < 0.01
            # Second derivative: -sin(x)
            @test maximum(abs.(grid.physical[:, 1, 3] .+ sin.(pts))) < 0.01
        end

        @testset "Gaussian derivatives" begin
            # Analytic test: u = exp(-(x/σ)²), u' = -2x/σ² * u, u'' = (4x²/σ⁴ - 2/σ²) * u
            sigma = 20.0
            gp = SpringsteelGridParameters(
                geometry = "R",
                num_cells = 100,
                iMin = -50.0,
                iMax = 50.0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)
            pts  = getGridpoints(grid)

            for i in eachindex(pts)
                grid.physical[i, 1, 1] = exp(-(pts[i] / sigma)^2)
            end

            spectralTransform!(grid)
            gridTransform!(grid)

            analytic_vals = [exp(-(x / sigma)^2)                          for x in pts]
            analytic_dx   = [-2x / sigma^2 * exp(-(x / sigma)^2)          for x in pts]
            analytic_dxx  = [(4x^2 / sigma^4 - 2 / sigma^2) * exp(-(x / sigma)^2) for x in pts]

            @test maximum(abs.(grid.physical[:, 1, 1] .- analytic_vals)) < 1e-5
            @test maximum(abs.(grid.physical[:, 1, 2] .- analytic_dx))   < 1e-4
            @test maximum(abs.(grid.physical[:, 1, 3] .- analytic_dxx))  < 1e-4
        end

        @testset "spectralTransform / gridTransform (explicit-array variants)" begin
            gp = SpringsteelGridParameters(
                geometry = "R",
                num_cells = 20,
                iMin = 0.0,
                iMax = 2π,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.PERIODIC),
                BCR = Dict("u" => CubicBSpline.PERIODIC))
            grid = createGrid(gp)
            pts  = getGridpoints(grid)

            for i in eachindex(pts)
                grid.physical[i, 1, 1] = sin(pts[i])
            end
            original = copy(grid.physical[:, 1, 1])

            # Explicit-array forward transform
            spec2 = zeros(Float64, size(grid.spectral))
            Springsteel.spectralTransform(grid, grid.physical, spec2)
            @test maximum(abs.(spec2 .- grid.spectral)) > 0.0   # non-trivial (grid.spectral still zero)
            @test maximum(abs.(spec2)) > 1e-10                  # non-trivial coefficients

            # Explicit-array inverse transform
            phys2 = zeros(Float64, size(grid.physical))
            Springsteel.gridTransform(grid, phys2, spec2)
            max_err = maximum(abs.(phys2[:, 1, 1] .- original))
            @test max_err < 1e-4
        end

    end  # 1D Cartesian roundtrip

    # ─────────────────────────────────────────────────────────────────────────
    # 2D Cartesian Transforms
    # SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, NoBasisArray}  (RR)
    # SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, ChebyshevBasisArray} (RZ)
    # ─────────────────────────────────────────────────────────────────────────
    @testset "2D Cartesian Transforms" begin

        @testset "RR getGridpoints" begin
            gp = SpringsteelGridParameters(
                geometry  = "RR",
                num_cells = 6,
                iMin = 0.0, iMax = 30.0,
                jMin = 0.0, jMax = 30.0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCU = Dict("u" => CubicBSpline.R0),
                BCD = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)
            pts  = getGridpoints(grid)

            @test size(pts, 2) == 2
            @test size(pts, 1) == grid.params.iDim * grid.params.jDim
            @test all(pts[:, 1] .>= gp.iMin)
            @test all(pts[:, 1] .<= gp.iMax)
            @test all(pts[:, 2] .>= gp.jMin)
            @test all(pts[:, 2] .<= gp.jMax)
        end

        @testset "RZ getGridpoints" begin
            kDim = 12
            gp = SpringsteelGridParameters(
                geometry  = "RZ",
                num_cells = 6,
                iMin = 0.0, iMax = 30.0,
                kMin = 0.0, kMax = 15.0,
                kDim = kDim,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCB = Dict("u" => Chebyshev.R0),
                BCT = Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)
            pts  = getGridpoints(grid)

            @test size(pts, 2) == 2
            @test size(pts, 1) == grid.params.iDim * grid.params.kDim
            @test all(pts[:, 1] .>= gp.iMin)
            @test all(pts[:, 1] .<= gp.iMax)
        end

        @testset "RR Spline×Spline roundtrip (quadratic)" begin
            # f(x,y) = (x/xmax)^2 * (y/ymax)  — polynomial, spline-exact
            num_cells = 8
            xmax = 40.0; ymax = 40.0
            gp = SpringsteelGridParameters(
                geometry  = "RR",
                num_cells = num_cells,
                iMin = 0.0, iMax = xmax,
                jMin = 0.0, jMax = ymax,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCU = Dict("u" => CubicBSpline.R0),
                BCD = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            iDim = grid.params.iDim
            jDim = grid.params.jDim
            for r in 1:iDim
                x = grid.ibasis.data[1, 1].mishPoints[r]
                for l in 1:jDim
                    y = grid.jbasis.data[r, 1].mishPoints[l]
                    grid.physical[(r-1)*jDim + l, 1, 1] = (x/xmax)^2 * (y/ymax)
                end
            end
            original = copy(grid.physical[:, 1, 1])

            spectralTransform!(grid)
            @test maximum(abs.(grid.spectral[:, 1])) > 1e-10  # non-trivial

            gridTransform!(grid)
            max_err = maximum(abs.(grid.physical[:, 1, 1] .- original))
            @test max_err < 1e-3

            # ∂/∂i and ∂/∂j should be non-trivial
            @test maximum(abs.(grid.physical[:, 1, 2])) > 1e-10
            @test maximum(abs.(grid.physical[:, 1, 4])) > 1e-10
        end

        @testset "RR multi-variable roundtrip" begin
            gp = SpringsteelGridParameters(
                geometry  = "RR",
                num_cells = 6,
                iMin = 0.0, iMax = 20.0,
                jMin = 0.0, jMax = 20.0,
                vars = Dict("u" => 1, "v" => 2),
                BCL = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                BCU = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                BCD = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0))
            grid = createGrid(gp)

            iDim = grid.params.iDim
            jDim = grid.params.jDim
            for r in 1:iDim
                x = grid.ibasis.data[1, 1].mishPoints[r]
                for l in 1:jDim
                    y = grid.jbasis.data[r, 1].mishPoints[l]
                    idx = (r-1)*jDim + l
                    grid.physical[idx, 1, 1] = x / 20.0
                    grid.physical[idx, 2, 1] = y / 20.0
                end
            end
            orig_u = copy(grid.physical[:, 1, 1])
            orig_v = copy(grid.physical[:, 2, 1])

            spectralTransform!(grid)
            gridTransform!(grid)

            @test maximum(abs.(grid.physical[:, 1, 1] .- orig_u)) < 1e-3
            @test maximum(abs.(grid.physical[:, 2, 1] .- orig_v)) < 1e-3
        end

        @testset "RR derivative accuracy ∂/∂i of f(x,y)=x" begin
            # f(x,y) = x/xmax  →  ∂f/∂i = 1/xmax everywhere; ∂f/∂j = 0
            xmax = 30.0
            gp = SpringsteelGridParameters(
                geometry  = "RR",
                num_cells = 10,
                iMin = 0.0, iMax = xmax,
                jMin = 0.0, jMax = xmax,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCU = Dict("u" => CubicBSpline.R0),
                BCD = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)
            iDim = grid.params.iDim
            jDim = grid.params.jDim
            for r in 1:iDim
                x = grid.ibasis.data[1, 1].mishPoints[r]
                for l in 1:jDim
                    grid.physical[(r-1)*jDim + l, 1, 1] = x / xmax
                end
            end

            spectralTransform!(grid)
            gridTransform!(grid)

            # ∂f/∂i ≈ 1/xmax at interior points
            @test maximum(abs.(grid.physical[:, 1, 2] .- (1.0/xmax))) < 0.05
            # ∂f/∂j ≈ 0
            @test maximum(abs.(grid.physical[:, 1, 4])) < 0.05
        end

        @testset "RR derivative accuracy ∂/∂j of f(x,y)=y" begin
            # f(x,y) = y/ymax  →  ∂f/∂j = 1/ymax; ∂f/∂i = 0
            ymax = 30.0
            gp = SpringsteelGridParameters(
                geometry  = "RR",
                num_cells = 10,
                iMin = 0.0, iMax = ymax,
                jMin = 0.0, jMax = ymax,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCU = Dict("u" => CubicBSpline.R0),
                BCD = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)
            iDim = grid.params.iDim
            jDim = grid.params.jDim
            for r in 1:iDim
                for l in 1:jDim
                    y = grid.jbasis.data[r, 1].mishPoints[l]
                    grid.physical[(r-1)*jDim + l, 1, 1] = y / ymax
                end
            end

            spectralTransform!(grid)
            gridTransform!(grid)

            # ∂f/∂j ≈ 1/ymax
            @test maximum(abs.(grid.physical[:, 1, 4] .- (1.0/ymax))) < 0.05
            # ∂f/∂i ≈ 0
            @test maximum(abs.(grid.physical[:, 1, 2])) < 0.05
        end

        @testset "RZ Spline×Chebyshev roundtrip (quadratic)" begin
            # f(x,z) = (x/xmax)^2 * (z/zmax)
            num_cells = 8
            kDim = 15
            xmax = 40.0; zmax = 20.0
            gp = SpringsteelGridParameters(
                geometry  = "RZ",
                num_cells = num_cells,
                iMin = 0.0, iMax = xmax,
                kMin = 0.0, kMax = zmax,
                kDim = kDim,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCB = Dict("u" => Chebyshev.R0),
                BCT = Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)

            iDim = grid.params.iDim
            for r in 1:iDim
                x = grid.ibasis.data[1, 1].mishPoints[r]
                for z in 1:kDim
                    zp = grid.kbasis.data[1].mishPoints[z]
                    grid.physical[(r-1)*kDim + z, 1, 1] = (x/xmax)^2 * (zp/zmax)
                end
            end
            original = copy(grid.physical[:, 1, 1])

            spectralTransform!(grid)
            @test maximum(abs.(grid.spectral[:, 1])) > 1e-10  # non-trivial

            gridTransform!(grid)
            max_err = maximum(abs.(grid.physical[:, 1, 1] .- original))
            @test max_err < 1e-3

            # ∂/∂i and ∂/∂k should be non-trivial
            @test maximum(abs.(grid.physical[:, 1, 2])) > 1e-10
            @test maximum(abs.(grid.physical[:, 1, 4])) > 1e-10
        end

        @testset "RZ multi-variable roundtrip" begin
            kDim = 12
            gp = SpringsteelGridParameters(
                geometry  = "RZ",
                num_cells = 6,
                iMin = 0.0, iMax = 30.0,
                kMin = 0.0, kMax = 15.0,
                kDim = kDim,
                vars = Dict("u" => 1, "v" => 2),
                BCL = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                BCB = Dict("u" => Chebyshev.R0, "v" => Chebyshev.R0),
                BCT = Dict("u" => Chebyshev.R0, "v" => Chebyshev.R0))
            grid = createGrid(gp)

            iDim = grid.params.iDim
            for r in 1:iDim
                x = grid.ibasis.data[1, 1].mishPoints[r]
                for z in 1:kDim
                    zp = grid.kbasis.data[1].mishPoints[z]
                    idx = (r-1)*kDim + z
                    grid.physical[idx, 1, 1] = x / 30.0
                    grid.physical[idx, 2, 1] = zp / 15.0
                end
            end
            orig_u = copy(grid.physical[:, 1, 1])
            orig_v = copy(grid.physical[:, 2, 1])

            spectralTransform!(grid)
            gridTransform!(grid)

            @test maximum(abs.(grid.physical[:, 1, 1] .- orig_u)) < 1e-3
            @test maximum(abs.(grid.physical[:, 2, 1] .- orig_v)) < 1e-3
        end

        @testset "RZ derivative accuracy ∂/∂i of f(x,z)=x" begin
            # f(x,z) = x/xmax  →  ∂f/∂i = 1/xmax; ∂f/∂k = 0
            xmax = 30.0
            kDim = 15
            gp = SpringsteelGridParameters(
                geometry  = "RZ",
                num_cells = 10,
                iMin = 0.0, iMax = xmax,
                kMin = 0.0, kMax = 15.0,
                kDim = kDim,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCB = Dict("u" => Chebyshev.R0),
                BCT = Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)
            iDim = grid.params.iDim
            for r in 1:iDim
                x = grid.ibasis.data[1, 1].mishPoints[r]
                for z in 1:kDim
                    grid.physical[(r-1)*kDim + z, 1, 1] = x / xmax
                end
            end

            spectralTransform!(grid)
            gridTransform!(grid)

            @test maximum(abs.(grid.physical[:, 1, 2] .- (1.0/xmax))) < 0.05
            @test maximum(abs.(grid.physical[:, 1, 4])) < 0.05
        end

        @testset "RZ derivative accuracy ∂/∂k of f(x,z)=z" begin
            # f(x,z) = z/zmax  →  ∂f/∂k = 1/zmax; ∂f/∂i = 0
            zmax = 15.0
            kDim = 15
            gp = SpringsteelGridParameters(
                geometry  = "RZ",
                num_cells = 8,
                iMin = 0.0, iMax = 30.0,
                kMin = 0.0, kMax = zmax,
                kDim = kDim,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCB = Dict("u" => Chebyshev.R0),
                BCT = Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)
            iDim = grid.params.iDim
            for r in 1:iDim
                for z in 1:kDim
                    zp = grid.kbasis.data[1].mishPoints[z]
                    grid.physical[(r-1)*kDim + z, 1, 1] = zp / zmax
                end
            end

            spectralTransform!(grid)
            gridTransform!(grid)

            @test maximum(abs.(grid.physical[:, 1, 4] .- (1.0/zmax))) < 0.05
            @test maximum(abs.(grid.physical[:, 1, 2])) < 0.05
        end

    end  # 2D Cartesian Transforms

    @testset "2D Cylindrical Transforms" begin

        @testset "RL roundtrip — wavenumber-0 (f(r,λ) = r)" begin
            gp = SpringsteelGridParameters(
                geometry  = "RL",
                num_cells = 10,
                iMin = 0.0, iMax = 100.0,
                vars = Dict("u" => 1),
                BCL  = Dict("u" => CubicBSpline.R0),
                BCR  = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            # f(r, λ) = r  — only wavenumber 0, constant in λ
            iDim = grid.params.iDim
            l2 = 0
            for r in 1:iDim
                ri      = r + grid.params.patchOffsetL
                lpoints = 4 + 4*ri
                l1 = l2 + 1
                l2 = l1 + lpoints - 1
                x  = grid.ibasis.data[1, 1].mishPoints[r]
                grid.physical[l1:l2, 1, 1] .= x
            end
            original = copy(grid.physical[:, 1, 1])

            spectralTransform!(grid)
            @test maximum(abs.(grid.spectral[:, 1])) > 1e-10   # non-trivial

            gridTransform!(grid)
            max_err = maximum(abs.(grid.physical[:, 1, 1] .- original))
            @test max_err < 1e-3

            # Radial derivative of f(r)=r should be ≈ 1
            @test maximum(abs.(grid.physical[:, 1, 2] .- 1.0)) < 0.05
            # Second radial derivative ≈ 0
            @test maximum(abs.(grid.physical[:, 1, 3])) < 0.05
        end

        @testset "RL roundtrip — wavenumber-1 (f(r,λ) = r·cos λ)" begin
            gp = SpringsteelGridParameters(
                geometry  = "RL",
                num_cells = 10,
                iMin = 0.0, iMax = 100.0,
                vars = Dict("u" => 1),
                BCL  = Dict("u" => CubicBSpline.R0),
                BCR  = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            # f(r, λ) = r * cos(λ) — wavenumber-1 mode
            iDim = grid.params.iDim
            l2 = 0
            for r in 1:iDim
                ri      = r + grid.params.patchOffsetL
                lpoints = 4 + 4*ri
                l1 = l2 + 1
                l2 = l1 + lpoints - 1
                x  = grid.ibasis.data[1, 1].mishPoints[r]
                λs = grid.jbasis.data[r, 1].mishPoints
                grid.physical[l1:l2, 1, 1] .= x .* cos.(λs)
            end
            original = copy(grid.physical[:, 1, 1])

            spectralTransform!(grid)
            gridTransform!(grid)

            max_err = maximum(abs.(grid.physical[:, 1, 1] .- original))
            @test max_err < 1e-2

            # Azimuthal derivative ∂/∂λ of r·cos(λ) = -r·sin(λ) — non-trivial
            @test maximum(abs.(grid.physical[:, 1, 4])) > 1e-6
        end

        @testset "RL multi-variable roundtrip" begin
            gp = SpringsteelGridParameters(
                geometry  = "RL",
                num_cells = 8,
                iMin = 0.0, iMax = 80.0,
                vars = Dict("u" => 1, "v" => 2),
                BCL  = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                BCR  = Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0))
            grid = createGrid(gp)

            iDim = grid.params.iDim
            l2 = 0
            for r in 1:iDim
                ri      = r + grid.params.patchOffsetL
                lpoints = 4 + 4*ri
                l1 = l2 + 1
                l2 = l1 + lpoints - 1
                x = grid.ibasis.data[1, 1].mishPoints[r]
                grid.physical[l1:l2, 1, 1] .= x
                grid.physical[l1:l2, 2, 1] .= x .* 2.0
            end
            orig_u = copy(grid.physical[:, 1, 1])
            orig_v = copy(grid.physical[:, 2, 1])

            spectralTransform!(grid)
            gridTransform!(grid)

            @test maximum(abs.(grid.physical[:, 1, 1] .- orig_u)) < 1e-3
            @test maximum(abs.(grid.physical[:, 2, 1] .- orig_v)) < 1e-3
        end

        @testset "RL derivative accuracy ∂/∂i of f(r,λ) = r" begin
            # ∂(r)/∂r = 1; ∂²(r)/∂r² = 0; ∂(r)/∂λ = 0
            gp = SpringsteelGridParameters(
                geometry  = "RL",
                num_cells = 12,
                iMin = 0.0, iMax = 100.0,
                vars = Dict("u" => 1),
                BCL  = Dict("u" => CubicBSpline.R0),
                BCR  = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            iDim = grid.params.iDim
            l2 = 0
            for r in 1:iDim
                ri      = r + grid.params.patchOffsetL
                lpoints = 4 + 4*ri
                l1 = l2 + 1
                l2 = l1 + lpoints - 1
                x = grid.ibasis.data[1, 1].mishPoints[r]
                grid.physical[l1:l2, 1, 1] .= x / 100.0
            end

            spectralTransform!(grid)
            gridTransform!(grid)

            # ∂f/∂r = 1/100
            @test maximum(abs.(grid.physical[:, 1, 2] .- 1.0/100.0)) < 0.05
            # ∂f/∂λ = 0
            @test maximum(abs.(grid.physical[:, 1, 4])) < 0.05
        end

    end  # 2D Cylindrical Transforms

    @testset "3D Transforms" begin

        # ── 3D Cartesian Spline×Spline×Spline (RRR) ──────────────────────────
        @testset "RRR roundtrip f(x,y,z) = x²·y·z" begin
            gp = SpringsteelGridParameters(
                geometry  = "RRR",
                num_cells = 4,
                iMin = 0.0, iMax = 50.0,
                jMin = 0.0, jMax = 50.0,
                kMin = 0.0, kMax = 50.0,
                vars = Dict("u" => 1),
                BCL  = Dict("u" => CubicBSpline.R0),
                BCR  = Dict("u" => CubicBSpline.R0),
                BCU  = Dict("u" => CubicBSpline.R0),
                BCD  = Dict("u" => CubicBSpline.R0),
                BCB  = Dict("u" => CubicBSpline.R0),
                BCT  = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)
            @test typeof(grid) == SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, SplineBasisArray}

            iDim = grid.params.iDim
            jDim = grid.params.jDim
            kDim = grid.params.kDim
            for r in 1:iDim, l in 1:jDim, z in 1:kDim
                xi = grid.ibasis.data[1, 1, 1].mishPoints[r]
                yj = grid.jbasis.data[r, 1, 1].mishPoints[l]
                zk = grid.kbasis.data[r, l, 1].mishPoints[z]
                flat = (r-1)*jDim*kDim + (l-1)*kDim + z
                grid.physical[flat, 1, 1] = xi^2 * yj * zk
            end
            original = copy(grid.physical[:, 1, 1])

            spectralTransform!(grid)
            @test maximum(abs.(grid.spectral[:, 1])) > 1e-10
            gridTransform!(grid)

            max_err = maximum(abs.(grid.physical[:, 1, 1] .- original))
            @test max_err < 1e-2
        end

        @testset "RRR derivative accuracy — BUG-2 and BUG-3 regression" begin
            # Bug-2: derivative slot 4 (∂/∂j) was dimension-mismatch
            # Bug-3: splineBuffer_l stale across r loop
            # Test with f(x,y,z) = y² so ∂f/∂j = 2y, ∂²f/∂j² = 2, and ∂f/∂i = ∂f/∂k = 0
            gp = SpringsteelGridParameters(
                geometry  = "RRR",
                num_cells = 4,
                iMin = 0.0, iMax = 50.0,
                jMin = 0.0, jMax = 50.0,
                kMin = 0.0, kMax = 50.0,
                vars = Dict("u" => 1),
                BCL  = Dict("u" => CubicBSpline.R0),
                BCR  = Dict("u" => CubicBSpline.R0),
                BCU  = Dict("u" => CubicBSpline.R0),
                BCD  = Dict("u" => CubicBSpline.R0),
                BCB  = Dict("u" => CubicBSpline.R0),
                BCT  = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            iDim = grid.params.iDim
            jDim = grid.params.jDim
            kDim = grid.params.kDim

            # f = y² — only y-dependence
            for r in 1:iDim, l in 1:jDim, z in 1:kDim
                yj   = grid.jbasis.data[r, 1, 1].mishPoints[l]
                flat = (r-1)*jDim*kDim + (l-1)*kDim + z
                grid.physical[flat, 1, 1] = yj^2
            end

            spectralTransform!(grid)
            gridTransform!(grid)

            # ∂f/∂i = 0  (slot 2)
            @test maximum(abs.(grid.physical[:, 1, 2])) < 0.1
            # ∂f/∂j = 2y  (slot 4)  — BUG-2 fix test
            for r in 1:iDim, l in 1:jDim, z in 1:kDim
                yj   = grid.jbasis.data[r, 1, 1].mishPoints[l]
                flat = (r-1)*jDim*kDim + (l-1)*kDim + z
                @test abs(grid.physical[flat, 1, 4] - 2.0*yj) < 0.2
            end
            # ∂f/∂k = 0  (slot 6)
            @test maximum(abs.(grid.physical[:, 1, 6])) < 0.1
        end

        @testset "RRR BUG-3 regression — radial variation preserved" begin
            # BUG-3: splineBuffer_l was overwritten per r, so only last r survived.
            # Test with f(x,y,z) = x·y·z so values at r=1 must differ from r=rDim.
            gp = SpringsteelGridParameters(
                geometry  = "RRR",
                num_cells = 4,
                iMin = 1.0, iMax = 50.0,
                jMin = 1.0, jMax = 50.0,
                kMin = 1.0, kMax = 50.0,
                vars = Dict("u" => 1),
                BCL  = Dict("u" => CubicBSpline.R0),
                BCR  = Dict("u" => CubicBSpline.R0),
                BCU  = Dict("u" => CubicBSpline.R0),
                BCD  = Dict("u" => CubicBSpline.R0),
                BCB  = Dict("u" => CubicBSpline.R0),
                BCT  = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            iDim = grid.params.iDim
            jDim = grid.params.jDim
            kDim = grid.params.kDim

            for r in 1:iDim, l in 1:jDim, z in 1:kDim
                xi   = grid.ibasis.data[1, 1, 1].mishPoints[r]
                yj   = grid.jbasis.data[r, 1, 1].mishPoints[l]
                zk   = grid.kbasis.data[r, l, 1].mishPoints[z]
                flat = (r-1)*jDim*kDim + (l-1)*kDim + z
                grid.physical[flat, 1, 1] = xi * yj * zk
            end
            original = copy(grid.physical[:, 1, 1])

            spectralTransform!(grid)
            gridTransform!(grid)

            # Values at r=1 and r=iDim must differ (BUG-3 caused them to be identical)
            r1_val = grid.physical[1*jDim*kDim - jDim*kDim + 1, 1, 1]  # r=1, l=1, z=1
            rN_val = grid.physical[(iDim-1)*jDim*kDim + 1, 1, 1]        # r=iDim, l=1, z=1
            @test abs(r1_val - rN_val) > 1.0   # must differ significantly

            # Also check roundtrip accuracy
            max_err = maximum(abs.(grid.physical[:, 1, 1] .- original))
            @test max_err < 1e-2
        end

        # ── 3D Cylindrical Spline×Fourier×Chebyshev (RLZ) ────────────────────
        @testset "RLZ roundtrip f(r,λ,z) = r·cos(λ)·z" begin
            gp = SpringsteelGridParameters(
                geometry  = "RLZ",
                num_cells = 4,
                iMin = 0.0, iMax = 60.0,
                kMin = 0.0, kMax = 10.0,
                kDim = 8,
                vars = Dict("u" => 1),
                BCL  = Dict("u" => CubicBSpline.R0),
                BCR  = Dict("u" => CubicBSpline.R0),
                BCB  = Dict("u" => Chebyshev.R0),
                BCT  = Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)
            @test typeof(grid) == SpringsteelGrid{CylindricalGeometry, SplineBasisArray, FourierBasisArray, ChebyshevBasisArray}

            iDim = grid.params.iDim
            kDim = grid.params.kDim
            zi   = 1
            for r in 1:iDim
                ri      = r + grid.params.patchOffsetL
                lpoints = 4 + 4*ri
                r_m     = grid.ibasis.data[1, 1].mishPoints[r]
                for l in 1:lpoints
                    l_m = grid.jbasis.data[r, 1].mishPoints[l]
                    for z in 1:kDim
                        z_m = grid.kbasis.data[1].mishPoints[z]
                        grid.physical[zi + (l-1)*kDim + (z-1), 1, 1] = r_m * cos(l_m) * z_m
                    end
                end
                zi += lpoints * kDim
            end
            original = copy(grid.physical[:, 1, 1])

            spectralTransform!(grid)
            @test maximum(abs.(grid.spectral[:, 1])) > 1e-10
            gridTransform!(grid)

            max_err = maximum(abs.(grid.physical[:, 1, 1] .- original))
            @test max_err < 1e-2
        end

        @testset "RLZ BUG-1 regression — patchOffsetL kDim fix" begin
            # BUG-1: gridTransform used kDim = iDim (wrong) instead of iDim + patchOffsetL.
            # With spectralIndexL=2 (patchOffsetL=3), z > 1 levels are read from wrong
            # spectral offsets. Verify roundtrip with Chebyshev z-variation.
            gp = SpringsteelGridParameters(
                geometry       = "RLZ",
                num_cells      = 4,
                spectralIndexL = 2,   # → patchOffsetL = 3
                iMin = 0.0, iMax = 40.0,
                kMin = 0.0, kMax = 10.0,
                kDim = 8,
                vars = Dict("u" => 1),
                BCL  = Dict("u" => CubicBSpline.R0),
                BCR  = Dict("u" => CubicBSpline.R0),
                BCB  = Dict("u" => Chebyshev.R0),
                BCT  = Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)
            @test grid.params.patchOffsetL == 3   # confirm setup

            iDim = grid.params.iDim
            kDim = grid.params.kDim
            zi   = 1
            for r in 1:iDim
                ri      = r + grid.params.patchOffsetL
                lpoints = 4 + 4*ri
                r_m     = grid.ibasis.data[1, 1].mishPoints[r]
                for l in 1:lpoints
                    l_m = grid.jbasis.data[r, 1].mishPoints[l]
                    for z in 1:kDim
                        z_m = grid.kbasis.data[1].mishPoints[z]
                        grid.physical[zi + (l-1)*kDim + (z-1), 1, 1] = r_m * z_m
                    end
                end
                zi += lpoints * kDim
            end
            original = copy(grid.physical[:, 1, 1])

            spectralTransform!(grid)
            gridTransform!(grid)

            # Without BUG-1 fix, z-dependent values would be wrong for patchOffsetL > 0
            max_err = maximum(abs.(grid.physical[:, 1, 1] .- original))
            @test max_err < 1e-2
        end

        @testset "RLZ radial derivative ∂f/∂r" begin
            gp = SpringsteelGridParameters(
                geometry  = "RLZ",
                num_cells = 6,
                iMin = 0.0, iMax = 100.0,
                kMin = 0.0, kMax = 20.0,
                kDim = 10,
                vars = Dict("u" => 1),
                BCL  = Dict("u" => CubicBSpline.R0),
                BCR  = Dict("u" => CubicBSpline.R0),
                BCB  = Dict("u" => Chebyshev.R0),
                BCT  = Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)

            iDim = grid.params.iDim
            kDim = grid.params.kDim
            zi   = 1
            for r in 1:iDim
                ri      = r + grid.params.patchOffsetL
                lpoints = 4 + 4*ri
                r_m     = grid.ibasis.data[1, 1].mishPoints[r]
                for l in 1:lpoints
                    for z in 1:kDim
                        grid.physical[zi + (l-1)*kDim + (z-1), 1, 1] = r_m / 100.0
                    end
                end
                zi += lpoints * kDim
            end

            spectralTransform!(grid)
            gridTransform!(grid)

            # ∂f/∂r = 1/100
            @test maximum(abs.(grid.physical[:, 1, 2] .- 1.0/100.0)) < 0.05
            # ∂f/∂λ = 0
            @test maximum(abs.(grid.physical[:, 1, 4])) < 0.05
        end

    end  # 3D Transforms

    # ── Regular Grid Transforms (2D/3D) ──────────────────────────────────
    @testset "Regular Grid Transforms" begin

        # ── RL regularGridTransform ───────────────────────────────────────
        @testset "RL getRegularGridpoints" begin
            gp = SpringsteelGridParameters(
                geometry="RL", num_cells=10, iMin=0.0, iMax=100.0,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)
            reg_pts = getRegularGridpoints(grid)

            n_r = grid.params.i_regular_out
            n_λ = grid.params.j_regular_out
            @test size(reg_pts) == (n_r * n_λ, 2)
            # r ranges from iMin to iMax
            @test minimum(reg_pts[:, 1]) ≈ gp.iMin
            @test maximum(reg_pts[:, 1]) ≈ gp.iMax
            # λ ranges in [0, 2π)
            @test minimum(reg_pts[:, 2]) ≈ 0.0
            @test maximum(reg_pts[:, 2]) < 2π
            # r-outer, λ-inner ordering: first n_λ points should have same r
            @test all(reg_pts[1:n_λ, 1] .≈ reg_pts[1, 1])
            # λ should be monotonically increasing within each r block
            @test all(diff(reg_pts[1:n_λ, 2]) .> 0)
        end

        @testset "RL regularGridTransform — axisymmetric f(r,λ) = r" begin
            gp = SpringsteelGridParameters(
                geometry="RL", num_cells=10, iMin=0.0, iMax=100.0,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            # Fill with f(r,λ) = r (axisymmetric, only wavenumber 0)
            iDim = grid.params.iDim
            l2 = 0
            for r in 1:iDim
                ri = r + grid.params.patchOffsetL
                lpoints = 4 + 4*ri
                l1 = l2 + 1
                l2 = l1 + lpoints - 1
                x = grid.ibasis.data[1, 1].mishPoints[r]
                grid.physical[l1:l2, 1, 1] .= x
            end

            spectralTransform!(grid)
            reg_pts = getRegularGridpoints(grid)
            reg_phys = regularGridTransform(grid, reg_pts)

            # Output shape should be (n_r*n_λ, nvars, 5)
            n_r = grid.params.i_regular_out
            n_λ = grid.params.j_regular_out
            @test size(reg_phys) == (n_r * n_λ, 1, 5)

            # Values: f(r,λ) = r — compare against radial coordinate
            max_err_val = 0.0
            for i in 1:n_r
                for j in 1:n_λ
                    idx = (i-1)*n_λ + j
                    r_val = reg_pts[idx, 1]
                    max_err_val = max(max_err_val, abs(reg_phys[idx, 1, 1] - r_val))
                end
            end
            @test max_err_val < 1e-3

            # Radial derivative ∂f/∂r = 1 (slot 2)
            @test maximum(abs.(reg_phys[:, 1, 2] .- 1.0)) < 0.05

            # Second radial derivative ∂²f/∂r² = 0 (slot 3)
            @test maximum(abs.(reg_phys[:, 1, 3])) < 0.05

            # Azimuthal derivatives should be 0 for axisymmetric function (slots 4-5)
            @test maximum(abs.(reg_phys[:, 1, 4])) < 1e-10
            @test maximum(abs.(reg_phys[:, 1, 5])) < 1e-10
        end

        @testset "RL regularGridTransform — wavenumber-1 f(r,λ) = r·cos(λ)" begin
            gp = SpringsteelGridParameters(
                geometry="RL", num_cells=10, iMin=0.0, iMax=100.0,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            # Fill with f(r,λ) = r·cos(λ) — wavenumber-1 mode
            iDim = grid.params.iDim
            l2 = 0
            for r in 1:iDim
                ri = r + grid.params.patchOffsetL
                lpoints = 4 + 4*ri
                l1 = l2 + 1
                l2 = l1 + lpoints - 1
                x = grid.ibasis.data[1, 1].mishPoints[r]
                λs = grid.jbasis.data[r, 1].mishPoints
                grid.physical[l1:l2, 1, 1] .= x .* cos.(λs)
            end

            spectralTransform!(grid)
            reg_pts = getRegularGridpoints(grid)
            reg_phys = regularGridTransform(grid, reg_pts)

            # Values: f(r,λ) = r·cos(λ)
            analytic_vals = [reg_pts[i, 1] * cos(reg_pts[i, 2]) for i in 1:size(reg_pts, 1)]
            max_err_val = maximum(abs.(reg_phys[:, 1, 1] .- analytic_vals))
            @test max_err_val < 1e-2

            # ∂f/∂r = cos(λ) (slot 2)
            analytic_dr = [cos(reg_pts[i, 2]) for i in 1:size(reg_pts, 1)]
            max_err_dr = maximum(abs.(reg_phys[:, 1, 2] .- analytic_dr))
            @test max_err_dr < 0.05

            # ∂f/∂λ = -r·sin(λ) (slot 4)
            analytic_dl = [-reg_pts[i, 1] * sin(reg_pts[i, 2]) for i in 1:size(reg_pts, 1)]
            max_err_dl = maximum(abs.(reg_phys[:, 1, 4] .- analytic_dl))
            @test max_err_dl < 1e-2

            # ∂²f/∂λ² = -r·cos(λ) (slot 5)
            analytic_dll = [-reg_pts[i, 1] * cos(reg_pts[i, 2]) for i in 1:size(reg_pts, 1)]
            max_err_dll = maximum(abs.(reg_phys[:, 1, 5] .- analytic_dll))
            @test max_err_dll < 1e-2
        end

        @testset "RL regularGridTransform — matrix-input wrapper" begin
            gp = SpringsteelGridParameters(
                geometry="RL", num_cells=8, iMin=0.0, iMax=80.0,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            # Fill with f(r,λ) = r
            iDim = grid.params.iDim
            l2 = 0
            for r in 1:iDim
                ri = r + grid.params.patchOffsetL
                lpoints = 4 + 4*ri
                l1 = l2 + 1
                l2 = l1 + lpoints - 1
                x = grid.ibasis.data[1, 1].mishPoints[r]
                grid.physical[l1:l2, 1, 1] .= x
            end

            spectralTransform!(grid)
            reg_pts = getRegularGridpoints(grid)

            # Call with matrix input — should produce same result as vector input
            phys_matrix = regularGridTransform(grid, reg_pts)
            r_pts = sort(unique(reg_pts[:, 1]))
            n_λ = div(size(reg_pts, 1), length(r_pts))
            λ_pts = reg_pts[1:n_λ, 2]
            phys_vectors = regularGridTransform(grid, r_pts, λ_pts)
            @test phys_matrix ≈ phys_vectors
        end

        # ── RLZ regularGridTransform ──────────────────────────────────────
        @testset "RLZ getRegularGridpoints" begin
            gp = SpringsteelGridParameters(
                geometry="RLZ", num_cells=4, iMin=0.0, iMax=60.0,
                kMin=0.0, kMax=10.0, kDim=8,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0),
                BCB=Dict("u" => Chebyshev.R0),
                BCT=Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)
            reg_pts = getRegularGridpoints(grid)

            n_r = grid.params.i_regular_out
            n_λ = grid.params.j_regular_out
            n_z = grid.params.k_regular_out
            @test size(reg_pts) == (n_r * n_λ * n_z, 3)
            # r ranges from iMin to iMax
            @test minimum(reg_pts[:, 1]) ≈ gp.iMin
            @test maximum(reg_pts[:, 1]) ≈ gp.iMax
            # λ in [0, 2π)
            @test minimum(reg_pts[:, 2]) ≈ 0.0
            @test maximum(reg_pts[:, 2]) < 2π
            # z ranges from kMin to kMax
            @test minimum(reg_pts[:, 3]) ≈ gp.kMin
            @test maximum(reg_pts[:, 3]) ≈ gp.kMax
        end

        @testset "RLZ regularGridTransform — axisymmetric f(r,λ,z) = r·z" begin
            gp = SpringsteelGridParameters(
                geometry="RLZ", num_cells=4, iMin=0.0, iMax=60.0,
                kMin=0.0, kMax=10.0, kDim=8,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0),
                BCB=Dict("u" => Chebyshev.R0),
                BCT=Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)

            # Fill with f(r,λ,z) = r·z (axisymmetric)
            iDim = grid.params.iDim
            kDim = grid.params.kDim
            zi = 1
            for r in 1:iDim
                ri = r + grid.params.patchOffsetL
                lpoints = 4 + 4*ri
                r_m = grid.ibasis.data[1, 1].mishPoints[r]
                for l in 1:lpoints
                    for z in 1:kDim
                        z_m = grid.kbasis.data[1].mishPoints[z]
                        grid.physical[zi + (l-1)*kDim + (z-1), 1, 1] = r_m * z_m
                    end
                end
                zi += lpoints * kDim
            end

            spectralTransform!(grid)
            reg_pts = getRegularGridpoints(grid)
            reg_phys = regularGridTransform(grid, reg_pts)

            # Output shape: (n_r*n_λ*n_z, nvars, 7)
            n_r = grid.params.i_regular_out
            n_λ = grid.params.j_regular_out
            n_z = grid.params.k_regular_out
            @test size(reg_phys) == (n_r * n_λ * n_z, 1, 7)

            # Values: f(r,λ,z) = r·z
            analytic_vals = [reg_pts[i, 1] * reg_pts[i, 3] for i in 1:size(reg_pts, 1)]
            max_err_val = maximum(abs.(reg_phys[:, 1, 1] .- analytic_vals))
            @test max_err_val < 0.1

            # ∂f/∂r = z (slot 2)
            analytic_dr = [reg_pts[i, 3] for i in 1:size(reg_pts, 1)]
            max_err_dr = maximum(abs.(reg_phys[:, 1, 2] .- analytic_dr))
            @test max_err_dr < 0.1

            # ∂f/∂λ = 0 (slot 4) — axisymmetric
            @test maximum(abs.(reg_phys[:, 1, 4])) < 1e-6

            # ∂f/∂z = r (slot 6) — excluding z endpoints where Chebyshev derivatives blow up
            # Interior z-points only
            interior_mask = (reg_pts[:, 3] .> gp.kMin + 0.1) .& (reg_pts[:, 3] .< gp.kMax - 0.1)
            if any(interior_mask)
                analytic_dz = [reg_pts[i, 1] for i in 1:size(reg_pts, 1)]
                interior_err = maximum(abs.(reg_phys[interior_mask, 1, 6] .- analytic_dz[interior_mask]))
                @test interior_err < 0.5
            end
        end

        @testset "RLZ regularGridTransform — wavenumber-1 f(r,λ,z) = r·cos(λ)·z" begin
            gp = SpringsteelGridParameters(
                geometry="RLZ", num_cells=4, iMin=0.0, iMax=60.0,
                kMin=0.0, kMax=10.0, kDim=8,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0),
                BCB=Dict("u" => Chebyshev.R0),
                BCT=Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)

            iDim = grid.params.iDim
            kDim = grid.params.kDim
            zi = 1
            for r in 1:iDim
                ri = r + grid.params.patchOffsetL
                lpoints = 4 + 4*ri
                r_m = grid.ibasis.data[1, 1].mishPoints[r]
                for l in 1:lpoints
                    l_m = grid.jbasis.data[r, 1].mishPoints[l]
                    for z in 1:kDim
                        z_m = grid.kbasis.data[1].mishPoints[z]
                        grid.physical[zi + (l-1)*kDim + (z-1), 1, 1] = r_m * cos(l_m) * z_m
                    end
                end
                zi += lpoints * kDim
            end

            spectralTransform!(grid)
            reg_pts = getRegularGridpoints(grid)
            reg_phys = regularGridTransform(grid, reg_pts)

            # Values: f(r,λ,z) = r·cos(λ)·z
            analytic_vals = [reg_pts[i, 1] * cos(reg_pts[i, 2]) * reg_pts[i, 3]
                             for i in 1:size(reg_pts, 1)]
            max_err_val = maximum(abs.(reg_phys[:, 1, 1] .- analytic_vals))
            @test max_err_val < 0.5

            # ∂f/∂λ = -r·sin(λ)·z (slot 4)
            analytic_dl = [-reg_pts[i, 1] * sin(reg_pts[i, 2]) * reg_pts[i, 3]
                           for i in 1:size(reg_pts, 1)]
            max_err_dl = maximum(abs.(reg_phys[:, 1, 4] .- analytic_dl))
            @test max_err_dl < 0.5
        end

        @testset "RLZ regularGridTransform — matrix-input wrapper" begin
            gp = SpringsteelGridParameters(
                geometry="RLZ", num_cells=4, iMin=0.0, iMax=60.0,
                kMin=0.0, kMax=10.0, kDim=8,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0),
                BCB=Dict("u" => Chebyshev.R0),
                BCT=Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)

            # Fill with f(r,λ,z) = r
            iDim = grid.params.iDim
            kDim = grid.params.kDim
            zi = 1
            for r in 1:iDim
                ri = r + grid.params.patchOffsetL
                lpoints = 4 + 4*ri
                r_m = grid.ibasis.data[1, 1].mishPoints[r]
                for l in 1:lpoints
                    for z in 1:kDim
                        grid.physical[zi + (l-1)*kDim + (z-1), 1, 1] = r_m
                    end
                end
                zi += lpoints * kDim
            end

            spectralTransform!(grid)
            reg_pts = getRegularGridpoints(grid)

            # Call with matrix input
            phys_matrix = regularGridTransform(grid, reg_pts)
            # Call with separate vectors
            r_pts = sort(unique(reg_pts[:, 1]))
            λ_pts = sort(unique(reg_pts[:, 2]))
            z_pts = sort(unique(reg_pts[:, 3]))
            phys_vectors = regularGridTransform(grid, r_pts, λ_pts, z_pts)
            @test phys_matrix ≈ phys_vectors
        end

        # ── RR regularGridTransform ───────────────────────────────────────
        @testset "RR getRegularGridpoints" begin
            gp = SpringsteelGridParameters(
                geometry="RR", num_cells=4,
                iMin=0.0, iMax=50.0,
                jMin=0.0, jMax=50.0,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0),
                BCU=Dict("u" => CubicBSpline.R0),
                BCD=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)
            reg_pts = getRegularGridpoints(grid)
            n_i = grid.params.i_regular_out
            n_j = grid.params.j_regular_out
            @test size(reg_pts) == (n_i * n_j, 2)
            @test minimum(reg_pts[:, 1]) ≈ gp.iMin
            @test maximum(reg_pts[:, 1]) ≈ gp.iMax
            @test minimum(reg_pts[:, 2]) ≈ gp.jMin
            @test maximum(reg_pts[:, 2]) ≈ gp.jMax
        end

        @testset "RR regularGridTransform — f(x,y) = x·y" begin
            gp = SpringsteelGridParameters(
                geometry="RR", num_cells=6,
                iMin=1.0, iMax=30.0,
                jMin=1.0, jMax=30.0,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0),
                BCU=Dict("u" => CubicBSpline.R0),
                BCD=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            iDim = grid.params.iDim
            jDim = grid.params.jDim
            for r in 1:iDim
                xi = grid.ibasis.data[1, 1].mishPoints[r]
                for l in 1:jDim
                    yj = grid.jbasis.data[r, 1].mishPoints[l]
                    grid.physical[(r-1)*jDim + l, 1, 1] = xi * yj
                end
            end

            spectralTransform!(grid)
            reg_pts  = getRegularGridpoints(grid)
            reg_phys = regularGridTransform(grid, reg_pts)

            n_i = grid.params.i_regular_out
            n_j = grid.params.j_regular_out
            @test size(reg_phys) == (n_i * n_j, 1, 5)

            # Values: f(x,y) = x·y
            analytic_vals = [reg_pts[i, 1] * reg_pts[i, 2] for i in 1:size(reg_pts, 1)]
            @test maximum(abs.(reg_phys[:, 1, 1] .- analytic_vals)) < 0.5

            # ∂f/∂x = y (slot 2)
            analytic_dx = [reg_pts[i, 2] for i in 1:size(reg_pts, 1)]
            @test maximum(abs.(reg_phys[:, 1, 2] .- analytic_dx)) < 0.5

            # ∂f/∂y = x (slot 4)
            analytic_dy = [reg_pts[i, 1] for i in 1:size(reg_pts, 1)]
            @test maximum(abs.(reg_phys[:, 1, 4] .- analytic_dy)) < 0.5
        end

        @testset "RR regularGridTransform — f(x,y) = x² derivative slots" begin
            gp = SpringsteelGridParameters(
                geometry="RR", num_cells=6,
                iMin=0.0, iMax=30.0,
                jMin=0.0, jMax=30.0,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0),
                BCU=Dict("u" => CubicBSpline.R0),
                BCD=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            iDim = grid.params.iDim
            jDim = grid.params.jDim
            for r in 1:iDim
                xi = grid.ibasis.data[1, 1].mishPoints[r]
                for l in 1:jDim
                    grid.physical[(r-1)*jDim + l, 1, 1] = xi^2
                end
            end

            spectralTransform!(grid)
            reg_pts  = getRegularGridpoints(grid)
            reg_phys = regularGridTransform(grid, reg_pts)

            # ∂f/∂x = 2x (slot 2)
            analytic_dx = [2.0 * reg_pts[i, 1] for i in 1:size(reg_pts, 1)]
            @test maximum(abs.(reg_phys[:, 1, 2] .- analytic_dx)) < 0.5

            # ∂²f/∂x² = 2 (slot 3)
            @test maximum(abs.(reg_phys[:, 1, 3] .- 2.0)) < 0.5

            # ∂f/∂y = 0 (slot 4), ∂²f/∂y² = 0 (slot 5)
            @test maximum(abs.(reg_phys[:, 1, 4])) < 0.1
            @test maximum(abs.(reg_phys[:, 1, 5])) < 0.1
        end

        @testset "RR regularGridTransform — matrix-input wrapper" begin
            gp = SpringsteelGridParameters(
                geometry="RR", num_cells=4,
                iMin=1.0, iMax=30.0,
                jMin=1.0, jMax=30.0,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0),
                BCU=Dict("u" => CubicBSpline.R0),
                BCD=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            iDim = grid.params.iDim
            jDim = grid.params.jDim
            for r in 1:iDim
                xi = grid.ibasis.data[1, 1].mishPoints[r]
                for l in 1:jDim
                    grid.physical[(r-1)*jDim + l, 1, 1] = xi
                end
            end

            spectralTransform!(grid)
            reg_pts = getRegularGridpoints(grid)

            phys_matrix  = regularGridTransform(grid, reg_pts)
            i_pts_vec = sort(unique(reg_pts[:, 1]))
            j_pts_vec = sort(unique(reg_pts[:, 2]))
            phys_vectors = regularGridTransform(grid, i_pts_vec, j_pts_vec)
            @test phys_matrix ≈ phys_vectors
        end

        # ── RZ regularGridTransform ───────────────────────────────────────
        @testset "RZ getRegularGridpoints" begin
            kDim = 12
            gp = SpringsteelGridParameters(
                geometry="RZ", num_cells=4,
                iMin=0.0, iMax=30.0,
                kMin=0.0, kMax=15.0,
                kDim=kDim,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0),
                BCB=Dict("u" => Chebyshev.R0),
                BCT=Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)
            reg_pts = getRegularGridpoints(grid)
            n_i = grid.params.i_regular_out
            n_k = grid.params.k_regular_out
            @test size(reg_pts) == (n_i * n_k, 2)
            @test minimum(reg_pts[:, 1]) ≈ gp.iMin
            @test maximum(reg_pts[:, 1]) ≈ gp.iMax
            @test minimum(reg_pts[:, 2]) ≈ gp.kMin
            @test maximum(reg_pts[:, 2]) ≈ gp.kMax
        end

        @testset "RZ regularGridTransform — f(x,z) = x·z" begin
            kDim = 15
            xmax = 30.0; zmax = 20.0
            gp = SpringsteelGridParameters(
                geometry="RZ", num_cells=6,
                iMin=1.0, iMax=xmax,
                kMin=1.0, kMax=zmax,
                kDim=kDim,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0),
                BCB=Dict("u" => Chebyshev.R0),
                BCT=Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)

            iDim = grid.params.iDim
            for r in 1:iDim
                xi = grid.ibasis.data[1, 1].mishPoints[r]
                for z in 1:kDim
                    zp = grid.kbasis.data[1].mishPoints[z]
                    grid.physical[(r-1)*kDim + z, 1, 1] = xi * zp
                end
            end

            spectralTransform!(grid)
            reg_pts  = getRegularGridpoints(grid)
            reg_phys = regularGridTransform(grid, reg_pts)

            n_i = grid.params.i_regular_out
            n_k = grid.params.k_regular_out
            @test size(reg_phys) == (n_i * n_k, 1, 5)

            # Values: f(x,z) = x·z
            analytic_vals = [reg_pts[i, 1] * reg_pts[i, 2] for i in 1:size(reg_pts, 1)]
            @test maximum(abs.(reg_phys[:, 1, 1] .- analytic_vals)) < 0.5

            # ∂f/∂x = z (slot 2)
            analytic_dx = [reg_pts[i, 2] for i in 1:size(reg_pts, 1)]
            @test maximum(abs.(reg_phys[:, 1, 2] .- analytic_dx)) < 0.5

            # ∂f/∂z = x (slot 4)
            analytic_dz = [reg_pts[i, 1] for i in 1:size(reg_pts, 1)]
            @test maximum(abs.(reg_phys[:, 1, 4] .- analytic_dz)) < 0.5
        end

        @testset "RZ regularGridTransform — f(x,z) = z² derivative slots" begin
            kDim = 15
            gp = SpringsteelGridParameters(
                geometry="RZ", num_cells=6,
                iMin=0.0, iMax=30.0,
                kMin=0.0, kMax=20.0,
                kDim=kDim,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0),
                BCB=Dict("u" => Chebyshev.R0),
                BCT=Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)

            iDim = grid.params.iDim
            for r in 1:iDim
                for z in 1:kDim
                    zp = grid.kbasis.data[1].mishPoints[z]
                    grid.physical[(r-1)*kDim + z, 1, 1] = zp^2
                end
            end

            spectralTransform!(grid)
            reg_pts  = getRegularGridpoints(grid)
            reg_phys = regularGridTransform(grid, reg_pts)

            # ∂f/∂x = 0 (slot 2)
            @test maximum(abs.(reg_phys[:, 1, 2])) < 0.1

            # ∂f/∂z = 2z (slot 4)
            analytic_dz = [2.0 * reg_pts[i, 2] for i in 1:size(reg_pts, 1)]
            @test maximum(abs.(reg_phys[:, 1, 4] .- analytic_dz)) < 1.0

            # ∂²f/∂z² = 2 (slot 5)
            @test maximum(abs.(reg_phys[:, 1, 5] .- 2.0)) < 1.0
        end

        @testset "RZ regularGridTransform — matrix-input wrapper" begin
            kDim = 12
            gp = SpringsteelGridParameters(
                geometry="RZ", num_cells=4,
                iMin=1.0, iMax=30.0,
                kMin=1.0, kMax=15.0,
                kDim=kDim,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0),
                BCB=Dict("u" => Chebyshev.R0),
                BCT=Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)

            iDim = grid.params.iDim
            for r in 1:iDim
                xi = grid.ibasis.data[1, 1].mishPoints[r]
                for z in 1:kDim
                    grid.physical[(r-1)*kDim + z, 1, 1] = xi
                end
            end

            spectralTransform!(grid)
            reg_pts = getRegularGridpoints(grid)

            phys_matrix  = regularGridTransform(grid, reg_pts)
            i_pts_vec = sort(unique(reg_pts[:, 1]))
            k_pts_vec = sort(unique(reg_pts[:, 2]))
            phys_vectors = regularGridTransform(grid, i_pts_vec, k_pts_vec)
            @test phys_matrix ≈ phys_vectors
        end

        # ── RRR regularGridTransform ──────────────────────────────────────
        @testset "RRR getRegularGridpoints" begin
            gp = SpringsteelGridParameters(
                geometry="RRR", num_cells=4,
                iMin=0.0, iMax=50.0,
                jMin=0.0, jMax=50.0,
                kMin=0.0, kMax=50.0,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0),
                BCU=Dict("u" => CubicBSpline.R0),
                BCD=Dict("u" => CubicBSpline.R0),
                BCB=Dict("u" => CubicBSpline.R0),
                BCT=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)
            reg_pts = getRegularGridpoints(grid)

            n_x = grid.params.i_regular_out
            n_y = grid.params.j_regular_out
            n_z = grid.params.k_regular_out
            @test size(reg_pts) == (n_x * n_y * n_z, 3)
            @test minimum(reg_pts[:, 1]) ≈ gp.iMin
            @test maximum(reg_pts[:, 1]) ≈ gp.iMax
            @test minimum(reg_pts[:, 2]) ≈ gp.jMin
            @test maximum(reg_pts[:, 2]) ≈ gp.jMax
            @test minimum(reg_pts[:, 3]) ≈ gp.kMin
            @test maximum(reg_pts[:, 3]) ≈ gp.kMax
        end

        @testset "RRR regularGridTransform — f(x,y,z) = x²·y·z" begin
            gp = SpringsteelGridParameters(
                geometry="RRR", num_cells=4,
                iMin=1.0, iMax=50.0,
                jMin=1.0, jMax=50.0,
                kMin=1.0, kMax=50.0,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0),
                BCU=Dict("u" => CubicBSpline.R0),
                BCD=Dict("u" => CubicBSpline.R0),
                BCB=Dict("u" => CubicBSpline.R0),
                BCT=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            iDim = grid.params.iDim
            jDim = grid.params.jDim
            kDim = grid.params.kDim
            for r in 1:iDim, l in 1:jDim, z in 1:kDim
                xi = grid.ibasis.data[1, 1, 1].mishPoints[r]
                yj = grid.jbasis.data[r, 1, 1].mishPoints[l]
                zk = grid.kbasis.data[r, l, 1].mishPoints[z]
                flat = (r-1)*jDim*kDim + (l-1)*kDim + z
                grid.physical[flat, 1, 1] = xi^2 * yj * zk
            end

            spectralTransform!(grid)
            reg_pts = getRegularGridpoints(grid)
            reg_phys = regularGridTransform(grid, reg_pts)

            n_x = grid.params.i_regular_out
            n_y = grid.params.j_regular_out
            n_z = grid.params.k_regular_out
            @test size(reg_phys) == (n_x * n_y * n_z, 1, 7)

            # Values: f(x,y,z) = x²·y·z
            analytic_vals = [reg_pts[i, 1]^2 * reg_pts[i, 2] * reg_pts[i, 3]
                             for i in 1:size(reg_pts, 1)]
            max_err_val = maximum(abs.(reg_phys[:, 1, 1] .- analytic_vals))
            @test max_err_val < 1.0   # 3D polynomial, relaxed tolerance

            # ∂f/∂x = 2x·y·z (slot 2)
            analytic_dx = [2.0 * reg_pts[i, 1] * reg_pts[i, 2] * reg_pts[i, 3]
                           for i in 1:size(reg_pts, 1)]
            max_err_dx = maximum(abs.(reg_phys[:, 1, 2] .- analytic_dx))
            @test max_err_dx < 1.0

            # ∂f/∂y = x²·z (slot 4)
            analytic_dy = [reg_pts[i, 1]^2 * reg_pts[i, 3]
                           for i in 1:size(reg_pts, 1)]
            max_err_dy = maximum(abs.(reg_phys[:, 1, 4] .- analytic_dy))
            @test max_err_dy < 1.0

            # ∂f/∂z = x²·y (slot 6)
            analytic_dz = [reg_pts[i, 1]^2 * reg_pts[i, 2]
                           for i in 1:size(reg_pts, 1)]
            max_err_dz = maximum(abs.(reg_phys[:, 1, 6] .- analytic_dz))
            @test max_err_dz < 1.0
        end

        @testset "RRR regularGridTransform — f(x,y,z) = y² (derivative slots)" begin
            gp = SpringsteelGridParameters(
                geometry="RRR", num_cells=4,
                iMin=0.0, iMax=50.0,
                jMin=0.0, jMax=50.0,
                kMin=0.0, kMax=50.0,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0),
                BCU=Dict("u" => CubicBSpline.R0),
                BCD=Dict("u" => CubicBSpline.R0),
                BCB=Dict("u" => CubicBSpline.R0),
                BCT=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            iDim = grid.params.iDim
            jDim = grid.params.jDim
            kDim = grid.params.kDim
            for r in 1:iDim, l in 1:jDim, z in 1:kDim
                yj = grid.jbasis.data[r, 1, 1].mishPoints[l]
                flat = (r-1)*jDim*kDim + (l-1)*kDim + z
                grid.physical[flat, 1, 1] = yj^2
            end

            spectralTransform!(grid)
            reg_pts = getRegularGridpoints(grid)
            reg_phys = regularGridTransform(grid, reg_pts)

            # ∂f/∂x = 0 (slot 2)
            @test maximum(abs.(reg_phys[:, 1, 2])) < 0.2

            # ∂f/∂y = 2y (slot 4)
            analytic_dy = [2.0 * reg_pts[i, 2] for i in 1:size(reg_pts, 1)]
            max_err_dy = maximum(abs.(reg_phys[:, 1, 4] .- analytic_dy))
            @test max_err_dy < 0.5

            # ∂²f/∂y² = 2 (slot 5)
            max_err_dyy = maximum(abs.(reg_phys[:, 1, 5] .- 2.0))
            @test max_err_dyy < 0.5

            # ∂f/∂z = 0 (slot 6)
            @test maximum(abs.(reg_phys[:, 1, 6])) < 0.2
        end

        @testset "RRR regularGridTransform — matrix-input wrapper" begin
            gp = SpringsteelGridParameters(
                geometry="RRR", num_cells=4,
                iMin=1.0, iMax=50.0,
                jMin=1.0, jMax=50.0,
                kMin=1.0, kMax=50.0,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0),
                BCU=Dict("u" => CubicBSpline.R0),
                BCD=Dict("u" => CubicBSpline.R0),
                BCB=Dict("u" => CubicBSpline.R0),
                BCT=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            iDim = grid.params.iDim
            jDim = grid.params.jDim
            kDim = grid.params.kDim
            for r in 1:iDim, l in 1:jDim, z in 1:kDim
                xi = grid.ibasis.data[1, 1, 1].mishPoints[r]
                flat = (r-1)*jDim*kDim + (l-1)*kDim + z
                grid.physical[flat, 1, 1] = xi
            end

            spectralTransform!(grid)
            reg_pts = getRegularGridpoints(grid)

            phys_matrix = regularGridTransform(grid, reg_pts)
            x_pts = sort(unique(reg_pts[:, 1]))
            y_pts = sort(unique(reg_pts[:, 2]))
            z_pts = sort(unique(reg_pts[:, 3]))
            phys_vectors = regularGridTransform(grid, x_pts, y_pts, z_pts)
            @test phys_matrix ≈ phys_vectors
        end

        # ── SL regularGridTransform ──────────────────────────────────────
        @testset "SL getRegularGridpoints" begin
            gp = SpringsteelGridParameters(
                geometry="SL", num_cells=8,
                iMin=0.0, iMax=Float64(π),
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)
            reg_pts = getRegularGridpoints(grid)

            n_θ = grid.params.i_regular_out
            n_λ = grid.params.j_regular_out
            @test size(reg_pts) == (n_θ * n_λ, 2)
            # θ spans [0, π]
            @test minimum(reg_pts[:, 1]) ≈ 0.0
            @test maximum(reg_pts[:, 1]) ≈ Float64(π)
            # λ spans [0, 2π)
            @test minimum(reg_pts[:, 2]) ≈ 0.0
            @test maximum(reg_pts[:, 2]) < 2π
            # θ-outer ordering: first n_λ points have same θ
            @test all(reg_pts[1:n_λ, 1] .≈ reg_pts[1, 1])
            # λ monotonically increasing within first θ block
            @test all(diff(reg_pts[1:n_λ, 2]) .> 0)
        end

        @testset "SL regularGridTransform — axisymmetric f(θ,λ) = sin(θ)" begin
            gp = SpringsteelGridParameters(
                geometry="SL", num_cells=10,
                iMin=0.0, iMax=Float64(π),
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            pts = getGridpoints(grid)
            npts = size(pts, 1)
            for n in 1:npts
                grid.physical[n, 1, 1] = sin(pts[n, 1])
            end

            spectralTransform!(grid)
            reg_pts = getRegularGridpoints(grid)
            reg_phys = regularGridTransform(grid, reg_pts)

            n_θ = grid.params.i_regular_out
            n_λ = grid.params.j_regular_out
            @test size(reg_phys) == (n_θ * n_λ, 1, 5)

            # Values: f(θ,λ) = sin(θ)
            analytic_vals = [sin(reg_pts[i, 1]) for i in 1:size(reg_pts, 1)]
            max_err_val = maximum(abs.(reg_phys[:, 1, 1] .- analytic_vals))
            @test max_err_val < 1e-2

            # ∂f/∂θ = cos(θ) (slot 2)
            analytic_dθ = [cos(reg_pts[i, 1]) for i in 1:size(reg_pts, 1)]
            max_err_dθ = maximum(abs.(reg_phys[:, 1, 2] .- analytic_dθ))
            @test max_err_dθ < 0.1

            # ∂f/∂λ = 0 for axisymmetric (slot 4)
            @test maximum(abs.(reg_phys[:, 1, 4])) < 1e-10
            # ∂²f/∂λ² = 0 for axisymmetric (slot 5)
            @test maximum(abs.(reg_phys[:, 1, 5])) < 1e-10
        end

        @testset "SL regularGridTransform — wavenumber-1 f(θ,λ) = sin(θ)·cos(λ)" begin
            gp = SpringsteelGridParameters(
                geometry="SL", num_cells=10,
                iMin=0.0, iMax=Float64(π),
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            pts = getGridpoints(grid)
            npts = size(pts, 1)
            for n in 1:npts
                grid.physical[n, 1, 1] = sin(pts[n, 1]) * cos(pts[n, 2])
            end

            spectralTransform!(grid)
            reg_pts = getRegularGridpoints(grid)
            reg_phys = regularGridTransform(grid, reg_pts)

            # Values: f(θ,λ) = sin(θ)·cos(λ)
            analytic_vals = [sin(reg_pts[i, 1]) * cos(reg_pts[i, 2]) for i in 1:size(reg_pts, 1)]
            max_err_val = maximum(abs.(reg_phys[:, 1, 1] .- analytic_vals))
            @test max_err_val < 1e-2

            # ∂f/∂θ = cos(θ)·cos(λ) (slot 2)
            analytic_dθ = [cos(reg_pts[i, 1]) * cos(reg_pts[i, 2]) for i in 1:size(reg_pts, 1)]
            max_err_dθ = maximum(abs.(reg_phys[:, 1, 2] .- analytic_dθ))
            @test max_err_dθ < 0.1

            # ∂f/∂λ = -sin(θ)·sin(λ) (slot 4)
            analytic_dλ = [-sin(reg_pts[i, 1]) * sin(reg_pts[i, 2]) for i in 1:size(reg_pts, 1)]
            max_err_dλ = maximum(abs.(reg_phys[:, 1, 4] .- analytic_dλ))
            @test max_err_dλ < 1e-2

            # ∂²f/∂λ² = -sin(θ)·cos(λ) (slot 5)
            analytic_dλλ = [-sin(reg_pts[i, 1]) * cos(reg_pts[i, 2]) for i in 1:size(reg_pts, 1)]
            max_err_dλλ = maximum(abs.(reg_phys[:, 1, 5] .- analytic_dλλ))
            @test max_err_dλλ < 1e-2
        end

        @testset "SL regularGridTransform — matrix-input wrapper" begin
            gp = SpringsteelGridParameters(
                geometry="SL", num_cells=8,
                iMin=0.0, iMax=Float64(π),
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)

            pts = getGridpoints(grid)
            for n in 1:size(pts, 1)
                grid.physical[n, 1, 1] = sin(pts[n, 1])
            end
            spectralTransform!(grid)
            reg_pts = getRegularGridpoints(grid)

            # matrix-input call should match vector-input call
            phys_matrix = regularGridTransform(grid, reg_pts)
            θ_pts = sort(unique(reg_pts[:, 1]))
            n_λ = div(size(reg_pts, 1), length(θ_pts))
            λ_pts = reg_pts[1:n_λ, 2]
            phys_vectors = regularGridTransform(grid, θ_pts, λ_pts)
            @test phys_matrix ≈ phys_vectors
        end

        # ── SLZ regularGridTransform ──────────────────────────────────────
        @testset "SLZ getRegularGridpoints" begin
            gp = SpringsteelGridParameters(
                geometry="SLZ", num_cells=6,
                iMin=0.0, iMax=Float64(π),
                kMin=0.0, kMax=10.0, kDim=6,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0),
                BCB=Dict("u" => Chebyshev.R0),
                BCT=Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)
            reg_pts = getRegularGridpoints(grid)

            n_θ = grid.params.i_regular_out
            n_λ = grid.params.j_regular_out
            n_z = grid.params.k_regular_out
            @test size(reg_pts) == (n_θ * n_λ * n_z, 3)
            # θ spans [0, π]
            @test minimum(reg_pts[:, 1]) ≈ 0.0
            @test maximum(reg_pts[:, 1]) ≈ Float64(π)
            # λ spans [0, 2π)
            @test minimum(reg_pts[:, 2]) ≈ 0.0
            @test maximum(reg_pts[:, 2]) < 2π
            # z spans [kMin, kMax]
            @test minimum(reg_pts[:, 3]) ≈ 0.0
            @test maximum(reg_pts[:, 3]) ≈ 10.0
        end

        @testset "SLZ regularGridTransform — axisymmetric f(θ,λ,z) = sin(θ)·z" begin
            gp = SpringsteelGridParameters(
                geometry="SLZ", num_cells=8,
                iMin=0.0, iMax=Float64(π),
                kMin=0.0, kMax=10.0, kDim=8,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0),
                BCB=Dict("u" => Chebyshev.R0),
                BCT=Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)

            pts = getGridpoints(grid)
            npts = size(pts, 1)
            for n in 1:npts
                grid.physical[n, 1, 1] = sin(pts[n, 1]) * pts[n, 3]
            end

            spectralTransform!(grid)
            reg_pts = getRegularGridpoints(grid)
            reg_phys = regularGridTransform(grid, reg_pts)

            n_θ = grid.params.i_regular_out
            n_λ = grid.params.j_regular_out
            n_z = grid.params.k_regular_out
            @test size(reg_phys) == (n_θ * n_λ * n_z, 1, 7)

            # Values: f(θ,λ,z) = sin(θ)·z
            analytic_vals = [sin(reg_pts[i, 1]) * reg_pts[i, 3] for i in 1:size(reg_pts, 1)]
            max_err_val = maximum(abs.(reg_phys[:, 1, 1] .- analytic_vals))
            @test max_err_val < 0.1

            # ∂f/∂θ = cos(θ)·z (slot 2)
            analytic_dθ = [cos(reg_pts[i, 1]) * reg_pts[i, 3] for i in 1:size(reg_pts, 1)]
            max_err_dθ = maximum(abs.(reg_phys[:, 1, 2] .- analytic_dθ))
            @test max_err_dθ < 0.5

            # ∂f/∂λ = 0 for axisymmetric (slot 4)
            @test maximum(abs.(reg_phys[:, 1, 4])) < 1e-6

            # ∂f/∂z = sin(θ) (slot 6) — interior points only (away from z BCs)
            interior_mask = (reg_pts[:, 3] .> 0.5) .& (reg_pts[:, 3] .< 9.5)
            if any(interior_mask)
                analytic_dz = [sin(reg_pts[i, 1]) for i in 1:size(reg_pts, 1)]
                interior_err = maximum(abs.(reg_phys[interior_mask, 1, 6] .- analytic_dz[interior_mask]))
                @test interior_err < 0.5
            end
        end

        @testset "SLZ regularGridTransform — wavenumber-1 f(θ,λ,z) = sin(θ)·cos(λ)·z" begin
            gp = SpringsteelGridParameters(
                geometry="SLZ", num_cells=8,
                iMin=0.0, iMax=Float64(π),
                kMin=0.0, kMax=10.0, kDim=8,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0),
                BCB=Dict("u" => Chebyshev.R0),
                BCT=Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)

            pts = getGridpoints(grid)
            npts = size(pts, 1)
            for n in 1:npts
                grid.physical[n, 1, 1] = sin(pts[n, 1]) * cos(pts[n, 2]) * pts[n, 3]
            end

            spectralTransform!(grid)
            reg_pts = getRegularGridpoints(grid)
            reg_phys = regularGridTransform(grid, reg_pts)

            # Values: f(θ,λ,z) = sin(θ)·cos(λ)·z
            analytic_vals = [sin(reg_pts[i, 1]) * cos(reg_pts[i, 2]) * reg_pts[i, 3]
                             for i in 1:size(reg_pts, 1)]
            max_err_val = maximum(abs.(reg_phys[:, 1, 1] .- analytic_vals))
            @test max_err_val < 0.5

            # ∂f/∂λ = -sin(θ)·sin(λ)·z (slot 4)
            analytic_dλ = [-sin(reg_pts[i, 1]) * sin(reg_pts[i, 2]) * reg_pts[i, 3]
                           for i in 1:size(reg_pts, 1)]
            max_err_dλ = maximum(abs.(reg_phys[:, 1, 4] .- analytic_dλ))
            @test max_err_dλ < 0.5
        end

        @testset "SLZ regularGridTransform — matrix-input wrapper" begin
            gp = SpringsteelGridParameters(
                geometry="SLZ", num_cells=6,
                iMin=0.0, iMax=Float64(π),
                kMin=0.0, kMax=10.0, kDim=6,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0),
                BCB=Dict("u" => Chebyshev.R0),
                BCT=Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)

            pts = getGridpoints(grid)
            for n in 1:size(pts, 1)
                grid.physical[n, 1, 1] = sin(pts[n, 1]) * pts[n, 3]
            end
            spectralTransform!(grid)
            reg_pts = getRegularGridpoints(grid)

            # matrix-input call should match vector-input call
            phys_matrix = regularGridTransform(grid, reg_pts)
            θ_pts = sort(unique(reg_pts[:, 1]))
            λ_pts = sort(unique(reg_pts[:, 2]))
            z_pts = sort(unique(reg_pts[:, 3]))
            phys_vectors = regularGridTransform(grid, θ_pts, λ_pts, z_pts)
            @test phys_matrix ≈ phys_vectors
        end

    end  # Regular Grid Transforms

    @testset "Spherical Transforms" begin

        @testset "2D Spherical roundtrip f(θ,λ)=sin(θ)cos(λ)" begin
            gp = SpringsteelGridParameters(
                geometry  = "SL",
                num_cells = 10,
                iMin      = 0.0,
                iMax      = Float64(π),
                vars      = Dict("u" => 1),
                BCL       = Dict("u" => CubicBSpline.R0),
                BCR       = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)
            pts  = getGridpoints(grid)
            @test size(pts, 2) == 2
            @test size(pts, 1) == grid.params.jDim
            npts = size(pts, 1)
            # f(θ, λ) = sin(θ) * cos(λ) — vanishes at poles, representable exactly
            for n in 1:npts
                theta = pts[n, 1]; lam = pts[n, 2]
                grid.physical[n, 1, 1] = sin(theta) * cos(lam)
            end
            spectralTransform!(grid)
            gridTransform!(grid)
            err = maximum(abs.(grid.physical[1:npts, 1, 1] .-
                               [sin(pts[n,1])*cos(pts[n,2]) for n in 1:npts]))
            @test err < 1e-2
        end

        @testset "2D Spherical pole values vanish" begin
            gp = SpringsteelGridParameters(
                geometry  = "SL",
                num_cells = 8,
                iMin      = 0.0,
                iMax      = Float64(π),
                vars      = Dict("u" => 1),
                BCL       = Dict("u" => CubicBSpline.R0),
                BCR       = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)
            pts  = getGridpoints(grid)
            npts = size(pts, 1)
            # f = sin(θ)*cos(λ) → values near poles ≈ 0 (BC enforced)
            for n in 1:npts
                grid.physical[n, 1, 1] = sin(pts[n,1]) * cos(pts[n,2])
            end
            spectralTransform!(grid)
            gridTransform!(grid)
            # Check that spline BC forces field to 0 at iMin and iMax
            # (gridpoints near 0 and π should have small values)
            min_theta = minimum(pts[:, 1])
            max_theta = maximum(pts[:, 1])
            near_pole_min = [n for n in 1:npts if abs(pts[n,1] - min_theta) < 0.01]
            near_pole_max = [n for n in 1:npts if abs(pts[n,1] - max_theta) < 0.01]
            if !isempty(near_pole_min)
                @test maximum(abs.(grid.physical[near_pole_min, 1, 1])) < 0.5
            end
            if !isempty(near_pole_max)
                @test maximum(abs.(grid.physical[near_pole_max, 1, 1])) < 0.5
            end
        end

        @testset "3D Spherical roundtrip f(θ,λ,z)=sin(θ)cos(λ)z" begin
            gp = SpringsteelGridParameters(
                geometry  = "SLZ",
                num_cells = 10,
                kDim      = 8,
                iMin      = 0.0,
                iMax      = Float64(π),
                kMin      = 0.0,
                kMax      = 20.0,
                vars      = Dict("u" => 1),
                BCL       = Dict("u" => CubicBSpline.R0),
                BCR       = Dict("u" => CubicBSpline.R0),
                BCB       = Dict("u" => Chebyshev.R0),
                BCT       = Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)
            pts  = getGridpoints(grid)
            @test size(pts, 2) == 3
            npts = size(pts, 1)
            for n in 1:npts
                theta = pts[n, 1]; lam = pts[n, 2]; z = pts[n, 3]
                grid.physical[n, 1, 1] = sin(theta) * cos(lam) * z
            end
            spectralTransform!(grid)
            gridTransform!(grid)
            err = maximum(abs.(grid.physical[1:npts, 1, 1] .-
                               [sin(pts[n,1])*cos(pts[n,2])*pts[n,3] for n in 1:npts]))
            @test err < 1e-2
        end

        @testset "3D Spherical radial derivative ∂f/∂θ" begin
            gp = SpringsteelGridParameters(
                geometry  = "SLZ",
                num_cells = 10,
                kDim      = 6,
                iMin      = 0.0,
                iMax      = Float64(π),
                kMin      = 1.0,
                kMax      = 10.0,
                vars      = Dict("u" => 1),
                BCL       = Dict("u" => CubicBSpline.R0),
                BCR       = Dict("u" => CubicBSpline.R0),
                BCB       = Dict("u" => Chebyshev.R0),
                BCT       = Dict("u" => Chebyshev.R0))
            grid = createGrid(gp)
            pts  = getGridpoints(grid)
            npts = size(pts, 1)
            # f = cos(θ)*z → ∂f/∂θ = -sin(θ)*z
            for n in 1:npts
                theta = pts[n,1]; z = pts[n,3]
                grid.physical[n, 1, 1] = cos(theta) * z
            end
            spectralTransform!(grid)
            gridTransform!(grid)
            # check slot 2 = ∂f/∂θ
            err = 0.0
            for n in 1:npts
                theta = pts[n,1]; z = pts[n,3]
                err = max(err, abs(grid.physical[n,1,2] - (-sin(theta)*z)))
            end
            @test err < 0.1
        end

    end  # Spherical Transforms

    @testset "num_deriv_slots dispatch" begin
        # 1D (R_Grid): NoBasisArray × NoBasisArray → 3 slots
        gp_1d = SpringsteelGridParameters(
            geometry = "R",
            num_cells = 4,
            iMin = 0.0, iMax = 1.0,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0))
        grid_1d = createGrid(gp_1d)
        @test size(grid_1d.physical, 3) == 3
        @test Springsteel.num_deriv_slots(grid_1d.jbasis, grid_1d.kbasis) == 3

        # 2D active-j spline (RR_Grid): SplineBasisArray × NoBasisArray → 5 slots
        gp_rr = SpringsteelGridParameters(
            geometry = "RR",
            num_cells = 4,
            iMin = 0.0, iMax = 1.0,
            jMin = 0.0, jMax = 1.0, jDim = 12,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0),
            BCU = Dict("u" => CubicBSpline.R0),
            BCD = Dict("u" => CubicBSpline.R0))
        grid_rr = createGrid(gp_rr)
        @test size(grid_rr.physical, 3) == 5
        @test grid_rr.jbasis isa Springsteel.SplineBasisArray
        @test grid_rr.kbasis isa Springsteel.NoBasisArray
        @test Springsteel.num_deriv_slots(grid_rr.jbasis, grid_rr.kbasis) == 5

        # 2D active-j Fourier (RL_Grid): FourierBasisArray × NoBasisArray → 5 slots
        gp_rl = SpringsteelGridParameters(
            geometry = "RL",
            iMin = 0.0, iMax = 20.0,
            num_cells = 4,
            patchOffsetL = 0,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0))
        grid_rl = createGrid(gp_rl)
        @test size(grid_rl.physical, 3) == 5
        @test grid_rl.jbasis isa Springsteel.FourierBasisArray
        @test grid_rl.kbasis isa Springsteel.NoBasisArray
        @test Springsteel.num_deriv_slots(grid_rl.jbasis, grid_rl.kbasis) == 5

        # 2D active-k (RZ_Grid): NoBasisArray × ChebyshevBasisArray → 5 slots
        gp_rz = SpringsteelGridParameters(
            geometry = "RZ",
            num_cells = 4,
            iMin = 0.0, iMax = 1.0,
            kMin = 0.0, kMax = 1.0, kDim = 8,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0),
            BCB = Dict("u" => Chebyshev.R0),
            BCT = Dict("u" => Chebyshev.R0))
        grid_rz = createGrid(gp_rz)
        @test size(grid_rz.physical, 3) == 5
        @test grid_rz.jbasis isa Springsteel.NoBasisArray
        @test grid_rz.kbasis isa Springsteel.ChebyshevBasisArray
        @test Springsteel.num_deriv_slots(grid_rz.jbasis, grid_rz.kbasis) == 5

        # 3D (RLZ_Grid): FourierBasisArray × ChebyshevBasisArray → 7 slots
        gp_rlz = SpringsteelGridParameters(
            geometry = "RLZ",
            num_cells = 3,
            iMin = 0.0, iMax = 20.0,
            patchOffsetL = 0,
            kMin = 0.0, kMax = 10.0, kDim = 8,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0),
            BCB = Dict("u" => Chebyshev.R0),
            BCT = Dict("u" => Chebyshev.R0))
        grid_rlz = createGrid(gp_rlz)
        @test size(grid_rlz.physical, 3) == 7
        @test grid_rlz.jbasis isa Springsteel.FourierBasisArray
        @test grid_rlz.kbasis isa Springsteel.ChebyshevBasisArray
        @test Springsteel.num_deriv_slots(grid_rlz.jbasis, grid_rlz.kbasis) == 7

        # 3D (RRR_Grid): SplineBasisArray × SplineBasisArray → 7 slots
        gp_rrr = SpringsteelGridParameters(
            geometry = "RRR",
            num_cells = 3,
            iMin = 0.0, iMax = 1.0,
            jMin = 0.0, jMax = 1.0,
            kMin = 0.0, kMax = 1.0,
            vars = Dict("u" => 1),
            BCL = Dict("u" => CubicBSpline.R0),
            BCR = Dict("u" => CubicBSpline.R0),
            BCU = Dict("u" => CubicBSpline.R0),
            BCD = Dict("u" => CubicBSpline.R0),
            BCB = Dict("u" => CubicBSpline.R0),
            BCT = Dict("u" => CubicBSpline.R0))
        grid_rrr = createGrid(gp_rrr)
        @test size(grid_rrr.physical, 3) == 7
        @test Springsteel.num_deriv_slots(grid_rrr.jbasis, grid_rrr.kbasis) == 7
    end  # num_deriv_slots dispatch

    # ────────────────────────────────────────────────────────────────────────
    # SpringsteelGrid Tiling
    # ────────────────────────────────────────────────────────────────────────


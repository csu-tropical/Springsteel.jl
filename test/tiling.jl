    @testset "SpringsteelGrid Tiling" begin

        @testset "1D Cartesian tiling" begin
            # Use geometry="R" which routes to SpringsteelGrid{CartesianGeometry, ...}
            gp = SpringsteelGridParameters(
                geometry = "R",
                num_cells = 12,
                iMin = 0.0, iMax = 100.0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0))
            patch = createGrid(gp)
            @test patch isa SpringsteelGrid

            # ── calcTileSizes ───────────────────────────────────────────────
            tiles = calcTileSizes(patch, 4)
            @test length(tiles) == 4

            # Each tile has ≥ 3 cells (≥ 9 gridpoints)
            for tile in tiles
                @test tile.params.num_cells >= 3
                @test tile.params.iDim >= 9
            end

            # Tile boundaries cover full domain
            @test tiles[1].params.iMin ≈ patch.params.iMin
            @test tiles[end].params.iMax ≈ patch.params.iMax

            # Cell counts sum to patch total
            @test sum(t.params.num_cells for t in tiles) == patch.params.num_cells

            # spectralIndexL continuity
            @test tiles[1].params.spectralIndexL == 1
            for i in 1:length(tiles)-1
                @test tiles[i+1].params.spectralIndexL ==
                      tiles[i].params.spectralIndexL + tiles[i].params.num_cells
            end

            # Too many tiles should throw DomainError (12 cells, 5 tiles → some tile < 9 gridpoints)
            @test_throws DomainError calcTileSizes(patch, 5)

            # ── calcPatchMap ────────────────────────────────────────────────
            patchMap = calcPatchMap(patch, tiles[1])
            @test patchMap isa SparseMatrixCSC

            siL = tiles[1].params.spectralIndexL
            siR_inner = tiles[1].params.spectralIndexR - 3   # inner (non-halo) end
            # patch map marks the inner region only
            @test count(!iszero, patchMap) == (siR_inner - siL + 1) * length(tiles[1].params.vars)

            # ── calcHaloMap ─────────────────────────────────────────────────
            haloMap = calcHaloMap(patch, tiles[1], tiles[2])
            @test haloMap isa SparseMatrixCSC
            @test count(!iszero, haloMap) == 3   # 3-row halo for 1 variable

            # ── num_columns ─────────────────────────────────────────────────
            @test num_columns(patch) >= 1

            # ── allocateSplineBuffer ─────────────────────────────────────────
            buf = allocateSplineBuffer(tiles[1])
            @test isa(buf, Array)
            @test length(buf) > 0

            # ── getBorderSpectral ────────────────────────────────────────────
            border = getBorderSpectral(tiles[1])
            @test border isa SparseMatrixCSC

            biL = tiles[1].params.spectralIndexR - 2
            biR = tiles[1].params.spectralIndexR
            # With sentinel spectral values, verify halo extraction
            tiles[1].spectral[:, 1] .= collect(1.0:tiles[1].params.b_iDim)
            border2 = getBorderSpectral(tiles[1])
            @test nnz(border2) == 3   # exactly 3 non-zeros for 1 variable
            tiL = tiles[1].params.b_iDim - 2
            @test Vector(border2[biL:biR, 1]) ≈ collect(Float64, tiL:tiles[1].params.b_iDim)

            # ── sumSpectralTile! ─────────────────────────────────────────────
            tiles[1].spectral .= 1.0
            patch.spectral .= 0.0
            sumSpectralTile!(patch, tiles[1])
            sR = tiles[1].params.spectralIndexR
            @test all(patch.spectral[siL:sR, :] .== 1.0)
            @test all(patch.spectral[sR+1:end, :] .== 0.0)

            # ── setSpectralTile! ─────────────────────────────────────────────
            patch.spectral .= 99.0
            tiles[1].spectral .= 2.0
            setSpectralTile!(patch, tiles[1])
            @test all(patch.spectral[siL:sR, :] .== 2.0)
            @test all(patch.spectral[1:siL-1, :] .== 0.0)
            @test all(patch.spectral[sR+1:end, :] .== 0.0)

            # ── gridTransform! on tile ───────────────────────────────────────
            # Use patch spectral to populate tile: let tile inherit patch spectral slice
            gp2 = SpringsteelGridParameters(
                geometry = "R",
                num_cells = 3,
                iMin = 0.0, iMax = 25.0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0))
            tile_grid = createGrid(gp2)
            @test tile_grid isa SpringsteelGrid
            gridTransform!(tile_grid)   # should not error on zero spectral
            @test size(tile_grid.physical, 1) == tile_grid.params.iDim
        end

        @testset "2D Cylindrical tiling (RL)" begin
            gp = SpringsteelGridParameters(
                geometry = "RL",
                num_cells = 12,
                iMin = 0.0, iMax = 100.0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0))
            patch = createGrid(gp)

            tiles = calcTileSizes(patch, 4)
            @test length(tiles) == 4

            for tile in tiles
                @test tile.params.num_cells >= 3
                @test tile.params.iDim >= 9
            end

            @test tiles[1].params.iMin ≈ patch.params.iMin
            @test tiles[end].params.iMax ≈ patch.params.iMax
            @test sum(t.params.num_cells for t in tiles) == patch.params.num_cells

            # calcPatchMap / calcHaloMap
            patchMap = calcPatchMap(patch, tiles[1])
            @test patchMap isa SparseMatrixCSC

            haloMap = calcHaloMap(patch, tiles[1], tiles[2])
            @test haloMap isa SparseMatrixCSC
            @test count(!iszero, haloMap) >= 1   # at least some non-zeros

            # num_columns / allocateSplineBuffer
            @test num_columns(patch) >= 0
            buf = allocateSplineBuffer(tiles[1])
            @test isa(buf, Array)
            @test length(buf) > 0

            # getBorderSpectral — nnz = 3 * nvars * complete_blocks_in_spectral
            # RL allocates b_jDim rows (not b_iDim*(1+2*kDim)), so the number of
            # complete b_iDim-row blocks that fit is div(n, b_iDim), not 1+2*kDim.
            nvars_bs = length(tiles[1].params.vars)
            n_bs     = size(tiles[1].spectral, 1)
            b_bs     = tiles[1].params.b_iDim
            tiles[1].spectral .= 1.0
            border = getBorderSpectral(tiles[1])
            @test border isa SparseMatrixCSC
            @test nnz(border) == 3 * nvars_bs * div(n_bs, b_bs)

            # Too many tiles
            @test_throws DomainError calcTileSizes(patch, 5)

            # ── splineTransform! / tileTransform! round-trip ───────────────
            @testset "RL distributed pipeline (splineTransform! / tileTransform!)" begin
                # patch is RL, 12 cells, r ∈ [0, 100], 1 variable "u".
                # Fill physical with the axisymmetric function f(r) = sin(π·r/100),
                # round-trip it through the spectral and tile pipelines, and
                # verify the recovered values and derivatives.

                pts_rl  = getGridpoints(patch)            # (jDim, 2)  cols: [r, λ]
                n_rl    = size(pts_rl, 1)
                rMax_rl = patch.params.iMax                # 100.0

                # --- seed physical with f(r) = sin(π·r/rMax) ----------------
                patch.physical .= 0.0
                for i in 1:n_rl
                    r = pts_rl[i, 1]
                    patch.physical[i, 1, 1] = sin(π * r / rMax_rl)
                end

                # physical → B-coefficients in patch.spectral
                spectralTransform!(patch)

                # Copy B-coefficients into a SharedArray (simulates distributed use)
                sharedSpectral_rl = SharedArray{Float64}(size(patch.spectral))
                sharedSpectral_rl[:, :] .= patch.spectral

                # splineTransform!: B (in sharedSpectral) → A (into patch.spectral)
                splineTransform!(sharedSpectral_rl, patch)

                # tileTransform!: A-coefficients → reconstructed physical
                physical_rl = zeros(Float64, size(patch.physical))
                tileTransform!(sharedSpectral_rl, patch, physical_rl, patch.spectral)

                # Analytic values and derivatives
                analytic_val = [sin(π * pts_rl[i, 1] / rMax_rl)          for i in 1:n_rl]
                analytic_dr  = [(π / rMax_rl) * cos(π * pts_rl[i, 1] / rMax_rl) for i in 1:n_rl]

                # Values: smooth function over 12 cells; cubic B-spline accuracy
                @test maximum(abs.(physical_rl[:, 1, 1] .- analytic_val)) < 5e-4

                # ∂f/∂r correctness
                @test maximum(abs.(physical_rl[:, 1, 2] .- analytic_dr))  < 0.05

                # Axisymmetric ⟹ azimuthal derivative ≈ 0
                @test maximum(abs.(physical_rl[:, 1, 4])) < 1e-10

                # All physical values must be finite
                @test all(isfinite, physical_rl)
            end  # RL distributed pipeline
        end

        @testset "3D Cylindrical tiling (RLZ)" begin
            gp = SpringsteelGridParameters(
                geometry = "RLZ",
                num_cells = 9,
                iMin = 0.0, iMax = 75.0,
                kMin = 0.0, kMax = 10.0, kDim = 6,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCB = Dict("u" => Chebyshev.R0),
                BCT = Dict("u" => Chebyshev.R0))
            patch = createGrid(gp)

            # ── calcTileSizes ───────────────────────────────────────────────
            tiles = calcTileSizes(patch, 3)
            @test length(tiles) == 3

            for tile in tiles
                @test tile.params.num_cells >= 3
                @test tile.params.iDim >= 9
            end
            @test tiles[1].params.iMin ≈ patch.params.iMin
            @test tiles[end].params.iMax ≈ patch.params.iMax
            @test sum(t.params.num_cells for t in tiles) == patch.params.num_cells

            # ── allocateSplineBuffer — shape (iDim, 3, b_kDim, nvars) for RLZ ─
            buf = allocateSplineBuffer(tiles[1])
            @test isa(buf, Array)
            @test ndims(buf) == 4
            @test size(buf, 1) == tiles[1].params.iDim
            @test size(buf, 2) == 3   # k=0 / real / imag columns
            @test size(buf, 3) == tiles[1].params.b_kDim
            @test size(buf, 4) == length(tiles[1].params.vars)

            # ── calcPatchMap — nnz = b_kDim * (1+2*kDim) * (b_iDim-4) * nvars ─
            patchMap = calcPatchMap(patch, tiles[1])
            @test patchMap isa SparseMatrixCSC
            kDim_t   = tiles[1].params.iDim + tiles[1].params.patchOffsetL
            b_kDim_t = tiles[1].params.b_kDim
            nvars_t  = length(tiles[1].params.vars)
            tShare   = tiles[1].params.b_iDim - 4
            @test nnz(patchMap) == b_kDim_t * (1 + 2*kDim_t) * tShare * nvars_t

            # ── calcHaloMap — nnz = b_kDim * (1+2*kDim) * 3 * nvars ─────────
            haloMap = calcHaloMap(patch, tiles[1], tiles[2])
            @test haloMap isa SparseMatrixCSC
            @test nnz(haloMap) == b_kDim_t * (1 + 2*kDim_t) * 3 * nvars_t

            # ── sumSpectralTile! — all z × wavenumber blocks updated ──────────
            tiles[1].spectral .= 1.0
            patch.spectral    .= 0.0
            sumSpectralTile!(patch, tiles[1])

            siL      = tiles[1].params.spectralIndexL
            b_iDim_t = tiles[1].params.b_iDim
            b_iDim_p = patch.params.b_iDim
            patch_kDim  = patch.params.iDim + patch.params.patchOffsetL
            wn_stride_p = b_iDim_p * (1 + 2 * patch_kDim)

            # z=1, k=0 block
            @test all(patch.spectral[siL:siL+b_iDim_t-1, :] .== 1.0)
            # z=1, k=1 real block (p=(1-1)*2=0 → offset b_iDim_p)
            pp1_k1r = (0+1)*b_iDim_p + siL
            @test all(patch.spectral[pp1_k1r:pp1_k1r+b_iDim_t-1, :] .== 1.0)
            # z=1, k=1 imag block (offset 2*b_iDim_p)
            pp1_k1i = (0+2)*b_iDim_p + siL
            @test all(patch.spectral[pp1_k1i:pp1_k1i+b_iDim_t-1, :] .== 1.0)
            # z=2, k=0 block
            pp1_z2  = wn_stride_p + siL
            @test all(patch.spectral[pp1_z2:pp1_z2+b_iDim_t-1, :] .== 1.0)
            # Rows before tile are untouched (siL == 1 for tile 1, nothing before)
            @test all(patch.spectral[1:siL-1, :] .== 0.0)   # empty range, always true

            # ── setSpectralTile! — zero-then-write all z × wavenumber blocks ─
            patch.spectral .= 99.0
            tiles[1].spectral .= 2.0
            setSpectralTile!(patch, tiles[1])

            # z=1, k=0 block written
            @test all(patch.spectral[siL:siL+b_iDim_t-1, :] .== 2.0)
            # z=2, k=0 block written
            @test all(patch.spectral[pp1_z2:pp1_z2+b_iDim_t-1, :] .== 2.0)
            # Rows outside any tile block are zeroed
            @test all(patch.spectral[siL+b_iDim_t:b_iDim_p, :] .== 0.0)

            # ── Too many tiles ──────────────────────────────────────────────
            @test_throws DomainError calcTileSizes(patch, 4)

            # ── splineTransform! / tileTransform! round-trip ───────────────
            @testset "RLZ distributed pipeline (splineTransform! / tileTransform!)" begin
                # patch is RLZ, 9 cells, r ∈ [0, 75], z ∈ [0, 10], 1 variable "u".
                # Fill physical with separable axisymmetric f(r, z) = sin(π·r/75) · (z/10)

                pts_rlz  = getGridpoints(patch)         # (jDim, 3)  cols: [r, λ, z]
                n_rlz    = size(pts_rlz, 1)
                rMax_rlz = patch.params.iMax            # 75.0
                zMax_rlz = patch.params.kMax            # 10.0

                patch.physical .= 0.0
                for i in 1:n_rlz
                    r = pts_rlz[i, 1];  z = pts_rlz[i, 3]
                    patch.physical[i, 1, 1] = sin(π * r / rMax_rlz) * (z / zMax_rlz)
                end

                spectralTransform!(patch)

                sharedSpectral_rlz = SharedArray{Float64}(size(patch.spectral))
                sharedSpectral_rlz[:, :] .= patch.spectral

                splineTransform!(sharedSpectral_rlz, patch)

                physical_rlz = zeros(Float64, size(patch.physical))
                tileTransform!(sharedSpectral_rlz, patch, physical_rlz, patch.spectral)

                analytic_val_rlz = [sin(π * pts_rlz[i,1] / rMax_rlz) * (pts_rlz[i,3] / zMax_rlz)
                                    for i in 1:n_rlz]

                # Values
                @test maximum(abs.(physical_rlz[:, 1, 1] .- analytic_val_rlz)) < 1e-3

                # Axisymmetric ⟹ azimuthal derivative ≈ 0
                @test maximum(abs.(physical_rlz[:, 1, 4])) < 1e-8

                # All physical values must be finite
                @test all(isfinite, physical_rlz)
            end  # RLZ distributed pipeline
        end

        @testset "2D Spherical tiling (SL)" begin
            gp = SpringsteelGridParameters(
                geometry = "SL",
                num_cells = 12,
                iMin = 0.0, iMax = Float64(π),
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0))
            patch = createGrid(gp)

            # ── calcTileSizes ───────────────────────────────────────────────
            tiles = calcTileSizes(patch, 4)
            @test length(tiles) == 4

            for tile in tiles
                @test tile.params.num_cells >= 3
                @test tile.params.iDim >= 9
            end

            @test tiles[1].params.iMin ≈ patch.params.iMin
            @test tiles[end].params.iMax ≈ patch.params.iMax
            @test sum(t.params.num_cells for t in tiles) == patch.params.num_cells

            # ── allocateSplineBuffer ────────────────────────────────────────
            buf = allocateSplineBuffer(tiles[1])
            @test isa(buf, Array)
            @test length(buf) > 0

            # ── calcPatchMap ────────────────────────────────────────────────
            patchMap = calcPatchMap(patch, tiles[1])
            @test patchMap isa SparseMatrixCSC
            # SL has (2*kDim+1) wavenumber blocks; patchMap should have significantly
            # more non-zeros than a 1D grid of the same size
            rl_gp = SpringsteelGridParameters(
                geometry = "R", num_cells = 12,
                iMin = 0.0, iMax = 10.0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0))
            rl_patch = createGrid(rl_gp)
            rl_tiles  = calcTileSizes(rl_patch, 4)
            rl_pmap   = calcPatchMap(rl_patch, rl_tiles[1])
            @test nnz(patchMap) > nnz(rl_pmap)

            # ── calcHaloMap ─────────────────────────────────────────────────
            haloMap = calcHaloMap(patch, tiles[1], tiles[2])
            @test haloMap isa SparseMatrixCSC
            @test nnz(haloMap) >= 1

            # ── getBorderSpectral — nnz = 3 * nvars * complete_blocks_in_spectral ──
            # SL allocates exactly b_iDim*(1+2*kDim) rows so all blocks fit.
            nvars_bs = length(tiles[1].params.vars)
            n_bs     = size(tiles[1].spectral, 1)
            b_bs     = tiles[1].params.b_iDim
            tiles[1].spectral .= 1.0
            border = getBorderSpectral(tiles[1])
            @test border isa SparseMatrixCSC
            @test nnz(border) == 3 * nvars_bs * div(n_bs, b_bs)

            # ── sumSpectralTile! — verifies all wavenumber blocks are updated ──
            tiles[1].spectral .= 1.0
            patch.spectral .= 0.0
            sumSpectralTile!(patch, tiles[1])

            siL = tiles[1].params.spectralIndexL
            siR = tiles[1].params.spectralIndexR
            # k=0 block
            @test all(patch.spectral[siL:siR, :] .== 1.0)
            # k=1 real block (p=2): rows patch.b_iDim+siL .. patch.b_iDim+siR
            kDim_tile = tiles[1].params.iDim + tiles[1].params.patchOffsetL
            if kDim_tile >= 1
                pp1 = patch.params.b_iDim + siL
                pp2 = patch.params.b_iDim + siR
                @test all(patch.spectral[pp1:pp2, :] .== 1.0)
            end
            # Rows outside tile domain are still zero
            @test all(patch.spectral[1:siL-1, :] .== 0.0)

            # ── setSpectralTile! — verifies zero-then-write, all blocks written ──
            patch.spectral .= 99.0
            tiles[1].spectral .= 2.0
            setSpectralTile!(patch, tiles[1])
            # k=0 block updated
            @test all(patch.spectral[siL:siR, :] .== 2.0)
            # Rows outside tile domain zeroed
            @test all(patch.spectral[1:siL-1, :] .== 0.0)
            @test all(patch.spectral[siR+1:patch.params.b_iDim, :] .== 0.0)

            # Too many tiles
            @test_throws DomainError calcTileSizes(patch, 5)

            # ── splineTransform! / tileTransform! round-trip ───────────────
            @testset "SL distributed pipeline (splineTransform! / tileTransform!)" begin
                # patch is SL, 12 cells, θ ∈ [0, π], 1 variable "u".
                # Fill physical with the axisymmetric function f(θ) = sin(θ),
                # which vanishes naturally at the poles (BCs satisfied).

                pts_sl = getGridpoints(patch)             # (jDim, 2)  cols: [θ, λ]
                n_sl   = size(pts_sl, 1)

                # --- seed physical with f(θ) = sin(θ) ----------------------
                patch.physical .= 0.0
                for i in 1:n_sl
                    θ = pts_sl[i, 1]
                    patch.physical[i, 1, 1] = sin(θ)
                end

                # physical → B-coefficients in patch.spectral
                spectralTransform!(patch)

                # Copy B-coefficients into a SharedArray
                sharedSpectral_sl = SharedArray{Float64}(size(patch.spectral))
                sharedSpectral_sl[:, :] .= patch.spectral

                # splineTransform!: B → A
                splineTransform!(sharedSpectral_sl, patch)

                # tileTransform!: A → physical
                physical_sl = zeros(Float64, size(patch.physical))
                tileTransform!(sharedSpectral_sl, patch, physical_sl, patch.spectral)

                # Analytic values and derivatives
                analytic_val = [sin(pts_sl[i, 1])  for i in 1:n_sl]
                analytic_dθ  = [cos(pts_sl[i, 1])  for i in 1:n_sl]

                # Values: smooth function over 12 cells; cubic B-spline accuracy
                @test maximum(abs.(physical_sl[:, 1, 1] .- analytic_val)) < 5e-4

                # ∂f/∂θ
                @test maximum(abs.(physical_sl[:, 1, 2] .- analytic_dθ))  < 0.05

                # Axisymmetric ⟹ azimuthal derivative ≈ 0
                @test maximum(abs.(physical_sl[:, 1, 4])) < 1e-10

                # All physical values must be finite
                @test all(isfinite, physical_sl)
            end  # SL distributed pipeline
        end

        @testset "3D Spherical tiling (SLZ)" begin
            gp = SpringsteelGridParameters(
                geometry = "SLZ",
                num_cells = 9,
                iMin = 0.0, iMax = Float64(π),
                kMin = 0.0, kMax = 10.0, kDim = 6,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCB = Dict("u" => Chebyshev.R0),
                BCT = Dict("u" => Chebyshev.R0))
            patch = createGrid(gp)

            # ── calcTileSizes ───────────────────────────────────────────────
            tiles = calcTileSizes(patch, 3)
            @test length(tiles) == 3

            for tile in tiles
                @test tile.params.num_cells >= 3
                @test tile.params.iDim >= 9
            end
            @test tiles[1].params.iMin ≈ patch.params.iMin
            @test tiles[end].params.iMax ≈ patch.params.iMax
            @test sum(t.params.num_cells for t in tiles) == patch.params.num_cells

            # ── allocateSplineBuffer ────────────────────────────────────────
            buf = allocateSplineBuffer(tiles[1])
            @test isa(buf, Array)
            @test ndims(buf) == 4
            @test size(buf, 1) == tiles[1].params.iDim
            @test size(buf, 2) == 3   # k=0 / real / imag columns
            @test size(buf, 3) == tiles[1].params.b_kDim
            @test size(buf, 4) == length(tiles[1].params.vars)

            patchMap = calcPatchMap(patch, tiles[1])
            @test patchMap isa SparseMatrixCSC
            kDim_t   = tiles[1].params.iDim + tiles[1].params.patchOffsetL
            b_kDim_t = tiles[1].params.b_kDim
            nvars_t  = length(tiles[1].params.vars)
            tShare   = tiles[1].params.b_iDim - 4
            @test nnz(patchMap) == b_kDim_t * (1 + 2*kDim_t) * tShare * nvars_t

            # ── calcHaloMap — nnz = b_kDim * (1+2*kDim) * 3 * nvars ─────────
            haloMap = calcHaloMap(patch, tiles[1], tiles[2])
            @test haloMap isa SparseMatrixCSC
            @test nnz(haloMap) == b_kDim_t * (1 + 2*kDim_t) * 3 * nvars_t

            # ── getBorderSpectral ───────────────────────────────────────────
            border = getBorderSpectral(tiles[1])
            @test border isa SparseMatrixCSC

            # ── sumSpectralTile! — all z × wavenumber blocks updated ──────────
            tiles[1].spectral .= 1.0
            patch.spectral    .= 0.0
            sumSpectralTile!(patch, tiles[1])

            siL      = tiles[1].params.spectralIndexL
            b_iDim_t = tiles[1].params.b_iDim
            b_iDim_p = patch.params.b_iDim
            patch_kDim  = patch.params.iDim + patch.params.patchOffsetL
            wn_stride_p = b_iDim_p * (1 + 2 * patch_kDim)

            # z=1, k=0 block
            @test all(patch.spectral[siL:siL+b_iDim_t-1, :] .== 1.0)
            # z=1, k=1 real block (p=(1-1)*2=0 → offset b_iDim_p)
            pp1_k1r = (0+1)*b_iDim_p + siL
            @test all(patch.spectral[pp1_k1r:pp1_k1r+b_iDim_t-1, :] .== 1.0)
            # z=1, k=1 imag block
            pp1_k1i = (0+2)*b_iDim_p + siL
            @test all(patch.spectral[pp1_k1i:pp1_k1i+b_iDim_t-1, :] .== 1.0)
            # z=2, k=0 block
            pp1_z2 = wn_stride_p + siL
            @test all(patch.spectral[pp1_z2:pp1_z2+b_iDim_t-1, :] .== 1.0)
            # Rows before tile are untouched
            @test all(patch.spectral[1:siL-1, :] .== 0.0)

            # ── setSpectralTile! — zero-then-write all z × wavenumber blocks ─
            patch.spectral .= 99.0
            tiles[1].spectral .= 2.0
            setSpectralTile!(patch, tiles[1])

            # z=1, k=0 block written
            @test all(patch.spectral[siL:siL+b_iDim_t-1, :] .== 2.0)
            # z=2, k=0 block written
            @test all(patch.spectral[pp1_z2:pp1_z2+b_iDim_t-1, :] .== 2.0)
            # Rows outside tile block are zeroed
            @test all(patch.spectral[siL+b_iDim_t:b_iDim_p, :] .== 0.0)

            # ── Too many tiles ──────────────────────────────────────────────
            @test_throws DomainError calcTileSizes(patch, 4)

            # ── splineTransform! / tileTransform! round-trip ───────────────
            @testset "SLZ distributed pipeline (splineTransform! / tileTransform!)" begin
                # patch is SLZ, 9 cells, θ ∈ [0, π], z ∈ [0, 10], 1 variable "u".
                # Fill physical with separable axisymmetric f(θ, z) = sin(θ) · (z/10)

                pts_slz  = getGridpoints(patch)         # (jDim, 3)  cols: [θ, λ, z]
                n_slz    = size(pts_slz, 1)
                zMax_slz = patch.params.kMax            # 10.0

                patch.physical .= 0.0
                for i in 1:n_slz
                    θ = pts_slz[i, 1];  z = pts_slz[i, 3]
                    patch.physical[i, 1, 1] = sin(θ) * (z / zMax_slz)
                end

                spectralTransform!(patch)

                sharedSpectral_slz = SharedArray{Float64}(size(patch.spectral))
                sharedSpectral_slz[:, :] .= patch.spectral

                splineTransform!(sharedSpectral_slz, patch)

                physical_slz = zeros(Float64, size(patch.physical))
                tileTransform!(sharedSpectral_slz, patch, physical_slz, patch.spectral)

                analytic_val_slz = [sin(pts_slz[i,1]) * (pts_slz[i,3] / zMax_slz)
                                    for i in 1:n_slz]

                # Values
                @test maximum(abs.(physical_slz[:, 1, 1] .- analytic_val_slz)) < 1e-3

                # Axisymmetric ⟹ azimuthal derivative ≈ 0
                @test maximum(abs.(physical_slz[:, 1, 4])) < 1e-8

                # All physical values must be finite
                @test all(isfinite, physical_slz)
            end  # SLZ distributed pipeline
        end

        @testset "Fallback: calcTileSizes for non-Spline-i-basis" begin
            # For now check that fallback for any SpringsteelGrid returns the grid itself for num_tiles=1
            gp = SpringsteelGridParameters(
                geometry = "R",
                num_cells = 6,
                iMin = 0.0, iMax = 10.0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)
            single_tile = calcTileSizes(grid, 1)
            @test length(single_tile) == 1
        end

        @testset "Multi-dim tiling" begin

            # ── 2D tiling on RR_Grid (3 i-tiles × 2 j-tiles = 6 tiles) ──────
            gp_rr = SpringsteelGridParameters(
                geometry  = "RR", num_cells = 12,
                iMin = 0.0, iMax = 100.0,
                jMin = 0.0, jMax = 100.0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCU = Dict("u" => CubicBSpline.R0),
                BCD = Dict("u" => CubicBSpline.R0))
            patch_rr = createGrid(gp_rr)
            tiles_rr = calcTileSizes(patch_rr, (i=3, j=2))
            @test length(tiles_rr) == 6  # 3 × 2

            # Each tile has valid physical dimensions
            for tile in tiles_rr
                @test tile.params.num_cells >= 3     # ≥ 3 i-cells
                @test tile.params.iDim >= 9          # ≥ 9 i-gridpoints
                @test tile.params.jDim > 0           # j-dimension set
                @test tile.params.b_jDim >= 6        # ≥ 3 j-cells → b_jDim = nc_j+3 ≥ 6
            end

            # i-cell counts sum to patch total (pick one j-strip: tiles at j=1)
            nc_i_strip = [tiles_rr[(ti-1)*2 + 1].params.num_cells for ti in 1:3]
            @test sum(nc_i_strip) == patch_rr.params.num_cells

            # spectralIndexL sequence is correct for each i-strip
            @test tiles_rr[1].params.spectralIndexL == 1
            @test tiles_rr[3].params.spectralIndexL == tiles_rr[1].params.num_cells + 1
            @test tiles_rr[5].params.spectralIndexL == tiles_rr[3].params.spectralIndexL + tiles_rr[3].params.num_cells

            # ── 3D tiling on RRR_Grid (2×2×2 = 8 tiles) ─────────────────────
            gp_rrr = SpringsteelGridParameters(
                geometry  = "RRR", num_cells = 6,
                iMin = 0.0, iMax = 50.0,
                jMin = 0.0, jMax = 50.0,
                kMin = 0.0, kMax = 50.0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0),
                BCU = Dict("u" => CubicBSpline.R0),
                BCD = Dict("u" => CubicBSpline.R0),
                BCB = Dict("u" => CubicBSpline.R0),
                BCT = Dict("u" => CubicBSpline.R0))
            patch_rrr = createGrid(gp_rrr)
            tiles_rrr = calcTileSizes(patch_rrr, (i=2, j=2, k=2))
            @test length(tiles_rrr) == 8  # 2 × 2 × 2

            for tile in tiles_rrr
                @test tile.params.num_cells >= 3
                @test tile.params.jDim > 0
                @test tile.params.kDim > 0
            end

            # ── allocateSplineBuffer — shape (iDim, b_jDim, b_kDim, nvars) for RRR
            buf_rrr = allocateSplineBuffer(tiles_rrr[1])
            @test isa(buf_rrr, Array)
            @test ndims(buf_rrr) == 4
            @test size(buf_rrr, 1) == tiles_rrr[1].params.iDim
            @test size(buf_rrr, 2) == tiles_rrr[1].params.b_jDim
            @test size(buf_rrr, 3) == tiles_rrr[1].params.b_kDim
            @test size(buf_rrr, 4) == length(tiles_rrr[1].params.vars)

            # ── Tiling non-Spline j-dimension should throw DomainError ────────
            gp_rl = SpringsteelGridParameters(
                geometry = "RL", num_cells = 10,
                iMin = 0.0, iMax = 100.0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0))
            patch_rl = createGrid(gp_rl)
            @test_throws DomainError calcTileSizes(patch_rl, (i=2, j=2))

            # ── 1-D NamedTuple delegation (i=N on a 1D R_Grid) ───────────────
            gp_r = SpringsteelGridParameters(
                geometry = "R", num_cells = 12,
                iMin = 0.0, iMax = 100.0,
                vars = Dict("u" => 1),
                BCL = Dict("u" => CubicBSpline.R0),
                BCR = Dict("u" => CubicBSpline.R0))
            patch_r = createGrid(gp_r)
            tiles_r = calcTileSizes(patch_r, (i=4,))
            @test length(tiles_r) == 4

            # ── calcPatchMap_multidim / calcHaloMap_multidim ─────────────────
            pm = calcPatchMap_multidim(patch_rr, tiles_rr[1])
            @test pm isa SparseMatrixCSC

            hm = calcHaloMap_multidim(patch_rr, tiles_rr[1], tiles_rr[3])  # adjacent in i
            @test hm isa SparseMatrixCSC

        end  # Multi-dim tiling

    end  # SpringsteelGrid Tiling


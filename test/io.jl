using DataFrames

    @testset "SpringsteelGrid I/O" begin

        @testset "Gridpoints" begin
            # 1D
            gp1d = SpringsteelGridParameters(
                geometry="R", num_cells=10,
                iMin=0.0, iMax=100.0,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0))
            grid1d = createGrid(gp1d)
            pts = getGridpoints(grid1d)
            @test length(pts) == grid1d.params.iDim
            @test pts[1] >= gp1d.iMin
            @test pts[end] <= gp1d.iMax
            @test all(diff(pts) .> 0)  # monotonically increasing
        end

        @testset "getRegularGridpoints" begin
            gp = SpringsteelGridParameters(
                geometry="R", num_cells=10,
                iMin=0.0, iMax=100.0,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)
            reg_pts = getRegularGridpoints(grid)
            @test length(reg_pts) == grid.params.i_regular_out
            @test reg_pts[1] ≈ gp.iMin
            @test reg_pts[end] ≈ gp.iMax
            @test all(diff(reg_pts) .> 0)
        end

        @testset "regularGridTransform roundtrip" begin
            gp = SpringsteelGridParameters(
                geometry="R", num_cells=60,
                iMin=0.0, iMax=10.0,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.PERIODIC),
                BCR=Dict("u" => CubicBSpline.PERIODIC))
            grid = createGrid(gp)
            pts = getGridpoints(grid)
            L = gp.iMax - gp.iMin
            for i in eachindex(pts)
                grid.physical[i, 1, 1] = sin(2π * pts[i] / L)
            end
            spectralTransform!(grid)
            reg_pts = getRegularGridpoints(grid)
            reg_phys = regularGridTransform(grid, reg_pts)
            # Values at regular points should match sin function closely
            # (tolerance matches the Spline1D_Grid regularGridTransform tests: 1e-5)
            max_err = maximum(abs.(reg_phys[:, 1, 1] .- sin.(2π .* reg_pts ./ L)))
            @test max_err < 1e-5
        end

        @testset "Write/read roundtrip" begin
            gp = SpringsteelGridParameters(
                geometry="R", num_cells=10,
                iMin=0.0, iMax=100.0,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)
            # Initialize physical with known values
            grid.physical[:, 1, 1] .= 1.0

            # Write to temp directory
            tmpdir = mktempdir()
            write_grid(grid, tmpdir, "test")
            # Verify files exist
            @test isfile(joinpath(tmpdir, "test_physical.csv"))
            @test isfile(joinpath(tmpdir, "test_spectral.csv"))
            @test isfile(joinpath(tmpdir, "test_gridded.csv"))
        end

        @testset "check_grid_dims" begin
            gp = SpringsteelGridParameters(
                geometry="R", num_cells=10,
                iMin=0.0, iMax=100.0,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)
            # Test with correct-sized DataFrame
            df_good = DataFrame(u = zeros(grid.params.iDim))
            @test check_grid_dims(df_good, grid) === nothing

            # Test with wrong-sized DataFrame
            df_bad = DataFrame(u = zeros(5))
            @test_throws DomainError check_grid_dims(df_bad, grid)
        end

        @testset "check_grid_dims 2D RL" begin
            gp = SpringsteelGridParameters(
                geometry="RL", num_cells=5,
                iMin=0.0, iMax=50.0,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)
            df_good = DataFrame(u = zeros(size(grid.physical, 1)))
            @test check_grid_dims(df_good, grid) === nothing
            df_bad  = DataFrame(u = zeros(3))
            @test_throws DomainError check_grid_dims(df_bad, grid)
        end

        @testset "write_grid 2D RL produces files" begin
            gp = SpringsteelGridParameters(
                geometry="RL", num_cells=5,
                iMin=0.0, iMax=50.0,
                vars=Dict("u" => 1),
                BCL=Dict("u" => CubicBSpline.R0),
                BCR=Dict("u" => CubicBSpline.R0))
            grid = createGrid(gp)
            tmpdir = mktempdir()
            write_grid(grid, tmpdir, "rl_test")
            @test isfile(joinpath(tmpdir, "rl_test_physical.csv"))
            @test isfile(joinpath(tmpdir, "rl_test_spectral.csv"))
        end

        # ─── JLD2 helpers for roundtrip tests ───────────────────────────────
        function _jld2_roundtrip_test(grid)
            pts = getGridpoints(grid)
            is1d = pts isa Vector
            for i in 1:size(grid.physical, 1)
                v = is1d ? pts[i] : pts[i, 1]
                grid.physical[i, 1, 1] = sin(v)
                grid.physical[i, 2, 1] = cos(v)
            end
            spectralTransform!(grid)
            tmpfile = joinpath(mktempdir(), "test_roundtrip.jld2")
            save_grid(tmpfile, grid)
            @test isfile(tmpfile)
            loaded = load_grid(tmpfile)
            @test loaded.params.geometry == grid.params.geometry
            @test loaded.params.num_cells == grid.params.num_cells
            @test size(loaded.spectral) == size(grid.spectral)
            @test size(loaded.physical) == size(grid.physical)
            @test loaded.spectral ≈ grid.spectral
            @test loaded.physical ≈ grid.physical
            gridTransform!(loaded)
            @test all(isfinite.(loaded.physical))
        end

        @testset "JLD2 save/load" begin

            @testset "save/load roundtrip R" begin
                gp = SpringsteelGridParameters(
                    geometry="R", num_cells=10,
                    iMin=0.0, iMax=100.0,
                    vars=Dict("u" => 1, "v" => 2),
                    BCL=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0))
                _jld2_roundtrip_test(createGrid(gp))
            end

            @testset "load pre-1.1 archive (added params fields)" begin
                # Fixture written before SpringsteelGridParameters gained
                # num_cells_i/j/k. JLD2 cannot reconstruct the struct and hands
                # back a ReconstructedStatic; load_grid must upgrade it.
                fixture = joinpath(@__DIR__, "fixtures", "legacy_grid_v1_rr.jld2")
                @test isfile(fixture)

                raw = jldopen(fixture, "r") do f; f["params"]; end
                @test !(raw isa SpringsteelGridParameters)   # genuinely legacy
                @test jldopen(fixture, "r") do f; f["format_version"]; end == "1.0"

                grid = load_grid(fixture)
                @test grid isa RR_Grid
                # Dimensions re-resolve to exactly what the old code derived
                @test grid.params.num_cells == 6
                @test grid.params.num_cells_i == 6
                @test grid.params.num_cells_j == 6
                @test grid.params.jDim == 18
                @test grid.params.b_jDim == 9
                # ... and the archived coefficients survive untouched
                @test sum(abs, grid.spectral) ≈ 278.57593577400144
                @test !any(isnan, grid.physical)

                # A fresh round-trip preserves the new fields
                mktempdir() do dir
                    f2 = joinpath(dir, "new.jld2")
                    save_grid(f2, grid)
                    @test jldopen(f2, "r") do f; f["format_version"]; end == "1.1"
                    g2 = load_grid(f2)
                    @test g2.params.num_cells_j == grid.params.num_cells_j
                    @test g2.params.num_cells_k == grid.params.num_cells_k
                    @test g2.spectral == grid.spectral
                end
            end

            @testset "load archive with widened Dict params" begin
                # Fixture written while vars/l_q/max_wavenumber and the filter fields
                # were still declared as a bare `Dict`, so the caller could store
                # genuinely widened dicts (Dict{String,Any}). Now that those fields
                # are concretely typed, JLD2 must *convert* the on-disk values rather
                # than reject them. Regenerate with test/fixtures/make_fixtures.jl.
                fixture = joinpath(@__DIR__, "fixtures", "widened_dicts_grid.jld2")
                @test isfile(fixture)

                # This exercises the *other* JLD2 path from the pre-1.1 fixture above.
                # There, fields were missing from the archive, so JLD2 gave up and handed
                # back a ReconstructedStatic for `_upgrade_params` to rebuild. Here every
                # field is present but three of them are stored at a wider type, so JLD2
                # builds the struct in place and `convert` narrows each field on the way
                # in — which is why `raw` is already a real SpringsteelGridParameters with
                # narrowed dicts, and why the widening is not observable after the fact.
                raw = jldopen(fixture, "r") do f; f["params"]; end
                @test raw isa SpringsteelGridParameters      # converted, not reconstructed
                @test raw.vars isa Dict{String,Int64}

                grid = load_grid(fixture)
                @test grid isa RL_Grid

                # ... and they come back narrowed, not widened.
                @test grid.params.vars isa Dict{String,Int64}
                @test grid.params.l_q isa Dict{String,Float64}
                @test grid.params.max_wavenumber isa Dict{String,Int64}
                @test grid.params.fourier_filter isa Dict{String,AbstractFilter}
                @test grid.params.spline_filter isa Dict{String,Dict{Symbol,AbstractFilter}}

                # Values survive the coercion intact
                @test grid.params.vars == Dict("u" => 1, "v" => 2)
                @test grid.params.l_q["default"] == 2.0
                @test grid.params.max_wavenumber["default"] == 4
                @test grid.params.spline_filter["u"][:i] isa GaussianFilter
                @test !any(isnan, grid.physical)
            end

            @testset "save/load roundtrip RL" begin
                gp = SpringsteelGridParameters(
                    geometry="RL", num_cells=5,
                    iMin=0.0, iMax=50.0,
                    vars=Dict("u" => 1, "v" => 2),
                    BCL=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0))
                _jld2_roundtrip_test(createGrid(gp))
            end

            @testset "save/load roundtrip RZ" begin
                gp = SpringsteelGridParameters(
                    geometry="RZ", num_cells=5,
                    iMin=0.0, iMax=50.0,
                    kMin=0.0, kMax=10.0, kDim=10,
                    vars=Dict("u" => 1, "v" => 2),
                    BCL=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                    BCB=Dict("u" => Chebyshev.R0, "v" => Chebyshev.R0),
                    BCT=Dict("u" => Chebyshev.R0, "v" => Chebyshev.R0))
                _jld2_roundtrip_test(createGrid(gp))
            end

            @testset "save/load roundtrip RLZ" begin
                gp = SpringsteelGridParameters(
                    geometry="RLZ", num_cells=5,
                    iMin=0.0, iMax=50.0,
                    kMin=0.0, kMax=10.0, kDim=10,
                    vars=Dict("u" => 1, "v" => 2),
                    BCL=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                    BCB=Dict("u" => Chebyshev.R0, "v" => Chebyshev.R0),
                    BCT=Dict("u" => Chebyshev.R0, "v" => Chebyshev.R0))
                _jld2_roundtrip_test(createGrid(gp))
            end

            @testset "save/load roundtrip RR" begin
                gp = SpringsteelGridParameters(
                    geometry="RR", num_cells=5,
                    iMin=0.0, iMax=50.0,
                    jMin=0.0, jMax=50.0,
                    vars=Dict("u" => 1, "v" => 2),
                    BCL=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                    BCU=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                    BCD=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0))
                _jld2_roundtrip_test(createGrid(gp))
            end

            @testset "save/load roundtrip RRR" begin
                gp = SpringsteelGridParameters(
                    geometry="RRR", num_cells=5,
                    iMin=0.0, iMax=50.0,
                    jMin=0.0, jMax=50.0,
                    kMin=0.0, kMax=50.0,
                    vars=Dict("u" => 1, "v" => 2),
                    BCL=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                    BCU=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                    BCD=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                    BCB=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                    BCT=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0))
                _jld2_roundtrip_test(createGrid(gp))
            end

            @testset "save/load roundtrip SL" begin
                gp = SpringsteelGridParameters(
                    geometry="SL", num_cells=5,
                    iMin=0.1, iMax=3.0,
                    vars=Dict("u" => 1, "v" => 2),
                    BCL=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0))
                _jld2_roundtrip_test(createGrid(gp))
            end

            @testset "save/load roundtrip SLZ" begin
                gp = SpringsteelGridParameters(
                    geometry="SLZ", num_cells=5,
                    iMin=0.1, iMax=3.0,
                    kMin=0.0, kMax=10.0, kDim=10,
                    vars=Dict("u" => 1, "v" => 2),
                    BCL=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                    BCB=Dict("u" => Chebyshev.R0, "v" => Chebyshev.R0),
                    BCT=Dict("u" => Chebyshev.R0, "v" => Chebyshev.R0))
                _jld2_roundtrip_test(createGrid(gp))
            end

            @testset "save/load compress=false" begin
                gp = SpringsteelGridParameters(
                    geometry="R", num_cells=10,
                    iMin=0.0, iMax=100.0,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0))
                grid = createGrid(gp)
                grid.physical[:, 1, 1] .= 1.0
                spectralTransform!(grid)
                tmpfile = joinpath(mktempdir(), "test_nocompress.jld2")
                save_grid(tmpfile, grid; compress=false)
                @test isfile(tmpfile)
                loaded = load_grid(tmpfile)
                @test loaded.spectral ≈ grid.spectral
                @test loaded.physical ≈ grid.physical
            end

            @testset "load_grid nonexistent file throws" begin
                @test_throws Exception load_grid("/nonexistent/path/missing.jld2")
            end

        end  # JLD2 save/load

        @testset "NetCDF I/O" begin

            @testset "write_netcdf 1D R" begin
                gp = SpringsteelGridParameters(geometry="R", num_cells=20,
                    iMin=0.0, iMax=10.0,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.PERIODIC),
                    BCR=Dict("u" => CubicBSpline.PERIODIC))
                grid = createGrid(gp)
                pts = getGridpoints(grid)
                for i in eachindex(pts)
                    grid.physical[i, 1, 1] = sin(2π * pts[i] / 10.0)
                end
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_r.nc")
                write_netcdf(tmpfile, grid)
                @test isfile(tmpfile)

                NCDataset(tmpfile, "r") do ds
                    @test haskey(ds.dim, "x")
                    @test ds.dim["x"] == grid.params.i_regular_out
                    @test haskey(ds, "u")
                    @test ds.attrib["Conventions"] == "CF-1.12"
                    x_vals = Array(ds["x"])
                    @test x_vals[1] ≈ gp.iMin
                    @test x_vals[end] ≈ gp.iMax
                    u_vals = Array(ds["u"])
                    @test length(u_vals) == grid.params.i_regular_out
                    @test maximum(abs.(u_vals .- sin.(2π .* x_vals ./ 10.0))) < 1e-4
                end
            end

            @testset "write_netcdf 1D with derivatives" begin
                gp = SpringsteelGridParameters(geometry="R", num_cells=20,
                    iMin=0.0, iMax=10.0,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.PERIODIC),
                    BCR=Dict("u" => CubicBSpline.PERIODIC))
                grid = createGrid(gp)
                pts = getGridpoints(grid)
                for i in eachindex(pts)
                    grid.physical[i, 1, 1] = sin(2π * pts[i] / 10.0)
                end
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_r_deriv.nc")
                write_netcdf(tmpfile, grid; include_derivatives=true)
                @test isfile(tmpfile)

                NCDataset(tmpfile, "r") do ds
                    @test haskey(ds, "u")
                    @test haskey(ds, "u_x")
                    @test haskey(ds, "u_xx")
                    # Derivative of sin(2π x/10) ≈ (2π/10)*cos(2π x/10)
                    x_vals = Array(ds["x"])
                    du_vals = Array(ds["u_x"])
                    expected_du = (2π / 10.0) .* cos.(2π .* x_vals ./ 10.0)
                    @test maximum(abs.(du_vals .- expected_du)) < 1e-3
                end
            end

            @testset "write_netcdf custom attributes" begin
                gp = SpringsteelGridParameters(geometry="R", num_cells=10,
                    iMin=0.0, iMax=10.0,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0))
                grid = createGrid(gp)
                grid.physical[:, 1, 1] .= 1.0
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_attrs.nc")
                write_netcdf(tmpfile, grid;
                    global_attributes=Dict{String,Any}("institution" => "Test Lab"))
                @test isfile(tmpfile)

                NCDataset(tmpfile, "r") do ds
                    @test ds.attrib["Conventions"] == "CF-1.12"
                    @test ds.attrib["institution"] == "Test Lab"
                    @test haskey(ds.attrib, "history")
                    @test haskey(ds.attrib, "source")
                end
            end

            @testset "write_netcdf multiple variables" begin
                gp = SpringsteelGridParameters(geometry="R", num_cells=15,
                    iMin=0.0, iMax=2π,
                    vars=Dict("u" => 1, "v" => 2),
                    BCL=Dict("u" => CubicBSpline.PERIODIC, "v" => CubicBSpline.PERIODIC),
                    BCR=Dict("u" => CubicBSpline.PERIODIC, "v" => CubicBSpline.PERIODIC))
                grid = createGrid(gp)
                pts = getGridpoints(grid)
                for i in eachindex(pts)
                    grid.physical[i, 1, 1] = sin(pts[i])
                    grid.physical[i, 2, 1] = cos(pts[i])
                end
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_multivars.nc")
                write_netcdf(tmpfile, grid)

                NCDataset(tmpfile, "r") do ds
                    @test haskey(ds, "u")
                    @test haskey(ds, "v")
                    @test length(Array(ds["u"])) == grid.params.i_regular_out
                    @test length(Array(ds["v"])) == grid.params.i_regular_out
                end
            end

            # ── 2D and 3D write_netcdf tests ──────────────────────────────

            @testset "write_netcdf 2D RL (cylindrical j-active)" begin
                gp = SpringsteelGridParameters(
                    geometry="RL", num_cells=5,
                    iMin=0.0, iMax=50.0,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0))
                grid = createGrid(gp)
                pts  = getGridpoints(grid)
                for p in 1:size(pts, 1)
                    r = pts[p, 1]; λ = pts[p, 2]
                    grid.physical[p, 1, 1] = sin(r / 50.0) * cos(λ)
                end
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_rl.nc")
                write_netcdf(tmpfile, grid)
                @test isfile(tmpfile)

                NCDataset(tmpfile, "r") do ds
                    @test haskey(ds.dim, "radius")
                    @test haskey(ds.dim, "azimuth")
                    @test ds.dim["radius"]  == grid.params.i_regular_out
                    @test ds.dim["azimuth"] == grid.params.j_regular_out
                    @test haskey(ds, "u")
                    @test ds.attrib["Conventions"] == "CF-1.12"
                    az_vals = Array(ds["azimuth"])
                    @test minimum(az_vals) >= 0.0
                    @test maximum(az_vals) < 360.0
                    r_vals = Array(ds["radius"])
                    @test r_vals[1] ≈ gp.iMin
                    @test r_vals[end] ≈ gp.iMax
                    u_data = Array(ds["u"])
                    @test size(u_data) == (grid.params.i_regular_out, grid.params.j_regular_out)
                    # Values should roughly match sin(r/50)*cos(λ)
                    az_rad = az_vals .* (π / 180.0)
                    ref = [sin(r / 50.0) * cos(az) for r in r_vals, az in az_rad]
                    @test maximum(abs.(u_data .- ref)) < 0.1
                end
            end

            @testset "write_netcdf 2D RZ (cartesian k-active)" begin
                gp = SpringsteelGridParameters(
                    geometry="RZ", num_cells=5,
                    iMin=0.0, iMax=50.0,
                    kMin=0.0, kMax=10.0, kDim=10,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0),
                    BCB=Dict("u" => Chebyshev.R0),
                    BCT=Dict("u" => Chebyshev.R0))
                grid = createGrid(gp)
                pts  = getGridpoints(grid)
                for p in 1:size(pts, 1)
                    grid.physical[p, 1, 1] = pts[p, 1] + pts[p, 2]
                end
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_rz.nc")
                write_netcdf(tmpfile, grid)
                @test isfile(tmpfile)

                NCDataset(tmpfile, "r") do ds
                    @test haskey(ds.dim, "x")
                    @test haskey(ds.dim, "z")
                    @test ds.dim["x"] == grid.params.i_regular_out
                    @test ds.dim["z"] == grid.params.k_regular_out
                    @test haskey(ds, "u")
                    @test ds.attrib["Conventions"] == "CF-1.12"
                    x_vals = Array(ds["x"])
                    z_vals = Array(ds["z"])
                    @test x_vals[1] ≈ gp.iMin
                    @test x_vals[end] ≈ gp.iMax
                    @test z_vals[1] ≈ gp.kMin
                    @test z_vals[end] ≈ gp.kMax
                    u_data = Array(ds["u"])
                    @test size(u_data) == (grid.params.i_regular_out, grid.params.k_regular_out)
                end
            end

            @testset "write_netcdf 2D SL (spherical j-active)" begin
                gp = SpringsteelGridParameters(
                    geometry="SL", num_cells=5,
                    iMin=0.1, iMax=3.0,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0))
                grid = createGrid(gp)
                pts  = getGridpoints(grid)
                for p in 1:size(pts, 1)
                    grid.physical[p, 1, 1] = sin(pts[p, 1])
                end
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_sl.nc")
                write_netcdf(tmpfile, grid)
                @test isfile(tmpfile)

                NCDataset(tmpfile, "r") do ds
                    @test haskey(ds.dim, "latitude")
                    @test haskey(ds.dim, "longitude")
                    @test ds.dim["latitude"]  == grid.params.i_regular_out
                    @test ds.dim["longitude"] == grid.params.j_regular_out
                    @test haskey(ds, "u")
                    @test ds.attrib["Conventions"] == "CF-1.12"
                    # CF standard_name and units attributes
                    @test ds["latitude"].attrib["standard_name"]  == "latitude"
                    @test ds["latitude"].attrib["units"]          == "degrees_north"
                    @test ds["longitude"].attrib["standard_name"] == "longitude"
                    @test ds["longitude"].attrib["units"]         == "degrees_east"
                    # Latitude must be sorted ascending (south to north)
                    lat_vals = Array(ds["latitude"])
                    @test issorted(lat_vals)
                    # Longitude in [0, 360)
                    lon_vals = Array(ds["longitude"])
                    @test minimum(lon_vals) >= 0.0
                    @test maximum(lon_vals) < 360.0
                end
            end

            @testset "write_netcdf 3D RLZ (cylindrical)" begin
                gp = SpringsteelGridParameters(
                    geometry="RLZ", num_cells=5,
                    iMin=0.0, iMax=50.0,
                    kMin=0.0, kMax=10.0, kDim=10,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0),
                    BCB=Dict("u" => Chebyshev.R0),
                    BCT=Dict("u" => Chebyshev.R0))
                grid = createGrid(gp)
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_rlz.nc")
                write_netcdf(tmpfile, grid)
                @test isfile(tmpfile)

                NCDataset(tmpfile, "r") do ds
                    @test haskey(ds.dim, "radius")
                    @test haskey(ds.dim, "azimuth")
                    @test haskey(ds.dim, "height")
                    @test length(ds.dim) == 3
                    @test ds.dim["radius"]  == grid.params.i_regular_out
                    @test ds.dim["azimuth"] == grid.params.j_regular_out
                    @test ds.dim["height"]  == grid.params.k_regular_out
                    @test haskey(ds, "u")
                    u_data = Array(ds["u"])
                    @test size(u_data) == (grid.params.i_regular_out,
                                           grid.params.j_regular_out,
                                           grid.params.k_regular_out)
                    az_vals = Array(ds["azimuth"])
                    @test minimum(az_vals) >= 0.0
                    @test maximum(az_vals) < 360.0
                end
            end

            @testset "write_netcdf 3D SLZ (spherical)" begin
                gp = SpringsteelGridParameters(
                    geometry="SLZ", num_cells=5,
                    iMin=0.1, iMax=3.0,
                    kMin=0.0, kMax=10.0, kDim=10,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0),
                    BCB=Dict("u" => Chebyshev.R0),
                    BCT=Dict("u" => Chebyshev.R0))
                grid = createGrid(gp)
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_slz.nc")
                write_netcdf(tmpfile, grid)
                @test isfile(tmpfile)

                NCDataset(tmpfile, "r") do ds
                    @test haskey(ds.dim, "latitude")
                    @test haskey(ds.dim, "longitude")
                    @test haskey(ds.dim, "height")
                    @test ds["latitude"].attrib["standard_name"]  == "latitude"
                    @test ds["latitude"].attrib["units"]          == "degrees_north"
                    @test ds["longitude"].attrib["standard_name"] == "longitude"
                    @test ds["longitude"].attrib["units"]         == "degrees_east"
                    lat_vals = Array(ds["latitude"])
                    @test issorted(lat_vals)
                    lon_vals = Array(ds["longitude"])
                    @test minimum(lon_vals) >= 0.0
                    @test maximum(lon_vals) < 360.0
                    h_vals = Array(ds["height"])
                    @test h_vals[1] ≈ gp.kMin
                    @test h_vals[end] ≈ gp.kMax
                    u_data = Array(ds["u"])
                    @test size(u_data) == (grid.params.i_regular_out,
                                           grid.params.j_regular_out,
                                           grid.params.k_regular_out)
                end
            end

            @testset "write_netcdf 2D RR (cartesian j-active)" begin
                gp = SpringsteelGridParameters(
                    geometry="RR", num_cells=5,
                    iMin=0.0, iMax=50.0,
                    jMin=0.0, jMax=50.0,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0),
                    BCU=Dict("u" => CubicBSpline.R0),
                    BCD=Dict("u" => CubicBSpline.R0))
                grid = createGrid(gp)
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_rr.nc")
                write_netcdf(tmpfile, grid)
                @test isfile(tmpfile)

                NCDataset(tmpfile, "r") do ds
                    @test haskey(ds.dim, "x")
                    @test haskey(ds.dim, "y")
                    @test ds.dim["x"] == grid.params.i_regular_out
                    @test ds.dim["y"] == grid.params.j_regular_out
                    @test haskey(ds, "u")
                    x_vals = Array(ds["x"])
                    y_vals = Array(ds["y"])
                    @test x_vals[1] ≈ gp.iMin
                    @test x_vals[end] ≈ gp.iMax
                    @test y_vals[1] ≈ gp.jMin
                    @test y_vals[end] ≈ gp.jMax
                end
            end

            @testset "write_netcdf 3D RRR (cartesian)" begin
                gp = SpringsteelGridParameters(
                    geometry="RRR", num_cells=5,
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
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_rrr.nc")
                write_netcdf(tmpfile, grid)
                @test isfile(tmpfile)

                NCDataset(tmpfile, "r") do ds
                    @test haskey(ds.dim, "x")
                    @test haskey(ds.dim, "y")
                    @test haskey(ds.dim, "z")
                    @test length(ds.dim) == 3
                    @test ds.dim["x"] == grid.params.i_regular_out
                    @test ds.dim["y"] == grid.params.j_regular_out
                    @test ds.dim["z"] == grid.params.k_regular_out
                    @test haskey(ds, "u")
                    u_data = Array(ds["u"])
                    @test size(u_data) == (grid.params.i_regular_out,
                                           grid.params.j_regular_out,
                                           grid.params.k_regular_out)
                end
            end

            @testset "write_netcdf 2D RL with derivatives" begin
                gp = SpringsteelGridParameters(
                    geometry="RL", num_cells=5,
                    iMin=0.0, iMax=50.0,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0))
                grid = createGrid(gp)
                pts  = getGridpoints(grid)
                for p in 1:size(pts, 1)
                    grid.physical[p, 1, 1] = sin(pts[p, 1] / 50.0)
                end
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_rl_deriv.nc")
                write_netcdf(tmpfile, grid; include_derivatives=true)
                @test isfile(tmpfile)

                NCDataset(tmpfile, "r") do ds
                    @test haskey(ds, "u")
                    @test haskey(ds, "u_r")
                    @test haskey(ds, "u_rr")
                    @test haskey(ds, "u_az")
                    @test haskey(ds, "u_azaz")
                end
            end

            # ── read_netcdf tests ─────────────────────────────────────────

            @testset "read_netcdf roundtrip 1D R" begin
                gp = SpringsteelGridParameters(geometry="R", num_cells=20,
                    iMin=0.0, iMax=10.0,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.PERIODIC),
                    BCR=Dict("u" => CubicBSpline.PERIODIC))
                grid = createGrid(gp)
                pts = getGridpoints(grid)
                for i in eachindex(pts)
                    grid.physical[i, 1, 1] = sin(2π * pts[i] / 10.0)
                end
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_r_read.nc")
                write_netcdf(tmpfile, grid)

                data = read_netcdf(tmpfile)

                @test data["dimensions"]["x"] == grid.params.i_regular_out
                reg_pts = getRegularGridpoints(grid)
                @test data["coordinates"]["x"] ≈ reg_pts
                @test haskey(data["variables"], "u")
                @test length(data["variables"]["u"]) == grid.params.i_regular_out
                x_vals = data["coordinates"]["x"]
                @test maximum(abs.(data["variables"]["u"] .- sin.(2π .* x_vals ./ 10.0))) < 1e-4
                @test data["attributes"]["Conventions"] == "CF-1.12"
            end

            @testset "read_netcdf roundtrip 2D RL" begin
                gp = SpringsteelGridParameters(
                    geometry="RL", num_cells=5,
                    iMin=0.0, iMax=50.0,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0))
                grid = createGrid(gp)
                pts  = getGridpoints(grid)
                for p in 1:size(pts, 1)
                    grid.physical[p, 1, 1] = sin(pts[p, 1] / 50.0) * cos(pts[p, 2])
                end
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_rl_read.nc")
                write_netcdf(tmpfile, grid)

                data = read_netcdf(tmpfile)

                @test haskey(data["coordinates"], "radius")
                @test haskey(data["coordinates"], "azimuth")
                @test haskey(data["variables"], "u")
                @test size(data["variables"]["u"]) == (grid.params.i_regular_out, grid.params.j_regular_out)
                @test data["attributes"]["Conventions"] == "CF-1.12"
            end

            @testset "read_netcdf roundtrip SL" begin
                gp = SpringsteelGridParameters(
                    geometry="SL", num_cells=5,
                    iMin=0.1, iMax=3.0,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0))
                grid = createGrid(gp)
                pts  = getGridpoints(grid)
                for p in 1:size(pts, 1)
                    grid.physical[p, 1, 1] = cos(pts[p, 1])
                end
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_sl_read.nc")
                write_netcdf(tmpfile, grid)

                data = read_netcdf(tmpfile)

                @test haskey(data["coordinates"], "latitude")
                @test haskey(data["coordinates"], "longitude")
                lat = data["coordinates"]["latitude"]
                @test issorted(lat)   # ascending south-to-north
                @test haskey(data["variables"], "u")
                @test data["attributes"]["Conventions"] == "CF-1.12"
            end

            @testset "read_netcdf with include_derivatives" begin
                gp = SpringsteelGridParameters(geometry="R", num_cells=20,
                    iMin=0.0, iMax=10.0,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.PERIODIC),
                    BCR=Dict("u" => CubicBSpline.PERIODIC))
                grid = createGrid(gp)
                pts = getGridpoints(grid)
                for i in eachindex(pts)
                    grid.physical[i, 1, 1] = sin(2π * pts[i] / 10.0)
                end
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_r_deriv_read.nc")
                write_netcdf(tmpfile, grid; include_derivatives=true)

                data = read_netcdf(tmpfile)

                @test haskey(data["variables"], "u")
                @test haskey(data["variables"], "u_x")
                @test haskey(data["variables"], "u_xx")
            end

            @testset "read_netcdf preserves CF time coordinate" begin
                # write_netcdf(...; time=t) writes a CF time axis (units
                # "seconds since 1970-...", calendar gregorian), which NCDatasets
                # decodes to DateTime. read_netcdf must preserve it, not coerce to
                # Float64. Regression: it previously threw a convert error.
                gp = SpringsteelGridParameters(geometry="R", num_cells=20,
                    iMin=0.0, iMax=10.0,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.PERIODIC),
                    BCR=Dict("u" => CubicBSpline.PERIODIC))
                grid = createGrid(gp)
                pts = getGridpoints(grid)
                for i in eachindex(pts)
                    grid.physical[i, 1, 1] = sin(2π * pts[i] / 10.0)
                end
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_r_time.nc")
                write_netcdf(tmpfile, grid; time=3600.0)

                data = read_netcdf(tmpfile)   # must not throw

                @test haskey(data["coordinates"], "time")
                t_vals = data["coordinates"]["time"]
                @test eltype(t_vals) <: Dates.AbstractTime
                # 3600 s since the 1970 epoch = 1970-01-01T01:00:00
                @test t_vals[1] == DateTime(1970, 1, 1, 1, 0, 0)
                # spatial coordinate still decodes to Float64
                @test data["coordinates"]["x"] isa AbstractVector{Float64}
                @test haskey(data["variables"], "u")
            end

            @testset "read_netcdf nonexistent file throws" begin
                @test_throws Exception read_netcdf("/nonexistent/path/file.nc")
            end

            # ── New keyword arguments ─────────────────────────────────────

            @testset "write_netcdf coordinate_attributes 1D" begin
                gp = SpringsteelGridParameters(geometry="R", num_cells=5,
                    iMin=0.0, iMax=100.0,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0))
                grid = createGrid(gp)
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_coord_attrs.nc")
                write_netcdf(tmpfile, grid;
                    coordinate_attributes=Dict{String,Dict{String,Any}}(
                        "x" => Dict{String,Any}("units" => "m", "long_name" => "easting")))
                @test isfile(tmpfile)

                NCDataset(tmpfile, "r") do ds
                    @test ds["x"].attrib["units"] == "m"
                    @test ds["x"].attrib["long_name"] == "easting"
                end
            end

            @testset "write_netcdf variable_attributes 1D" begin
                gp = SpringsteelGridParameters(geometry="R", num_cells=5,
                    iMin=0.0, iMax=100.0,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0))
                grid = createGrid(gp)
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_var_attrs.nc")
                write_netcdf(tmpfile, grid;
                    variable_attributes=Dict{String,Dict{String,Any}}(
                        "u" => Dict{String,Any}("units" => "m/s", "long_name" => "velocity")))
                @test isfile(tmpfile)

                NCDataset(tmpfile, "r") do ds
                    @test ds["u"].attrib["units"] == "m/s"
                    @test ds["u"].attrib["long_name"] == "velocity"
                end
            end

            @testset "write_netcdf time keyword 1D" begin
                gp = SpringsteelGridParameters(geometry="R", num_cells=5,
                    iMin=0.0, iMax=100.0,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0))
                grid = createGrid(gp)
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_time.nc")
                write_netcdf(tmpfile, grid; time=1234567890.0)
                @test isfile(tmpfile)

                NCDataset(tmpfile, "r") do ds
                    @test haskey(ds.dim, "time")
                    @test ds.dim["time"] == 1
                    @test ds["time"].attrib["units"] == "seconds since 1970-01-01T00:00:00Z"
                    # Data variable should have time as leading dimension
                    @test "time" in dimnames(ds["u"])
                    @test size(ds["u"]) == (1, grid.params.i_regular_out)
                end
            end

            @testset "write_netcdf grid_mapping keyword 1D" begin
                gp = SpringsteelGridParameters(geometry="R", num_cells=5,
                    iMin=0.0, iMax=100.0,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0))
                grid = createGrid(gp)
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_grid_mapping.nc")
                write_netcdf(tmpfile, grid;
                    grid_mapping=Dict{String,Any}(
                        "grid_mapping_name" => "transverse_mercator",
                        "latitude_of_projection_origin" => 35.0,
                        "longitude_of_central_meridian" => -97.0))
                @test isfile(tmpfile)

                NCDataset(tmpfile, "r") do ds
                    @test haskey(ds, "grid_mapping")
                    @test ds["grid_mapping"].attrib["grid_mapping_name"] == "transverse_mercator"
                    @test ds["grid_mapping"].attrib["latitude_of_projection_origin"] ≈ 35.0
                    @test ds["grid_mapping"].attrib["longitude_of_central_meridian"] ≈ -97.0
                end
            end

            @testset "write_netcdf all keywords combined 1D" begin
                gp = SpringsteelGridParameters(geometry="R", num_cells=5,
                    iMin=0.0, iMax=100.0,
                    vars=Dict("u" => 1, "v" => 2),
                    BCL=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0, "v" => CubicBSpline.R0))
                grid = createGrid(gp)
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_all_kw.nc")
                write_netcdf(tmpfile, grid;
                    include_derivatives=true,
                    global_attributes=Dict{String,Any}("institution" => "CSU"),
                    coordinate_attributes=Dict{String,Dict{String,Any}}(
                        "x" => Dict{String,Any}("units" => "km")),
                    variable_attributes=Dict{String,Dict{String,Any}}(
                        "u" => Dict{String,Any}("units" => "m/s")),
                    time=100.0,
                    grid_mapping=Dict{String,Any}(
                        "grid_mapping_name" => "latitude_longitude"))
                @test isfile(tmpfile)

                NCDataset(tmpfile, "r") do ds
                    @test ds.attrib["institution"] == "CSU"
                    @test ds["x"].attrib["units"] == "km"
                    @test ds["u"].attrib["units"] == "m/s"
                    @test haskey(ds.dim, "time")
                    @test haskey(ds, "grid_mapping")
                    @test haskey(ds, "u_x")   # derivatives
                    @test haskey(ds, "u_xx")
                    # Data vars should have time dimension
                    @test size(ds["u"], 1) == 1
                end
            end

            @testset "write_netcdf keywords 2D RR" begin
                gp = SpringsteelGridParameters(geometry="RR", num_cells=3,
                    iMin=0.0, iMax=10.0,
                    jMin=0.0, jMax=10.0,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0),
                    BCU=Dict("u" => CubicBSpline.R0),
                    BCD=Dict("u" => CubicBSpline.R0))
                grid = createGrid(gp)
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_2d_kw.nc")
                write_netcdf(tmpfile, grid;
                    coordinate_attributes=Dict{String,Dict{String,Any}}(
                        "x" => Dict{String,Any}("units" => "m"),
                        "y" => Dict{String,Any}("units" => "m")),
                    variable_attributes=Dict{String,Dict{String,Any}}(
                        "u" => Dict{String,Any}("units" => "dBZ")),
                    time=200.0,
                    grid_mapping=Dict{String,Any}(
                        "grid_mapping_name" => "transverse_mercator"))
                @test isfile(tmpfile)

                NCDataset(tmpfile, "r") do ds
                    @test ds["x"].attrib["units"] == "m"
                    @test ds["y"].attrib["units"] == "m"
                    @test ds["u"].attrib["units"] == "dBZ"
                    @test haskey(ds.dim, "time")
                    @test ds.dim["time"] == 1
                    @test haskey(ds, "grid_mapping")
                    # Data shape: (time, x, y)
                    @test size(ds["u"]) == (1, grid.params.i_regular_out, grid.params.j_regular_out)
                end
            end

            @testset "write_netcdf keywords 2D RZ" begin
                gp = SpringsteelGridParameters(geometry="RZ", num_cells=3,
                    iMin=0.0, iMax=10.0,
                    kMin=0.0, kMax=5.0, kDim=6,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0),
                    BCB=Dict("u" => Chebyshev.R0),
                    BCT=Dict("u" => Chebyshev.R0))
                grid = createGrid(gp)
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_rz_kw.nc")
                write_netcdf(tmpfile, grid;
                    variable_attributes=Dict{String,Dict{String,Any}}(
                        "u" => Dict{String,Any}("units" => "K")),
                    time=300.0,
                    grid_mapping=Dict{String,Any}(
                        "grid_mapping_name" => "transverse_mercator"))
                @test isfile(tmpfile)

                NCDataset(tmpfile, "r") do ds
                    @test ds["u"].attrib["units"] == "K"
                    @test haskey(ds.dim, "time")
                    @test haskey(ds, "grid_mapping")
                    @test size(ds["u"]) == (1, grid.params.i_regular_out, grid.params.k_regular_out)
                end
            end

            @testset "write_netcdf keywords 3D RRR" begin
                gp = SpringsteelGridParameters(geometry="RRR", num_cells=2,
                    iMin=0.0, iMax=10.0,
                    jMin=0.0, jMax=10.0,
                    kMin=0.0, kMax=5.0,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0),
                    BCU=Dict("u" => CubicBSpline.R0),
                    BCD=Dict("u" => CubicBSpline.R0),
                    BCB=Dict("u" => CubicBSpline.R0),
                    BCT=Dict("u" => CubicBSpline.R0))
                gp = compute_derived_params(gp)
                grid = createGrid(gp)
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_3d_kw.nc")
                write_netcdf(tmpfile, grid;
                    coordinate_attributes=Dict{String,Dict{String,Any}}(
                        "x" => Dict{String,Any}("units" => "m"),
                        "y" => Dict{String,Any}("units" => "m"),
                        "z" => Dict{String,Any}("units" => "m")),
                    variable_attributes=Dict{String,Dict{String,Any}}(
                        "u" => Dict{String,Any}("units" => "m/s")),
                    time=400.0,
                    grid_mapping=Dict{String,Any}(
                        "grid_mapping_name" => "transverse_mercator",
                        "latitude_of_projection_origin" => 40.0))
                @test isfile(tmpfile)

                NCDataset(tmpfile, "r") do ds
                    @test ds["x"].attrib["units"] == "m"
                    @test ds["y"].attrib["units"] == "m"
                    @test ds["z"].attrib["units"] == "m"
                    @test ds["u"].attrib["units"] == "m/s"
                    @test haskey(ds.dim, "time")
                    @test ds.dim["time"] == 1
                    @test haskey(ds, "grid_mapping")
                    @test ds["grid_mapping"].attrib["latitude_of_projection_origin"] ≈ 40.0
                    @test size(ds["u"]) == (1, grid.params.i_regular_out, grid.params.j_regular_out, grid.params.k_regular_out)
                end
            end

            @testset "write_netcdf no time keyword preserves original behavior" begin
                gp = SpringsteelGridParameters(geometry="R", num_cells=5,
                    iMin=0.0, iMax=100.0,
                    vars=Dict("u" => 1),
                    BCL=Dict("u" => CubicBSpline.R0),
                    BCR=Dict("u" => CubicBSpline.R0))
                grid = createGrid(gp)
                spectralTransform!(grid)

                tmpfile = joinpath(mktempdir(), "test_no_time.nc")
                write_netcdf(tmpfile, grid)
                @test isfile(tmpfile)

                NCDataset(tmpfile, "r") do ds
                    @test !haskey(ds.dim, "time")
                    @test size(ds["u"]) == (grid.params.i_regular_out,)
                end
            end

        end  # NetCDF I/O

    end  # SpringsteelGrid I/O

    # ─────────────────────────────────────────────────────────────────────────
    # Backward Compatibility
    # ─────────────────────────────────────────────────────────────────────────

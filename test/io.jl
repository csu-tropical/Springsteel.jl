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

                @test data["dimensions"]["x"] == gp.i_regular_out
                reg_pts = getRegularGridpoints(grid)
                @test data["coordinates"]["x"] ≈ reg_pts
                @test haskey(data["variables"], "u")
                @test length(data["variables"]["u"]) == gp.i_regular_out
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
                @test size(data["variables"]["u"]) == (gp.i_regular_out, gp.j_regular_out)
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

            @testset "read_netcdf nonexistent file throws" begin
                @test_throws Exception read_netcdf("/nonexistent/path/file.nc")
            end

        end  # NetCDF I/O

    end  # SpringsteelGrid I/O

    # ─────────────────────────────────────────────────────────────────────────
    # Backward Compatibility
    # ─────────────────────────────────────────────────────────────────────────

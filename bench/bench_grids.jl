#!/usr/bin/env julia
# Scaling benchmark for Springsteel grid transforms.
#
# Usage:
#   julia --project bench/bench_grids.jl [output.csv]
#
# Covers all major geometries at nc ∈ {10, 50, 100} for a representative
# check of gridTransform! and spectralTransform! at steady state.
# Output CSV columns: grid, nc, transform, time_us, allocs, mem_b.
#
# Run before major changes and compare against prior CSVs to detect
# regressions in per-call cost or allocation footprint.

using Springsteel
using BenchmarkTools
using Printf

const RESULTS = Tuple{String,Int,String,Float64,Int,Int}[]

function bench(grid_name::String, nc::Int, gp)
    g = createGrid(gp)
    pts = getGridpoints(g)
    if ndims(pts) == 1
        g.physical[:, 1, 1] .= sin.(π .* pts ./ maximum(pts))
    else
        for i in 1:size(g.physical, 1)
            g.physical[i, 1, 1] = sin(π * pts[i, 1] / maximum(pts[:, 1]))
        end
    end
    spectralTransform!(g)
    gridTransform!(g)

    b_st = @benchmark spectralTransform!($g)
    b_gt = @benchmark gridTransform!($g)

    push!(RESULTS, (grid_name, nc, "spectralTransform!",
                    minimum(b_st.times) / 1e3, b_st.allocs, b_st.memory))
    push!(RESULTS, (grid_name, nc, "gridTransform!",
                    minimum(b_gt.times) / 1e3, b_gt.allocs, b_gt.memory))
end

function run_all()
    for nc in [10, 50]
        println("── nc=$nc ────────────────────────────")

        # R (1D spline)
        bench("R", nc, SpringsteelGridParameters(geometry="R", iMin=0.0, iMax=1.0,
            num_cells=nc, vars=Dict("u"=>1),
            BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0)))

        # RR (2D spline×spline)
        bench("RR", nc, SpringsteelGridParameters(geometry="RR", iMin=0.0, iMax=1.0,
            jMin=0.0, jMax=1.0, num_cells=nc, vars=Dict("u"=>1),
            BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0),
            BCD=Dict("u"=>CubicBSpline.R0), BCU=Dict("u"=>CubicBSpline.R0),
            max_wavenumber=Dict("default"=>0)))

        # RZ (2D spline×Chebyshev)
        bench("RZ", nc, SpringsteelGridParameters(geometry="RZ", iMin=0.0, iMax=1.0,
            kMin=0.0, kMax=1.0, num_cells=nc, kDim=min(nc, 32),
            vars=Dict("u"=>1),
            BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0),
            BCB=Dict("u"=>Chebyshev.R0), BCT=Dict("u"=>Chebyshev.R0),
            max_wavenumber=Dict("default"=>0)))

        # RL (2D spline×Fourier)
        bench("RL", nc, SpringsteelGridParameters(geometry="RL", iMin=0.0, iMax=Float64(nc),
            num_cells=nc, vars=Dict("u"=>1),
            BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0),
            max_wavenumber=Dict("default"=>-1)))

        if nc <= 50
            # RLZ (3D spline×Fourier×Chebyshev)
            bench("RLZ", nc, SpringsteelGridParameters(geometry="RLZ", iMin=0.0, iMax=Float64(nc),
                kMin=0.0, kMax=10.0, num_cells=nc, kDim=min(nc, 16),
                vars=Dict("u"=>1),
                BCL=Dict("u"=>CubicBSpline.R0), BCR=Dict("u"=>CubicBSpline.R0),
                BCB=Dict("u"=>Chebyshev.R0), BCT=Dict("u"=>Chebyshev.R0),
                max_wavenumber=Dict("default"=>-1)))
        end

        for (g, n, t, us, a, m) in RESULTS
            if n == nc
                @printf("  %-5s %-20s %8.2f μs  %d allocs  %d B\n", g, t, us, a, m)
            end
        end
    end
end

function write_csv(path::String)
    open(path, "w") do io
        println(io, "grid,nc,transform,time_us,allocs,mem_b")
        for (g, n, t, us, a, m) in RESULTS
            @printf(io, "%s,%d,%s,%.2f,%d,%d\n", g, n, t, us, a, m)
        end
    end
    println("\nCSV written to $path")
end

run_all()

out = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "baseline", "grids_$(readchomp(`date +%Y-%m-%d`)).csv")
mkpath(dirname(out))
write_csv(out)
